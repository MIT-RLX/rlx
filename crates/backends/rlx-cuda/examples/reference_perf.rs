// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Reference-anchored performance** — how far is rlx's own kernel from the
//! best-known implementation on this device?
//!
//! Every speedup rlx measures today is against *rlx*. `tune_dispatch` reports
//! 1.97× at decode, and its own contract says what that means:
//! `references: the compile-time default tile only; no external library
//! baseline`. Beating your own default 1.97× is a real improvement to the
//! fallback path and says **nothing** about whether that path is at 40% or 95%
//! of what the hardware gives a good implementation.
//!
//! CAKE reports the other number: Table 4 is relative performance per kernel
//! against TensorRT-LLM, CUTLASS, DeepGEMM and FA-4, and treats "below
//! reference" as a signal to act on rather than a fact to omit. Given
//! per-backend peak performance is this project's stated goal, that is the
//! measurement that makes every other number mean something.
//!
//! The reference here is **cuBLAS**: already linked, already the default path
//! for dense f32 GEMM, and a genuine best-known implementation for this shape
//! class. Both paths are measured *in the same process on the same device
//! state* by flipping `no_cublas` through `install_runtime_config`, so the
//! comparison cannot drift on clocks, thermals or driver state between runs.
//!
//! What this is not: a comparison against theoretical peak FLOP/s. "94% of
//! cuBLAS" is a claim about a real, tuned implementation; it is not a claim of
//! near-optimality, and cuBLAS itself is not optimal at every shape (the
//! ratios below go above 1.0 at some decode shapes, which is worth reading as
//! "cuBLAS is not tuned here", not "rlx beats NVIDIA").
//!
//! ```sh
//! cargo run --release -p rlx-cuda --example reference_perf
//! ```

use std::time::Instant;

use rlx_cuda::backend::CudaExecutable;
use rlx_ir::{DType, Graph, Shape};

const WARMUP: usize = 5;
const ITERS: usize = 30;
const L2_FLUSH_BYTES: usize = 128 << 20;

/// Above this GPU utilization, another process is running and every number
/// below is contention. Same gate as `tune_dispatch`.
const MAX_BUSY_PERCENT: f32 = 15.0;

/// Relative tolerance for rlx-vs-reference outputs.
///
/// NOT bit-exact, deliberately: cuBLAS and the tiled kernel accumulate in a
/// different order, so identical results would be the surprise. A ratio is only
/// meaningful if both sides computed the same thing, so this is a correctness
/// gate on the comparison itself.
const REL_TOL: f32 = 2e-3;

/// Shapes to anchor. Spans the decode/prefill regimes the dispatch table
/// separates, so the report says where rlx stands in each rather than averaging
/// them into one uninformative number.
const SHAPES: &[(usize, usize, usize, &str)] = &[
    (1, 1024, 1024, "decode, small hidden"),
    (1, 4096, 4096, "decode, LM hidden"),
    (32, 4096, 4096, "small batch"),
    (128, 4096, 4096, "medium batch"),
    (512, 2048, 2048, "medium prefill"),
    (2048, 2048, 2048, "large prefill"),
    (4096, 4096, 4096, "large square"),
];

fn build(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("ref_mm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn flush_l2() {
    use std::sync::{Mutex, OnceLock};
    static SCRATCH: OnceLock<Mutex<Option<cudarc::driver::CudaSlice<u8>>>> = OnceLock::new();
    let Some(ctx) = rlx_cuda::device::cuda_context() else {
        return;
    };
    let cell = SCRATCH
        .get_or_init(|| Mutex::new(ctx.default_stream().alloc_zeros::<u8>(L2_FLUSH_BYTES).ok()));
    let mut guard = cell.lock().expect("l2 scratch poisoned");
    if let Some(buf) = guard.as_mut() {
        let _ = ctx.default_stream().memset_zeros(buf);
        let _ = ctx.default_stream().synchronize();
    }
}

/// Median of `ITERS` samples, L2 flushed before each. Same protocol as the
/// tuner: a mean over back-to-back iterations measures a warm cache.
fn measure(m: usize, k: usize, n: usize, xv: &[f32], wv: &[f32]) -> (Vec<f32>, f64) {
    let mut exe = CudaExecutable::compile(build(m, k, n));
    exe.set_param("w", wv);
    for _ in 0..WARMUP {
        let _ = exe.run(&[("x", xv)]);
    }
    let mut samples = Vec::with_capacity(ITERS);
    let mut out = Vec::new();
    for _ in 0..ITERS {
        flush_l2();
        let t0 = Instant::now();
        let r = exe.run(&[("x", xv)]);
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
        out = r[0].clone();
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    (out, samples[samples.len() / 2])
}

/// Flip the cuBLAS tiers on or off for the *next* compile, in-process.
fn set_cublas(enabled: bool) {
    let mut cfg = rlx_cuda::config::CudaRuntimeConfig::from_env();
    cfg.no_cublas = !enabled;
    rlx_cuda::config::install_runtime_config(cfg);
}

/// Foreign CUDA processes on device 0, excluding this one.
///
/// The precise signal. NVML's utilization number cannot tell a compositor
/// drawing a desktop from another training job: on the CUDA rig with no compute
/// client at all, utilization sits at a sustained 17%, and a 15% threshold
/// refuses to measure on a GPU that is in fact free. A foreign *compute*
/// process is what actually contends — that is what produced the ~25x slower
/// timings and a different winner earlier in this work.
fn foreign_compute_processes() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-compute-apps=pid", "--format=csv,noheader"])
        .output()
        .ok()?;
    let me = std::process::id().to_string();
    Some(
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && *l != me)
            .count(),
    )
}

/// `Err` only when something else is actually computing on the GPU.
///
/// Elevated utilization with no compute client is *reported, not refused* —
/// hiding it would be as wrong as blocking on it, so the number goes in the
/// header where a reader can weigh it.
fn contention_ok() -> Result<String, String> {
    if rlx_ir::env::flag("RLX_ALLOW_THROTTLE") {
        return Ok("RLX_ALLOW_THROTTLE=1 — gate bypassed".into());
    }
    match foreign_compute_processes() {
        Some(0) => {}
        Some(n) => {
            return Err(format!(
                "{n} other CUDA process(es) are computing on this device"
            ));
        }
        None => return Ok("nvidia-smi unavailable — contention unverified".into()),
    }
    let mut min_util: Option<f32> = None;
    for i in 0..5 {
        let Some(s) = rlx_cuda::nvml::sample(0) else {
            return Ok("no foreign compute clients; NVML unavailable for utilization".into());
        };
        if let Some(u) = s.util_percent {
            min_util = Some(min_util.map_or(u, |m: f32| m.min(u)));
        }
        if i < 4 {
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }
    match min_util {
        Some(u) if u > MAX_BUSY_PERCENT => Ok(format!(
            "no foreign compute clients, but {u:.0}% utilization (display/compositor) — \
             timings carry that baseline"
        )),
        Some(u) => Ok(format!(
            "idle: no foreign compute clients, {u:.0}% util (min of 5)"
        )),
        None => Ok("no foreign compute clients; utilization unreadable".into()),
    }
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / (1.0 + y.abs()))
        .fold(0.0f32, f32::max)
}

/// Attention shapes: `(batch, heads, seq, head_dim, label)`.
///
/// Decode (`seq_q = 1`) and prefill, at head dims a real model uses.
const ATTN_SHAPES: &[(usize, usize, usize, usize, &str)] = &[
    (1, 32, 128, 64, "prefill, short"),
    (1, 32, 512, 64, "prefill, medium"),
    (1, 32, 1024, 64, "prefill, long"),
    (4, 32, 512, 64, "prefill, batch 4"),
    (1, 32, 512, 128, "prefill, head_dim 128"),
];

/// rlx's fused `Op::Attention`.
fn attn_fused(b: usize, h: usize, s: usize, d: usize) -> Graph {
    let qkv = Shape::new(&[b, h, s, d], DType::F32);
    let mut g = Graph::new("attn_fused");
    let q = g.input("q", qkv.clone());
    let k = g.input("k", qkv.clone());
    let v = g.input("v", qkv.clone());
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: rlx_ir::op::MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        qkv,
    );
    g.set_outputs(vec![y]);
    g
}

/// The reference: attention composed from primitives, so the two GEMMs go
/// through cuBLAS.
///
/// **This is not FlashAttention.** It is the unfused formulation a vendor BLAS
/// gives you — `QK^T`, scaled softmax, `PV` — and it materializes the full
/// `[B*H, S, S]` score matrix, which is exactly what a fused kernel exists to
/// avoid. Read the ratio as "what fusion buys over vendor GEMMs", not as a
/// comparison against a tuned attention library. rlx links no such library.
fn attn_unfused(b: usize, h: usize, s: usize, d: usize) -> Graph {
    let bh = b * h;
    let f = DType::F32;
    let mut g = Graph::new("attn_unfused");
    let q4 = g.input("q", Shape::new(&[b, h, s, d], f));
    let k4 = g.input("k", Shape::new(&[b, h, s, d], f));
    let v4 = g.input("v", Shape::new(&[b, h, s, d], f));

    let q = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![bh as i64, s as i64, d as i64],
        },
        vec![q4],
        Shape::new(&[bh, s, d], f),
    );
    let k = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![bh as i64, s as i64, d as i64],
        },
        vec![k4],
        Shape::new(&[bh, s, d], f),
    );
    let v = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![bh as i64, s as i64, d as i64],
        },
        vec![v4],
        Shape::new(&[bh, s, d], f),
    );
    // K^T over the trailing two axes.
    let kt = g.add_node(
        rlx_ir::Op::Transpose {
            perm: vec![0, 2, 1],
        },
        vec![k],
        Shape::new(&[bh, d, s], f),
    );
    let scores = g.matmul(q, kt, Shape::new(&[bh, s, s], f));
    // 1/sqrt(d), as a constant multiply. `Op::Constant` carries raw bytes.
    let scale = g.add_node(
        rlx_ir::Op::Constant {
            data: (1.0f32 / (d as f32).sqrt()).to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], f),
    );
    let scaled = g.add_node(
        rlx_ir::Op::Binary(rlx_ir::op::BinaryOp::Mul),
        vec![scores, scale],
        Shape::new(&[bh, s, s], f),
    );
    let probs = g.add_node(
        rlx_ir::Op::Softmax { axis: -1 },
        vec![scaled],
        Shape::new(&[bh, s, s], f),
    );
    let out3 = g.matmul(probs, v, Shape::new(&[bh, s, d], f));
    let out = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![b as i64, h as i64, s as i64, d as i64],
        },
        vec![out3],
        Shape::new(&[b, h, s, d], f),
    );
    g.set_outputs(vec![out]);
    g
}

/// Time a graph with three named inputs.
fn measure3(g: Graph, feeds: &[(&str, &[f32])]) -> (Vec<f32>, f64) {
    let mut exe = CudaExecutable::compile(g);
    for _ in 0..WARMUP {
        let _ = exe.run(feeds);
    }
    let mut samples = Vec::with_capacity(ITERS);
    let mut out = Vec::new();
    for _ in 0..ITERS {
        flush_l2();
        let t0 = Instant::now();
        let r = exe.run(feeds);
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
        out = r[0].clone();
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("finite timings"));
    (out, samples[samples.len() / 2])
}

fn run_attention() {
    println!("\n\nrlx fused `Op::Attention` vs unfused (cuBLAS GEMMs + softmax)");
    println!("reference = the UNFUSED formulation with vendor GEMMs — NOT FlashAttention\n");
    println!(
        "  {:<24} {:>28}  {:>10} {:>12}  {:>8}",
        "shape", "case", "fused ms", "unfused ms", "rel"
    );
    let mut ratios: Vec<f64> = Vec::new();
    for &(b, h, s, d, label) in ATTN_SHAPES {
        let n = b * h * s * d;
        let qv: Vec<f32> = (0..n).map(|i| ((i % 61) as f32) * 0.01 - 0.3).collect();
        let kv: Vec<f32> = (0..n).map(|i| ((i % 47) as f32) * 0.01 - 0.23).collect();
        let vv: Vec<f32> = (0..n).map(|i| ((i % 53) as f32) * 0.01 - 0.26).collect();
        let feeds: Vec<(&str, &[f32])> = vec![
            ("q", qv.as_slice()),
            ("k", kv.as_slice()),
            ("v", vv.as_slice()),
        ];

        set_cublas(true);
        let (fo, fms) = measure3(attn_fused(b, h, s, d), &feeds);
        let (uo, ums) = measure3(attn_unfused(b, h, s, d), &feeds);

        let rel_err = max_rel(&fo, &uo);
        let shape = format!("{b}x{h}x{s}x{d}");
        if rel_err > REL_TOL {
            println!(
                "  {shape:<24} {label:>28}  MISMATCH (max rel {rel_err:.2e}) — ratio withheld"
            );
            continue;
        }
        let rel = ums / fms;
        ratios.push(rel);
        println!("  {shape:<24} {label:>28}  {fms:>10.3} {ums:>12.3}  {rel:>7.2}x");
    }
    if !ratios.is_empty() {
        let gm = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
        println!(
            "\n  geometric mean over {} shape(s): fusion is {gm:.2}x the unfused path",
            ratios.len()
        );
        println!(
            "  Above 1.00x means fusion wins. This says nothing about how either\n               compares to a tuned attention library — rlx links none."
        );
    }
}

fn main() {
    if !rlx_cuda::is_available() {
        println!("CUDA not available on this host — nothing to anchor.");
        return;
    }
    match contention_ok() {
        Ok(note) => println!("GPU state: {note}"),
        Err(why) => {
            eprintln!(
                "REFUSING to measure: {why}.\n\
                 A reference ratio taken under contention is not a reference ratio."
            );
            std::process::exit(1);
        }
    }

    println!("\nrlx tiled `matmul` vs cuBLAS — same process, same device state");
    println!("reference = cuBLAS (a real tuned implementation, NOT theoretical peak)\n");
    println!(
        "  {:<22} {:>16}  {:>10} {:>10}  {:>8}  {:>9}",
        "shape", "case", "rlx ms", "cuBLAS ms", "rel", "GFLOP/s"
    );

    let mut ratios: Vec<f64> = Vec::new();
    let mut worst: Option<(String, f64)> = None;
    for &(m, k, n, label) in SHAPES {
        let xv: Vec<f32> = (0..m * k).map(|i| ((i % 97) as f32) * 1e-2 - 0.5).collect();
        let wv: Vec<f32> = (0..k * n).map(|i| ((i % 89) as f32) * 1e-2 - 0.5).collect();

        set_cublas(false);
        let (rlx_out, rlx_ms) = measure(m, k, n, &xv, &wv);
        set_cublas(true);
        let (ref_out, ref_ms) = measure(m, k, n, &xv, &wv);

        // A ratio between two different computations is meaningless.
        let rel_err = max_rel(&rlx_out, &ref_out);
        if rel_err > REL_TOL {
            println!(
                "  {:<22} {:>16}  MISMATCH vs reference (max rel {rel_err:.2e}) — ratio withheld",
                format!("{m}x{k}x{n}"),
                label
            );
            continue;
        }

        let rel = ref_ms / rlx_ms; // >1 = rlx faster
        let gflops = 2.0 * (m * k * n) as f64 / (rlx_ms * 1e-3) / 1e9;
        ratios.push(rel);
        if worst.as_ref().is_none_or(|(_, w)| rel < *w) {
            worst = Some((format!("{m}x{k}x{n} ({label})"), rel));
        }
        println!(
            "  {:<22} {:>16}  {rlx_ms:>10.3} {ref_ms:>10.3}  {rel:>7.2}x  {gflops:>9.1}",
            format!("{m}x{k}x{n}"),
            label
        );
    }

    if ratios.is_empty() {
        println!("\nno comparable shapes — every case mismatched the reference.");
        return;
    }
    // Geometric mean: ratios compose multiplicatively, so an arithmetic mean
    // would let one 3x outlier hide several 0.5x shapes.
    let gm = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
    println!(
        "\n  geometric mean over {} shape(s): {gm:.2}x of cuBLAS",
        ratios.len()
    );
    if let Some((shape, r)) = worst {
        // Name the worst case explicitly. An aggregate that hides where the
        // implementation is weakest is the failure this report exists to avoid.
        println!("  worst shape: {shape} at {r:.2}x");
    }
    println!(
        "\n  Read this as distance from a tuned library, not from peak. Ratios above\n  \
         1.00x mean cuBLAS is not tuned for that shape, not that rlx is optimal."
    );

    run_attention();
}
