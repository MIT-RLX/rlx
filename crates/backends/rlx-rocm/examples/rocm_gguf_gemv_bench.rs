// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **How close to memory speed is the GGUF decode GEMV on ROCm?**
//!
//! A decode step's `DequantMatMul` (m = 1, or 2 for a speculative verify)
//! reads every weight byte once and does little else, so its time against the
//! weight bytes is the measure: GB/s, set against what the device's memory
//! can do. One graph holds `--copies` independent matmuls of one shape (each
//! its own weights, all reading one `x`), another holds one; the difference
//! over `copies − 1` is one matmul without the run's fixed costs.
//!
//! Shapes are Qwen3.5-0.8B's (hidden 1024, FFN 3584). Q8_0 by default.
//!
//! ```sh
//! cargo run --release -p rlx-rocm --example rocm_gguf_gemv_bench
//! cargo run --release -p rlx-rocm --example rocm_gguf_gemv_bench -- --iters 200 --copies 16
//! ```
//!
//! Run it on an idle GPU: on a contended one it measures the contention.

use rlx_ir::{DType, Graph, QuantScheme, Shape};
use rlx_rocm::backend::RocmExecutable;

/// `n` Q8_0 rows of `k` weights: an f16 scale (≈ 1/3) then 32 int8 per block.
/// The scale and `x` are not short binary fractions, so the sums round and
/// the output hash tells one summation order from another (with scale 1.0
/// and `x` in 64ths every order gave the same bits).
fn q8_0_weights(n: usize, k: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2654435761).max(1);
    let mut out = Vec::with_capacity(n * k / 32 * 34);
    for _ in 0..n * k / 32 {
        out.extend_from_slice(&0x3555u16.to_le_bytes());
        for _ in 0..32 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            out.push((s >> 24) as u8);
        }
    }
    out
}

fn graph(m: usize, n: usize, k: usize, copies: usize) -> Graph {
    let mut g = Graph::new("gguf_gemv_bench");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let bytes = n * k / 32 * 34;
    let outs: Vec<_> = (0..copies)
        .map(|c| {
            let w = g.param(format!("w{c}"), Shape::new(&[bytes], DType::U8));
            g.add_node(
                rlx_ir::Op::DequantMatMul {
                    scheme: QuantScheme::GgufQ8_0,
                },
                vec![x, w],
                Shape::new(&[m, n], DType::F32),
            )
        })
        .collect();
    g.set_outputs(outs);
    g
}

/// Mean seconds a run of `copies` matmuls takes, and an FNV-1a hash of the
/// first output's bits (a kernel change that should keep every bit can be
/// checked by it).
fn time(
    m: usize,
    n: usize,
    k: usize,
    copies: usize,
    iters: usize,
    gap_ms: u64,
    warm: bool,
    spin: bool,
) -> (f64, u64) {
    // `--warm`: spend each gap running a small matmul instead of sleeping.
    let mut warmer = warm.then(|| {
        let mut e = RocmExecutable::compile(graph(1, 1024, 1024, 1));
        e.set_param_bytes("w0", &q8_0_weights(1024, 1024, 99));
        e
    });
    let wx = vec![0.5f32; 1024];
    let mut exe = RocmExecutable::compile(graph(m, n, k, copies));
    for c in 0..copies {
        exe.set_param_bytes(&format!("w{c}"), &q8_0_weights(n, k, c as u32 + 1));
    }
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.7311).sin()).collect();
    let out = exe.run(&[("x", &x)]).remove(0);
    let hash = out.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3)
    });
    for _ in 0..2 {
        exe.run(&[("x", &x)]);
    }
    // `--gap-ms`: idle that long before each timed run, as a pipeline stage
    // waiting on its neighbours does (an idle GPU lowers its clock).
    let mut total = 0u64;
    for _ in 0..iters {
        if gap_ms > 0 {
            let until = std::time::Instant::now() + std::time::Duration::from_millis(gap_ms);
            match warmer.as_mut() {
                Some(w) => {
                    while std::time::Instant::now() < until {
                        w.run(&[("x", &wx)]);
                    }
                }
                // `--spin`: keep this CPU core busy instead (the GPU idle):
                // tells a host-side slowdown from a GPU one.
                None if spin => {
                    while std::time::Instant::now() < until {
                        std::hint::spin_loop();
                    }
                }
                None => std::thread::sleep(until - std::time::Instant::now()),
            }
        }
        let t = rlx_ir::Tick::now();
        exe.run(&[("x", &x)]);
        total += rlx_ir::Tick::now().elapsed_ns(t);
    }
    (total as f64 * 1e-9 / iters as f64, hash)
}

/// Mean seconds one f32 `[1, k] · [k, n]` matmul takes, same method: what
/// the memory gives a kernel that only streams weights.
fn time_f32(n: usize, k: usize, copies: usize, iters: usize) -> f64 {
    let run = |copies: usize| {
        let mut g = Graph::new("f32_gemv_bench");
        let x = g.input("x", Shape::new(&[1, k], DType::F32));
        let outs: Vec<_> = (0..copies)
            .map(|c| {
                let w = g.param(format!("w{c}"), Shape::new(&[k, n], DType::F32));
                g.matmul(x, w, Shape::new(&[1, n], DType::F32))
            })
            .collect();
        g.set_outputs(outs);
        let mut exe = RocmExecutable::compile(g);
        let w: Vec<f32> = (0..k * n)
            .map(|i| ((i % 89) as f32 - 44.0) / 512.0)
            .collect();
        for c in 0..copies {
            exe.set_param(&format!("w{c}"), &w);
        }
        let x: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 - 48.0) / 64.0).collect();
        for _ in 0..3 {
            exe.run(&[("x", &x)]);
        }
        let t = rlx_ir::Tick::now();
        for _ in 0..iters {
            exe.run(&[("x", &x)]);
        }
        rlx_ir::Tick::now().elapsed_ns(t) as f64 * 1e-9 / iters as f64
    };
    (run(copies) - run(1)) / (copies - 1) as f64
}

fn main() {
    if !rlx_rocm::is_available() {
        println!("no ROCm device — nothing to measure");
        return;
    }
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: usize| -> usize {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let iters = arg("--iters", 100);
    let copies = arg("--copies", 12).max(2);
    let gap = arg("--gap-ms", 0) as u64;
    let warm = args.iter().any(|a| a == "--warm");
    let spin = args.iter().any(|a| a == "--spin");
    println!(
        "Q8_0 DequantMatMul, {copies} copies a graph, {iters} runs (µs a matmul, weight GB/s)"
    );
    println!(
        "{:>3} {:>6} {:>6} {:>10} {:>8}  {:>16}",
        "m", "n", "k", "µs", "GB/s", "output bits"
    );
    let f = time_f32(3584, 1024, copies, iters);
    println!(
        "reference: f32 matmul m=1 3584×1024: {:.1} µs, {:.1} GB/s",
        f * 1e6,
        (3584 * 1024 * 4) as f64 / f / 1e9
    );
    for (n, k) in [
        (3584, 1024),
        (1024, 3584),
        (2048, 1024),
        (1024, 2048),
        (6144, 1024),
    ] {
        for m in [1, 2] {
            let (one, hash) = time(m, n, k, 1, iters, gap, warm, spin);
            let (many, _) = time(m, n, k, copies, iters, gap, warm, spin);
            let per = (many - one) / (copies - 1) as f64;
            let bytes = (n * k / 32 * 34) as f64;
            println!(
                "{m:>3} {n:>6} {k:>6} {:>10.1} {:>8.1}  {hash:016x}",
                per * 1e6,
                bytes / per / 1e9
            );
        }
    }
}
