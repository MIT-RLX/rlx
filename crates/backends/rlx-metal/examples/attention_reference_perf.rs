// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **How far is rlx's Metal attention from Apple's?**
//!
//! `block_profile` measured where a transformer block's Metal time goes:
//!
//! | | seq=1 | seq=256 | seq=1024 |
//! |---|---|---|---|
//! | matmul | 48.7% | 53.9% | 23.0% |
//! | **attention** | 11.5% | 33.2% | **72.0%** |
//!
//! Attention scales O(S²) while the projections scale O(S), so at long context
//! it is almost the whole cost — and it had never been compared to anything.
//! `reference_perf` anchors GEMM against `MPSMatrixMultiplication` and found no
//! gap; this asks the same question for the op that actually dominates.
//!
//! Answering it *first* is the point. A day went into tuning a GEMM tile edge
//! before `reference_perf` showed rlx already at MPS parity with correct
//! routing — optimising before anchoring. This is the anchor.
//!
//! # The two arms
//!
//! Same graph, same process, same device state, flipped through
//! `install_runtime_config`:
//!
//! * **rlx** — `RLX_DISABLE_MPSGRAPH`: the backend's own attention thunk.
//! * **MPSGraph** — Apple's `scaledDotProductAttention` (macOS 14.4+), which
//!   `mps_graph_lower` selects for a 4-D `Op::Attention`.
//!
//! # Why Q/K/V are graph inputs, and why that is not cheating
//!
//! `mps_graph.rs` documents a real MPSGraph defect: when Q/K/V are slice-views
//! of a *computed* tensor (narrows of a fused-QKV MatMul), MPSGraph returns
//! ~100% relative error, and Apple's own SDPA builtin hits it too. The lowering
//! already bails to the thunk in that case (`attn_qkv_feeds_mps_safe`).
//!
//! Feeding Q/K/V as leaves is what isolates *attention* rather than measuring a
//! projection chain — the same reason `reference_perf` hands GEMM its operands
//! directly. It also lands on the path the bail leaves intact, so the MPSGraph
//! arm is the one Apple actually intends. The correctness gate below is not a
//! formality given that history: if the arms disagree, the number is discarded.
//!
//! ```sh
//! cargo run --release -p rlx-metal --example attention_reference_perf
//! ```

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("Metal is macOS-only.");
}

#[cfg(target_os = "macos")]
fn main() {
    use rlx_ir::op::MaskKind;
    use rlx_ir::{DType, Graph, Shape};
    use rlx_metal::backend::MetalExecutable;
    use rlx_metal::{install_runtime_config, runtime_config};

    const F: DType = DType::F32;
    const WARMUP: usize = 3;
    const ITERS: usize = 20;

    /// Causal attention over host-fed Q/K/V. `[B, H, S, D]`.
    fn attn(b: usize, h: usize, s: usize, d: usize) -> Graph {
        let mut g = Graph::new("attn_ref");
        let q = g.input("q", Shape::new(&[b, h, s, d], F));
        let k = g.input("k", Shape::new(&[b, h, s, d], F));
        let v = g.input("v", Shape::new(&[b, h, s, d], F));
        let y = g.add_node(
            rlx_ir::Op::Attention {
                num_heads: h,
                head_dim: d,
                v_head_dim: None,
                mask_kind: MaskKind::Causal,
                score_scale: None,
                attn_logit_softcap: None,
            },
            vec![q, k, v],
            Shape::new(&[b, h, s, d], F),
        );
        g.set_outputs(vec![y]);
        g
    }

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    /// Median and the p50/min ratio. A p50 far above the floor means the
    /// machine, not the kernel — the same disclosure `reference_perf` makes.
    fn stat(v: &mut [f64]) -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        (v[v.len() / 2], v[v.len() / 2] / v[0].max(f64::MIN_POSITIVE))
    }

    /// Device time of the ATTENTION THUNK, not of the whole run.
    ///
    /// This distinction cost a wrong conclusion. Timing the whole run of an
    /// attention-only graph includes uploading Q/K/V — 12.6 MB per iteration at
    /// B=1,H=16,S=1024,D=64 — and reading the output back. That I/O dominated,
    /// and reporting it as attention put the op at "6% of this chip's sgemm
    /// ceiling", which implied large headroom that does not exist. Measured as
    /// a thunk, the same shape runs at GEMM-class throughput.
    ///
    /// `RLX_METAL_THUNK_PROFILE` makes the backend submit one command buffer
    /// per thunk and record its own GPU span, which is what isolates the op.
    fn attention_thunk_ms() -> Option<f64> {
        rlx_metal::thunk_profile::total_ms()
    }

    /// Device span of a whole run — kept because the ratio between the two arms
    /// is still meaningful (both pay the same I/O), and because the gap between
    /// this and `attention_thunk_ms` IS the I/O cost.
    fn timed(exe: &mut MetalExecutable, feeds: &[(&str, &[f32])]) -> (Vec<f32>, f64) {
        use std::time::Instant;
        rlx_metal::gpu_span::reset();
        let t0 = Instant::now();
        let r = exe.run(feeds);
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        (r[0].clone(), rlx_metal::gpu_span::last_ms().unwrap_or(wall))
    }

    // (label, B, H, S, D)
    const SHAPES: &[(&str, usize, usize, usize, usize)] = &[
        ("short prefill", 1, 16, 128, 64),
        ("prefill", 1, 16, 256, 64),
        ("long prefill", 1, 16, 512, 64),
        ("longer prefill", 1, 16, 1024, 64),
        ("batch 4, prefill", 4, 16, 256, 64),
    ];

    println!("rlx's Metal attention vs Apple MPSGraph SDPA — same process, same device\n");
    println!("  why      : attention is 72% of a transformer block at seq=1024 (block_profile)");
    println!("  arms     : RLX_DISABLE_MPSGRAPH on/off, same graph");
    println!("  timing   : command-buffer device span, median of {ITERS}, {WARMUP} warmup");
    println!("  gate     : outputs must agree to 1e-3 relative — MPSGraph has a documented");
    println!("             attention defect on computed-slice Q/K/V, so agreement is checked");
    println!("             rather than assumed\n");

    // Empirical ceiling, not a theoretical one.
    //
    // "Parity with MPS" says nothing about whether BOTH are leaving the machine
    // on the table — and a vendor library is not automatically near peak. The
    // useful denominator is what this same chip demonstrably reaches on a real
    // kernel, so the ceiling here is the cost model's measured sgemm figure
    // rather than a datasheet number nobody hit.
    let hw = rlx_metal::cost::hw_model();
    // `sgemm_simd_4x4_flops`, NOT `sgemm_simd_flops`.
    //
    // `Simd` is the 8x8-output variant — a small, rarely-selected kernel that
    // calibrates at 61-66 GFLOP/s on this chip. `Simd4x4` is the 32x32 path the
    // cascade actually picks, and it calibrates at 2152-2617 GFLOP/s. Using the
    // former as a ceiling made attention read "202% of ceiling", which is the
    // sort of number that should stop a reader rather than be published.
    let ceiling_flops = hw.sgemm_simd_4x4_flops;
    println!(
        "  ceiling  : {:.0} GFLOP/s — this chip's MEASURED simd4x4 sgemm throughput\n\
         \x20            (calibration cache), i.e. what it demonstrably reaches on a real\n\
         \x20            kernel. Not a datasheet figure.\n",
        ceiling_flops / 1e9
    );

    println!(
        "  {:<18} {:>10} {:>10} {:>8} {:>8} {:>7} {:>8} {:>7}",
        "shape", "run ms", "MPS ms", "rel", "noise", "GF/s", "of ceil", "agree"
    );

    let mut logsum = 0.0f64;
    let mut counted = 0usize;
    let mut noisy = 0usize;

    for (label, b, h, s, d) in SHAPES {
        let (b, h, s, d) = (*b, *h, *s, *d);
        let qv = fill(b * h * s * d, 0x5eed);
        let kv = fill(b * h * s * d, 0xbeef);
        let vv = fill(b * h * s * d, 0xf00d);
        let feeds: Vec<(&str, &[f32])> = vec![("q", &qv), ("k", &kv), ("v", &vv)];

        let arm = |disable_mps: bool| -> (Vec<f32>, f64, f64) {
            let mut cfg = runtime_config();
            cfg.disable_mpsgraph = disable_mps;
            install_runtime_config(cfg);
            let mut exe = MetalExecutable::compile(attn(b, h, s, d));
            for _ in 0..WARMUP {
                timed(&mut exe, &feeds);
            }
            let mut samples = Vec::with_capacity(ITERS);
            let mut out = Vec::new();
            for _ in 0..ITERS {
                let (o, ms) = timed(&mut exe, &feeds);
                samples.push(ms);
                out = o;
            }
            let (med, ratio) = stat(&mut samples);
            (out, med, ratio)
        };

        // Second pass with per-thunk profiling on, to separate the op from the
        // I/O that surrounds it.
        let thunk_only = {
            rlx_ir::env::set("RLX_METAL_THUNK_PROFILE", "1");
            let mut cfg = runtime_config();
            cfg.disable_mpsgraph = true;
            install_runtime_config(cfg);
            let mut exe = MetalExecutable::compile(attn(b, h, s, d));
            let _ = exe.run(&feeds);
            rlx_metal::thunk_profile::reset();
            const N: usize = 5;
            for _ in 0..N {
                let _ = exe.run(&feeds);
            }
            let t = attention_thunk_ms().map(|ms| ms / N as f64);
            rlx_ir::env::unset("RLX_METAL_THUNK_PROFILE");
            t
        };

        let (rlx_out, rlx_ms, rlx_noise) = arm(true);
        let (mps_out, mps_ms, mps_noise) = arm(false);

        // Different algorithms, so not bit-exact. Relative error against the
        // magnitude of the reference, which is what a tolerance should compare.
        let denom = rlx_out.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let rel_err = rlx_out
            .iter()
            .zip(&mps_out)
            .fold(0.0f32, |m, (a, c)| m.max((a - c).abs()))
            / denom;
        let agree = rel_err <= 1e-3;
        let noise = rlx_noise.max(mps_noise);
        let rel = rlx_ms / mps_ms.max(f64::MIN_POSITIVE);

        // Causal attention: QK^T and PV are S^2*D MACs each, so 4*S^2*D FLOPs
        // per head; the causal mask skips ~half the score matrix.
        let flops = 2.0 * (b * h) as f64 * (s * s) as f64 * d as f64;
        // Roofline against the THUNK time. Using the run time here is what
        // produced the bogus "6% of ceiling".
        let op_ms = thunk_only.unwrap_or(rlx_ms);
        let gflops = flops / (op_ms / 1e3) / 1e9;
        let of_ceiling = 100.0 * (gflops * 1e9) / ceiling_flops;

        println!(
            "  {label:<18} {rlx_ms:>10.4} {mps_ms:>10.4} {:>7.2}x {:>7.0}% {gflops:>7.0} \
             {of_ceiling:>7.0}% {:>7}",
            rel,
            (noise - 1.0) * 100.0,
            if agree {
                "ok".to_string()
            } else {
                format!("{rel_err:.1e}")
            }
        );

        if noise > 1.15 {
            noisy += 1;
        }
        if agree && noise <= 1.15 {
            logsum += rel.ln();
            counted += 1;
        }
    }

    println!();
    if counted == 0 {
        println!(
            "  AGGREGATE WITHHELD: no shape was both in agreement and quiet enough.\n\
             \x20 {noisy}/{} shape(s) were contention-dominated. Re-run on an idle machine.",
            SHAPES.len()
        );
    } else {
        println!(
            "  geomean over {counted}/{} clean shape(s): {:.3}x",
            SHAPES.len(),
            (logsum / counted as f64).exp()
        );
        if noisy > 0 {
            println!("  ({noisy} shape(s) excluded as contention-dominated.)");
        }
    }
    println!(
        "\n  `rel` below 1.00x means rlx is FASTER than Apple's; the gap either way is the\n\
         \x20 headroom against a tuned library. `of ceil` is the other question entirely:\n\
         \x20 if BOTH arms sit far below what this chip reaches on sgemm, then parity just\n\
         \x20 means both implementations are leaving the same machine unused, and there IS\n\
         \x20 room to improve — it is simply not reachable by copying Apple.\n\
         \x20 On a contended GPU the ms column (and so GF/s) is inflated; only trust rows\n\
         \x20 whose noise is small."
    );

    // Leave the process as we found it.
    let mut restore = runtime_config();
    restore.disable_mpsgraph = false;
    install_runtime_config(restore);
}
