#![cfg(feature = "cpu")]
//! Benchmark: composition VQ (matmul+argmin) vs a fused single-pass loop, to
//! decide whether a fused native kernel is worth building. Run with:
//!   cargo test -p rlx-runtime --release --test vq_bench -- --nocapture

use rlx_ir::ops::vq::VqMetric;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};
use std::time::Instant;

fn const_f32(g: &mut Graph, xs: &[f32], dims: &[usize]) -> NodeId {
    let mut b = Vec::with_capacity(xs.len() * 4);
    for x in xs {
        b.extend_from_slice(&x.to_le_bytes());
    }
    g.add_node(
        Op::Constant { data: b },
        vec![],
        Shape::new(dims, DType::F32),
    )
}
fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Naive single-threaded fused loop (the original — kept for the A/B).
fn vq_assign_l2_naive(x: &[f32], cb: &[f32], n: usize, d: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for i in 0..n {
        let xi = &x[i * d..(i + 1) * d];
        let mut best = f32::INFINITY;
        let mut best_j = 0usize;
        for j in 0..k {
            let cj = &cb[j * d..(j + 1) * d];
            let mut dist = 0.0f32;
            for t in 0..d {
                let diff = xi[t] - cj[t];
                dist += diff * diff;
            }
            if dist < best {
                best = dist;
                best_j = j;
            }
        }
        out[i] = best_j as f32;
    }
    out
}

/// Fast fused VQ: rayon over rows, precomputed codebook norms, and the *same*
/// `‖C‖²−2·x·Cᵀ` proxy the composition uses (so results match modulo f32
/// summation order). Running argmin per row → never materializes the [N,K]
/// distance matrix. The inner dot auto-vectorizes at -O.
fn vq_assign_l2_fast(
    x: &[f32],
    cb: &[f32],
    cb_norm: &[f32],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut out = vec![0f32; n];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let xi = &x[i * d..(i + 1) * d];
        let mut best = f32::INFINITY;
        let mut best_j = 0usize;
        for j in 0..k {
            let cj = &cb[j * d..(j + 1) * d];
            let mut dot = 0.0f32;
            for t in 0..d {
                dot += xi[t] * cj[t];
            }
            let dist = cb_norm[j] - 2.0 * dot; // drop ‖x‖² (constant per row)
            if dist < best {
                best = dist;
                best_j = j;
            }
        }
        *o = best_j as f32;
    });
    out
}

fn seeded(n: usize, salt: u32) -> Vec<f32> {
    // Cheap deterministic pseudo-data (no rng).
    (0..n)
        .map(|i| {
            let z = (i as u32).wrapping_mul(2654435761).wrapping_add(salt);
            ((z >> 9) as f32 / (1u32 << 23) as f32) - 0.5
        })
        .collect()
}

#[test]
fn bench_vq_composition_vs_fused() {
    for &(n, d, k) in &[
        (256usize, 128usize, 1024usize),
        (512, 128, 4096),
        (256, 64, 8192),
    ] {
        let x = seeded(n * d, 1);
        let cb = seeded(k * d, 2);

        // Composition graph (matmul + argmin + gather idx).
        let mut g = Graph::new("vq");
        let xn = const_f32(&mut g, &x, &[n, d]);
        let cbn = const_f32(&mut g, &cb, &[k, d]);
        let (idx, _q) = g.vector_quantize(xn, cbn, VqMetric::L2);
        g.set_outputs(vec![idx]);
        let mut compiled = Session::new(Device::Cpu).compile(g);

        // Perf micro-bench. Warm up (spin the rayon pool + reach steady-state
        // clocks) then take the *minimum* per-iteration time — the least-
        // contended sample, which reflects the kernel's true steady-state cost.
        // A mean over iters lets a single scheduler hiccup on the all-cores
        // rayon loop dominate (that contention noise is what made this flaky),
        // so we report the best sample instead.
        let iters = 30;
        let warmup = 5;

        // composition (matmul + argmin + gather)
        let comp_idx = bytes_to_f32s(&compiled.run_typed(&[])[0].0);
        for _ in 0..warmup {
            let _ = compiled.run_typed(&[]);
        }
        let mut comp_ns = u128::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let _ = compiled.run_typed(&[]);
            comp_ns = comp_ns.min(t.elapsed().as_nanos());
        }

        // precompute codebook norms ‖C_j‖²
        let cb_norm: Vec<f32> = (0..k)
            .map(|j| cb[j * d..(j + 1) * d].iter().map(|&v| v * v).sum())
            .collect();

        // fast fused (rayon + norm-proxy)
        let fused_idx = vq_assign_l2_fast(&x, &cb, &cb_norm, n, d, k);
        for _ in 0..warmup {
            let _ = vq_assign_l2_fast(&x, &cb, &cb_norm, n, d, k);
        }
        let mut fused_ns = u128::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let _ = vq_assign_l2_fast(&x, &cb, &cb_norm, n, d, k);
            fused_ns = fused_ns.min(t.elapsed().as_nanos());
        }

        // naive single-threaded (for context)
        let mut naive_ns = u128::MAX;
        for _ in 0..iters {
            let t = Instant::now();
            let _ = vq_assign_l2_naive(&x, &cb, n, d, k);
            naive_ns = naive_ns.min(t.elapsed().as_nanos());
        }

        // Same proxy as the composition → agree except for f32 summation-order
        // tie-flips between near-equidistant random codes (informational).
        let mism = comp_idx
            .iter()
            .zip(fused_idx.iter())
            .filter(|(a, b)| a != b)
            .count();
        let speedup = comp_ns as f64 / fused_ns as f64;
        println!(
            "N={n:>4} D={d:>3} K={k:>5}  composition={:>9}ns  fused(fast)={:>9}ns  naive={:>9}ns  speedup={speedup:>5.2}x  divergent={mism}/{n}",
            comp_ns, fused_ns, naive_ns,
        );
        // Fused must beat the naive single-threaded loop everywhere — a
        // comfortable, hardware-independent invariant (naive is single-core;
        // fused is rayon + the norm proxy).
        assert!(
            fused_ns < naive_ns,
            "fused ({fused_ns}ns) should beat naive ({naive_ns}ns)"
        );
        // Fused's edge over the matmul+argmin composition is that it never
        // materialises / re-scans the [N,K] distance matrix, so its win grows
        // with the working set N*K. At small K on a fast BLAS (e.g. Accelerate)
        // that [N,K] matrix fits in cache, so the composition's GEMM is *close*
        // to fused — a near-crossover where run-to-run scheduling noise on the
        // all-cores rayon loop flips the winner (this is why we don't hard-gate
        // it there; it's also why rlx-vq keeps the composition path). The win
        // is robust and large (≈2–5×) once the matrix spills L2 (N*K ≳ 2M).
        if n * k >= 2_000_000 {
            assert!(
                speedup > 1.0,
                "fused should beat the composition at N={n} D={d} K={k} \
                 (N*K={}, got {speedup:.2}x)",
                n * k
            );
        }
    }
}

/// Correctness on well-separated codes: with each input placed next to a
/// distinct code, both paths must agree exactly (no tie ambiguity).
#[test]
fn fused_matches_composition_on_separable_data() {
    let (n, d, k) = (64usize, 16usize, 64usize);
    // codebook: code j is the one-hot-ish vector j*10 on axis (j % d).
    let mut cb = vec![0f32; k * d];
    for j in 0..k {
        cb[j * d + (j % d)] = 10.0 + j as f32;
    }
    // input i sits right on code (i % k).
    let mut x = vec![0f32; n * d];
    for i in 0..n {
        let j = i % k;
        x[i * d + (j % d)] = 10.0 + j as f32 + 0.01;
    }
    let cb_norm: Vec<f32> = (0..k)
        .map(|j| cb[j * d..(j + 1) * d].iter().map(|&v| v * v).sum())
        .collect();

    let mut g = Graph::new("vq_ok");
    let xn = const_f32(&mut g, &x, &[n, d]);
    let cbn = const_f32(&mut g, &cb, &[k, d]);
    let (idx, _q) = g.vector_quantize(xn, cbn, VqMetric::L2);
    g.set_outputs(vec![idx]);
    let comp = bytes_to_f32s(&Session::new(Device::Cpu).compile(g).run_typed(&[])[0].0);
    let fused = vq_assign_l2_fast(&x, &cb, &cb_norm, n, d, k);
    assert_eq!(
        comp, fused,
        "fused must match composition on separable data"
    );
    for i in 0..n {
        assert_eq!(fused[i] as usize, i % k, "nearest code should be i%k");
    }
}
