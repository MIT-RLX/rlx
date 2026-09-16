// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Recompute a fused `matmul + bias (+ activation)` from dumped operands and
//! check it against an f64 reference, row by row.
//!
//! When two backends disagree on one node, the first question is which of them
//! is right — and the cheapest way to answer it is to recompute the node
//! outside the runtime. `RLX_CPU_DUMP_NODE_DATA=<ids>` with `RLX_CPU_DUMP_DIR`
//! writes any node's raw `f32` arena buffer (parameters included) to
//! `node_<id>.f32`; this example reads three of those back, runs the same
//! `sgemm_accumulate` the `FusedMmBiasAct` thunk runs, and compares each
//! requested row against a straightforward f64 dot product.
//!
//! ```sh
//! DUMP_DIR=/tmp/dump A=48 W=49 BIAS=50 M=1387 K=192 N=768 ROWS=1386,1385,174 \
//!   cargo run -p rlx-cpu --release --example sgemm_tail
//! ```
//!
//! It was written for a `[1, 1387, 768]` MLP up-projection whose last row was
//! ~18% off. The answer it gave was **negative and useful**: the GEMM was exact
//! (max|diff| 3.8e-7 on every row, including the last), which ruled out the
//! matmul and left the activation fused after it — where the bug actually was,
//! in `par_gelu_inplace`'s tail handling. Note that this checks the GEMM only:
//! no activation is applied to either side, deliberately, so that the two
//! halves of a fused op can be attributed separately.

fn read(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn env_usize(key: &str) -> usize {
    std::env::var(key)
        .unwrap_or_else(|_| panic!("set {key}"))
        .parse()
        .unwrap_or_else(|e| panic!("{key}: {e}"))
}

fn main() {
    let dir = std::env::var("DUMP_DIR").expect("set DUMP_DIR");
    let node = |key: &str| format!("{dir}/node_{}.f32", std::env::var(key).expect("set {key}"));
    let (m, k, n) = (env_usize("M"), env_usize("K"), env_usize("N"));
    let (a, w, bias) = (read(&node("A")), read(&node("W")), read(&node("BIAS")));
    assert_eq!(a.len(), m * k, "A is not [{m}, {k}]");
    assert_eq!(w.len(), k * n, "W is not [{k}, {n}]");
    assert_eq!(bias.len(), n, "bias is not [{n}]");

    // Exactly what `exec_fused_mm_bias_act` does: fill with bias, then
    // accumulate into it (`beta = 1`), which is a different CBLAS path from the
    // usual `beta = 0` matmul and worth exercising as the runtime calls it.
    let mut c = vec![0f32; m * n];
    for row in 0..m {
        c[row * n..(row + 1) * n].copy_from_slice(&bias);
    }
    rlx_cpu::blas::sgemm_accumulate(&a, &w, &mut c, m, k, n);

    let rows: Vec<usize> = std::env::var("ROWS")
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![m - 1]);
    let mut worst = 0f64;
    for row in rows {
        assert!(row < m, "row {row} is out of range for m = {m}");
        let got = &c[row * n..(row + 1) * n];
        let want: Vec<f64> = (0..n)
            .map(|j| {
                let mut acc = bias[j] as f64;
                for p in 0..k {
                    acc += a[row * k + p] as f64 * w[p * n + j] as f64;
                }
                acc
            })
            .collect();
        let err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - w).abs())
            .fold(0f64, f64::max);
        worst = worst.max(err);
        println!(
            "row {row:>6}: sgemm tabs={:>13.6}  f64 tabs={:>13.6}  max|diff|={err:.3e}",
            got.iter().map(|v| v.abs() as f64).sum::<f64>(),
            want.iter().map(|v| v.abs()).sum::<f64>(),
        );
    }
    println!("worst max|diff| = {worst:.3e}");
}
