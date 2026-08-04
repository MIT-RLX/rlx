// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ScaledGroupedMatMul` — native low-precision (MXFP4) *grouped* (MoE)
//! GEMM. Verifies the CPU reference oracle against a plain-f32 grouped matmul
//! and that the portable decompose lowering (ScaledDequantize + GroupedMatMul)
//! reproduces the fused CPU result.

use rlx_ir::*;
use rlx_runtime::{Device, Session};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

// Pure-f32 grouped matmul: out[i,j] = Σ_p x[i,p]·w[e,j,p], e = expert[i].
// Weight is [E,N,K] (TN), so out[i] = x[i] · w[e]ᵀ.
fn reference(
    x: &[f32],
    w: &[f32],
    expert: &[f32],
    m: usize,
    k: usize,
    n: usize,
    e_cnt: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        let e = expert[i] as usize;
        assert!(e < e_cnt);
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[i * k + p] * w[(e * n + j) * k + p];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn build(m: usize, k: usize, n: usize, e_cnt: usize) -> Graph {
    let mut g = Graph::new("sgmm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.input("w", Shape::new(&[e_cnt, n, k], DType::F32));
    let idx = g.input("idx", Shape::new(&[m], DType::F32));
    let y = g.scaled_grouped_matmul(x, w, idx, ScaledFormat::F4E2M1, ScaleLayout::mx());
    g.set_outputs(vec![y]);
    g
}

fn inputs(m: usize, k: usize, n: usize, e_cnt: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.11).sin() * 1.3).collect();
    let w: Vec<f32> = (0..e_cnt * n * k)
        .map(|i| (i as f32 * 0.037).cos() * 1.1)
        .collect();
    // Round-robin expert routing so every expert gets tokens.
    let idx: Vec<f32> = (0..m).map(|i| (i % e_cnt) as f32).collect();
    (x, w, idx)
}

#[test]
fn cpu_scaled_grouped_matmul_tracks_f32() {
    let (m, k, n, e_cnt) = (6usize, 64usize, 8usize, 3usize);
    let (x, w, idx) = inputs(m, k, n, e_cnt);

    let mut c = Session::new(Device::Cpu).compile(build(m, k, n, e_cnt));
    let out = c.run(&[("x", &x), ("w", &w), ("idx", &idx)]).remove(0);

    assert_eq!(out.len(), m * n);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite output");

    let reference = reference(&x, &w, &idx, m, k, n, e_cnt);
    let cos = cosine(&out, &reference);
    eprintln!("MXFP4 scaled_grouped_matmul: cosine_vs_f32={cos:.4}");
    assert!(cos >= 0.9, "MXFP4 grouped cosine {cos} < 0.9");
}

// The decompose lowering (used by every backend without a native FP4-grouped
// kernel) must reproduce the fused CPU oracle. Force it via the fusion pass and
// run the decomposed graph on CPU.
#[test]
fn decompose_matches_fused_oracle() {
    use rlx_fusion::LowerScaledGroupedMatMul;
    use rlx_fusion::pass::Pass;

    let (m, k, n, e_cnt) = (6usize, 64usize, 8usize, 3usize);
    let (x, w, idx) = inputs(m, k, n, e_cnt);

    // Fused CPU oracle.
    let mut fused = Session::new(Device::Cpu).compile(build(m, k, n, e_cnt));
    let out_fused = fused.run(&[("x", &x), ("w", &w), ("idx", &idx)]).remove(0);

    // Decomposed graph (ScaledDequantize + Transpose + GroupedMatMul).
    let lowered = LowerScaledGroupedMatMul.run(build(m, k, n, e_cnt));
    assert!(
        !lowered
            .nodes()
            .iter()
            .any(|nd| matches!(nd.op, Op::ScaledGroupedMatMul { .. })),
        "decompose left a ScaledGroupedMatMul node"
    );
    let mut dec = Session::new(Device::Cpu).compile(lowered);
    let out_dec = dec.run(&[("x", &x), ("w", &w), ("idx", &idx)]).remove(0);

    let cos = cosine(&out_fused, &out_dec);
    eprintln!("decompose vs fused: cosine={cos:.6}");
    assert!(cos >= 0.999, "decompose diverges from oracle: cosine {cos}");
}
