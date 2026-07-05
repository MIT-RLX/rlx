// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! `CompileOptions::scaled_quant` — the execution-flow knob that rewrites every
//! 2-D matmul in a graph into a native low-precision `ScaledMatMul` in a chosen
//! element format (including a parameterized `ScaledFormat::Custom` like
//! `f4e3m0`) at compile time. Verifies the policy runs end-to-end on CPU and the
//! quantized output tracks the plain-f32 result.

use rlx_ir::*;
use rlx_runtime::{CompileOptions, Device, ScaledQuantConfig, Session};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn build(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[m, k], DType::F32));
    let w = g.input("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    g
}

#[test]
fn session_scaled_quant_policy_f4e3m0_tracks_f32() {
    let (m, k, n) = (4usize, 64usize, 8usize);
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();

    // Plain f32 reference.
    let mut c0 = Session::new(Device::Cpu).compile(build(m, k, n));
    let reference = c0.run(&[("x", &x), ("w", &w)]).remove(0);

    // Same graph, compiled with the f4e3m0 scaled-quant policy — the plain
    // matmul is rewritten to a ScaledMatMul before the rest of the pipeline.
    let cfg = ScaledQuantConfig {
        lhs_format: ScaledFormat::custom(3, 0),
        rhs_format: ScaledFormat::custom(3, 0),
        scale_layout: ScaleLayout::mx(),
    };
    let opts = CompileOptions::new().scaled_quant(cfg);
    let mut c = Session::new(Device::Cpu).compile_with(build(m, k, n), &opts);
    let out = c.run(&[("x", &x), ("w", &w)]).remove(0);

    assert_eq!(out.len(), m * n);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "policy produced non-finite"
    );
    let cos = cosine(&out, &reference);
    eprintln!("scaled_quant f4e3m0 policy: cosine_vs_f32={cos:.4}");
    assert!(cos >= 0.9, "scaled_quant policy cosine {cos} < 0.9");

    // A named fp8 policy should be near-lossless on this smooth data.
    let opts8 = CompileOptions::new().scaled_quant(ScaledQuantConfig::mxfp8_e4m3());
    let mut c8 = Session::new(Device::Cpu).compile_with(build(m, k, n), &opts8);
    let out8 = c8.run(&[("x", &x), ("w", &w)]).remove(0);
    let cos8 = cosine(&out8, &reference);
    eprintln!("scaled_quant mxfp8 policy: cosine_vs_f32={cos8:.5}");
    assert!(cos8 >= 0.99, "mxfp8 policy cosine {cos8} < 0.99");
}
