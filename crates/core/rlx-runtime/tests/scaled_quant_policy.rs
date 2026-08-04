// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Per-tensor FP8 `ScaledMatMul` on CUDA. This is the NATIVE cuBLASLt FP8
/// tensor-core path — which only exists on Ada (sm_89) / Hopper (sm_90)+. On
/// Ampere (sm_86, e.g. the msi RTX 3080 Ti) the cuBLASLt FP8 GEMM returns
/// NOT_SUPPORTED and USED TO PANIC; it now falls back to the software
/// decode-and-accumulate path. Either way it must run + track f32. On an FP8-TC
/// GPU this validates the tensor-core GEMM numerically. Runs on the msi rig
/// (`RLX_PARITY_DEVICE` selects the device); no-ops without CUDA.
#[test]
fn cuda_per_tensor_fp8_scaled_matmul_runs_and_tracks_f32() {
    let dev = match std::env::var("RLX_PARITY_DEVICE") {
        Ok(s) => rlx_runtime::parse_device(&s).unwrap_or(Device::Cuda),
        Err(_) => Device::Cuda,
    };
    if !rlx_runtime::is_available(dev) {
        eprintln!("skip cuda_per_tensor_fp8 ({dev:?} unavailable)");
        return;
    }
    let (m, k, n) = (8usize, 128usize, 16usize);
    let x: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.11).sin() * 1.3).collect();
    let w: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.05).cos()).collect();

    let reference = Session::new(Device::Cpu)
        .compile(build(m, k, n))
        .run(&[("x", &x), ("w", &w)])
        .remove(0);

    let cfg = ScaledQuantConfig {
        lhs_format: ScaledFormat::F8E4M3,
        rhs_format: ScaledFormat::F8E4M3,
        scale_layout: ScaleLayout::PerTensor,
    };
    let opts = CompileOptions::new().scaled_quant(cfg);
    // MUST NOT panic on non-Ada CUDA (the pre-fix behavior was `.expect`).
    let out = Session::new(dev)
        .compile_with(build(m, k, n), &opts)
        .run(&[("x", &x), ("w", &w)])
        .remove(0);

    assert_eq!(out.len(), m * n);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "per-tensor fp8 non-finite"
    );
    let cos = cosine(&out, &reference);
    eprintln!("[cuda] per-tensor fp8 ScaledMatMul: cosine_vs_f32={cos:.4}");
    assert!(cos >= 0.95, "per-tensor fp8 cosine {cos} < 0.95");
}
