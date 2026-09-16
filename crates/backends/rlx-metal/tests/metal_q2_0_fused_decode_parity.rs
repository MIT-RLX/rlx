// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The fused Q2_0 decode kernels must match an exact host reference,
//! in every fusion and inner-loop configuration.
//!
//! `metal_fused_decode_mlp_parity.rs` covers this shape for Q4_K and Q5_0
//! only, so `q2_0_dual_mv_f32_sg`, `q2_0_swiglu_mv_f32_sg` and
//! `q2_0_mv_residual_f32_sg` had no test at all — they are reached solely by
//! pattern-fusion of a decode MLP, which the Q4_K/Q5_0 cases never trigger for
//! Q2_0 weights. That matters now that all four Q2_0 GEMVs share one 16-bit
//! `q2_0_dot16` inner loop: a mistake in it would be caught only in
//! `q2_0_mv_f32_sg` without this file.
//!
//! The reference is computed **on the host in exact f32**, not by running the
//! graph on `Device::Cpu`. rlx-cpu's `m == 1` Q2_0 GEMV quantizes activations
//! to int8 (`q2_0_dot_q8_block`, llama.cpp-style) and is deliberately
//! approximate — against this graph it lands 2.2% off, because the SwiGLU
//! product feeding `down` has a wide dynamic range. Comparing Metal to that
//! would either hide a real kernel bug behind a loose tolerance or fail for
//! reasons that have nothing to do with the kernel.
//!
//! Each variant runs in its own `#[test]` because the env switches are
//! process-wide and read once:
//!   * fused (default),
//!   * every Q2_0 fusion disabled (plain `mv` kernels),
//!   * the scalar byte inner loop (`RLX_METAL_Q2_0_SCALAR=1`) — the code the
//!     16-bit path replaced, so the two are checked against one reference.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use rlx_runtime::{Device, Session};

const H: usize = 512;
const I: usize = 1024;

fn weight(seed: f32, k: usize, n: usize) -> Vec<u8> {
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * seed).sin() * 0.5)
        .collect();
    rlx_gguf::quantize(&w, rlx_gguf::GgmlType::Q2_0).expect("quantize Q2_0")
}

/// RmsNorm → (gate, up) → SwiGLU → down → residual add: the decode MLP shape
/// the Metal backend pattern-fuses into `q2_0_swiglu_mv` / `q2_0_dual_mv` /
/// `q2_0_mv_residual`.
fn build(lens: (usize, usize, usize)) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("q2_0_fused_decode");
    let r = g.input("r", Shape::new(&[1, H], f));
    let gamma = g.input("gamma", Shape::new(&[H], f));
    let beta = g.input("beta", Shape::new(&[H], f));
    let gate_w = g.param("gate_w", Shape::new(&[lens.0], DType::U8));
    let up_w = g.param("up_w", Shape::new(&[lens.1], DType::U8));
    let down_w = g.param("down_w", Shape::new(&[lens.2], DType::U8));
    let dq = |g: &mut Graph, x: NodeId, w: NodeId, n: usize| {
        g.add_node(
            Op::DequantMatMul {
                scheme: QuantScheme::GgufQ2_0,
            },
            vec![x, w],
            Shape::new(&[1, n], DType::F32),
        )
    };

    let normed = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-5,
        },
        vec![r, gamma, beta],
        Shape::new(&[1, H], f),
    );
    let gate = dq(&mut g, normed, gate_w, I);
    let up = dq(&mut g, normed, up_w, I);
    let gate_act = g.add_node(
        Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[1, I], f),
    );
    let prod = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![gate_act, up],
        Shape::new(&[1, I], f),
    );
    let down = dq(&mut g, prod, down_w, H);
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![r, down],
        Shape::new(&[1, H], f),
    );
    g.set_outputs(vec![out]);
    g
}

fn weights() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        weight(0.013, H, I),
        weight(0.021, H, I),
        weight(0.017, I, H),
    )
}

fn residual() -> Vec<f32> {
    (0..H).map(|i| ((i as f32) * 0.03).cos() * 0.4).collect()
}

fn run(device: Device) -> Vec<f32> {
    let (gate_p, up_p, down_p) = weights();
    let g = build((gate_p.len(), up_p.len(), down_p.len()));
    let mut c = Session::new(device).compile(g);
    c.set_param_typed("gate_w", &gate_p, DType::U8);
    c.set_param_typed("up_w", &up_p, DType::U8);
    c.set_param_typed("down_w", &down_p, DType::U8);
    let r = residual();
    let gamma = vec![1.0f32; H];
    let beta = vec![0.0f32; H];
    c.run(&[
        ("r", r.as_slice()),
        ("gamma", gamma.as_slice()),
        ("beta", beta.as_slice()),
    ])
    .remove(0)
}

/// `x @ Wᵀ` with `W` given as packed Q2_0 `[n, k]`, in exact f32.
fn dq_matmul(x: &[f32], packed: &[u8], k: usize, n: usize) -> Vec<f32> {
    let w = rlx_gguf::q2_dequant::dequant_q2_0(packed, n * k).expect("dequant");
    (0..n)
        .map(|c| (0..k).map(|j| x[j] * w[c * k + j]).sum())
        .collect()
}

/// The whole graph, on the host, in exact f32.
fn reference(gate_p: &[u8], up_p: &[u8], down_p: &[u8], r: &[f32]) -> Vec<f32> {
    let ms = r.iter().map(|v| v * v).sum::<f32>() / H as f32;
    let inv = 1.0 / (ms + 1e-5).sqrt();
    let normed: Vec<f32> = r.iter().map(|v| v * inv).collect();
    let gate = dq_matmul(&normed, gate_p, H, I);
    let up = dq_matmul(&normed, up_p, H, I);
    let prod: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();
    let down = dq_matmul(&prod, down_p, I, H);
    r.iter().zip(&down).map(|(a, b)| a + b).collect()
}

fn check(label: &str) {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (gate_p, up_p, down_p) = weights();
    let r = residual();
    let want = reference(&gate_p, &up_p, &down_p, &r);
    let got = run(Device::Metal);
    assert_eq!(want.len(), got.len(), "{label}: output length");
    assert!(
        got.iter().any(|v| *v != 0.0),
        "{label}: Metal returned all zeros"
    );
    let scale = want
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // Same arithmetic, different summation order (simd_sum tree vs serial),
    // so only f32 reassociation should separate them.
    assert!(
        worst / scale < 1e-4,
        "{label}: Metal vs exact host rel |Δ| = {} (worst {worst}, scale {scale})",
        worst / scale
    );
}

#[test]
fn q2_0_fused_decode_mlp_matches_reference() {
    check("fused");
}

/// Same graph with every Q2_0 decode fusion off, so the plain `mv` kernels
/// carry it. Guards the fallback the fused paths are compared against.
#[test]
fn q2_0_unfused_decode_mlp_matches_reference() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // SAFETY: single-threaded test process; read once at first encode.
    unsafe {
        std::env::set_var("RLX_METAL_Q2_0_FUSED_DISABLE", "1");
        std::env::set_var("RLX_METAL_Q2_DUAL_DISABLE", "1");
    }
    check("fusions disabled");
}

/// Same graph forced onto the scalar byte inner loop — the code the 16-bit
/// `q2_0_dot16` replaced. If the two disagree, the rewrite is wrong.
#[test]
fn q2_0_scalar_inner_loop_matches_reference() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // SAFETY: single-threaded test process; read once at first encode.
    unsafe {
        std::env::set_var("RLX_METAL_Q2_0_SCALAR", "1");
    }
    check("scalar inner loop");
}
