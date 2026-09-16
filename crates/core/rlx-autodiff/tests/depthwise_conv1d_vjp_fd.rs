// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Depthwise causal `Op::Conv` VJP against central finite differences.
//!
//! Qwen3.5/3.6 gated-delta-net blocks carry a short causal depthwise conv over
//! the sequence, expressed as NCHW with the sequence length in `H` and the
//! causal padding materialized into the input: `[1, C, L+K-1, 1]` convolved
//! with `[C, 1, K, 1]` at `groups = C` and zero padding. This is the only op in
//! that block that mixes positions besides the recurrent scan itself.
//!
//! Conv is *linear* in its input, so finite differences are exact here up to
//! rounding, and the gradient must reach **every** input position — position
//! `j` feeds outputs `j-K+1 ..= j`. A rule that only credits the aligned
//! position would leave the model's cross-position sensitivity with no gradient
//! path at all, which no forward test would notice.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::{DType, Graph, Op, Shape};

const C: usize = 3;
const K: usize = 4;
const L_OUT: usize = 5;
const L_IN: usize = L_OUT + K - 1;
const N_IN: usize = C * L_IN;
const N_W: usize = C * K;
const N_OUT: usize = C * L_OUT;

fn conv_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("dwconv");
    let x = g.input("x", Shape::new(&[1, C, L_IN, 1], f));
    let w = g.param("w", Shape::new(&[C, 1, K, 1], f));
    let y = g.add_node(
        Op::Conv {
            kernel_size: vec![K, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: C,
        },
        vec![x, w],
        Shape::new(&[1, C, L_OUT, 1], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

#[test]
fn depthwise_conv_vjp_matches_finite_differences() {
    let x0: Vec<f32> = (0..N_IN).map(|i| 0.5 * hashed(1, i)).collect();
    let w: Vec<f32> = (0..N_W).map(|i| 0.5 * hashed(2, i)).collect();
    let cot: Vec<f32> = (0..N_OUT).map(|i| 0.5 + 0.25 * ((i % 5) as f32)).collect();

    let bwd = grad_with_loss_wrt(
        &conv_graph(),
        &[Wrt::Leaf("x".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("w", &w);
    let grad = compiled.run(&[("x", &x0[..]), ("d_output", &cot[..])])[1].clone();
    assert_eq!(grad.len(), N_IN);

    let mut fwd = rlx::Session::new(rlx::Device::Cpu).compile(conv_graph());
    fwd.set_param("w", &w);
    let mut probe = |x: &[f32]| -> f64 {
        fwd.run(&[("x", x)])[0]
            .iter()
            .zip(&cot)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let eps = 1e-2f32; // linear in x, so a large step is exact and low-noise
    let mut worst = 0.0f64;
    let mut zero_but_sensitive = Vec::new();
    for i in 0..N_IN {
        let mut xp = x0.clone();
        let mut xm = x0.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (probe(&xp) - probe(&xm)) / (2.0 * eps as f64);
        worst = worst.max((fd - grad[i] as f64).abs());
        if grad[i] == 0.0 && fd.abs() > 1e-3 {
            zero_but_sensitive.push((i, fd));
        }
    }
    assert!(
        zero_but_sensitive.is_empty(),
        "VJP is exactly zero at input positions the output genuinely depends on: {zero_but_sensitive:?}"
    );
    assert!(
        worst < 1e-3,
        "worst |autodiff - finite difference| = {worst}"
    );
    eprintln!("depthwise conv dx: worst FD delta = {worst:.2e}");
}

/// Every input position must receive gradient — the interior ones feed `K`
/// outputs each, the edges fewer, but none zero.
#[test]
fn depthwise_conv_vjp_reaches_every_input_position() {
    let x0: Vec<f32> = (0..N_IN).map(|i| 0.5 * hashed(1, i)).collect();
    // All-ones weights and cotangent: every input position then has a strictly
    // positive gradient, so a zero is unambiguously a missing path.
    let w = vec![1.0f32; N_W];
    let cot = [1.0f32; N_OUT];

    let bwd = grad_with_loss_wrt(
        &conv_graph(),
        &[Wrt::Leaf("x".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("w", &w);
    let grad = compiled.run(&[("x", &x0[..]), ("d_output", &cot[..])])[1].clone();

    for c in 0..C {
        for l in 0..L_IN {
            let g = grad[c * L_IN + l];
            assert!(
                g > 0.0,
                "channel {c} position {l} received gradient {g}; every input position \
                 feeds at least one output"
            );
        }
    }
    eprintln!("depthwise conv dx reaches all {N_IN} input positions");
}
