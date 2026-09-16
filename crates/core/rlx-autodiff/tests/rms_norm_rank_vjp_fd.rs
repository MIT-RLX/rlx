// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::RmsNorm` VJP against central finite differences, by input rank.
//!
//! Qwen3-family models apply RMSNorm **per attention head** — on a rank-4
//! `[batch, seq, heads, head_dim]` tensor with `axis = -1` — not just to the
//! rank-3 residual stream. A gradient rule that assumes the normalized axis is
//! the only non-batch axis produces a plausible-looking but wrong `dx` there:
//! the forward is fine, so nothing downstream complains.
//!
//! Rank 2 and 3 are included as controls — if those pass and rank 4 fails, the
//! defect is in how the rule folds the leading axes, not in the RMSNorm
//! derivative itself.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::{DType, Graph, Op, Shape};

/// `y = rms_norm(x, gamma, beta, axis = -1)` over `dims`.
fn rms_norm_graph(dims: &[usize]) -> Graph {
    let f = DType::F32;
    let norm_dim = *dims.last().expect("at least one dim");
    let shape = Shape::new(dims, f);
    let mut g = Graph::new("rmsnorm");
    let x = g.input("x", shape.clone());
    let gamma = g.param("gamma", Shape::new(&[norm_dim], f));
    let beta = g.param("beta", Shape::new(&[norm_dim], f));
    let y = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x, gamma, beta],
        shape,
    );
    g.set_outputs(vec![y]);
    g
}

fn check(dims: &[usize]) -> f64 {
    let n: usize = dims.iter().product();
    let norm_dim = *dims.last().unwrap();

    let x0: Vec<f32> = (0..n)
        .map(|i| 0.35 * (((i * 7) % 11) as f32 - 5.0) / 5.0)
        .collect();
    let gamma: Vec<f32> = (0..norm_dim).map(|i| 0.8 + 0.1 * i as f32).collect();
    let beta: Vec<f32> = (0..norm_dim).map(|i| 0.05 * i as f32).collect();
    let cot: Vec<f32> = (0..n).map(|i| 0.5 + 0.25 * ((i % 7) as f32)).collect();

    let bwd = grad_with_loss_wrt(
        &rms_norm_graph(dims),
        &[Wrt::Leaf("x".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("gamma", &gamma);
    compiled.set_param("beta", &beta);
    let grad = compiled.run(&[("x", &x0[..]), ("d_output", &cot[..])])[1].clone();

    let mut fwd = rlx::Session::new(rlx::Device::Cpu).compile(rms_norm_graph(dims));
    fwd.set_param("gamma", &gamma);
    fwd.set_param("beta", &beta);
    let weighted = |fwd: &mut rlx::CompiledGraph, x: &[f32]| -> f64 {
        fwd.run(&[("x", x)])[0]
            .iter()
            .zip(&cot)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let eps = 1e-3f32;
    let mut worst = 0.0f64;
    for i in 0..n {
        let mut xp = x0.clone();
        let mut xm = x0.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (weighted(&mut fwd, &xp) - weighted(&mut fwd, &xm)) / (2.0 * eps as f64);
        worst = worst.max((fd - grad[i] as f64).abs());
    }
    eprintln!("dims {dims:?}: worst |autodiff - fd| = {worst:.6}");
    worst
}

#[test]
fn rms_norm_vjp_matches_finite_differences_rank2() {
    assert!(check(&[6, 4]) < 1e-3);
}

#[test]
fn rms_norm_vjp_matches_finite_differences_rank3() {
    assert!(check(&[1, 6, 4]) < 1e-3);
}

/// The Qwen3 per-head Q/K norm shape.
#[test]
fn rms_norm_vjp_matches_finite_differences_rank4() {
    assert!(check(&[1, 6, 4, 4]) < 1e-3);
}

/// Rank 4 with a non-trivial leading batch — a rule that collapses only one
/// leading axis passes `[1, …]` and fails here.
#[test]
fn rms_norm_vjp_matches_finite_differences_rank4_batched() {
    assert!(check(&[2, 3, 4, 4]) < 1e-3);
}
