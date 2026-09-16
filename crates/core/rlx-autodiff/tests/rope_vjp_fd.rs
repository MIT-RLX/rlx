// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Rope` VJP against central finite differences, per sequence position.
//!
//! RoPE is *linear* in `x` — a per-position rotation — so its VJP is the
//! transposed rotation and finite differences are exact up to rounding. That
//! makes this a sharp test: any position-dependent error (a sign flip on the
//! sine term, a pair-offset mistake, a mis-indexed cos/sin row) shows up as a
//! clean mismatch at the positions where the rotation angle is large, and as
//! nothing at position 0 where the rotation is the identity.
//!
//! Both the full-rotation case (`n_rot == head_dim`) and the partial case
//! Qwen3.5/3.6 uses (`n_rot < head_dim`, trailing dims copied through) are
//! covered — the partial path has a tail that must receive gradient unchanged.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::op::RopeStyle;
use rlx_ir::{DType, Graph, Op, Shape};

const BATCH: usize = 1;
const SEQ: usize = 6;
const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const D: usize = HEADS * HEAD_DIM;
const N: usize = BATCH * SEQ * D;
const MAX_POS: usize = 8;

fn rope_graph(n_rot: usize) -> Graph {
    let f = DType::F32;
    let shape = Shape::new(&[BATCH, SEQ, D], f);
    let table = Shape::new(&[MAX_POS, n_rot / 2], f);
    let mut g = Graph::new("rope");
    let x = g.input("x", shape.clone());
    let cos = g.input("cos", table.clone());
    let sin = g.input("sin", table);
    let y = g.add_node(
        Op::Rope {
            head_dim: HEAD_DIM,
            n_rot,
            style: RopeStyle::NeoX,
        },
        vec![x, cos, sin],
        shape,
    );
    g.set_outputs(vec![y]);
    g
}

/// A real RoPE table: `theta_i = 10000^(-2i/n_rot)`, angle = position · theta.
fn tables(n_rot: usize) -> (Vec<f32>, Vec<f32>) {
    let half = n_rot / 2;
    let mut cos = vec![0.0f32; MAX_POS * half];
    let mut sin = vec![0.0f32; MAX_POS * half];
    for p in 0..MAX_POS {
        for i in 0..half {
            let theta = 10000f64.powf(-2.0 * i as f64 / n_rot as f64);
            let angle = p as f64 * theta;
            cos[p * half + i] = angle.cos() as f32;
            sin[p * half + i] = angle.sin() as f32;
        }
    }
    (cos, sin)
}

fn check(n_rot: usize) {
    let (cos, sin) = tables(n_rot);
    let bwd = grad_with_loss_wrt(
        &rope_graph(n_rot),
        &[Wrt::Leaf("x".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );

    let x0: Vec<f32> = (0..N)
        .map(|i| 0.3 * (((i * 7) % 11) as f32 - 5.0) / 5.0)
        .collect();
    let cot: Vec<f32> = (0..N).map(|i| 0.5 + 0.25 * ((i % 7) as f32)).collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    let outs = compiled.run(&[
        ("x", &x0[..]),
        ("cos", &cos[..]),
        ("sin", &sin[..]),
        ("d_output", &cot[..]),
    ]);
    let grad = &outs[1];

    let mut fwd = rlx::Session::new(rlx::Device::Cpu).compile(rope_graph(n_rot));
    let weighted = |fwd: &mut rlx::CompiledGraph, x: &[f32]| -> f64 {
        fwd.run(&[("x", x), ("cos", &cos[..]), ("sin", &sin[..])])[0]
            .iter()
            .zip(&cot)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let eps = 1e-2f32; // RoPE is linear in x, so a large step is exact and low-noise.
    let mut worst_by_pos = vec![0.0f64; SEQ];
    for i in 0..N {
        let mut xp = x0.clone();
        let mut xm = x0.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (weighted(&mut fwd, &xp) - weighted(&mut fwd, &xm)) / (2.0 * eps as f64);
        let pos = i / D;
        worst_by_pos[pos] = worst_by_pos[pos].max((fd - grad[i] as f64).abs());
    }
    eprintln!("n_rot={n_rot}: worst |autodiff - fd| by position = {worst_by_pos:?}");
    for (pos, w) in worst_by_pos.iter().enumerate() {
        assert!(
            *w < 1e-3,
            "n_rot={n_rot} position {pos}: worst |autodiff - fd| = {w}"
        );
    }
    let magnitude = grad.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    assert!(magnitude > 1e-3, "gradient is ~zero — agreement is vacuous");
}

#[test]
fn full_rotation_rope_vjp_matches_finite_differences() {
    check(HEAD_DIM);
}

#[test]
fn partial_rotation_rope_vjp_matches_finite_differences() {
    // n_rot < head_dim: the trailing dims are copied through and must receive
    // their cotangent unchanged.
    check(HEAD_DIM / 2);
}
