// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ScatterAdd` VJP on a non-zero axis, against finite differences.
//!
//! Scatter-add and gather are transposes of each other **along the same axis**.
//! The rule previously hardcoded `Gather { axis: 0 }`, which stays correctly
//! *shaped* for any axis — so a wrong axis produces a gradient that passes every
//! shape check and every "does it run" test while permuting the values. Finite
//! differences is the only thing that sees it.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

fn sum_sq_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let shape = g.node(y).shape.clone();
    let y2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![y, y], shape);
    let rank = g.node(y2).shape.rank();
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: (0..rank).collect(),
            keep_dim: false,
        },
        vec![y2],
        Shape::from_dims(&[], DType::F32),
    )
}

fn check(upd_dims: &[usize], out_dims: &[usize], axis: usize, idx: &[f32]) {
    let n: usize = upd_dims.iter().product();

    let build = |g: &mut Graph, updates: NodeId| -> NodeId {
        let data: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
        let i = g.add_node(
            Op::Constant { data },
            vec![],
            Shape::new(&[idx.len()], DType::F32),
        );
        g.add_node(
            Op::ScatterAdd { axis },
            vec![updates, i],
            Shape::new(out_dims, DType::F32),
        )
    };

    let mut g = Graph::new("scatter_grad");
    let x = g.param("x", Shape::new(upd_dims, DType::F32));
    let y = build(&mut g, x);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    // Distinct magnitudes so a permuted gradient cannot coincide with the right one.
    let x_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.41 - 1.3).collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    let d_x = compiled.run(&[("d_output", &[1.0f32])])[1].clone();
    assert_eq!(d_x.len(), n);

    let loss_at = |xv: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let xi = fg.input("x", Shape::new(upd_dims, DType::F32));
        let y = build(&mut fg, xi);
        let loss = sum_sq_loss(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("x", xv)])
            .pop()
            .unwrap()[0]
    };

    let eps = 1e-3f32;
    for i in 0..n {
        let mut xp = x_init.clone();
        let mut xm = x_init.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (loss_at(&xp) - loss_at(&xm)) / (2.0 * eps);
        assert!(
            (fd - d_x[i]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "scatter grad[{i}] upd={upd_dims:?} out={out_dims:?} axis={axis}: \
             analytic {} vs FD {fd}",
            d_x[i]
        );
    }
}

#[test]
fn scatter_add_vjp_axis_zero() {
    check(&[4, 3], &[6, 3], 0, &[0.0, 2.0, 2.0, 5.0]);
}

#[test]
fn scatter_add_vjp_axis_one() {
    check(&[3, 4], &[3, 6], 1, &[1.0, 1.0, 4.0, 0.0]);
}

#[test]
fn scatter_add_vjp_rank_three_every_axis() {
    for axis in 0..3 {
        let mut upd = [2usize, 3, 4];
        let mut out = [2usize, 3, 4];
        upd[axis] = 3;
        out[axis] = 5;
        check(&upd, &out, axis, &[4.0, 0.0, 4.0]);
    }
}

/// With duplicate indices several updates share a destination, so the gradient
/// of each is the same upstream value — the case where a gather-based transpose
/// is doing real work rather than a permutation.
#[test]
fn scatter_add_vjp_with_duplicate_indices() {
    check(&[4, 2], &[3, 2], 0, &[1.0, 1.0, 1.0, 1.0]);
    check(&[2, 4], &[2, 3], 1, &[2.0, 2.0, 0.0, 2.0]);
}
