// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Roll` VJP vs finite differences. Loss `sum(roll(x)^2)`.
//!
//! A roll is an orthogonal permutation, so its transpose is its inverse — the
//! roll by the negated shifts. That makes a sign error in the rule completely
//! self-consistent (the gradient still has the right shape and norm, just the
//! wrong elements), so it is invisible to anything but an external oracle.
//! Finite differences is that oracle here, for the same reason it was what
//! caught the RoPE table-stride bug rather than cross-backend parity.

use rlx_autodiff::grad_with_loss;
use rlx_ir::infer::GraphExt;
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

fn check(dims: &[usize], shifts: &[i64], axes: &[usize]) {
    let n: usize = dims.iter().product();
    let mut g = Graph::new("roll_grad");
    let x = g.param("x", Shape::new(dims, DType::F32));
    let y = g.roll_(x, shifts.to_vec(), axes.to_vec());
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    // Distinct magnitudes per element: with `sum(y^2)` the gradient is `2·y`
    // permuted, so a mis-permuted gradient can only match by coincidence if the
    // values are distinct.
    let x_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 1.1).collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    let outs = compiled.run(&[("d_output", &[1.0f32])]);
    let d_x = outs[1].clone();
    assert_eq!(d_x.len(), n);

    let loss_at = |xv: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let xi = fg.input("x", Shape::new(dims, DType::F32));
        let y = fg.roll_(xi, shifts.to_vec(), axes.to_vec());
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
            "roll grad[{i}] dims={dims:?} shifts={shifts:?} axes={axes:?}: \
             analytic {} vs FD {fd}",
            d_x[i]
        );
    }
}

#[test]
fn roll_vjp_1d() {
    check(&[6], &[1], &[0]);
    check(&[6], &[-2], &[0]);
    check(&[6], &[5], &[0]);
    check(&[6], &[0], &[0]); // identity
    check(&[6], &[6], &[0]); // full period
    check(&[6], &[13], &[0]); // wraps twice
}

#[test]
fn roll_vjp_2d_each_axis() {
    check(&[3, 4], &[1], &[0]);
    check(&[3, 4], &[-1], &[0]);
    check(&[3, 4], &[2], &[1]);
    check(&[3, 4], &[-3], &[1]);
}

#[test]
fn roll_vjp_multi_axis() {
    check(&[2, 3, 4], &[1, -1, 2], &[0, 1, 2]);
    check(&[2, 3], &[1, 1], &[0, 1]);
}

/// The gradient of a permutation must itself be a permutation of `2·x`.
///
/// Independent of finite differences: it checks the *structure* of the answer,
/// so it fails on a gradient that is smoothly wrong (e.g. scaled) as well as
/// one that is mis-permuted.
#[test]
fn roll_grad_is_a_permutation_of_two_x() {
    let dims = [8usize];
    let n = 8;
    let mut g = Graph::new("roll_grad_perm");
    let x = g.param("x", Shape::new(&dims, DType::F32));
    let y = g.roll_(x, vec![3], vec![0]);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    let x_init: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    let d_x = compiled.run(&[("d_output", &[1.0f32])])[1].clone();

    // d/dx sum(roll(x)^2) = 2x exactly: rolling does not change which element
    // each x_i becomes, only where it lands, and the loss sums over all of them.
    for i in 0..n {
        assert!(
            (d_x[i] - 2.0 * x_init[i]).abs() < 1e-4,
            "grad[{i}] = {} want {}",
            d_x[i],
            2.0 * x_init[i]
        );
    }
}
