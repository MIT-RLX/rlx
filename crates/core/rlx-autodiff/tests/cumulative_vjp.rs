// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::{CumProd, CumMax}` VJP vs finite differences (loss = sum(op(x)^2)).
//! CumProd uses strictly-positive inputs (the `/x` grad form); CumMax uses
//! distinct values so the argmax routing is unambiguous.

use rlx_autodiff::{grad_with_loss, jvp};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

fn sum_sq(g: &mut Graph, y: NodeId) -> NodeId {
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

fn fd_check(dims: &[usize], build: impl Fn(&mut Graph, NodeId) -> NodeId, x_init: &[f32]) {
    let n: usize = dims.iter().product();
    let mut g = Graph::new("s");
    let x = g.param("x", Shape::new(dims, DType::F32));
    let y = build(&mut g, x);
    let loss = sum_sq(&mut g, y);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[x]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("x", x_init);
    let d = c.run(&[("d_output", &[1.0f32])])[1].clone();

    let loss_at = |xv: &[f32]| -> f32 {
        let mut fg = Graph::new("f");
        let xi = fg.input("x", Shape::new(dims, DType::F32));
        let y = build(&mut fg, xi);
        let loss = sum_sq(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("x", xv)])
            .pop()
            .unwrap()[0]
    };
    let eps = 1e-3f32;
    for i in 0..n {
        let mut xp = x_init.to_vec();
        let mut xm = x_init.to_vec();
        xp[i] += eps;
        xm[i] -= eps;
        let fd = (loss_at(&xp) - loss_at(&xm)) / (2.0 * eps);
        assert!(
            (fd - d[i]).abs() <= 3e-2 * (1.0 + fd.abs()),
            "grad[{i}]: analytic {} vs FD {fd}",
            d[i]
        );
    }
}

#[test]
fn cumprod_vjp_inclusive() {
    let x = vec![1.3f32, 0.8, 1.5, 0.9, 1.1, 0.7];
    fd_check(&[2, 3], |g, x| g.cumprod_(x, -1, false), &x);
}

#[test]
fn cumprod_vjp_exclusive() {
    let x = vec![1.2f32, 0.9, 1.4, 0.85, 1.05, 0.75];
    fd_check(&[2, 3], |g, x| g.cumprod_(x, 1, true), &x);
}

#[test]
fn cummax_vjp_inclusive() {
    // Distinct values → unambiguous argmax routing.
    let x = vec![
        1.0f32, 3.0, 2.0, 5.0, 4.0, 6.5, -1.0, -3.0, 0.5, -2.0, 2.5, 7.0,
    ];
    fd_check(&[2, 6], |g, x| g.cummax_(x, -1, false), &x);
}

/// Forward-mode: tangent output of `jvp` vs a finite-difference directional
/// derivative of the primal in direction `t`.
fn jvp_check(dims: &[usize], build: impl Fn(&mut Graph, NodeId) -> NodeId, x: &[f32], t: &[f32]) {
    let mut g = Graph::new("j");
    let xin = g.input("x", Shape::new(dims, DType::F32));
    let y = build(&mut g, xin);
    g.set_outputs(vec![y]);
    let jg = jvp(&g, &[xin]);
    let out = rlx::Session::new(rlx::Device::Cpu)
        .compile(jg)
        .run(&[("x", x), ("tangent_x", t)]);
    let tan = out[1].clone();

    let fwd = |xv: &[f32]| -> Vec<f32> {
        let mut fg = Graph::new("f");
        let xi = fg.input("x", Shape::new(dims, DType::F32));
        let y = build(&mut fg, xi);
        fg.set_outputs(vec![y]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("x", xv)])
            .pop()
            .unwrap()
    };
    let eps = 1e-3f32;
    let xp: Vec<f32> = x.iter().zip(t).map(|(a, b)| a + eps * b).collect();
    let xm: Vec<f32> = x.iter().zip(t).map(|(a, b)| a - eps * b).collect();
    let (yp, ym) = (fwd(&xp), fwd(&xm));
    for i in 0..tan.len() {
        let fd = (yp[i] - ym[i]) / (2.0 * eps);
        assert!(
            (fd - tan[i]).abs() <= 3e-2 * (1.0 + fd.abs()),
            "tangent[{i}]: analytic {} vs FD {fd}",
            tan[i]
        );
    }
}

#[test]
fn cumprod_jvp_inclusive() {
    let x = vec![1.3f32, 0.8, 1.5, 0.9, 1.1, 0.7];
    let t = vec![0.5f32, -0.3, 0.2, 0.7, -0.4, 0.6];
    jvp_check(&[2, 3], |g, x| g.cumprod_(x, -1, false), &x, &t);
}

#[test]
fn cummax_jvp_inclusive() {
    let x = vec![
        1.0f32, 3.0, 2.0, 5.0, 4.0, 6.5, -1.0, -3.0, 0.5, -2.0, 2.5, 7.0,
    ];
    let t = vec![
        0.4f32, -0.5, 0.3, 0.9, -0.2, 0.6, 0.1, -0.7, 0.8, -0.3, 0.5, -0.6,
    ];
    jvp_check(&[2, 6], |g, x| g.cummax_(x, -1, false), &x, &t);
}
