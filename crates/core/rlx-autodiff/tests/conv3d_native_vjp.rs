// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native `Op::Conv3d` / `Op::ConvTranspose3d` VJP vs finite differences (CPU).

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const XS: [usize; 5] = [1, 1, 3, 3, 3];
const WS: [usize; 5] = [1, 1, 2, 2, 2];
const CT_WS: [usize; 5] = [1, 1, 2, 2, 2]; // [C_in, C_out, kD, kH, kW]

fn sum_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let rank = g.node(y).shape.rank();
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: (0..rank).collect(),
            keep_dim: false,
        },
        vec![y],
        Shape::from_dims(&[], DType::F32),
    )
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label} len");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{label}[{i}]: got {a} want {b} (tol {tol})"
        );
    }
}

#[test]
fn conv3d_vjp_matches_fd() {
    let mut g = Graph::new("c3d");
    let x = g.param("x", Shape::new(&XS, DType::F32));
    let w = g.param("w", Shape::new(&WS, DType::F32));
    let y = g.conv3d(x, w, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x, w]);
    let x_init: Vec<f32> = (0..XS.iter().product::<usize>())
        .map(|i| (i as f32) * 0.1 - 0.5)
        .collect();
    let w_init: Vec<f32> = (0..WS.iter().product::<usize>())
        .map(|i| (i as f32) * 0.05 - 0.2)
        .collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    compiled.set_param("w", &w_init);
    let outs = compiled.run(&[("d_output", &[1.0f32])]);
    assert!(outs.len() >= 3, "expected loss + d_x + d_w");
    let d_x = &outs[1];
    let d_w = &outs[2];

    let loss_at = |xv: &[f32], wv: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let xi = fg.input("x", Shape::new(&XS, DType::F32));
        let wi = fg.input("w", Shape::new(&WS, DType::F32));
        let y = fg.conv3d(xi, wi, [1, 1, 1], [0, 0, 0], [1, 1, 1], 1);
        let loss = sum_loss(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("x", xv), ("w", wv)])
            .pop()
            .unwrap()[0]
    };

    let eps = 1e-3f32;
    let mut fd_x = vec![0f32; x_init.len()];
    for i in 0..x_init.len() {
        let mut p = x_init.clone();
        let mut m = x_init.clone();
        p[i] += eps;
        m[i] -= eps;
        fd_x[i] = (loss_at(&p, &w_init) - loss_at(&m, &w_init)) / (2.0 * eps);
    }
    let mut fd_w = vec![0f32; w_init.len()];
    for i in 0..w_init.len() {
        let mut p = w_init.clone();
        let mut m = w_init.clone();
        p[i] += eps;
        m[i] -= eps;
        fd_w[i] = (loss_at(&x_init, &p) - loss_at(&x_init, &m)) / (2.0 * eps);
    }

    assert_close(d_x, &fd_x, 2e-2, "conv3d d_x");
    assert_close(d_w, &fd_w, 2e-2, "conv3d d_w");
}

#[test]
fn conv_transpose3d_vjp_matches_fd() {
    let mut g = Graph::new("ct3d");
    let x = g.param("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
    let w = g.param("w", Shape::new(&CT_WS, DType::F32));
    let y = g.conv_transpose3d(x, w, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x, w]);
    let x_init: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w_init: Vec<f32> = (0..8).map(|i| (i as f32) * 0.05 - 0.15).collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    compiled.set_param("w", &w_init);
    let outs = compiled.run(&[("d_output", &[1.0f32])]);
    let d_x = &outs[1];
    let d_w = &outs[2];

    let loss_at = |xv: &[f32], wv: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let xi = fg.input("x", Shape::new(&[1, 1, 2, 2, 2], DType::F32));
        let wi = fg.input("w", Shape::new(&CT_WS, DType::F32));
        let y = fg.conv_transpose3d(xi, wi, [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], 1);
        let loss = sum_loss(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("x", xv), ("w", wv)])
            .pop()
            .unwrap()[0]
    };

    let eps = 1e-3f32;
    let mut fd_x = vec![0f32; x_init.len()];
    for i in 0..x_init.len() {
        let mut p = x_init.clone();
        let mut m = x_init.clone();
        p[i] += eps;
        m[i] -= eps;
        fd_x[i] = (loss_at(&p, &w_init) - loss_at(&m, &w_init)) / (2.0 * eps);
    }
    let mut fd_w = vec![0f32; w_init.len()];
    for i in 0..w_init.len() {
        let mut p = w_init.clone();
        let mut m = w_init.clone();
        p[i] += eps;
        m[i] -= eps;
        fd_w[i] = (loss_at(&x_init, &p) - loss_at(&x_init, &m)) / (2.0 * eps);
    }

    assert_close(d_x, &fd_x, 3e-2, "conv_transpose3d d_x");
    assert_close(d_w, &fd_w, 3e-2, "conv_transpose3d d_w");
}
