// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::ConvTranspose2d` VJP (dx, dw) vs finite differences — strided case
//! (stride 2), the one that motivates transpose-conv upsampling.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const XS: [usize; 4] = [1, 2, 3, 3]; // [N, C_in, H, W]
const WS: [usize; 4] = [2, 3, 2, 2]; // [C_in, C_out, kH, kW] (transpose layout)

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

fn convt(g: &mut Graph, x: NodeId, w: NodeId) -> NodeId {
    g.conv_transpose2d(x, w, [2, 2], [2, 2], [0, 0], [1, 1], [0, 0], 1)
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
fn conv_transpose2d_vjp_matches_fd() {
    let mut g = Graph::new("ct");
    let x = g.param("x", Shape::new(&XS, DType::F32));
    let w = g.param("w", Shape::new(&WS, DType::F32));
    let y = convt(&mut g, x, w);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x, w]);
    let x_init: Vec<f32> = (0..XS.iter().product::<usize>())
        .map(|i| (i as f32) * 0.1 - 0.4)
        .collect();
    let w_init: Vec<f32> = (0..WS.iter().product::<usize>())
        .map(|i| (i as f32) * 0.05 - 0.25)
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
        let y = convt(&mut fg, xi, wi);
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

    assert_close(d_x, &fd_x, 2e-2, "conv_transpose2d d_x");
    assert_close(d_w, &fd_w, 2e-2, "conv_transpose2d d_w");
}
