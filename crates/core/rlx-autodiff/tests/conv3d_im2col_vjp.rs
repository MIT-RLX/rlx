// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Graph::conv3d_im2col` — an all-backend, pure-decomposition 3-D conv.
//!
//! TEST 1 (forward parity): the im2col decomposition matches the native CPU
//! `Op::Conv3d` kernel (stride-1 same-pad and strided cases).
//!
//! TEST 2 (gradient check): the decomposition's autodiff-for-free `dx`/`dw`
//! (`grad_with_loss`) matches central finite differences.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// Deterministic pseudo-random f32 in roughly `[-1, 1]` (no `rand` dep).
fn seq(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..len)
        .map(|_| {
            // xorshift32
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

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
    let mut max_abs = 0f32;
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let e = (a - b).abs();
        max_abs = max_abs.max(e);
        assert!(
            e <= tol,
            "{label}[{i}]: got {a} want {b} (|err| {e} > tol {tol}); max_abs so far {max_abs}"
        );
    }
}

/// TEST 1 — forward parity vs the native `Op::Conv3d` CPU kernel.
#[test]
fn conv3d_im2col_matches_native_conv3d() {
    const XS: [usize; 5] = [1, 2, 5, 5, 5]; // [N, C_in, D, H, W]
    const WS: [usize; 5] = [3, 2, 3, 3, 3]; // [C_out, C_in, kD, kH, kW]

    let x_init = seq(XS.iter().product(), 1);
    let w_init = seq(WS.iter().product(), 2);

    let cases: [([usize; 3], [usize; 3], [usize; 3]); 2] = [
        ([1, 1, 1], [1, 1, 1], [1, 1, 1]),
        ([2, 2, 2], [1, 1, 1], [1, 1, 1]),
    ];

    for (stride, padding, dilation) in cases {
        let mut g = Graph::new("conv3d_parity");
        let x = g.input("x", Shape::new(&XS, DType::F32));
        let w = g.input("w", Shape::new(&WS, DType::F32));
        let y_ref = g.conv3d(x, w, stride, padding, dilation, 1);
        let y_im = g.conv3d_im2col(x, w, stride, padding, dilation, 1);
        // Sanity: both builders infer the same output shape.
        assert_eq!(
            g.node(y_ref).shape.dims(),
            g.node(y_im).shape.dims(),
            "shape mismatch for stride {stride:?}"
        );
        g.set_outputs(vec![y_ref, y_im]);

        let outs = rlx::Session::new(rlx::Device::Cpu)
            .compile(g)
            .run(&[("x", &x_init), ("w", &w_init)]);
        assert!(outs.len() >= 2, "expected two outputs");
        assert_close(
            &outs[1],
            &outs[0],
            1e-4,
            &format!("conv3d_im2col fwd (stride {stride:?})"),
        );
    }
}

/// TEST 2 — `dx`/`dw` via `grad_with_loss` vs central finite differences.
#[test]
fn conv3d_im2col_vjp_matches_fd() {
    const XS: [usize; 5] = [1, 2, 4, 4, 4]; // [N, C_in, D, H, W]
    const WS: [usize; 5] = [2, 2, 3, 3, 3]; // [C_out, C_in, kD, kH, kW]
    const STRIDE: [usize; 3] = [1, 1, 1];
    const PAD: [usize; 3] = [1, 1, 1];
    const DIL: [usize; 3] = [1, 1, 1];

    let mut g = Graph::new("conv3d_im2col_bwd");
    let x = g.param("x", Shape::new(&XS, DType::F32));
    let w = g.param("w", Shape::new(&WS, DType::F32));
    let y = g.conv3d_im2col(x, w, STRIDE, PAD, DIL, 1);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x, w]);
    let x_init = seq(XS.iter().product(), 7);
    let w_init = seq(WS.iter().product(), 9);

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    compiled.set_param("w", &w_init);
    let outs = compiled.run(&[("d_output", &[1.0f32])]);
    assert!(outs.len() >= 3, "expected loss + d_x + d_w");
    let d_x = &outs[1];
    let d_w = &outs[2];

    // Forward-only loss for finite differences.
    let loss_at = |xv: &[f32], wv: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let xi = fg.input("x", Shape::new(&XS, DType::F32));
        let wi = fg.input("w", Shape::new(&WS, DType::F32));
        let y = fg.conv3d_im2col(xi, wi, STRIDE, PAD, DIL, 1);
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

    assert_close(d_x, &fd_x, 2e-2, "conv3d_im2col d_x");
    assert_close(d_w, &fd_w, 2e-2, "conv3d_im2col d_w");
}
