// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Pad` VJP vs finite differences for every differentiable mode. Loss is
//! `sum(pad(x)^2)`, whose gradient exposes the fold multiplicities (edge/
//! reflected/wrapped positions accumulate more than the interior).

use rlx_autodiff::grad_with_loss;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, PadMode, Shape};

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

fn check_mode(dims: &[usize], pads: Vec<[usize; 2]>, mode: PadMode) {
    let n: usize = dims.iter().product();
    let mut g = Graph::new("pad_grad");
    let x = g.param("x", Shape::new(dims, DType::F32));
    let y = g.pad_(x, pads.clone(), mode);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    let x_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 0.7).collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    let outs = compiled.run(&[("d_output", &[1.0f32])]);
    let d_x = outs[1].clone();
    assert_eq!(d_x.len(), n, "{mode:?}: grad length");

    let loss_at = |xv: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let xi = fg.input("x", Shape::new(dims, DType::F32));
        let y = fg.pad_(xi, pads.clone(), mode);
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
            "{mode:?} grad[{i}]: analytic {} vs FD {fd}",
            d_x[i]
        );
    }
}

#[test]
fn pad_vjp_1d_all_modes() {
    for mode in [
        PadMode::Constant(0.5),
        PadMode::Reflect,
        PadMode::Replicate,
        PadMode::Circular,
    ] {
        check_mode(&[4], vec![[2, 2]], mode);
    }
}

#[test]
fn pad_vjp_2d_all_modes() {
    for mode in [
        PadMode::Constant(-0.3),
        PadMode::Reflect,
        PadMode::Replicate,
        PadMode::Circular,
    ] {
        check_mode(&[3, 4], vec![[1, 2], [2, 1]], mode);
    }
}
