// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! 3-D `Op::Pool` (Max / Mean) VJP vs finite differences (CPU).

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const XS: [usize; 5] = [1, 1, 2, 2, 2];

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

fn pool(g: &mut Graph, x: NodeId, kind: ReduceOp) -> NodeId {
    g.add_node(
        Op::Pool {
            kind,
            kernel_size: vec![2, 2, 2],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
        },
        vec![x],
        Shape::new(&[1, 1, 1, 1, 1], DType::F32),
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

fn check_pool_vjp(kind: ReduceOp, label: &str) {
    let mut g = Graph::new(label);
    let x = g.param("x", Shape::new(&XS, DType::F32));
    let y = pool(&mut g, x, kind);
    let loss = sum_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[x]);
    // Distinct values so Max has a unique argmax (avoids tie ambiguity).
    let x_init = vec![0.1, 0.3, -0.2, 0.5, 0.4, -0.1, 0.2, 0.8];

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("x", &x_init);
    let outs = compiled.run(&[("d_output", &[1.0f32])]);
    let d_x = &outs[1];

    let loss_at = |xv: &[f32]| -> f32 {
        let mut fg = Graph::new("fwd");
        let xi = fg.input("x", Shape::new(&XS, DType::F32));
        let y = pool(&mut fg, xi, kind);
        let loss = sum_loss(&mut fg, y);
        fg.set_outputs(vec![loss]);
        rlx::Session::new(rlx::Device::Cpu)
            .compile(fg)
            .run(&[("x", xv)])
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
        fd_x[i] = (loss_at(&p) - loss_at(&m)) / (2.0 * eps);
    }

    assert_close(d_x, &fd_x, 2e-2, label);
}

#[test]
fn maxpool3d_vjp_matches_fd() {
    check_pool_vjp(ReduceOp::Max, "maxpool3d d_x");
}

#[test]
fn avgpool3d_vjp_matches_fd() {
    check_pool_vjp(ReduceOp::Mean, "avgpool3d d_x");
}
