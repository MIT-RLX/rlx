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

//! `Op::{Clamp, Tile, Trilu}` VJP vs finite differences (loss = sum(op(x)^2)).

use rlx_autodiff::grad_with_loss;
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
            (fd - d[i]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "grad[{i}]: analytic {} vs FD {fd}",
            d[i]
        );
    }
}

#[test]
fn clamp_vjp() {
    // Values away from the kinks at min=-1 / max=2.5.
    let x = vec![-0.5f32, 0.3, 1.0, 2.0, -3.0, 4.0];
    fd_check(&[6], |g, x| g.clamp_(x, -1.0, 2.5), &x);
}

#[test]
fn tile_vjp() {
    let x = vec![0.5f32, -0.7, 1.3, -0.2, 0.9, -1.1];
    fd_check(&[2, 3], |g, x| g.tile_(x, vec![2, 2]), &x);
}

#[test]
fn trilu_vjp() {
    let x: Vec<f32> = (0..9).map(|i| (i as f32) * 0.3 - 1.0).collect();
    fd_check(&[3, 3], |g, x| g.trilu_(x, true, 0), &x);
    fd_check(&[3, 3], |g, x| g.trilu_(x, false, -1), &x);
}
