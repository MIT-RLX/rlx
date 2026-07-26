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

//! `atan2(a, b)` forward vs `f32::atan2`, and its VJP (differentiable in BOTH
//! operands: `∂/∂a = b/(a²+b²)`, `∂/∂b = -a/(a²+b²)`) vs finite differences.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

// Points in general position — away from the origin (undefined) and the b<0,a→0
// branch cut where atan2 and its FD are ill-conditioned.
const A: &[f32] = &[0.5, -0.7, 1.4, -1.6, 0.3, 2.2];
const B: &[f32] = &[1.2, 0.8, -0.9, 1.5, 2.0, -1.1];

fn sum_sq_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let shape = g.node(y).shape.clone();
    let y2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![y, y], shape);
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![y2],
        Shape::from_dims(&[], DType::F32),
    )
}

fn atan2_node(g: &mut Graph, a: NodeId, b: NodeId, n: usize) -> NodeId {
    g.add_node(
        Op::Binary(BinaryOp::Atan2),
        vec![a, b],
        Shape::new(&[n], DType::F32),
    )
}

#[test]
fn atan2_forward_matches_f32() {
    let n = A.len();
    let mut g = Graph::new("atan2_fwd");
    let a = g.input("a", Shape::new(&[n], DType::F32));
    let b = g.input("b", Shape::new(&[n], DType::F32));
    let y = atan2_node(&mut g, a, b, n);
    g.set_outputs(vec![y]);
    let out = rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", A), ("b", B)])
        .pop()
        .unwrap();
    for i in 0..n {
        let want = A[i].atan2(B[i]);
        assert!(
            (out[i] - want).abs() < 1e-6,
            "atan2[{i}]: {} vs {want}",
            out[i]
        );
    }
}

fn loss_at(av: &[f32], bv: &[f32]) -> f32 {
    let n = av.len();
    let mut g = Graph::new("atan2_loss");
    let a = g.input("a", Shape::new(&[n], DType::F32));
    let b = g.input("b", Shape::new(&[n], DType::F32));
    let y = atan2_node(&mut g, a, b, n);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", av), ("b", bv)])
        .pop()
        .unwrap()[0]
}

#[test]
fn atan2_vjp_matches_fd() {
    let n = A.len();
    let mut g = Graph::new("atan2_grad");
    let a = g.param("a", Shape::new(&[n], DType::F32));
    let b = g.param("b", Shape::new(&[n], DType::F32));
    let y = atan2_node(&mut g, a, b, n);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[a, b]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("a", A);
    c.set_param("b", B);
    let outs = c.run(&[("d_output", &[1.0f32])]);
    let (da, db) = (&outs[1], &outs[2]);

    let eps = 1e-3f32;
    for i in 0..n {
        let (mut ap, mut am) = (A.to_vec(), A.to_vec());
        ap[i] += eps;
        am[i] -= eps;
        let fd_a = (loss_at(&ap, B) - loss_at(&am, B)) / (2.0 * eps);
        assert!(
            (fd_a - da[i]).abs() <= 2e-2 * (1.0 + fd_a.abs()),
            "d/da[{i}]: analytic {} vs FD {fd_a}",
            da[i]
        );

        let (mut bp, mut bm) = (B.to_vec(), B.to_vec());
        bp[i] += eps;
        bm[i] -= eps;
        let fd_b = (loss_at(A, &bp) - loss_at(A, &bm)) / (2.0 * eps);
        assert!(
            (fd_b - db[i]).abs() <= 2e-2 * (1.0 + fd_b.abs()),
            "d/db[{i}]: analytic {} vs FD {fd_b}",
            db[i]
        );
    }
}
