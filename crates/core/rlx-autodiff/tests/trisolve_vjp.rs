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

//! VJP of `Op::TriangularSolve` vs finite differences, in both operands. The A
//! gradient must be zero on the unused (upper) triangle — the `Trilu` mask.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const N: usize = 4;
const K: usize = 2;

fn lower_a() -> Vec<f32> {
    let mut a = vec![0f32; N * N];
    for i in 0..N {
        for j in 0..=i {
            a[i * N + j] = if i == j {
                (i as f32) + 2.5
            } else {
                0.4 * ((i + j) as f32) - 0.3
            };
        }
    }
    a
}
fn b_rhs() -> Vec<f32> {
    (0..N * K)
        .map(|k| ((k as f32) * 0.7 + 0.2).sin() * 0.8)
        .collect()
}

fn sum_sq_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let shape = g.node(y).shape.clone();
    let n = shape.num_elements().unwrap();
    let y2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![y, y], shape);
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![n as i64],
        },
        vec![y2],
        Shape::new(&[n], DType::F32),
    );
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![flat],
        Shape::from_dims(&[], DType::F32),
    )
}

fn loss_at(a: &[f32], b: &[f32]) -> f32 {
    let mut g = Graph::new("l");
    let ai = g.input("a", Shape::new(&[N, N], DType::F32));
    let bi = g.input("b", Shape::new(&[N, K], DType::F32));
    let x = g.triangular_solve(ai, bi, true, false, Shape::new(&[N, K], DType::F32));
    let loss = sum_sq_loss(&mut g, x);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", a), ("b", b)])
        .pop()
        .unwrap()[0]
}

#[test]
fn trisolve_vjp_matches_fd() {
    let a = lower_a();
    let b = b_rhs();

    let mut g = Graph::new("g");
    let ai = g.param("a", Shape::new(&[N, N], DType::F32));
    let bi = g.param("b", Shape::new(&[N, K], DType::F32));
    let x = g.triangular_solve(ai, bi, true, false, Shape::new(&[N, K], DType::F32));
    let loss = sum_sq_loss(&mut g, x);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[ai, bi]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("a", &a);
    c.set_param("b", &b);
    let outs = c.run(&[("d_output", &[1.0f32])]);
    let (da, db) = (&outs[1], &outs[2]);

    let eps = 1e-3f32;
    for i in 0..N {
        for j in 0..N {
            let (mut ap, mut am) = (a.clone(), a.clone());
            ap[i * N + j] += eps;
            am[i * N + j] -= eps;
            let fd = (loss_at(&ap, &b) - loss_at(&am, &b)) / (2.0 * eps);
            assert!(
                (fd - da[i * N + j]).abs() <= 2e-2 * (1.0 + fd.abs()),
                "dA[{i}][{j}]: analytic {} vs FD {fd}",
                da[i * N + j]
            );
        }
    }
    for idx in 0..b.len() {
        let (mut bp, mut bm) = (b.clone(), b.clone());
        bp[idx] += eps;
        bm[idx] -= eps;
        let fd = (loss_at(&a, &bp) - loss_at(&a, &bm)) / (2.0 * eps);
        assert!(
            (fd - db[idx]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "dB[{idx}]: analytic {} vs FD {fd}",
            db[idx]
        );
    }
}
