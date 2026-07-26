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

//! VJP of `Op::Cholesky` vs finite differences. `potrf` reads only A's lower
//! triangle, so the A-gradient is zero on the strict upper triangle.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const N: usize = 4;

fn spd() -> Vec<f32> {
    let m: Vec<f32> = (0..N * N)
        .map(|i| ((i as f32) * 0.29 + 0.1).sin() * 0.5)
        .collect();
    let mut a = vec![0f32; N * N];
    for i in 0..N {
        for j in 0..N {
            let mut s = 0.0f32;
            for k in 0..N {
                s += m[i * N + k] * m[j * N + k];
            }
            a[i * N + j] = s + if i == j { N as f32 } else { 0.0 };
        }
    }
    a
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

fn loss_at(a: &[f32]) -> f32 {
    let mut g = Graph::new("l");
    let ai = g.input("a", Shape::new(&[N, N], DType::F32));
    let l = g.cholesky(ai, Shape::new(&[N, N], DType::F32));
    let loss = sum_sq_loss(&mut g, l);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", a)])
        .pop()
        .unwrap()[0]
}

#[test]
fn cholesky_vjp_matches_fd() {
    let a = spd();
    let mut g = Graph::new("g");
    let ai = g.param("a", Shape::new(&[N, N], DType::F32));
    let l = g.cholesky(ai, Shape::new(&[N, N], DType::F32));
    let loss = sum_sq_loss(&mut g, l);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[ai]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("a", &a);
    let da = c.run(&[("d_output", &[1.0f32])])[1].clone();

    let eps = 1e-3f32;
    for i in 0..N {
        for j in 0..N {
            let (mut ap, mut am) = (a.clone(), a.clone());
            ap[i * N + j] += eps;
            am[i * N + j] -= eps;
            let fd = (loss_at(&ap) - loss_at(&am)) / (2.0 * eps);
            assert!(
                (fd - da[i * N + j]).abs() <= 3e-2 * (1.0 + fd.abs()),
                "dA[{i}][{j}]: analytic {} vs FD {fd}",
                da[i * N + j]
            );
        }
    }
}
