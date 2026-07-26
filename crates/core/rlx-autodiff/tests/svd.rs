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

//! `Op::Svd` — thin SVD reconstruction (`A = U·diag(S)·Vᵀ`) and the singular-
//! value VJP (`Ā = U·diag(ū)·Vᵀ`) vs finite differences.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp, SvdPart};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const M: usize = 4;
const N: usize = 3;
const K: usize = 3; // min(M, N)

fn a_mat() -> Vec<f32> {
    (0..M * N)
        .map(|i| ((i as f32) * 0.41 + 0.2).sin() * 2.0 + 0.1 * i as f32)
        .collect()
}

fn part_shape(part: SvdPart) -> Shape {
    match part {
        SvdPart::U => Shape::new(&[M, K], DType::F32),
        SvdPart::S => Shape::new(&[K], DType::F32),
        SvdPart::Vt => Shape::new(&[K, N], DType::F32),
    }
}

fn run_part(a: &[f32], part: SvdPart) -> Vec<f32> {
    let mut g = Graph::new("svd");
    let ai = g.input("a", Shape::new(&[M, N], DType::F32));
    let o = g.svd(ai, part, part_shape(part));
    g.set_outputs(vec![o]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", a)])
        .pop()
        .unwrap()
}

#[test]
fn svd_reconstructs() {
    let a = a_mat();
    let u = run_part(&a, SvdPart::U);
    let s = run_part(&a, SvdPart::S);
    let vt = run_part(&a, SvdPart::Vt);
    for i in 0..M {
        for j in 0..N {
            let mut acc = 0.0f32;
            for r in 0..K {
                acc += u[i * K + r] * s[r] * vt[r * N + j];
            }
            assert!((acc - a[i * N + j]).abs() < 1e-4, "recon[{i}][{j}]");
        }
    }
    // Singular values non-negative and descending.
    for r in 0..K {
        assert!(s[r] >= -1e-6, "sigma {r} negative");
    }
    for r in 1..K {
        assert!(s[r] <= s[r - 1] + 1e-5, "sigma not descending");
    }
}

// loss = Σ (r+1)·σ_r — distinct weights, so the gradient tests the full VJP.
fn svd_loss_graph(g: &mut Graph, a: NodeId) -> NodeId {
    let s = g.svd(a, SvdPart::S, Shape::new(&[K], DType::F32));
    let cw: Vec<f32> = (0..K).map(|r| (r as f32) + 1.0).collect();
    let c_bytes: Vec<u8> = cw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c = g.add_node(
        Op::Constant { data: c_bytes },
        vec![],
        Shape::new(&[K], DType::F32),
    );
    let sc = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![s, c],
        Shape::new(&[K], DType::F32),
    );
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![sc],
        Shape::from_dims(&[], DType::F32),
    )
}

fn loss_at(av: &[f32]) -> f32 {
    let mut g = Graph::new("l");
    let a = g.input("a", Shape::new(&[M, N], DType::F32));
    let loss = svd_loss_graph(&mut g, a);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", av)])
        .pop()
        .unwrap()[0]
}

#[test]
fn svd_singular_value_vjp_matches_fd() {
    let a = a_mat();
    let mut g = Graph::new("g");
    let ai = g.param("a", Shape::new(&[M, N], DType::F32));
    let loss = svd_loss_graph(&mut g, ai);
    g.set_outputs(vec![loss]);
    let bwd = grad_with_loss(&g, &[ai]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("a", &a);
    let da = c.run(&[("d_output", &[1.0f32])])[1].clone();

    let eps = 1e-3f32;
    for k in 0..M * N {
        let (mut ap, mut am) = (a.clone(), a.clone());
        ap[k] += eps;
        am[k] -= eps;
        let fd = (loss_at(&ap) - loss_at(&am)) / (2.0 * eps);
        assert!(
            (fd - da[k]).abs() <= 3e-2 * (1.0 + fd.abs()),
            "dA[{k}]: analytic {} vs FD {fd}",
            da[k]
        );
    }
}
