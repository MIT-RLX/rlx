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

//! `Op::Qr` — thin QR (`A = Q·R`, Q orthonormal, R upper) and both VJPs vs FD.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, QrPart, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

const M: usize = 4;
const N: usize = 3;
const K: usize = 3;

fn a_mat() -> Vec<f32> {
    (0..M * N)
        .map(|i| ((i as f32) * 0.53 + 0.3).sin() * 2.0 + 0.15 * i as f32)
        .collect()
}
fn pshape(part: QrPart) -> Shape {
    match part {
        QrPart::Q => Shape::new(&[M, K], DType::F32),
        QrPart::R => Shape::new(&[K, N], DType::F32),
    }
}
fn run_part(a: &[f32], part: QrPart) -> Vec<f32> {
    let mut g = Graph::new("qr");
    let ai = g.input("a", Shape::new(&[M, N], DType::F32));
    let o = g.qr(ai, part, pshape(part));
    g.set_outputs(vec![o]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", a)])
        .pop()
        .unwrap()
}

#[test]
fn qr_reconstructs() {
    let a = a_mat();
    let q = run_part(&a, QrPart::Q); // [M,K]
    let r = run_part(&a, QrPart::R); // [K,N]
    // Q·R == A.
    for i in 0..M {
        for j in 0..N {
            let acc: f32 = (0..K).map(|k| q[i * K + k] * r[k * N + j]).sum();
            assert!((acc - a[i * N + j]).abs() < 1e-4, "QR recon [{i}][{j}]");
        }
    }
    // Qᵀ·Q == I.
    for p in 0..K {
        for qq in 0..K {
            let s: f32 = (0..M).map(|i| q[i * K + p] * q[i * K + qq]).sum();
            let want = if p == qq { 1.0 } else { 0.0 };
            assert!((s - want).abs() < 1e-4, "QᵀQ[{p}][{qq}]");
        }
    }
    // R upper-triangular.
    for i in 0..K {
        for j in 0..i.min(N) {
            assert!(r[i * N + j].abs() < 1e-5, "R[{i}][{j}] not zero");
        }
    }
}

fn weighted_loss(g: &mut Graph, x: NodeId, len: usize) -> NodeId {
    let s = Shape::new(&[len], DType::F32);
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![len as i64],
        },
        vec![x],
        s.clone(),
    );
    let cw: Vec<f32> = (0..len).map(|k| ((k as f32) * 0.3 + 0.5).sin()).collect();
    let c_bytes: Vec<u8> = cw.iter().flat_map(|v| v.to_le_bytes()).collect();
    let c = g.add_node(Op::Constant { data: c_bytes }, vec![], s.clone());
    let xc = g.add_node(Op::Binary(BinaryOp::Mul), vec![flat, c], s);
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![xc],
        Shape::from_dims(&[], DType::F32),
    )
}

fn loss_graph(g: &mut Graph, a: NodeId, part: QrPart) -> NodeId {
    let o = g.qr(a, part, pshape(part));
    let len = match part {
        QrPart::Q => M * K,
        QrPart::R => K * N,
    };
    weighted_loss(g, o, len)
}

fn loss_at(av: &[f32], part: QrPart) -> f32 {
    let mut g = Graph::new("l");
    let a = g.input("a", Shape::new(&[M, N], DType::F32));
    let loss = loss_graph(&mut g, a, part);
    g.set_outputs(vec![loss]);
    rlx::Session::new(rlx::Device::Cpu)
        .compile(g)
        .run(&[("a", av)])
        .pop()
        .unwrap()[0]
}

#[test]
fn qr_vjp_matches_fd() {
    let a = a_mat();
    for part in [QrPart::Q, QrPart::R] {
        let mut g = Graph::new("g");
        let ai = g.param("a", Shape::new(&[M, N], DType::F32));
        let loss = loss_graph(&mut g, ai, part);
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
            let fd = (loss_at(&ap, part) - loss_at(&am, part)) / (2.0 * eps);
            assert!(
                (fd - da[k]).abs() <= 3e-2 * (1.0 + fd.abs()),
                "{part:?} dA[{k}]: analytic {} vs FD {fd}",
                da[k]
            );
        }
    }
}
