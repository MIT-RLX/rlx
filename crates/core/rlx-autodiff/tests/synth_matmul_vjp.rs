// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::SynthMatMul` (codebook weight-synthesis matmul) VJP vs finite
//! differences. Differentiable in `x` and `codebook`; `indices` (u8) is a
//! fixed, non-differentiable param. `d_codebook` accumulates via scatter-add
//! over reused indices — the FD check exercises that accumulation.

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape, SynthKind};

const M: usize = 3;
const K: usize = 8;
const N: usize = 4;
const D: usize = 2;
const NE: usize = 4;

fn kb() -> usize {
    K / D
}
fn idx_data() -> Vec<u8> {
    // Spread across all NE entries, with reuse (tests scatter accumulation).
    (0..N * kb()).map(|i| ((i * 3 + 1) % NE) as u8).collect()
}
fn x_data() -> Vec<f32> {
    (0..M * K).map(|i| (i as f32 * 0.5).sin() * 0.9).collect()
}
fn cb_data() -> Vec<f32> {
    (0..NE * D).map(|i| (i as f32 * 0.4).cos() * 0.7).collect()
}

fn kind() -> SynthKind {
    SynthKind::Codebook {
        entry_dim: D as u32,
        num_entries: NE as u32,
    }
}

fn synth(g: &mut Graph, x: NodeId, idx: NodeId, cb: NodeId) -> NodeId {
    g.synth_matmul(x, idx, cb, kind(), Shape::new(&[M, N], DType::F32))
}

fn sum_sq_loss(g: &mut Graph, y: NodeId) -> NodeId {
    let sh = g.node(y).shape.clone();
    let n: usize = (0..sh.rank()).map(|i| sh.dim(i).unwrap_static()).product();
    let y2 = g.add_node(Op::Binary(BinaryOp::Mul), vec![y, y], sh);
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

fn loss_at(xv: &[f32], cbv: &[f32], idx: &[u8]) -> f32 {
    let mut g = Graph::new("synth_loss");
    let x = g.input("x", Shape::new(&[M, K], DType::F32));
    let cb = g.input("cb", Shape::new(&[NE, D], DType::F32));
    let id = g.param("idx", Shape::new(&[N, kb()], DType::U8));
    let y = synth(&mut g, x, id, cb);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(g);
    c.set_param_typed("idx", idx, DType::U8);
    c.run(&[("x", xv), ("cb", cbv)]).pop().unwrap()[0]
}

#[test]
fn synth_vjp_matches_fd() {
    let (x, cb, idx) = (x_data(), cb_data(), idx_data());
    let mut g = Graph::new("synth_grad");
    let xp = g.param("x", Shape::new(&[M, K], DType::F32));
    let cbp = g.param("cb", Shape::new(&[NE, D], DType::F32));
    let idp = g.param("idx", Shape::new(&[N, kb()], DType::U8));
    let y = synth(&mut g, xp, idp, cbp);
    let loss = sum_sq_loss(&mut g, y);
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[xp, cbp]);
    let mut c = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    c.set_param("x", &x);
    c.set_param("cb", &cb);
    c.set_param_typed("idx", &idx, DType::U8);
    let outs = c.run(&[("d_output", &[1.0f32])]);
    let (dx, dcb) = (&outs[1], &outs[2]);
    assert_eq!(dx.len(), x.len());
    assert_eq!(dcb.len(), cb.len());

    let eps = 1e-3f32;
    // ∂loss/∂x
    for i in 0..x.len() {
        let (mut xp_, mut xm_) = (x.clone(), x.clone());
        xp_[i] += eps;
        xm_[i] -= eps;
        let fd = (loss_at(&xp_, &cb, &idx) - loss_at(&xm_, &cb, &idx)) / (2.0 * eps);
        assert!(
            (fd - dx[i]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "d/dx[{i}]: analytic {} vs FD {fd}",
            dx[i]
        );
    }
    // ∂loss/∂codebook (accumulated over reused indices)
    for i in 0..cb.len() {
        let (mut cp_, mut cm_) = (cb.clone(), cb.clone());
        cp_[i] += eps;
        cm_[i] -= eps;
        let fd = (loss_at(&x, &cp_, &idx) - loss_at(&x, &cm_, &idx)) / (2.0 * eps);
        assert!(
            (fd - dcb[i]).abs() <= 2e-2 * (1.0 + fd.abs()),
            "d/dcodebook[{i}]: analytic {} vs FD {fd}",
            dcb[i]
        );
    }
}
