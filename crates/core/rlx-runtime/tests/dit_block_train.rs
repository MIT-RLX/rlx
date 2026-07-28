// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tiny DiT-style block train step: adaLN → linear → gated residual → sum loss.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::AdaNormKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 3;
const D: usize = 4;
const EPS: f32 = 1e-5;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| seed * (0.11 * (i as f32) - 0.07 * ((i % 3) as f32)))
        .collect()
}

/// Build `out = gated(x, linear(ada(x)), gate)` then scalar sum loss.
fn dit_block_loss() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("dit_block");
    let x = g.param("x", Shape::new(&[B, S, D], f));
    let scale = g.input("scale", Shape::new(&[B, 1, D], f));
    let shift = g.input("shift", Shape::new(&[B, 1, D], f));
    let gate = g.input("gate", Shape::new(&[B, 1, D], f));
    let w = g.param("w", Shape::new(&[D, D], f));
    let n = g.ada_layer_norm(x, scale, shift, AdaNormKind::LayerNorm, EPS);
    // [B,S,D] @ [D,D] — reshape to 2-D matmul then back.
    let flat = g.reshape_(n, vec![(B * S) as i64, D as i64]);
    let y2 = g.mm(flat, w);
    let y = g.reshape_(y2, vec![B as i64, S as i64, D as i64]);
    let out = g.gated_residual(x, y, gate);
    let loss = g.sum(out, vec![0, 1, 2], false);
    g.set_outputs(vec![loss]);
    g
}

#[test]
fn dit_block_one_train_step_cpu() {
    let g = dit_block_loss();
    let x_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Param { name } if name == "x"))
        .map(|n| n.id)
        .unwrap();
    let w_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Param { name } if name == "w"))
        .map(|n| n.id)
        .unwrap();
    let bwd = grad_with_loss(&g, &[x_id, w_id]);
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AdaLayerNormBackward { .. }))
    );
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidualBackward))
    );

    let x0 = fill(B * S * D, 1.0);
    let w0 = fill(D * D, 0.05);
    let scale = fill(B * D, 0.2);
    let shift = fill(B * D, -0.1);
    let gate = fill(B * D, 0.3);
    let d_output = [1.0f32];

    let mut c = Session::new(Device::Cpu).compile(bwd);
    c.set_param("x", &x0);
    c.set_param("w", &w0);
    let outs = c.run(&[
        ("scale", &scale),
        ("shift", &shift),
        ("gate", &gate),
        ("d_output", &d_output),
    ]);
    let loss = outs[0][0];
    let dx = &outs[1];
    let dw = &outs[2];
    assert!(loss.is_finite(), "loss={loss}");
    assert!(dx.iter().all(|v| v.is_finite()));
    assert!(dw.iter().all(|v| v.is_finite()));
    // Non-trivial gradient: at least one entry moves.
    assert!(
        dx.iter().any(|v| v.abs() > 1e-8) || dw.iter().any(|v| v.abs() > 1e-8),
        "expected non-zero grads"
    );
}

#[test]
fn hir_ada_layer_norm_lowers() {
    use rlx_ir::HirGraphExt;
    use rlx_ir::hir::{HirModule, HirMut};

    let f = DType::F32;
    let mut hir = HirModule::new("hir_ada");
    let mut gb = HirMut::new(&mut hir);
    let x = gb.input("x", Shape::new(&[B, S, D], f));
    let scale = gb.input("scale", Shape::new(&[B, 1, D], f));
    let shift = gb.input("shift", Shape::new(&[B, 1, D], f));
    let out = gb.ada_layer_norm(x, scale, shift, AdaNormKind::RmsNorm, EPS);
    gb.set_outputs(vec![out]);
    let mir = hir.lower_to_mir().expect("HIR lower");
    let g = mir.into_graph();
    assert!(
        g.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::AdaLayerNorm { .. }))
    );
}
