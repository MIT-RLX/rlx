// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Regression: a Param built AFTER a multi-axis reduce must still receive its
//! correct gradient.
//!
//! `prepare_graph_for_ad` legalizes `Reduce{_, all-axes}` into several
//! single-axis reduces (+ reshape), renumbering every node created after it.
//! `wrt` NodeIds are captured against the pre-prepare graph, so without the
//! by-name remap in `grad_with_loss_opts`, a Param built after the reduce
//! resolves to a STALE id → the wrong gradient (silent miscompute) or a panic.
//! Here `loss = sum(x)·w + b`, so d(loss)/dw = sum(x) and d(loss)/db = 1 — the
//! stale-id path yields a different value, so the numeric asserts fail loudly.

use rlx_ir::op::{BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};

#[test]
fn grad_wrt_params_after_multi_axis_reduce_is_correct() {
    let f = DType::F32;
    let mut g = Graph::new("mreduce");
    let x = g.input("x", Shape::new(&[2, 2, 2], f)); // 8 elements
    let r = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1, 2],
        false,
        Shape::from_dims(&[], f),
    );
    // Params built AFTER the multi-axis reduce (their ids shift under prepare).
    let w = g.param("w", Shape::new(&[1], f));
    let b = g.param("b", Shape::new(&[1], f));
    let rw = g.binary(BinaryOp::Mul, r, w, Shape::new(&[1], f)); // sum(x)·w
    let loss = g.binary(BinaryOp::Add, rw, b, Shape::new(&[1], f));
    g.set_outputs(vec![loss]);

    let bwd = rlx_autodiff::grad_with_loss(&g, &[w, b]);

    let x_ones = [1.0f32; 8]; // sum(x) = 8
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("w", &[1.0]);
    compiled.set_param("b", &[0.0]);
    let outs = compiled.run(&[("x", &x_ones[..]), ("d_output", &[1.0f32])]);

    assert!(outs.len() >= 3, "expect [loss, dw, db]");
    // d(loss)/dw = sum(x) = 8 ; the stale-id path would return something else.
    assert!(
        (outs[1][0] - 8.0).abs() < 1e-4,
        "dw: got {} want 8.0",
        outs[1][0]
    );
    // d(loss)/db = 1.
    assert!(
        (outs[2][0] - 1.0).abs() < 1e-4,
        "db: got {} want 1.0",
        outs[2][0]
    );
}
