// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `prepare_graph_for_ad` must preserve `graph.outputs` **positionally**.
//!
//! This is the invariant `Wrt::Output` rests on. Prepare renumbers nodes freely
//! (reduce legalization, `LowerDotGeneral`, fused-op unfusing, `If`/`While`
//! inlining) — that's fine, provided each pass remaps `outputs` in place so
//! `outputs[i]` still denotes the same *value*. It does today, but only as an
//! emergent property of every pass being written that way; nothing asserted it.
//!
//! A pass that dropped, reordered, or appended an output would make
//! `Wrt::Output(i)` differentiate the wrong tensor — silently, since the shapes
//! would usually still line up. So check values, not just count.

use rlx_autodiff::prepare_graph_for_ad;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};

/// Compile `graph` and `prepare_graph_for_ad(graph)` and assert every output
/// agrees elementwise.
fn assert_outputs_stable(graph: Graph, params: &[(&str, Vec<f32>)], inputs: &[(&str, Vec<f32>)]) {
    let n_outputs = graph.outputs.len();
    let prepared = prepare_graph_for_ad(graph.clone());
    assert_eq!(
        prepared.outputs.len(),
        n_outputs,
        "prepare changed the output count"
    );

    let run = |g: Graph| -> Vec<Vec<f32>> {
        let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(g);
        for (name, data) in params {
            compiled.set_param(name, data);
        }
        let feed: Vec<(&str, &[f32])> = inputs.iter().map(|(n, d)| (*n, d.as_slice())).collect();
        compiled.run(&feed)
    };

    let before = run(graph);
    let after = run(prepared);
    for i in 0..n_outputs {
        assert_eq!(
            before[i].len(),
            after[i].len(),
            "output {i} changed length across prepare"
        );
        for (j, (b, a)) in before[i].iter().zip(&after[i]).enumerate() {
            assert!(
                (b - a).abs() < 1e-5,
                "output {i}[{j}] moved across prepare: {b} → {a}"
            );
        }
    }
}

/// Multi-axis `Reduce` is legalized into per-axis reduces + reshape — the
/// rewrite that renumbers most aggressively.
#[test]
fn multi_axis_reduce_preserves_output_positions() {
    let f = DType::F32;
    let mut g = Graph::new("multi_axis_reduce");
    let x = g.input("x", Shape::new(&[2, 2, 2], f));
    let s = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1, 2],
        false,
        Shape::from_dims(&[], f),
    );
    let w = g.param("w", Shape::new(&[1], f));
    let h = g.binary(BinaryOp::Mul, s, w, Shape::new(&[1], f));
    let y = g.binary(BinaryOp::Add, h, h, Shape::new(&[1], f));
    // Three outputs with distinct values, so a reorder can't pass unnoticed.
    g.set_outputs(vec![y, h, x]);

    assert_outputs_stable(
        g,
        &[("w", vec![3.0])],
        &[("x", (0..8).map(|i| i as f32).collect())],
    );
}

/// `LowerDotGeneral` rewrites every `MatMul`; `tanh` rides along as an
/// elementwise op that unfusing may re-express.
#[test]
fn matmul_stack_preserves_output_positions() {
    let f = DType::F32;
    let (rows, d_in, d_hid) = (2usize, 3usize, 4usize);
    let mut g = Graph::new("matmul_stack");
    let x = g.input("x", Shape::new(&[rows, d_in], f));
    let w1 = g.param("w1", Shape::new(&[d_in, d_hid], f));
    let a = g.matmul(x, w1, Shape::new(&[rows, d_hid], f));
    let t = g.activation(Activation::Tanh, a, Shape::new(&[rows, d_hid], f));
    let w2 = g.param("w2", Shape::new(&[d_hid, d_in], f));
    let y = g.matmul(t, w2, Shape::new(&[rows, d_in], f));
    // outputs[1..] are the intermediates a lens would tap.
    g.set_outputs(vec![y, a, t]);

    assert_outputs_stable(
        g,
        &[
            (
                "w1",
                (0..d_in * d_hid).map(|i| 0.1 * (i as f32 - 5.0)).collect(),
            ),
            (
                "w2",
                (0..d_hid * d_in).map(|i| 0.1 * (i as f32 - 6.0)).collect(),
            ),
        ],
        &[(
            "x",
            (0..rows * d_in).map(|i| 0.2 * (i as f32) - 0.5).collect(),
        )],
    );
}
