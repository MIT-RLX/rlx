// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "test-support", feature = "serialize"))]

//! `.rlxmodel` over **every** op: each [`sample_ops`] op, in every setting
//! that changes its arity, prints and reads back to the same bytes.
//!
//! `tests/fixtures/opset1.rlxmodel` holds every op as this crate wrote it at
//! opset 1. It must keep reading: a renamed variant or field without
//! `#[serde(alias)]`, or a new field without `#[serde(default)]`, breaks every
//! file written before it, and this test is where that shows. Adding an op is
//! fine; refresh the file with `RLX_BLESS_RLXMODEL=1` to cover the new one.

use rlx_ir::{DType, Graph, Shape, rlxmodel, sample_ops};

fn one_op_graph(name: &str, op: rlx_ir::Op) -> Graph {
    let s = Shape::new(&[4], DType::F32);
    let mut g = Graph::new(name);
    let inputs: Vec<_> = (0..op.num_inputs())
        .map(|i| g.input(format!("x{i}"), s.clone()))
        .collect();
    let y = g.add_node(op, inputs, s);
    g.set_outputs(vec![y]);
    g
}

fn every_op() -> Vec<Graph> {
    let mut out = Vec::new();
    for (kind, op) in sample_ops::all() {
        out.push(one_op_graph(&format!("{kind:?}"), op));
        for (i, v) in sample_ops::variants(kind).into_iter().enumerate() {
            out.push(one_op_graph(&format!("{kind:?}_{i}"), v));
        }
    }
    out
}

#[test]
fn every_op_round_trips_to_the_same_bytes() {
    let graphs = every_op();
    let refs: Vec<&Graph> = graphs.iter().collect();
    let text = rlxmodel::write_graphs(&refs).unwrap();
    let back = rlxmodel::read_graphs(&text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(back.len(), graphs.len());
    for (a, b) in graphs.iter().zip(&back) {
        assert_eq!(
            bincode::serialize(a).unwrap(),
            bincode::serialize(b).unwrap(),
            "{} differs after a round trip:\n{}",
            a.name,
            rlxmodel::to_text(a).unwrap()
        );
    }
    assert_eq!(
        rlxmodel::write_graphs(&back.iter().collect::<Vec<_>>()).unwrap(),
        text
    );
}

#[test]
fn files_written_at_opset_1_still_read() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/opset1.rlxmodel"
    );
    if rlx_ir::env::var_os("RLX_BLESS_RLXMODEL").is_some() {
        let graphs = every_op();
        let text = rlxmodel::write_graphs(&graphs.iter().collect::<Vec<_>>()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    let text = std::fs::read_to_string(path).expect("the opset 1 fixture");
    let graphs = rlxmodel::read_graphs(&text).unwrap_or_else(|e| {
        panic!("{path} no longer reads — a serde alias or default is missing: {e}")
    });
    // Each graph is named after the op it holds.
    for g in &graphs {
        let op = &g.node(g.outputs[0]).op;
        let kind = format!("{:?}", op.kind());
        assert!(
            g.name == kind || g.name.starts_with(&format!("{kind}_")),
            "{} now reads as {kind}",
            g.name
        );
    }
    assert!(graphs.len() >= sample_ops::ALL_KINDS.len());
}
