// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Every binding sees the same error *kinds* for caller mistakes: a wrong
//! argument is `Type`, an unavailable device `Runtime` — not `Panic` ("a Rust
//! assertion inside rlx"), which they all used to be.

#![cfg(feature = "runtime")]

use rlx_bind::ErrorKind;
use rlx_bind::runtime::session::{CompileRequest, Session};
use rlx_ir::{DType, Graph, GraphExt, Op, Shape};

fn graph() -> Graph {
    let mut g = Graph::new("mlp");
    let x = g.input("tokens", Shape::new(&[2, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 3], DType::F32));
    let h = g.mm(x, w);
    let y = g.relu(h);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn mistakes_have_the_right_kind_and_message() {
    let s = Session::new("cpu", "f32", None).expect("cpu session");
    let mut c = s
        .compile(graph(), &CompileRequest::default())
        .expect("compile");

    let e = c.run(&[("token", &[0.0; 8])]).err().unwrap();
    assert_eq!(e.kind, ErrorKind::Type);
    assert!(e.message.contains("did you mean `tokens`"), "{}", e.message);

    let e = c.run(&[("tokens", &[0.0; 3])]).err().unwrap();
    assert_eq!(e.kind, ErrorKind::Type);
    assert!(e.message.contains("holds 8"), "{}", e.message);

    let e = c.set_param("w", &[0.0; 5]).err().unwrap();
    assert_eq!(e.kind, ErrorKind::Type);

    c.set_param("w", &[1.0; 12]).unwrap();
    let y = c.run(&[("tokens", &[1.0; 8])]).unwrap();
    assert_eq!(y[0][0], 4.0);
}

#[test]
fn an_invalid_graph_is_a_type_error() {
    let mut g = Graph::new("bad");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let w = g.param("w", Shape::new(&[4, 5], DType::F32));
    let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[2, 5], DType::F32));
    g.set_outputs(vec![y]);
    let s = Session::new("cpu", "f32", None).unwrap();
    let e = s.compile(g, &CompileRequest::default()).err().unwrap();
    assert_eq!(e.kind, ErrorKind::Type, "{}", e.message);
    assert!(e.message.contains("invalid"), "{}", e.message);
}

#[test]
fn a_device_not_built_in_is_a_runtime_error() {
    if rlx_runtime::feature_compiled(rlx_runtime::Device::Tpu) {
        return;
    }
    let e = Session::new("tpu", "f32", None).err().unwrap();
    assert_eq!(e.kind, ErrorKind::Runtime, "{}", e.message);
    assert!(e.message.contains("not in this build"), "{}", e.message);
}

#[test]
fn an_unknown_output_name_is_a_range_error() {
    let mut g = graph();
    g.name_output(0, "probs");
    let s = Session::new("cpu", "f32", None).unwrap();
    let c = s.compile(g, &CompileRequest::default()).unwrap();
    assert_eq!(c.output_index("probs").unwrap(), 0);
    let e = c.output_index("prob").err().unwrap();
    assert_eq!(e.kind, ErrorKind::Range);
    assert!(e.message.contains("did you mean `probs`"), "{}", e.message);
}
