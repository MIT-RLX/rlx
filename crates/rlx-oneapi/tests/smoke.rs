// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Smoke tests for the Intel oneAPI (Level Zero) backend.
//!
//! Unlike the GPU-only backends, `OneApiExecutable` always has a correct
//! execution path: when no Level Zero device is present (this macOS dev box /
//! CI) the whole legalized graph runs through the `rlx-cpu` reference, so the
//! compute assertions below execute *everywhere* and validate the legalize +
//! interpreter end-to-end. On Intel hardware the same graphs additionally
//! exercise the native SPIR-V dispatch path.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_oneapi::backend::OneApiExecutable;

fn s(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

fn run1(g: Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    OneApiExecutable::compile(g)
        .run(inputs)
        .into_iter()
        .next()
        .unwrap()
}

#[test]
fn device_discovery_is_graceful() {
    // Must never panic regardless of host. Driverless → false + no name;
    // with a Level Zero GPU → true + a device name.
    let avail = rlx_oneapi::is_available();
    if avail {
        assert!(rlx_oneapi::device_name().is_some());
        eprintln!(
            "[rlx-oneapi] Level Zero device: {:?} (native kernels: {})",
            rlx_oneapi::device_name(),
            rlx_oneapi::has_native_kernels()
        );
    } else {
        assert!(rlx_oneapi::device_name().is_none());
        eprintln!("[rlx-oneapi] no Level Zero device — compute runs via CPU reference");
    }
}

#[test]
fn add_then_relu() {
    let mut g = Graph::new("add_relu");
    let a = g.input("a", s(&[4]));
    let b = g.input("b", s(&[4]));
    let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], s(&[4]));
    let out = g.add_node(Op::Activation(Activation::Relu), vec![sum], s(&[4]));
    g.set_outputs(vec![out]);
    let r = run1(
        g,
        &[
            ("a", &[1.0, -5.0, 3.0, -2.0]),
            ("b", &[0.5, 1.0, -1.0, -1.0]),
        ],
    );
    assert_eq!(r, vec![1.5, 0.0, 2.0, 0.0]);
}

#[test]
fn matmul_2x3_3x2() {
    let mut g = Graph::new("matmul");
    let a = g.input("a", s(&[2, 3]));
    let b = g.input("b", s(&[3, 2]));
    let out = g.add_node(Op::MatMul, vec![a, b], s(&[2, 2]));
    g.set_outputs(vec![out]);
    // A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]] -> [[58,64],[139,154]]
    let r = run1(
        g,
        &[
            ("a", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("b", &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ],
    );
    assert_eq!(r, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn softmax_uniform() {
    let mut g = Graph::new("sm");
    let x = g.input("x", s(&[3]));
    let o = g.add_node(Op::Softmax { axis: -1 }, vec![x], s(&[3]));
    g.set_outputs(vec![o]);
    let r = run1(g, &[("x", &[0.0, 0.0, 0.0])]);
    for v in &r {
        assert!((v - 1.0 / 3.0).abs() < 1e-6, "softmax uniform: {v}");
    }
}

#[test]
fn param_upload_and_mul() {
    // y = x * w  (w supplied as a param, not a graph input)
    let mut g = Graph::new("pmul");
    let x = g.input("x", s(&[3]));
    let w = g.add_node(Op::Param { name: "w".into() }, vec![], s(&[3]));
    let o = g.add_node(Op::Binary(BinaryOp::Mul), vec![x, w], s(&[3]));
    g.set_outputs(vec![o]);
    let mut exe = OneApiExecutable::compile(g);
    exe.set_param("w", &[2.0, 3.0, 4.0]);
    let r = exe.run(&[("x", &[1.0, 1.0, 1.0])]);
    assert_eq!(r[0], vec![2.0, 3.0, 4.0]);
}
