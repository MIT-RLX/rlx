// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    // Backend is always selectable (CPU-ref without L0). Must never panic.
    assert!(rlx_oneapi::is_available());
    match rlx_oneapi::device_name() {
        Some(name) => eprintln!(
            "[rlx-oneapi] Level Zero device: {name:?} (native kernels: {})",
            rlx_oneapi::has_native_kernels()
        ),
        None => {
            assert!(!rlx_oneapi::has_level_zero_device());
            eprintln!("[rlx-oneapi] no Level Zero device — compute runs via CPU reference");
        }
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

#[test]
fn dit_packed_gated_residual_backward_runs() {
    let (b, seq, d) = (2usize, 3usize, 4usize);
    let mut g = Graph::new("gate_bwd");
    let x = g.input("x", s(&[b, seq, d]));
    let y = g.input("y", s(&[b, seq, d]));
    let gate = g.input("gate", s(&[b, 1, d]));
    let dy = g.input("dy", s(&[b, seq, d]));
    let o = g.gated_residual_backward(x, y, gate, dy);
    g.set_outputs(vec![o]);
    let nx = b * seq * d;
    let ng = b * d;
    let r = run1(
        g,
        &[
            ("x", &vec![0.1f32; nx]),
            ("y", &vec![0.2f32; nx]),
            ("gate", &vec![0.5f32; ng]),
            ("dy", &vec![1.0f32; nx]),
        ],
    );
    assert_eq!(r.len(), nx + nx + ng);
    assert!(r.iter().all(|v| v.is_finite()));
    // dx = dy = 1; dy_out = gate*dy = 0.5; dgate = sum_s y*dy = 3*0.2
    assert!((r[0] - 1.0).abs() < 1e-5);
    assert!((r[nx] - 0.5).abs() < 1e-5);
    assert!((r[2 * nx] - 0.6).abs() < 1e-5);
}
