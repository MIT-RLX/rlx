// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Third-order AD through smooth activations vs closed-form references.

#![cfg(feature = "cpu")]

use rlx_autodiff::nth_order_grad;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn f64s(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f64b(b: &[u8]) -> f64 {
    f64::from_le_bytes(b[..8].try_into().unwrap())
}

fn eval_graph(g: &Graph, x: f64) -> f64 {
    let mut c = Session::new(Device::Cpu).compile(g.clone());
    let outs = c.run_typed(&[("x", &f64s(&[x]), DType::F64)]);
    f64b(&outs[0].0)
}

fn activation_graph(kind: Activation) -> Graph {
    let mut g = Graph::new("act");
    let x = g.input("x", Shape::scalar(DType::F64));
    let y = g.activation(kind, x, Shape::scalar(DType::F64));
    g.set_outputs(vec![y]);
    g
}

#[test]
fn tanh_third_derivative_closed_form() {
    let forward = activation_graph(Activation::Tanh);
    let g3 = nth_order_grad(&forward, "x", 3);
    for &x_val in &[0.37, 0.5, -0.8] {
        let got = eval_graph(&g3, x_val);
        let tx = x_val.tanh();
        let sech2 = (1.0_f64 / x_val.cosh()).powi(2);
        let want = -2.0 * sech2 * (1.0 - 3.0 * tx * tx);
        assert!(
            (got - want).abs() < 1e-9,
            "tanh''' at {x_val}: got {got} want {want}"
        );
    }
}

#[test]
fn sin_third_derivative_runs() {
    let forward = activation_graph(Activation::Sin);
    let g3 = nth_order_grad(&forward, "x", 3);
    let got = eval_graph(&g3, 1.1);
    assert!(got.is_finite(), "sin third deriv: {got}");
}

#[test]
fn sigmoid_third_derivative_runs() {
    let forward = activation_graph(Activation::Sigmoid);
    let g3 = nth_order_grad(&forward, "x", 3);
    let got = eval_graph(&g3, 0.42);
    assert!(got.is_finite(), "sigmoid third deriv: {got}");
}

#[test]
fn silu_and_gelu_third_derivative_runs() {
    for kind in [Activation::Silu, Activation::Gelu] {
        let forward = activation_graph(kind);
        let g3 = nth_order_grad(&forward, "x", 3);
        let got = eval_graph(&g3, 0.25);
        assert!(
            got.is_finite(),
            "{kind:?} third deriv must be finite, got {got}"
        );
    }
}

#[test]
fn relu_third_derivative_is_zero() {
    let forward = activation_graph(Activation::Relu);
    let g1 = nth_order_grad(&forward, "x", 1);
    let g2 = nth_order_grad(&forward, "x", 2);
    let g3 = nth_order_grad(&forward, "x", 3);
    for &x_val in &[1.0, 0.5, -2.0] {
        let got1 = eval_graph(&g1, x_val);
        let want1 = if x_val > 0.0 { 1.0 } else { 0.0 };
        assert!(
            (got1 - want1).abs() < 1e-12,
            "relu' at {x_val}: got {got1} want {want1}"
        );
        assert!(eval_graph(&g2, x_val).abs() < 1e-12, "relu'' at {x_val}");
        assert!(eval_graph(&g3, x_val).abs() < 1e-12, "relu''' at {x_val}");
    }
}

#[test]
fn abs_third_derivative_is_zero() {
    let forward = activation_graph(Activation::Abs);
    let g3 = nth_order_grad(&forward, "x", 3);
    assert!(eval_graph(&g3, 1.5).abs() < 1e-12);
    assert!(eval_graph(&g3, -1.5).abs() < 1e-12);
}
