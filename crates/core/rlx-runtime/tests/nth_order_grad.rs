// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU execution tests for higher-order reverse-mode AD.

#![cfg(feature = "cpu")]

use rlx_autodiff::{directional_nth_grad, nth_order_grad};
use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn f64s(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f64b(b: &[u8]) -> f64 {
    f64::from_le_bytes(b[..8].try_into().unwrap())
}

#[test]
fn nth_order_x_cubed_derivatives() {
    let mut g = Graph::new("x3");
    let x = g.input("x", Shape::scalar(DType::F64));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F64));
    let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F64));
    g.set_outputs(vec![x3]);

    let x_val: f64 = 2.5;
    for (order, want) in [(1, 3.0 * x_val.powi(2)), (2, 6.0 * x_val), (3, 6.0)] {
        let hg = nth_order_grad(&g, "x", order);
        let mut c = Session::new(Device::Cpu).compile(hg);
        let outs = c.run_typed(&[("x", &f64s(&[x_val]), DType::F64)]);
        let got = f64b(&outs[0].0);
        assert!(
            (got - want).abs() < 1e-10,
            "order {order}: got {got}, want {want}"
        );
    }
}

#[test]
fn nth_order_x_cubed_f32() {
    let mut g = Graph::new("x3_f32");
    let x = g.input("x", Shape::scalar(DType::F32));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F32));
    let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F32));
    g.set_outputs(vec![x3]);

    let hg = nth_order_grad(&g, "x", 3);
    let mut c = Session::new(Device::Cpu).compile(hg);
    let x_val = 1.5f32;
    let bytes: Vec<u8> = x_val.to_le_bytes().into_iter().collect();
    let outs = c.run_typed(&[("x", &bytes, DType::F32)]);
    let got = f32::from_le_bytes(outs[0].0[..4].try_into().unwrap());
    assert!((got - 6.0).abs() < 1e-5, "f32 3rd deriv: {got}");
}

#[test]
fn nth_order_tanh_third_derivative() {
    let mut g = Graph::new("tanh");
    let x = g.input("x", Shape::scalar(DType::F64));
    let tx = g.activation(Activation::Tanh, x, Shape::scalar(DType::F64));
    g.set_outputs(vec![tx]);

    let x_val: f64 = 0.5;
    let txv = x_val.tanh();
    let sech2 = (1.0_f64 / x_val.cosh()).powi(2);
    let want = -2.0 * sech2 * (1.0 - 3.0 * txv * txv);

    let hg = nth_order_grad(&g, "x", 3);
    let mut c = Session::new(Device::Cpu).compile(hg);
    let outs = c.run_typed(&[("x", &f64s(&[x_val]), DType::F64)]);
    let got = f64b(&outs[0].0);
    assert!((got - want).abs() < 1e-9, "tanh''' got {got}, want {want}");
}

#[test]
fn directional_second_order_hessian_vector() {
    // f(x) = sum(x²), x ∈ R^n. H·v = 2v.
    let n = 4;
    let mut g = Graph::new("sum_sq");
    let x = g.input("x", Shape::new(&[n], DType::F64));
    let xx = g.binary(BinaryOp::Mul, x, x, Shape::new(&[n], DType::F64));
    let f = g.reduce(xx, ReduceOp::Sum, vec![0], false, Shape::scalar(DType::F64));
    g.set_outputs(vec![f]);

    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![0.5, -0.25, 1.0, -1.5];
    let hg = directional_nth_grad(&g, "x", &["v", "v"]);
    let mut c = Session::new(Device::Cpu).compile(hg);
    let v_bytes = f64s(&v);
    let outs = c.run_typed(&[
        ("x", &f64s(&x_data), DType::F64),
        ("dir_0", &v_bytes, DType::F64),
        ("dir_1", &v_bytes, DType::F64),
    ]);
    let hv = f64b(&outs[0].0);
    // <v, Hv> for f=sum(x²): H=2I, so <v,2v> = 2||v||²
    let want = 2.0 * v.iter().map(|x| x * x).sum::<f64>();
    assert!((hv - want).abs() < 1e-10, "directional 2nd: {hv} vs {want}");
}

#[test]
fn fourth_order_polynomial() {
    // f(x) = x^4 → f'''' = 24
    let mut g = Graph::new("x4");
    let x = g.input("x", Shape::scalar(DType::F64));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F64));
    let x4 = g.binary(BinaryOp::Mul, x2, x2, Shape::scalar(DType::F64));
    g.set_outputs(vec![x4]);

    let hg = nth_order_grad(&g, "x", 4);
    let mut c = Session::new(Device::Cpu).compile(hg);
    let outs = c.run_typed(&[("x", &f64s(&[1.0]), DType::F64)]);
    let got = f64b(&outs[0].0);
    assert!((got - 24.0).abs() < 1e-9, "4th deriv: {got}");
}
