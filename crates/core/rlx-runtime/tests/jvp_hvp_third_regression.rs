// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `jvp(hvp(f))` does not yield the third derivative — use `nth_order_grad`.

#![cfg(feature = "cpu")]

use rlx_autodiff::autodiff_fwd::hvp;
use rlx_autodiff::{jvp, nth_order_grad};
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn f64s(xs: &[f64]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f64b(b: &[u8]) -> f64 {
    f64::from_le_bytes(b[..8].try_into().unwrap())
}

fn build_x3() -> (Graph, NodeId) {
    let mut g = Graph::new("x3");
    let x = g.input("x", Shape::scalar(DType::F64));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F64));
    let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F64));
    g.set_outputs(vec![x3]);
    (g, x)
}

#[test]
fn jvp_of_hvp_is_not_third_derivative() {
    let (g, x) = build_x3();
    let x_val = 2.5;
    let want_third = 6.0;

    let hg = hvp(&g, &[x]);
    let x_in = hg
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "x"))
        .map(|n| n.id)
        .expect("x input");
    let jg = jvp(&hg, &[x_in]);
    let mut c = Session::new(Device::Cpu).compile(jg);
    let outs = c.run_typed(&[
        ("x", &f64s(&[x_val]), DType::F64),
        ("tangent_x", &f64s(&[1.0]), DType::F64),
    ]);
    let jvp_hvp = f64b(&outs[outs.len() - 1].0);
    assert!(
        jvp_hvp.abs() < 1e-12,
        "jvp(hvp) wrongly reports third deriv as {jvp_hvp}"
    );
    assert!((jvp_hvp - want_third).abs() > 1.0);

    let ng = nth_order_grad(&g, "x", 3);
    let mut c = Session::new(Device::Cpu).compile(ng);
    let outs = c.run_typed(&[("x", &f64s(&[x_val]), DType::F64)]);
    let got = f64b(&outs[0].0);
    assert!(
        (got - want_third).abs() < 1e-10,
        "nth_order_grad: {got} vs {want_third}"
    );
}
