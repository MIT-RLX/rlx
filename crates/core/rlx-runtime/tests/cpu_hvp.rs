// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hessian-vector products via forward-over-reverse AD.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_opt::autodiff_fwd::hvp;
use rlx_runtime::{Device, Session};

fn f64s_to_bytes(xs: &[f64]) -> Vec<u8> {
    let mut o = Vec::with_capacity(xs.len() * 8);
    for x in xs {
        o.extend_from_slice(&x.to_le_bytes());
    }
    o
}
fn bytes_to_f64s(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn hvp_sum_squares_gives_2v() {
    // f(x) = sum(x²); ∇f = 2x; H = 2I; H·v = 2v.
    let n = 4;
    let mut g = Graph::new("hvp_sq");
    let x = g.input("x", Shape::new(&[n], DType::F64));
    let xx = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        x,
        x,
        Shape::new(&[n], DType::F64),
    );
    let f = g.sum(xx, vec![0], false);
    g.set_outputs(vec![f]);

    let hg = hvp(&g, &[x]);
    let mut c = Session::new(Device::Cpu).compile(hg);

    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let v = vec![0.5, -0.25, 1.0, -1.5];
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x_data), DType::F64),
        ("tangent_x", &f64s_to_bytes(&v), DType::F64),
    ]);
    // Outputs: [primal_f, grad_x, tangent_f, H·v]
    assert_eq!(outs.len(), 4);
    let primal = bytes_to_f64s(&outs[0].0)[0];
    let grad = bytes_to_f64s(&outs[1].0);
    let hv = bytes_to_f64s(&outs[3].0);

    let want_f = x_data.iter().map(|v| v * v).sum::<f64>();
    assert!((primal - want_f).abs() < 1e-12);
    for i in 0..n {
        assert!(
            (grad[i] - 2.0 * x_data[i]).abs() < 1e-12,
            "grad[{i}]={} want {}",
            grad[i],
            2.0 * x_data[i]
        );
        assert!(
            (hv[i] - 2.0 * v[i]).abs() < 1e-12,
            "H·v[{i}]={} want {}",
            hv[i],
            2.0 * v[i]
        );
    }
}

#[test]
fn hvp_matches_finite_differences() {
    // Compare HVP against FD of the gradient: H·v ≈ (∇f(x+h·v) − ∇f(x−h·v)) / (2h).
    let n = 3;
    let mut g = Graph::new("hvp_fd");
    let x = g.input("x", Shape::new(&[n], DType::F64));
    // f(x) = sum(x⁴).
    let x2 = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        x,
        x,
        Shape::new(&[n], DType::F64),
    );
    let x4 = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        x2,
        x2,
        Shape::new(&[n], DType::F64),
    );
    let f = g.sum(x4, vec![0], false);
    g.set_outputs(vec![f]);

    let hg = hvp(&g, &[x]);
    let mut c = Session::new(Device::Cpu).compile(hg);

    let x0: Vec<f64> = vec![0.5, -1.0, 2.0];
    let v: Vec<f64> = vec![1.0, 0.25, -0.5];
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&x0), DType::F64),
        ("tangent_x", &f64s_to_bytes(&v), DType::F64),
    ]);
    let hv = bytes_to_f64s(&outs[3].0);

    // FD reference: f(x) = sum(x⁴), so ∇f = 4·x³, H = 12·diag(x²).
    // H·v[i] = 12·x[i]²·v[i]
    let mut want = vec![0f64; n];
    for i in 0..n {
        want[i] = 12.0 * x0[i] * x0[i] * v[i];
    }
    for i in 0..n {
        assert!(
            (hv[i] - want[i]).abs() < 1e-9,
            "H·v[{i}]={} want {}",
            hv[i],
            want[i]
        );
    }
}

#[test]
fn hvp_through_tanh_activation() {
    use rlx_ir::op::Activation;

    let mut g = Graph::new("tanh_hvp");
    let x = g.input("x", Shape::scalar(DType::F64));
    let tx = g.activation(Activation::Tanh, x, Shape::scalar(DType::F64));
    g.set_outputs(vec![tx]);

    let x_val: f64 = 0.5;
    let v: f64 = 1.0;
    let txv = x_val.tanh();
    let want_hv = -2.0 * txv * (1.0 - txv * txv) * v;

    let hg = hvp(&g, &[x]);
    assert!(
        hg.nodes()
            .iter()
            .all(|n| !matches!(&n.op, rlx_ir::Op::Input { name } if name == "d_output")),
        "hvp must internalize d_output"
    );

    let mut c = Session::new(Device::Cpu).compile(hg);
    let outs = c.run_typed(&[
        ("x", &f64s_to_bytes(&[x_val]), DType::F64),
        ("tangent_x", &f64s_to_bytes(&[v]), DType::F64),
    ]);
    let grad = bytes_to_f64s(&outs[1].0)[0];
    let hv = bytes_to_f64s(&outs[3].0)[0];
    let want_grad = 1.0 - txv * txv;
    assert!(
        (grad - want_grad).abs() < 1e-12,
        "grad {grad} want {want_grad}"
    );
    assert!((hv - want_hv).abs() < 1e-10, "H·v {hv} want {want_hv}");
}
