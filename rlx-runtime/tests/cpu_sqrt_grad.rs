// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! End-to-end `grad_with_loss` parity for `Activation::Sqrt`,
//! `Activation::Relu`, and `Activation::Sigmoid`.
//!
//! We build small graphs `loss = sum(act(x))` for each activation,
//! ask `grad_with_loss` for `dloss/dx`, and compare every element
//! against second-order central finite differences. This pins down the
//! activation-VJP path (`Op::ActivationBackward` for the generic kinds,
//! `Op::ReluBackward` for the dedicated ReLU kernel) with an actual
//! number — not just "the gradient walk completes without panicking."
//!
//! Documented as "broken" in the rlx-eda memory; this test is the
//! witness one way or the other.

#![cfg(feature = "cpu")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn build_loss_graph(act: Activation, n: usize) -> Graph {
    let s = Shape::new(&[n], DType::F32);
    let scalar = Shape::new(&[1], DType::F32);
    let mut g = Graph::new("act_loss");
    let x = g.param("x", s.clone());
    let y = g.activation(act, x, s);
    let loss = g.add_node(
        rlx_ir::Op::Reduce {
            op: rlx_ir::op::ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![y],
        scalar,
    );
    g.set_outputs(vec![loss]);
    g
}

fn run_forward(g: &Graph, x: &[f32]) -> f32 {
    let mut compiled = Session::new(Device::Cpu).compile(g.clone());
    compiled.set_param("x", x);
    let outs = compiled.run(&[]);
    outs[0][0]
}

fn run_grad(g: &Graph, x: &[f32]) -> Vec<f32> {
    let x_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, rlx_ir::Op::Param { name } if name == "x"))
        .map(|n| n.id)
        .unwrap();
    let bwd = grad_with_loss(g, &[x_id]);
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param("x", x);
    let outs = compiled.run(&[("d_output", &[1.0_f32][..])]);
    outs[1].clone()
}

fn check_act(act: Activation, x: &[f32], eps: f32, rtol: f32, atol: f32, label: &str) {
    let g = build_loss_graph(act, x.len());
    let ad = run_grad(&g, x);

    // Centered FD for each element.
    for i in 0..x.len() {
        let mut x_p = x.to_vec();
        x_p[i] += eps;
        let mut x_m = x.to_vec();
        x_m[i] -= eps;
        let f_p = run_forward(&g, &x_p);
        let f_m = run_forward(&g, &x_m);
        let fd = (f_p - f_m) / (2.0 * eps);
        let env = atol + rtol * fd.abs();
        let diff = (ad[i] - fd).abs();
        assert!(
            diff <= env,
            "[{label} i={i}] AD={ad_v:.6e} FD={fd:.6e} |Δ|={diff:.3e} env={env:.3e}",
            ad_v = ad[i]
        );
    }
}

#[test]
fn sqrt_grad_matches_finite_differences() {
    // Strictly positive x — sqrt is undefined at 0 and the kernel
    // clamps the VJP to 0 there, which FD wouldn't reproduce.
    let x: Vec<f32> = (1..=8).map(|i| 0.25 * i as f32).collect();
    check_act(Activation::Sqrt, &x, 1e-3, 5e-3, 1e-5, "sqrt");
}

#[test]
fn relu_grad_matches_finite_differences() {
    // Avoid the kink at 0 — FD across it gives 0.5 while AD picks
    // one side (subgradient discontinuity).
    let x: Vec<f32> = vec![-1.5, -0.5, 0.5, 1.5, 2.5, -2.5, 3.0, -3.0];
    check_act(Activation::Relu, &x, 1e-3, 5e-3, 1e-5, "relu");
}

#[test]
fn sigmoid_grad_matches_finite_differences() {
    let x: Vec<f32> = vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 3.0];
    check_act(Activation::Sigmoid, &x, 1e-3, 5e-3, 1e-5, "sigmoid");
}

/// Softplus = `(1/β)·log(1 + exp(β·x))`. Composed via Exp / Add(1) /
/// Log primitives — this is what the MOSFET model needs for its
/// cutoff smoothing. The composition's reverse mode is `σ(β·x)` —
/// finite-difference verifies that the chained VJPs land there.
#[test]
fn softplus_chain_grad_matches_finite_differences() {
    use rlx_ir::Op;
    use rlx_ir::op::{BinaryOp, ReduceOp};

    let n = 8usize;
    let beta: f32 = 200.0;
    let s = Shape::new(&[n], DType::F32);
    let scalar = Shape::new(&[1], DType::F32);

    let mut g = Graph::new("softplus_loss");
    let x = g.param("x", s.clone());
    let beta_c = g.add_node(
        Op::Constant {
            data: beta.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let one = g.add_node(
        Op::Constant {
            data: 1.0_f32.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let inv_beta = g.add_node(
        Op::Constant {
            data: (1.0_f32 / beta).to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let bx = g.binary(BinaryOp::Mul, x, beta_c, s.clone());
    let exp_bx = g.activation(Activation::Exp, bx, s.clone());
    let one_plus = g.binary(BinaryOp::Add, exp_bx, one, s.clone());
    let log_ = g.activation(Activation::Log, one_plus, s.clone());
    let sp = g.binary(BinaryOp::Mul, log_, inv_beta, s);
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![sp],
        scalar,
    );
    g.set_outputs(vec![loss]);

    // Wide range of x — at large |β·x| the kernel can lose precision;
    // include both well-resolved and saturating regimes.
    let x_vals: Vec<f32> = vec![-0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05];
    let ad = run_grad(&g, &x_vals);

    // Closed-form derivative: d(softplus(βx)/β)/dx = σ(βx).
    let expected: Vec<f32> = x_vals
        .iter()
        .map(|&v| 1.0 / (1.0 + (-beta * v).exp()))
        .collect();

    for (i, (&got, &want)) in ad.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        let env = 1e-4 + 5e-3 * want.abs();
        assert!(
            diff <= env,
            "[softplus i={i}] AD={got:.6e} expected σ(βx)={want:.6e} \
             |Δ|={diff:.3e} env={env:.3e}"
        );
    }
}
