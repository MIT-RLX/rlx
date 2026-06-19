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

//! f64 sibling of `cpu_sqrt_grad.rs`. Same activation-VJP parity check,
//! exercised on f64 graphs.
//!
//! The motivation for splitting f64 out: the f32 thunks read each slot
//! as `&[f32]`, which on an f64 graph silently halves every value. That
//! manifests as backward gradients of `0` for `Op::ReluBackward` and
//! garbage for the others until the dtype-aware lowering lands. The
//! `cpu_sqrt_grad.rs` test passes those cases on f32 because the slot
//! views match dtype; this test pins f64 so the dispatch stays honest.

#![cfg(feature = "cpu")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

fn build_loss_graph(act: Activation, n: usize) -> Graph {
    let s = Shape::new(&[n], DType::F64);
    let scalar = Shape::new(&[1], DType::F64);
    let mut g = Graph::new("act_loss_f64");
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

fn run_forward(g: &Graph, x: &[f64]) -> f64 {
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut compiled = Session::new(Device::Cpu).compile(g.clone());
    compiled.set_param_typed("x", &bytes, DType::F64);
    let outs = compiled.run_typed(&[]);
    f64::from_le_bytes(outs[0].0[..8].try_into().unwrap())
}

fn run_grad(g: &Graph, x: &[f64]) -> Vec<f64> {
    let x_id = g
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, rlx_ir::Op::Param { name } if name == "x"))
        .map(|n| n.id)
        .unwrap();
    let bwd = grad_with_loss(g, &[x_id]);
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut compiled = Session::new(Device::Cpu).compile(bwd);
    compiled.set_param_typed("x", &bytes, DType::F64);
    let one = 1.0_f64.to_le_bytes();
    let outs = compiled.run_typed(&[("d_output", &one, DType::F64)]);
    outs[1]
        .0
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn check_act(act: Activation, x: &[f64], eps: f64, rtol: f64, atol: f64, label: &str) {
    let g = build_loss_graph(act, x.len());
    let ad = run_grad(&g, x);

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
fn sqrt_grad_matches_finite_differences_f64() {
    let x: Vec<f64> = (1..=8).map(|i| 0.25 * i as f64).collect();
    check_act(Activation::Sqrt, &x, 1e-5, 1e-6, 1e-9, "sqrt");
}

#[test]
fn relu_grad_matches_finite_differences_f64() {
    // Avoid kink at 0. Pre-fix this test would all return 0 (the f64
    // graph went through the f32 ReluBackward thunk, which read each
    // 8-byte slot as two f32s and silently used the wrong half).
    let x: Vec<f64> = vec![-1.5, -0.5, 0.5, 1.5, 2.5, -2.5, 3.0, -3.0];
    check_act(Activation::Relu, &x, 1e-5, 1e-6, 1e-9, "relu");
}

#[test]
fn sigmoid_grad_matches_finite_differences_f64() {
    let x: Vec<f64> = vec![-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 3.0];
    check_act(Activation::Sigmoid, &x, 1e-5, 1e-6, 1e-9, "sigmoid");
}

/// Softplus chain on f64: same shape as the f32 test in
/// `cpu_sqrt_grad.rs`. With `β=200` and `|x| ≤ 1` the closed-form
/// `σ(βx)` saturates well — we sample a wider window than f32 because
/// f64's dynamic range tolerates `exp(±200)` cleanly.
#[test]
fn softplus_chain_grad_matches_finite_differences_f64() {
    use rlx_ir::Op;
    use rlx_ir::op::{BinaryOp, ReduceOp};

    let n = 8usize;
    let beta: f64 = 200.0;
    let s = Shape::new(&[n], DType::F64);
    let scalar = Shape::new(&[1], DType::F64);

    let mut g = Graph::new("softplus_loss_f64");
    let x = g.param("x", s.clone());
    let beta_c = g.add_node(
        Op::Constant {
            data: beta.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F64),
    );
    let one = g.add_node(
        Op::Constant {
            data: 1.0_f64.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F64),
    );
    let inv_beta = g.add_node(
        Op::Constant {
            data: (1.0_f64 / beta).to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F64),
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

    let x_vals: Vec<f64> = vec![-0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05];
    let ad = run_grad(&g, &x_vals);

    let expected: Vec<f64> = x_vals
        .iter()
        .map(|&v| 1.0 / (1.0 + (-beta * v).exp()))
        .collect();

    for (i, (&got, &want)) in ad.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        let env = 1e-9 + 1e-6 * want.abs();
        assert!(
            diff <= env,
            "[softplus i={i}] AD={got:.6e} expected σ(βx)={want:.6e} \
             |Δ|={diff:.3e} env={env:.3e}"
        );
    }
}
