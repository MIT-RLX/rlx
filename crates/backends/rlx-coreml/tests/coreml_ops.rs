// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Per-op parity tests for the MIL lowering, run through CoreML on-device.
#![cfg(any(target_os = "macos", target_os = "ios"))]
// The erf reference constants are deliberately written at full published
// precision; they're truncated to f32 at compile time.
#![allow(clippy::excessive_precision)]

use rlx_coreml::CoremlExecutable;
use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mx = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        mx <= tol,
        "max abs diff {mx} > {tol}\n got {a:?}\n want {b:?}"
    );
}

fn run_unary(act: Activation, x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mut g = Graph::new("unary");
    let xi = g.input("x", Shape::new(&[n], DType::F32));
    let y = g.activation(act, xi, Shape::new(&[n], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CoremlExecutable::compile(g);
    exe.run(&[("x", x)]).expect("run").remove(0)
}

#[test]
fn gelu_exact() {
    let x = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let got = run_unary(Activation::Gelu, &x);
    // exact gelu: x * 0.5 * (1 + erf(x/sqrt(2)))
    let want: Vec<f32> = x
        .iter()
        .map(|&v| v * 0.5 * (1.0 + libm_erf(v / std::f32::consts::SQRT_2)))
        .collect();
    approx(&got, &want, 1e-3);
}

// Minimal erf for the reference (Abramowitz–Stegun 7.1.26).
fn libm_erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

#[test]
fn silu_and_sigmoid() {
    let x = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
    let sig = run_unary(Activation::Sigmoid, &x);
    let silu = run_unary(Activation::Silu, &x);
    let want_sig: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
    let want_silu: Vec<f32> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
    approx(&sig, &want_sig, 1e-4);
    approx(&silu, &want_silu, 1e-4);
}

#[test]
fn neg_exp_sqrt() {
    approx(
        &run_unary(Activation::Neg, &[1.0, -2.0, 3.0]),
        &[-1.0, 2.0, -3.0],
        1e-5,
    );
    approx(
        &run_unary(Activation::Exp, &[0.0, 1.0, 2.0]),
        &[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)],
        1e-3,
    );
    approx(
        &run_unary(Activation::Sqrt, &[1.0, 4.0, 9.0]),
        &[1.0, 2.0, 3.0],
        1e-5,
    );
}

#[test]
fn softmax_last_axis() {
    let mut g = Graph::new("softmax");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.softmax(x, -1, Shape::new(&[2, 3], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("x", &[1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0])])
        .expect("run")
        .remove(0);

    let row0 = softmax_ref(&[1.0, 2.0, 3.0]);
    let row1 = softmax_ref(&[1.0, 1.0, 1.0]);
    approx(&out[0..3], &row0, 1e-4);
    approx(&out[3..6], &row1, 1e-4);
}

fn softmax_ref(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&v| v / s).collect()
}

#[test]
fn transpose_2d() {
    let mut g = Graph::new("transpose");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.append_node(
        rlx_ir::Op::Transpose { perm: vec![1, 0] },
        vec![x],
        Shape::new(&[3, 2], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);
    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("x", &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])])
        .expect("run")
        .remove(0);
    approx(&out, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
}

#[test]
fn reshape_flatten() {
    let mut g = Graph::new("reshape");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.reshape(x, vec![6], Shape::new(&[6], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("x", &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])])
        .expect("run")
        .remove(0);
    approx(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
}

#[test]
fn layer_norm_affine() {
    // LayerNorm over last dim with gamma/beta params.
    let mut g = Graph::new("layernorm");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let gamma = g.param("g", Shape::new(&[4], DType::F32));
    let beta = g.param("b", Shape::new(&[4], DType::F32));
    let y = g.layer_norm(x, gamma, beta, -1, 1e-5, Shape::new(&[2, 4], DType::F32));
    g.set_outputs(vec![y]);

    let xs = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0];
    let gv = [1.0f32, 1.0, 1.0, 1.0];
    let bv = [0.0f32, 0.0, 0.0, 0.0];

    let mut exe = CoremlExecutable::compile(g);
    exe.set_param("g", &gv);
    exe.set_param("b", &bv);
    let out = exe.run(&[("x", &xs)]).expect("run").remove(0);

    let want: Vec<f32> = [&xs[0..4], &xs[4..8]]
        .iter()
        .flat_map(|row| layer_norm_ref(row, 1e-5))
        .collect();
    approx(&out, &want, 1e-3);
}

fn layer_norm_ref(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter().map(|&v| (v - mean) * inv).collect()
}

#[test]
fn rms_norm_affine() {
    let mut g = Graph::new("rmsnorm");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let gamma = g.param("g", Shape::new(&[4], DType::F32));
    let y = g.append_node(
        rlx_ir::Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        },
        vec![x, gamma],
        Shape::new(&[2, 4], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);

    let xs = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 1.5, 2.5];
    let gv = [1.0f32, 0.5, 2.0, 1.0];

    let mut exe = CoremlExecutable::compile(g);
    exe.set_param("g", &gv);
    let out = exe.run(&[("x", &xs)]).expect("run").remove(0);

    let want: Vec<f32> = [&xs[0..4], &xs[4..8]]
        .iter()
        .flat_map(|row| rms_norm_ref(row, &gv, 1e-6))
        .collect();
    approx(&out, &want, 1e-3);
}

fn rms_norm_ref(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms = x.iter().map(|&v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(gamma).map(|(&v, &g)| v * inv * g).collect()
}

#[test]
fn reduce_sum_axis() {
    let mut g = Graph::new("reduce");
    let x = g.input("x", Shape::new(&[2, 3], DType::F32));
    let y = g.reduce(
        x,
        rlx_ir::op::ReduceOp::Sum,
        vec![1],
        false,
        Shape::new(&[2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("x", &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])])
        .expect("run")
        .remove(0);
    approx(&out, &[6.0, 15.0], 1e-4);
}

#[test]
fn concat_axis0() {
    let mut g = Graph::new("concat");
    let a = g.input("a", Shape::new(&[1, 3], DType::F32));
    let b = g.input("b", Shape::new(&[1, 3], DType::F32));
    let y = g.concat(vec![a, b], 0, Shape::new(&[2, 3], DType::F32));
    g.set_outputs(vec![y]);
    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("a", &[1.0f32, 2.0, 3.0]), ("b", &[4.0f32, 5.0, 6.0])])
        .expect("run")
        .remove(0);
    approx(&out, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
}

#[test]
fn narrow_slice() {
    // Slice columns [1,3) of a [2,4] tensor.
    let mut g = Graph::new("narrow");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let y = g.append_node(
        rlx_ir::Op::Narrow {
            axis: 1,
            start: 1,
            len: 2,
        },
        vec![x],
        Shape::new(&[2, 2], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);
    let mut exe = CoremlExecutable::compile(g);
    let out = exe
        .run(&[("x", &[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])])
        .expect("run")
        .remove(0);
    approx(&out, &[1.0, 2.0, 5.0, 6.0], 1e-5);
}
