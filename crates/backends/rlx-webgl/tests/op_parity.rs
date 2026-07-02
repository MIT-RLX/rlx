// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.
//
// Per-op-family parity: the WebGL lowering (planner + CPU executor) must match
// RLX's CPU backend. Covers the kernels added beyond the MLP set.

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, GraphExt, Shape};

fn f32s(shape: &[usize], f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..shape.iter().product::<usize>()).map(f).collect()
}

/// Run `g` through RLX's CPU backend and through the WebGL lowering; assert equal.
fn parity(g: Graph, inputs: &[(&str, &[f32])]) {
    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(g.clone());
    let reference = sess.run(inputs);

    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let got = rlx_webgl::run_cpu(&plan, inputs).expect("run_cpu");

    assert_eq!(reference.len(), got.len(), "output count");
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(r.len(), gv.len(), "output {oi} length");
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: reference={a} webgl={b}"
            );
        }
    }
}

fn shape2(r: usize, c: usize) -> Shape {
    Shape::new(&[r, c], DType::F32)
}

#[test]
fn binary_max_min_pow() {
    let a = f32s(&[2, 3], |i| i as f32 - 2.5);
    let b = f32s(&[2, 3], |i| 1.5 - i as f32);
    for op in [BinaryOp::Max, BinaryOp::Min, BinaryOp::Pow] {
        let mut g = Graph::new("bin");
        let ai = g.input("a", shape2(2, 3));
        let bi = g.input("b", shape2(2, 3));
        // Pow needs a positive base to stay real.
        let (av, bv) = if op == BinaryOp::Pow {
            (
                f32s(&[2, 3], |i| 0.5 + i as f32 * 0.3),
                f32s(&[2, 3], |i| 1.0 + i as f32 * 0.2),
            )
        } else {
            (a.clone(), b.clone())
        };
        let y = g.binary(op, ai, bi, shape2(2, 3));
        g.set_outputs(vec![y]);
        parity(g, &[("a", &av), ("b", &bv)]);
    }
}

#[test]
fn activations() {
    // Mixed-sign domain for the everywhere-defined activations.
    let xs = f32s(&[2, 4], |i| i as f32 * 0.5 - 2.0);
    for act in [
        Activation::Relu,
        Activation::Neg,
        Activation::Exp,
        Activation::Sigmoid,
        Activation::Tanh,
        Activation::Abs,
        Activation::Sin,
        Activation::Cos,
        Activation::Silu,
    ] {
        let mut g = Graph::new("act");
        let x = g.input("x", shape2(2, 4));
        let y = g.activation(act, x, shape2(2, 4));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
    // Positive domain for Log / Sqrt / Rsqrt.
    let xp = f32s(&[2, 4], |i| 0.25 + i as f32 * 0.5);
    for act in [Activation::Log, Activation::Sqrt, Activation::Rsqrt] {
        let mut g = Graph::new("act_pos");
        let x = g.input("x", shape2(2, 4));
        let y = g.activation(act, x, shape2(2, 4));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xp)]);
    }
}

#[test]
fn reduce_ops() {
    let xs = f32s(&[3, 4], |i| (i as f32 * 0.37).sin() + 0.5);
    for op in [
        ReduceOp::Sum,
        ReduceOp::Mean,
        ReduceOp::Max,
        ReduceOp::Min,
        ReduceOp::Prod,
    ] {
        let mut g = Graph::new("red");
        let x = g.input("x", shape2(3, 4));
        let y = g.reduce(x, op, vec![1], false, Shape::new(&[3], DType::F32));
        g.set_outputs(vec![y]);
        parity(g, &[("x", &xs)]);
    }
    // Reduce all axes → scalar.
    let mut g = Graph::new("red_all");
    let x = g.input("x", shape2(3, 4));
    let y = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1],
        false,
        Shape::scalar(DType::F32),
    );
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn softmax_last_axis() {
    let xs = f32s(&[2, 5], |i| i as f32 * 0.3 - 1.0);
    let mut g = Graph::new("sm");
    let x = g.input("x", shape2(2, 5));
    let y = g.softmax(x, -1, shape2(2, 5));
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

#[test]
fn reverse_axis() {
    let xs = f32s(&[2, 4], |i| i as f32);
    let mut g = Graph::new("rev");
    let x = g.input("x", shape2(2, 4));
    let y = g.reverse(x, vec![1]);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

/// Backward through a non-ReLU activation exercises `ActivationBackward`.
#[test]
fn sigmoid_mlp_backward_matches_cpu() {
    let (in_dim, hidden, out_dim) = (3usize, 4usize, 2usize);
    let mut g = Graph::new("sig_mlp");
    let x = g.input("x", shape2(1, in_dim));
    let w1 = g.param("w1", shape2(in_dim, hidden));
    let b1 = g.param("b1", shape2(1, hidden));
    let w2 = g.param("w2", shape2(hidden, out_dim));
    let b2 = g.param("b2", shape2(1, out_dim));
    let h = g.matmul(x, w1, shape2(1, hidden));
    let h = g.add(h, b1);
    let h = g.activation(Activation::Sigmoid, h, shape2(1, hidden));
    let y = g.matmul(h, w2, shape2(1, out_dim));
    let y = g.add(y, b2);
    let t = g.input("target", shape2(1, out_dim));
    let d = g.sub(y, t);
    let sq = g.mul(d, d);
    let loss = g.sum(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);

    let bwd = rlx_autodiff::grad_with_loss(&g, &[w1, b1, w2, b2]);

    let xd = vec![0.5f32, -0.3, 0.8];
    let td = vec![0.2f32, 0.7];
    let w1d = f32s(&[in_dim, hidden], |i| 0.1 * i as f32 - 0.2);
    let b1d = vec![0.0f32; hidden];
    let w2d = f32s(&[hidden, out_dim], |i| 0.15 * i as f32 - 0.1);
    let b2d = vec![0.0f32; out_dim];

    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(bwd.clone());
    sess.set_param("w1", &w1d);
    sess.set_param("b1", &b1d);
    sess.set_param("w2", &w2d);
    sess.set_param("b2", &b2d);
    let reference = sess.run(&[("x", &xd), ("target", &td), ("d_output", &[1.0])]);

    let plan = rlx_webgl::build_plan(&bwd).expect("plan");
    let got = rlx_webgl::run_cpu(
        &plan,
        &[
            ("x", &xd),
            ("target", &td),
            ("d_output", &[1.0]),
            ("w1", &w1d),
            ("b1", &b1d),
            ("w2", &w2d),
            ("b2", &b2d),
        ],
    )
    .expect("run_cpu");

    assert_eq!(reference.len(), got.len());
    for (r, gv) in reference.iter().zip(&got) {
        for (a, b) in r.iter().zip(gv) {
            assert!((a - b).abs() <= 1e-4 * (1.0 + a.abs()), "ref={a} webgl={b}");
        }
    }
}
