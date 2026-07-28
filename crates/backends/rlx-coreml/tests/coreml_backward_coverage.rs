// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Coverage audit: gradient of each major forward op runs on the CoreML (ANE)
// backend and matches the CPU oracle. For each op we build `loss = sum(op(...))`,
// differentiate w.r.t. a parameter, run the backward graph on the ANE (which
// decomposes / natively-lowers it), and compare to the SAME backward graph
// pre-decomposed and run on the CPU.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_ir::op::{Activation, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_runtime::{Device, Session};

fn scalar_sum(g: &mut Graph, x: NodeId, shape: &Shape) -> NodeId {
    let rank = shape.rank();
    let axes: Vec<usize> = (0..rank).collect();
    g.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes,
            keep_dim: false,
        },
        vec![x],
        Shape::from_dims(&[], DType::F32),
    )
}

/// Run a differentiated graph on the ANE and on the CPU (pre-decomposed) and
/// assert the gradient is finite and points the same way (cosine ≈ 1). Returns
/// `false` (and prints) if the ANE compile fails — used to flag known gaps.
fn grad_parity(
    name: &str,
    bwd: &Graph,
    params: &[(&str, Vec<f32>)],
    inputs: &[(&str, &[f32])],
    grad_out_index: usize,
) {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let cpu_g = rlx_opt::rlx_autodiff::decompose_backward_ops_except(bwd.clone(), &[]);
    let mut feeds: Vec<(&str, &[f32])> = inputs.to_vec();
    let seed = [1.0f32];
    feeds.push(("d_output", &seed));

    let run = |device: Device, graph: &Graph| -> Vec<f32> {
        let mut c = Session::new(device).compile(graph.clone());
        for (n, d) in params {
            c.set_param(n, d);
        }
        c.run(&feeds).remove(grad_out_index)
    };
    let ane = run(Device::Ane, bwd);
    let cpu = run(Device::Cpu, &cpu_g);
    assert_eq!(ane.len(), cpu.len(), "{name}: grad length mismatch");
    assert!(
        ane.iter().all(|v| v.is_finite()),
        "{name}: non-finite ANE grad: {ane:?}"
    );
    let (mut dot, mut na, mut nc) = (0.0f32, 0.0f32, 0.0f32);
    for (&a, &c) in ane.iter().zip(&cpu) {
        dot += a * c;
        na += a * a;
        nc += c * c;
    }
    let cosine = if na > 0.0 && nc > 0.0 {
        dot / (na.sqrt() * nc.sqrt())
    } else {
        1.0
    };
    assert!(
        cosine > 0.999,
        "{name}: ANE vs CPU grad diverged (cosine={cosine})\n  ane={ane:?}\n  cpu={cpu:?}"
    );
    eprintln!("OK {name}: cosine={cosine:.5}");
}

#[test]
fn grad_layer_norm() {
    let (rows, h) = (2usize, 4usize);
    let mut g = Graph::new("ln");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[h], DType::F32));
    let beta = g.param("beta", Shape::new(&[h], DType::F32));
    let y = g.layer_norm(x, gamma, beta, -1, 1e-5, Shape::new(&[rows, h], DType::F32));
    let loss = scalar_sum(&mut g, y, &Shape::new(&[rows, h], DType::F32));
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[gamma]);
    let x_data: Vec<f32> = (0..rows * h).map(|i| i as f32 * 0.2 - 0.4).collect();
    grad_parity(
        "layer_norm dgamma",
        &bwd,
        &[("gamma", vec![1.0; h]), ("beta", vec![0.0; h])],
        &[("x", &x_data)],
        1,
    );
}

#[test]
fn grad_group_norm() {
    let (n, c, hh, w) = (1usize, 4usize, 2usize, 2usize);
    let mut g = Graph::new("gn");
    let x = g.input("x", Shape::new(&[n, c, hh, w], DType::F32));
    let gamma = g.param("gamma", Shape::new(&[c], DType::F32));
    let beta = g.param("beta", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x, gamma, beta, 2, 1e-5);
    let loss = scalar_sum(&mut g, y, &Shape::new(&[n, c, hh, w], DType::F32));
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[gamma]);
    let x_data: Vec<f32> = (0..n * c * hh * w).map(|i| i as f32 * 0.1 - 0.5).collect();
    grad_parity(
        "group_norm dgamma",
        &bwd,
        &[("gamma", vec![1.0; c]), ("beta", vec![0.0; c])],
        &[("x", &x_data)],
        1,
    );
}

#[test]
fn grad_softmax() {
    let (rows, h) = (2usize, 4usize);
    let mut g = Graph::new("sm");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let w = g.param("W", Shape::new(&[h, h], DType::F32));
    let z = g.matmul(x, w, Shape::new(&[rows, h], DType::F32));
    let y = g.softmax(z, -1, Shape::new(&[rows, h], DType::F32));
    let loss = scalar_sum(&mut g, y, &Shape::new(&[rows, h], DType::F32));
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[w]);
    let x_data: Vec<f32> = (0..rows * h).map(|i| i as f32 * 0.1).collect();
    grad_parity(
        "softmax dW",
        &bwd,
        &[("W", vec![0.1; h * h])],
        &[("x", &x_data)],
        1,
    );
}

/// Conv backward w.r.t. INPUT on the ANE (native `conv_transpose` = the conv
/// adjoint). Validated against an ANALYTIC identity, NOT the CPU decompose: the
/// autodiff `compose_conv2d_backward_input` is BROKEN (emits a plain `conv`, the
/// wrong gradient — confirmed: CPU returns 0.168 where truth is 0.994). Since
/// `loss = sum(conv(x·S, W))` is linear in the scalar S, the true `dInput`
/// gradient satisfies `dS == loss` at S=1 — and the ANE's `conv_transpose` hits it.
#[test]
fn grad_conv2d_input_native_matches_analytic() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (n, c, hh, w) = (1usize, 1usize, 4usize, 4usize);
    let (co, kh, kw) = (2usize, 3usize, 3usize);
    let mut g = Graph::new("conv_in");
    let x = g.input("x", Shape::new(&[n, c, hh, w], DType::F32));
    let s = g.param("S", Shape::new(&[1, c, 1, 1], DType::F32));
    let scaled = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        x,
        s,
        Shape::new(&[n, c, hh, w], DType::F32),
    );
    let weight = g.param("Wc", Shape::new(&[co, c, kh, kw], DType::F32));
    let y = g.conv2d(scaled, weight, [kh, kw], [1, 1], [0, 0], [1, 1], 1);
    let y_shape = g.node(y).shape.clone();
    let loss = scalar_sum(&mut g, y, &y_shape);
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[s]);
    let x_data: Vec<f32> = (0..n * c * hh * w).map(|i| i as f32 * 0.1 - 0.5).collect();
    let wdata: Vec<f32> = (0..co * c * kh * kw)
        .map(|i| (i as f32 * 0.03).sin())
        .collect();

    let mut comp = Session::new(Device::Ane).compile(bwd);
    comp.set_param("S", &[1.0]);
    comp.set_param("Wc", &wdata);
    let out = comp.run(&[("x", &x_data), ("d_output", &[1.0f32])]);
    let (loss_v, ds) = (out[0][0], out[1][0]);
    assert!(ds.is_finite(), "conv dInput non-finite: {ds}");
    assert!(
        (ds - loss_v).abs() < 1e-3,
        "conv dInput grad wrong: dS={ds} but analytic truth (=loss) is {loss_v}"
    );
}

#[test]
fn grad_relu_activation() {
    let (rows, h) = (2usize, 4usize);
    let mut g = Graph::new("relu");
    let x = g.input("x", Shape::new(&[rows, h], DType::F32));
    let w = g.param("W", Shape::new(&[h, h], DType::F32));
    let z = g.matmul(x, w, Shape::new(&[rows, h], DType::F32));
    let a = g.activation(Activation::Relu, z, Shape::new(&[rows, h], DType::F32));
    let loss = scalar_sum(&mut g, a, &Shape::new(&[rows, h], DType::F32));
    g.set_outputs(vec![loss]);
    let bwd = rlx_opt::grad_with_loss(&g, &[w]);
    let x_data: Vec<f32> = (0..rows * h).map(|i| i as f32 * 0.1 - 0.3).collect();
    grad_parity(
        "relu dW",
        &bwd,
        &[("W", vec![0.2; h * h])],
        &[("x", &x_data)],
        1,
    );
}

/// Conv backward w.r.t. WEIGHT on the ANE (native transpose-conv kernel),
/// validated against FINITE DIFFERENCES (the decompose oracle is broken / needs
/// Im2Col, so it can't be the reference). The native `dW` must match the
/// numerical gradient of `loss = sum(conv(x, W))`.
#[test]
fn grad_conv2d_weight_native_matches_fd() {
    if !rlx_runtime::is_available(Device::Ane) {
        eprintln!("skip: Device::Ane not available");
        return;
    }
    let (n, c, hh, w) = (1usize, 1usize, 4usize, 4usize);
    let (co, kh, kw) = (2usize, 1usize, 2usize);
    let build = || -> (Graph, NodeId) {
        let mut g = Graph::new("conv_w");
        let x = g.input("x", Shape::new(&[n, c, hh, w], DType::F32));
        let weight = g.param("W", Shape::new(&[co, c, kh, kw], DType::F32));
        let y = g.conv2d(x, weight, [kh, kw], [1, 1], [0, 0], [1, 1], 1);
        let ys = g.node(y).shape.clone();
        let loss = scalar_sum(&mut g, y, &ys);
        g.set_outputs(vec![loss]);
        (g, weight)
    };
    let x_data: Vec<f32> = (0..n * c * hh * w)
        .map(|i| (i as f32 * 0.21).cos())
        .collect();
    let w_base: Vec<f32> = (0..co * c * kh * kw)
        .map(|i| i as f32 * 0.1 - 0.15)
        .collect();

    // Finite-difference gradient on the CPU forward conv.
    let (fwd, _) = build();
    let loss_at = |wv: &[f32]| -> f32 {
        let mut cc = Session::new(Device::Cpu).compile(fwd.clone());
        cc.set_param("W", wv);
        cc.run(&[("x", &x_data)])[0][0]
    };
    let base = loss_at(&w_base);
    let eps = 5e-3f32;
    let fd: Vec<f32> = (0..w_base.len())
        .map(|i| {
            let mut wp = w_base.clone();
            wp[i] += eps;
            (loss_at(&wp) - base) / eps
        })
        .collect();

    // ANE backward dW.
    let (g, wnode) = build();
    let bwd = rlx_opt::grad_with_loss(&g, &[wnode]);
    let mut comp = Session::new(Device::Ane).compile(bwd);
    comp.set_param("W", &w_base);
    let dw = comp
        .run(&[("x", &x_data), ("d_output", &[1.0f32])])
        .remove(1);

    assert_eq!(dw.len(), fd.len(), "dW length mismatch");
    assert!(dw.iter().all(|v| v.is_finite()), "non-finite dW: {dw:?}");
    let (mut dot, mut na, mut nc) = (0.0f32, 0.0f32, 0.0f32);
    let mut max_abs = 0.0f32;
    for (&a, &f) in dw.iter().zip(&fd) {
        dot += a * f;
        na += a * a;
        nc += f * f;
        max_abs = max_abs.max((a - f).abs());
    }
    let cosine = dot / (na.sqrt() * nc.sqrt());
    assert!(
        cosine > 0.999 && max_abs < 5e-2,
        "conv dW vs finite-diff: cosine={cosine}, max_abs={max_abs}\n  dW={dw:?}\n  fd={fd:?}"
    );
    eprintln!("OK conv dWeight: cosine={cosine:.5}, max_abs={max_abs:.4}");
}
