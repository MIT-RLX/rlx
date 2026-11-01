// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Parity for composite ops: those rlx-webgl lowers natively (LayerNorm,
// RmsNorm, StopGradient) and those it gets "for free" by legalizing to
// primitives (GroupNorm, BatchNormInference) — all checked vs RLX's CPU
// backend. Also covers LayerNorm's backward via the decomposition pass.

use rlx_ir::{DType, Graph, GraphExt, Shape};

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}
fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

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

#[test]
fn layer_norm() {
    let xs = f32s(8, |i| i as f32 * 0.4 - 1.5);
    let gs = f32s(4, |i| 1.0 + i as f32 * 0.1);
    let bs = f32s(4, |i| i as f32 * 0.05);
    let mut g = Graph::new("ln");
    let x = g.input("x", sh(&[2, 4]));
    let gamma = g.input("g", sh(&[4]));
    let beta = g.input("b", sh(&[4]));
    let y = g.ln(x, gamma, beta, 1e-5);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("g", &gs), ("b", &bs)]);
}

#[test]
fn rms_norm() {
    let xs = f32s(8, |i| i as f32 * 0.4 - 1.5);
    let gs = f32s(4, |i| 1.0 + i as f32 * 0.1);
    let bs = f32s(4, |i| i as f32 * 0.05);
    let mut g = Graph::new("rms");
    let x = g.input("x", sh(&[2, 4]));
    let gamma = g.input("g", sh(&[4]));
    let beta = g.input("b", sh(&[4]));
    let y = g.rms_norm(x, gamma, beta, 1e-6);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("g", &gs), ("b", &bs)]);
}

#[test]
fn stop_gradient_is_identity() {
    let xs = f32s(6, |i| i as f32 - 3.0);
    let mut g = Graph::new("sg");
    let x = g.input("x", sh(&[2, 3]));
    let y = g.stop_gradient(x);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs)]);
}

/// GroupNorm has no native kernel — it legalizes to primitives inside build_plan.
#[test]
fn group_norm_via_legalization() {
    // NCHW (the lowering pass expects 4D).
    let (n, c, h, w) = (1usize, 4usize, 2usize, 2usize);
    let xs = f32s(n * c * h * w, |i| (i as f32 * 0.3).cos() + 0.2);
    let gs = f32s(c, |i| 1.0 + i as f32 * 0.1);
    let bs = f32s(c, |i| i as f32 * 0.02);
    let mut g = Graph::new("gn");
    let x = g.input("x", sh(&[n, c, h, w]));
    let gamma = g.input("g", sh(&[c]));
    let beta = g.input("b", sh(&[c]));
    let y = g.group_norm(x, gamma, beta, 2, 1e-5);
    g.set_outputs(vec![y]);
    parity(g, &[("x", &xs), ("g", &gs), ("b", &bs)]);
}

/// BatchNormInference also legalizes to primitives inside build_plan.
/// RLX normalizes over the **last** axis (channel-last).
#[test]
fn batch_norm_inference_via_legalization() {
    let (n, c) = (2usize, 4usize);
    let xs = f32s(n * c, |i| i as f32 * 0.2 - 1.0);
    let gs = f32s(c, |i| 1.0 + i as f32 * 0.1);
    let bs = f32s(c, |i| i as f32 * 0.05);
    let mean = f32s(c, |i| i as f32 * 0.1);
    let var = f32s(c, |i| 0.5 + i as f32 * 0.2);
    let mut g = Graph::new("bn");
    let x = g.input("x", sh(&[n, c]));
    let gamma = g.input("g", sh(&[c]));
    let beta = g.input("b", sh(&[c]));
    let rmean = g.input("m", sh(&[c]));
    let rvar = g.input("v", sh(&[c]));
    let y = g.batch_norm_inference(x, gamma, beta, rmean, rvar, 1e-5);
    g.set_outputs(vec![y]);
    parity(
        g,
        &[
            ("x", &xs),
            ("g", &gs),
            ("b", &bs),
            ("m", &mean),
            ("v", &var),
        ],
    );
}

/// Backward through LayerNorm: the dedicated LayerNormBackward* ops are
/// decomposed to primitives by build_plan's legalization pass.
#[test]
fn layer_norm_backward_via_decomposition() {
    let xs = f32s(8, |i| i as f32 * 0.3 - 1.0);
    let gs = f32s(4, |i| 1.0 + i as f32 * 0.1);
    let bs = f32s(4, |i| i as f32 * 0.05);

    let mut g = Graph::new("lnbwd");
    let x = g.input("x", sh(&[2, 4]));
    let gamma = g.param("gamma", sh(&[4]));
    let beta = g.param("beta", sh(&[4]));
    let y = g.ln(x, gamma, beta, 1e-5);
    let sq = g.mul(y, y);
    let loss = g.sum(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);

    let bwd = rlx_autodiff::grad_with_loss(&g, &[gamma, beta]);

    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(bwd.clone());
    sess.set_param("gamma", &gs);
    sess.set_param("beta", &bs);
    let reference = sess.run(&[("x", &xs), ("d_output", &[1.0])]);

    let plan = rlx_webgl::build_plan(&bwd).expect("plan");
    let got = rlx_webgl::run_cpu(
        &plan,
        &[
            ("x", &xs),
            ("d_output", &[1.0]),
            ("gamma", &gs),
            ("beta", &bs),
        ],
    )
    .expect("run_cpu");

    assert_eq!(reference.len(), got.len());
    for (r, gv) in reference.iter().zip(&got) {
        for (a, b) in r.iter().zip(gv) {
            assert!((a - b).abs() <= 2e-4 * (1.0 + a.abs()), "ref={a} webgl={b}");
        }
    }
}

/// Backward through Conv3d (training): Conv3dBackwardInput / Weight are
/// decomposed by build_plan's legalization pass and then run on the WebGL
/// kernels. Checked for plain and grouped convs.
fn conv3d_backward(groups: usize) {
    let (n, cin, d, h, w) = (1usize, 4usize, 3usize, 4usize, 4usize);
    let (cout, k) = (4usize, 2usize);
    let xs = f32s(n * cin * d * h * w, |i| (i as f32 * 0.23).sin());
    let ws = f32s(cout * (cin / groups) * k * k * k, |i| {
        0.04 * (i as f32 % 9.0) - 0.15
    });

    let mut g = Graph::new("conv3d_bwd");
    let x = g.param("x", sh(&[n, cin, d, h, w]));
    let weight = g.param("w", sh(&[cout, cin / groups, k, k, k]));
    let y = g.conv3d(x, weight, [1, 2, 1], [1, 0, 1], [1, 1, 1], groups);
    let sq = g.mul(y, y);
    let loss = g.sum(sq, vec![0, 1, 2, 3, 4], false);
    g.set_outputs(vec![loss]);

    let bwd = rlx_autodiff::grad_with_loss(&g, &[x, weight]);

    let mut sess = rlx_runtime::Session::new(rlx_runtime::Device::Cpu).compile(bwd.clone());
    sess.set_param("x", &xs);
    sess.set_param("w", &ws);
    let reference = sess.run(&[("d_output", &[1.0])]);

    let plan = rlx_webgl::build_plan(&bwd).expect("plan");
    let got = rlx_webgl::run_cpu(&plan, &[("d_output", &[1.0]), ("x", &xs), ("w", &ws)])
        .expect("run_cpu");

    assert_eq!(reference.len(), got.len());
    for (oi, (r, gv)) in reference.iter().zip(&got).enumerate() {
        assert_eq!(r.len(), gv.len(), "output {oi} length");
        for (i, (a, b)) in r.iter().zip(gv).enumerate() {
            assert!(
                (a - b).abs() <= 2e-4 * (1.0 + a.abs()),
                "output {oi}[{i}]: ref={a} webgl={b}"
            );
        }
    }
}

#[test]
fn conv3d_backward_via_decomposition() {
    conv3d_backward(1);
}

#[test]
fn conv3d_grouped_backward_via_decomposition() {
    conv3d_backward(2);
}
