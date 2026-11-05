// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// 2-D conv training gradients through the decomposition pass, against rlx-cpu's
// native `Conv2dBackward*` kernels, at batch > 1 (the batch axis is where a
// `[N, Cout, Ho, Wo] → [Cout, N·Ho·Wo]` layout can go wrong), and the plan's
// size: the static decomposition used to build im2col one element at a time
// (a scalar Gather + its index Constant per patch element), so EEGNet's training
// graph planned to 7.4 M steps and 30–107 GB.

use rlx_ir::{DType, Graph, GraphExt, Shape};

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}
fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

/// d(Σ y²)/d{x, w} for `y = conv2d(x, w)`: rlx-cpu vs the WebGL plan; returns
/// the plan's step count.
#[allow(clippy::too_many_arguments)]
fn conv2d_backward(
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    k: [usize; 2],
    stride: [usize; 2],
    pad: [usize; 2],
    groups: usize,
) -> usize {
    let xs = f32s(n * cin * h * w, |i| (i as f32 * 0.23).sin());
    let ws = f32s(cout * (cin / groups) * k[0] * k[1], |i| {
        0.04 * (i as f32 % 9.0) - 0.15
    });

    let mut g = Graph::new("conv2d_bwd");
    let x = g.param("x", sh(&[n, cin, h, w]));
    let weight = g.param("w", sh(&[cout, cin / groups, k[0], k[1]]));
    let y = g.conv2d(x, weight, k, stride, pad, [1, 1], groups);
    let sq = g.mul(y, y);
    let loss = g.sum(sq, vec![0, 1, 2, 3], false);
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
                "n={n} groups={groups} output {oi}[{i}]: ref={a} webgl={b}"
            );
        }
    }
    plan.steps.len()
}

#[test]
fn batched_conv2d_backward_matches_cpu() {
    conv2d_backward(2, 3, 6, 7, 4, [3, 3], [1, 1], [0, 0], 1);
    conv2d_backward(3, 2, 7, 9, 5, [2, 3], [2, 1], [1, 2], 1);
}

#[test]
fn batched_grouped_conv2d_backward_matches_cpu() {
    conv2d_backward(2, 4, 6, 7, 4, [3, 3], [1, 1], [1, 1], 2);
    // depthwise, EEGNet's spatial filter shape: a (C, 1) kernel over all channels
    conv2d_backward(2, 4, 5, 9, 8, [5, 1], [1, 1], [0, 0], 4);
}

/// The plan does not grow with the number of patch elements: an EEGNet-shaped
/// temporal conv (a 1 × 16 kernel along time, 4 × 48 inputs, batch 2) has
/// 2·4·49·16 = 6272 patch entries, and plans in far fewer steps (one per
/// entry it was).
#[test]
fn conv2d_backward_plan_is_not_per_element() {
    let steps = conv2d_backward(2, 1, 4, 48, 4, [1, 16], [1, 1], [0, 8], 1);
    assert!(
        steps < 400,
        "{steps} steps: the decomposition is per element again"
    );
}
