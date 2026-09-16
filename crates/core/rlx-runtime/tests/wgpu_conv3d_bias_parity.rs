// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 3-D convolution with a per-output-channel bias must apply it on wgpu.
//!
//! `Op::Conv` and `Op::Conv3d` are both **2-input** by contract
//! (`Op::num_inputs`, enforced by `verify`) — "bias via Add". A 3-D conv that
//! carries its own bias reaches wgpu as [`Op::FusedConvBiasAct`] (3 inputs),
//! which `rlx_wgpu::unfuse::fold_conv_bias_into_conv3d` rewrites to a
//! bias-carrying `Op::Conv3d` so `conv3d.wgsl` can apply it in its store —
//! saving a full-size `Expand` of the bias plus a full-size `Add`.
//!
//! This guards that folded path end-to-end against CPU. The failure it exists
//! for is silent: if the lowering does not hand `bias_off` / `has_bias` to the
//! shader, the shader skips `acc + arena[bias_off + co]` and the bias simply
//! vanishes — a wrong tensor, not a panic. (The rank-dispatched `Op::Conv` arm
//! had exactly that defect, unreachable only because `verify` rejects a
//! 3-input `Op::Conv`.)

//! The whole module is gated on `gpu`: the helpers below exist only to feed
//! the one wgpu test, so without that feature they are dead code — and under
//! `-D warnings` dead code is a build failure, not a warning.
#![cfg(feature = "gpu")]

mod common;

use common::skip_unless_available;
use rlx_ir::op::Op;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const N: usize = 1;
const C_IN: usize = 2;
const C_OUT: usize = 3;
const D: usize = 4;
const H: usize = 4;
const W: usize = 4;
const K: usize = 3;
// 'valid' convolution, stride 1, no padding.
const D_OUT: usize = D - K + 1;
const H_OUT: usize = H - K + 1;
const W_OUT: usize = W - K + 1;

fn shapes() -> (Shape, Shape, Shape, Shape) {
    (
        Shape::new(&[N, C_IN, D, H, W], DType::F32),
        Shape::new(&[C_OUT, C_IN, K, K, K], DType::F32),
        Shape::new(&[C_OUT], DType::F32),
        Shape::new(&[N, C_OUT, D_OUT, H_OUT, W_OUT], DType::F32),
    )
}

/// Rank-5 `FusedConvBiasAct` with no activation and no residual — the shape
/// `fold_conv_bias_into_conv3d` rewrites into a bias-carrying `Op::Conv3d`.
fn conv_graph() -> Graph {
    let (in_s, w_s, b_s, out_s) = shapes();
    let mut g = Graph::new("conv3d_bias");
    let x = g.input("x", in_s);
    let w = g.param("w", w_s);
    let b = g.param("b", b_s);
    let y = g.add_node(
        Op::FusedConvBiasAct {
            kernel_size: vec![K, K, K],
            stride: vec![1, 1, 1],
            padding: vec![0, 0, 0],
            dilation: vec![1, 1, 1],
            groups: 1,
            activation: None,
            has_residual: false,
        },
        vec![x, w, b],
        out_s,
    );
    g.set_outputs(vec![y]);
    g
}

fn weights() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..N * C_IN * D * H * W)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.25)
        .collect();
    let w: Vec<f32> = (0..C_OUT * C_IN * K * K * K)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
        .collect();
    // Distinct, clearly non-zero per-channel bias: if it is dropped, every
    // output channel is off by a different amount.
    let b: Vec<f32> = (0..C_OUT).map(|c| 1.0 + c as f32).collect();
    (x, w, b)
}

fn run(device: Device) -> Vec<f32> {
    let (x, w, b) = weights();
    let mut compiled = Session::new(device).compile(conv_graph());
    compiled.set_param("w", &w);
    compiled.set_param("b", &b);
    compiled
        .run(&[("x", &x)])
        .into_iter()
        .next()
        .expect("one output")
}

#[test]
fn wgpu_conv3d_bias_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }

    let cpu = run(Device::Cpu);
    let gpu = run(Device::Gpu);
    assert_eq!(gpu.len(), cpu.len());

    let max = gpu
        .iter()
        .zip(&cpu)
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max < 1e-5,
        "wgpu 3-D conv disagrees with CPU by {max:.3e} — the folded per-channel \
         bias was dropped in lowering"
    );

    // Guard the guard: if BOTH backends dropped the bias they would still
    // agree. The biases are 1.0 / 2.0 / 3.0 and the unbiased sums are small,
    // so each channel's mean must sit near its own bias.
    let per_ch = D_OUT * H_OUT * W_OUT;
    let means: Vec<f32> = (0..C_OUT)
        .map(|c| gpu[c * per_ch..(c + 1) * per_ch].iter().sum::<f32>() / per_ch as f32)
        .collect();
    for (c, m) in means.iter().enumerate() {
        let want = 1.0 + c as f32;
        assert!(
            (m - want).abs() < 0.75,
            "channel {c} mean {m:.3} is not near its bias {want:.1}; means={means:?} \
             — the bias is missing from both backends"
        );
    }
}
