// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native `Op::BatchNormInference` and its three backwards on wgpu vs rlx-cpu.
//!
//! wgpu was the only f32-uniform arena backend host-routing these — CUDA and
//! ROCm share `batch_norm_inference.cu`, Vulkan has four `.comp`. A readback per
//! BN layer is dozens per forward pass in a CNN.
//!
//! The gates split exactly where the two implementations stop being the same
//! computation, and the split was measured, not assumed:
//!
//! | kernel | expression | gate |
//! |---|---|---|
//! | `bwd_input` | `dy · γ · inv` | **bit-exact** |
//! | `bwd_beta`  | `Σ dy` | **bit-exact** |
//! | forward | `γ · x̂ + β` | 1e-6 relative |
//! | `bwd_gamma` | `acc + dy · x̂` | 1e-6 relative |
//!
//! The two loose ones are precisely the two containing a multiply-**add**, which
//! Metal contracts into an `fma`: one rounding where the CPU does two. That is a
//! *more* accurate result, not a less accurate one, and demanding equality would
//! be demanding the GPU not use its fused multiply-add. The two without a
//! multiply-add come out bit-identical, which is what makes that diagnosis solid
//! rather than a shrug — a decode or addressing error would break all four.
//!
//! Getting even those two exact took work worth keeping: the reductions
//! accumulate down the rows in the CPU's order (f32 addition is not associative,
//! so the faster tree shape would not be the same computation), and `inv` is
//! `1 / sqrt(var + eps)` with a Newton-corrected sqrt and a correctly-rounded
//! divide. A bare `1.0 / sqrt(x)` under Metal's fast math was 1 ULP off and
//! broke `bwd_input` too; the Vulkan twins still use approximate `inversesqrt`.

#![cfg(all(feature = "gpu", feature = "cpu"))]

use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

fn wave(n: usize, phase: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * phase).sin() * amp).collect()
}

fn run_both(build: impl Fn() -> Graph, inputs: &[(&str, Vec<f32>)]) -> (Vec<f32>, Vec<f32>) {
    let borrowed: Vec<(&str, &[f32])> = inputs.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    let run = |d: Device| Session::new(d).compile(build()).run(&borrowed).remove(0);
    (run(Device::Gpu), run(Device::Cpu))
}

/// For the kernels with no multiply-add: nothing may differ.
fn both(what: &str, build: impl Fn() -> Graph, inputs: &[(&str, Vec<f32>)]) {
    let (gpu, cpu) = run_both(build, inputs);
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    for (i, (a, b)) in gpu.iter().zip(&cpu).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: element {i} — wgpu {a} vs cpu {b}"
        );
    }
}

/// For the kernels whose multiply-add Metal contracts into an `fma`. Tight
/// enough that only the contraction fits: an FMA differs by at most 1 ULP
/// (~1.2e-7 relative), while any addressing or decode error is orders larger.
fn close(what: &str, build: impl Fn() -> Graph, inputs: &[(&str, Vec<f32>)]) {
    let (gpu, cpu) = run_both(build, inputs);
    assert_eq!(gpu.len(), cpu.len(), "{what}: length");
    let scale = cpu.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    let mut worst = 0f32;
    for (i, (a, b)) in gpu.iter().zip(&cpu).enumerate() {
        let rel = (a - b).abs() / scale;
        worst = worst.max(rel);
        assert!(
            rel <= 1e-6,
            "{what}: element {i} — wgpu {a} vs cpu {b} (rel {rel:e})"
        );
    }
    eprintln!("{what}: max rel diff {worst:.2e}");
}

const ROWS: usize = 7;
const C: usize = 5;

fn stats() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = wave(ROWS * C, 0.29, 2.0);
    let gamma = wave(C, 0.7, 1.3);
    let beta = wave(C, 1.1, 0.6);
    let mean = wave(C, 0.4, 0.5);
    // Strictly positive variance, spanning a couple of decades so `inv_std`
    // covers more than one exponent.
    let var: Vec<f32> = (0..C)
        .map(|c| 0.01 * (c as f32 + 1.0).powi(2) + 0.05)
        .collect();
    (x, gamma, beta, mean, var)
}

#[test]
fn forward_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let (x, gamma, beta, mean, var) = stats();
    for eps in [1e-5f32, 1e-3, 0.5] {
        close(
            &format!("bn forward eps={eps}"),
            || {
                let mut g = Graph::new("bn");
                let xi = g.input("x", Shape::new(&[ROWS, C], DType::F32));
                let gi = g.input("gamma", Shape::new(&[C], DType::F32));
                let bi = g.input("beta", Shape::new(&[C], DType::F32));
                let mi = g.input("mean", Shape::new(&[C], DType::F32));
                let vi = g.input("var", Shape::new(&[C], DType::F32));
                let y = g.add_node(
                    Op::BatchNormInference { eps },
                    vec![xi, gi, bi, mi, vi],
                    Shape::new(&[ROWS, C], DType::F32),
                );
                g.set_outputs(vec![y]);
                g
            },
            &[
                ("x", x.clone()),
                ("gamma", gamma.clone()),
                ("beta", beta.clone()),
                ("mean", mean.clone()),
                ("var", var.clone()),
            ],
        );
    }
}

#[test]
fn backward_input_matches_cpu_bit_exactly() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let (x, gamma, _beta, mean, var) = stats();
    let dy = wave(ROWS * C, 0.53, 1.1);
    both(
        "bn bwd input",
        || {
            let mut g = Graph::new("bn_dx");
            let xi = g.input("x", Shape::new(&[ROWS, C], DType::F32));
            let gi = g.input("gamma", Shape::new(&[C], DType::F32));
            let mi = g.input("mean", Shape::new(&[C], DType::F32));
            let vi = g.input("var", Shape::new(&[C], DType::F32));
            let di = g.input("dy", Shape::new(&[ROWS, C], DType::F32));
            let y = g.add_node(
                Op::BatchNormInferenceBackwardInput { eps: 1e-4 },
                vec![xi, gi, mi, vi, di],
                Shape::new(&[ROWS, C], DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[
            ("x", x),
            ("gamma", gamma),
            ("mean", mean),
            ("var", var),
            ("dy", dy),
        ],
    );
}

#[test]
fn backward_gamma_matches_cpu() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let (x, _gamma, _beta, mean, var) = stats();
    let dy = wave(ROWS * C, 0.37, 1.7);
    close(
        "bn bwd gamma",
        || {
            let mut g = Graph::new("bn_dg");
            let xi = g.input("x", Shape::new(&[ROWS, C], DType::F32));
            let mi = g.input("mean", Shape::new(&[C], DType::F32));
            let vi = g.input("var", Shape::new(&[C], DType::F32));
            let di = g.input("dy", Shape::new(&[ROWS, C], DType::F32));
            let y = g.add_node(
                Op::BatchNormInferenceBackwardGamma { eps: 1e-4 },
                vec![xi, mi, vi, di],
                Shape::new(&[C], DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[("x", x), ("mean", mean), ("var", var), ("dy", dy.clone())],
    );
}

#[test]
fn backward_beta_matches_cpu_bit_exactly() {
    let _gpu = common::serialize_gpu();
    if common::skip_unless_available(Device::Gpu, "wgpu") {
        return;
    }
    let dy = wave(ROWS * C, 0.37, 1.7);
    both(
        "bn bwd beta",
        || {
            let mut g = Graph::new("bn_db");
            let di = g.input("dy", Shape::new(&[ROWS, C], DType::F32));
            let y = g.add_node(
                Op::BatchNormInferenceBackwardBeta,
                vec![di],
                Shape::new(&[C], DType::F32),
            );
            g.set_outputs(vec![y]);
            g
        },
        &[("dy", dy)],
    );
}
