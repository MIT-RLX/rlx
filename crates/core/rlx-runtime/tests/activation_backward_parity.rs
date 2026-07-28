// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend parity for `Op::ActivationBackward` (`dx = act'(x)·dy`) vs the
//! CPU reference — the backward counterpart of `elementwise_backend_parity`.
//!
//! Covers the native-backward set (relu-first opcode 0..=17); the tail
//! decomposes at the AD level and has no native kernel. Once the backward
//! kernels are generated from the rlxsl manifest (auto-differentiated from the
//! forward), this test proves the generated derivative matches the CPU oracle
//! on every backend that claims the op.
//!
//!   cargo test -p rlx-runtime --features metal --test activation_backward_parity
//! On the rig: `--features cuda,vulkan,gpu`.

#![cfg(feature = "cpu")]

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available, supports_graph};

const N: usize = 64;
const TOL: f32 = 5e-3;

#[allow(clippy::vec_init_then_push)]
fn available_backends() -> Vec<Device> {
    let mut v: Vec<Device> = Vec::new();
    #[cfg(all(feature = "metal", target_os = "macos"))]
    v.push(Device::Metal);
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    v.push(Device::Mlx);
    #[cfg(feature = "gpu")]
    v.push(Device::Gpu);
    #[cfg(feature = "cuda")]
    v.push(Device::Cuda);
    #[cfg(feature = "rocm")]
    v.push(Device::Rocm);
    #[cfg(feature = "vulkan")]
    v.push(Device::Vulkan);
    v.retain(|&d| is_available(d));
    v
}

/// Activations with a native backward kernel: relu-first opcode 0..=17.
fn native_backward_acts() -> Vec<Activation> {
    Activation::ALL
        .iter()
        .copied()
        .filter(|a| a.opcode_relu_first() < 18)
        .collect()
}

fn bwd_graph(kind: Activation) -> Graph {
    let mut g = Graph::new("act_bwd");
    let x = g.input("x", Shape::new(&[N], DType::F32));
    let dy = g.input("dy", Shape::new(&[N], DType::F32));
    let dx = g.activation_backward(kind, x, dy);
    g.set_outputs(vec![dx]);
    g
}

/// Domain-safe `x` for each activation's derivative (positive for
/// log/sqrt/rsqrt, away from 0 for recip).
fn x_input(kind: Activation) -> Vec<f32> {
    let base: Vec<f32> = (0..N)
        .map(|i| -2.0 + 4.0 * (i as f32) / (N as f32 - 1.0))
        .collect();
    match kind {
        Activation::Log | Activation::Sqrt | Activation::Rsqrt => {
            base.iter().map(|v| v.abs() + 0.5).collect()
        }
        Activation::Recip => base
            .iter()
            .map(|v| if v.abs() < 0.4 { 0.6 } else { *v })
            .collect(),
        Activation::Tan => base.iter().map(|v| v * 0.6).collect(),
        _ => base,
    }
}

fn run(device: Device, g: &Graph, inputs: &[(&str, &[f32])]) -> Vec<f32> {
    Session::new(device)
        .compile(g.clone())
        .run(inputs)
        .pop()
        .expect("no output")
}

fn worst_rel(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
        .fold(0.0, f32::max)
}

#[test]
fn activation_backward_parity() {
    let backends = available_backends();
    eprintln!("activation-backward parity over backends: {backends:?}");
    let dy: Vec<f32> = (0..N).map(|i| 0.3 + (i as f32 * 0.07).sin()).collect();
    for kind in native_backward_acts() {
        let g = bwd_graph(kind);
        let x = x_input(kind);
        let cpu = run(Device::Cpu, &g, &[("x", &x), ("dy", &dy)]);
        assert!(
            cpu.iter().all(|v| v.is_finite()),
            "CPU backward oracle for {kind:?} produced non-finite output"
        );
        for &dev in &backends {
            if !supports_graph(dev, &g) {
                eprintln!("skip {kind:?} on {dev:?} (unsupported)");
                continue;
            }
            let got = run(dev, &g, &[("x", &x), ("dy", &dy)]);
            let err = worst_rel(&cpu, &got);
            assert!(
                err < TOL,
                "activation-backward {kind:?} on {dev:?}: worst_rel={err:.3e}"
            );
        }
    }
}
