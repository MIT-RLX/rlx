// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU vs GPU parity for higher-order reverse-mode AD (`nth_order_grad`).
//!
//! Covers CUDA, wgpu (`Device::Gpu`), Metal, and MLX when compiled in.
//!
//! ```sh
//! cargo test -p rlx-runtime --features cpu,cuda --test third_order_gpu_parity   # CUDA rig
//! cargo test -p rlx-runtime --features cpu,apple --test third_order_gpu_parity # macOS
//! ```

#![cfg(all(
    feature = "cpu",
    any(
        feature = "cuda",
        feature = "rocm",
        feature = "gpu",
        all(feature = "metal", target_os = "macos"),
        all(feature = "mlx", target_os = "macos")
    )
))]

use rlx_autodiff::nth_order_grad;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

fn f32_bytes(x: f32) -> Vec<u8> {
    x.to_le_bytes().to_vec()
}
fn f32_out(b: &[u8]) -> f32 {
    f32::from_le_bytes(b[..4].try_into().unwrap())
}

fn eval_f32(device: Device, g: Graph, inputs: &[(&str, &[u8], DType)]) -> f32 {
    f32_out(&Session::new(device).compile(g).run_typed(inputs)[0].0)
}

fn assert_matches_cpu(
    device: Device,
    g: Graph,
    inputs: &[(&str, &[u8], DType)],
    tol: f32,
    label: &str,
) {
    if !is_available(device) {
        eprintln!("skip third_order_gpu_parity {label} on {device:?} (unavailable)");
        return;
    }
    let cpu = eval_f32(Device::Cpu, g.clone(), inputs);
    let gpu = eval_f32(device, g, inputs);
    assert!(
        (cpu - gpu).abs() < tol,
        "{label} {device:?}: cpu={cpu} gpu={gpu} tol={tol}"
    );
}

fn input_x(x_val: f32) -> [(&'static str, Vec<u8>, DType); 1] {
    [("x", f32_bytes(x_val), DType::F32)]
}

fn build_x3_f32() -> Graph {
    let mut g = Graph::new("x3_gpu");
    let x = g.input("x", Shape::scalar(DType::F32));
    let x2 = g.binary(BinaryOp::Mul, x, x, Shape::scalar(DType::F32));
    let x3 = g.binary(BinaryOp::Mul, x2, x, Shape::scalar(DType::F32));
    g.set_outputs(vec![x3]);
    g
}

fn build_relu_f32() -> Graph {
    let mut g = Graph::new("relu_gpu");
    let x = g.input("x", Shape::scalar(DType::F32));
    let y = g.activation(Activation::Relu, x, Shape::scalar(DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn build_tanh_f32() -> Graph {
    let mut g = Graph::new("tanh_gpu");
    let x = g.input("x", Shape::scalar(DType::F32));
    let y = g.activation(Activation::Tanh, x, Shape::scalar(DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn build_gelu_f32() -> Graph {
    let mut g = Graph::new("gelu_gpu");
    let x = g.input("x", Shape::scalar(DType::F32));
    let y = g.activation(Activation::Gelu, x, Shape::scalar(DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn build_silu_f32() -> Graph {
    let mut g = Graph::new("silu_gpu");
    let x = g.input("x", Shape::scalar(DType::F32));
    let y = g.activation(Activation::Silu, x, Shape::scalar(DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn third_order_x_cubed(device: Device) {
    let forward = build_x3_f32();
    let hg = nth_order_grad(&forward, "x", 3);
    let ins = input_x(1.5);
    let inputs = [("x", ins[0].1.as_slice(), DType::F32)];
    assert_matches_cpu(device, hg, &inputs, 1e-3, "x^3 third deriv");
    let cpu = eval_f32(Device::Cpu, nth_order_grad(&forward, "x", 3), &inputs);
    assert!((cpu - 6.0).abs() < 1e-2, "x^3 third deriv reference: {cpu}");
}

fn third_order_relu(device: Device) {
    let forward = build_relu_f32();
    let hg = nth_order_grad(&forward, "x", 3);
    let ins = input_x(1.0);
    let inputs = [("x", ins[0].1.as_slice(), DType::F32)];
    assert_matches_cpu(device, hg, &inputs, 1e-4, "relu third deriv");
    let cpu = eval_f32(Device::Cpu, nth_order_grad(&forward, "x", 3), &inputs);
    assert!(cpu.abs() < 1e-3, "relu third deriv should be ~0, got {cpu}");
}

fn third_order_tanh(device: Device) {
    let forward = build_tanh_f32();
    let hg = nth_order_grad(&forward, "x", 3);
    let x_val = 0.5f32;
    let ins = input_x(x_val);
    let inputs = [("x", ins[0].1.as_slice(), DType::F32)];
    assert_matches_cpu(device, hg, &inputs, 1e-3, "tanh third deriv");
    let tx = x_val.tanh();
    let sech2 = (1.0_f32 / x_val.cosh()).powi(2);
    let want = -2.0 * sech2 * (1.0 - 3.0 * tx * tx);
    let cpu = eval_f32(Device::Cpu, nth_order_grad(&forward, "x", 3), &inputs);
    assert!(
        (cpu - want).abs() < 1e-2,
        "tanh third deriv ref: cpu={cpu} want={want}"
    );
}

fn third_order_gelu(device: Device) {
    let forward = build_gelu_f32();
    let hg = nth_order_grad(&forward, "x", 3);
    let x_val = 0.75f32;
    let ins = input_x(x_val);
    let inputs = [("x", ins[0].1.as_slice(), DType::F32)];
    assert_matches_cpu(device, hg, &inputs, 1e-3, "gelu third deriv");
    let cpu = eval_f32(Device::Cpu, nth_order_grad(&forward, "x", 3), &inputs);
    assert!(
        cpu.is_finite(),
        "gelu third deriv should be finite, got {cpu}"
    );
}

fn third_order_silu(device: Device) {
    let forward = build_silu_f32();
    let hg = nth_order_grad(&forward, "x", 3);
    let x_val = 0.5f32;
    let ins = input_x(x_val);
    let inputs = [("x", ins[0].1.as_slice(), DType::F32)];
    assert_matches_cpu(device, hg, &inputs, 1e-3, "silu third deriv");
    let cpu = eval_f32(Device::Cpu, nth_order_grad(&forward, "x", 3), &inputs);
    assert!(
        cpu.is_finite(),
        "silu third deriv should be finite, got {cpu}"
    );
}

macro_rules! third_order_parity_suite {
    ($mod_name:ident, $device:expr, $($cfg:meta),+) => {
        $(#[$cfg])*
        mod $mod_name {
            use super::*;
            #[test]
            fn x_cubed_third_derivative() {
                third_order_x_cubed($device);
            }
            #[test]
            fn relu_third_derivative() {
                third_order_relu($device);
            }
            #[test]
            fn tanh_third_derivative() {
                third_order_tanh($device);
            }
            #[test]
            fn gelu_third_derivative() {
                third_order_gelu($device);
            }
            #[test]
            fn silu_third_derivative() {
                third_order_silu($device);
            }
        }
    };
}

third_order_parity_suite!(cuda, Device::Cuda, cfg(feature = "cuda"));
third_order_parity_suite!(rocm, Device::Rocm, cfg(feature = "rocm"));
third_order_parity_suite!(wgpu, Device::Gpu, cfg(feature = "gpu"));
third_order_parity_suite!(
    metal,
    Device::Metal,
    cfg(all(feature = "metal", target_os = "macos"))
);
third_order_parity_suite!(
    mlx,
    Device::Mlx,
    cfg(all(feature = "mlx", target_os = "macos"))
);
