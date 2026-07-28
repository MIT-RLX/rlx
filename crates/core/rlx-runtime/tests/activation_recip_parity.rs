// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reciprocal / exp / tanh activation parity vs CPU on available GPUs.
//! Covers the vmath-aligned unaries (`vvrecf` / `vvexpf` / `vvtanhf`).

use rlx_ir::op::Activation;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session, is_available};

#[allow(unused)]
fn graph(act: Activation) -> Graph {
    let mut graph = Graph::new("vmath_act_parity");
    let shape = Shape::new(&[8], DType::F32);
    let x = graph.input("x", shape.clone());
    let y = graph.activation(act, x, shape);
    graph.set_outputs(vec![y]);
    graph
}

#[allow(unused)]
fn backward_graph(act: Activation) -> Graph {
    let mut graph = Graph::new("vmath_act_backward_parity");
    let shape = Shape::new(&[8], DType::F32);
    let x = graph.input("x", shape.clone());
    let dy = graph.input("dy", shape);
    let dx = graph.activation_backward(act, x, dy);
    graph.set_outputs(vec![dx]);
    graph
}

#[allow(unused)]
fn run(device: Device, act: Activation, x: &[f32]) -> Vec<f32> {
    Session::new(device).compile(graph(act)).run(&[("x", x)])[0].clone()
}

fn run_backward(device: Device, act: Activation, x: &[f32], dy: &[f32]) -> Vec<f32> {
    Session::new(device)
        .compile(backward_graph(act))
        .run(&[("x", x), ("dy", dy)])[0]
        .clone()
}

#[allow(unused)]
fn check(device: Device, act: Activation, x: &[f32], tol: f32) {
    if !is_available(device) {
        return;
    }
    let expected = run(Device::Cpu, act, x);
    let actual = run(device, act, x);
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{device:?} {act:?}[{i}]: got {got}, expected {want} (tol={tol})"
        );
    }
}

#[allow(unused)]
fn check_backward(device: Device, act: Activation, x: &[f32], dy: &[f32], tol: f32) {
    if !is_available(device) {
        return;
    }
    let expected = run_backward(Device::Cpu, act, x, dy);
    let actual = run_backward(device, act, x, dy);
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tol,
            "{device:?} {act:?} backward[{i}]: got {got}, expected {want} (tol={tol})"
        );
    }
}

#[allow(unused)]
fn recip_x() -> [f32; 8] {
    [-4.0, -2.0, -0.5, -0.25, 0.25, 0.5, 2.0, 4.0]
}

#[allow(unused)]
fn exp_tanh_x() -> [f32; 8] {
    [-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0]
}

#[allow(unused)]
fn backward_dy() -> [f32; 8] {
    [-1.0, -0.75, -0.25, 0.1, 0.3, 0.5, 0.8, 1.0]
}

#[test]
#[cfg(feature = "metal")]
fn reciprocal_matches_cpu_on_metal() {
    check(Device::Metal, Activation::Recip, &recip_x(), 1e-5);
}

#[test]
#[cfg(feature = "gpu")]
fn reciprocal_matches_cpu_on_wgpu() {
    check(Device::Gpu, Activation::Recip, &recip_x(), 1e-5);
}

#[test]
#[cfg(feature = "cuda")]
fn reciprocal_matches_cpu_on_cuda() {
    check(Device::Cuda, Activation::Recip, &recip_x(), 1e-5);
}

#[test]
#[cfg(feature = "cuda")]
fn exp_tanh_match_cpu_on_cuda() {
    let x = exp_tanh_x();
    check(Device::Cuda, Activation::Exp, &x, 1e-5);
    check(Device::Cuda, Activation::Tanh, &x, 1e-5);
}

#[test]
#[cfg(feature = "metal")]
fn activation_backward_matches_cpu_on_metal() {
    let dy = backward_dy();
    check_backward(Device::Metal, Activation::Recip, &recip_x(), &dy, 1e-5);
    check_backward(Device::Metal, Activation::Exp, &exp_tanh_x(), &dy, 1e-5);
    check_backward(Device::Metal, Activation::Tanh, &exp_tanh_x(), &dy, 2e-5);
}

#[test]
#[cfg(feature = "gpu")]
fn activation_backward_matches_cpu_on_wgpu() {
    let dy = backward_dy();
    check_backward(Device::Gpu, Activation::Recip, &recip_x(), &dy, 1e-5);
    check_backward(Device::Gpu, Activation::Exp, &exp_tanh_x(), &dy, 1e-5);
    check_backward(Device::Gpu, Activation::Tanh, &exp_tanh_x(), &dy, 2e-5);
}

#[test]
#[cfg(feature = "cuda")]
fn activation_backward_matches_cpu_on_cuda() {
    let dy = backward_dy();
    check_backward(Device::Cuda, Activation::Recip, &recip_x(), &dy, 1e-5);
    check_backward(Device::Cuda, Activation::Exp, &exp_tanh_x(), &dy, 1e-5);
    check_backward(Device::Cuda, Activation::Tanh, &exp_tanh_x(), &dy, 2e-5);
}
