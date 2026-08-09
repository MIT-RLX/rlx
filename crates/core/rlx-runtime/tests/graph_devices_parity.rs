// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CPU vs GPU parity through [`GraphDevices`] (multi-backend runner API).

#![cfg(feature = "cpu")]
#![allow(dead_code)]

use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, GraphDevices, is_available};

fn matmul_graph() -> Graph {
    let mut g = Graph::new("gd_mm");
    let x = g.input("x", Shape::new(&[2, 4], DType::F32));
    let w = g.param("w", Shape::new(&[4, 3], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
    g.set_outputs(vec![y]);
    g
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label} len mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < tol, "{label}[{i}]: cpu/gpu {x} vs {y}");
    }
}

fn parity_on(device: Device, tol: f32, label: &str) {
    if !is_available(device) {
        eprintln!("skip graph_devices_parity {label} on {device:?} (unavailable)");
        return;
    }
    let g = matmul_graph();
    let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let x: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.5).collect();
    let mut runner = GraphDevices::new(g);
    runner.set_param("w", &w);
    let cpu = runner.run(Device::Cpu, &[("x", &x)]).expect("cpu run");
    let gpu = runner.run(device, &[("x", &x)]).expect("gpu run");
    assert_close(&cpu[0], &gpu[0], tol, label);
}

#[test]
fn graph_devices_cpu_identity() {
    let mut g = Graph::new("id");
    let x = g.input("x", Shape::new(&[4], DType::F32));
    g.set_outputs(vec![x]);
    let mut runner = GraphDevices::new(g);
    let out = runner
        .run(Device::Cpu, &[("x", &[1.0, 2.0, 3.0, 4.0])])
        .unwrap();
    assert_eq!(out[0], vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn graph_devices_matmul_metal_parity() {
    parity_on(Device::Metal, 1e-3, "metal");
}

#[test]
#[cfg(all(feature = "mlx", rlx_mlx_host))]
fn graph_devices_matmul_mlx_parity() {
    parity_on(Device::Mlx, 1e-3, "mlx");
}

#[test]
#[cfg(feature = "gpu")]
fn graph_devices_matmul_wgpu_parity() {
    parity_on(Device::Gpu, 1e-2, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn graph_devices_matmul_cuda_parity() {
    parity_on(Device::Cuda, 1e-2, "cuda");
}

#[test]
#[cfg(feature = "rocm")]
fn graph_devices_matmul_rocm_parity() {
    parity_on(Device::Rocm, 1e-2, "rocm");
}

#[test]
fn graph_devices_run_try_falls_back_to_cpu() {
    let mut g = Graph::new("id");
    let x = g.input("x", Shape::new(&[2], DType::F32));
    g.set_outputs(vec![x]);
    let mut runner = GraphDevices::new(g);
    // Lead with a device THIS BUILD genuinely cannot run, so the fallback path
    // is exercised on every host. Hard-coding `Device::Cuda` only passed on
    // machines without CUDA — on the CUDA rig `run_try` correctly selected Cuda
    // and the unconditional `== Cpu` assertion failed.
    let chain: Vec<Device> = match [Device::Cuda, Device::Rocm, Device::Vulkan, Device::Metal]
        .into_iter()
        .find(|d| !is_available(*d))
    {
        Some(unavailable) => vec![unavailable, Device::Cpu],
        None => vec![Device::Cpu],
    };
    let (dev, out) = runner
        .run_try(&chain, &[("x", &[3.0, 4.0])])
        .expect("fallback");
    assert_eq!(dev, Device::Cpu);
    assert_eq!(out[0], vec![3.0, 4.0]);
}

#[test]
fn graph_devices_add_param_sync() {
    let mut g = Graph::new("add");
    let x = g.input("x", Shape::new(&[2], DType::F32));
    let b = g.param("b", Shape::new(&[2], DType::F32));
    let y = g.binary(BinaryOp::Add, x, b, Shape::new(&[2], DType::F32));
    g.set_outputs(vec![y]);
    let mut runner = GraphDevices::new(g);
    runner.set_param("b", &[1.0, 2.0]);
    let out = runner.run(Device::Cpu, &[("x", &[3.0, 4.0])]).unwrap();
    assert_eq!(out[0], vec![4.0, 6.0]);
}
