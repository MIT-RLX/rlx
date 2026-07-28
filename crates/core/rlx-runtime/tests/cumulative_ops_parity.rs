// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::{CumProd, CumMax}` forward parity. CPU carries a native O(L) scan;
//! the GPU backends without a native kernel legalize to the masked-reduce
//! decomposition oracle, so this pins both against a reference and each other.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run_cumprod(device: Device, dims: &[usize], axis: i32, excl: bool, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("cumprod");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.cumprod_(inp, axis, excl);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

fn run_cummax(device: Device, dims: &[usize], axis: i32, excl: bool, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("cummax");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.cummax_(inp, axis, excl);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

/// Reference last-axis scan (inclusive/exclusive).
fn scan_ref(dims: &[usize], excl: bool, x: &[f32], is_max: bool) -> Vec<f32> {
    let cols = *dims.last().unwrap();
    let rows: usize = x.len() / cols;
    let mut out = vec![0f32; x.len()];
    for r in 0..rows {
        let mut acc = if is_max { f32::NEG_INFINITY } else { 1.0 };
        for c in 0..cols {
            let v = x[r * cols + c];
            if excl {
                out[r * cols + c] = acc;
                acc = if is_max { acc.max(v) } else { acc * v };
            } else {
                acc = if is_max { acc.max(v) } else { acc * v };
                out[r * cols + c] = acc;
            }
        }
    }
    out
}

fn approx(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label} length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        // Both finite (the decomposition uses a finite -inf sentinel).
        if x.is_finite() && y.is_finite() {
            assert!(
                (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
                "{label}[{i}]: {x} vs {y}"
            );
        }
    }
}

#[test]
fn cumprod_matches_reference() {
    let dims = [2usize, 4];
    let x: Vec<f32> = vec![1.5, 0.5, 2.0, -1.0, 0.9, 1.1, 0.8, 1.2];
    approx(
        &run_cumprod(Device::Cpu, &dims, -1, false, &x),
        &scan_ref(&dims, false, &x, false),
        1e-5,
        "cumprod incl",
    );
    approx(
        &run_cumprod(Device::Cpu, &dims, 1, true, &x),
        &scan_ref(&dims, true, &x, false),
        1e-5,
        "cumprod excl",
    );
}

#[test]
fn cummax_matches_reference() {
    let dims = [2usize, 4];
    let x: Vec<f32> = vec![1.0, 3.0, 2.0, 5.0, -1.0, -3.0, 0.0, -2.0];
    approx(
        &run_cummax(Device::Cpu, &dims, -1, false, &x),
        &scan_ref(&dims, false, &x, true),
        1e-5,
        "cummax incl",
    );
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    feature = "gpu",
    feature = "cuda",
    feature = "mlx",
    feature = "vulkan",
    feature = "coreml"
))]
fn check_device(device: Device, label: &str) {
    let dims = [3usize, 5];
    let px: Vec<f32> = vec![
        1.2, 0.8, 1.5, 0.9, 1.1, 0.7, 1.3, 0.95, 1.05, 0.85, 1.4, 0.6, 1.25, 0.75, 1.15,
    ];
    approx(
        &run_cumprod(device, &dims, -1, false, &px),
        &run_cumprod(Device::Cpu, &dims, -1, false, &px),
        1e-4,
        &format!("{label} cumprod"),
    );
    let mx: Vec<f32> = vec![
        1.0, 3.0, 2.0, 5.0, 4.0, -1.0, -3.0, 0.0, -2.0, 6.0, 2.5, 2.5, 1.0, 9.0, 3.0,
    ];
    approx(
        &run_cummax(device, &dims, -1, false, &mx),
        &run_cummax(Device::Cpu, &dims, -1, false, &mx),
        1e-4,
        &format!("{label} cummax"),
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn cumulative_metal_matches_cpu() {
    check_device(Device::Metal, "metal");
}

#[test]
#[cfg(feature = "gpu")]
fn cumulative_wgpu_matches_cpu() {
    check_device(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn cumulative_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    check_device(Device::Cuda, "cuda");
}

#[test]
#[cfg(feature = "mlx")]
fn cumulative_mlx_matches_cpu() {
    if !rlx_runtime::is_available(Device::Mlx) {
        return;
    }
    check_device(Device::Mlx, "mlx");
}

#[test]
#[cfg(feature = "vulkan")]
fn cumulative_vulkan_matches_cpu() {
    if !rlx_runtime::is_available(Device::Vulkan) {
        return;
    }
    check_device(Device::Vulkan, "vulkan");
}

#[test]
#[cfg(feature = "coreml")]
fn cumulative_coreml_matches_cpu() {
    if !rlx_runtime::is_available(Device::Ane) {
        return;
    }
    check_device(Device::Ane, "coreml");
}
