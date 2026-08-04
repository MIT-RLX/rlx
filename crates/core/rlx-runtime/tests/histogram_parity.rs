// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Histogram` forward parity. CPU carries a native O(n) bucketize
//! (`Thunk::Histogram`); every other backend legalizes to the
//! Compare+mul+Reduce::Sum+Concat decomposition oracle (`LowerHistogram`).
//! This pins the native kernel against a reference, the decomposition against
//! the native kernel (hardware-free, by running the lowered graph on CPU), and
//! any available GPU backend against CPU.

#![cfg(feature = "cpu")]

use rlx_fusion::LowerHistogram;
use rlx_fusion::pass::Pass;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn hist_graph(dims: &[usize], bins: usize, min: f32, max: f32) -> Graph {
    let mut g = Graph::new("histogram");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.histogram(inp, bins, min, max);
    g.set_outputs(vec![y]);
    g
}

/// Native path: compile the histogram graph directly (CPU keeps it native;
/// GPU backends legalize/decompose it internally).
fn run_native(
    device: Device,
    dims: &[usize],
    bins: usize,
    min: f32,
    max: f32,
    x: &[f32],
) -> Vec<f32> {
    Session::new(device)
        .compile(hist_graph(dims, bins, min, max))
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

/// Decomposition oracle: apply `LowerHistogram` up front, then run the
/// primitive-only graph on CPU. Exercises the exact path GPU backends take.
fn run_decomposed(dims: &[usize], bins: usize, min: f32, max: f32, x: &[f32]) -> Vec<f32> {
    let lowered = LowerHistogram.run(hist_graph(dims, bins, min, max));
    Session::new(Device::Cpu)
        .compile(lowered)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

/// Reference — mirrors the CPU kernel exactly (half-open buckets, closed top,
/// out-of-range dropped, `x == max` in the last bin).
fn hist_ref(bins: usize, min: f32, max: f32, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; bins];
    // `max > min` is also false when a bound is NaN, so this bails then too.
    let range_valid = max > min;
    if bins == 0 || !range_valid {
        return out;
    }
    let inv = bins as f32 / (max - min);
    for &v in x {
        if v.is_nan() || v < min || v > max {
            continue;
        }
        let mut idx = ((v - min) * inv) as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        out[idx] += 1.0;
    }
    out
}

/// Low-discrepancy sequence in `[min, max)` — golden-ratio offsets so values
/// never fall on an exact bin edge (where floor vs edge-compare could disagree).
fn spread(n: usize, min: f32, max: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let frac = (i as f32 * 0.618_034).fract();
            min + frac * (max - min)
        })
        .collect()
}

#[test]
fn histogram_matches_reference() {
    // 4 bins over [0,4], width 1. Hand-checkable:
    //   0.5→b0 | 1.5,1.9→b1 | 2.0→b2 | 3.99,4.0→b3 | -1.0,5.0 dropped.
    let x = vec![0.5, 1.5, 1.9, 2.0, 3.99, 4.0, -1.0, 5.0];
    let got = run_native(Device::Cpu, &[8], 4, 0.0, 4.0, &x);
    assert_eq!(got, vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(got, hist_ref(4, 0.0, 4.0, &x));
    // Total counted == in-range element count.
    assert_eq!(got.iter().sum::<f32>(), 6.0);
}

#[test]
fn histogram_native_matches_reference_multidim() {
    // ND input is flattened before bucketize; shape of counts is [bins].
    let x = spread(120, -2.0, 3.0);
    let got = run_native(Device::Cpu, &[3, 5, 8], 16, -2.0, 3.0, &x);
    assert_eq!(got, hist_ref(16, -2.0, 3.0, &x));
    assert_eq!(got.iter().sum::<f32>(), 120.0); // all in range
}

#[test]
fn histogram_decompose_matches_native() {
    for (bins, min, max) in [(4usize, 0.0f32, 4.0f32), (16, -2.0, 3.0), (7, -1.0, 1.0)] {
        let x = spread(500, min, max);
        let native = run_native(Device::Cpu, &[500], bins, min, max, &x);
        let decomp = run_decomposed(&[500], bins, min, max, &x);
        assert_eq!(
            native,
            hist_ref(bins, min, max, &x),
            "native vs ref (bins={bins})"
        );
        assert_eq!(decomp, native, "decompose vs native (bins={bins})");
    }
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    feature = "gpu",
    feature = "cuda",
    feature = "mlx",
    feature = "vulkan"
))]
fn check_device(device: Device, label: &str) {
    let (bins, min, max) = (16usize, -2.0f32, 3.0f32);
    let x = spread(500, min, max);
    let dev = run_native(device, &[500], bins, min, max, &x);
    let cpu = run_native(Device::Cpu, &[500], bins, min, max, &x);
    assert_eq!(dev.len(), cpu.len(), "{label} length");
    for (i, (a, b)) in dev.iter().zip(&cpu).enumerate() {
        assert!((a - b).abs() < 0.5, "{label}[{i}]: {a} vs {b}");
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn histogram_metal_matches_cpu() {
    check_device(Device::Metal, "metal");
}

#[test]
#[cfg(feature = "gpu")]
fn histogram_wgpu_matches_cpu() {
    check_device(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn histogram_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    check_device(Device::Cuda, "cuda");
}

#[test]
#[cfg(feature = "mlx")]
fn histogram_mlx_matches_cpu() {
    if !rlx_runtime::is_available(Device::Mlx) {
        return;
    }
    check_device(Device::Mlx, "mlx");
}

#[test]
#[cfg(feature = "vulkan")]
fn histogram_vulkan_matches_cpu() {
    if !rlx_runtime::is_available(Device::Vulkan) {
        return;
    }
    check_device(Device::Vulkan, "vulkan");
}
