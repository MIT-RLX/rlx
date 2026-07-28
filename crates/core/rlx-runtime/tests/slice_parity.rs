// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Slice` (strided slice) forward parity. `LowerSlice` (narrow / reverse /
//! gather) is the oracle for every backend except Metal/CUDA; pinned to NumPy
//! `x[start:*:step]` semantics.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run(
    device: Device,
    dims: &[usize],
    axis: usize,
    start: usize,
    len: usize,
    step: i64,
    x: &[f32],
) -> Vec<f32> {
    let mut g = Graph::new("slice");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.slice_(inp, axis, start, len, step);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

/// out[..,j,..] = in[.., start + j*step, ..] along `axis`.
fn reference(
    dims: &[usize],
    axis: usize,
    start: usize,
    len: usize,
    step: i64,
    x: &[f32],
) -> Vec<f32> {
    let rank = dims.len();
    let mut out_dims = dims.to_vec();
    out_dims[axis] = len;
    let stride = |d: &[usize]| {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * d[i + 1];
        }
        s
    };
    let in_stride = stride(dims);
    let out_stride = stride(&out_dims);
    let total: usize = out_dims.iter().product();
    let mut out = vec![0f32; total];
    for o in 0..total {
        let mut rem = o;
        let mut inflat = 0usize;
        for ax in 0..rank {
            let oc = rem / out_stride[ax];
            rem %= out_stride[ax];
            let ic = if ax == axis {
                (start as i64 + oc as i64 * step) as usize
            } else {
                oc
            };
            inflat += ic * in_stride[ax];
        }
        out[o] = x[inflat];
    }
    out
}

// (dims, axis, start, len, step) — valid indices for every access.
fn cases() -> Vec<(Vec<usize>, usize, usize, usize, i64)> {
    vec![
        (vec![6], 0, 2, 3, 1),        // narrow  → [2,3,4]
        (vec![6], 0, 0, 3, 2),        // x[::2]  → [0,2,4]
        (vec![6], 0, 1, 3, 2),        // x[1::2] → [1,3,5]
        (vec![6], 0, 5, 6, -1),       // x[::-1] → [5,4,3,2,1,0]
        (vec![6], 0, 5, 3, -2),       // x[5::-2]→ [5,3,1]
        (vec![3, 6], 1, 0, 3, 2),     // 2-D strided cols
        (vec![3, 6], 1, 5, 3, -2),    // 2-D reverse-strided cols
        (vec![2, 3, 4], 2, 3, 2, -1), // last-axis reverse window
    ]
}

#[test]
fn slice_1d_matches_numpy() {
    let x: Vec<f32> = (0..6).map(|i| i as f32).collect();
    assert_eq!(run(Device::Cpu, &[6], 0, 0, 3, 2, &x), vec![0., 2., 4.]);
    assert_eq!(run(Device::Cpu, &[6], 0, 1, 3, 2, &x), vec![1., 3., 5.]);
    assert_eq!(
        run(Device::Cpu, &[6], 0, 5, 6, -1, &x),
        vec![5., 4., 3., 2., 1., 0.]
    );
    assert_eq!(run(Device::Cpu, &[6], 0, 5, 3, -2, &x), vec![5., 3., 1.]);
    assert_eq!(run(Device::Cpu, &[6], 0, 2, 3, 1, &x), vec![2., 3., 4.]);
}

#[test]
fn slice_cpu_all_cases() {
    for (dims, axis, start, len, step) in cases() {
        let n: usize = dims.iter().product();
        let x: Vec<f32> = (0..n).map(|i| (i % 9) as f32 * 0.5 - 2.0).collect();
        assert_eq!(
            run(Device::Cpu, &dims, axis, start, len, step, &x),
            reference(&dims, axis, start, len, step, &x),
            "cpu slice dims={dims:?} axis={axis} start={start} len={len} step={step}"
        );
    }
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    feature = "gpu",
    feature = "cuda"
))]
fn check_device(device: Device, label: &str) {
    for (dims, axis, start, len, step) in cases() {
        let n: usize = dims.iter().product();
        let x: Vec<f32> = (0..n).map(|i| (i % 9) as f32 * 0.5 - 2.0).collect();
        assert_eq!(
            run(device, &dims, axis, start, len, step, &x),
            run(Device::Cpu, &dims, axis, start, len, step, &x),
            "{label} slice dims={dims:?} axis={axis} start={start} len={len} step={step}"
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn slice_metal_matches_cpu() {
    check_device(Device::Metal, "metal");
}

#[test]
#[cfg(feature = "gpu")]
fn slice_wgpu_matches_cpu() {
    check_device(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn slice_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    check_device(Device::Cuda, "cuda");
}
