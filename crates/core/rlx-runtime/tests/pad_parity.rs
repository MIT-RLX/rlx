// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Pad` forward parity. The `LowerPad` decomposition
//! (`full`/`narrow`/`reverse`/`expand`/`concat`) is the semantic oracle for
//! every backend except Metal/CUDA; this pins it to the NumPy `pad` /
//! PyTorch `F.pad` convention for all four `PadMode`s.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, PadMode, Shape};
use rlx_runtime::{Device, Session};

fn run(
    device: Device,
    dims: &[usize],
    pads: Vec<[usize; 2]>,
    mode: PadMode,
    x: &[f32],
) -> Vec<f32> {
    let mut g = Graph::new("pad");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.pad_(inp, pads, mode);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

/// Map an output coordinate `j` along an axis of input length `n` back to the
/// source index (or `None` = constant fill) for `mode`, `before` pad width.
fn src_index(j: usize, before: usize, n: usize, mode: PadMode) -> Option<usize> {
    let p = j as isize - before as isize;
    match mode {
        PadMode::Constant(_) => {
            if p < 0 || p >= n as isize {
                None
            } else {
                Some(p as usize)
            }
        }
        PadMode::Replicate => Some(p.clamp(0, n as isize - 1) as usize),
        PadMode::Circular => {
            let nn = n as isize;
            Some((((p % nn) + nn) % nn) as usize)
        }
        PadMode::Reflect => {
            let period = 2 * (n as isize - 1);
            let mut i = ((p % period) + period) % period;
            if i >= n as isize {
                i = period - i;
            }
            Some(i as usize)
        }
    }
}

/// Pad a single `axis` of a row-major tensor (composed to get N-D, matching the
/// per-axis-sequential decomposition and NumPy/PyTorch corner behavior).
fn pad_axis(
    data: &[f32],
    dims: &[usize],
    axis: usize,
    before: usize,
    after: usize,
    mode: PadMode,
) -> (Vec<f32>, Vec<usize>) {
    let n = dims[axis];
    let mut out_dims = dims.to_vec();
    out_dims[axis] = n + before + after;
    let rank = dims.len();
    let stride = |d: &[usize]| {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * d[i + 1];
        }
        s
    };
    let in_stride = stride(dims);
    let out_total: usize = out_dims.iter().product();
    let fill = if let PadMode::Constant(v) = mode {
        v
    } else {
        0.0
    };
    let mut out = vec![0f32; out_total];
    let out_stride = stride(&out_dims);
    for o in 0..out_total {
        let mut rem = o;
        let mut coord = vec![0usize; rank];
        for ax in 0..rank {
            coord[ax] = rem / out_stride[ax];
            rem %= out_stride[ax];
        }
        match src_index(coord[axis], before, n, mode) {
            None => out[o] = fill,
            Some(si) => {
                let mut inp = 0usize;
                for ax in 0..rank {
                    let c = if ax == axis { si } else { coord[ax] };
                    inp += c * in_stride[ax];
                }
                out[o] = data[inp];
            }
        }
    }
    (out, out_dims)
}

fn reference(dims: &[usize], pads: &[[usize; 2]], mode: PadMode, x: &[f32]) -> Vec<f32> {
    let mut data = x.to_vec();
    let mut cur = dims.to_vec();
    for (axis, &[b, a]) in pads.iter().enumerate() {
        if b == 0 && a == 0 {
            continue;
        }
        let (nd, ndims) = pad_axis(&data, &cur, axis, b, a, mode);
        data = nd;
        cur = ndims;
    }
    data
}

// ── 1-D: pin each mode to its NumPy vector on [1,2,3,4] pad (2,2). ──
#[test]
fn pad_1d_matches_numpy() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let cases: &[(PadMode, [f32; 8])] = &[
        (PadMode::Constant(0.0), [0., 0., 1., 2., 3., 4., 0., 0.]),
        (PadMode::Reflect, [3., 2., 1., 2., 3., 4., 3., 2.]),
        (PadMode::Replicate, [1., 1., 1., 2., 3., 4., 4., 4.]),
        (PadMode::Circular, [3., 4., 1., 2., 3., 4., 1., 2.]),
    ];
    for (mode, want) in cases {
        let got = run(Device::Cpu, &[4], vec![[2, 2]], *mode, &x);
        assert_eq!(got.as_slice(), want.as_slice(), "1-D {mode:?}");
        // reference() must agree with the pinned NumPy vector too.
        assert_eq!(
            reference(&[4], &[[2, 2]], *mode, &x).as_slice(),
            want.as_slice(),
            "ref {mode:?}"
        );
    }
}

// ── Asymmetric constant pad with a non-zero fill. ──
#[test]
fn pad_1d_asymmetric_constant() {
    let x = [5.0, 6.0, 7.0];
    let got = run(Device::Cpu, &[3], vec![[1, 2]], PadMode::Constant(-1.0), &x);
    assert_eq!(got, vec![-1.0, 5.0, 6.0, 7.0, -1.0, -1.0]);
}

// ── 2-D: all four modes vs the sequential reference (corners included). ──
#[test]
fn pad_2d_all_modes() {
    let dims = [3usize, 4];
    let x: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    for mode in [
        PadMode::Constant(9.0),
        PadMode::Reflect,
        PadMode::Replicate,
        PadMode::Circular,
    ] {
        let pads = vec![[1, 2], [2, 1]];
        let got = run(Device::Cpu, &dims, pads.clone(), mode, &x);
        let want = reference(&dims, &pads, mode, &x);
        assert_eq!(got, want, "2-D {mode:?}");
    }
}

#[allow(unused)]
fn all_modes() -> [PadMode; 4] {
    [
        PadMode::Constant(9.0),
        PadMode::Reflect,
        PadMode::Replicate,
        PadMode::Circular,
    ]
}

/// (dims, pads) configs — all valid for `Reflect` (every pad < its axis length).
/// Exercises 1D, multi-axis interaction, a zero-before axis, a fully-padded 2D,
/// and a size-1 axis left unpadded (the kernel's identity fast path).
#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    feature = "gpu",
    feature = "cuda"
))]
fn cases() -> Vec<(Vec<usize>, Vec<[usize; 2]>)> {
    vec![
        (vec![5], vec![[2, 2]]),
        (vec![2, 3, 4], vec![[1, 1], [1, 2], [0, 3]]),
        (vec![3, 3], vec![[2, 1], [1, 2]]),
        (vec![4, 1, 3], vec![[1, 2], [0, 0], [2, 1]]),
    ]
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    feature = "gpu",
    feature = "cuda"
))]
fn check_device_matches_cpu(device: Device, label: &str) {
    for (dims, pads) in cases() {
        let n: usize = dims.iter().product();
        let x: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
        for mode in all_modes() {
            assert_eq!(
                run(device, &dims, pads.clone(), mode, &x),
                run(Device::Cpu, &dims, pads.clone(), mode, &x),
                "{label} pad {mode:?} dims={dims:?} pads={pads:?}"
            );
        }
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn pad_metal_matches_cpu() {
    check_device_matches_cpu(Device::Metal, "metal");
}

#[test]
#[cfg(feature = "gpu")]
fn pad_wgpu_matches_cpu() {
    check_device_matches_cpu(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn pad_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    check_device_matches_cpu(Device::Cuda, "cuda");
}

// ── Padding only a subset of axes (zero pads must be no-ops). ──
#[test]
fn pad_2d_single_axis() {
    let dims = [2usize, 3];
    let x: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let pads = vec![[0, 0], [1, 1]];
    let got = run(Device::Cpu, &dims, pads.clone(), PadMode::Replicate, &x);
    let want = reference(&dims, &pads, PadMode::Replicate, &x);
    assert_eq!(got, want);
    // row 0: [1,1,2,3,3]; row 1: [4,4,5,6,6]
    assert_eq!(got, vec![1., 1., 2., 3., 3., 4., 4., 5., 6., 6.]);
}
