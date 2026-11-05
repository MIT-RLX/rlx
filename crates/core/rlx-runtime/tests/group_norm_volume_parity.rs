// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::GroupNorm` over NCDHW volumes against an f64 reference.
//!
//! Two failures this guards, both found porting MONAI 3-D segmentation nets:
//! - rank 5: the CPU thunk panicked (`conv_nchw_dims: expected rank 3 or 4`)
//!   and wgpu normalized over dims 2 and 3 only — wrong values, no error;
//! - precision: an instance norm spans a whole volume. A running f32 sum on
//!   CPU and a single-pass `E[x^2] - mean^2` on Metal gave relative errors of
//!   1.7e-4 / 2e-3 (CPU, offset 0 / 50) and 7e-4 (Metal, offset 50) over one
//!   96^3 group.

mod common;

use common::skip_unless_available;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 * 2.0
        })
        .collect()
}

/// f64 group norm of `x` `[n, c, spatial]` with per-channel affine.
fn reference(
    x: &[f32],
    g: &[f32],
    b: &[f32],
    n: usize,
    c: usize,
    groups: usize,
    eps: f64,
) -> Vec<f64> {
    let per = x.len() / (n * c);
    let cpg = c / groups;
    let mut out = vec![0.0; x.len()];
    for bi in 0..n {
        for gi in 0..groups {
            let lo = (bi * c + gi * cpg) * per;
            let hi = lo + cpg * per;
            let m = x[lo..hi].iter().map(|&v| v as f64).sum::<f64>() / (hi - lo) as f64;
            let var = x[lo..hi]
                .iter()
                .map(|&v| (v as f64 - m).powi(2))
                .sum::<f64>()
                / (hi - lo) as f64;
            let inv = 1.0 / (var + eps).sqrt();
            for i in lo..hi {
                let ch = i / per % c;
                out[i] = (x[i] as f64 - m) * inv * g[ch] as f64 + b[ch] as f64;
            }
        }
    }
    out
}

fn check(device: Device, dims: [usize; 5], groups: usize, offset: f32, rel_tol: f64) {
    if skip_unless_available(device, "group_norm_volume") {
        return;
    }
    let [n, c, ..] = dims;
    let mut g = Graph::new("gn5");
    let x = g.input("x", Shape::new(&dims, DType::F32));
    let gamma = g.param("g", Shape::new(&[c], DType::F32));
    let beta = g.param("b", Shape::new(&[c], DType::F32));
    let y = g.group_norm(x, gamma, beta, groups, 1e-5);
    g.set_outputs(vec![y]);
    let xs: Vec<f32> = noise(dims.iter().product(), 1)
        .iter()
        .map(|v| v + offset)
        .collect();
    let gs: Vec<f32> = noise(c, 2).iter().map(|v| 1.0 + v).collect();
    let bs = noise(c, 3);
    let mut compiled = Session::new(device).compile(g);
    compiled.set_param("g", &gs);
    compiled.set_param("b", &bs);
    let got = compiled.run(&[("x", &xs)]).remove(0);
    let want = reference(&xs, &gs, &bs, n, c, groups, 1e-5);
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (*a as f64 - b).abs())
        .fold(0f64, f64::max)
        / scale;
    eprintln!("{device:?} {dims:?} groups {groups} offset {offset}: rel err {err:.2e}");
    assert!(
        err <= rel_tol,
        "{device:?} {dims:?} offset {offset}: rel err {err:.2e}"
    );
}

/// Small rank-5 shapes, several groups, batch 2: layout coverage.
fn layout(device: Device) {
    check(device, [2, 8, 4, 6, 5], 4, 0.0, 1e-5);
    check(device, [1, 6, 3, 4, 7], 6, 0.0, 1e-5);
}

/// One 96^3 instance-norm group with a DC offset: precision.
fn volume(device: Device) {
    check(device, [1, 2, 96, 96, 96], 2, 0.0, 2e-6);
    check(device, [1, 2, 96, 96, 96], 2, 50.0, 1e-4);
}

#[test]
fn cpu_rank5_layout() {
    layout(Device::Cpu);
}

#[test]
fn cpu_rank5_volume_precision() {
    volume(Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn metal_rank5_layout() {
    layout(Device::Metal);
}

#[cfg(feature = "metal")]
#[test]
fn metal_rank5_volume_precision() {
    volume(Device::Metal);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_rank5_layout() {
    layout(Device::Gpu);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_rank5_volume_precision() {
    volume(Device::Gpu);
}
