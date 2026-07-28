// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::{Clamp, Tile, Trilu}` forward parity. All three decompose (max/min,
//! concat, mul-by-mask) so this pins the decomposition oracle to reference
//! semantics and checks it on the GPU backends too.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run_clamp(device: Device, dims: &[usize], min: f32, max: f32, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("clamp");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.clamp_(inp, min, max);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

fn run_tile(device: Device, dims: &[usize], reps: Vec<usize>, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("tile");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.tile_(inp, reps);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

fn run_trilu(device: Device, dims: &[usize], upper: bool, diag: i64, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("trilu");
    let inp = g.input("x", Shape::new(dims, DType::F32));
    let y = g.trilu_(inp, upper, diag);
    g.set_outputs(vec![y]);
    Session::new(device)
        .compile(g)
        .run(&[("x", x)])
        .pop()
        .unwrap()
}

fn tile_ref(dims: &[usize], reps: &[usize], x: &[f32]) -> Vec<f32> {
    let rank = dims.len();
    let out_dims: Vec<usize> = (0..rank).map(|i| dims[i] * reps[i]).collect();
    let stride = |d: &[usize]| {
        let mut s = vec![1usize; rank];
        for i in (0..rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * d[i + 1];
        }
        s
    };
    let ins = stride(dims);
    let outs = stride(&out_dims);
    let total: usize = out_dims.iter().product();
    let mut out = vec![0f32; total];
    for o in 0..total {
        let mut rem = o;
        let mut inflat = 0usize;
        for ax in 0..rank {
            let c = rem / outs[ax];
            rem %= outs[ax];
            inflat += (c % dims[ax]) * ins[ax];
        }
        out[o] = x[inflat];
    }
    out
}

fn trilu_ref(dims: &[usize], upper: bool, diag: i64, x: &[f32]) -> Vec<f32> {
    let rank = dims.len();
    let (rows, cols) = (dims[rank - 2], dims[rank - 1]);
    let mut out = x.to_vec();
    let planes: usize = dims[..rank - 2].iter().product::<usize>().max(1);
    for p in 0..planes {
        for r in 0..rows {
            for c in 0..cols {
                let keep = if upper {
                    (c as i64 - r as i64) >= diag
                } else {
                    (c as i64 - r as i64) <= diag
                };
                if !keep {
                    out[p * rows * cols + r * cols + c] = 0.0;
                }
            }
        }
    }
    out
}

#[test]
fn clamp_matches_reference() {
    let x: Vec<f32> = vec![-3.0, -0.5, 0.0, 1.2, 4.0, 2.0, -1.0, 3.5];
    let got = run_clamp(Device::Cpu, &[8], -1.0, 2.5, &x);
    let want: Vec<f32> = x.iter().map(|v| v.clamp(-1.0, 2.5)).collect();
    assert_eq!(got, want);
}

#[test]
fn tile_matches_reference() {
    let dims = [2usize, 3];
    let x: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    let reps = vec![2usize, 2];
    assert_eq!(
        run_tile(Device::Cpu, &dims, reps.clone(), &x),
        tile_ref(&dims, &reps, &x)
    );
}

#[test]
fn trilu_matches_reference() {
    let dims = [4usize, 4];
    let x: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    for (upper, diag) in [(true, 0), (false, 0), (true, 1), (false, -1)] {
        assert_eq!(
            run_trilu(Device::Cpu, &dims, upper, diag, &x),
            trilu_ref(&dims, upper, diag, &x),
            "trilu upper={upper} diag={diag}"
        );
    }
}

#[cfg(any(
    all(target_os = "macos", feature = "metal"),
    feature = "gpu",
    feature = "cuda"
))]
fn check_device(device: Device, label: &str) {
    let cx: Vec<f32> = vec![-3.0, -0.5, 0.0, 1.2, 4.0, 2.0, -1.0, 3.5];
    assert_eq!(
        run_clamp(device, &[8], -1.0, 2.5, &cx),
        run_clamp(Device::Cpu, &[8], -1.0, 2.5, &cx),
        "{label} clamp"
    );
    let td = [2usize, 3];
    let tx: Vec<f32> = (1..=6).map(|i| i as f32).collect();
    assert_eq!(
        run_tile(device, &td, vec![2, 2], &tx),
        run_tile(Device::Cpu, &td, vec![2, 2], &tx),
        "{label} tile"
    );
    let hd = [4usize, 4];
    let hx: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    for (u, d) in [(true, 0), (false, -1)] {
        assert_eq!(
            run_trilu(device, &hd, u, d, &hx),
            run_trilu(Device::Cpu, &hd, u, d, &hx),
            "{label} trilu {u} {d}"
        );
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
fn structural_metal_matches_cpu() {
    check_device(Device::Metal, "metal");
}

#[test]
#[cfg(feature = "gpu")]
fn structural_wgpu_matches_cpu() {
    check_device(Device::Gpu, "wgpu");
}

#[test]
#[cfg(feature = "cuda")]
fn structural_cuda_matches_cpu() {
    if !rlx_runtime::is_available(Device::Cuda) {
        return;
    }
    check_device(Device::Cuda, "cuda");
}
