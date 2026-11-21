// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Softmax on ROCm against the CPU, across its three kernels: a warp a vector
//! (contiguous, axis <= 64), a thread a vector (strided, axis <= 64) and a
//! block a vector (longer axes) — and over a middle axis, as SynthSeg's
//! labels are.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn data(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 9) as f32 / 4_194_304.0 - 1.0) * 8.0
        })
        .collect()
}

fn softmax(device: Device, dims: &[usize], axis: i32, x: &[f32]) -> Vec<f32> {
    let mut g = Graph::new("softmax");
    let shape = Shape::new(dims, DType::F32);
    let i = g.input("x", shape.clone());
    let y = g.softmax(i, axis, shape);
    g.set_outputs(vec![y]);
    Session::new(device).compile(g).run(&[("x", x)]).remove(0)
}

#[test]
fn softmax_short_axes_match_cpu() {
    for (dims, axis) in [
        (&[1000usize, 33][..], -1),
        (&[7, 64][..], 1),
        (&[5, 65][..], -1),
        (&[3, 1][..], -1),
        (&[2, 33, 500][..], 1),
        (&[1, 33, 16, 16, 16][..], 1),
        (&[4, 300][..], -1),
    ] {
        let x = data(
            dims.iter().product(),
            dims.len() as u32 * 31 + dims[0] as u32,
        );
        let want = softmax(Device::Cpu, dims, axis, &x);
        let got = softmax(Device::Rocm, dims, axis, &x);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{dims:?} axis {axis}: max |diff| {err:.2e}");
        assert!(err <= 1e-6, "{dims:?} axis {axis}: max |diff| {err}");
    }
}
