// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! wgpu MLX affine dequant-matmul with more than 65535 workgroups.
//!
//! The kernel runs one workgroup per (output column, 8-row tile) and was
//! dispatched as a 1-D grid of `n * ceil(m / 8)` groups, which WebGPU caps at
//! 65535 per dimension: a vocabulary head (`n = 262208`, MedGemma) failed
//! validation even for a single token, and so did a 2048-wide projection over
//! a 275-token prompt. The grid is now folded into two dimensions.
#![cfg(feature = "gpu")]

mod common;

use common::skip_unless_available;
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

fn run(
    device: Device,
    m: usize,
    k: usize,
    n: usize,
    x: &[f32],
    w: &[f32],
    s: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let mut g = Graph::new("dequant_wide");
    let xi = g.input("x", Shape::new(&[m, k], DType::F32));
    let wq = g.param("w", Shape::new(&[n * k / 8], DType::F32));
    let sc = g.param("s", Shape::new(&[n, k / 64], DType::F32));
    let bi = g.param("b", Shape::new(&[n, k / 64], DType::F32));
    let scheme = QuantScheme::MlxAffine {
        bits: 4,
        group_size: 64,
    };
    let y = g.dequant_matmul(xi, wq, sc, bi, scheme, Shape::new(&[m, n], DType::F32));
    g.set_outputs(vec![y]);
    let mut c = Session::new(device).compile(g);
    c.set_param("w", w);
    c.set_param("s", s);
    c.set_param("b", b);
    c.run(&[("x", x)]).remove(0)
}

#[test]
fn wide_and_tall_match_cpu() {
    if skip_unless_available(Device::Gpu, "wgpu_mlx_dequant_wide") {
        return;
    }
    // (m, k, n): one token over a 70000-column head (70000 groups), and 17
    // rows (3 row tiles) over 24000 columns (72000 groups).
    for (m, k, n) in [(1usize, 128usize, 70_000usize), (17, 64, 24_000)] {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        // Packed 4-bit codes travel bit-for-bit as f32 words.
        let w: Vec<f32> = (0..n * k / 8)
            .map(|_| f32::from_bits(next() & 0x3f3f_3f3f))
            .collect();
        let s: Vec<f32> = (0..n * k / 64)
            .map(|_| (next() % 1000) as f32 * 1e-4 + 0.01)
            .collect();
        let b: Vec<f32> = (0..n * k / 64)
            .map(|_| (next() % 1000) as f32 * 1e-4 - 0.05)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|_| (next() % 2000) as f32 * 1e-3 - 1.0)
            .collect();
        let want = run(Device::Cpu, m, k, n, &x, &w, &s, &b);
        let got = run(Device::Gpu, m, k, n, &x, &w, &s, &b);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let diff = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(
            diff <= 1e-5 * scale.max(1.0),
            "m={m} k={k} n={n}: max diff {diff} (scale {scale})"
        );
    }
}
