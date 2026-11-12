// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One query row against a bucket of keys — a decode step's attention — in
//! the `[batch, seq, heads · head_dim]` layout decode graphs use, scored
//! against a reference computed here.
//!
//! On aarch64 a single query row has its own path (a task per head, dots over
//! the live keys only), which runs sequentially below a size threshold and
//! splits heads over the pool above it. Both halves are covered, with and
//! without a key-padding mask, plus a short bucket and a single head.

#![cfg(feature = "cpu")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
}

/// `softmax(q·Kᵀ·scale over kept keys) · V` per head.
fn reference(q: &[f32], k: &[f32], v: &[f32], keep: &[bool], h: usize, dh: usize) -> Vec<f32> {
    let ks = keep.len();
    let scale = 1.0 / (dh as f32).sqrt();
    let mut out = vec![0f32; h * dh];
    for hi in 0..h {
        let sc: Vec<f64> = (0..ks)
            .map(|j| {
                let dot: f64 = (0..dh)
                    .map(|d| q[hi * dh + d] as f64 * k[j * h * dh + hi * dh + d] as f64)
                    .sum();
                if keep[j] {
                    dot * scale as f64
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        let max = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = sc.iter().map(|s| (s - max).exp()).collect();
        let z: f64 = w.iter().sum();
        for d in 0..dh {
            out[hi * dh + d] = (0..ks)
                .map(|j| w[j] / z * v[j * h * dh + hi * dh + d] as f64)
                .sum::<f64>() as f32;
        }
    }
    out
}

fn check(h: usize, dh: usize, ks: usize, keep: Option<&dyn Fn(usize) -> bool>) {
    let d = h * dh;
    let mut g = Graph::new("decode_attention");
    let q = g.input("q", Shape::new(&[1, 1, d], F));
    let k = g.input("k", Shape::new(&[1, ks, d], F));
    let v = g.input("v", Shape::new(&[1, ks, d], F));
    let out_shape = Shape::new(&[1, 1, d], F);
    let o = match keep {
        Some(_) => {
            let m = g.input("m", Shape::new(&[1, ks], F));
            g.attention(q, k, v, m, h, dh, out_shape)
        }
        None => g.attention_kind(q, k, v, h, dh, MaskKind::None, out_shape),
    };
    g.set_outputs(vec![o]);
    let mut c = Session::new(Device::Cpu).compile(g);

    let mut seed = (h * 131 + ks) as u32;
    let mut fill = |n: usize| (0..n).map(|_| lcg(&mut seed)).collect::<Vec<f32>>();
    let (qd, kd, vd) = (fill(d), fill(ks * d), fill(ks * d));
    let kept: Vec<bool> = (0..ks).map(|j| keep.is_none_or(|f| f(j))).collect();
    let mask: Vec<f32> = kept.iter().map(|&b| b as u8 as f32).collect();
    let mut inputs: Vec<(&str, &[f32])> = vec![("q", &qd), ("k", &kd), ("v", &vd)];
    if keep.is_some() {
        inputs.push(("m", &mask));
    }
    let got = c.run(&inputs).remove(0);
    let want = reference(&qd, &kd, &vd, &kept, h, dh);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        err < 1e-5,
        "h={h} dh={dh} keys={ks} masked={}: max err {err:.2e}",
        keep.is_some()
    );
}

#[test]
fn padded_decode_bucket() {
    // 129 rows = past 0..60 real, padding, the new key last (`bucket_decode_mask`).
    check(16, 64, 129, Some(&|j| j < 60 || j == 128));
}

#[test]
fn long_context_splits_heads() {
    // 16 heads × 1025 keys × 128 crosses the pool threshold.
    check(16, 64, 1025, None);
    check(16, 64, 1025, Some(&|j| j % 3 != 0));
}

#[test]
fn short_bucket_and_single_head() {
    check(16, 64, 24, Some(&|j| j < 10 || j == 23));
    check(1, 64, 129, Some(&|j| j < 100));
}
