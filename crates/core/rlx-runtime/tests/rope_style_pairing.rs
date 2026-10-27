// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Does `Op::Rope`'s `style` field actually reach every executor?
//!
//! `rlx-flow/src/rope.rs` (`rotate_half_gptj`) records an open question:
//!
//! > `RopeStyle::GptJ` is honoured in the fused *thunk* path
//! > (`rlx-cpu/src/thunk/ops/attention.rs` sets `interleaved` from it), but the
//! > standalone `Op::Rope` executor arm destructures
//! > `Op::Rope { head_dim, n_rot, .. }` and implements only the NeoX pairing.
//! > Whether a lowering pass routes GptJ away from that arm was not established.
//!
//! Reading confirms the premise: `rlx-cpu/src/executor.rs`'s arm binds only
//! `{ head_dim, n_rot, .. }` and never looks at `style`, while
//! `rlx-cpu/src/thunk/compile_dispatch.rs` does bind it. This test settles the
//! consequence empirically, per backend, by comparing against a host reference
//! for **each** pairing:
//!
//! * NeoX (rotate-half) pairs lane `i` with lane `i + head_dim/2`:
//!   `out[i] = x[i]·cos[i] − x[i+h]·sin[i]`, `out[i+h] = x[i+h]·cos[i] + x[i]·sin[i]`.
//! * GptJ (interleaved) pairs adjacent lanes `2i` and `2i+1`:
//!   `out[2i] = x[2i]·cos[i] − x[2i+1]·sin[i]`,
//!   `out[2i+1] = x[2i+1]·cos[i] + x[2i]·sin[i]`.
//!
//! Both read the same half-width table (one angle per pair), so a backend that
//! silently applies the NeoX pairing to a `GptJ` node produces a *plausible*
//! tensor of the right shape and magnitude — the failure mode that makes this
//! worth pinning with a test rather than a comment.
//!
//! The angles here are deliberately far from 0 and from π/2, so NeoX and GptJ
//! disagree by O(1) on this input: with equal angles per pair the two pairings
//! coincide, and such a test would pass either way.

#![cfg(feature = "cpu")]

use rlx_ir::infer::GraphExt as _;
use rlx_ir::op::RopeStyle;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;

/// `[b, h, n, d]` input with distinct, well-separated values.
fn make_x(b: usize, h: usize, n: usize, d: usize) -> Vec<f32> {
    (0..b * h * n * d)
        .map(|i| 1.0 + (i % 13) as f32 * 0.37)
        .collect()
}

/// Half-width `cos`/`sin` tables: one angle per lane pair, per position.
fn make_tables(n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let half = d / 2;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for p in 0..n {
        for i in 0..half {
            // not a multiple of pi/2, and different per lane, so the two
            // pairings cannot coincide
            let a = 0.3 + 0.41 * (p as f32) + 0.17 * (i as f32);
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

/// Host reference for one pairing.
fn reference(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    b: usize,
    h: usize,
    n: usize,
    d: usize,
    style: RopeStyle,
) -> Vec<f32> {
    let half = d / 2;
    let mut out = x.to_vec();
    for bi in 0..b {
        for hi in 0..h {
            for p in 0..n {
                let base = ((bi * h + hi) * n + p) * d;
                for i in 0..half {
                    let (c, s) = (cos[p * half + i], sin[p * half + i]);
                    let (lo, hi_lane) = match style {
                        RopeStyle::NeoX => (i, i + half),
                        RopeStyle::GptJ => (2 * i, 2 * i + 1),
                    };
                    let (a, bb) = (x[base + lo], x[base + hi_lane]);
                    out[base + lo] = a * c - bb * s;
                    out[base + hi_lane] = bb * c + a * s;
                }
            }
        }
    }
    out
}

fn check(device: Device, style: RopeStyle) {
    let (b, h, n, d) = (1usize, 2usize, 3usize, 8usize);
    let x = make_x(b, h, n, d);
    let (cos, sin) = make_tables(n, d);

    let mut g = Graph::new("rope_style");
    let xi = g.input("x", Shape::new(&[b, h, n, d], F));
    let ci = g.param("cos", Shape::new(&[n, d / 2], F));
    let si = g.param("sin", Shape::new(&[n, d / 2], F));
    let out = g.rope_styled(xi, ci, si, d, style);
    g.set_outputs(vec![out]);

    let mut c = Session::new(device).compile(g);
    c.set_param("cos", &cos);
    c.set_param("sin", &sin);
    c.finalize_params();
    let got = c.run(&[("x", &x)]).into_iter().next().unwrap();

    let want = reference(&x, &cos, &sin, b, h, n, d, style);
    let other = reference(
        &x,
        &cos,
        &sin,
        b,
        h,
        n,
        d,
        match style {
            RopeStyle::NeoX => RopeStyle::GptJ,
            RopeStyle::GptJ => RopeStyle::NeoX,
        },
    );

    let err = |r: &[f32]| {
        got.iter()
            .zip(r)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    };
    let (e_want, e_other) = (err(&want), err(&other));
    eprintln!(
        "[rope-style:{style:?} on {device:?}] max|Δ| vs {style:?} = {e_want:e}, \
         vs the OTHER pairing = {e_other:e}"
    );
    // The two pairings must be distinguishable on this input, or the test is vacuous.
    assert!(
        e_other > 1e-2,
        "the two pairings agree on this input ({e_other:e}) — the test would pass either way"
    );
    assert!(
        e_want < 1e-5,
        "{device:?} did not apply the {style:?} pairing (max|Δ| {e_want:e}); \
         distance to the other pairing is {e_other:e}"
    );
}

#[test]
fn cpu_honours_both_pairings() {
    check(Device::Cpu, RopeStyle::NeoX);
    check(Device::Cpu, RopeStyle::GptJ);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_honours_both_pairings() {
    let _gpu = common::serialize_gpu();
    check(Device::Metal, RopeStyle::NeoX);
    check(Device::Metal, RopeStyle::GptJ);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_honours_both_pairings() {
    let _gpu = common::serialize_gpu();
    check(Device::Gpu, RopeStyle::NeoX);
    check(Device::Gpu, RopeStyle::GptJ);
}

#[cfg(all(feature = "coreml", target_vendor = "apple"))]
#[test]
fn coreml_honours_both_pairings() {
    let _gpu = common::serialize_gpu();
    check(Device::Ane, RopeStyle::NeoX);
    check(Device::Ane, RopeStyle::GptJ);
}

#[cfg(feature = "mlx")]
#[test]
fn mlx_honours_both_pairings() {
    let _gpu = common::serialize_gpu();
    check(Device::Mlx, RopeStyle::NeoX);
    check(Device::Mlx, RopeStyle::GptJ);
}
