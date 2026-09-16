// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Every spelling of one key-padding mask must give one answer, everywhere.**
//!
//! `MaskKind::Custom` is documented as a binary `[batch, key_len]` mask
//! (`1.0` = valid, `<0.5` = ignored). `[1, S_k]` — one padding row broadcast
//! over the batch — is that same mask, and so are its rank-3 and rank-4
//! spellings `[1, 1, S_k]`, `[B, 1, 1, S_k]`, and so on. All must agree.
//!
//! They did not. Two independent copies of the same mistake:
//!
//! * The stride-driven backends (CUDA, ROCm, wgpu) derived each mask stride
//!   from a product of dims, giving an axis of extent 1 a non-zero stride
//!   instead of 0. `rlx_ir::mask_strides_for_shape` is now the single copy, and
//!   zeroes broadcast axes; wgpu had a byte-identical private duplicate, which
//!   is how the two drifted apart in the first place.
//! * CPU, Metal and Vulkan index the mask at a hard-coded `mask[b * S_k + k]`
//!   with no strides at all, so `[1, S_k]` read **past the end of the tensor**
//!   for every batch above 0 — into whatever the arena placed next to it.
//!   `rlx_opt::legalize_custom_attention_mask` materializes the broadcast in
//!   the IR ahead of them.
//!
//! Scored against a reference computed here, not against another backend: they
//! disagreed with each other, so picking one as the oracle would beg the
//! question. Reading a neighbouring tensor also tends to produce *plausible*
//! numbers, which is why this needs an independent answer rather than a
//! cross-check.
//!
//! **Not tested here, because it is now refused:** a per-query mask
//! (`[S_q, S_k]`, `[B, H, S_q, S_k]`, …). `Custom` does not define one —
//! `MaskKind::Bias` is the per-head, per-query tensor. MLX and wgpu happen to
//! broadcast such a mask per query; CPU, Metal and Vulkan read the same tensor
//! as key padding. Pinning that would freeze a disagreement rather than test a
//! contract, so `rlx_ir::repr_check` rejects it instead (rule R8) and
//! `repr_check::tests::flags_per_query_custom_mask` covers it. The legal
//! spellings enumerated below are the ones that rule must keep accepting.

use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;
const B: usize = 2;
const H: usize = 2;
const S: usize = 4;
const DH: usize = 8;

fn build(mask_dims: &[usize]) -> Graph {
    let mut g = Graph::new("attn_mask");
    let q = g.input("q", Shape::new(&[B, H, S, DH], F));
    let k = g.input("k", Shape::new(&[B, H, S, DH], F));
    let v = g.input("v", Shape::new(&[B, H, S, DH], F));
    let m = g.input("m", Shape::new(mask_dims, F));
    let o = g.attention(q, k, v, m, H, DH, Shape::new(&[B, H, S, DH], F));
    g.set_outputs(vec![o]);
    g
}

/// `softmax(QKᵀ/√d, masked) · V`, with `keep(b, j)` deciding which keys survive.
fn reference(q: &[f32], k: &[f32], v: &[f32], keep: &dyn Fn(usize, usize) -> bool) -> Vec<f32> {
    let scale = 1.0 / (DH as f32).sqrt();
    let mut out = vec![0.0f32; B * H * S * DH];
    for b in 0..B {
        for h in 0..H {
            let base = (b * H + h) * S * DH;
            for i in 0..S {
                let mut sc = [0.0f32; S];
                for (j, s) in sc.iter_mut().enumerate() {
                    let dot: f32 = (0..DH)
                        .map(|d| q[base + i * DH + d] * k[base + j * DH + d])
                        .sum();
                    *s = if keep(b, j) {
                        dot * scale
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut den = 0.0f32;
                for s in sc.iter_mut() {
                    *s = if s.is_finite() { (*s - mx).exp() } else { 0.0 };
                    den += *s;
                }
                for d in 0..DH {
                    let acc: f32 = (0..S).map(|j| sc[j] * v[base + j * DH + d]).sum();
                    out[base + i * DH + d] = if den > 0.0 { acc / den } else { 0.0 };
                }
            }
        }
    }
    out
}

/// Deliberately NOT filtered by availability — `check` routes each one through
/// `skip_unless_available`, so a device this host cannot instantiate is
/// *reported* (and, under `RLX_REQUIRE_DEVICE=1`, fails). Pre-filtering here
/// would drop it silently, and a run covering only CPU would report `ok` as
/// loudly as one covering six backends.
const DEVICES: &[(&str, Device)] = &[
    ("cpu", Device::Cpu),
    ("metal", Device::Metal),
    ("mlx", Device::Mlx),
    ("wgpu", Device::Gpu),
    ("cuda", Device::Cuda),
    ("rocm", Device::Rocm),
    ("vulkan", Device::Vulkan),
];

fn qkv() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = B * H * S * DH;
    (
        (0..n).map(|i| (i as f32 * 0.13).sin()).collect(),
        (0..n).map(|i| (i as f32 * 0.17).cos()).collect(),
        (0..n).map(|i| (i as f32 * 0.11).sin()).collect(),
    )
}

/// Run `dims`/`data` on every backend and require agreement with `keep`.
fn check(label: &str, dims: &[usize], data: &[f32], keep: &dyn Fn(usize, usize) -> bool) {
    let (q, k, v) = qkv();
    let want = reference(&q, &k, &v, keep);
    // A mask that keeps everything would agree with any misreading; make sure
    // the fixture actually discriminates before trusting a pass.
    let unmasked = reference(&q, &k, &v, &|_, _| true);
    let spread = want
        .iter()
        .zip(&unmasked)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        spread > 1e-3,
        "{label}: this mask barely changes the output ({spread:.2e}) — the case \
         cannot tell a correct read from a wrong one"
    );

    let mut ran = Vec::new();
    for (name, dev) in DEVICES {
        if common::skip_unless_available(*dev, name) {
            continue;
        }
        let dev = *dev;
        let got = Session::new(dev)
            .compile(build(dims))
            .run(&[("q", &q), ("k", &k), ("v", &v), ("m", data)])
            .remove(0);
        let d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            d < 1e-4,
            "{name}: mask {dims:?} ({label}) differs from the reference by \
             {d:.3e}. A size-1 axis is broadcast — check that its stride is 0 \
             (`rlx_ir::mask_strides_for_shape`) or that the broadcast was \
             materialized (`rlx_opt::legalize_custom_attention_mask`)."
        );
        ran.push(*name);
    }
    eprintln!("  {label} {dims:?}: checked on {ran:?}");
}

/// The documented shape, per batch — the case that already worked everywhere.
#[test]
fn per_batch_key_padding_matches_the_reference() {
    let _gpu = common::serialize_gpu();
    let keep = |b: usize, j: usize| j + b < S;
    let data: Vec<f32> = (0..B * S)
        .map(|x| if keep(x / S, x % S) { 1.0 } else { 0.0 })
        .collect();
    for dims in [vec![B, S], vec![B, 1, S], vec![B, 1, 1, S]] {
        check("per-batch key padding", &dims, &data, &keep);
    }
}

/// One padding row broadcast over the batch — the case that was wrong on all
/// six GPU backends, in two different ways, and read out of bounds on three.
#[test]
fn batch_broadcast_key_padding_matches_the_reference() {
    let _gpu = common::serialize_gpu();
    let keep = |_b: usize, j: usize| j + 1 < S;
    let data: Vec<f32> = (0..S).map(|j| if keep(0, j) { 1.0 } else { 0.0 }).collect();
    for dims in [vec![1, S], vec![1, 1, S], vec![1, 1, 1, S]] {
        check("batch-broadcast key padding", &dims, &data, &keep);
    }
}

/// Every spelling is the same mask, so every backend must also agree with
/// *itself* across them — a backend could match the reference on `[B, S_k]` and
/// still mis-index the broadcast form.
#[test]
fn all_spellings_of_one_mask_agree_on_each_backend() {
    let _gpu = common::serialize_gpu();
    let (q, k, v) = qkv();
    let keep = |_b: usize, j: usize| j + 1 < S;
    let data: Vec<f32> = (0..S).map(|j| if keep(0, j) { 1.0 } else { 0.0 }).collect();
    // The same mask written out per batch, for the rank-2 documented form.
    let per_batch: Vec<f32> = (0..B * S)
        .map(|x| if keep(0, x % S) { 1.0 } else { 0.0 })
        .collect();

    for (name, dev) in DEVICES {
        if common::skip_unless_available(*dev, name) {
            continue;
        }
        let dev = *dev;
        let run = |dims: &[usize], d: &[f32]| {
            Session::new(dev)
                .compile(build(dims))
                .run(&[("q", &q), ("k", &k), ("v", &v), ("m", d)])
                .remove(0)
        };
        let base = run(&[B, S], &per_batch);
        for dims in [vec![1, S], vec![1, 1, S], vec![1, 1, 1, S]] {
            let got = run(&dims, &data);
            let d = base
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                d < 1e-5,
                "{name}: mask {dims:?} disagrees with the same mask written as \
                 [B, S_k] by {d:.3e} — one of the two spellings is being \
                 mis-indexed"
            );
        }
    }
}
