// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// `MaskKind::Custom` attention on CoreML, in BOTH layouts the rest of the stack
// produces: the raw `[B, S_k]` key-padding vector, and the `[B, H, S_q, S_k]`
// score-shaped mask that `rlx-unfuse` passes through untouched
// (`mask_dims == target_dims`).
//
// The MIL lowering used to assume the first layout unconditionally and reshape
// the mask to `[B, 1, .., 1, S_k]`. On a score-shaped mask that drops
// `H · S_q` elements — an `mps.reshape` whose operands disagree. `rlx_ir` does
// not verify element counts on a fully-concrete reshape target (ONNX importers
// depend on that), and the CPU backend just reinterprets the buffer, so nothing
// upstream caught it: MPSGraph did, by `abort()`ing the process with
// `original module failed verification`. An `abort()` inside a vendor framework
// cannot be caught, so this test is the guard.
#![cfg(any(target_os = "macos", target_os = "ios"))]

use rlx_coreml::CoremlExecutable;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

const B: usize = 1;
const H: usize = 2;
const S: usize = 3;
const D: usize = 4;

fn ramp(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.31 + seed).sin() * 0.5)
        .collect()
}

/// `q/k/v` are `[B, H, S, D]`; `mask` has `mask_dims` and is 1.0 = keep.
fn build(mask_dims: &[usize]) -> Graph {
    let mut g = Graph::new("attn_custom");
    let q = g.input("q", Shape::new(&[B, H, S, D], DType::F32));
    let k = g.input("k", Shape::new(&[B, H, S, D], DType::F32));
    let v = g.input("v", Shape::new(&[B, H, S, D], DType::F32));
    let m = g.input("mask", Shape::new(mask_dims, DType::F32));
    let y = g.append_node(
        Op::Attention {
            num_heads: H,
            head_dim: D,
            v_head_dim: None,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, m],
        Shape::new(&[B, H, S, D], DType::F32),
        None,
    );
    g.set_outputs(vec![y]);
    g
}

fn cpu(mask_dims: &[usize], mask: &[f32], q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(build(mask_dims));
    c.run(&[("q", q), ("k", k), ("v", v), ("mask", mask)])
        .remove(0)
}

fn coreml(mask_dims: &[usize], mask: &[f32], q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut e = CoremlExecutable::compile(build(mask_dims));
    e.run(&[("q", q), ("k", k), ("v", v), ("mask", mask)])
        .expect("coreml run")
        .remove(0)
}

fn check(mask_dims: &[usize], mask: &[f32]) {
    let (q, k, v) = (
        ramp(B * H * S * D, 0.0),
        ramp(B * H * S * D, 1.0),
        ramp(B * H * S * D, 2.0),
    );
    let want = cpu(mask_dims, mask, &q, &k, &v);
    let got = coreml(mask_dims, mask, &q, &k, &v);
    assert_eq!(got.len(), want.len(), "len {} vs {}", got.len(), want.len());
    let mx = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        mx <= 1e-4,
        "mask {mask_dims:?}: max abs diff {mx}\n got {got:?}\n ref {want:?}"
    );
}

/// The historical layout: a `[B, S_k]` key-padding vector, broadcast over heads
/// and queries by the lowering.
#[test]
fn custom_mask_key_padding_matches_cpu() {
    check(&[B, S], &[1.0, 1.0, 0.0]);
}

/// The layout that used to emit an invalid `reshape`: already score-shaped,
/// because `rlx-unfuse` normalises a Custom mask to `[B, H, S_q, S_k]` before
/// rebuilding `Op::Attention`.
///
/// Note what the reference actually is. Both CPU paths read a Custom mask as
/// `mask_data[bi * k_s + ki]` — the leading `B · S_k` floats — no matter what
/// rank the buffer declares, so the extra `H · S_q` values are *ignored*, not
/// applied per head/query. CoreML has to reproduce that, not "improve" on it;
/// CPU is the reference the whole workspace is anchored to.
#[test]
fn custom_mask_score_shaped_matches_cpu() {
    // Varies across every axis, so a lowering that read the WRONG leading slice
    // (or applied the mask elementwise) would disagree with CPU.
    let mut m = vec![1.0f32; B * H * S * S];
    for h in 0..H {
        for qi in 0..S {
            for ki in 0..S {
                if ki > qi + h {
                    m[(h * S + qi) * S + ki] = 0.0;
                }
            }
        }
    }
    check(&[B, H, S, S], &m);
}
