// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `attention_coop` (one workgroup per attention row, used for few rows —
//! a decode step) vs CPU.
//!
//! The shapes are the ones the kernel exists for and the ones that break a
//! chunked reduction: a single query over more keys than one shared-memory
//! chunk (2048), a custom keep-mask with holes, causal masking with a
//! KV-cache offset (`k_s > q_s`), and head dims below / at the 256 ceiling.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_vulkan::backend::VulkanExecutable;

mod common;

fn skip() -> bool {
    rlx_ir::env::skip_unless_device("vulkan", true, rlx_vulkan::is_available())
}

fn data(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 2001) as f32 / 1000.0) - 1.0)
        .collect()
}

struct Case {
    b: usize,
    q_s: usize,
    k_s: usize,
    nh: usize,
    dh: usize,
    mask: Option<MaskKind>, // None = custom keep-mask input
}

fn build(c: &Case) -> Graph {
    let f = DType::F32;
    let hs = c.nh * c.dh;
    let mut g = Graph::new("attn");
    let q = g.input("q", Shape::new(&[c.b, c.q_s, hs], f));
    let k = g.input("k", Shape::new(&[c.b, c.k_s, hs], f));
    let v = g.input("v", Shape::new(&[c.b, c.k_s, hs], f));
    let out = Shape::new(&[c.b, c.q_s, hs], f);
    let y: NodeId = match c.mask {
        Some(kind) => g.attention_kind(q, k, v, c.nh, c.dh, kind, out),
        None => {
            let m = g.input("mask", Shape::new(&[c.b, c.k_s], f));
            g.attention(q, k, v, m, c.nh, c.dh, out)
        }
    };
    g.set_outputs(vec![y]);
    g
}

fn check(c: Case) {
    let hs = c.nh * c.dh;
    let q = data(c.b * c.q_s * hs, 1);
    let k = data(c.b * c.k_s * hs, 2);
    let v = data(c.b * c.k_s * hs, 3);
    // Keep 1, drop 0: every seventh key masked, never all of a row.
    let mask: Vec<f32> = (0..c.b * c.k_s).map(|i| f32::from(i % 7 != 3)).collect();
    let mut inputs: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
    if c.mask.is_none() {
        inputs.push(("mask", &mask));
    }
    let want = {
        use rlx::prelude::*;
        Session::new(Device::Cpu)
            .compile(build(&c))
            .run(&inputs)
            .remove(0)
    };
    let got = VulkanExecutable::compile(build(&c)).run(&inputs).remove(0);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "attention b={} q_s={} k_s={} nh={} dh={} {:?}: max |vk − cpu| = {worst:e}",
        c.b,
        c.q_s,
        c.k_s,
        c.nh,
        c.dh,
        c.mask
    );
}

#[test]
fn one_query_over_several_chunks_custom_mask() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        b: 1,
        q_s: 1,
        k_s: 5000,
        nh: 4,
        dh: 64,
        mask: None,
    });
}

#[test]
fn causal_with_kv_offset_and_batch() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        b: 2,
        q_s: 3,
        k_s: 37,
        nh: 2,
        dh: 32,
        mask: Some(MaskKind::Causal),
    });
}

#[test]
fn unmasked_wide_heads() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        b: 1,
        q_s: 1,
        k_s: 300,
        nh: 2,
        dh: 256,
        mask: Some(MaskKind::None),
    });
}
