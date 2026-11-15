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
//!
//! Few rows over many keys take the split-K pair (`attention_split` +
//! `attention_combine`) instead: the 5000-key case and the causal one with
//! an uneven last slice (four or more 256-key slices). `RLX_VULKAN_ATTN_SPLIT=0` runs them
//! all on `attention_coop`.
//!
//! Sixteen or more queries take `attention_tiled` (16 queries of a head per
//! workgroup, K/V tiles in shared memory): the `prefill_*` cases — a ViT
//! encoder's 577 tokens, causal with a KV offset and a head dim that is not a
//! multiple of 16, and a custom mask at dh 128. `RLX_VULKAN_ATTN_TILED=0`
//! runs them on `attention_coop`.
//!
//! The `kv_concat_*` cases build K and V as `concat(past, new)` along the
//! sequence axis, the way a decode step extends its cache; `attention_coop`
//! reads the two parts in place (`RLX_VULKAN_KV_FOLD=0` copies them first).

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

#[test]
fn causal_split_keys_uneven_last_slice() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        b: 1,
        q_s: 2,
        k_s: 1027,
        nh: 4,
        dh: 48,
        mask: Some(MaskKind::Causal),
    });
}

#[test]
fn prefill_vit_encoder_unmasked() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        b: 1,
        q_s: 577,
        k_s: 577,
        nh: 6,
        dh: 64,
        mask: Some(MaskKind::None),
    });
}

#[test]
fn prefill_causal_offset_odd_head_dim() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        b: 2,
        q_s: 50,
        k_s: 70,
        nh: 3,
        dh: 40,
        mask: Some(MaskKind::Causal),
    });
}

#[test]
fn prefill_custom_mask_wide_heads() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check(Case {
        b: 1,
        q_s: 33,
        k_s: 100,
        nh: 2,
        dh: 128,
        mask: None,
    });
}

/// `attention(q, concat(past_k, new_k), concat(past_v, new_v))`, rank 3
/// (`[B, S, H·D]`, concat on axis 1) or rank 4 (`[B, H, S, D]`, axis 2).
fn check_kv_concat(
    b: usize,
    past: usize,
    nh: usize,
    dh: usize,
    rank4: bool,
    mask: Option<MaskKind>,
) {
    let f = DType::F32;
    let hs = nh * dh;
    let k_s = past + 1;
    let shape = |s: usize| {
        if rank4 {
            Shape::new(&[b, nh, s, dh], f)
        } else {
            Shape::new(&[b, s, hs], f)
        }
    };
    let axis = if rank4 { 2 } else { 1 };
    let build = || {
        let mut g = Graph::new("kv-concat");
        let q = g.input("q", shape(1));
        let pk = g.input("pk", shape(past));
        let nk = g.input("nk", shape(1));
        let pv = g.input("pv", shape(past));
        let nv = g.input("nv", shape(1));
        let k = g.add_node(rlx_ir::Op::Concat { axis }, vec![pk, nk], shape(k_s));
        let v = g.add_node(rlx_ir::Op::Concat { axis }, vec![pv, nv], shape(k_s));
        let y = match mask {
            Some(kind) => g.attention_kind(q, k, v, nh, dh, kind, shape(1)),
            None => {
                let m = g.input("mask", Shape::new(&[b, k_s], f));
                g.attention(q, k, v, m, nh, dh, shape(1))
            }
        };
        g.set_outputs(vec![y]);
        g
    };
    let n1 = b * hs;
    let np = b * past * hs;
    let (q, pk, nk, pv, nv) = (
        data(n1, 1),
        data(np, 2),
        data(n1, 3),
        data(np, 4),
        data(n1, 5),
    );
    let keep: Vec<f32> = (0..b * k_s).map(|i| f32::from(i % 5 != 2)).collect();
    let mut inputs: Vec<(&str, &[f32])> = vec![
        ("q", &q),
        ("pk", &pk),
        ("nk", &nk),
        ("pv", &pv),
        ("nv", &nv),
    ];
    if mask.is_none() {
        inputs.push(("mask", &keep));
    }
    let want = {
        use rlx::prelude::*;
        Session::new(Device::Cpu)
            .compile(build())
            .run(&inputs)
            .remove(0)
    };
    let got = VulkanExecutable::compile(build()).run(&inputs).remove(0);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst < 1e-4,
        "kv-concat attention b={b} past={past} nh={nh} dh={dh} rank4={rank4} {mask:?}: \
         max |vk − cpu| = {worst:e}"
    );
}

#[test]
fn kv_concat_decode_rank3_custom_mask() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    // TrOCR's decoder self-attention: 128-row cache + this step's row.
    check_kv_concat(1, 128, 4, 64, false, None);
    check_kv_concat(2, 37, 3, 40, false, None);
}

#[test]
fn kv_concat_decode_rank4_causal() {
    let _gpu = common::serialize_gpu();
    if skip() {
        return;
    }
    check_kv_concat(2, 50, 4, 32, true, Some(MaskKind::Causal));
    check_kv_concat(1, 1, 2, 64, true, Some(MaskKind::None));
}
