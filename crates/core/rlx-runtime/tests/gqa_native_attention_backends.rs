// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Which backends' attention reads shared K/V heads itself.**
//!
//! `rlx_ir::retarget::gqa_native_attention` drops `repeat_kv`'s broadcast and
//! hands `Attention` K / V with fewer heads than Q. Every backend must run
//! that graph and give what the CPU gives on the original: CPU, Metal and
//! wgpu index the shared heads; CUDA, ROCm, Vulkan and MLX get them repeated
//! first (`rlx_fusion::expand_shared_kv`) — CUDA and Vulkan returned a wrong
//! tensor without it.

use rlx_ir::op::{MaskKind, Op};
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

mod common;

const F: DType = DType::F32;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const DH: usize = 16;
const SEQ: usize = 3;
const KV_SEQ: usize = 7;

/// `q[1, SEQ, HEADS*DH]` against K / V of `KV_HEADS` broadcast to `HEADS`.
fn graph() -> Graph {
    let f = |d: &[usize]| Shape::new(d, F);
    let group = HEADS / KV_HEADS;
    let mut g = Graph::new("gqa");
    let q = g.input("q", f(&[1, SEQ, HEADS * DH]));
    let mut kv = Vec::new();
    for name in ["k", "v"] {
        let x = g.input(name, f(&[1, KV_SEQ, KV_HEADS * DH]));
        let split = g.reshape(
            x,
            vec![1, KV_SEQ as i64, KV_HEADS as i64, 1, DH as i64],
            f(&[1, KV_SEQ, KV_HEADS, 1, DH]),
        );
        let wide = g.add_node(
            Op::Expand {
                target_shape: vec![1, KV_SEQ as i64, KV_HEADS as i64, group as i64, DH as i64],
            },
            vec![split],
            f(&[1, KV_SEQ, KV_HEADS, group, DH]),
        );
        kv.push(g.reshape(
            wide,
            vec![1, KV_SEQ as i64, (HEADS * DH) as i64],
            f(&[1, KV_SEQ, HEADS * DH]),
        ));
    }
    let a = g.add_node(
        Op::Attention {
            num_heads: HEADS,
            head_dim: DH,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, kv[0], kv[1]],
        f(&[1, SEQ, HEADS * DH]),
    );
    g.set_outputs(vec![a]);
    g
}

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 500.0 - 1.0
        })
        .collect()
}

#[test]
fn attention_on_shared_heads_matches_the_broadcast() {
    let _gpu = common::serialize_gpu();
    let (native, n) = rlx_ir::retarget::gqa_native_attention(&graph()).unwrap();
    assert_eq!(n, 1);
    let q = data(SEQ * HEADS * DH, 1);
    let k = data(KV_SEQ * KV_HEADS * DH, 2);
    let v = data(KV_SEQ * KV_HEADS * DH, 3);
    let feeds = [
        ("q", q.as_slice()),
        ("k", k.as_slice()),
        ("v", v.as_slice()),
    ];
    let want = Session::new(Device::Cpu)
        .compile(graph())
        .run(&feeds)
        .remove(0);
    for (name, dev) in [
        ("cpu", Device::Cpu),
        ("metal", Device::Metal),
        ("mlx", Device::Mlx),
        ("wgpu", Device::Gpu),
        ("cuda", Device::Cuda),
        ("rocm", Device::Rocm),
        ("vulkan", Device::Vulkan),
    ] {
        if common::skip_unless(dev) {
            continue;
        }
        let got = Session::new(dev)
            .compile(native.clone())
            .run(&feeds)
            .remove(0);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("{name}: max |diff| {err:.2e}");
        assert!(
            err < 1e-4,
            "{name}: attention on shared K/V heads differs by {err}"
        );
    }
}
