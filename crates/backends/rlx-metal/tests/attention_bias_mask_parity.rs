// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Attention` with `MaskKind::Bias` vs CPU — rlx-metal.
//!
//! W2v-BERT uses `position_embeddings_type = "relative_key"`, which builds a
//! Shaw-style per-head relative-position bias and feeds it to the fast
//! attention path as an additive `[B, H, Sq, Sk]` tensor. It is the one part of
//! `rlx-tribev2-audio` not yet exonerated on metal (its LayerNorms, its whole
//! FFN1 chain and its depthwise conv are all bit-exact there), so this covers
//! the bias path directly at that model's shape: 16 heads, head_dim 64, seq 16.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn case(b: usize, nh: usize, hd: usize, seq: usize) -> Option<f32> {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_metal::is_available()) {
        eprintln!("skip: metal unavailable");
        return None;
    }
    let dim = nh * hd;
    let mut g = Graph::new("attn_bias");
    let q = g.input("q", Shape::new(&[b, seq, dim], DType::F32));
    let k = g.input("k", Shape::new(&[b, seq, dim], DType::F32));
    let v = g.input("v", Shape::new(&[b, seq, dim], DType::F32));
    let bias = g.input("bias", Shape::new(&[b, nh, seq, seq], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::Bias,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, bias],
        Shape::new(&[b, seq, dim], DType::F32),
    );
    g.set_outputs(vec![y]);

    let qs: Vec<f32> = (0..b * seq * dim)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let ks: Vec<f32> = (0..b * seq * dim)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let vs: Vec<f32> = (0..b * seq * dim)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    // A relative-position bias: depends on (qi - ki), clamped to a window, with
    // distinct left/right extents like W2v-BERT's 64/8.
    let bs: Vec<f32> = (0..b * nh * seq * seq)
        .map(|i| {
            let ki = (i % seq) as i64;
            let qi = ((i / seq) % seq) as i64;
            let h = (i / (seq * seq)) % nh;
            let d = (qi - ki).clamp(-64, 8) as f32;
            (d * 0.05 + h as f32 * 0.01).tanh()
        })
        .collect();

    let run = |d: Device| -> Vec<f32> {
        let mut c = Session::new(d).compile(g.clone());
        c.run(&[
            ("q", qs.as_slice()),
            ("k", ks.as_slice()),
            ("v", vs.as_slice()),
            ("bias", bs.as_slice()),
        ])
        .remove(0)
    };
    let cpu = run(Device::Cpu);
    let met = run(Device::Metal);
    Some(
        cpu.iter()
            .zip(&met)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max),
    )
}

#[test]
fn attention_bias_mask_matches_cpu() {
    // (b, heads, head_dim, seq); the last is W2v-BERT's real geometry.
    let cases = [
        (1usize, 4usize, 32usize, 8usize),
        (1, 16, 64, 16),
        (2, 16, 64, 16),
    ];
    let mut bad = Vec::new();
    for (b, nh, hd, s) in cases {
        let Some(d) = case(b, nh, hd, s) else { return };
        eprintln!("attn bias b={b} heads={nh} hd={hd} seq={s}: max_abs={d:.6e}");
        if d > 1e-4 {
            bad.push((b, nh, hd, s, d));
        }
    }
    assert!(
        bad.is_empty(),
        "metal attention+bias disagrees with CPU: {bad:?}"
    );
}
