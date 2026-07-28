// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PACKED-input causal `Op::Attention` (`[B, S, H*D]`, internal head split) on
//! Metal vs CPU — the exact form the Voxtral LM prefill uses (q_rope is
//! `[B, S, 32*128]`), with head_dim=128. The other Metal attention tests pass
//! pre-split `[B, H, S, D]`, so this layout is untested.

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_packed_causal_attn(b: usize, s: usize, h: usize, d: usize) -> Graph {
    let f = DType::F32;
    let w = h * d;
    let mut g = Graph::new("packed_causal_attn");
    let q = g.input("q", Shape::new(&[b, s, w], f));
    let k = g.input("k", Shape::new(&[b, s, w], f));
    let v = g.input("v", Shape::new(&[b, s, w], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, w], f),
    );
    g.set_outputs(vec![y]);
    g
}

#[test]
fn metal_packed_causal_attention_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    // Voxtral-Mini-3B text: 32 heads, head_dim 128 (so packed width 4096).
    let (b, s, h, d) = (1, 64, 32, 128);
    let w = h * d;
    let n = b * s * w;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0007).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0011).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0005).sin()).collect();

    let g = build_packed_causal_attn(b, s, h, d);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);

    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let cpu_sum: f64 = cpu.iter().map(|&x| x as f64).sum();
    let metal_sum: f64 = metal.iter().map(|&x| x as f64).sum();
    eprintln!(
        "packed causal attn (hd=128): max_abs={max_abs:.6} cpu_sum={cpu_sum:.4} metal_sum={metal_sum:.4}"
    );
    assert!(max_abs < 1e-4, "packed causal attention max_abs={max_abs}");
}
