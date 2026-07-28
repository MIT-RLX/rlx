// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Attention` prefill parity across head_dim + GQA — wgpu vs CPU.
//!
//! Regression probe for Gemma 4 E2B. Sliding layers: nh=8, nkv=1, head_dim=256
//! (works, bit-exact). Full-attention (global) layers: nh=8, nkv=1,
//! head_dim=512 — GQA at head_dim 512, the config the wgpu attention kernel
//! had never exercised before E2B (gemma2/3 are all head_dim 256).

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn run_case(nh: usize, nkv: usize, hd: usize, seq: usize) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let b = 1usize;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd; // GQA: nkv < nh (E2B global layers: nh=8, nkv=1)
    // Gemma 4 regime: Q/K per-head RMS-normed (‖head‖≈√hd), score_scale=1.0, so
    // scores ≈ hd. Unit-rms Q/K reproduce that large-score softmax regime.
    let q: Vec<f32> = (0..b * seq * q_dim)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let k: Vec<f32> = (0..b * seq * kv_dim)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..b * seq * kv_dim)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();

    let mut g = Graph::new("attn_hd");
    let qi = g.input("q", Shape::new(&[b, seq, q_dim], DType::F32));
    let ki = g.input("k", Shape::new(&[b, seq, kv_dim], DType::F32));
    let vi = g.input("v", Shape::new(&[b, seq, kv_dim], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            mask_kind: MaskKind::Causal,
            score_scale: Some(1.0), // Gemma 4: unit scale (Q pre-normed)
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi],
        Shape::new(&[b, seq, q_dim], DType::F32),
    );
    g.set_outputs(vec![y]);

    let run = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        c.run(&[
            ("q", q.as_slice()),
            ("k", k.as_slice()),
            ("v", v.as_slice()),
        ])
        .remove(0)
    };
    let gpu = run(Device::Gpu);
    let cpu = run(Device::Cpu);
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("attention nh={nh} nkv={nkv} hd={hd} seq={seq}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

#[test]
fn wgpu_attention_gqa_head_dim_parity() {
    // (nh, nkv, hd): E2B sliding = (8,1,256), E2B global = (8,1,512).
    let cases: &[(usize, usize, usize)] = &[
        (8, 8, 512), // MHA hd512 (isolation baseline — passes)
        (8, 1, 256), // GQA hd256 (E2B sliding — passes)
        (8, 1, 512), // GQA hd512 (E2B global — the suspect)
    ];
    for &(nh, nkv, hd) in cases {
        let Some(max_abs) = run_case(nh, nkv, hd, 5) else {
            return;
        };
        assert!(
            max_abs <= 1e-2,
            "wgpu attention nh={nh} nkv={nkv} hd={hd} max_abs {max_abs} > 1e-2"
        );
    }
}
