// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Asymmetric `Op::Attention { v_head_dim: Some(v) }` (DeepSeek/Kimi MLA)
//! parity — wgpu vs CPU.
//!
//! Q/K scores contract over `head_dim`; V rows + output are `v_head_dim`
//! wide (v != head_dim). The CPU backend already implements the asymmetric
//! path (`dh`=head_dim for Q/K, `dh_v`=v_head_dim for V-read/score@V/O-write);
//! this pins the wgpu kernel to it.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

/// Rank-3 `[B, S, H·head_dim]` Q/K, `[B, S, H·v_head_dim]` V/out, causal.
fn run_case(nh: usize, hd: usize, v_hd: usize, seq: usize) -> Option<f32> {
    if !rlx_runtime::is_available(Device::Gpu) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let b = 1usize;
    let q_dim = nh * hd; // Q/K per-head width = head_dim
    let v_dim = nh * v_hd; // V/out per-head width = v_head_dim (asymmetric)

    let q: Vec<f32> = (0..b * seq * q_dim)
        .map(|i| ((i as f32) * 0.021).sin())
        .collect();
    let k: Vec<f32> = (0..b * seq * q_dim)
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let v: Vec<f32> = (0..b * seq * v_dim)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();

    let mut g = Graph::new("attn_vhd");
    let qi = g.input("q", Shape::new(&[b, seq, q_dim], DType::F32));
    let ki = g.input("k", Shape::new(&[b, seq, q_dim], DType::F32));
    let vi = g.input("v", Shape::new(&[b, seq, v_dim], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: Some(v_hd), // asymmetric MLA-style SDPA
            mask_kind: MaskKind::Causal,
            score_scale: None, // default = 1/sqrt(head_dim)
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi],
        Shape::new(&[b, seq, v_dim], DType::F32),
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
    assert_eq!(
        gpu.len(),
        b * seq * v_dim,
        "gpu output width == H·v_head_dim"
    );
    assert_eq!(
        cpu.len(),
        b * seq * v_dim,
        "cpu output width == H·v_head_dim"
    );
    let max_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("attn asym nh={nh} hd={hd} v_hd={v_hd} seq={seq}: max_abs={max_abs:.6e}");
    Some(max_abs)
}

#[test]
fn wgpu_attention_asymmetric_v_head_dim_parity() {
    // head_dim=8, v_head_dim=4, 2 heads, short seq, Causal, rank-3.
    let Some(max_abs) = run_case(2, 8, 4, 6) else {
        return;
    };
    assert!(
        max_abs < 1e-3,
        "wgpu asymmetric v_head_dim max_abs {max_abs} >= 1e-3"
    );
}
