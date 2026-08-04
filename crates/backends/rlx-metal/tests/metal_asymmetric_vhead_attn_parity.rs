// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Asymmetric `v_head_dim` (MLA) `Op::Attention` on Metal vs CPU.
//!
//! DeepSeek/Kimi MLA read V with `v_head_dim` (128) while Q/K scores use the
//! larger `head_dim` (192); the output is `num_heads * v_head_dim` wide. The
//! CPU backend is the reference. These tests exercise the three f32 SDPA
//! kernels the standard rank-3 `Op::Attention` path uses:
//!   - `sdpa`            (short causal prefill, Lq == Lk ≤ 64)
//!   - `sdpa_long`       (long causal prefill, Lq == Lk > 64)
//!   - `sdpa_decode_m1`  (decode step, Lq = 1, Lk > 1)
//! with `head_dim != v_head_dim` on each.

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

/// Rank-3 `[B, S, H*head_dim]` Q/K, `[B, Sk, H*v_head_dim]` V, causal.
fn build_asym_attn(b: usize, h: usize, sq: usize, sk: usize, d: usize, vd: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("asym_vhead_attn");
    let q = g.input("q", Shape::new(&[b, sq, h * d], f));
    let k = g.input("k", Shape::new(&[b, sk, h * d], f));
    let v = g.input("v", Shape::new(&[b, sk, h * vd], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: Some(vd),
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, sq, h * vd], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn max_abs(metal: &[f32], cpu: &[f32]) -> f32 {
    assert_eq!(metal.len(), cpu.len(), "length mismatch metal vs cpu");
    cpu.iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn run_case(label: &str, b: usize, h: usize, sq: usize, sk: usize, d: usize, vd: usize) {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let nq = b * sq * h * d;
    let nk = b * sk * h * d;
    let nv = b * sk * h * vd;
    let q: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.031).sin()).collect();
    let k: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.047).cos()).collect();
    let v: Vec<f32> = (0..nv).map(|i| ((i as f32) * 0.013).sin()).collect();

    let g = build_asym_attn(b, h, sq, sk, d, vd);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);

    // Output must be H*v_head_dim wide, not H*head_dim.
    assert_eq!(cpu.len(), b * sq * h * vd, "cpu output width wrong");
    let mx = max_abs(&metal, &cpu);
    eprintln!("{label}: max_abs={mx:.6} (out_len={})", cpu.len());
    assert!(
        mx < 1e-3,
        "{label} asymmetric v_head_dim Metal vs CPU max_abs={mx}"
    );
}

// Short causal prefill (Lq == Lk ≤ 64) → `sdpa` kernel.
#[test]
fn metal_asym_vhead_prefill_short_matches_cpu() {
    run_case("asym short prefill (sdpa)", 1, 3, 5, 5, 6, 4);
}

// Batched short prefill → `sdpa`, exercises per-batch strides.
#[test]
fn metal_asym_vhead_prefill_batch2_matches_cpu() {
    run_case("asym short prefill B=2 (sdpa)", 2, 4, 7, 7, 8, 5);
}

// Long causal prefill (Lq == Lk > 64) → `sdpa_long` kernel.
#[test]
fn metal_asym_vhead_prefill_long_matches_cpu() {
    run_case("asym long prefill (sdpa_long)", 1, 3, 80, 80, 8, 6);
}

// Decode step (Lq = 1, Lk > 1) → `sdpa_decode_m1` kernel.
#[test]
fn metal_asym_vhead_decode_matches_cpu() {
    run_case("asym decode (sdpa_decode_m1)", 1, 3, 1, 17, 8, 6);
}
