// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! Prefill SLIDING-WINDOW `Op::Attention` (Gemma/Mistral-style) on Metal vs CPU.
//!
//! `MaskKind::SlidingWindow(w)` = causal **and** lookback ≤ `w`. The Metal
//! native sdpa thunk handles this directly (`mask_kind == 4`); the MPSGraph
//! fast-path intentionally bails to it (the MPSGraph custom-mask attention has a
//! documented slice-view bug). This guards that the windowed path on Metal
//! matches the CPU reference — and that for `w ≥ S` it degenerates to plain
//! causal.

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_sliding(b: usize, h: usize, s: usize, d: usize, window: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("sliding_attn");
    let q = g.input("q", Shape::new(&[b, h, s, d], f));
    let k = g.input("k", Shape::new(&[b, h, s, d], f));
    let v = g.input("v", Shape::new(&[b, h, s, d], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            mask_kind: MaskKind::SlidingWindow(window),
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, h, s, d], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn run_both(g: Graph, q: &[f32], k: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("q", q), ("k", k), ("v", v)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("q", q), ("k", k), ("v", v)]).remove(0);
    (metal, cpu)
}

#[test]
fn metal_sliding_window_attention_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (b, h, s, d) = (1usize, 8usize, 96usize, 64usize);
    let n = b * h * s * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();

    // A genuine window (< S) so the lookback actually constrains rows.
    let (metal, cpu) = run_both(build_sliding(b, h, s, d, 24), &q, &k, &v);
    let max_abs = cpu
        .iter()
        .zip(&metal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("sliding-window attn (w=24): max_abs={max_abs:.6}");
    assert!(max_abs < 1e-4, "sliding-window attention max_abs={max_abs}");

    // w ≥ S must equal plain causal (full lookback).
    let (metal_full, cpu_full) = run_both(build_sliding(b, h, s, d, s + 8), &q, &k, &v);
    let max_full = cpu_full
        .iter()
        .zip(&metal_full)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_full < 1e-4, "wide-window attention max_abs={max_full}");
}
