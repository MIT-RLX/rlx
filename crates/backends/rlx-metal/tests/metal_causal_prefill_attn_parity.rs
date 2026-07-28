// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prefill CAUSAL `Op::Attention` (Lq = Lk = S > 1) on Metal vs CPU.
//!
//! Covers `[B,H,S,D]` MaskKind::Causal — historically the Metal 4-D MPSGraph
//! path discarded the causal mask (all-zero). Also covers CFG-style B=2 and
//! Custom bucket masks in `[B,S,H,D]` (Zonos decode).

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn build_causal_prefill_attn(b: usize, h: usize, s: usize, d: usize) -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("causal_prefill_attn");
    let q = g.input("q", Shape::new(&[b, h, s, d], f));
    let k = g.input("k", Shape::new(&[b, h, s, d], f));
    let v = g.input("v", Shape::new(&[b, h, s, d], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, h, s, d], f),
    );
    g.set_outputs(vec![y]);
    g
}

fn assert_metal_cpu_close(label: &str, metal: &[f32], cpu: &[f32], tol: f32) {
    let max_abs = cpu
        .iter()
        .zip(metal.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{label}: max_abs={max_abs:.6}");
    assert!(max_abs < tol, "{label} max_abs={max_abs}");
}

#[test]
fn metal_causal_prefill_attention_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }

    // S > 1 so the causal mask actually constrains multiple query rows.
    let (b, h, s, d) = (1, 8, 96, 64);
    let n = b * h * s * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin()).collect();

    let g = build_causal_prefill_attn(b, h, s, d);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);

    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
    assert_metal_cpu_close("causal prefill B=1", &metal, &cpu, 1e-4);
}

#[test]
fn metal_causal_prefill_attention_batch2_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    let (b, h, s, d) = (2, 4, 32, 32);
    let n = b * h * s * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.02).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.04).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.015).sin()).collect();

    let g = build_causal_prefill_attn(b, h, s, d);
    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c.run(&[("q", &q), ("k", &k), ("v", &v)]).remove(0);
    assert_metal_cpu_close("causal prefill B=2", &metal, &cpu, 1e-4);
}

#[test]
fn metal_custom_bucket_mask_bshd_matches_cpu() {
    if !rlx_runtime::is_available(Device::Metal) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // Decode-shaped: Lq=1, Lk=upper+1, Custom keep-mask (Zonos CFG layout).
    let (b, h, lq, lk, d) = (2, 4, 1usize, 17usize, 32usize);
    let f = DType::F32;
    let mut g = Graph::new("custom_bucket_attn");
    let q = g.input("q", Shape::new(&[b, lq, h, d], f));
    let k = g.input("k", Shape::new(&[b, lk, h, d], f));
    let v = g.input("v", Shape::new(&[b, lk, h, d], f));
    let mask = g.input("mask", Shape::new(&[b, lk], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            mask_kind: MaskKind::Custom,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v, mask],
        Shape::new(&[b, lq, h, d], f),
    );
    g.set_outputs(vec![y]);

    let nq = b * lq * h * d;
    let nk = b * lk * h * d;
    let qv: Vec<f32> = (0..nq).map(|i| ((i as f32) * 0.03).sin()).collect();
    let kv: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.05).cos()).collect();
    let vv: Vec<f32> = (0..nk).map(|i| ((i as f32) * 0.01).sin()).collect();
    // Keep first 8 keys + last (decode "current") — pad the middle.
    let past = 8usize;
    let mut mv = vec![0.0f32; b * lk];
    for bi in 0..b {
        for i in 0..lk {
            let keep = i < past || i + 1 == lk;
            mv[bi * lk + i] = if keep { 1.0 } else { 0.0 };
        }
    }

    let mut m = Session::new(Device::Metal).compile(g.clone());
    let metal = m
        .run(&[("q", &qv), ("k", &kv), ("v", &vv), ("mask", &mv)])
        .remove(0);
    let mut c = Session::new(Device::Cpu).compile(g);
    let cpu = c
        .run(&[("q", &qv), ("k", &kv), ("v", &vv), ("mask", &mv)])
        .remove(0);
    assert_metal_cpu_close("custom bucket B=2 BSHD", &metal, &cpu, 1e-4);
}
