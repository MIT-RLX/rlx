// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `attention_tiled` (many query rows: 64 per workgroup, K/V tiles in
//! workgroup memory) — wgpu vs CPU over every case the kernel claims: all
//! mask kinds, GQA, soft-cap, fewer queries than keys (absolute positions),
//! ragged query and key tails, head widths 16–128, batch > 1.
//!
//! Each case also runs with `RLX_WGPU_ATTENTION_UNTILED=1` (the per-row
//! kernels) and requires the two wgpu results to differ in the last bits:
//! proof the tiled kernel is the one that ran, so a parity pass cannot come
//! from the old path.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

struct Case {
    name: &'static str,
    batch: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    seq_q: usize,
    seq_k: usize,
    mask: MaskKind,
    softcap: Option<f32>,
}

/// `case(name, batch, heads, kv_heads, head_dim, (seq_q, seq_k), mask, softcap)`.
#[allow(clippy::too_many_arguments)]
fn case(
    name: &'static str,
    batch: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    (seq_q, seq_k): (usize, usize),
    mask: MaskKind,
    softcap: Option<f32>,
) -> Case {
    Case {
        name,
        batch,
        nh,
        nkv,
        hd,
        seq_q,
        seq_k,
        mask,
        softcap,
    }
}

fn wave(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * a + b).sin()).collect()
}

/// Errors relative to `max |cpu|`: `(tiled vs cpu, untiled vs cpu, tiled vs
/// untiled)`.
fn run(case: &Case) -> Option<(f32, f32, f32)> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let Case {
        batch: b,
        nh,
        nkv,
        hd,
        seq_q,
        seq_k,
        ..
    } = *case;
    let q = wave(b * seq_q * nh * hd, 0.011, 0.0);
    let k = wave(b * seq_k * nkv * hd, 0.013, 1.0);
    let v = wave(b * seq_k * nkv * hd, 0.017, 2.0);

    let mut g = Graph::new("attn_tiled");
    let qi = g.input("q", Shape::new(&[b, seq_q, nh * hd], DType::F32));
    let ki = g.input("k", Shape::new(&[b, seq_k, nkv * hd], DType::F32));
    let vi = g.input("v", Shape::new(&[b, seq_k, nkv * hd], DType::F32));
    let mut inputs = vec![qi, ki, vi];
    let mut mask = Vec::new();
    match case.mask {
        MaskKind::Custom => {
            // Key padding: drop every 7th key and the last 3 of batch 1.
            mask = (0..b * seq_k)
                .map(|i| {
                    let (bi, s) = (i / seq_k, i % seq_k);
                    if s % 7 == 3 || (bi == 1 && s + 3 >= seq_k) {
                        0.0
                    } else {
                        1.0
                    }
                })
                .collect();
            inputs.push(g.input("mask", Shape::new(&[b, seq_k], DType::F32)));
        }
        MaskKind::Bias => {
            mask = wave(b * nh * seq_q * seq_k, 0.003, 0.5)
                .into_iter()
                .map(|x| 2.0 * x)
                .collect();
            inputs.push(g.input("mask", Shape::new(&[b, nh, seq_q, seq_k], DType::F32)));
        }
        _ => {}
    }
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: case.mask,
            score_scale: None,
            attn_logit_softcap: case.softcap,
        },
        inputs,
        Shape::new(&[b, seq_q, nh * hd], DType::F32),
    );
    g.set_outputs(vec![y]);

    let exec = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        let mut feed = vec![
            ("q", q.as_slice()),
            ("k", k.as_slice()),
            ("v", v.as_slice()),
        ];
        if !mask.is_empty() {
            feed.push(("mask", mask.as_slice()));
        }
        c.run(&feed).remove(0)
    };
    let cpu = exec(Device::Cpu);
    let tiled = exec(Device::Gpu);
    rlx_ir::env::set("RLX_WGPU_ATTENTION_UNTILED", "1");
    let untiled = exec(Device::Gpu);
    rlx_ir::env::remove("RLX_WGPU_ATTENTION_UNTILED");

    let scale = cpu.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-12);
    let rel = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
            / scale
    };
    assert!(
        tiled.iter().all(|x| x.is_finite()),
        "{}: non-finite output",
        case.name
    );
    let (t, u, p) = (
        rel(&tiled, &cpu),
        rel(&untiled, &cpu),
        rel(&tiled, &untiled),
    );
    eprintln!(
        "{}: vs cpu tiled {t:.3e} / untiled {u:.3e}; tiled vs untiled {p:.3e}",
        case.name
    );
    Some((t, u, p))
}

#[test]
fn wgpu_attention_tiled_matches_cpu() {
    // name, batch, heads, kv heads, head dim, (queries, keys), mask, soft-cap
    #[rustfmt::skip]
    let cases = [
        case("encoder hd64", 1, 6, 6, 64, (200, 200), MaskKind::None, None),
        case("ragged tails", 1, 2, 2, 64, (97, 131), MaskKind::None, None),
        case("cross, k ≫ q", 1, 2, 2, 64, (70, 1031), MaskKind::Custom, None),
        case("hd16", 1, 4, 4, 16, (80, 80), MaskKind::None, None),
        case("hd32", 1, 4, 4, 32, (80, 80), MaskKind::None, None),
        case("hd128", 1, 2, 2, 128, (80, 90), MaskKind::None, None),
        case("causal prefill", 1, 4, 4, 64, (150, 150), MaskKind::Causal, None),
        case("causal, past keys", 1, 4, 4, 64, (40, 120), MaskKind::Causal, None),
        case("sliding window", 1, 4, 4, 64, (130, 130), MaskKind::SlidingWindow(17), None),
        case("padding mask, batch 2", 2, 3, 3, 32, (66, 66), MaskKind::Custom, None),
        case("additive bias", 2, 2, 2, 32, (48, 72), MaskKind::Bias, None),
        case("GQA 8:2", 1, 8, 2, 64, (100, 100), MaskKind::Causal, None),
        case("soft-cap", 1, 4, 4, 64, (90, 90), MaskKind::Causal, Some(5.0)),
    ];
    for case in &cases {
        let Some((tiled, untiled, paths)) = run(case) else {
            return;
        };
        // Reduction-order noise vs the CPU grows with head width (the per-row
        // kernel itself is 2.3e-5 off at hd128); the tiled kernel must stay
        // within that band, not under a fixed bar.
        assert!(
            tiled < (2.0 * untiled).max(2e-5),
            "{}: tiled {tiled:.3e} vs cpu, untiled {untiled:.3e}",
            case.name
        );
        assert!(
            paths > 0.0,
            "{}: tiled and untiled agree to the bit — the tiled kernel did not run",
            case.name
        );
    }
}
