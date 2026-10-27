// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `sdpa_prefill_fa2_d{64,128}` against the CPU backend: every mask kind,
//! both layouts, GQA, cross-attention lengths, ragged tails (lengths that are
//! not multiples of the 32-row / 32-key tiles), batch 2 and a logit softcap.
//! `RLX_METAL_PREFILL_FA2=0` runs the same cases on the old kernel.

#![cfg(target_os = "macos")]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

#[derive(Clone, Copy, Debug)]
struct Case {
    b: usize,
    h: usize,
    kv_h: usize,
    sq: usize,
    sk: usize,
    d: usize,
    mask: MaskKind,
    bhsd: bool,
    softcap: Option<f32>,
}

fn data(n: usize, a: f32, b: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * a).sin() * b).collect()
}

/// Build the graph and its feeds.
fn build(c: Case) -> (Graph, Vec<(&'static str, Vec<f32>)>) {
    let f = DType::F32;
    let mut g = Graph::new("fa2");
    let (qs, ks) = if c.bhsd {
        (
            Shape::new(&[c.b, c.h, c.sq, c.d], f),
            Shape::new(&[c.b, c.kv_h, c.sk, c.d], f),
        )
    } else {
        (
            Shape::new(&[c.b, c.sq, c.h * c.d], f),
            Shape::new(&[c.b, c.sk, c.kv_h * c.d], f),
        )
    };
    let q = g.input("q", qs.clone());
    let k = g.input("k", ks.clone());
    let v = g.input("v", ks);
    let mut feeds = vec![
        ("q", data(c.b * c.sq * c.h * c.d, 0.031, 1.3)),
        ("k", data(c.b * c.sk * c.kv_h * c.d, 0.047, 1.1)),
        ("v", data(c.b * c.sk * c.kv_h * c.d, 0.013, 1.0)),
    ];
    let mut inputs = vec![q, k, v];
    match c.mask {
        MaskKind::Custom => {
            let m = g.input("mask", Shape::new(&[c.b, c.sk], f));
            inputs.push(m);
            // Each batch row keeps a different prefix of keys.
            let mut mk = vec![0.0f32; c.b * c.sk];
            for bi in 0..c.b {
                let keep = c.sk - bi * (c.sk / 3);
                for j in 0..keep {
                    mk[bi * c.sk + j] = 1.0;
                }
            }
            feeds.push(("mask", mk));
        }
        MaskKind::Bias => {
            let m = g.input("mask", Shape::new(&[c.b, c.h, c.sq, c.sk], f));
            inputs.push(m);
            feeds.push(("mask", data(c.b * c.h * c.sq * c.sk, 0.017, 2.0)));
        }
        _ => {}
    }
    let y = g.add_node(
        Op::Attention {
            num_heads: c.h,
            head_dim: c.d,
            v_head_dim: None,
            mask_kind: c.mask,
            score_scale: None,
            attn_logit_softcap: c.softcap,
        },
        inputs,
        qs,
    );
    g.set_outputs(vec![y]);
    (g, feeds)
}

fn run(dev: Device, g: Graph, feeds: &[(&'static str, Vec<f32>)]) -> Vec<f32> {
    let mut s = Session::new(dev).compile(g);
    let refs: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    s.run(&refs).remove(0)
}

fn check(c: Case) {
    let (g, feeds) = build(c);
    let metal = run(Device::Metal, g.clone(), &feeds);
    let cpu = run(Device::Cpu, g, &feeds);
    let worst = metal
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // The CPU softcap uses a polynomial tanh (`rlx_cpu::vmath::tanh_poly`,
    // absolute-error bounded); the Metal kernels use `precise::tanh`. Both
    // Metal kernels differ from the CPU by the same ~3e-3 there.
    let tol = if c.softcap.is_some() { 5e-3 } else { 2e-4 };
    eprintln!("{c:?}: max_abs {worst:.2e}");
    assert!(worst.is_finite() && worst < tol, "{c:?}: max_abs {worst}");
}

fn metal() -> bool {
    !rlx_ir::env::skip_unless_device(
        "metal",
        cfg!(target_os = "macos"),
        rlx_runtime::is_available(Device::Metal),
    )
}

fn base(d: usize) -> Case {
    Case {
        b: 1,
        h: 4,
        kv_h: 4,
        sq: 77,
        sk: 77,
        d,
        mask: MaskKind::None,
        bhsd: false,
        softcap: None,
    }
}

#[test]
fn fa2_masks_and_head_dims_match_cpu() {
    if !metal() {
        return;
    }
    for d in [64, 128] {
        for mask in [
            MaskKind::None,
            MaskKind::Causal,
            MaskKind::Custom,
            MaskKind::Bias,
            MaskKind::SlidingWindow(20),
        ] {
            check(Case { mask, ..base(d) });
        }
    }
}

#[test]
fn fa2_layouts_gqa_and_lengths_match_cpu() {
    if !metal() {
        return;
    }
    let d = 128;
    // BHSD layout, batch 2.
    check(Case {
        b: 2,
        bhsd: true,
        ..base(d)
    });
    // GQA: 8 query heads over 2 KV heads, causal.
    check(Case {
        h: 8,
        kv_h: 2,
        mask: MaskKind::Causal,
        ..base(d)
    });
    // Cross-attention lengths (queries at the end of a longer context).
    check(Case {
        sq: 40,
        sk: 131,
        ..base(d)
    });
    check(Case {
        sq: 40,
        sk: 131,
        mask: MaskKind::Causal,
        ..base(d)
    });
    // Exact multiples of the tiles, and a padding mask over batch 2.
    check(Case {
        b: 2,
        sq: 64,
        sk: 64,
        mask: MaskKind::Custom,
        ..base(d)
    });
    // Logit softcap.
    check(Case {
        softcap: Some(30.0),
        ..base(64)
    });
}

/// FLUX-3-sized attention (batch 2, 24 heads, 1664 tokens, head_dim 128,
/// key mask): parity, and the time per call of the new kernel and of the
/// 8-row kernel it replaces (scalar and MMA), back to back in one process —
/// timings across processes on a shared machine vary more than the kernels.
#[test]
fn fa2_long_sequence_matches_cpu() {
    if !metal() {
        return;
    }
    let c = Case {
        b: 2,
        h: 24,
        kv_h: 24,
        sq: 1664,
        sk: 1664,
        d: 128,
        mask: MaskKind::Custom,
        bhsd: false,
        softcap: None,
    };
    let (g, feeds) = build(c);
    let refs: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let flops = 4.0 * (c.b * c.h) as f64 * (c.sq * c.sk * c.d) as f64;
    let mut first = Vec::new();
    for (label, fa2, mma) in [
        ("fa2", None, None),
        ("fa (scalar)", Some("0"), Some("0")),
        ("fa (mma)", Some("0"), Some("1")),
    ] {
        for (key, val) in [
            ("RLX_METAL_PREFILL_FA2", fa2),
            ("RLX_METAL_PREFILL_FA_MMA", mma),
        ] {
            match val {
                Some(v) => rlx_ir::env::set(key, v),
                None => rlx_ir::env::remove(key),
            }
        }
        let mut s = Session::new(Device::Metal).compile(g.clone());
        let out = s.run(&refs).remove(0);
        let reps = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let _ = s.run(&refs);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "{label}: {:.1} ms/call, {:.2} TFLOPS",
            dt * 1e3,
            flops / dt / 1e12
        );
        if first.is_empty() {
            first = out;
        }
    }
    rlx_ir::env::remove("RLX_METAL_PREFILL_FA2");
    rlx_ir::env::remove("RLX_METAL_PREFILL_FA_MMA");
    let cpu = run(Device::Cpu, g, &feeds);
    let worst = first
        .iter()
        .zip(&cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("long: max_abs {worst:.2e}");
    assert!(worst < 2e-4, "long: max_abs {worst}");
}
