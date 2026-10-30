// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `sdpa_prefill_fa3_d64` (`RLX_METAL_PREFILL_FA3=1`) against the CPU backend:
//! every mask kind, both layouts, GQA, cross-attention lengths, ragged tails,
//! batch 2 and a logit softcap. Then the time per call against
//! `sdpa_prefill_fa2_d64` at V-JEPA2 ViT-g's shape (22 heads, 8192 tokens,
//! head_dim 64, key-padding mask).
//!
//! One test, because the kernel switch is process-wide: a second test running
//! in parallel would see it flip.

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
fn fa3_matches_cpu_and_times_against_fa2() {
    if !metal() {
        return;
    }
    let d = 64;
    for mode in ["1", "64"] {
        rlx_ir::env::set("RLX_METAL_PREFILL_FA3", mode);
        eprintln!("RLX_METAL_PREFILL_FA3={mode}");
        for mask in [
            MaskKind::None,
            MaskKind::Causal,
            MaskKind::Custom,
            MaskKind::Bias,
            MaskKind::SlidingWindow(20),
        ] {
            check(Case { mask, ..base(d) });
        }
        check(Case {
            b: 2,
            bhsd: true,
            ..base(d)
        });
        check(Case {
            h: 8,
            kv_h: 2,
            mask: MaskKind::Causal,
            ..base(d)
        });
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
        check(Case {
            b: 2,
            sq: 64,
            sk: 64,
            mask: MaskKind::Custom,
            ..base(d)
        });
        check(Case {
            b: 2,
            sq: 64,
            sk: 64,
            mask: MaskKind::Custom,
            bhsd: true,
            ..base(d)
        });
        check(Case {
            sq: 17,
            sk: 33,
            ..base(d)
        });
        check(Case {
            softcap: Some(30.0),
            ..base(d)
        });
    }

    // V-JEPA2 ViT-g: 22 heads x 64, 8192 tokens, all keys valid.
    let c = Case {
        b: 1,
        h: 22,
        kv_h: 22,
        sq: 8192,
        sk: 8192,
        d,
        mask: MaskKind::Custom,
        bhsd: false,
        softcap: None,
    };
    let (g, mut feeds) = build(c);
    for (n, v) in feeds.iter_mut() {
        if *n == "mask" {
            v.iter_mut().for_each(|x| *x = 1.0);
        }
    }
    let refs: Vec<(&str, &[f32])> = feeds.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let flops = 4.0 * (c.b * c.h) as f64 * (c.sq * c.sk * c.d) as f64;
    let mut outs = Vec::new();
    // Both orders: a shared machine drifts between runs.
    for (label, fa3) in [
        ("fa3-64", "64"),
        ("fa3", "1"),
        ("fa2", "0"),
        ("fa2", "0"),
        ("fa3", "1"),
        ("fa3-64", "64"),
    ] {
        rlx_ir::env::set("RLX_METAL_PREFILL_FA3", fa3);
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
        outs.push(out);
    }
    rlx_ir::env::remove("RLX_METAL_PREFILL_FA3");
    // outs: fa3-64, fa3, fa2, ...
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let d64 = max_diff(&outs[0], &outs[2]);
    eprintln!("vjepa2 shape: fa3-64 vs fa2 max_abs {d64:.2e}");
    assert!(d64 < 2e-4 && d64 > 0.0, "fa3-64 vs fa2: max_abs {d64}");
    let diff = outs[1]
        .iter()
        .zip(&outs[2])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("vjepa2 shape: fa3 vs fa2 max_abs {diff:.2e}");
    assert!(
        diff < 2e-4,
        "fa3 vs fa2 at the V-JEPA2 shape: max_abs {diff}"
    );
    // The two kernels round differently (exp2, pre-scaled Q), so identical
    // output would mean the switch did not reach the dispatch.
    assert!(
        diff > 0.0,
        "fa3 and fa2 outputs are bit-identical: RLX_METAL_PREFILL_FA3 was inert"
    );
}
