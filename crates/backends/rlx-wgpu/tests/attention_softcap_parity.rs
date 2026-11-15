// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Op::Attention { attn_logit_softcap: Some(c) }` parity, wgpu vs CPU.
//!
//! Scores become `c · tanh(s / c)` before masking (Gemma 2's attention soft-cap).
//! The wgpu kernel used to drop the field and run uncapped. Scores here reach
//! about ±60 against a cap of 5, so a missing cap changes the output by a wide
//! margin; each case also checks that it does.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn run(
    nh: usize,
    hd: usize,
    seq: usize,
    mask: MaskKind,
    cap: Option<f32>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        eprintln!("skip: wgpu unavailable");
        return None;
    }
    let dim = nh * hd;
    let q: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.021).sin()).collect();
    let k: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.017).cos()).collect();
    let v: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.013).sin()).collect();
    // An additive bias: 0 to attend, large-negative to drop (every other key
    // beyond the first, so no row is fully masked).
    let bias: Vec<f32> = (0..nh * seq * seq)
        .map(|i| {
            if (i % seq) % 2 == 1 && i % seq > 0 {
                -1e4
            } else {
                0.0
            }
        })
        .collect();

    let mut g = Graph::new("attn_softcap");
    let qi = g.input("q", Shape::new(&[1, seq, dim], DType::F32));
    let ki = g.input("k", Shape::new(&[1, seq, dim], DType::F32));
    let vi = g.input("v", Shape::new(&[1, seq, dim], DType::F32));
    let mut inputs = vec![qi, ki, vi];
    if matches!(mask, MaskKind::Bias) {
        inputs.push(g.input("bias", Shape::new(&[1, nh, seq, seq], DType::F32)));
    }
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: mask,
            // Unscaled scores: up to ±hd, well past the cap.
            score_scale: Some(1.0),
            attn_logit_softcap: cap,
        },
        inputs,
        Shape::new(&[1, seq, dim], DType::F32),
    );
    g.set_outputs(vec![y]);
    let exec = |device: Device| -> Vec<f32> {
        let mut c = Session::new(device).compile(g.clone());
        let mut feed: Vec<(&str, &[f32])> = vec![("q", &q), ("k", &k), ("v", &v)];
        if matches!(mask, MaskKind::Bias) {
            feed.push(("bias", &bias));
        }
        c.run(&feed).remove(0)
    };
    Some((exec(Device::Gpu), exec(Device::Cpu)))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn case(nh: usize, hd: usize, seq: usize, mask: MaskKind) {
    let Some((gpu, cpu)) = run(nh, hd, seq, mask, Some(5.0)) else {
        return;
    };
    let (_, uncapped) = run(nh, hd, seq, mask, None).expect("device checked above");
    let err = max_abs(&gpu, &cpu);
    let effect = max_abs(&cpu, &uncapped);
    eprintln!(
        "softcap nh={nh} hd={hd} seq={seq} {mask:?}: wgpu vs cpu {err:.2e}, cap changes output by {effect:.2e}"
    );
    assert!(
        effect > 1e-2,
        "the cap does not matter for this case, so it tests nothing"
    );
    assert!(
        err < 1e-4,
        "wgpu softcap attention differs from CPU by {err}"
    );
}

#[test]
fn softcap_causal_short() {
    case(4, 64, 8, MaskKind::Causal);
}

#[test]
fn softcap_causal_long() {
    case(4, 64, 200, MaskKind::Causal);
}

#[test]
fn softcap_bias_short() {
    case(2, 64, 16, MaskKind::Bias);
}

#[test]
fn softcap_bias_long() {
    case(2, 64, 130, MaskKind::Bias);
}

#[test]
fn softcap_none_mask() {
    case(2, 32, 40, MaskKind::None);
}
