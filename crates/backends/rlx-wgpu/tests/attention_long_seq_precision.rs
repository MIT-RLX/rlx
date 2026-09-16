// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Attention precision on wgpu as the sequence crosses the kernel-selection
//! boundary.
//!
//! `rlx-brainbert` is 12 heads x 64 head_dim at **seq = 186**, and its relative
//! L2 on wgpu is ~240x CPU's while every isolated op (matmul, LayerNorm, GELU,
//! softmax) is bit-exact or near it. Short sequences take a different attention
//! kernel from long ones, and the earlier cross-backend check only covered
//! seq = 16 — i.e. not the path this model uses.

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let den: f64 = a
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .max(1e-30);
    (num / den).sqrt()
}

fn case_masked(nh: usize, hd: usize, seq: usize, masked: bool) -> Option<f64> {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_runtime::is_available(Device::Gpu)) {
        return None;
    }
    let dim = nh * hd;
    let mut g = Graph::new("attn_long");
    let q = g.input("q", Shape::new(&[1, seq, dim], DType::F32));
    let k = g.input("k", Shape::new(&[1, seq, dim], DType::F32));
    let v = g.input("v", Shape::new(&[1, seq, dim], DType::F32));
    // BrainBERT feeds a padding mask ([batch, key_len]; 1 = valid).
    let mask = g.input("mask", Shape::new(&[1, seq], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: if masked {
                MaskKind::Custom
            } else {
                MaskKind::None
            },
            score_scale: None,
            attn_logit_softcap: None,
        },
        if masked {
            vec![q, k, v, mask]
        } else {
            vec![q, k, v]
        },
        Shape::new(&[1, seq, dim], DType::F32),
    );
    g.set_outputs(vec![y]);
    let qs: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.011).sin()).collect();
    let ks: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.013).cos()).collect();
    let vs: Vec<f32> = (0..seq * dim).map(|i| ((i as f32) * 0.017).sin()).collect();
    // ~15% of positions masked out, as BrainBERT's pretraining does.
    let ms: Vec<f32> = (0..seq)
        .map(|i| if i % 7 == 0 { 0.0 } else { 1.0 })
        .collect();
    let run = |d: Device| -> Vec<f32> {
        let mut c = Session::new(d).compile(g.clone());
        if masked {
            c.run(&[
                ("q", qs.as_slice()),
                ("k", ks.as_slice()),
                ("v", vs.as_slice()),
                ("mask", ms.as_slice()),
            ])
            .remove(0)
        } else {
            c.run(&[
                ("q", qs.as_slice()),
                ("k", ks.as_slice()),
                ("v", vs.as_slice()),
            ])
            .remove(0)
        }
    };
    Some(rel_l2(&run(Device::Cpu), &run(Device::Gpu)))
}

#[test]
fn attention_precision_across_the_seq_boundary() {
    // 12 x 64 is BrainBERT's geometry; 186 is its sequence. 16 and 64 sit on the
    // short-kernel side of the selection gate, 65 and 186 on the long side.
    let mut rows = Vec::new();
    for seq in [16usize, 64, 65, 186] {
        let Some(r) = case_masked(12, 64, seq, false) else {
            eprintln!("skip: wgpu unavailable");
            return;
        };
        eprintln!("attention 12x64 seq={seq:<4} rel_l2 = {r:.3e}");
        rows.push((seq, r));
        if let Some(rm) = case_masked(12, 64, seq, true) {
            eprintln!("attention 12x64 seq={seq:<4} MASKED rel_l2 = {rm:.3e}");
            rows.push((seq, rm));
        }
    }
    let worst = rows
        .iter()
        .cloned()
        .fold((0usize, 0f64), |a, b| if b.1 > a.1 { b } else { a });
    assert!(
        worst.1 < 1e-5,
        "wgpu attention loses precision at seq={} (rel_l2 {:.3e}) — this is the \
         path rlx-brainbert takes",
        worst.0,
        worst.1
    );
}
