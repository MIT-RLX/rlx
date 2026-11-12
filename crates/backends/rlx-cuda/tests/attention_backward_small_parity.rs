// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::AttentionBackward` against CPU around the `attention_bwd_small` bounds.
//!
//! CUDA routes shapes with seq_q·seq_k ≤ 4096 and seq_k·(head_dim+1) ≤ 2560 to
//! the block-cooperative `attention_bwd_small` kernel and everything else to
//! `attention_bwd`. This sweeps mask kinds and sizes on both sides of those
//! bounds (s64·d32 is the RF encoder shape that motivated the small kernel).

#![cfg(target_os = "linux")]

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn run(device: Device, mask: MaskKind, b: usize, h: usize, s: usize, d: usize) -> Vec<Vec<f32>> {
    let f = DType::F32;
    let mut g = Graph::new("attn");
    let q = g.input("q", Shape::new(&[b, s, h * d], f));
    let k = g.input("k", Shape::new(&[b, s, h * d], f));
    let v = g.input("v", Shape::new(&[b, s, h * d], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: mask,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, h * d], f),
    );
    g.set_outputs(vec![y]);
    let bwd = grad_with_loss(&g, &[q, k, v]);
    let n = b * s * h * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
    let dy: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos()).collect();
    Session::new(device)
        .compile(bwd)
        .run(&[("q", &q), ("k", &k), ("v", &v), ("d_output", &dy)])
}

#[test]
fn attention_backward_matches_cpu_around_small_kernel_bounds() {
    if rlx_ir::env::skip_unless_device("cuda", true, rlx_cuda::is_available()) {
        return;
    }
    let mut bad = Vec::new();
    for mask in [MaskKind::None, MaskKind::Causal, MaskKind::SlidingWindow(5)] {
        // (s, d): small-kernel shapes, then ones just past each bound.
        for (s, d) in [
            (1, 8),
            (7, 24),
            (37, 40),
            (64, 32),
            (64, 39),
            (65, 32),
            (16, 160),
        ] {
            let cpu = run(Device::Cpu, mask, 2, 3, s, d);
            let gpu = run(Device::Cuda, mask, 2, 3, s, d);
            let mut worst_rel: f32 = 0.0;
            for (a, c) in cpu.iter().zip(&gpu) {
                assert_eq!(a.len(), c.len());
                let num: f32 = a.iter().zip(c).map(|(x, y)| (x - y) * (x - y)).sum();
                let den: f32 = a.iter().map(|x| x * x).sum::<f32>().max(f32::MIN_POSITIVE);
                worst_rel = worst_rel.max((num / den).sqrt());
            }
            eprintln!("{mask:?} s{s} d{d}: worst relF = {worst_rel:.3e}");
            if !(worst_rel < 1e-4) {
                bad.push(format!("{mask:?} s{s} d{d}: {worst_rel:.3e}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "AttentionBackward disagrees with CPU: {bad:?}"
    );
}
