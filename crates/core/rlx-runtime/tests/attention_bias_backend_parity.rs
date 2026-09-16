// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-backend parity for `Op::Attention` with `MaskKind::Bias` — the
//! additive per-head `[B, H, Sq, Sk]` position bias.
//!
//! rlx-metal silently DROPPED this bias for short sequences: its long/flash
//! kernels all branched on `mask_kind == 3u`, but `sdpa` / `sdpa_simd` /
//! `sdpa_simd_h16` / `sdpa_h` handled only causal, padding and sliding-window,
//! so a model with a learned position bias and `seq <= 64` quietly got plain
//! attention. `rlx-tribev2-audio` (W2v-BERT, `position_embeddings_type =
//! "relative_key"`, 16 frames) came out at cos 0.64 while every other backend
//! was exact.
//!
//! The geometry below is that model's: 16 heads, head_dim 64, seq 16 — short
//! enough to miss every fast path — plus a small shape and a batched one.

#![allow(dead_code)]

use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_runtime::{Device, Session};

mod common;

const CASES: &[(usize, usize, usize, usize)] = &[
    (1, 4, 32, 8),   // small
    (1, 16, 64, 16), // W2v-BERT
    (2, 16, 64, 16), // batched
];

fn tensors(b: usize, nh: usize, hd: usize, seq: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = nh * hd;
    let q = (0..b * seq * dim)
        .map(|i| ((i as f32) * 0.011).sin())
        .collect();
    let k = (0..b * seq * dim)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let v = (0..b * seq * dim)
        .map(|i| ((i as f32) * 0.017).sin())
        .collect();
    // Shaw-style relative bias: a function of (qi - ki), clamped to an
    // asymmetric window like W2v-BERT's left 64 / right 8, plus a per-head term
    // so a kernel that collapses the head axis is caught too.
    let bias = (0..b * nh * seq * seq)
        .map(|i| {
            let ki = (i % seq) as i64;
            let qi = ((i / seq) % seq) as i64;
            let h = (i / (seq * seq)) % nh;
            let d = (qi - ki).clamp(-64, 8) as f32;
            (d * 0.05 + h as f32 * 0.01).tanh()
        })
        .collect();
    (q, k, v, bias)
}

fn run(device: Device, b: usize, nh: usize, hd: usize, seq: usize) -> Vec<f32> {
    let dim = nh * hd;
    let mut g = Graph::new("attn_bias");
    let qi = g.input("q", Shape::new(&[b, seq, dim], DType::F32));
    let ki = g.input("k", Shape::new(&[b, seq, dim], DType::F32));
    let vi = g.input("v", Shape::new(&[b, seq, dim], DType::F32));
    let bi = g.input("bias", Shape::new(&[b, nh, seq, seq], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::Bias,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi, bi],
        Shape::new(&[b, seq, dim], DType::F32),
    );
    g.set_outputs(vec![y]);
    let (q, k, v, bias) = tensors(b, nh, hd, seq);
    Session::new(device)
        .compile(g)
        .run(&[("q", &q), ("k", &k), ("v", &v), ("bias", &bias)])
        .remove(0)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// The bias must actually change the output — otherwise a backend that ignores
/// it entirely would still "match" a reference that also ignored it.
#[test]
fn the_bias_is_not_a_no_op_on_cpu() {
    let (b, nh, hd, seq) = CASES[1];
    let with = run(Device::Cpu, b, nh, hd, seq);
    let dim = nh * hd;
    let mut g = Graph::new("attn_nobias");
    let qi = g.input("q", Shape::new(&[b, seq, dim], DType::F32));
    let ki = g.input("k", Shape::new(&[b, seq, dim], DType::F32));
    let vi = g.input("v", Shape::new(&[b, seq, dim], DType::F32));
    let y = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind: MaskKind::None,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![qi, ki, vi],
        Shape::new(&[b, seq, dim], DType::F32),
    );
    g.set_outputs(vec![y]);
    let (q, k, v, _) = tensors(b, nh, hd, seq);
    let without = Session::new(Device::Cpu)
        .compile(g)
        .run(&[("q", &q), ("k", &k), ("v", &v)])
        .remove(0);
    let delta = max_abs(&with, &without);
    assert!(
        delta > 1e-3,
        "the bias barely moves the output ({delta:.3e}); this test would not \
         catch a backend that discards it"
    );
    eprintln!("bias moves the CPU output by {delta:.3e}");
}

macro_rules! backend_parity {
    ($name:ident, $feat:meta, $dev:expr) => {
        #[test]
        #[$feat]
        fn $name() {
            let _gpu = common::serialize_gpu();
            if common::skip_unless($dev) {
                eprintln!("skip: {:?} unavailable", $dev);
                return;
            }
            for &(b, nh, hd, seq) in CASES {
                let cpu = run(Device::Cpu, b, nh, hd, seq);
                let dev = run($dev, b, nh, hd, seq);
                let err = max_abs(&cpu, &dev);
                eprintln!(
                    "{:?} attn+bias b={b} heads={nh} hd={hd} seq={seq}: max_abs={err:.3e}",
                    $dev
                );
                assert!(
                    err < 1e-4,
                    "{:?} attention+bias parity failed at b={b} heads={nh} hd={hd} \
                     seq={seq}: {err:.3e}",
                    $dev
                );
            }
        }
    };
}

backend_parity!(
    attn_bias_metal,
    cfg(all(feature = "metal", target_os = "macos")),
    Device::Metal
);
backend_parity!(
    attn_bias_mlx,
    cfg(all(feature = "mlx", target_os = "macos")),
    Device::Mlx
);
backend_parity!(attn_bias_gpu, cfg(feature = "gpu"), Device::Gpu);
backend_parity!(attn_bias_cuda, cfg(feature = "cuda"), Device::Cuda);
backend_parity!(attn_bias_rocm, cfg(feature = "rocm"), Device::Rocm);
