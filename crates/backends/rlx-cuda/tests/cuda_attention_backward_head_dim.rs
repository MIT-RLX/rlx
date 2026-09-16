// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::AttentionBackward` against CPU, sweeping `head_dim`.
//!
//! A Jacobian-lens fit of a Qwen3.5 attention block disagreed with CPU by 0.56
//! relative on CUDA and 1.0 (i.e. output ≈ zero) on ROCm, with a node-level diff
//! putting the first divergence on this op once the forward was fixed. Its
//! `head_dim` is 256, which is larger than most flash-attention backward kernels
//! tile for — so this sweeps head_dim to separate "the kernel is wrong" from
//! "the kernel has an unguarded size limit and silently does nothing".
//!
//! Run with `RLX_CUDA_NO_TF32=1`: TF32 alone costs ~1e-4 relative here and would
//! blur the thing being measured.

#![cfg(target_os = "linux")]

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn attention_graph(b: usize, h: usize, s: usize, d: usize) -> (Graph, Vec<rlx_ir::NodeId>) {
    let f = DType::F32;
    let mut g = Graph::new("attn");
    // Rank-3 `[B, S, H·D]`, which is what a transformer block emits and what
    // each backend's unfuse promotes from. Declaring rank-4 here is ambiguous:
    // Metal reads `[B, S, H, D]` and the CUDA/ROCm promoted path reads
    // `[B, H, S, D]`, so the same graph means different things per backend.
    let q = g.input("q", Shape::new(&[b, s, h * d], f));
    let k = g.input("k", Shape::new(&[b, s, h * d], f));
    let v = g.input("v", Shape::new(&[b, s, h * d], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, h * d], f),
    );
    g.set_outputs(vec![y]);
    (g, vec![q, k, v])
}

fn run(device: Device, b: usize, h: usize, s: usize, d: usize) -> Vec<Vec<f32>> {
    let (fwd, wrt) = attention_graph(b, h, s, d);
    let bwd = grad_with_loss(&fwd, &wrt);
    let n = b * s * h * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
    let dy: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos()).collect();
    let mut g = Session::new(device).compile(bwd);
    g.run(&[("q", &q), ("k", &k), ("v", &v), ("d_output", &dy)])
}

fn check(device: Device, label: &str, b: usize, h: usize, s: usize, d: usize) -> bool {
    let cpu = run(Device::Cpu, b, h, s, d);
    let gpu = run(device, b, h, s, d);
    let mut worst_rel: f32 = 0.0;
    let mut all_zero = true;
    for (a, c) in cpu.iter().zip(&gpu) {
        if c.iter().any(|v| *v != 0.0) {
            all_zero = false;
        }
        let num: f32 = a.iter().zip(c).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum::<f32>().max(f32::MIN_POSITIVE);
        worst_rel = worst_rel.max((num / den).sqrt());
    }
    eprintln!(
        "{label} b{b} h{h} s{s} head_dim{d}: worst relF = {worst_rel:.3e}{}",
        if all_zero {
            "   [GPU OUTPUT IS ALL ZERO]"
        } else {
            ""
        }
    );
    worst_rel < 1e-4
}

/// `head_dim > 128` used to return **exact zeros**: the kernel bailed at its
/// early-return guard having written nothing, so the gradient was silently all
/// zero and training simply learned nothing from these tensors. `attention_bwd.cu`
/// now tiles the one head_dim-sized accumulator, and head_dim 256 (Qwen3.5)
/// agrees with CPU to ~9e-7 like every narrower head.
///
/// No longer ignored — it needs a GPU, and skips itself when there is none.
#[test]
fn attention_backward_matches_cpu_across_head_dim() {
    let (device, label) = if rlx_runtime::is_available(Device::Cuda) {
        (Device::Cuda, "cuda")
    } else if rlx_runtime::is_available(Device::Rocm) {
        (Device::Rocm, "rocm")
    } else {
        eprintln!("skip: no CUDA/ROCm");
        return;
    };
    // Report every size before asserting: the shape of the failure across
    // head_dim is the diagnosis, and stopping at the first one hides it.
    let mut bad = Vec::new();
    for d in [32usize, 64, 128, 256] {
        if !check(device, label, 2, 4, 16, d) {
            bad.push(d);
        }
    }
    assert!(
        bad.is_empty(),
        "{label} AttentionBackward disagrees with CPU at head_dim {bad:?} \
         (passing at the others) — looks like an unguarded tile-size limit"
    );
}
