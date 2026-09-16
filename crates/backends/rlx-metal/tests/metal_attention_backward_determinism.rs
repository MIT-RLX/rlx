// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backward passes must give the same answer twice.
//!
//! A backward that is only reproducible on average is a real problem for anything
//! that *accumulates* gradients across many runs — an averaged-Jacobian lens
//! compounds one block's spread across every block it composes.
//!
//! Both cases here regressed on the same bug. A threadgroup reduction leaves its
//! result in `partial[0]`, every thread reads it, and the kernel then reuses
//! `partial` for a second reduction — with no barrier in between. A threadgroup
//! spans several SIMD groups, which diverge freely, so one group could overwrite
//! `partial[0]` while another was still reading it. It was intermittent (~1 run
//! in 3 here) and it affected `rms_norm_bwd`, `softmax_lastax`,
//! `softmax_lastax_causal`, `softmax_lastax_h`, `layer_norm_bwd`,
//! `group_norm_bwd_input` and both AdaLayerNorm backwards — the softmax one is
//! what the causal attention backward runs, which is why the attention case
//! below caught it. The cross-entropy softmax kernels already had the barrier.

#![cfg(target_os = "macos")]

use rlx_autodiff::grad_with_loss;
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

fn attention_graph(
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    causal: bool,
) -> (Graph, Vec<rlx_ir::NodeId>) {
    let f = DType::F32;
    let mut g = Graph::new("attn");
    let q = g.input("q", Shape::new(&[b, s, h, d], f));
    let k = g.input("k", Shape::new(&[b, s, h, d], f));
    let v = g.input("v", Shape::new(&[b, s, h, d], f));
    let y = g.add_node(
        rlx_ir::Op::Attention {
            num_heads: h,
            head_dim: d,
            v_head_dim: None,
            mask_kind: if causal {
                MaskKind::Causal
            } else {
                MaskKind::None
            },
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q, k, v],
        Shape::new(&[b, s, h, d], f),
    );
    g.set_outputs(vec![y]);
    (g, vec![q, k, v])
}

fn run_backward(b: usize, h: usize, s: usize, d: usize, causal: bool) -> Vec<Vec<f32>> {
    let (fwd, wrt) = attention_graph(b, h, s, d, causal);
    let bwd = grad_with_loss(&fwd, &wrt);
    let n = b * s * h * d;
    let q: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.031).sin()).collect();
    let k: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.017).cos()).collect();
    let v: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.011).sin()).collect();
    let dy: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.007).cos()).collect();
    let mut g = Session::new(Device::Metal).compile(bwd);
    g.run(&[("q", &q), ("k", &k), ("v", &v), ("d_output", &dy)])
}

fn check(b: usize, h: usize, s: usize, d: usize, causal: bool) {
    let first = run_backward(b, h, s, d, causal);
    let second = run_backward(b, h, s, d, causal);
    assert_eq!(
        first.len(),
        second.len(),
        "output count changed between runs"
    );
    for (i, (a, c)) in first.iter().zip(&second).enumerate() {
        let num: f32 = a.iter().zip(c).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum::<f32>().max(f32::MIN_POSITIVE);
        let rel = (num / den).sqrt();
        let worst = a
            .iter()
            .zip(c)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "b{b} h{h} s{s} d{d} causal={causal} output {i}: relF = {rel:.3e}  worst = {worst:.3e}"
        );
        assert!(
            worst == 0.0,
            "attention backward is not reproducible: output {i} differs by {worst:.3e} \
             (relF {rel:.3e}) between two identical runs"
        );
    }
}

#[test]
fn attention_backward_is_reproducible() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    // The shapes a Jacobian-lens fit uses, plus a couple of neighbours: the
    // spread showed up at some sizes and not others, so a single shape would
    // have called this clean.
    check(8, 16, 24, 64, true);
    check(8, 16, 48, 64, true);
    check(8, 16, 48, 64, false);
    check(1, 8, 128, 64, true);
}

fn norm_graph(b: usize, s: usize, h: usize, layer_norm: bool) -> (Graph, Vec<rlx_ir::NodeId>) {
    let f = DType::F32;
    let mut g = Graph::new("norm");
    let x = g.input("x", Shape::new(&[b, s, h], f));
    let gamma = g.param("gamma", Shape::new(&[h], f));
    let beta = g.param("beta", Shape::new(&[h], f));
    let op = if layer_norm {
        rlx_ir::Op::LayerNorm {
            axis: -1,
            eps: 1e-5,
        }
    } else {
        rlx_ir::Op::RmsNorm {
            axis: -1,
            eps: 1e-6,
        }
    };
    let y = g.add_node(op, vec![x, gamma, beta], Shape::new(&[b, s, h], f));
    g.set_outputs(vec![y]);
    (g, vec![x, gamma, beta])
}

fn run_norm_backward(b: usize, s: usize, h: usize, layer_norm: bool) -> Vec<Vec<f32>> {
    let (fwd, wrt) = norm_graph(b, s, h, layer_norm);
    let bwd = grad_with_loss(&fwd, &wrt);
    let n = b * s * h;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.023).sin() * 1.7).collect();
    let dy: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.009).cos()).collect();
    let gamma: Vec<f32> = (0..h).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let beta: Vec<f32> = (0..h).map(|i| (i as f32) * 0.0005).collect();
    let mut g = Session::new(Device::Metal).compile(bwd);
    g.set_param("gamma", &gamma);
    g.set_param("beta", &beta);
    g.run(&[("x", &x), ("d_output", &dy)])
}

/// Normalization backwards are the kernels the barrier bug actually lived in.
///
/// `h` is deliberately varied: the threadgroup is `min(256, h)` threads, so a
/// row wider than 256 exercises the strided load path and a narrow one exercises
/// a small threadgroup. 1536 is not a power of two either, which is the second
/// thing that was wrong — the reductions halved with `tsize / 2` and would drop
/// the odd element at every level.
#[test]
fn norm_backwards_are_reproducible() {
    if rlx_ir::env::skip_unless_device("metal", true, rlx_runtime::is_available(Device::Metal)) {
        eprintln!("skip: Metal unavailable");
        return;
    }
    for (b, s, h) in [(8usize, 48usize, 1024usize), (4, 16, 128), (2, 8, 1536)] {
        for layer_norm in [false, true] {
            let first = run_norm_backward(b, s, h, layer_norm);
            for attempt in 0..4 {
                let again = run_norm_backward(b, s, h, layer_norm);
                for (i, (a, c)) in first.iter().zip(&again).enumerate() {
                    let worst = a
                        .iter()
                        .zip(c)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        worst == 0.0,
                        "{} backward b{b} s{s} h{h}: output {i} differs by {worst:.3e} on \
                         attempt {attempt} — a threadgroup reduction is racing",
                        if layer_norm { "layer_norm" } else { "rms_norm" }
                    );
                }
            }
            eprintln!(
                "{} b{b} s{s} h{h}: reproducible over 5 runs",
                if layer_norm { "layer_norm" } else { "rms_norm" }
            );
        }
    }
}
