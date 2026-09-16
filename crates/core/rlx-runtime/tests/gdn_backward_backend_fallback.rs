// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A backend that runs `Op::GatedDeltaNet` but not its backward must still be
//! able to compile a gradient — automatically.
//!
//! Eight backends claim the forward; three claim `Op::GatedDeltaNetBackward`
//! (CPU, Metal, MLX). For the other five the gradient walk emits an op they
//! cannot execute, and before this the only escape was setting
//! `RLX_GDN_UNFUSE_FOR_AD=1` *upstream of autodiff* — a global env flag
//! answering a per-backend question, at a point where the caller has usually
//! already built the graph.
//!
//! `rewrite_for_backend` is the seam that already resolves this class of
//! mismatch for every other fused backward op, so the check here is the real
//! path rather than the decomposition in isolation
//! (`rlx-autodiff/tests/gated_delta_net_backward_decompose.rs` covers that):
//! take a backend's actual `supported_ops`, remove the one kind, and require
//! that the result both legalizes and computes the same gradients.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_compile::{legalize_for_backend, rewrite_for_backend};
use rlx_ir::{DType, Graph, Op, OpKind, Shape};
use rlx_runtime::{Device, Session};

const B: usize = 2;
const S: usize = 4;
const H: usize = 2;
const N: usize = 8;

fn hashed(seed: u64, i: usize) -> f32 {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x >> 40) as f32) / 8_388_608.0 - 1.0
}

fn backward_graph() -> Graph {
    let f = DType::F32;
    let bshn = Shape::new(&[B, S, H, N], f);
    let mut g = Graph::new("gdn");
    let q = g.input("q", bshn.clone());
    let k = g.input("k", bshn.clone());
    let v = g.input("v", bshn.clone());
    let gate = g.input("g", Shape::new(&[B, S, H], f));
    let beta = g.input("beta", Shape::new(&[B, S, H], f));
    let y = g.add_node(
        Op::GatedDeltaNet {
            state_size: N,
            carry_state: false,
            gate_per_channel: false,
        },
        vec![q, k, v, gate, beta],
        bshn,
    );
    g.set_outputs(vec![y]);

    grad_with_loss_wrt(
        &g,
        &[
            Wrt::Leaf("q".into()),
            Wrt::Leaf("k".into()),
            Wrt::Leaf("v".into()),
            Wrt::Leaf("g".into()),
            Wrt::Leaf("beta".into()),
        ],
        GradWithLossOptions::STRICT.with_aux(false),
    )
}

fn run(graph: Graph) -> Vec<Vec<f32>> {
    let qkv = B * S * H * N;
    let bsh = B * S * H;
    let q: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(1, i)).collect();
    let k: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(2, i)).collect();
    let v: Vec<f32> = (0..qkv).map(|i| 0.4 * hashed(3, i)).collect();
    let gate: Vec<f32> = (0..bsh).map(|i| -0.4 + 0.15 * hashed(4, i)).collect();
    let beta: Vec<f32> = (0..bsh).map(|i| 0.5 + 0.2 * hashed(5, i)).collect();
    let cot: Vec<f32> = (0..qkv).map(|i| 0.5 + 0.25 * ((i % 7) as f32)).collect();

    let mut compiled = Session::new(Device::Cpu).compile(graph);
    compiled.run(&[
        ("q", &q[..]),
        ("k", &k[..]),
        ("v", &v[..]),
        ("g", &gate[..]),
        ("beta", &beta[..]),
        ("d_output", &cot[..]),
    ])
}

/// Model a CUDA/ROCm/wgpu/TPU/CoreML-shaped backend: everything CPU claims,
/// minus the one fused backward kernel they lack.
fn forward_only_backend_ops() -> Vec<OpKind> {
    let backend = rlx_runtime::backend_for(Device::Cpu).expect("CPU backend must resolve");
    let ops: Vec<OpKind> = backend
        .supported_ops()
        .iter()
        .copied()
        .filter(|k| *k != OpKind::GatedDeltaNetBackward)
        .collect();
    assert!(
        ops.contains(&OpKind::GatedDeltaNet),
        "the modelled backend must still run the forward — otherwise this \
         tests a different fallback"
    );
    ops
}

#[test]
fn rewrite_for_backend_removes_the_unsupported_fused_backward() {
    let supported = forward_only_backend_ops();
    let bwd = backward_graph();
    assert!(
        bwd.nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedDeltaNetBackward { .. })),
        "expected the fused backward op before rewriting"
    );

    let rewritten = rewrite_for_backend(bwd, &supported);
    assert!(
        !rewritten
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedDeltaNetBackward { .. })),
        "rewrite_for_backend left an op the backend cannot execute"
    );
    if let Err(bad) = legalize_for_backend(&rewritten, &supported) {
        let kinds: Vec<OpKind> = bad.into_iter().map(|(_, k)| k).collect();
        panic!("rewritten gradient graph is still illegal for the backend: {kinds:?}");
    }
}

#[test]
fn the_rewritten_gradient_is_numerically_the_same() {
    let supported = forward_only_backend_ops();
    let fused = run(backward_graph());
    let rewritten = run(rewrite_for_backend(backward_graph(), &supported));

    assert_eq!(fused.len(), rewritten.len(), "output count differs");
    let labels = ["y", "dq", "dk", "dv", "dg", "dbeta"];
    let mut worst = 0.0f32;
    for (i, (a, b)) in fused.iter().zip(&rewritten).enumerate() {
        let label = labels.get(i).copied().unwrap_or("?");
        assert_eq!(a.len(), b.len(), "{label} length differs");
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            let d = (x - y).abs();
            worst = worst.max(d);
            assert!(d < 2e-4, "{label}[{j}]: fused {x} vs rewritten {y}");
        }
        let magnitude = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            magnitude > 1e-4,
            "{label} is ~zero (max {magnitude}) — the comparison is vacuous"
        );
    }
    eprintln!("worst |fused - rewritten| = {worst:.2e}");
}
