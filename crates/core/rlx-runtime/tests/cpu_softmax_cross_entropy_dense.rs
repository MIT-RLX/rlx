// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::SoftmaxCrossEntropy` — the dense / soft-label classification loss.
//!
//! `loss[n] = logsumexp(logits[n]) - Σ_c targets[n,c]·logits[n,c]`.
//!
//! Covers three things on the CPU backend (the native fused thunk):
//!   1. Forward values match a hand-rolled numerically-stable reference.
//!   2. A one-hot target row reproduces the sparse
//!      `SoftmaxCrossEntropyWithLogits` loss exactly.
//!   3. The autodiff gradient w.r.t. logits matches both the analytic
//!      `(softmax(logits) - targets) · d_loss` and central finite
//!      differences.

#![cfg(feature = "cpu")]

use rlx_ir::op::ReduceOp;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};
use rlx_opt::autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

const N: usize = 4;
const C: usize = 5;

/// Forward graph returning the per-row `[N]` loss. `logits` is declared
/// as a param (so the gradient graph can target it); `targets` is a
/// runtime input.
fn build_per_row() -> (Graph, NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("sce_dense_per_row");
    let logits = g.param("logits", Shape::new(&[N, C], f));
    let targets = g.input("targets", Shape::new(&[N, C], f));
    let per = g.softmax_cross_entropy(logits, targets);
    g.set_outputs(vec![per]);
    (g, logits)
}

/// Forward graph returning the scalar mean loss.
fn build_mean() -> (Graph, NodeId) {
    let f = DType::F32;
    let mut g = Graph::new("sce_dense_mean");
    let logits = g.param("logits", Shape::new(&[N, C], f));
    let targets = g.input("targets", Shape::new(&[N, C], f));
    let per = g.softmax_cross_entropy(logits, targets);
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0],
            keep_dim: false,
        },
        vec![per],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    (g, logits)
}

/// Numerically-stable per-row reference.
fn ref_loss(logits: &[f32], targets: &[f32]) -> Vec<f32> {
    (0..N)
        .map(|n| {
            let row = &logits[n * C..(n + 1) * C];
            let trow = &targets[n * C..(n + 1) * C];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            let lse = m + sum.ln();
            let dot: f32 = (0..C).map(|c| trow[c] * row[c]).sum();
            lse - dot
        })
        .collect()
}

fn softmax_row(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
}

fn sample_logits() -> Vec<f32> {
    // Spread across positive / negative so the max-subtract path matters.
    (0..N * C)
        .map(|i| 0.4 * (i as f32) - 1.3 * ((i % 3) as f32) - 0.7)
        .collect()
}

/// Rows that sum to 1 (a valid soft-label distribution).
fn sample_targets() -> Vec<f32> {
    let mut t = vec![0f32; N * C];
    for n in 0..N {
        let raw: Vec<f32> = (0..C).map(|c| 0.2 + 0.5 * (((n + c) % 4) as f32)).collect();
        let s: f32 = raw.iter().sum();
        for c in 0..C {
            t[n * C + c] = raw[c] / s;
        }
    }
    t
}

#[test]
fn forward_matches_reference() {
    let (g, _logits) = build_per_row();
    let logits = sample_logits();
    let targets = sample_targets();

    let session = Session::new(Device::Cpu);
    let mut compiled = session.compile(g);
    compiled.set_param("logits", &logits);
    let outs = compiled.run(&[("targets", &targets)]);

    let got = &outs[0];
    let want = ref_loss(&logits, &targets);
    assert_eq!(got.len(), N, "per-row loss length");
    for n in 0..N {
        assert!(
            (got[n] - want[n]).abs() < 1e-5,
            "loss[{n}]: got {} want {}",
            got[n],
            want[n]
        );
    }
}

#[test]
fn lowered_primitives_match_native_thunk() {
    // The GPU backends (CUDA/Metal/Vulkan/WGPU/...) don't have a native
    // dense-SCE kernel — `rewrite_for_backend` rewrites the op to
    // primitives via `LowerSoftmaxCrossEntropy`. Run that lowered graph
    // on the CPU and confirm it reproduces the fused thunk to the bit.
    use rlx_fusion::LowerSoftmaxCrossEntropy;
    use rlx_fusion::pass::Pass;

    let logits = sample_logits();
    let targets = sample_targets();

    let (g, _) = build_per_row();
    let lowered = LowerSoftmaxCrossEntropy.run(g);
    assert!(
        !lowered
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::SoftmaxCrossEntropy)),
        "dense SCE op should be gone after lowering"
    );

    let mut compiled = Session::new(Device::Cpu).compile(lowered);
    compiled.set_param("logits", &logits);
    let got = compiled.run(&[("targets", &targets)]);

    let want = ref_loss(&logits, &targets);
    for n in 0..N {
        assert!(
            (got[0][n] - want[n]).abs() < 1e-5,
            "lowered loss[{n}]: got {} want {}",
            got[0][n],
            want[n]
        );
    }
}

#[test]
fn one_hot_target_matches_sparse_loss() {
    let logits = sample_logits();

    // One-hot targets: class index n % C for row n.
    let labels: Vec<f32> = (0..N).map(|n| (n % C) as f32).collect();
    let mut targets = vec![0f32; N * C];
    for n in 0..N {
        targets[n * C + (n % C)] = 1.0;
    }

    let session = Session::new(Device::Cpu);

    // Dense path.
    let (gd, _) = build_per_row();
    let mut cd = session.compile(gd);
    cd.set_param("logits", &logits);
    let dense = cd.run(&[("targets", &targets)]);

    // Sparse path.
    let f = DType::F32;
    let mut gs = Graph::new("sce_sparse");
    let lg = gs.param("logits", Shape::new(&[N, C], f));
    let lb = gs.input("labels", Shape::new(&[N], f));
    let per = gs.softmax_cross_entropy_with_logits(lg, lb);
    gs.set_outputs(vec![per]);
    let mut cs = session.compile(gs);
    cs.set_param("logits", &logits);
    let sparse = cs.run(&[("labels", &labels)]);

    for n in 0..N {
        assert!(
            (dense[0][n] - sparse[0][n]).abs() < 1e-5,
            "row {n}: dense {} vs sparse {}",
            dense[0][n],
            sparse[0][n]
        );
    }
}

#[test]
fn grad_matches_analytic_and_finite_differences() {
    let logits = sample_logits();
    let targets = sample_targets();

    // Autodiff gradient of the mean loss w.r.t. logits.
    let (g, params) = {
        let (g, lid) = build_mean();
        (g, vec![lid])
    };
    let bwd_g = grad_with_loss(&g, &params);
    let session = Session::new(Device::Cpu);
    let mut bwd = session.compile(bwd_g);
    bwd.set_param("logits", &logits);
    let d_output = vec![1.0f32];
    let outs = bwd.run(&[("targets", &targets), ("d_output", &d_output)]);
    let ad = &outs[1]; // outs[0] = loss, outs[1] = grad(logits)
    assert_eq!(ad.len(), N * C, "logits grad length");

    // Analytic: dlogits[n,c] = (softmax(logits[n])[c] - targets[n,c]) / N.
    for n in 0..N {
        let sm = softmax_row(&logits[n * C..(n + 1) * C]);
        for c in 0..C {
            let want = (sm[c] - targets[n * C + c]) / (N as f32);
            let got = ad[n * C + c];
            assert!(
                (got - want).abs() < 1e-5,
                "analytic grad[{n},{c}]: got {got} want {want}",
            );
        }
    }

    // Central finite differences on the scalar mean loss.
    let mean_loss = |lg: &[f32]| -> f32 {
        let (g, _) = build_mean();
        let mut c = Session::new(Device::Cpu).compile(g);
        c.set_param("logits", lg);
        c.run(&[("targets", &targets)])[0][0]
    };
    let eps = 1e-3f32;
    for idx in 0..N * C {
        let mut lp = logits.clone();
        let mut lm = logits.clone();
        lp[idx] += eps;
        lm[idx] -= eps;
        let fd = (mean_loss(&lp) - mean_loss(&lm)) / (2.0 * eps);
        let abs_err = (fd - ad[idx]).abs();
        let rel_err = abs_err / fd.abs().max(1e-6);
        assert!(
            abs_err < 5e-3 || rel_err < 5e-3,
            "FD grad[{idx}]: autodiff {} vs FD {fd} (abs {abs_err:e}, rel {rel_err:e})",
            ad[idx],
        );
    }
}
