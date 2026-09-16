// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::Attention { mask_kind: Causal }` VJP against central finite differences.
//!
//! Isolates the attention op from any surrounding block: a bare
//! `attention_kind(q, k, v, Causal)` graph, an arbitrary cotangent on the
//! output, and `∂(Σ c·out)/∂{q,k,v}` checked numerically. Inputs are scaled so
//! the softmax sits well away from saturation, where finite differences are
//! well-conditioned and a wrong VJP has nowhere to hide.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::op::MaskKind;
use rlx_ir::{DType, Graph, Shape};

const BATCH: usize = 1;
const SEQ: usize = 6;
const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const D: usize = HEADS * HEAD_DIM;
const N: usize = BATCH * SEQ * D;

fn attention_graph(mask: MaskKind) -> Graph {
    let f = DType::F32;
    let shape = Shape::new(&[BATCH, SEQ, D], f);
    let mut g = Graph::new("attn");
    let q = g.input("q", shape.clone());
    let k = g.input("k", shape.clone());
    let v = g.input("v", shape.clone());
    let o = g.attention_kind(q, k, v, HEADS, HEAD_DIM, mask, shape);
    g.set_outputs(vec![o]);
    g
}

/// Deterministic, well-scaled, and distinct per tensor — identical q/k/v would
/// let an index mix-up between them pass.
fn tensor(seed: usize) -> Vec<f32> {
    (0..N)
        .map(|i| 0.3 * (((i * 7 + seed * 13) % 11) as f32 - 5.0) / 5.0)
        .collect()
}

fn check(mask: MaskKind, label: &str) {
    let g = attention_graph(mask);
    let bwd = grad_with_loss_wrt(
        &g,
        &[
            Wrt::Leaf("q".into()),
            Wrt::Leaf("k".into()),
            Wrt::Leaf("v".into()),
        ],
        GradWithLossOptions::STRICT.with_aux(false),
    );

    let (qv, kv, vv) = (tensor(0), tensor(1), tensor(2));
    let cot: Vec<f32> = (0..N).map(|i| 0.5 + 0.25 * ((i % 7) as f32)).collect();

    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    let outs = compiled.run(&[
        ("q", &qv[..]),
        ("k", &kv[..]),
        ("v", &vv[..]),
        ("d_output", &cot[..]),
    ]);
    // [out, dq, dk, dv]
    assert_eq!(outs.len(), 4);

    let mut fwd = rlx::Session::new(rlx::Device::Cpu).compile(attention_graph(mask));
    let weighted = |fwd: &mut rlx::CompiledGraph, q: &[f32], k: &[f32], v: &[f32]| -> f64 {
        fwd.run(&[("q", q), ("k", k), ("v", v)])[0]
            .iter()
            .zip(&cot)
            .map(|(a, b)| (*a as f64) * (*b as f64))
            .sum()
    };

    let eps = 1e-3f32;
    for (t, name) in [(0usize, "dq"), (1, "dk"), (2, "dv")] {
        let grad = &outs[t + 1];
        let mut worst = 0.0f64;
        for i in 0..N {
            let (mut q, mut k, mut v) = (qv.clone(), kv.clone(), vv.clone());
            let target = match t {
                0 => &mut q,
                1 => &mut k,
                _ => &mut v,
            };
            let saved = target[i];
            target[i] = saved + eps;
            let plus = weighted(&mut fwd, &q, &k, &v);
            let target = match t {
                0 => &mut q,
                1 => &mut k,
                _ => &mut v,
            };
            target[i] = saved - eps;
            let minus = weighted(&mut fwd, &q, &k, &v);
            let fd = (plus - minus) / (2.0 * eps as f64);
            let delta = (fd - grad[i] as f64).abs();
            worst = worst.max(delta);
            assert!(
                delta < 1e-3,
                "{label} {name}[{i}]: autodiff {} vs finite-difference {fd}",
                grad[i]
            );
        }
        let magnitude = grad.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(
            magnitude > 1e-3,
            "{label} {name} is ~zero (max {magnitude}) — agreement is vacuous"
        );
        eprintln!("{label} {name}: max |grad| = {magnitude:.5}, worst FD delta = {worst:.2e}");
    }
}

#[test]
fn causal_attention_vjp_matches_finite_differences() {
    check(MaskKind::Causal, "causal");
}

/// Same op without the mask — separates "attention VJP is wrong" from
/// "the causal mask is not applied in the backward".
#[test]
fn unmasked_attention_vjp_matches_finite_differences() {
    check(MaskKind::None, "unmasked");
}
