// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A pass reporting `Unchanged` is load-bearing, so it gets checked against
//! ground truth.
//!
//! The runner acts on the report: it skips whole-graph re-verification and
//! keeps every cached analysis alive. A pass that rewrote the IR but reported
//! `Unchanged` would therefore hand later passes a stale `UseCounts` or
//! `OpKindIndex` describing a graph that no longer exists — a silent
//! wrong-answer bug of exactly the kind that is hardest to trace back.
//!
//! The fusion passes derive their report from `Rewriter::fired()` rather than
//! a hand-maintained flag. This checks that derivation empirically: run each
//! pass over a range of graphs and require
//!
//! ```text
//!   reported Unchanged  ⇒  the output really is structurally identical
//! ```
//!
//! The converse is deliberately *not* required — over-reporting `Changed` is
//! merely wasteful, never wrong.

use rlx_fusion::fusion::{
    FuseAdaLayerNorm, FuseAttentionBlock, FuseConvAffineAct, FuseConvBiasAct, FuseGatedResidual,
    FuseMatMulBiasAct, FuseMatMulResidual, FuseResidualLN, FuseResidualRmsNorm, FuseRmsNormReshape,
    FuseSharedInputMatMul, FuseSwiGLU, FuseSwiGLUDualMatmul, FuseTransformerLayer,
    MarkElementwiseRegions, UnfuseElementwiseRegions,
};
use rlx_fusion::pass::{IRStatus, Pass};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, IgnoreConfig, Op, Shape};

fn all_fusion_passes() -> Vec<&'static dyn Pass> {
    vec![
        &FuseMatMulBiasAct,
        &FuseMatMulResidual,
        &FuseSwiGLU,
        &FuseSwiGLUDualMatmul,
        &FuseTransformerLayer,
        &FuseResidualLN,
        &FuseResidualRmsNorm,
        &FuseRmsNormReshape,
        &FuseSharedInputMatMul,
        &FuseAdaLayerNorm,
        &FuseGatedResidual,
        &FuseAttentionBlock,
        &FuseConvBiasAct,
        &FuseConvAffineAct,
        &MarkElementwiseRegions,
        &UnfuseElementwiseRegions::FOR_CPU,
    ]
}

/// A transformer-ish block: the shape these passes are written against, so
/// several of them genuinely fire on it.
fn transformer_block(layers: usize) -> Graph {
    let mut g = Graph::new("block");
    let d = 32;
    let s = Shape::new(&[2, 8, d], DType::F32);
    let mut h = g.input("x", s.clone());
    for l in 0..layers {
        let w = g.param(format!("w{l}"), Shape::new(&[d, d], DType::F32));
        let b = g.param(format!("b{l}"), Shape::new(&[d], DType::F32));
        let mm = g.matmul(h, w, s.clone());
        let bias = g.binary(BinaryOp::Add, mm, b, s.clone());
        let act = g.add_node(Op::Activation(Activation::Gelu), vec![bias], s.clone());
        let gamma = g.param(format!("g{l}"), Shape::new(&[d], DType::F32));
        let beta = g.param(format!("be{l}"), Shape::new(&[d], DType::F32));
        let ln = g.add_node(
            Op::LayerNorm { eps: 1e-5, axis: 2 },
            vec![act, gamma, beta],
            s.clone(),
        );
        h = g.binary(BinaryOp::Add, ln, h, s.clone());
    }
    g.set_outputs(vec![h]);
    g
}

/// Elementwise-only, so the region-marking passes have something to chew on.
fn elementwise_chain() -> Graph {
    let mut g = Graph::new("chain");
    let s = Shape::new(&[4, 8], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let a = g.binary(BinaryOp::Add, x, y, s.clone());
    let m = g.binary(BinaryOp::Mul, a, x, s.clone());
    let r = g.add_node(Op::Activation(Activation::Relu), vec![m], s.clone());
    let t = g.add_node(Op::Activation(Activation::Tanh), vec![r], s);
    g.set_outputs(vec![t]);
    g
}

/// Nothing any fusion pass is looking for.
fn barren() -> Graph {
    let mut g = Graph::new("barren");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let t = g.add_node(Op::Transpose { perm: vec![0] }, vec![x], s);
    g.set_outputs(vec![t]);
    g
}

fn fixtures() -> Vec<(&'static str, Graph)> {
    vec![
        ("transformer_1", transformer_block(1)),
        ("transformer_3", transformer_block(3)),
        ("elementwise_chain", elementwise_chain()),
        ("barren", barren()),
    ]
}

#[test]
fn unchanged_reports_are_backed_by_identical_ir() {
    let mut verified = 0usize;
    for pass in all_fusion_passes() {
        for (fixture, graph) in fixtures() {
            let result = pass.run_with_status(graph.clone());
            if result.ir_changed == IRStatus::Unchanged {
                assert!(
                    graph.structurally_eq(&result.graph, IgnoreConfig::SEMANTIC),
                    "`{}` reported Unchanged on `{fixture}` but rewrote the IR — the \
                     runner would have kept stale analyses and skipped verification",
                    pass.name()
                );
                verified += 1;
            }
        }
    }
    assert!(
        verified > 0,
        "no pass reported Unchanged on any fixture — the check was vacuous"
    );
}

#[test]
fn changed_reports_are_backed_by_different_ir() {
    // Not a correctness requirement (over-reporting is only wasteful), but a
    // pass that reports Changed while rebuilding an identical graph is the
    // exact waste this work removes — so hold the line where it is already
    // clean, and name any pass that regresses.
    let mut sloppy = Vec::new();
    for pass in all_fusion_passes() {
        for (fixture, graph) in fixtures() {
            let result = pass.run_with_status(graph.clone());
            if result.ir_changed == IRStatus::Changed
                && graph.structurally_eq(&result.graph, IgnoreConfig::SEMANTIC)
            {
                sloppy.push(format!("{} on {fixture}", pass.name()));
            }
        }
    }
    assert!(
        sloppy.is_empty(),
        "these passes reported Changed without changing anything, costing a \
         redundant verification and an analysis-cache flush each: {sloppy:#?}"
    );
}

#[test]
fn a_pass_that_fires_still_reports_changed() {
    // Guard the other direction on a graph FuseMatMulBiasAct is built for:
    // under-reporting here would silently disable the pass's effect downstream.
    let graph = transformer_block(1);
    let result = FuseMatMulBiasAct.run_with_status(graph.clone());
    assert_eq!(result.ir_changed, IRStatus::Changed);
    assert!(!graph.structurally_eq(&result.graph, IgnoreConfig::SEMANTIC));
}

/// The analysis cache only pays if passes actually share it. This pins that:
/// a pipeline run must serve most `UseCounts` requests from cache rather than
/// rebuilding per pass, which was the state before the passes were
/// parameterised over where their counts come from.
#[test]
fn fusion_passes_share_one_use_counts_relation() {
    use rlx_fusion::analysis::AnalysisManager;
    use rlx_fusion::pass::run_passes_tracked;

    let graph = transformer_block(6);
    let passes = all_fusion_passes();
    let mut analyses = AnalysisManager::default();
    let _ = run_passes_tracked(graph, &passes, false, &mut analyses);

    let (hits, computed) = analyses.stats();
    assert!(
        hits > computed,
        "expected the shared relation to be reused across passes, got \
         {hits} hits / {computed} builds — a pass is probably building its own"
    );
}
