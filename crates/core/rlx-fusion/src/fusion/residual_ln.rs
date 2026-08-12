// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `residual_ln` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::analysis::{AnalysisManager, LazyUseCounts, OpKindIndex, UseCounts};
use crate::pass::{Pass, PassResult};
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;
use rlx_ir::OpKind;

/// Fuses `add(x, residual) → layer_norm` into FusedResidualLN.
///
/// Also detects `add(x, residual) → add(bias) → layer_norm` for the
/// bias variant (used in BERT's output projection).
pub struct FuseResidualLN;

/// `FusedResidualLN` computes `LayerNorm(x + residual)` as one elementwise
/// pass and infers its output shape from operand 0 (`unary_shape(in_shape(0))`).
/// That only holds when both `Add` operands already carry the full output shape.
/// A conditioning-add — e.g. a `[1,1,512]` global bias broadcast over a
/// `[1,83,512]` stream (ChatterBox S3Gen) — has a broadcast operand, so folding
/// it would infer a `[1,1,512]` output and fail the IR verifier. Only fuse when
/// both operands match the `Add` output shape (a true same-shape residual).
fn add_operands_match_output(graph: &Graph, add: &rlx_ir::Node) -> bool {
    if add.inputs.len() != 2 {
        return false;
    }
    let out = &add.shape;
    graph.node(add.inputs[0]).shape.dims() == out.dims()
        && graph.node(add.inputs[1]).shape.dims() == out.dims()
}

impl FuseResidualLN {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        // Graph outputs hold implicit references to their producing
        // nodes that don't show up in any node's `inputs` (use_count
        // walks node inputs only). Treat being-a-graph-output as a
        // use so we don't fuse-away an intermediate the caller still
        // wants to read — this used to silently corrupt multi-block
        // encoders (e.g. SAM 2 stage outputs) by collapsing the
        // residual add of block N into block N+1's LN.
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }
        // Pre-scan: find all Add nodes consumed by LayerNorm
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::LayerNorm { .. } = &node.op {
                let ln_input_id = node.inputs[0];
                let ln_input = graph.node(ln_input_id);
                if matches!(ln_input.op, Op::Binary(BinaryOp::Add))
                    && uses.use_count(ln_input_id) == 1
                    && !is_output.contains_key(&ln_input_id)
                    && add_operands_match_output(&graph, ln_input)
                {
                    fused_away.insert(ln_input_id, ());
                }
            }
        }

        // Nothing matched: hand back the original graph instead of rebuilding
        // it node-for-node into an identical copy. On a graph that merely
        // *contains* this pass's anchor op without the full pattern, that
        // rebuild was the pass's entire cost.
        if fused_away.is_empty() {
            return PassResult::unchanged(graph);
        }
        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::LayerNorm { eps, .. } = &node.op {
                let ln_input_id = node.inputs[0];
                let ln_input = graph.node(ln_input_id);

                if matches!(ln_input.op, Op::Binary(BinaryOp::Add))
                    && fused_away.contains_key(&ln_input_id)
                {
                    let (x_id, residual_id) = (ln_input.inputs[0], ln_input.inputs[1]);
                    let gamma_id = node.inputs[1];
                    let beta_id = node.inputs[2];

                    let fused_id = rw.add_fused(
                        Op::FusedResidualLN {
                            has_bias: false,
                            eps: *eps,
                        },
                        &[x_id, residual_id, gamma_id, beta_id],
                        node.shape.clone(),
                    );

                    rw.replace(ln_input_id, fused_id);
                    rw.replace(node.id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseResidualLN {
    // Required by construction: the pattern requires a LayerNorm.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::LayerNorm]
    }

    fn name(&self) -> &str {
        "fuse_residual_ln"
    }

    fn run(&self, graph: Graph) -> Graph {
        self.fuse_with(graph, None).graph
    }

    fn run_with_status(&self, graph: Graph) -> PassResult {
        if !self.can_fire(&graph) {
            return PassResult::unchanged(graph);
        }
        self.fuse_with(graph, None)
    }

    fn run_with_analyses(&self, graph: Graph, analyses: &mut AnalysisManager) -> PassResult {
        // Answer the trigger check from the shared op-kind index first: a pass
        // whose anchor op is absent must not even pay for the use relation.
        let triggers = self.trigger_kinds();
        if !triggers.is_empty() && !analyses.get::<OpKindIndex>(&graph).contains_any(triggers) {
            return PassResult::unchanged(graph);
        }
        let shared = analyses.get::<UseCounts>(&graph);
        self.fuse_with(graph, Some(shared))
    }
}

// ── Pass 2b: Add(residual) + RmsNorm → FusedResidualRmsNorm ─────────────
