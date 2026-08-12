// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `residual_rmsnorm` — extracted from the `fusion` module for navigability (see `mod.rs`).

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

/// Fuses `add(x, residual) → rms_norm` into [`Op::FusedResidualRmsNorm`].
pub struct FuseResidualRmsNorm;

impl FuseResidualRmsNorm {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::RmsNorm { .. } = &node.op {
                let rn_input_id = node.inputs[0];
                let rn_input = graph.node(rn_input_id);
                if matches!(rn_input.op, Op::Binary(BinaryOp::Add))
                    && uses.use_count(rn_input_id) == 1
                    && !is_output.contains_key(&rn_input_id)
                {
                    fused_away.insert(rn_input_id, ());
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::RmsNorm { eps, .. } = &node.op {
                let rn_input_id = node.inputs[0];
                let rn_input = graph.node(rn_input_id);

                if matches!(rn_input.op, Op::Binary(BinaryOp::Add))
                    && fused_away.contains_key(&rn_input_id)
                {
                    let (x_id, residual_id) = (rn_input.inputs[0], rn_input.inputs[1]);
                    let gamma_id = node.inputs[1];
                    let beta_id = node.inputs[2];

                    let fused_id = rw.add_fused(
                        Op::FusedResidualRmsNorm {
                            has_bias: false,
                            eps: *eps,
                        },
                        &[x_id, residual_id, gamma_id, beta_id],
                        node.shape.clone(),
                    );

                    rw.replace(rn_input_id, fused_id);
                    rw.replace(node.id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseResidualRmsNorm {
    // Required by construction: the pattern requires an RmsNorm.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::RmsNorm]
    }

    fn name(&self) -> &str {
        "fuse_residual_rms_norm"
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

// ── Pass 2c: RmsNorm → Reshape(leading flatten) ─────────────────────────
