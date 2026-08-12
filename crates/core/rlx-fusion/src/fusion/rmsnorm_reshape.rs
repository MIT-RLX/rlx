// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rmsnorm_reshape` — extracted from the `fusion` module for navigability (see `mod.rs`).

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

/// Fuses `rms_norm([…, H]) → reshape([∏leading, H])` into a single
/// `RmsNorm` with the flattened output shape, eliminating a memcpy.
///
/// Matches the Qwen3.5 pre-norm pattern where normalized activations
/// are immediately reshaped to 2-D for matmul.
pub struct FuseRmsNormReshape;

impl FuseRmsNormReshape {
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

        let mut flat_shape: HashMap<NodeId, Shape> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if let Op::RmsNorm { .. } = &node.op {
                if uses.use_count(node.id) != 1 || is_output.contains_key(&node.id) {
                    continue;
                }
                let Some(reshape_id) = sole_consumer(&graph, node.id) else {
                    continue;
                };
                if is_output.contains_key(&reshape_id) {
                    continue;
                }
                let reshape = graph.node(reshape_id);
                if let Op::Reshape { new_shape } = &reshape.op {
                    if let Some(flat) = leading_flatten_shape(&node.shape, new_shape) {
                        flat_shape.insert(node.id, flat);
                        fused_away.insert(reshape_id, ());
                    }
                }
            }
        }

        let mut rw = Rewriter::new(&graph.name);

        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }

            if let Op::RmsNorm { axis, eps, .. } = &node.op {
                if let Some(flat) = flat_shape.get(&node.id) {
                    let Some(reshape_id) = sole_consumer(&graph, node.id) else {
                        rw.copy_node(node);
                        continue;
                    };
                    let fused_id = rw.add_fused(
                        Op::RmsNorm {
                            axis: *axis,
                            eps: *eps,
                        },
                        &node.inputs,
                        flat.clone(),
                    );
                    rw.replace(node.id, fused_id);
                    rw.replace(reshape_id, fused_id);
                    continue;
                }
            }

            rw.copy_node(node);
        }

        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseRmsNormReshape {
    // Required by construction: the pattern requires an RmsNorm.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::RmsNorm]
    }

    fn name(&self) -> &str {
        "fuse_rms_norm_reshape"
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

// ── Pass 3b: Dual MatMul SwiGLU (gate+up before shared-input concat) ─────
