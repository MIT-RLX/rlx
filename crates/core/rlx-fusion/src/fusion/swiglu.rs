// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `swiglu` — extracted from the `fusion` module for navigability (see `mod.rs`).

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

/// Detects the post-`FuseSharedInputMatMul` SwiGLU pattern and replaces it
/// with a single `Op::FusedSwiGLU` node consuming the concatenated matmul.
///
/// Pattern (after `FuseSharedInputMatMul` has fused fc11+fc12 into one mm):
///   %cat   = matmul(%x, concat(%fc11_w, %fc12_w))   ; shape [..., 2N]
///   %up    = narrow(%cat, axis=-1, 0, N)            ; shape [..., N]
///   %gate  = narrow(%cat, axis=-1, N, N)            ; shape [..., N]
///   %silu  = silu(%gate)
///   %out   = mul(%up, %silu)
///
/// Becomes:
///   %out   = fused_swiglu(%cat)
///
/// Saves three kernel launches (two narrows + silu + mul → one kernel) and
/// keeps up/gate resident in registers.
///
/// Single-use guard: only fuses when each intermediate (narrow, narrow, silu)
/// has exactly one consumer. The mul may have any number of consumers.
pub struct FuseSwiGLU;

impl FuseSwiGLU {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        // Scan for Mul nodes whose two inputs match the SwiGLU pattern.
        // Collect rewrites first, then rebuild.
        // up_narrow_id / silu_id / gate_narrow_id are kept for pattern-shape
        // self-documentation even though only the rewrite path reads
        // mul_id / cat_id / out_n.
        #[allow(dead_code)]
        struct Match {
            mul_id: NodeId,
            up_narrow_id: NodeId,
            silu_id: NodeId,
            gate_narrow_id: NodeId,
            cat_id: NodeId,
            out_n: usize,
            gate_first: bool,
        }

        let mut matches: Vec<Match> = Vec::new();
        let mut consumed: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            // Looking for: mul(narrow(cat, 0, n), silu(narrow(cat, n, n)))
            //   — or symmetrically with up/gate swapped.
            if !matches!(node.op, Op::Binary(BinaryOp::Mul)) {
                continue;
            }
            let lhs_id = node.inputs[0];
            let rhs_id = node.inputs[1];
            let lhs = graph.node(lhs_id);
            let rhs = graph.node(rhs_id);

            // Decide which side is silu(gate) — the silu branch.
            let (up_narrow, silu_id, silu_node) =
                if matches!(rhs.op, Op::Activation(Activation::Silu)) {
                    (lhs, rhs_id, rhs)
                } else if matches!(lhs.op, Op::Activation(Activation::Silu)) {
                    (rhs, lhs_id, lhs)
                } else {
                    continue;
                };

            // up side must be a Narrow.
            let (up_axis, up_start, up_len) = match &up_narrow.op {
                Op::Narrow { axis, start, len } => (*axis, *start, *len),
                _ => continue,
            };
            // silu input must be a Narrow.
            let gate_narrow_id = silu_node.inputs[0];
            let gate_narrow = graph.node(gate_narrow_id);
            let (g_axis, g_start, g_len) = match &gate_narrow.op {
                Op::Narrow { axis, start, len } => (*axis, *start, *len),
                _ => continue,
            };

            // Both narrows must come from the same source on the same axis,
            // covering the two halves: (0..N) and (N..2N).
            if up_narrow.inputs[0] != gate_narrow.inputs[0] {
                continue;
            }
            if up_axis != g_axis {
                continue;
            }
            if up_len != g_len {
                continue;
            }
            let n = up_len;
            // Canonical: up @ 0, gate @ N. Swapped (gate-first builders): gate @ 0, up @ N.
            let gate_first = up_start == n && g_start == 0;
            if !(gate_first || (up_start == 0 && g_start == n)) {
                continue;
            }

            // Single-use checks: narrows feed only into silu+mul, silu feeds
            // only into mul. The cat itself can have arbitrary other users.
            if uses.use_count(up_narrow.id) != 1 {
                continue;
            }
            if uses.use_count(gate_narrow_id) != 1 {
                continue;
            }
            if uses.use_count(silu_id) != 1 {
                continue;
            }

            matches.push(Match {
                mul_id: node.id,
                up_narrow_id: up_narrow.id,
                silu_id,
                gate_narrow_id,
                cat_id: up_narrow.inputs[0],
                out_n: n,
                gate_first,
            });
            consumed.insert(up_narrow.id, ());
            consumed.insert(gate_narrow_id, ());
            consumed.insert(silu_id, ());
        }

        if matches.is_empty() {
            return PassResult::unchanged(graph);
        }

        // Rebuild graph, replacing matched mul nodes with FusedSwiGLU.
        let mut rw = Rewriter::new(&graph.name);
        let match_by_mul: HashMap<NodeId, &Match> = matches.iter().map(|m| (m.mul_id, m)).collect();

        for node in graph.nodes() {
            if consumed.contains_key(&node.id) {
                continue;
            }

            if let Some(m) = match_by_mul.get(&node.id) {
                // Output shape = mul's output shape (= [..., N]).
                let out_shape = node.shape.clone();
                debug_assert_eq!(
                    out_shape.dim(out_shape.rank() - 1).unwrap_static(),
                    m.out_n,
                    "FuseSwiGLU: output last dim should be N"
                );
                let fused_id = rw.add_fused(
                    Op::FusedSwiGLU {
                        cast_to: None,
                        gate_first: m.gate_first,
                    },
                    &[m.cat_id],
                    out_shape,
                );
                rw.replace(node.id, fused_id);
                continue;
            }

            rw.copy_node(node);
        }

        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseSwiGLU {
    // Required by construction: both halves of the gate come from `Op::Narrow` slices of one Concat.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Narrow]
    }

    fn name(&self) -> &str {
        "fuse_swiglu"
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

// ── Pass 5: Fuse Attention Block (QKV → SDPA → OutProj) ────────────────
