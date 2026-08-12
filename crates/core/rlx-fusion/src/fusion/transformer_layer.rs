// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `transformer_layer` — extracted from the `fusion` module for navigability (see `mod.rs`).

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

/// Fuses an entire BERT-style transformer layer (attention block + residual+LN +
/// FFN + residual+LN) into one [`Op::FusedTransformerLayer`] node.
///
/// Pattern (after [`FuseMatMulBiasAct`], [`FuseResidualLN`], and
/// [`FuseAttentionBlock`] have run — order matters):
///
/// ```text
///   skip ──┬─→ FusedAttentionBlock(qkv_w, out_w, mask, qkv_b, out_b) ─→ attn_out
///          └─→ FusedResidualLN(attn_out, skip, ln1_g, ln1_b) ─→ h1
///                                                                ├─→ FusedMatMulBiasAct(fc1_w, fc1_b, GeLU) ─→ ffn_int
///                                                                │                                              ↓
///                                                                │           FusedMatMulBiasAct(fc2_w, fc2_b, None) ─→ ffn_out
///                                                                └────────────────────→ FusedResidualLN(ffn_out, h1, ln2_g, ln2_b) ─→ out
/// ```
///
/// All five nodes collapse into a single `FusedTransformerLayer { num_heads,
/// head_dim, intermediate_size, eps1, eps2, activation, has_bias: true }`
/// with the 14-input layout consumed by `rlx-mlx`'s lowering at
/// `rlx-mlx/src/lower.rs:1528`:
/// `[hidden, qkv_w, qkv_b, out_w, out_b, ln1_g, ln1_b, fc1_w, fc1_b, fc2_w, fc2_b, ln2_g, ln2_b, mask]`.
///
/// Threshold is the same as [`FuseAttentionBlock`] (`RLX_FUSE_ATTN_THRESHOLD`,
/// default 64). Backends that don't natively support `FusedTransformerLayer`
/// un-fuse it back to primitives at compile time; backends that do (MLX) can
/// emit one monolithic kernel per layer.
pub struct FuseTransformerLayer;

impl FuseTransformerLayer {
    fn should_fuse(graph: &Graph) -> bool {
        // Same gate as FuseAttentionBlock — single-source of truth for
        // "this graph is small enough for L1-resident block fusion".
        FuseAttentionBlock::should_fuse(graph)
    }
}

impl FuseTransformerLayer {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        if !Self::should_fuse(&graph) {
            return PassResult::unchanged(graph);
        }

        // Graph-output guard: any intermediate we'd absorb must not be an
        // explicit output, otherwise a downstream caller would see the
        // collapsed result instead of the per-stage tensor it expects.
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }

        struct LayerMatch {
            attn_id: NodeId,
            ln1_id: NodeId,
            fc1_id: NodeId,
            fc2_id: NodeId,
            ln2_id: NodeId,
            inputs: [NodeId; 14],
            num_heads: usize,
            head_dim: usize,
            intermediate_size: usize,
            eps1: f32,
            eps2: f32,
            activation: Activation,
            out_shape: Shape,
        }

        let mut matches: Vec<LayerMatch> = Vec::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            // Anchor on each FusedAttentionBlock — every BERT layer starts here.
            let Some((num_heads, head_dim, hidden_id, qkv_w, out_w, mask, qkv_b, out_b)) =
                fused_attn_block_bert(node)
            else {
                continue;
            };
            let attn_id = node.id;
            // Attention's only consumer must be the post-attn FusedResidualLN.
            if uses.use_count(attn_id) != 1 || is_output.contains_key(&attn_id) {
                continue;
            }
            let ln1_id = match graph
                .nodes()
                .iter()
                .find(|n| n.inputs.contains(&attn_id))
                .map(|n| n.id)
            {
                Some(id) => id,
                None => continue,
            };
            let ln1_node = graph.node(ln1_id);
            let Some((ln1_x, ln1_res, ln1_g, ln1_b, eps1)) = fused_residual_ln_no_bias(ln1_node)
            else {
                continue;
            };
            // Order in the residual+LN: x = attn_out, residual = skip (= hidden).
            if ln1_x != attn_id || ln1_res != hidden_id {
                continue;
            }
            // h1 must have exactly 2 consumers (FFN.1 input AND ln2 residual).
            if uses.use_count(ln1_id) != 2 || is_output.contains_key(&ln1_id) {
                continue;
            }

            // Find FFN.1: FusedMatMulBiasAct(h1, fc1_w, fc1_b) with GeLU.
            let mut fc1_candidate: Option<NodeId> = None;
            let mut ln2_candidate: Option<NodeId> = None;
            for cn in graph.nodes() {
                if !cn.inputs.contains(&ln1_id) {
                    continue;
                }
                if fused_mm_bias_act(cn).is_some() && cn.inputs[0] == ln1_id {
                    fc1_candidate = Some(cn.id);
                } else if fused_residual_ln_no_bias(cn).is_some() && cn.inputs[1] == ln1_id {
                    ln2_candidate = Some(cn.id);
                }
            }
            let (Some(fc1_id), Some(ln2_id)) = (fc1_candidate, ln2_candidate) else {
                continue;
            };
            let fc1_node = graph.node(fc1_id);
            let Some((_, fc1_w, fc1_b, activation)) = fused_mm_bias_act(fc1_node) else {
                continue;
            };
            // FFN.1 output → FFN.2 (single consumer).
            if uses.use_count(fc1_id) != 1 || is_output.contains_key(&fc1_id) {
                continue;
            }
            let fc2_id = match graph
                .nodes()
                .iter()
                .find(|n| n.inputs.contains(&fc1_id))
                .map(|n| n.id)
            {
                Some(id) => id,
                None => continue,
            };
            let fc2_node = graph.node(fc2_id);
            // FFN.2 must be FusedMatMulBiasAct with activation=None.
            let Some((fc2_in, fc2_w, fc2_b)) = fused_mm_bias_none(fc2_node) else {
                continue;
            };
            if fc2_in != fc1_id {
                continue;
            }
            if uses.use_count(fc2_id) != 1 || is_output.contains_key(&fc2_id) {
                continue;
            }
            // Final residual+LN: x = ffn_out, residual = h1, gamma/beta + eps2.
            let ln2_node = graph.node(ln2_id);
            let Some((ln2_x, ln2_res, ln2_g, ln2_b, eps2)) = fused_residual_ln_no_bias(ln2_node)
            else {
                continue;
            };
            if ln2_x != fc2_id || ln2_res != ln1_id {
                continue;
            }
            // intermediate_size from fc1_w (`[H, intermediate_size]`).
            let intermediate_size = {
                let s = &graph.node(fc1_w).shape;
                if s.rank() != 2 {
                    continue;
                }
                let d = s.dim(s.rank() - 1);
                if !d.is_static() {
                    continue;
                }
                d.unwrap_static()
            };

            matches.push(LayerMatch {
                attn_id,
                ln1_id,
                fc1_id,
                fc2_id,
                ln2_id,
                inputs: [
                    hidden_id, qkv_w, qkv_b, out_w, out_b, ln1_g, ln1_b, fc1_w, fc1_b, fc2_w,
                    fc2_b, ln2_g, ln2_b, mask,
                ],
                num_heads,
                head_dim,
                intermediate_size,
                eps1,
                eps2,
                activation,
                out_shape: ln2_node.shape.clone(),
            });
            fused_away.insert(attn_id, ());
            fused_away.insert(ln1_id, ());
            fused_away.insert(fc1_id, ());
            fused_away.insert(fc2_id, ());
            fused_away.insert(ln2_id, ());
        }

        if matches.is_empty() {
            return PassResult::unchanged(graph);
        }

        // Index by ln2 (the layer's terminal node) so we know when to emit.
        let mut by_terminal: HashMap<NodeId, &LayerMatch> = HashMap::new();
        for m in &matches {
            by_terminal.insert(m.ln2_id, m);
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                if let Some(m) = by_terminal.get(&node.id) {
                    rw.ensure_mapped(&graph, &m.inputs);
                    let fused_id = rw.add_fused(
                        Op::FusedTransformerLayer {
                            num_heads: m.num_heads,
                            head_dim: m.head_dim,
                            intermediate_size: m.intermediate_size,
                            eps1: m.eps1,
                            eps2: m.eps2,
                            activation: m.activation,
                            has_bias: true,
                        },
                        &m.inputs,
                        m.out_shape.clone(),
                    );
                    rw.replace(m.attn_id, fused_id);
                    rw.replace(m.ln1_id, fused_id);
                    rw.replace(m.fc1_id, fused_id);
                    rw.replace(m.fc2_id, fused_id);
                    rw.replace(node.id, fused_id);
                }
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseTransformerLayer {
    // Required by construction: the layer is anchored on an already-fused attention block.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::FusedAttentionBlock]
    }

    fn name(&self) -> &str {
        "fuse_transformer_layer"
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

// ── PLAN L2: MarkElementwiseRegions ─────────────────────────────────────
//
// Walk the graph and collapse maximal chains of element-wise ops
// (Activation / Cast / Binary / Compare) into a single
// `Op::ElementwiseRegion`. Conditions for inclusion in a chain:
//   - Op is element-wise per `is_elementwise()` (excluding Where which
//     has a 3-input mask semantic that doesn't compose into a single
//     scalar register chain cleanly — keep as separate op for now).
//   - Output shape exactly equals every input shape (no broadcast —
//     broadcast scalar/vector adds register-pattern complexity, defer).
//   - Every intermediate (chain-internal) value has exactly one
//     consumer in the *whole* graph. Multi-consumer values must
//     materialize.
// The chain start can read graph-level inputs / params / earlier-fused
// nodes; the chain end is the last single-consumer or terminal node.
// This is the simplest correct cut — N-ary chain fusion replaces the
// pairwise `fuse_elementwise_chains` pattern in each backend with one
// IR-level pass + a single backend kernel. See PLAN L2.
//
// Fusion boundaries: chains do not extend across inputs whose producer
// satisfies [`rlx_ir::Op::is_fusion_boundary`] (BLAS, Gaussian splat, …).
