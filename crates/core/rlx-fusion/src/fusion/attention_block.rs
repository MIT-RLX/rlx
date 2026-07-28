// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `attention_block` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

/// Fuses `matmul(QKV) → narrow(Q,K,V) → [rope] → attention → matmul(out)`
/// into a single FusedAttentionBlock when batch*seq is small.
///
/// The optimizer auto-detects batch size from graph input shapes. For small
/// inputs (batch*seq ≤ 64), intermediate tensors fit in L1 cache, making a
/// monolithic kernel faster than separate BLAS calls.
///
/// Threshold is configurable via `RLX_FUSE_ATTN_THRESHOLD` (default: 64).
pub struct FuseAttentionBlock;

impl FuseAttentionBlock {
    /// Check if the graph has small enough inputs to benefit from fusion.
    ///
    /// Returns `true` when any 2-D+ input has `dim(0) * dim(1) ≤ threshold`,
    /// where `threshold` defaults to 64 (overridable via
    /// `RLX_FUSE_ATTN_THRESHOLD`). The cutoff matches the L1-cache budget for
    /// keeping Q/K/V resident on CPU and reflects the dispatch-overhead
    /// crossover for small-batch BERT-family encoders on GPU backends.
    pub(crate) fn should_fuse(graph: &Graph) -> bool {
        let threshold: usize = rlx_ir::env::var("RLX_FUSE_ATTN_THRESHOLD")
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        for node in graph.nodes() {
            if let Op::Input { .. } = &node.op
                && node.shape.rank() >= 2
            {
                let d0 = node.shape.dim(0);
                let d1 = node.shape.dim(1);
                if d0.is_static() && d1.is_static() {
                    let b = d0.unwrap_static();
                    let s = d1.unwrap_static();
                    if b * s <= threshold {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Pass for FuseAttentionBlock {
    fn name(&self) -> &str {
        "fuse_attention_block"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Bail when graph input shape is too large to benefit (the L1-resident
        // / single-dispatch win disappears once Q/K/V no longer fit on-chip).
        if !Self::should_fuse(&graph) {
            return graph;
        }

        // We rewrite the chain
        //   hidden ─ FusedMatMulBiasAct(qkv_w, qkv_b) ─ narrow×3 ─ Attention(mask) ─ FusedMatMulBiasAct(out_w, out_b)
        // into a single `Op::FusedAttentionBlock { has_bias: true, has_rope: false }`.
        //
        // Pattern preconditions:
        //   * QKV producer's only consumers are the three narrows (and not a graph
        //     output) — otherwise we'd duplicate compute on un-fuse.
        //   * Each narrow has exactly one consumer (the attention).
        //   * The attention has `MaskKind::Custom` (caller-supplied mask tensor).
        //   * The attention's only consumer is the OutProj `FusedMatMulBiasAct`.
        //   * The OutProj is not a graph output of an *intermediate* block (i.e.
        //     fusing it is safe — its result is the layer's actual output).
        //
        // When any precondition fails we fall back to copying the chain through.

        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }

        // Pre-scan: for each Attention with Custom mask, decide whether the
        // surrounding chain matches. If yes, record the IDs that get folded away.
        struct Match {
            attn_id: NodeId,
            qkv_mm_id: NodeId,
            out_mm_id: NodeId,
            narrows: [NodeId; 3],
            hidden_id: NodeId,
            qkv_w: NodeId,
            qkv_b: NodeId,
            out_w: NodeId,
            out_b: NodeId,
            mask: NodeId,
            num_heads: usize,
            head_dim: usize,
            out_shape: Shape,
        }
        let mut matches: Vec<Match> = Vec::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            let Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            } = &node.op
            else {
                continue;
            };
            // Only the BERT-style mask form (caller-supplied [B, S] tensor),
            // no score scale tweaks, no soft-cap.
            if !matches!(mask_kind, MaskKind::Custom)
                || score_scale.is_some()
                || attn_logit_softcap.is_some()
                || node.inputs.len() != 4
            {
                continue;
            }
            let (q, k, v, mask) = (
                node.inputs[0],
                node.inputs[1],
                node.inputs[2],
                node.inputs[3],
            );

            // All three of Q, K, V must be Narrows on the same parent at
            // start=0,h,2h with len=h on the last (innermost) axis.
            let qn = graph.node(q);
            let kn = graph.node(k);
            let vn = graph.node(v);
            let (qp, q_axis, q_start, q_len) = match narrow_parent(qn) {
                Some(p) => p,
                None => continue,
            };
            let (kp, k_axis, k_start, k_len) = match narrow_parent(kn) {
                Some(p) => p,
                None => continue,
            };
            let (vp, v_axis, v_start, v_len) = match narrow_parent(vn) {
                Some(p) => p,
                None => continue,
            };
            if qp != kp || kp != vp {
                continue;
            }
            let h = num_heads * head_dim;
            let parent_rank = graph.node(qp).shape.rank();
            let last_ax = parent_rank.saturating_sub(1);
            if q_axis != last_ax || k_axis != last_ax || v_axis != last_ax {
                continue;
            }
            if q_len != h || k_len != h || v_len != h {
                continue;
            }
            if q_start != 0 || k_start != h || v_start != 2 * h {
                continue;
            }
            // Narrows must be single-consumer to be safely consumed.
            if graph.use_count(q) != 1
                || graph.use_count(k) != 1
                || graph.use_count(v) != 1
                || is_output.contains_key(&q)
                || is_output.contains_key(&k)
                || is_output.contains_key(&v)
            {
                continue;
            }

            // Parent must be FusedMatMulBiasAct (post-FuseMatMulBiasAct shape).
            let qkv_mm_node = graph.node(qp);
            let (hidden_id, qkv_w, qkv_b) = match fused_mm_bias_none(qkv_mm_node) {
                Some(t) => t,
                None => continue,
            };
            // The QKV MM must have exactly the three narrows as consumers and
            // must not be a graph output itself.
            if graph.use_count(qp) != 3 || is_output.contains_key(&qp) {
                continue;
            }

            // Find the OutProj consumer of the Attention.
            if graph.use_count(node.id) != 1 || is_output.contains_key(&node.id) {
                continue;
            }
            let out_consumer_id = match graph
                .nodes()
                .iter()
                .find(|n| n.inputs.contains(&node.id))
                .map(|n| n.id)
            {
                Some(id) => id,
                None => continue,
            };
            let out_mm_node = graph.node(out_consumer_id);
            let (out_in, out_w, out_b) = match fused_mm_bias_none(out_mm_node) {
                Some(t) if t.0 == node.id => t,
                _ => continue,
            };
            let _ = out_in;

            // All checks passed — record the match.
            matches.push(Match {
                attn_id: node.id,
                qkv_mm_id: qp,
                out_mm_id: out_consumer_id,
                narrows: [q, k, v],
                hidden_id,
                qkv_w,
                qkv_b,
                out_w,
                out_b,
                mask,
                num_heads: *num_heads,
                head_dim: *head_dim,
                out_shape: out_mm_node.shape.clone(),
            });
            fused_away.insert(qp, ());
            fused_away.insert(q, ());
            fused_away.insert(k, ());
            fused_away.insert(v, ());
            fused_away.insert(node.id, ());
            fused_away.insert(out_consumer_id, ());
        }

        if matches.is_empty() {
            return graph;
        }

        // Index matches by the out-projection node id so we can swap it in-place.
        let mut by_out: HashMap<NodeId, &Match> = HashMap::new();
        for m in &matches {
            by_out.insert(m.out_mm_id, m);
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                if let Some(m) = by_out.get(&node.id) {
                    // Make sure all referenced inputs are already in the new graph.
                    rw.ensure_mapped(
                        &graph,
                        &[m.hidden_id, m.qkv_w, m.out_w, m.mask, m.qkv_b, m.out_b],
                    );
                    let fused_id = rw.add_fused(
                        Op::FusedAttentionBlock {
                            num_heads: m.num_heads,
                            head_dim: m.head_dim,
                            has_bias: true,
                            has_rope: false,
                        },
                        &[m.hidden_id, m.qkv_w, m.out_w, m.mask, m.qkv_b, m.out_b],
                        m.out_shape.clone(),
                    );
                    // Wire every old chain node to the new fused id so any
                    // downstream consumer (residual add, LN, etc.) picks it up.
                    rw.replace(m.qkv_mm_id, fused_id);
                    rw.replace(m.narrows[0], fused_id);
                    rw.replace(m.narrows[1], fused_id);
                    rw.replace(m.narrows[2], fused_id);
                    rw.replace(m.attn_id, fused_id);
                    rw.replace(node.id, fused_id);
                }
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}

// ── Pass 5b: Full BERT layer → FusedTransformerLayer ────────────────────
