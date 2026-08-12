// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `matmul_bias_act` — extracted from the `fusion` module for navigability (see `mod.rs`).

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

/// Fuses `matmul → add(bias) → activation` into a single FusedMatMulBiasAct.
///
/// This is the single most impactful fusion — it eliminates two intermediate
/// tensors and three memory passes (matmul write, bias read+write, act read+write)
/// down to one (matmul write with inline bias+activation).
///
/// Also fuses `matmul → add(bias)` without activation.
///
/// Epilogue activations are fused only when every backend can apply them
/// inline with the matmul (today: Gelu and Silu). Other activations — e.g.
/// Exp in qwen35 softplus — stay as separate ops so Metal does not silently
/// drop the epilogue.
pub struct FuseMatMulBiasAct;

impl FuseMatMulBiasAct {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        // Phase 1 — scan only. Measured on a real Qwen3-0.6B prefill graph,
        // this pass attempts 112 fusions and misses every one (the biases are
        // rank-3, not the rank-1 the epilogue kernel needs), then rebuilt all
        // 1104 nodes into an identical copy. Deciding first costs one scan.
        let mut matches: HashMap<NodeId, Fusion> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }
            if let Some(m) = match_matmul_bias_act(&graph, &uses, node) {
                fused_away.insert(m.add_id, ());
                if let Some(aid) = m.act_id {
                    fused_away.insert(aid, ());
                }
                matches.insert(node.id, m);
            }
        }

        if matches.is_empty() {
            return PassResult::unchanged(graph);
        }

        // Phase 2 — rebuild.
        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }
            if let Some(m) = matches.get(&node.id) {
                // Bias may be declared after the matmul in the source graph —
                // copy it early instead of requiring builders to order params
                // first.
                rw.ensure_mapped(&graph, &m.operands);
                let fused_id = rw.add_fused(
                    Op::FusedMatMulBiasAct {
                        activation: m.activation,
                    },
                    &m.operands,
                    m.out_shape.clone(),
                );
                rw.replace(node.id, fused_id);
                rw.replace(m.add_id, fused_id);
                if let Some(aid) = m.act_id {
                    rw.replace(aid, fused_id);
                }
                continue;
            }
            rw.copy_node(node);
        }

        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseMatMulBiasAct {
    // Required by construction: the pattern starts at a MatMul.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::MatMul]
    }

    fn name(&self) -> &str {
        "fuse_matmul_bias_act"
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

// ── Pass 2: Add(residual) + LayerNorm → FusedResidualLN ─────────────────

/// A matched `MatMul → Add(bias) → [Activation]` chain.
struct Fusion {
    operands: [NodeId; 3],
    activation: Option<Activation>,
    add_id: NodeId,
    act_id: Option<NodeId>,
    out_shape: rlx_ir::Shape,
}

/// Does `node` start a fusible matmul epilogue?
///
/// Extracted so the scan and the rebuild ask exactly one question — two copies
/// of a fusion predicate are two things that can drift apart, and the symptom
/// is a fusion that silently stops firing.
fn match_matmul_bias_act(
    graph: &Graph,
    uses: &LazyUseCounts,
    node: &rlx_ir::Node,
) -> Option<Fusion> {
    // Only fuse an F32-weight matmul: the fused epilogue kernel reads the
    // weight (`inputs[1]`) as f32, so a half-width F16/BF16 rhs would be read
    // as f32 garbage. Non-F32 weights must stay a standalone MatMul so they hit
    // the dtype-aware sgemm path (`SgemmF16`/`SgemmBf16`). This is the qwen35
    // vision F16-weight fix (the mm+bias fusion was silently reinterpreting the
    // half weights).
    if !matches!(node.op, Op::MatMul) || graph.shape(node.inputs[1]).dtype() != DType::F32 {
        return None;
    }
    let mm_id = node.id;

    // The matmul's result must be consumed solely by the bias add.
    let mm_users = uses.users(mm_id);
    if mm_users.len() != 1 {
        return None;
    }
    let add_node = graph.node(mm_users[0]);
    let Op::Binary(BinaryOp::Add) = &add_node.op else {
        return None;
    };

    // The non-matmul operand carries the bias, and the epilogue kernel adds it
    // per output channel — so it must be rank-≤1.
    let bias_id = if add_node.inputs[0] == mm_id {
        add_node.inputs[1]
    } else {
        add_node.inputs[0]
    };
    if graph.shape(bias_id).rank() > 1 {
        return None;
    }

    // Optional single-use activation epilogue.
    let mut activation = None;
    let mut act_id = None;
    let add_users = uses.users(add_node.id);
    if add_users.len() == 1 {
        let act_node = graph.node(add_users[0]);
        if let Op::Activation(a) = &act_node.op
            && fusible_mm_bias_epilogue_activation(*a)
        {
            activation = Some(*a);
            act_id = Some(act_node.id);
        }
    }

    let out_shape = match act_id {
        Some(aid) => graph.shape(aid).clone(),
        None => add_node.shape.clone(),
    };
    Some(Fusion {
        operands: [node.inputs[0], node.inputs[1], bias_id],
        activation,
        add_id: add_node.id,
        act_id,
        out_shape,
    })
}
