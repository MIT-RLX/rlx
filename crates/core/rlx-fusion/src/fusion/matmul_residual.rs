// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `matmul_residual` — fuse `Add(MatMul(a, b), residual)` into
//! [`Op::FusedMatMulResidual`] so a backend can fold the transformer residual
//! add into the matmul's store instead of a separate elementwise-add dispatch.
//! Registered only for backends that claim `OpKind::FusedMatMulResidual`
//! (today: Metal, where the saving matters on a launch-latency-bound decode);
//! everyone else keeps the plain `MatMul` + `Add`.

#![allow(unused_imports)]

use crate::analysis::{AnalysisManager, LazyUseCounts, OpKindIndex, UseCounts};
use crate::graph_rewrite::Rewriter;
use crate::pass::{Pass, PassResult};
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

use super::*;
use rlx_ir::OpKind;

/// Fuses `matmul → add(residual)` into a single [`Op::FusedMatMulResidual`].
///
/// The residual is a **full** `[m, n]` tensor (same shape as the matmul
/// result) — the transformer's `add(skip, o_proj)` / `add(h, down_proj)`. This
/// is deliberately distinct from [`FuseMatMulBiasAct`], which only matches a
/// rank-≤1 broadcast bias; the two never compete for the same `Add`.
pub struct FuseMatMulResidual;

impl FuseMatMulResidual {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        // Phase 1 — scan only. No graph is allocated until something matches.
        let mut matches: HashMap<NodeId, (NodeId, [NodeId; 3], rlx_ir::Shape)> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }
            if let Some(m) = match_matmul_residual(&graph, &uses, node) {
                fused_away.insert(m.0, ());
                matches.insert(node.id, m);
            }
        }

        // Nothing matched: return the original graph rather than rebuilding it
        // node-for-node into an identical copy.
        if matches.is_empty() {
            return PassResult::unchanged(graph);
        }

        // Phase 2 — rebuild.
        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }
            if let Some((add_id, operands, out_shape)) = matches.get(&node.id) {
                rw.ensure_mapped(&graph, operands);
                let fused_id = rw.add_fused(Op::FusedMatMulResidual, operands, out_shape.clone());
                rw.replace(node.id, fused_id);
                rw.replace(*add_id, fused_id);
                continue;
            }
            rw.copy_node(node);
        }

        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseMatMulResidual {
    // Required by construction: the pattern starts at a MatMul.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::MatMul]
    }

    fn name(&self) -> &str {
        "fuse_matmul_residual"
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

/// Does `node` start a fusible `MatMul → Add(residual)` pair?
///
/// Returns `(add_id, [lhs, rhs, residual], out_shape)`. Extracted so the
/// cheap pre-scan and the rebuild ask exactly the same question — duplicating
/// the predicate would let the two drift, which is how a fusion silently stops
/// firing.
fn match_matmul_residual(
    graph: &Graph,
    uses: &LazyUseCounts,
    node: &rlx_ir::Node,
) -> Option<(NodeId, [NodeId; 3], rlx_ir::Shape)> {
    if !matches!(node.op, Op::MatMul) {
        return None;
    }
    let mm_id = node.id;
    // The matmul must feed ONLY the add (its result is consumed solely by the
    // residual). The add's own output may have any number of users (skip
    // stream + norm) — that is fine.
    let mm_users = uses.users(mm_id);
    if mm_users.len() != 1 {
        return None;
    }
    let add_node = graph.node(mm_users[0]);
    let Op::Binary(BinaryOp::Add) = &add_node.op else {
        return None;
    };
    let residual_id = if add_node.inputs[0] == mm_id {
        add_node.inputs[1]
    } else {
        add_node.inputs[0]
    };

    // Only fuse a genuine elementwise residual: the added tensor must match the
    // matmul output shape exactly (no broadcast) and be rank > 1 (rank-≤1 is
    // the bias fusion's domain). The residual-epilogue kernel is f32-only
    // (output, residual AND weight) — notably an f16-resident weight routes to
    // the half-precision gemv instead.
    let mm_shape = graph.shape(mm_id);
    let res_shape = graph.shape(residual_id);
    let rhs_shape = graph.shape(node.inputs[1]);
    let same_shape = mm_shape.dtype() == DType::F32
        && res_shape.dtype() == DType::F32
        && rhs_shape.dtype() == DType::F32
        && mm_shape.rank() == res_shape.rank()
        && mm_shape.rank() > 1
        && (0..mm_shape.rank()).all(|d| mm_shape.dim(d) == res_shape.dim(d));
    if !same_shape {
        return None;
    }
    Some((
        add_node.id,
        [node.inputs[0], node.inputs[1], residual_id],
        add_node.shape.clone(),
    ))
}
