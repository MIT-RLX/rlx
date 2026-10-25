// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `swiglu_dual` — extracted from the `fusion` module for navigability (see `mod.rs`).

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

/// Fuses the common LLM FFN pattern in one rewrite:
///   gate = matmul(x, wg); up = matmul(x, wu); out = mul(silu(gate), up)
///
/// Becomes:
///   cat = matmul(x, concat(wu, wg))   // up weights first for kernel layout
///   out = fused_swiglu(cat)
///
/// Eliminates two `[..., N]` matmul outputs plus a silu buffer — the
/// largest memory win on transformer FFN blocks.
pub struct FuseSwiGLUDualMatmul;

impl FuseSwiGLUDualMatmul {
    fn match_dual_swiglu(
        graph: &Graph,
        uses: &LazyUseCounts,
        mul_node: &Node,
    ) -> Option<(NodeId, NodeId, NodeId, NodeId, NodeId)> {
        if !matches!(mul_node.op, Op::Binary(BinaryOp::Mul)) {
            return None;
        }
        let lhs = graph.node(mul_node.inputs[0]);
        let rhs = graph.node(mul_node.inputs[1]);
        let (up_mm, silu_id, silu_node) = if matches!(rhs.op, Op::Activation(Activation::Silu)) {
            (lhs, mul_node.inputs[1], rhs)
        } else if matches!(lhs.op, Op::Activation(Activation::Silu)) {
            (rhs, mul_node.inputs[0], lhs)
        } else {
            return None;
        };
        if !matches!(up_mm.op, Op::MatMul) {
            return None;
        }
        let gate_mm = graph.node(silu_node.inputs[0]);
        if !matches!(gate_mm.op, Op::MatMul) {
            return None;
        }
        if up_mm.inputs[0] != gate_mm.inputs[0] {
            return None;
        }
        // The fused node reads `Concat(up_w, gate_w)`, an activation of the
        // weights' dtype; Metal and wgpu/Vulkan widen a BF16 activation to f32
        // and their Concat misread packed BF16 params (see
        // `FuseSharedInputMatMul`). Same dtype only, and not BF16.
        let (wu_dt, wg_dt) = (
            graph.shape(up_mm.inputs[1]).dtype(),
            graph.shape(gate_mm.inputs[1]).dtype(),
        );
        if wu_dt != wg_dt || wu_dt == rlx_ir::DType::BF16 {
            return None;
        }
        // `[K, N]` weights only: a batched `[B, K, N]` pair (an expert per
        // batch row) concatenates along another axis than the one below.
        if graph.shape(up_mm.inputs[1]).rank() != 2 || graph.shape(gate_mm.inputs[1]).rank() != 2 {
            return None;
        }
        // All three of these are absorbed into the fused node below, so all
        // three must be safe to delete: exactly one consumer and not exported.
        // Checking only the silu left the two matmuls to be dropped even when
        // something else still needed them — invisible in an ordinary model
        // graph, but a graph that publishes an interior activation as an output
        // (`split_vjp`'s save half, an instrumentation tap) loses the value and
        // `Rewriter::finish` panics mapping the outputs.
        if !uses.has_single_use(silu_id)
            || !uses.has_single_use(gate_mm.id)
            || !uses.has_single_use(up_mm.id)
        {
            return None;
        }
        Some((mul_node.id, gate_mm.id, up_mm.id, up_mm.inputs[0], silu_id))
    }
}

impl FuseSwiGLUDualMatmul {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        let mut matches: Vec<(NodeId, NodeId, NodeId, NodeId, NodeId)> = Vec::new();
        let mut consumed: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if let Some((mul_id, gate_mm, up_mm, _, silu_id)) =
                Self::match_dual_swiglu(&graph, &uses, node)
            {
                matches.push((mul_id, gate_mm, up_mm, graph.node(up_mm).inputs[0], silu_id));
                consumed.insert(gate_mm, ());
                consumed.insert(up_mm, ());
                consumed.insert(silu_id, ());
            }
        }

        if matches.is_empty() {
            return PassResult::unchanged(graph);
        }

        let match_by_mul: HashMap<NodeId, (NodeId, NodeId, NodeId)> = matches
            .into_iter()
            .map(|(mul, gate, up, input, _silu)| (mul, (gate, up, input)))
            .collect();

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if consumed.contains_key(&node.id) {
                continue;
            }
            if let Some(&(gate_mm, up_mm, input_id)) = match_by_mul.get(&node.id) {
                let gate = graph.node(gate_mm);
                let up = graph.node(up_mm);
                let wg = gate.inputs[1];
                let wu = up.inputs[1];
                rw.ensure_mapped(&graph, &[input_id, wg, wu]);

                let wu_shape = graph.shape(wu);
                let wg_shape = graph.shape(wg);
                let k = wu_shape.dim(0).unwrap_static();
                let n_up = wu_shape.dim(1).unwrap_static();
                let n_gate = wg_shape.dim(1).unwrap_static();
                debug_assert_eq!(wu_shape.dim(0), wg_shape.dim(0));

                // Up weights first → canonical FusedSwiGLU layout (gate_first=false).
                let concat_shape = Shape::new(&[k, n_up + n_gate], wu_shape.dtype());
                let concat_w = rw.add_fused(Op::Concat { axis: 1 }, &[wu, wg], concat_shape);

                let out_rank = up.shape.rank();
                let mut mm_dims: Vec<Dim> = (0..out_rank).map(|i| up.shape.dim(i)).collect();
                mm_dims[out_rank - 1] = Dim::Static(n_up + n_gate);
                let cat_shape = Shape::from_dims(&mm_dims, up.shape.dtype());
                let cat_id =
                    rw.new_graph
                        .add_node(Op::MatMul, vec![rw.map(input_id), concat_w], cat_shape);

                let fused_id = rw.new_graph.add_node(
                    Op::FusedSwiGLU {
                        cast_to: None,
                        gate_first: false,
                    },
                    vec![cat_id],
                    node.shape.clone(),
                );
                rw.replace(node.id, fused_id);
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseSwiGLUDualMatmul {
    // Required by construction: both the up and gate branches must be MatMuls.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::MatMul]
    }

    fn name(&self) -> &str {
        "fuse_swiglu_dual_matmul"
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

// ── Pass 3: Shared-input MatMul concat (QKV, SwiGLU fc11+fc12) ──────────

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;

    /// `silu(x·wg) · (x·wu)` for weights `w` (`[.., K, N]`), `x` `[.., 4, K]`.
    fn swiglu(w: &[usize]) -> Graph {
        let mut g = Graph::new("swiglu");
        let r = w.len();
        let mut x_dims = w[..r - 2].to_vec();
        x_dims.extend([4, w[r - 2]]);
        let x = g.input("x", Shape::new(&x_dims, DType::F32));
        let wg = g.param("wg", Shape::new(w, DType::F32));
        let wu = g.param("wu", Shape::new(w, DType::F32));
        let gate = g.mm(x, wg);
        let up = g.mm(x, wu);
        let a = g.silu(gate);
        let out = g.mul(a, up);
        g.set_outputs(vec![out]);
        g
    }

    /// `[K, N]` weights fuse; a batched `[B, K, N]` pair (an expert per batch
    /// row) is left as it is.
    #[test]
    fn only_two_dimensional_weights_fuse() {
        let fused = FuseSwiGLUDualMatmul.run(swiglu(&[8, 5]));
        assert!(
            fused
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedSwiGLU { .. }))
        );
        let batched = FuseSwiGLUDualMatmul.run(swiglu(&[2, 8, 5]));
        assert!(
            !batched
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::FusedSwiGLU { .. }))
        );
    }
}
