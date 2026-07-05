// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! `swiglu_dual` — extracted from the `fusion` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

// ── Helper: graph rewriter ──────────────────────────────────────────────

use crate::graph_rewrite::Rewriter;

// ── Pass 1: MatMul + Bias + Activation → FusedMatMulBiasAct ─────────────

use super::*;

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
        if graph.use_count(silu_id) != 1 {
            return None;
        }
        Some((mul_node.id, gate_mm.id, up_mm.id, up_mm.inputs[0], silu_id))
    }
}

impl Pass for FuseSwiGLUDualMatmul {
    fn name(&self) -> &str {
        "fuse_swiglu_dual_matmul"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut matches: Vec<(NodeId, NodeId, NodeId, NodeId, NodeId)> = Vec::new();
        let mut consumed: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if let Some((mul_id, gate_mm, up_mm, _, silu_id)) =
                Self::match_dual_swiglu(&graph, node)
            {
                matches.push((mul_id, gate_mm, up_mm, graph.node(up_mm).inputs[0], silu_id));
                consumed.insert(gate_mm, ());
                consumed.insert(up_mm, ());
                consumed.insert(silu_id, ());
            }
        }

        if matches.is_empty() {
            return graph;
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
        rw.finish(&graph.outputs)
    }
}

// ── Pass 3: Shared-input MatMul concat (QKV, SwiGLU fc11+fc12) ──────────
