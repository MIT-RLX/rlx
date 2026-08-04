// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Common Subexpression Elimination — merge structurally identical nodes.
//!
//! Two interior nodes with the same op, the same (already-remapped) inputs, and
//! the same shape compute bit-identically, so the second is redundant. Walking
//! the graph in topological order and value-numbering each node collapses every
//! such duplicate to its first occurrence.
//!
//! Why it matters (backward graphs especially): reverse-mode AD emits the same
//! subexpression many times. The prime example is **multi-stage weight synthesis**
//! (`rlx-tiny`): `q = x·W₀ + x·W₁ + …` makes every stage's weight-gradient
//! `grad_Wₛ = upstreamᵀ·x` — *identical* across stages (same `upstream`, same `x`).
//! Without CSE each stage recomputes the transpose **and** the GEMM; CSE keeps one
//! copy, cutting a `Transpose`+`MatMul` per extra stage per projection. Unlike
//! MPS transpose-folding (which loses to per-call overhead at small matmul scale),
//! this removes the work outright on whatever kernel the backend already picks.
//!
//! Only **interior** nodes (non-empty inputs) are value-numbered: two `Op::Input`s
//! or `Op::Param`s can share a shape yet denote different values, so leaves are
//! always kept distinct. Merging is bit-exact — the surviving node has identical
//! op/inputs/shape, hence identical output.

use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, NodeId};
use std::collections::HashMap;

pub struct CommonSubexpressionElimination;

impl Pass for CommonSubexpressionElimination {
    fn name(&self) -> &str {
        "common_subexpression_elimination"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        // Value-number table: (op, remapped inputs, shape) → surviving new id. The
        // op/shape are keyed by their `Debug` form so every embedded field (perm,
        // axes, dtype, spline grid, …) participates without an `Eq`/`Hash` bound.
        let mut seen: HashMap<(String, Vec<NodeId>, String), NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_inputs: Vec<NodeId> = node.inputs.iter().map(|id| id_map[id]).collect();

            // Leaves (Input/Param/Constant/… — no inputs) are never merged: a shared
            // shape does not make two inputs the same value.
            if !new_inputs.is_empty() {
                let key = (
                    format!("{:?}", node.op),
                    new_inputs.clone(),
                    format!("{:?}", node.shape),
                );
                if let Some(&existing) = seen.get(&key) {
                    id_map.insert(node.id, existing);
                    continue;
                }
                let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
                if node.name.is_some() || node.origin.is_some() {
                    let n = new_graph.node_mut(new_id);
                    n.name = node.name.clone();
                    n.origin = node.origin.clone();
                }
                seen.insert(key, new_id);
                id_map.insert(node.id, new_id);
            } else {
                let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
                if node.name.is_some() || node.origin.is_some() {
                    let n = new_graph.node_mut(new_id);
                    n.name = node.name.clone();
                    n.origin = node.origin.clone();
                }
                id_map.insert(node.id, new_id);
            }
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|id| id_map[id]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Op, Shape};

    fn f32s(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn merges_identical_transpose_matmul() {
        // Two identical `Transpose(u)` and `MatMul(Transpose(u), x)` — the shape a
        // 2-stage synth backward produces — collapse to one each.
        let mut g = Graph::new("bwd");
        let u = g.input("u", f32s(&[4, 3]));
        let x = g.input("x", f32s(&[4, 5]));
        let ut0 = g.add_node(Op::Transpose { perm: vec![1, 0] }, vec![u], f32s(&[3, 4]));
        let ut1 = g.add_node(Op::Transpose { perm: vec![1, 0] }, vec![u], f32s(&[3, 4]));
        let gw0 = g.add_node(Op::MatMul, vec![ut0, x], f32s(&[3, 5]));
        let gw1 = g.add_node(Op::MatMul, vec![ut1, x], f32s(&[3, 5]));
        // Consume both so neither is dead.
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![gw0, gw1], f32s(&[3, 5]));
        g.set_outputs(vec![sum]);
        let before = g.nodes().len();

        let out = CommonSubexpressionElimination.run(g);
        let after = out.nodes().len();
        // One Transpose + one MatMul removed (2 inputs + 1 T + 1 MM + 1 Add = 5).
        assert_eq!(after, before - 2, "CSE should drop the duplicate T+MM");
        assert_eq!(after, 5);
    }

    #[test]
    fn keeps_distinct_inputs() {
        // Two same-shape inputs must NOT be merged.
        let mut g = Graph::new("leaves");
        let a = g.input("a", f32s(&[2, 2]));
        let b = g.input("b", f32s(&[2, 2]));
        let s = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f32s(&[2, 2]));
        g.set_outputs(vec![s]);
        let out = CommonSubexpressionElimination.run(g);
        // a, b, add all survive.
        assert_eq!(out.nodes().len(), 3);
    }
}
