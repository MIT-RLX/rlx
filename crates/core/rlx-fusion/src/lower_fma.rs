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

//! Lower `Op::Fma` to `Mul` + `Add` for backends without a native fused
//! multiply-add.
//!
//! NOTE: this fallback uses TWO roundings, so it does **not** preserve the
//! single-rounding (error-free-transform) precision that `Op::Fma` exists for.
//! Backends that want compensated / double-word arithmetic must implement
//! `Op::Fma` natively (e.g. WGSL/MSL/CUDA `fma()`). This pass only guarantees
//! correctness-of-value on FMA-less backends (e.g. CoreML/ANE) so a graph that
//! uses `Op::Fma` runs everywhere instead of erroring at legalization.

use crate::pass::Pass;
use rlx_ir::op::BinaryOp;
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerFma;

impl Pass for LowerFma {
    fn name(&self) -> &str {
        "lower_fma"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph.nodes().iter().any(|n| matches!(n.op, Op::Fma)) {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::Fma => {
                    let a = id_map[&node.inputs[0]];
                    let b = id_map[&node.inputs[1]];
                    let c = id_map[&node.inputs[2]];
                    let prod = new_graph.add_node(
                        Op::Binary(BinaryOp::Mul),
                        vec![a, b],
                        node.shape.clone(),
                    );
                    new_graph.add_node(Op::Binary(BinaryOp::Add), vec![prod, c], node.shape.clone())
                }
                _ => {
                    let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                    new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
                }
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_shape(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    #[test]
    fn lowers_fma_to_mul_add() {
        let mut g = Graph::new("fma_lower");
        let a = g.input("a", f32_shape(&[8]));
        let b = g.input("b", f32_shape(&[8]));
        let c = g.input("c", f32_shape(&[8]));
        let fma = g.add_node(Op::Fma, vec![a, b, c], f32_shape(&[8]));
        g.set_outputs(vec![fma]);

        let out = LowerFma.run(g);

        // No Op::Fma survives; it became one Mul and one Add.
        assert!(
            !out.nodes().iter().any(|n| matches!(n.op, Op::Fma)),
            "LowerFma left an Op::Fma behind"
        );
        let muls = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
            .count();
        let adds = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Add)))
            .count();
        assert_eq!((muls, adds), (1, 1), "expected exactly one Mul and one Add");

        // The Add must consume the Mul's product and `c` (a*b + c), and the
        // graph's single output is that Add.
        let out_id = out.outputs[0];
        let add = out.node(out_id);
        assert!(matches!(add.op, Op::Binary(BinaryOp::Add)));
        let prod_id = add.inputs[0];
        assert!(matches!(out.node(prod_id).op, Op::Binary(BinaryOp::Mul)));
    }

    #[test]
    fn no_fma_is_a_noop() {
        // A graph with no Op::Fma passes through unchanged (same node count).
        let mut g = Graph::new("no_fma");
        let a = g.input("a", f32_shape(&[4]));
        let b = g.input("b", f32_shape(&[4]));
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f32_shape(&[4]));
        g.set_outputs(vec![sum]);
        let before = g.nodes().len();
        let out = LowerFma.run(g);
        assert_eq!(out.nodes().len(), before);
    }
}
