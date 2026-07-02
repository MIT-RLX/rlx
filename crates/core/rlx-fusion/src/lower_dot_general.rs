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

//! Lower `Op::DotGeneral` to primitive ops (MatMul + Transpose + Reshape).
//!
//! DotGeneral is XLA's fully general matmul: arbitrary contracting axes,
//! batch dimensions, etc. Implementing it as a backend primitive is a lot
//! of work. Instead we rewrite it to MatMul at the IR level. The existing
//! matmul kernels then handle dispatch — same code path as user-written
//! MatMuls, including all fusion benefits.
//!
//! Currently handles the common pattern that matters in practice:
//!   `dot_general(lhs[m, k], rhs[k, n], lhs_contracting=[1], rhs_contracting=[0])`
//! collapses to a plain MatMul. Other patterns (batched, non-standard
//! contracting axes) bail out — those are future work, but the coverage
//! report will tell us when one shows up.

use crate::pass::Pass;
use rlx_ir::*;
use std::collections::HashMap;

pub struct LowerDotGeneral;

impl Pass for LowerDotGeneral {
    fn name(&self) -> &str {
        "lower_dot_general"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Quick scan: is there anything to lower?
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::DotGeneral { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = match &node.op {
                Op::DotGeneral {
                    lhs_contracting,
                    rhs_contracting,
                    lhs_batch,
                    rhs_batch,
                } => {
                    // Only the canonical 2D pattern (no batch dims, contract on
                    // lhs's last axis and rhs's first axis) reduces to a plain
                    // MatMul. For everything else, leave the node intact —
                    // the coverage report flags it as MISSING and we fix it
                    // when a model needs it.
                    if lhs_batch.is_empty()
                        && rhs_batch.is_empty()
                        && lhs_contracting.as_slice() == [1]
                        && rhs_contracting.as_slice() == [0]
                    {
                        let lhs = id_map[&node.inputs[0]];
                        let rhs = id_map[&node.inputs[1]];
                        new_graph.add_node(Op::MatMul, vec![lhs, rhs], node.shape.clone())
                    } else {
                        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                        new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
                    }
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
