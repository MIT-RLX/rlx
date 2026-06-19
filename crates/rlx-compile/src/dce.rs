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

//! Dead Code Elimination — drop nodes that aren't reachable from any output.
//!
//! Walks the graph backwards from `graph.outputs`, marks every transitively
//! consumed node as live, then rebuilds the graph keeping only live nodes.
//!
//! Why it matters:
//! - Frees arena memory for buffers nobody reads.
//! - Avoids running kernels whose outputs are discarded.
//! - Catches accidental dead code (e.g., the early Vision graph builder
//!   emitted a patch projection that wasn't actually wired into the encoder).

use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, NodeId};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct DeadCodeElimination;

impl Pass for DeadCodeElimination {
    fn name(&self) -> &str {
        "dead_code_elimination"
    }

    fn run(&self, graph: Graph) -> Graph {
        // BFS backwards from outputs to find all reachable nodes.
        let mut live: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<NodeId> = graph.outputs.iter().copied().collect();
        while let Some(id) = queue.pop_front() {
            if !live.insert(id) {
                continue;
            }
            for &input in &graph.node(id).inputs {
                queue.push_back(input);
            }
        }

        // Rebuild graph keeping only live nodes (preserves topological order).
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            if !live.contains(&node.id) {
                continue;
            }
            // Inputs and Params are kept as-is; everything else gets remapped inputs.
            let new_inputs: Vec<NodeId> = node.inputs.iter().map(|id| id_map[id]).collect();
            let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
            if node.name.is_some() || node.origin.is_some() {
                let n = new_graph.node_mut(new_id);
                n.name = node.name.clone();
                n.origin = node.origin.clone();
            }
            id_map.insert(node.id, new_id);
        }
        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|id| id_map[id]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::*;

    #[test]
    fn drops_unreferenced_nodes() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let _dead = g.param("unused", Shape::new(&[8], DType::F32)); // never referenced
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![mm]);

        // Original has 4 nodes (x, w, unused, mm)
        assert_eq!(g.len(), 4);
        let after = DeadCodeElimination.run(g);
        // After DCE: 3 nodes (x, w, mm) — `unused` is gone
        assert_eq!(after.len(), 3);
    }

    #[test]
    fn keeps_used_nodes() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let y = g.input("y", Shape::new(&[4], DType::F32));
        let z = g.binary(op::BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![z]);

        let before = g.len();
        let after = DeadCodeElimination.run(g);
        assert_eq!(after.len(), before); // nothing dead
    }
}
