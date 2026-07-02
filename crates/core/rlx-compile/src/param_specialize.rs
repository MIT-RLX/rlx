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

//! Replace selected `Op::Param` nodes with `Op::Constant` before compile-time opts.
//!
//! Deploy graphs (e.g. pruned ternary FFT) often fix gate masks and zero twiddles
//! at specialization time while still building the graph with `Graph::param`.
//! Baking those values here lets constant folding and DCE remove dead paths.

use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

fn encode_f32(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Substitute listed params with constants. Unlisted params are unchanged.
pub fn specialize_params(graph: &Graph, bindings: &HashMap<String, Vec<f32>>) -> Graph {
    if bindings.is_empty() {
        return graph.clone();
    }
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_id = match &node.op {
            Op::Param { name } => {
                if let Some(values) = bindings.get(name) {
                    let expected = node.shape.num_elements().unwrap_or(values.len());
                    assert_eq!(
                        values.len(),
                        expected,
                        "param '{name}' binding len {} != shape elements {expected}",
                        values.len()
                    );
                    out.add_node(
                        Op::Constant {
                            data: encode_f32(values),
                        },
                        vec![],
                        node.shape.clone(),
                    )
                } else {
                    out.add_node(node.op.clone(), vec![], node.shape.clone())
                }
            }
            _ => {
                let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                out.add_node(node.op.clone(), new_inputs, node.shape.clone())
            }
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    out
}

/// Pass wrapper for the fusion pipeline / runtime preprocess hook.
pub struct SpecializeParams {
    pub bindings: HashMap<String, Vec<f32>>,
}

impl SpecializeParams {
    pub fn new(bindings: HashMap<String, Vec<f32>>) -> Self {
        Self { bindings }
    }
}

impl Pass for SpecializeParams {
    fn name(&self) -> &str {
        "specialize_params"
    }

    fn run(&self, graph: Graph) -> Graph {
        specialize_params(&graph, &self.bindings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::Shape;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::*;

    #[test]
    fn replaces_bound_param_with_constant() {
        let s = Shape::new(&[2], DType::F32);
        let mut g = Graph::new("t");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let y = g.binary(BinaryOp::Mul, x, w, s.clone());
        g.set_outputs(vec![y]);

        let mut bindings = HashMap::new();
        bindings.insert("w".into(), vec![0.0, 1.0]);
        let out = specialize_params(&g, &bindings);
        let w_node = out.node(out.nodes()[1].id);
        assert!(matches!(w_node.op, Op::Constant { .. }));
    }
}
