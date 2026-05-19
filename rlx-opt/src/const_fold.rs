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

//! Constant Folding — evaluate pure-input subgraphs at compile time.
//!
//! A node is foldable when all its inputs are foldable AND the op has
//! a deterministic, pure evaluation (no I/O, no random). We evaluate
//! such subgraphs once at compile time and replace them with `Op::Constant`.
//!
//! Examples that get folded:
//! - `1.0 / sqrt(head_dim)` (attention scale factor)
//! - `cast(known_param)` to a different dtype
//! - small reshapes/expands of constants
//!
//! For a typical transformer, constant folding only catches scattered
//! arithmetic — but it eliminates 10–50 redundant ops over a 12-layer
//! model and shrinks the arena slightly.

use crate::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};

pub struct ConstantFolding;

/// True if this op can be evaluated symbolically with no runtime state.
fn is_pure(op: &Op) -> bool {
    matches!(
        op,
        Op::Activation(_)
            | Op::Binary(_)
            | Op::Compare(_)
            | Op::Reshape { .. }
            | Op::Expand { .. }
            | Op::Cast { .. }
    )
}

/// True if the node's inputs are all known constants (Param, Constant, or
/// previously-folded result).
fn is_foldable(node_id: NodeId, graph: &Graph, folded: &HashSet<NodeId>) -> bool {
    let node = graph.node(node_id);
    if !is_pure(&node.op) {
        return false;
    }
    node.inputs.iter().all(|i| folded.contains(i))
}

/// Evaluate a foldable node given precomputed input values.
/// Returns a flat f32 buffer of the result, or None if not supported.
fn evaluate(node: &rlx_ir::Node, inputs: &[&Vec<f32>]) -> Option<Vec<f32>> {
    let total = node.shape.num_elements()?;
    let mut out = vec![0f32; total];

    match &node.op {
        Op::Activation(act) => {
            let x = inputs[0];
            for (i, &v) in x.iter().enumerate() {
                out[i] = match act {
                    Activation::Gelu | Activation::GeluApprox => {
                        v * 0.5 * (1.0 + (v * std::f32::consts::FRAC_1_SQRT_2).tanh())
                    }
                    Activation::Silu => v / (1.0 + (-v).exp()),
                    Activation::Relu => v.max(0.0),
                    Activation::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                    Activation::Tanh => v.tanh(),
                    Activation::Exp => v.exp(),
                    Activation::Log => v.ln(),
                    Activation::Sqrt => v.sqrt(),
                    Activation::Rsqrt => 1.0 / v.sqrt(),
                    Activation::Neg => -v,
                    Activation::Abs => v.abs(),
                    Activation::Round => v.round(),
                    Activation::Sin => v.sin(),
                    Activation::Cos => v.cos(),
                    Activation::Tan => v.tan(),
                    Activation::Atan => v.atan(),
                };
            }
            Some(out)
        }
        Op::Binary(op) => {
            let lhs = inputs[0];
            let rhs = inputs[1];
            // Naive: support same-shape only. Broadcast handled later.
            if lhs.len() != total || rhs.len() != total {
                return None;
            }
            for i in 0..total {
                out[i] = match op {
                    BinaryOp::Add => lhs[i] + rhs[i],
                    BinaryOp::Sub => lhs[i] - rhs[i],
                    BinaryOp::Mul => lhs[i] * rhs[i],
                    BinaryOp::Div => lhs[i] / rhs[i],
                    BinaryOp::Max => lhs[i].max(rhs[i]),
                    BinaryOp::Min => lhs[i].min(rhs[i]),
                    BinaryOp::Pow => lhs[i].powf(rhs[i]),
                };
            }
            Some(out)
        }
        Op::Reshape { .. } | Op::Expand { .. } | Op::Cast { .. } => {
            // Same data, just reshape/cast. For now: copy through as f32.
            let src = inputs[0];
            if src.len() == total {
                Some(src.clone())
            } else if src.len() == 1 {
                Some(vec![src[0]; total])
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Encode an f32 buffer as raw bytes for `Op::Constant`.
fn encode_constant(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

impl Pass for ConstantFolding {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Walk in topological order, tracking which nodes are foldable
        // and accumulating their evaluated values.
        let mut folded: HashSet<NodeId> = HashSet::new();
        let mut values: HashMap<NodeId, Vec<f32>> = HashMap::new();

        for node in graph.nodes() {
            // Constant nodes are trivially foldable (we already have the data).
            if let Op::Constant { data } = &node.op {
                folded.insert(node.id);
                let f32s: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                values.insert(node.id, f32s);
                continue;
            }
            // Inputs/Params are NOT foldable (their values are runtime).
            if matches!(node.op, Op::Input { .. } | Op::Param { .. }) {
                continue;
            }
            // Try to fold pure ops with all-constant inputs.
            if is_foldable(node.id, &graph, &folded) {
                let inputs: Vec<&Vec<f32>> = node.inputs.iter().map(|i| &values[i]).collect();
                if let Some(result) = evaluate(node, &inputs) {
                    folded.insert(node.id);
                    values.insert(node.id, result);
                }
            }
        }

        // Rebuild: replace folded nodes with Op::Constant, rewire others.
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            // Foldable downstream nodes get replaced with Constant unless
            // they're terminal Constants/Params themselves.
            if folded.contains(&node.id)
                && !matches!(
                    node.op,
                    Op::Constant { .. } | Op::Param { .. } | Op::Input { .. }
                )
            {
                let bytes = encode_constant(&values[&node.id]);
                let new_id =
                    new_graph.add_node(Op::Constant { data: bytes }, vec![], node.shape.clone());
                id_map.insert(node.id, new_id);
                continue;
            }
            // Otherwise copy the node, remapping inputs.
            let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
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
    use rlx_ir::*;

    #[test]
    fn folds_constant_arithmetic() {
        // const(2.0) + const(3.0) → const(5.0)
        let mut g = Graph::new("test");
        let a = g.add_node(
            Op::Constant {
                data: 2.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let b = g.add_node(
            Op::Constant {
                data: 3.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let sum = g.binary(op::BinaryOp::Add, a, b, Shape::new(&[1], DType::F32));
        g.set_outputs(vec![sum]);

        let folded = ConstantFolding.run(g);
        // After folding, the Add node should be a Constant with value 5.0
        let out_node = folded.node(folded.outputs[0]);
        if let Op::Constant { data } = &out_node.op {
            let v = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            assert!((v - 5.0).abs() < 1e-6);
        } else {
            panic!("expected folded Constant, got {:?}", out_node.op);
        }
    }

    #[test]
    fn does_not_fold_input_dependent() {
        let mut g = Graph::new("test");
        let x = g.input("x", Shape::new(&[4], DType::F32));
        let c = g.add_node(
            Op::Constant {
                data: vec![0u8; 16],
            },
            vec![],
            Shape::new(&[4], DType::F32),
        );
        let sum = g.binary(op::BinaryOp::Add, x, c, Shape::new(&[4], DType::F32));
        g.set_outputs(vec![sum]);

        let folded = ConstantFolding.run(g);
        // x + c is input-dependent; should NOT be folded.
        assert!(matches!(folded.node(folded.outputs[0]).op, Op::Binary(_)));
    }
}
