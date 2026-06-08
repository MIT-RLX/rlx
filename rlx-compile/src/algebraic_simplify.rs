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

//! Algebraic simplification on graphs with constant leaves.
//!
//! Complements [`crate::const_fold::ConstantFolding`] by rewriting ops that have
//! one constant operand and one dynamic operand (e.g. `mul(x, 0) → 0`).

use rlx_fusion::pass::Pass;
use rlx_ir::op::BinaryOp;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;

fn decode_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn encode_f32(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

fn constant_f32_values(graph: &Graph, id: NodeId) -> Option<Vec<f32>> {
    match &graph.node(id).op {
        Op::Constant { data } => Some(decode_f32(data)),
        _ => None,
    }
}

fn is_all_zero(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

fn is_all_one(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 1.0)
}

fn zeros_like(graph: &mut Graph, shape: &rlx_ir::Shape) -> NodeId {
    let n = shape.num_elements().unwrap_or(1);
    graph.add_node(
        Op::Constant {
            data: encode_f32(&vec![0.0; n]),
        },
        vec![],
        shape.clone(),
    )
}

/// One pass of local binary simplification.
pub fn algebraic_simplify(graph: &Graph) -> Graph {
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let simplified = if let Op::Binary(op) = &node.op {
            if new_inputs.len() != 2 {
                None
            } else {
                let (a, b) = (new_inputs[0], new_inputs[1]);
                let a_const = constant_f32_values(&out, a);
                let b_const = constant_f32_values(&out, b);
                let out_elems = node.shape.num_elements().unwrap_or(0);
                let const_matches = |c: &[f32]| c.len() == out_elems || c.len() == 1;
                match (op, a_const.as_deref(), b_const.as_deref()) {
                    (BinaryOp::Add, Some(c), None) if const_matches(c) && is_all_zero(c) => Some(b),
                    (BinaryOp::Add, None, Some(c)) if const_matches(c) && is_all_zero(c) => Some(a),
                    (BinaryOp::Sub, None, Some(c)) if const_matches(c) && is_all_zero(c) => Some(a),
                    (BinaryOp::Mul, Some(c), None)
                        if const_matches(c) && (is_all_zero(c) || is_all_one(c)) =>
                    {
                        if is_all_zero(c) {
                            Some(zeros_like(&mut out, &node.shape))
                        } else {
                            Some(b)
                        }
                    }
                    (BinaryOp::Mul, None, Some(c))
                        if const_matches(c) && (is_all_zero(c) || is_all_one(c)) =>
                    {
                        if is_all_zero(c) {
                            Some(zeros_like(&mut out, &node.shape))
                        } else {
                            Some(a)
                        }
                    }
                    _ => None,
                }
            }
        } else {
            None
        };

        let new_id = if let Some(reuse_id) = simplified {
            reuse_id
        } else {
            out.add_node(node.op.clone(), new_inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    out
}

pub struct AlgebraicSimplify;

impl Pass for AlgebraicSimplify {
    fn name(&self) -> &str {
        "algebraic_simplify"
    }

    fn run(&self, graph: Graph) -> Graph {
        algebraic_simplify(&graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::Shape;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::*;

    #[test]
    fn mul_by_zero_scalar_zeros_output() {
        let s = Shape::new(&[4], DType::F32);
        let mut g = Graph::new("t");
        let x = g.input("x", s.clone());
        let z = g.add_node(
            Op::Constant {
                data: 0.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F32),
        );
        let y = g.binary(BinaryOp::Mul, x, z, s.clone());
        g.set_outputs(vec![y]);

        let out = algebraic_simplify(&g);
        assert!(matches!(out.node(out.outputs[0]).op, Op::Constant { .. }));
    }
}
