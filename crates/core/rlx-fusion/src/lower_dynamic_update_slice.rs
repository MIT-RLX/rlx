// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::DynamicUpdateSlice` to a `ScatterNd`, for backends with no
//! native kernel.
//!
//! The native op writes `update` into `operand`'s own buffer at a runtime
//! `start`: one row of a decode cache, O(1) in the context. A `ScatterNd`
//! copies `operand` and writes the rows into the copy, so it is O(operand)
//! — correct everywhere, and what a backend without the native write gets.
//!
//! The indices are `[i₀, …, iₐ₋₁, start + r]` for every position `i` of the
//! axes before `axis` and each row `r` of `update` (`start` clamped to
//! `[0, extent - rows]`, as the native kernels and JAX clamp it): a batched
//! cache `[batch, cap, width]` gets each sequence's rows at the same
//! `start`.

use std::collections::HashMap;

use rlx_ir::op::ScatterNdReduction;
use rlx_ir::{DType, Dim, Graph, NodeId, Op, OpKind, Shape};

use crate::Pass;

/// `ScatterNd(operand, indices, rows)` for `DynamicUpdateSlice(operand,
/// update, start)` along `axis`; `None` when a dim is not static or the
/// update's axes before `axis` differ from the operand's.
pub fn lower_dynamic_update_slice(
    g: &mut Graph,
    operand: NodeId,
    update: NodeId,
    start: NodeId,
    axis: usize,
) -> Option<NodeId> {
    let static_dims = |s: &Shape| -> Option<Vec<usize>> {
        s.dims()
            .iter()
            .map(|d| match d {
                Dim::Static(n) => Some(*n),
                Dim::Dynamic(_) => None,
            })
            .collect()
    };
    let op_dims = static_dims(&g.node(operand).shape)?;
    let up_dims = static_dims(&g.node(update).shape)?;
    if op_dims[..axis] != up_dims[..axis] {
        return None;
    }
    let lead = &op_dims[..axis];
    let outer: usize = lead.iter().product();
    let rows = up_dims[axis];
    let extent = op_dims[axis];
    let f32s = |d: &[usize]| Shape::new(d, DType::F32);
    let start = match g.node(start).shape.dtype() {
        DType::F32 => start,
        _ => g.add_node(Op::Cast { to: DType::F32 }, vec![start], f32s(&[1])),
    };
    let start = g.add_node(
        Op::Reshape {
            new_shape: vec![1, 1],
        },
        vec![start],
        f32s(&[1, 1]),
    );
    let start = g.add_node(
        Op::Clamp {
            min: 0.0,
            max: extent.saturating_sub(rows) as f32,
        },
        vec![start],
        f32s(&[1, 1]),
    );
    // For each position before `axis` (row-major), row `r` at `start + r`:
    // the order `update` flattens in.
    let n = outer * rows;
    let offsets: Vec<u8> = (0..n)
        .flat_map(|i| ((i % rows) as f32).to_le_bytes())
        .collect();
    let offsets = g.add_node(Op::Constant { data: offsets }, vec![], f32s(&[n, 1]));
    let at = g.add_node(
        Op::Binary(rlx_ir::op::BinaryOp::Add),
        vec![offsets, start],
        f32s(&[n, 1]),
    );
    let index = if axis == 0 {
        at
    } else {
        let mut lead_index = Vec::with_capacity(n * axis * 4);
        for o in 0..outer {
            // `o` as an index into `lead`, row-major.
            let mut digits = vec![0usize; axis];
            let mut rest = o;
            for a in (0..axis).rev() {
                digits[a] = rest % lead[a];
                rest /= lead[a];
            }
            for _ in 0..rows {
                for &d in &digits {
                    lead_index.extend_from_slice(&(d as f32).to_le_bytes());
                }
            }
        }
        let lead_index = g.add_node(Op::Constant { data: lead_index }, vec![], f32s(&[n, axis]));
        g.add_node(
            Op::Concat { axis: 1 },
            vec![lead_index, at],
            f32s(&[n, axis + 1]),
        )
    };
    // `ScatterNd`'s updates are `indices.shape[:-1] + operand.shape[axis+1:]`.
    let mut row_dims = vec![n];
    row_dims.extend_from_slice(&up_dims[axis + 1..]);
    let dtype = g.node(update).shape.dtype();
    let rows_of = g.add_node(
        Op::Reshape {
            new_shape: row_dims.iter().map(|&d| d as i64).collect(),
        },
        vec![update],
        Shape::new(&row_dims, dtype),
    );
    let shape = g.node(operand).shape.clone();
    Some(g.add_node(
        Op::ScatterNd {
            reduction: ScatterNdReduction::None,
        },
        vec![operand, index, rows_of],
        shape,
    ))
}

/// Pass form: every `DynamicUpdateSlice` a backend does not claim becomes a
/// `ScatterNd`.
pub struct LowerDynamicUpdateSlice;

impl Pass for LowerDynamicUpdateSlice {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::DynamicUpdateSlice]
    }

    fn name(&self) -> &str {
        "lower_dynamic_update_slice"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::DynamicUpdateSlice { .. }))
        {
            return graph;
        }
        let mut out = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let lowered = match node.op {
                Op::DynamicUpdateSlice { axis } => {
                    lower_dynamic_update_slice(&mut out, inputs[0], inputs[1], inputs[2], axis)
                }
                _ => None,
            };
            let id = match lowered {
                Some(id) => id,
                None => {
                    let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
                    out.node_mut(id).name = node.name.clone();
                    id
                }
            };
            id_map.insert(node.id, id);
        }
        out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
        out.set_vars(graph.vars().iter().map(|(i, o)| (id_map[i], *o)).collect());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lowering_is_a_scatter_with_the_operand_shape() {
        let mut g = Graph::new("dus");
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let cache = g.input("cache", f(&[1, 8, 4]));
        let rows = g.input("rows", f(&[1, 2, 4]));
        let start = g.input("start", f(&[1]));
        let y = g.add_node(
            Op::DynamicUpdateSlice { axis: 1 },
            vec![cache, rows, start],
            f(&[1, 8, 4]),
        );
        g.set_outputs(vec![y]);
        let g = LowerDynamicUpdateSlice.run(g);
        assert!(
            !g.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::DynamicUpdateSlice { .. }))
        );
        let out = g.node(g.outputs[0]);
        assert!(matches!(out.op, Op::ScatterNd { .. }));
        assert_eq!(out.shape, f(&[1, 8, 4]));
        assert!(
            rlx_ir::verify::verify_all(&g).is_empty(),
            "{:?}",
            rlx_ir::verify::verify_all(&g)
        );
    }
}
