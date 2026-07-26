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

//! Lower `Op::{Clamp, Tile, Trilu}` to primitives that are native on every
//! backend (`max`/`min`, `concat`, mul-by-constant-mask) — the decomposition is
//! both the semantic oracle and the peak-perf path, so no dedicated kernels are
//! needed. Runs in the legalize loop like `LowerPad`/`LowerSlice`.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::BinaryOp;
use rlx_ir::*;
use std::collections::HashMap;

fn static_dims(g: &Graph, x: NodeId) -> Vec<usize> {
    g.shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect()
}

/// `clamp(x, min, max) = min(max(x, lo), hi)`.
pub fn lower_clamp(g: &mut Graph, x: NodeId, min: f32, max: f32) -> NodeId {
    let dtype = g.shape(x).dtype();
    let shape = g.shape(x).clone();
    let lo = g.full(&[1], min, dtype);
    let hi = g.full(&[1], max, dtype);
    let m = g.add_node(Op::Binary(BinaryOp::Max), vec![x, lo], shape.clone());
    g.add_node(Op::Binary(BinaryOp::Min), vec![m, hi], shape)
}

/// `tile(x, reps)` — per axis, concat `reps[axis]` copies of the current tensor.
pub fn lower_tile(g: &mut Graph, mut cur: NodeId, reps: &[usize]) -> NodeId {
    for (axis, &r) in reps.iter().enumerate() {
        if r <= 1 {
            continue;
        }
        let copies = vec![cur; r];
        cur = g.concat_(copies, axis);
    }
    cur
}

/// `trilu(x, upper, diagonal)` = `x * mask`, mask over the last two axes:
/// keep where `upper ? (col - row >= diagonal) : (col - row <= diagonal)`.
pub fn lower_trilu(g: &mut Graph, x: NodeId, upper: bool, diagonal: i64) -> NodeId {
    let dims = static_dims(g, x);
    let rank = dims.len();
    let dtype = g.shape(x).dtype();
    let (rows, cols) = (dims[rank - 2], dims[rank - 1]);
    let mut mask = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let keep = if upper {
                (c as i64 - r as i64) >= diagonal
            } else {
                (c as i64 - r as i64) <= diagonal
            };
            mask[r * cols + c] = keep as u8 as f32;
        }
    }
    let data: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mask_node = g.add_node(
        Op::Constant { data },
        vec![],
        Shape::new(&[rows, cols], dtype),
    );
    // Reshape to [1,…,1,rows,cols] so the multiply broadcasts over batch dims.
    let mut bshape = vec![1i64; rank];
    bshape[rank - 2] = rows as i64;
    bshape[rank - 1] = cols as i64;
    let mask_b = g.reshape_(mask_node, bshape);
    let shape = g.shape(x).clone();
    g.add_node(Op::Binary(BinaryOp::Mul), vec![x, mask_b], shape)
}

/// Rewrite every `Op::{Clamp, Tile, Trilu}` node into primitives.
pub struct LowerStructural;

impl Pass for LowerStructural {
    fn name(&self) -> &str {
        "lower_structural"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Clamp { .. } | Op::Tile { .. } | Op::Trilu { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let new_id = match &node.op {
                Op::Clamp { min, max } => lower_clamp(&mut new_graph, inputs[0], *min, *max),
                Op::Tile { reps } => lower_tile(&mut new_graph, inputs[0], reps),
                Op::Trilu { upper, diagonal } => {
                    lower_trilu(&mut new_graph, inputs[0], *upper, *diagonal)
                }
                _ => new_graph.add_node(node.op.clone(), inputs, node.shape.clone()),
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}
