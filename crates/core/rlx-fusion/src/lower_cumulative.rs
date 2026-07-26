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

//! Lower `Op::{CumProd, CumMax}` to a masked reduce over an inserted query axis —
//! the decomposition oracle for backends without a native scan kernel
//! (CPU/Metal/CUDA carry hand-written O(L) scans; everyone else legalizes here).
//!
//! For a length-`L` scan axis we insert a size-1 *query* axis in front of the
//! *key* axis, then for each query `i` reduce over the keys `j` masked to the
//! prefix `j <= i` (or `j < i` when exclusive):
//!
//! * CumProd — masked keys become the multiplicative identity `1`, then
//!   `reduce_prod` over the key axis.
//! * CumMax  — masked keys become a large negative sentinel, then `reduce_max`.
//!
//! Broadcasting expands the size-1 query axis against the `[L, L]` mask, so no
//! explicit `Expand` node is needed. It's O(L²) — the native kernels are the
//! perf path; this is the correctness fallback.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::ReduceOp;
use rlx_ir::*;
use std::collections::HashMap;

/// Sentinel standing in for `-inf` on the masked-out keys of a CumMax reduce.
/// Finite so the `keep * x + (1-keep) * SENT` arithmetic can't hit `0 * -inf =
/// NaN`; the native kernels emit a true `-inf` for the exclusive `[0]` slot.
const NEG_SENTINEL: f32 = -3.0e38;

fn static_dims(g: &Graph, x: NodeId) -> Vec<usize> {
    g.shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect()
}

/// Build the `[L, L]` prefix mask (reshaped to broadcast at the query/key axes)
/// as an f32 constant: `keep[i, j] = exclusive ? j < i : j <= i`.
fn prefix_mask(g: &mut Graph, rank: usize, axis: usize, len: usize, exclusive: bool) -> NodeId {
    let mut mask = vec![0f32; len * len];
    for i in 0..len {
        for j in 0..len {
            let keep = if exclusive { j < i } else { j <= i };
            mask[i * len + j] = keep as u8 as f32;
        }
    }
    let data: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
    let node = g.add_node(
        Op::Constant { data },
        vec![],
        Shape::new(&[len, len], DType::F32),
    );
    // Reshape to [1,…,L(query),L(key),…,1] so it broadcasts over the batch dims.
    let mut bshape = vec![1i64; rank + 1];
    bshape[axis] = len as i64;
    bshape[axis + 1] = len as i64;
    g.reshape_(node, bshape)
}

/// Reduce the (inserted) key axis, collapsing back to the input's rank/shape.
fn reduce_key(
    g: &mut Graph,
    x: NodeId,
    key_axis: usize,
    out_dims: &[usize],
    op: ReduceOp,
) -> NodeId {
    let dtype = g.shape(x).dtype();
    g.reduce(x, op, vec![key_axis], false, Shape::new(out_dims, dtype))
}

/// `cumprod(x)[…,i,…] = prod_{j<=i} x[…,j,…]` (or `j<i` when exclusive).
pub fn lower_cumprod(g: &mut Graph, x: NodeId, axis: i32, exclusive: bool) -> NodeId {
    let dims = static_dims(g, x);
    let rank = dims.len();
    let axis = if axis < 0 {
        (axis + rank as i32).max(0) as usize
    } else {
        axis as usize
    };
    let len = dims[axis];
    let dtype = g.shape(x).dtype();

    // Insert a size-1 query axis at `axis`; the key axis is now `axis + 1`.
    let mut q_dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    q_dims.insert(axis, 1);
    let xq = g.reshape_(x, q_dims);

    // masked = 1 + keep * (x - 1)  → keeps x on the prefix, 1 (identity) elsewhere.
    let keep = prefix_mask(g, rank, axis, len, exclusive);
    let one = g.full(&[1], 1.0, dtype);
    let x_m1 = g.sub(xq, one);
    let scaled = g.mul(keep, x_m1);
    let one2 = g.full(&[1], 1.0, dtype);
    let masked = g.add(scaled, one2);

    reduce_key(g, masked, axis + 1, &dims, ReduceOp::Prod)
}

/// `cummax(x)[…,i,…] = max_{j<=i} x[…,j,…]` (or `j<i` when exclusive).
pub fn lower_cummax(g: &mut Graph, x: NodeId, axis: i32, exclusive: bool) -> NodeId {
    let dims = static_dims(g, x);
    let rank = dims.len();
    let axis = if axis < 0 {
        (axis + rank as i32).max(0) as usize
    } else {
        axis as usize
    };
    let len = dims[axis];
    let dtype = g.shape(x).dtype();

    let mut q_dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    q_dims.insert(axis, 1);
    let xq = g.reshape_(x, q_dims);

    // masked = keep * x + (1 - keep) * SENT  → x on the prefix, -inf sentinel elsewhere.
    let keep = prefix_mask(g, rank, axis, len, exclusive);
    let kx = g.mul(keep, xq);
    let one = g.full(&[1], 1.0, dtype);
    let inv = g.sub(one, keep); // 1 - keep
    let sent = g.full(&[1], NEG_SENTINEL, dtype);
    let masked_gap = g.mul(inv, sent);
    let masked = g.add(kx, masked_gap);

    reduce_key(g, masked, axis + 1, &dims, ReduceOp::Max)
}

/// Rewrite every `Op::{CumProd, CumMax}` node that survives to the legalize loop.
pub struct LowerCumulative;

impl Pass for LowerCumulative {
    fn name(&self) -> &str {
        "lower_cumulative"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::CumProd { .. } | Op::CumMax { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let new_id = match &node.op {
                Op::CumProd { axis, exclusive } => {
                    lower_cumprod(&mut new_graph, inputs[0], *axis, *exclusive)
                }
                Op::CumMax { axis, exclusive } => {
                    lower_cummax(&mut new_graph, inputs[0], *axis, *exclusive)
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
