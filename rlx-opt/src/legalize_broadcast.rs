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

//! Broadcast legalization pass.
//!
//! `rlx-cpu`'s `Op::Binary` lowering uses `out[i] = lhs[i % lhs_len] OP
//! rhs[i % rhs_len]` (the `BiasAdd` / `BinaryFull` thunks). That's
//! correct iff each operand's shape is either:
//!
//! * identical to the output shape, or
//! * a scalar (single element), or
//! * a *trailing* slice of the output's shape, with no leading
//!   broadcast 1s — e.g. `[N, C] + [C]` where `C` is the trailing axis.
//!
//! Anything else — particularly the canonical conv-bias pattern
//! `[N, C, H, W] + [1, C, 1, 1]` — would interleave bias values across
//! all positions instead of channel-broadcasting them. Rather than
//! grow the thunks into a stride-aware kernel for every backend, this
//! pass rewrites the IR ahead of lowering: inserts a real `Op::Expand`
//! before each problem operand, materializing the broadcast in a
//! dedicated kernel that *does* understand axis strides. The output of
//! the pass is a graph whose `Op::Binary` nodes are all element-wise
//! over identically-shaped operands, so every backend's modulo-style
//! binary lowering becomes correct.
//!
//! This costs an extra arena buffer per legalized binary, but the
//! correctness win dominates — and the buffer's lifetime is tight
//! (single consumer), so memory planning recycles it aggressively.

use rlx_ir::shape::Dim;
use rlx_ir::{Graph, Node, NodeId, Op, Shape};
use std::collections::HashMap;

use crate::pass::Pass;

/// Pass that materializes non-trailing broadcasts via `Op::Expand`.
pub struct LegalizeBroadcast;

impl LegalizeBroadcast {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LegalizeBroadcast {
    fn default() -> Self {
        Self::new()
    }
}

impl Pass for LegalizeBroadcast {
    fn name(&self) -> &str {
        "legalize_broadcast"
    }
    fn run(&self, graph: Graph) -> Graph {
        run(graph)
    }
}

/// Free-function form for callers that don't go through the `Pass`
/// trait machinery (most backends).
pub fn run(graph: Graph) -> Graph {
    run_with_remap(graph).0
}

/// Run the pass and additionally return a `NodeId` remap from old →
/// new graph. Useful when callers need to translate references
/// (inputs, params, outputs) they captured before legalization.
pub fn run_with_remap(graph: Graph) -> (Graph, HashMap<NodeId, NodeId>) {
    let mut out = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_id = legalize_node(node, &graph, &id_map, &mut out);
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|id| id_map[id]).collect();
    out.set_outputs(new_outputs);
    (out, id_map)
}

fn legalize_node(
    node: &Node,
    fwd_graph: &Graph,
    id_map: &HashMap<NodeId, NodeId>,
    out: &mut Graph,
) -> NodeId {
    let new_inputs: Vec<NodeId> = node.inputs.iter().map(|id| id_map[id]).collect();

    // For Op::Binary, expand each operand to the output shape if it's
    // mid-broadcast (i.e., shape differs and isn't a clean trailing
    // bias). Equal-shape, scalar, and trailing-broadcast operands are
    // left alone — those paths are already correct.
    if matches!(node.op, Op::Binary(_)) && node.inputs.len() == 2 {
        let out_shape = &node.shape;
        let lhs_shape = fwd_graph.node(node.inputs[0]).shape.clone();
        let rhs_shape = fwd_graph.node(node.inputs[1]).shape.clone();

        let lhs_id = maybe_expand(new_inputs[0], &lhs_shape, out_shape, out);
        let rhs_id = maybe_expand(new_inputs[1], &rhs_shape, out_shape, out);
        return out.add_node(node.op.clone(), vec![lhs_id, rhs_id], node.shape.clone());
    }

    // Pass-through for everything else: copy node verbatim (with input
    // remapping).
    out.add_node(node.op.clone(), new_inputs, node.shape.clone())
}

fn maybe_expand(id: NodeId, src: &Shape, target: &Shape, out: &mut Graph) -> NodeId {
    if shape_eq(src, target) {
        return id;
    }
    if is_scalar(src) {
        return id;
    }
    if is_clean_trailing_broadcast(src, target) {
        return id;
    }

    // Non-trivial broadcast: materialize via Op::Expand.
    let target_dims_i64: Vec<i64> = target
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            Dim::Dynamic(_) => -1,
        })
        .collect();
    out.add_node(
        Op::Expand {
            target_shape: target_dims_i64,
        },
        vec![id],
        target.clone(),
    )
}

fn shape_eq(a: &Shape, b: &Shape) -> bool {
    a.dims() == b.dims() && a.dtype() == b.dtype()
}

fn is_scalar(s: &Shape) -> bool {
    let n: usize = s
        .dims()
        .iter()
        .filter_map(|d| match d {
            Dim::Static(n) => Some(*n),
            _ => None,
        })
        .product();
    n == 1
}

/// Returns `true` when `src` is a "clean trailing slice" of `target`:
/// every dim of `src` matches the corresponding right-aligned dim of
/// `target` exactly (no broadcast-via-1 within `src`). This is the
/// pattern the modulo-indexed `BinaryFull` kernel handles correctly.
fn is_clean_trailing_broadcast(src: &Shape, target: &Shape) -> bool {
    let s_dims = src.dims();
    let t_dims = target.dims();
    if s_dims.len() > t_dims.len() {
        return false;
    }
    let off = t_dims.len() - s_dims.len();
    for i in 0..s_dims.len() {
        match (s_dims[i], t_dims[off + i]) {
            (Dim::Static(a), Dim::Static(b)) if a == b => {}
            // Matching dynamic dims (same symbol) — treat as equal.
            (Dim::Dynamic(a), Dim::Dynamic(b)) if a == b => {}
            // Anything else (including 1 vs N) — not a clean trailing
            // pattern; needs Expand.
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::*;
    use rlx_ir::*;

    #[test]
    fn passthrough_for_equal_shapes() {
        let f = DType::F32;
        let mut g = Graph::new("eq");
        let a = g.input("a", Shape::new(&[4, 8], f));
        let b = g.input("b", Shape::new(&[4, 8], f));
        let c = g.binary(BinaryOp::Add, a, b, Shape::new(&[4, 8], f));
        g.set_outputs(vec![c]);
        let n_before = g.len();
        let g2 = run(g);
        assert_eq!(g2.len(), n_before, "no Expand inserted for equal shapes");
    }

    #[test]
    fn passthrough_for_trailing_bias() {
        let f = DType::F32;
        let mut g = Graph::new("trail");
        let a = g.input("a", Shape::new(&[4, 8], f));
        let b = g.input("b", Shape::new(&[8], f));
        let c = g.binary(BinaryOp::Add, a, b, Shape::new(&[4, 8], f));
        g.set_outputs(vec![c]);
        let n_before = g.len();
        let g2 = run(g);
        assert_eq!(
            g2.len(),
            n_before,
            "no Expand inserted for trailing-bias broadcast"
        );
    }

    #[test]
    fn inserts_expand_for_channel_broadcast() {
        let f = DType::F32;
        let mut g = Graph::new("chan");
        let a = g.input("a", Shape::new(&[1, 2, 3, 3], f));
        let b = g.input("b", Shape::new(&[1, 2, 1, 1], f));
        let c = g.binary(BinaryOp::Add, a, b, Shape::new(&[1, 2, 3, 3], f));
        g.set_outputs(vec![c]);
        let n_before = g.len();
        let g2 = run(g);
        assert!(
            g2.len() > n_before,
            "Expand should be inserted for [1,2,1,1] → [1,2,3,3]"
        );
        // Find the Expand node.
        let has_expand = g2.nodes().iter().any(|n| matches!(n.op, Op::Expand { .. }));
        assert!(has_expand);
    }
}
