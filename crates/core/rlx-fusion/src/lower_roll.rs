// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::Roll` (cyclic shift) to primitives. Semantic oracle for every
//! backend — no backend claims `OpKind::Roll`, so this pass is the only
//! implementation, which means it is also the definition.
//!
//! One axis, shift `k` (already reduced into `0..n`):
//!
//! ```text
//! roll(x, k, axis) = concat([narrow(x, axis, n-k, k), narrow(x, axis, 0, n-k)], axis)
//! ```
//!
//! because `out[i] = x[(i − k) mod n]` puts the last `k` elements first. Both
//! halves are contiguous narrows, so the cost is exactly one copy of the tensor
//! regardless of `k`.
//!
//! Multiple `(shift, dim)` pairs compose left to right. A shift that reduces to
//! zero is dropped rather than emitted as a degenerate concat of an empty
//! narrow — an empty narrow is not universally legal, and the identity is the
//! right answer anyway.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// Decompose one `Op::Roll` (input `x` already remapped) into narrows + concats.
///
/// Returns `x` unchanged when every shift is a no-op.
pub fn lower_roll(g: &mut Graph, x: NodeId, shifts: &[i64], dims: &[usize]) -> NodeId {
    let mut cur = x;
    for (&shift, &axis) in shifts.iter().zip(dims.iter()) {
        let n = match g.shape(cur).dims().get(axis).copied() {
            Some(Dim::Static(n)) => n,
            // A dynamic axis has no compile-time length, so the split point is
            // unknown. Leaving the node alone is wrong (no backend runs it), so
            // this is a hard error rather than a silent pass-through.
            _ => panic!("roll: axis {axis} must have a static length"),
        };
        if n == 0 {
            continue;
        }
        // Rust's `%` keeps the sign of the dividend; rem_euclid gives the
        // mathematical modulus, so negative shifts land in `0..n` as intended.
        let k = shift.rem_euclid(n as i64) as usize;
        if k == 0 {
            continue;
        }
        let tail = g.narrow_(cur, axis, n - k, k);
        let head = g.narrow_(cur, axis, 0, n - k);
        cur = g.concat_(vec![tail, head], axis);
    }
    cur
}

/// Rewrite every `Op::Roll` node into primitives.
pub struct LowerRoll;

impl Pass for LowerRoll {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Roll]
    }

    fn name(&self) -> &str {
        "lower_roll"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Roll { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::Roll { shifts, dims } = &node.op {
                let x = id_map[&node.inputs[0]];
                lower_roll(&mut new_graph, x, shifts, dims)
            } else {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}
