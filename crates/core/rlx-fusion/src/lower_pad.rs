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

//! Lower `Op::Pad` to primitives (`full`/`narrow`/`reverse`/`expand`/`concat`).
//!
//! This decomposition is the semantic oracle for [`PadMode`] on every backend
//! that does not claim `OpKind::Pad` (i.e. everything except Metal/CUDA, which
//! keep the native kernel). It is applied per-axis on the progressively padded
//! array, which matches NumPy `pad` / PyTorch `F.pad` corner behavior for the
//! `Reflect`/`Replicate`/`Circular` modes.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

fn static_dims(g: &Graph, x: NodeId) -> Vec<usize> {
    g.shape(x)
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect()
}

/// Flip element order along a single `axis` (shape unchanged).
fn reverse_axis(g: &mut Graph, x: NodeId, axis: usize) -> NodeId {
    let shape = g.shape(x).clone();
    g.add_node(Op::Reverse { axes: vec![axis] }, vec![x], shape)
}

/// Broadcast a size-1 `axis` up to `size` (the other dims must already match).
fn expand_axis(g: &mut Graph, x: NodeId, axis: usize, size: usize) -> NodeId {
    let mut dims = static_dims(g, x);
    dims[axis] = size;
    let target: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    let shape = Shape::new(&dims, g.shape(x).dtype());
    g.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![x],
        shape,
    )
}

/// Decompose one padded axis into a concat of the before-block, `cur`, and the
/// after-block, per `mode`. `cur` may already carry padding on earlier axes.
fn pad_axis(
    g: &mut Graph,
    cur: NodeId,
    axis: usize,
    before: usize,
    after: usize,
    mode: PadMode,
) -> NodeId {
    let dims = static_dims(g, cur);
    let n = dims[axis];
    let dtype = g.shape(cur).dtype();

    let slab = |g: &mut Graph, size: usize, value: f32| {
        let mut d = dims.clone();
        d[axis] = size;
        g.full(&d, value, dtype)
    };

    let mut parts: Vec<NodeId> = Vec::with_capacity(3);
    match mode {
        PadMode::Constant(v) => {
            if before > 0 {
                parts.push(slab(g, before, v));
            }
            parts.push(cur);
            if after > 0 {
                parts.push(slab(g, after, v));
            }
        }
        PadMode::Reflect => {
            assert!(
                before < n && after < n,
                "pad(reflect): pad ({before},{after}) must be < axis length {n} on axis {axis}"
            );
            // before block, output order x[before], x[before-1], …, x[1].
            if before > 0 {
                let head = g.narrow_(cur, axis, 1, before);
                parts.push(reverse_axis(g, head, axis));
            }
            parts.push(cur);
            // after block, output order x[n-2], x[n-3], …, x[n-1-after].
            if after > 0 {
                let tail = g.narrow_(cur, axis, n - 1 - after, after);
                parts.push(reverse_axis(g, tail, axis));
            }
        }
        PadMode::Replicate => {
            if before > 0 {
                let edge = g.narrow_(cur, axis, 0, 1);
                parts.push(expand_axis(g, edge, axis, before));
            }
            parts.push(cur);
            if after > 0 {
                let edge = g.narrow_(cur, axis, n - 1, 1);
                parts.push(expand_axis(g, edge, axis, after));
            }
        }
        PadMode::Circular => {
            assert!(
                before <= n && after <= n,
                "pad(circular): pad ({before},{after}) must be ≤ axis length {n} on axis {axis}"
            );
            // before block wraps the tail: x[n-before], …, x[n-1].
            if before > 0 {
                parts.push(g.narrow_(cur, axis, n - before, before));
            }
            parts.push(cur);
            // after block wraps the head: x[0], …, x[after-1].
            if after > 0 {
                parts.push(g.narrow_(cur, axis, 0, after));
            }
        }
    }

    if parts.len() == 1 {
        return parts.pop().unwrap();
    }
    g.concat_(parts, axis)
}

/// Lower a single `Op::Pad` (already remapped input `x`) to its decomposition.
pub fn lower_pad(g: &mut Graph, mut cur: NodeId, pads: &[[usize; 2]], mode: PadMode) -> NodeId {
    for (axis, &[before, after]) in pads.iter().enumerate() {
        if before == 0 && after == 0 {
            continue;
        }
        cur = pad_axis(g, cur, axis, before, after, mode);
    }
    cur
}

/// Rewrite every `Op::Pad` node into primitives.
pub struct LowerPad;

impl Pass for LowerPad {
    fn name(&self) -> &str {
        "lower_pad"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph.nodes().iter().any(|n| matches!(n.op, Op::Pad { .. })) {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::Pad { pads, mode } = &node.op {
                let x = id_map[&node.inputs[0]];
                lower_pad(&mut new_graph, x, pads, *mode)
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
