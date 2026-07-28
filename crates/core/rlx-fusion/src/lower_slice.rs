// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::Slice` (strided slice) to primitives. Semantic oracle for every
//! backend that does not claim `OpKind::Slice` (i.e. all except Metal/CUDA):
//! - `step == 1`  → `narrow` (contiguous).
//! - `step == -1` → `reverse(narrow(window))`.
//! - otherwise    → `gather` with a constant `i64` index `[start + j*step]`.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// Decompose one `Op::Slice` (input `x` already remapped) to primitives.
pub fn lower_slice(
    g: &mut Graph,
    x: NodeId,
    axis: usize,
    start: usize,
    len: usize,
    step: i64,
) -> NodeId {
    if step == 1 {
        return g.narrow_(x, axis, start, len);
    }
    if step == -1 {
        // Reversed contiguous window: indices [start-(len-1) .. start].
        let lo = start - (len - 1);
        let win = g.narrow_(x, axis, lo, len);
        let shape = g.shape(win).clone();
        return g.add_node(Op::Reverse { axes: vec![axis] }, vec![win], shape);
    }
    // General stride: gather with a constant i64 index tensor.
    let idx: Vec<i64> = (0..len).map(|j| start as i64 + j as i64 * step).collect();
    let data: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
    let idx_node = g.add_node(
        Op::Constant { data },
        vec![],
        Shape::new(&[len], DType::I64),
    );
    g.gather_(x, idx_node, axis)
}

/// Rewrite every `Op::Slice` node into primitives.
pub struct LowerSlice;

impl Pass for LowerSlice {
    fn name(&self) -> &str {
        "lower_slice"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Slice { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::Slice {
                axis,
                start,
                len,
                step,
            } = &node.op
            {
                let x = id_map[&node.inputs[0]];
                lower_slice(&mut new_graph, x, *axis, *start, *len, *step)
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
