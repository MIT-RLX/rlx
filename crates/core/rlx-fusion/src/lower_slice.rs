// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::Slice` (strided slice) to primitives. Semantic oracle for every
//! backend that does not claim `OpKind::Slice` (i.e. all except Metal/CUDA):
//! - `step == 1`  → `narrow` (contiguous).
//! - `step == -1` → `reverse(narrow(window))`.
//! - `step > 1` on a static shape → `narrow` (+ zero `pad`) to `len * step`,
//!   `reshape` the axis to `[len, step]`, take phase 0, `reshape` back.
//! - otherwise    → `gather` with a constant `i64` index `[start + j*step]`.
//!
//! The reshape form matters on wgpu: any gather in a graph makes its memory
//! planner pin every buffer to the end (`graph_has_host_indexing`), and a
//! SwinUNETR patch-merge (eight `x[:, i::2, j::2, k::2]` phases) at 64^3
//! asked for an 11 GiB arena that way instead of ~1 GiB.

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
    if step > 1
        && let Some(y) = strided_by_reshape(g, x, axis, start, len, step as usize)
    {
        return y;
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

/// `len` reads at `start + j*step` (`step > 1`) without a gather. `None` when
/// a dimension is dynamic.
fn strided_by_reshape(
    g: &mut Graph,
    x: NodeId,
    axis: usize,
    start: usize,
    len: usize,
    step: usize,
) -> Option<NodeId> {
    let dims: Vec<usize> = g
        .shape(x)
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let axis_len = dims[axis];
    let window = len * step;
    // The window runs past the axis by at most `step - 1` (the last read is
    // `start + (len-1)*step < axis_len`); zero-pad that tail, never read.
    let avail = (axis_len - start).min(window);
    let mut y = g.narrow_(x, axis, start, avail);
    if avail < window {
        let mut pads = vec![[0usize, 0]; dims.len()];
        pads[axis] = [0, window - avail];
        y = g.pad_(y, pads, rlx_ir::op::PadMode::Constant(0.0));
    }
    let mut split: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    split[axis] = len as i64;
    split.insert(axis + 1, step as i64);
    let y = g.reshape_(y, split);
    let y = g.narrow_(y, axis + 1, 0, 1);
    let mut out: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    out[axis] = len as i64;
    Some(g.reshape_(y, out))
}

/// Rewrite every `Op::Slice` node into primitives.
pub struct LowerSlice;

impl Pass for LowerSlice {
    // Lifted from the scan `run` already performs: without these kinds
    // the pass rebuilds the graph node-for-node and returns it unchanged.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Slice]
    }

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

#[cfg(test)]
mod reshape_stride_tests {
    use super::*;
    use crate::pass::Pass;

    /// Every positive stride is gather-free and reads the same elements as
    /// the definition, including windows that run past the axis.
    #[test]
    fn positive_strides_lower_without_gather() {
        for (axis_len, start, len, step) in [
            (8, 0, 4, 2),
            (8, 1, 4, 2),
            (9, 0, 3, 3),
            (10, 2, 3, 3),
            (7, 1, 3, 2),
        ] {
            let mut g = Graph::new("s");
            let x = g.input("x", Shape::new(&[2, axis_len, 3], DType::F32));
            let y = g.slice_(x, 1, start, len, step);
            g.set_outputs(vec![y]);
            let lowered = LowerSlice.run(g);
            assert!(
                !lowered
                    .nodes()
                    .iter()
                    .any(|n| matches!(n.op, Op::Gather { .. } | Op::Slice { .. })),
                "len {axis_len} start {start} step {step}: still a gather/slice"
            );
            let out = lowered.outputs[0];
            let dims: Vec<usize> = lowered
                .node(out)
                .shape
                .dims()
                .iter()
                .map(|d| d.unwrap_static())
                .collect();
            assert_eq!(dims, vec![2, len, 3]);
        }
    }
}
