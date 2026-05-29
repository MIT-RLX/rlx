// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Lower `Op::Reduce` on non-last axes (and multi-axis reduce) for backends
//! that only implement reduction along the trailing dimension (e.g. wgpu).

use std::collections::HashMap;

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::ReduceOp;
use rlx_ir::*;

fn normalize_axes(axes: &[usize], rank: usize) -> Vec<usize> {
    let mut out: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if (a as i32) < 0 {
                (rank as i32 + a as i32) as usize
            } else {
                a
            }
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn needs_lower(axes: &[usize], rank: usize) -> bool {
    if axes.is_empty() || rank == 0 {
        return false;
    }
    let axes = normalize_axes(axes, rank);
    axes.len() > 1 || axes[0] != rank - 1
}

/// One reduce step at `axis` on the current tensor (wgpu: last axis only).
fn reduce_one_axis(
    g: &mut Graph,
    input: NodeId,
    op: ReduceOp,
    axis: usize,
    keep_dim: bool,
) -> NodeId {
    let in_shape = g.shape(input);
    let rank = in_shape.rank();
    let expected = shape::reduce_shape(in_shape, &[axis], keep_dim).expect("reduce shape");
    debug_assert!(axis < rank);
    if axis == rank - 1 {
        return g.reduce(input, op, vec![axis], keep_dim, expected);
    }
    // Move reduced axis to the end, reduce there, then restore layout if needed.
    let mut perm: Vec<usize> = (0..rank).filter(|i| *i != axis).collect();
    perm.push(axis);
    let t = g.transpose_(input, perm);
    let new_rank = g.shape(t).rank();
    let mid = g.reduce(
        t,
        op,
        vec![new_rank - 1],
        keep_dim,
        shape::reduce_shape(g.shape(t), &[new_rank - 1], keep_dim).expect("reduce shape"),
    );
    if g.shape(mid) == &expected {
        return mid;
    }
    let dims: Vec<i64> = expected
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    g.reshape_(mid, dims)
}

fn lower_reduce(
    g: &mut Graph,
    input: NodeId,
    op: ReduceOp,
    axes: &[usize],
    keep_dim: bool,
    out_shape: Shape,
) -> NodeId {
    let rank = g.shape(input).rank();
    let mut axes = normalize_axes(axes, rank);
    if axes.is_empty() {
        return input;
    }
    if !needs_lower(&axes, rank) {
        return g.reduce(input, op, axes, keep_dim, out_shape);
    }
    // Collapse from highest axis index first so indices stay valid.
    axes.sort_unstable_by(|a, b| b.cmp(a));
    let mut h = input;
    for (step, &ax) in axes.iter().enumerate() {
        let last = step + 1 == axes.len();
        let kd = last && keep_dim;
        h = reduce_one_axis(g, h, op, ax, kd);
    }
    if g.shape(h) != &out_shape {
        let dims: Vec<i64> = out_shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        h = g.reshape_(h, dims);
    }
    h
}

/// Rewrite reduces that are not last-axis-only into transpose + last-axis reduce.
pub struct LowerNonLastAxisReduce;

impl Pass for LowerNonLastAxisReduce {
    fn name(&self) -> &str {
        "lower_non_last_axis_reduce"
    }

    fn run(&self, graph: Graph) -> Graph {
        let needs = graph.nodes().iter().any(|n| {
            if let Op::Reduce { axes, .. } = &n.op {
                needs_lower(
                    axes,
                    n.inputs
                        .first()
                        .map(|&i| graph.shape(i).rank())
                        .unwrap_or(0),
                )
            } else {
                false
            }
        });
        if !needs {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::Reduce { op, axes, keep_dim } = &node.op {
                let input = id_map[&node.inputs[0]];
                lower_reduce(
                    &mut new_graph,
                    input,
                    *op,
                    axes,
                    *keep_dim,
                    node.shape.clone(),
                )
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
