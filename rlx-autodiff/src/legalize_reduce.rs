// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Decompose multi-axis `Op::Reduce` into single-axis chains for backends
// that only support one reduction axis at a time (e.g. WGPU).

use rlx_ir::op::ReduceOp;
use rlx_ir::shape::Dim;
use rlx_ir::{Graph, NodeId, Op, Shape};

/// Replace every `Reduce` with `axes.len() > 1` by a chain of single-axis
/// reductions (`keep_dim=true` on each step; final reshape drops dims if needed).
pub fn legalize_multi_axis_reduce(mut g: Graph) -> Graph {
    let targets: Vec<(NodeId, ReduceOp, Vec<usize>, bool, Shape)> = g
        .nodes()
        .iter()
        .filter_map(|n| {
            if let Op::Reduce { op, axes, keep_dim } = &n.op {
                (axes.len() > 1).then_some((n.id, *op, axes.clone(), *keep_dim, n.shape.clone()))
            } else {
                None
            }
        })
        .collect();

    let mut remap: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();

    for (id, op, axes, keep_dim, final_shape) in targets {
        let input = g.node(id).inputs[0];
        let dtype = g.node(input).shape.dtype();
        let mut cur = input;
        let mut shape = g.node(cur).shape.clone();
        let mut sorted = axes;
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for &ax in &sorted {
            let mut dims: Vec<Dim> = shape.dims().to_vec();
            dims[ax] = Dim::Static(1);
            let step_shape = Shape::from_dims(&dims, dtype);
            cur = g.add_node(
                Op::Reduce {
                    op,
                    axes: vec![ax],
                    keep_dim: true,
                },
                vec![cur],
                step_shape,
            );
            shape = g.node(cur).shape.clone();
        }
        if !keep_dim {
            let new_shape_dims: Vec<i64> = final_shape
                .dims()
                .iter()
                .map(|d| match d {
                    Dim::Static(n) => *n as i64,
                    Dim::Dynamic(_) => -1,
                })
                .collect();
            cur = g.add_node(
                Op::Reshape {
                    new_shape: new_shape_dims,
                },
                vec![cur],
                final_shape,
            );
        }
        remap.insert(id, cur);
    }

    if remap.is_empty() {
        return g;
    }

    for node in g.nodes_mut() {
        for inp in &mut node.inputs {
            if let Some(&r) = remap.get(inp) {
                *inp = r;
            }
        }
    }
    for out in &mut g.outputs {
        if let Some(&r) = remap.get(out) {
            *out = r;
        }
    }
    g
}
