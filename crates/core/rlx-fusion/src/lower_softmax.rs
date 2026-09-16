// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Normalize [`Op::Softmax`] to the innermost axis, the one form backends
//! implement.
//!
//! Every backend's softmax reduces contiguous runs: the kernel is handed a
//! `rows x cols` pair and normalizes each run of `cols` neighbouring elements.
//! That is the softmax over the *innermost* axis and nothing else. Along any
//! other axis the elements are strided, so the same `rows`/`cols` describe
//! groups that are not the ones asked for.
//!
//! Nothing catches this today. `Op::Softmax` is in every backend's
//! `supported_ops`, so the "legalize what the backend rejects" path never
//! fires, and the kernel cannot tell a wrong grouping from a right one — it
//! returns a tensor that is a softmax of *something*, silently. On a 4-D input
//! it answers `axis: 2` with the axis-3 result, because `total / dim(2)` and
//! `total / dim(3)` coincide whenever the last two dimensions are equal.
//!
//! So this pass rewrites `axis != rank - 1` to `transpose(axis <-> innermost)`,
//! softmax, `transpose` back. The transposes are real copies, so a strided
//! kernel would be faster — but that is an optimization, and this is what makes
//! the op *correct* everywhere first.

use crate::pass::Pass;
use rlx_ir::*;
use std::collections::HashMap;

/// `perm` that swaps `axis` with the innermost one. A transposition is its own
/// inverse, so the same permutation undoes it.
fn swap_with_last_perm(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(axis, rank - 1);
    perm
}

/// Rewrite one `Softmax` (input already remapped) into the innermost-axis form.
pub fn lower_softmax(g: &mut Graph, x: NodeId, axis: i32, out_shape: Shape) -> NodeId {
    let rank = out_shape.rank();
    assert!(rank > 0, "Softmax on a rank-0 tensor");
    let ax = if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    };
    assert!(
        ax < rank,
        "Softmax axis {axis} out of range for rank {rank}"
    );

    let last = rank - 1;
    if ax == last {
        // Canonical already. Normalize a negative axis to its positive form so
        // backends do not each have to.
        return g.add_node(Op::Softmax { axis: last as i32 }, vec![x], out_shape);
    }

    let perm = swap_with_last_perm(rank, ax);
    let swapped_shape = shape::transpose_shape(g.shape(x), &perm).expect("softmax transpose");
    let swapped = g.add_node(
        Op::Transpose { perm: perm.clone() },
        vec![x],
        swapped_shape.clone(),
    );
    let reduced = g.add_node(
        Op::Softmax { axis: last as i32 },
        vec![swapped],
        swapped_shape,
    );
    g.add_node(Op::Transpose { perm }, vec![reduced], out_shape)
}

/// Rewrite every non-innermost `Op::Softmax` in the graph.
pub struct LowerSoftmaxAxis;

impl Pass for LowerSoftmaxAxis {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Softmax]
    }

    fn name(&self) -> &str {
        "lower_softmax_axis"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Self-gating: attention softmaxes the innermost axis, which is already
        // canonical, so the common case costs one scan and returns untouched.
        let needs_work = graph.nodes().iter().any(|n| match &n.op {
            Op::Softmax { axis } => {
                let rank = n.shape.rank();
                rank > 0 && *axis != (rank - 1) as i32
            }
            _ => false,
        });
        if !needs_work {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::Softmax { axis } = &node.op {
                let x = id_map[&node.inputs[0]];
                lower_softmax(&mut new_graph, x, *axis, node.shape.clone())
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
mod tests {
    use super::*;

    #[test]
    fn swapping_with_the_last_axis_is_its_own_inverse() {
        for rank in 1..6 {
            for axis in 0..rank {
                let perm = swap_with_last_perm(rank, axis);
                let dims: Vec<usize> = (0..rank).collect();
                let after: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
                let back: Vec<usize> = perm.iter().map(|&i| after[i]).collect();
                assert_eq!(back, dims, "rank {rank} axis {axis}");
                assert_eq!(after[rank - 1], axis, "axis must land innermost");
            }
        }
    }

    fn graph_with_softmax(dims: &[usize], axis: i32) -> Graph {
        let mut g = Graph::new("sm");
        let x = g.input("x", Shape::new(dims, DType::F32));
        let s = Shape::new(dims, DType::F32);
        let y = g.add_node(Op::Softmax { axis }, vec![x], s);
        g.set_outputs(vec![y]);
        g
    }

    /// Attention's softmax is already innermost. Rewriting it would put two
    /// full transposes into the hottest loop in the tree.
    #[test]
    fn the_canonical_form_is_left_alone() {
        let before = graph_with_softmax(&[2, 4, 8, 8], 3);
        let count = before.nodes().len();
        let after = LowerSoftmaxAxis.run(before);
        assert_eq!(
            after.nodes().len(),
            count,
            "canonical softmax was rewritten"
        );
    }

    #[test]
    fn a_non_innermost_axis_is_wrapped_in_transposes() {
        let after = LowerSoftmaxAxis.run(graph_with_softmax(&[2, 4, 8, 6], 1));
        let transposes = after
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Transpose { .. }))
            .count();
        assert_eq!(transposes, 2, "expected a transpose on each side");
        for n in after.nodes() {
            if let Op::Softmax { axis } = &n.op {
                assert_eq!(*axis, 3, "the softmax must end up on the innermost axis");
            }
        }
        // The graph still produces the shape it did before.
        let out = after.outputs[0];
        let shape = &after.node(out).shape;
        assert_eq!(
            (0..4)
                .map(|i| shape.dim(i).unwrap_static())
                .collect::<Vec<_>>(),
            vec![2, 4, 8, 6]
        );
    }

    /// A negative axis names the innermost one and must not grow transposes.
    #[test]
    fn a_negative_axis_is_resolved_before_it_is_judged() {
        let after = LowerSoftmaxAxis.run(graph_with_softmax(&[2, 4, 8, 6], -1));
        let transposes = after
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Transpose { .. }))
            .count();
        assert_eq!(transposes, 0, "-1 is already the innermost axis");
    }
}
