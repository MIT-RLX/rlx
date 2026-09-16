// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Normalize [`Op::ScatterAdd`] to the one form backends implement.
//!
//! Every backend's scatter kernel does the same thing: walk `[num_updates,
//! trailing]`, read an `f32`-encoded row index, accumulate into
//! `[out_dim, trailing]`. Scatter along any other axis, or an `i64` index
//! tensor, would need twelve separate kernel changes.
//!
//! Instead this pass rewrites both away before backend lowering:
//!
//! * **`axis != 0`** → `transpose(axis → front)`, scatter along 0,
//!   `transpose` back. The transposes are real copies, so a native strided
//!   kernel would be faster — but that is an optimization, and this is the
//!   thing that makes the op *correct* everywhere first.
//! * **`i64` indices** → `cast` to `f32`, matching what the kernels read.
//!
//! Running it unconditionally matters. `Op::ScatterAdd` is claimed by every
//! backend's `supported_ops`, so the usual "legalize what the backend rejects"
//! path never fires for it — a backend would happily accept `axis: 2` and
//! scatter along axis 0 anyway. That is the exact failure mode already recorded
//! for ROCm's partially-supported ops, so the axis-0 assumption is asserted in
//! the CPU thunk as a backstop.

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// `perm` that moves `axis` to the front, and its inverse.
///
/// With rlx's convention `out.dims[i] = in.dims[perm[i]]`, moving axis `a` to
/// the front is `[a, 0, 1, …, a-1, a+1, …]`.
fn move_to_front_perm(rank: usize, axis: usize) -> (Vec<usize>, Vec<usize>) {
    let mut fwd = Vec::with_capacity(rank);
    fwd.push(axis);
    fwd.extend((0..rank).filter(|&i| i != axis));
    let mut inv = vec![0usize; rank];
    for (i, &src) in fwd.iter().enumerate() {
        inv[src] = i;
    }
    (fwd, inv)
}

/// Rewrite one `ScatterAdd` (inputs already remapped) into the axis-0,
/// f32-index form. Returns the node producing `out_shape`.
pub fn lower_scatter_add(
    g: &mut Graph,
    updates: NodeId,
    indices: NodeId,
    axis: usize,
    out_shape: Shape,
) -> NodeId {
    // Kernels read the index buffer as f32 regardless of its declared dtype.
    let idx = if g.shape(indices).dtype() == DType::I64 {
        g.cast(indices, DType::F32)
    } else {
        indices
    };

    if axis == 0 {
        return g.add_node(Op::ScatterAdd { axis: 0 }, vec![updates, idx], out_shape);
    }

    let rank = out_shape.rank();
    assert!(
        axis < rank,
        "ScatterAdd axis {axis} out of range for rank {rank}"
    );
    let (fwd, inv) = move_to_front_perm(rank, axis);

    let upd_t = {
        let s = shape::transpose_shape(g.shape(updates), &fwd).expect("updates transpose");
        g.add_node(Op::Transpose { perm: fwd.clone() }, vec![updates], s)
    };
    let out_t_shape = shape::transpose_shape(&out_shape, &fwd).expect("output transpose");
    let scattered = g.add_node(Op::ScatterAdd { axis: 0 }, vec![upd_t, idx], out_t_shape);
    g.add_node(Op::Transpose { perm: inv }, vec![scattered], out_shape)
}

/// Rewrite every non-canonical `Op::ScatterAdd` in the graph.
pub struct LowerScatterAddAxis;

impl Pass for LowerScatterAddAxis {
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::ScatterAdd]
    }

    fn name(&self) -> &str {
        "lower_scatter_add_axis"
    }

    fn run(&self, graph: Graph) -> Graph {
        // Self-gating: the common axis-0 + f32 case is already canonical, so the
        // pass costs one scan and returns the graph untouched.
        let needs_work = graph.nodes().iter().any(|n| match &n.op {
            Op::ScatterAdd { axis } => {
                *axis != 0 || graph.node(n.inputs[1]).shape.dtype() == DType::I64
            }
            _ => false,
        });
        if !needs_work {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::ScatterAdd { axis } = &node.op {
                let updates = id_map[&node.inputs[0]];
                let indices = id_map[&node.inputs[1]];
                lower_scatter_add(&mut new_graph, updates, indices, *axis, node.shape.clone())
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
    fn move_to_front_perm_is_an_involution_pair() {
        for rank in 1..6 {
            for axis in 0..rank {
                let (fwd, inv) = move_to_front_perm(rank, axis);
                assert_eq!(fwd.len(), rank);
                // Applying fwd then inv is the identity.
                let dims: Vec<usize> = (0..rank).collect();
                let after: Vec<usize> = fwd.iter().map(|&i| dims[i]).collect();
                let back: Vec<usize> = inv.iter().map(|&i| after[i]).collect();
                assert_eq!(back, dims, "rank {rank} axis {axis}");
                assert_eq!(fwd[0], axis, "axis must land at the front");
            }
        }
    }

    /// The pass must be a no-op on the canonical form — otherwise every MoE
    /// unpermute in the tree grows two transposes.
    #[test]
    fn canonical_form_is_untouched() {
        let mut g = Graph::new("canon");
        let upd = g.input("u", Shape::new(&[4, 3], DType::F32));
        let idx = g.input("i", Shape::new(&[4], DType::F32));
        let out = g.add_node(
            Op::ScatterAdd { axis: 0 },
            vec![upd, idx],
            Shape::new(&[8, 3], DType::F32),
        );
        g.set_outputs(vec![out]);
        let before = g.nodes().len();
        let after = LowerScatterAddAxis.run(g);
        assert_eq!(after.nodes().len(), before);
        assert!(
            after
                .nodes()
                .iter()
                .all(|n| !matches!(n.op, Op::Transpose { .. }))
        );
    }

    /// A non-zero axis becomes transpose / scatter-0 / transpose.
    #[test]
    fn non_zero_axis_is_rewritten_to_axis_zero() {
        let mut g = Graph::new("ax1");
        let upd = g.input("u", Shape::new(&[3, 4], DType::F32));
        let idx = g.input("i", Shape::new(&[4], DType::F32));
        let out = g.add_node(
            Op::ScatterAdd { axis: 1 },
            vec![upd, idx],
            Shape::new(&[3, 8], DType::F32),
        );
        g.set_outputs(vec![out]);
        let after = LowerScatterAddAxis.run(g);

        assert!(
            after
                .nodes()
                .iter()
                .all(|n| !matches!(n.op, Op::ScatterAdd { axis } if axis != 0)),
            "no non-zero axis may survive the pass"
        );
        assert_eq!(
            after
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Transpose { .. }))
                .count(),
            2
        );
        // Output shape is preserved.
        let out_id = *after.outputs.first().unwrap();
        assert_eq!(after.node(out_id).shape.dims().len(), 2);
    }

    /// i64 indices are normalized to the f32 the kernels read.
    #[test]
    fn i64_indices_are_cast() {
        let mut g = Graph::new("i64");
        let upd = g.input("u", Shape::new(&[4, 3], DType::F32));
        let idx = g.input("i", Shape::new(&[4], DType::I64));
        let out = g.add_node(
            Op::ScatterAdd { axis: 0 },
            vec![upd, idx],
            Shape::new(&[8, 3], DType::F32),
        );
        g.set_outputs(vec![out]);
        let after = LowerScatterAddAxis.run(g);

        let sc = after
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::ScatterAdd { .. }))
            .expect("scatter survives");
        assert_eq!(
            after.node(sc.inputs[1]).shape.dtype(),
            DType::F32,
            "the index operand reaching a backend must be f32"
        );
    }
}
