// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lane masks — per-lane reset expressed as **data**, not control flow.
//!
//! A batched workload runs `n_lanes` independent problems in one graph: parallel
//! RL environments, or a batch of decode sequences. Lanes finish at different
//! times, so periodically some subset needs to be reset while the rest keeps
//! running.
//!
//! The obvious implementation resets on the host — rebuild the state, rebind the
//! buffers, run again. That works, and it throws away the captured graph every
//! time it happens, because the graph is what encodes which buffers are read.
//! On CUDA the capture is dropped and re-taken (see `captured_graph = None` in
//! the rlx-cuda dispatch loop); the cost lands exactly on the boundary between
//! episodes, which for short episodes is most of them.
//!
//! The alternative is to make the reset part of the graph:
//!
//! ```text
//! state' = where(mask[lane] != 0, reset_value, state)
//! ```
//!
//! with `mask` a plain `[n_lanes]` **graph input**. Its *contents* change every
//! step; the *graph* never does. Nothing is rebound, no shape changes, so a
//! captured schedule stays valid across an arbitrary sequence of resets.
//!
//! This is the trick the Warp HOME-LBM environments use for their parallel
//! worlds: `partial_reset` launches mask-guarded kernels that early-return on
//! lanes which should not reset, and — unlike their full `reset` — never
//! invalidates the captured graph.
//!
//! # No new op
//!
//! This needs [`Op::Where`], [`Op::Compare`], [`Op::Reshape`] and [`Op::Expand`],
//! all of which every backend already has. Like the MRoPE seam, the useful thing
//! here is the composition and the guarantee attached to it, not a new opcode.
//!
//! ```
//! use rlx_ir::infer::GraphExt;
//! use rlx_ir::lanes::LaneExt;
//! use rlx_ir::{DType, Graph, Shape};
//!
//! let mut g = Graph::new("batched");
//! let state = g.input("state", Shape::new(&[4, 8], DType::F32));
//! let mask = g.input("reset", Shape::new(&[4], DType::F32));
//! let fresh = g.reset_lanes_to_(state, 0.0, mask);
//! g.set_outputs(vec![fresh]);
//! ```

use crate::infer::GraphExt;
use crate::op::CmpOp;
use crate::{DType, Dim, Graph, NodeId, Op, Shape};

/// Builders for lane-masked state updates.
///
/// A *lane mask* is a rank-1 `f32` tensor of length `n_lanes`. Nonzero selects
/// the lane for reset. `f32` rather than `bool` so it can be fed straight from
/// the host through the ordinary input path, with the comparison to zero done
/// inside the graph.
pub trait LaneExt {
    /// Broadcast a `[n_lanes]` mask to a boolean tensor shaped like `like`,
    /// i.e. `[n_lanes, 1, 1, …]` expanded across the trailing dims.
    ///
    /// Returns `None` if `like` is rank-0, has a dynamic leading dim, or its
    /// leading dim disagrees with the mask length — all cases where the
    /// broadcast is not well defined and silently guessing would be worse than
    /// refusing.
    fn lane_mask_like_(&mut self, mask: NodeId, like: NodeId) -> Option<NodeId>;

    /// `out[l, …] = if mask[l] != 0 { reset_to[l, …] } else { state[l, …] }`.
    ///
    /// `reset_to` must have the same shape as `state`.
    fn reset_lanes_(&mut self, state: NodeId, reset_to: NodeId, mask: NodeId) -> NodeId;

    /// [`reset_lanes_`](LaneExt::reset_lanes_) against a uniform scalar.
    fn reset_lanes_to_(&mut self, state: NodeId, value: f64, mask: NodeId) -> NodeId;

    /// Apply one mask to several state tensors at once — the usual case, since
    /// a lane's state is spread over many tensors (six moment fields, or a K
    /// and a V cache).
    ///
    /// The mask broadcast is built once per distinct shape, so `n` fields of the
    /// same shape cost one reshape/compare/expand chain rather than `n`.
    fn reset_lanes_many_(
        &mut self,
        states: &[NodeId],
        reset_to: &[NodeId],
        mask: NodeId,
    ) -> Vec<NodeId>;
}

/// Static dims of `shape`, or `None` if any is dynamic.
fn static_dims(shape: &Shape) -> Option<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            Dim::Dynamic(_) => None,
        })
        .collect()
}

impl LaneExt for Graph {
    fn lane_mask_like_(&mut self, mask: NodeId, like: NodeId) -> Option<NodeId> {
        let target = static_dims(&self.node(like).shape.clone())?;
        if target.is_empty() {
            return None;
        }
        let mask_dims = static_dims(&self.node(mask).shape.clone())?;
        if mask_dims.len() != 1 || mask_dims[0] != target[0] {
            return None;
        }
        let n_lanes = target[0];
        let dtype = self.node(mask).shape.dtype();

        // [L] -> [L, 1, 1, ...] so the lane axis lines up and the rest broadcast.
        let mut keep = vec![n_lanes];
        keep.extend(std::iter::repeat_n(1usize, target.len() - 1));
        let reshaped = self.add_node(
            Op::Reshape {
                new_shape: keep.iter().map(|&d| d as i64).collect(),
            },
            vec![mask],
            Shape::new(&keep, dtype),
        );

        // != 0 -> Bool. Comparing in the graph (rather than taking a Bool input)
        // keeps the host side a plain f32 buffer.
        let zero = self.constant(0.0, dtype);
        let cond = self.add_node(
            Op::Compare(CmpOp::Ne),
            vec![reshaped, zero],
            Shape::new(&keep, DType::Bool),
        );

        Some(self.add_node(
            Op::Expand {
                target_shape: target.iter().map(|&d| d as i64).collect(),
            },
            vec![cond],
            Shape::new(&target, DType::Bool),
        ))
    }

    fn reset_lanes_(&mut self, state: NodeId, reset_to: NodeId, mask: NodeId) -> NodeId {
        let shape = self.node(state).shape.clone();
        assert_eq!(
            shape,
            self.node(reset_to).shape,
            "reset_lanes_: state and reset value must have the same shape"
        );
        let cond = self
            .lane_mask_like_(mask, state)
            .expect("reset_lanes_: mask must be [n_lanes] matching the state's leading dim");
        self.add_node(Op::Where, vec![cond, reset_to, state], shape)
    }

    fn reset_lanes_to_(&mut self, state: NodeId, value: f64, mask: NodeId) -> NodeId {
        let shape = self.node(state).shape.clone();
        let dims = static_dims(&shape).expect("reset_lanes_to_: static shape required");
        let fill = self.full(&dims, value as f32, shape.dtype());
        self.reset_lanes_(state, fill, mask)
    }

    fn reset_lanes_many_(
        &mut self,
        states: &[NodeId],
        reset_to: &[NodeId],
        mask: NodeId,
    ) -> Vec<NodeId> {
        assert_eq!(
            states.len(),
            reset_to.len(),
            "reset_lanes_many_: one reset value per state tensor"
        );
        // Cache the broadcast per shape: the six-field case would otherwise emit
        // six identical reshape/compare/expand chains and lean on CSE to undo it.
        let mut cache: Vec<(Shape, NodeId)> = Vec::new();
        let mut out = Vec::with_capacity(states.len());
        for (&s, &r) in states.iter().zip(reset_to.iter()) {
            let shape = self.node(s).shape.clone();
            let cond = match cache.iter().find(|(sh, _)| *sh == shape) {
                Some((_, c)) => *c,
                None => {
                    let c = self
                        .lane_mask_like_(mask, s)
                        .expect("reset_lanes_many_: mask must match each state's leading dim");
                    cache.push((shape.clone(), c));
                    c
                }
            };
            out.push(self.add_node(Op::Where, vec![cond, r, s], shape));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_shape_signature(g: &Graph) -> Vec<(String, Vec<Dim>)> {
        g.nodes()
            .iter()
            .map(|n| (format!("{:?}", n.op.kind()), n.shape.dims().to_vec()))
            .collect()
    }

    #[test]
    fn lane_mask_broadcasts_to_state_rank() {
        let mut g = Graph::new("t");
        let state = g.input("s", Shape::new(&[4, 3, 2], DType::F32));
        let mask = g.input("m", Shape::new(&[4], DType::F32));
        let cond = g.lane_mask_like_(mask, state).expect("broadcast");
        assert_eq!(g.node(cond).shape.dims(), state_dims(&g, state).as_slice());
        assert_eq!(g.node(cond).shape.dtype(), DType::Bool);
    }

    fn state_dims(g: &Graph, id: NodeId) -> Vec<Dim> {
        g.node(id).shape.dims().to_vec()
    }

    /// A mismatched or dynamic mask must be refused, not silently broadcast to
    /// something plausible.
    #[test]
    fn lane_mask_refuses_ill_defined_broadcasts() {
        let mut g = Graph::new("t");
        let state = g.input("s", Shape::new(&[4, 3], DType::F32));
        let wrong_len = g.input("m1", Shape::new(&[5], DType::F32));
        assert!(g.lane_mask_like_(wrong_len, state).is_none());

        let rank2 = g.input("m2", Shape::new(&[4, 1], DType::F32));
        assert!(g.lane_mask_like_(rank2, state).is_none());

        let scalar = g.input("sc", Shape::from_dims(&[], DType::F32));
        let m = g.input("m3", Shape::new(&[4], DType::F32));
        assert!(g.lane_mask_like_(m, scalar).is_none());

        let dynamic = g.input(
            "d",
            Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(3)], DType::F32),
        );
        assert!(g.lane_mask_like_(m, dynamic).is_none());
    }

    /// **The load-bearing property.** The graph must not depend on which lanes
    /// are being reset — that is the whole reason this is worth doing, since a
    /// graph that changes is a capture that gets thrown away.
    ///
    /// Built twice with the mask as an input and compared structurally; the mask
    /// values never enter graph construction, so they cannot.
    #[test]
    fn graph_structure_is_independent_of_mask_contents() {
        let build = || {
            let mut g = Graph::new("b");
            let s = g.input("s", Shape::new(&[4, 8], DType::F32));
            let m = g.input("m", Shape::new(&[4], DType::F32));
            let out = g.reset_lanes_to_(s, 0.0, m);
            g.set_outputs(vec![out]);
            g
        };
        let a = build();
        let b = build();
        assert_eq!(graph_shape_signature(&a), graph_shape_signature(&b));
        // And the mask is a genuine input, not folded into a constant.
        assert!(
            a.nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Input { name } if name == "m")),
            "the mask must remain a graph input"
        );
    }

    /// Applying one mask to several same-shaped fields must build the broadcast
    /// once — six copies would be six extra reshape/compare/expand chains per
    /// step, relying on CSE to clean up after us.
    #[test]
    fn many_reuses_the_broadcast_per_shape() {
        let mut g = Graph::new("many");
        let shape = Shape::new(&[3, 5], DType::F32);
        let states: Vec<NodeId> = (0..6)
            .map(|i| g.input(format!("s{i}"), shape.clone()))
            .collect();
        let zeros: Vec<NodeId> = states
            .iter()
            .map(|_| g.zeros(&[3, 5], DType::F32))
            .collect();
        let mask = g.input("m", Shape::new(&[3], DType::F32));
        let out = g.reset_lanes_many_(&states, &zeros, mask);
        assert_eq!(out.len(), 6);
        let expands = g
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Expand { .. }))
            .count();
        assert_eq!(expands, 1, "broadcast should be built once for one shape");
        let wheres = g.nodes().iter().filter(|n| n.op == Op::Where).count();
        assert_eq!(wheres, 6);
    }
}
