// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Split a backward graph into **save** and **replay** halves.
//!
//! [`grad_with_loss`](crate::grad_with_loss) mirrors the whole forward into the
//! backward graph so gradient kernels can recompute the activations they need.
//! That is the right default for training, where each backward runs against a
//! fresh forward. It is the wrong shape for anything that reruns the *same*
//! forward under many cotangents — computing a Jacobian one block of output
//! dimensions at a time, per-sample gradients, influence functions — because
//! the forward is recomputed once per cotangent. PyTorch's `retain_graph`
//! covers this case; a graph has no tape to retain, so it needs the split.
//!
//! [`split_vjp`] cuts the backward graph in two:
//!
//! * **save** — the mirrored forward. Its outputs are the backward graph's own
//!   forward-side outputs, followed by every activation the backward half reads.
//! * **replay** — the gradient computation. The saved activations arrive as
//!   `Op::Param`, not `Op::Input`: parameters are bound once and persist across
//!   runs, so replaying `N` cotangents costs one bind rather than `N` feeds of
//!   the same bytes. `d_output` stays an input, since it is what varies.
//!
//! Run `save` once, bind its trailing outputs onto `replay`, then run `replay`
//! per cotangent.
//!
//! The cut is found structurally, with no cooperation from the gradient walk: a
//! node belongs to the backward half exactly when it is reachable from
//! `d_output`. Everything else is forward. Leaves (`Param` / `Input` /
//! `Constant`) that the backward half reads are *copied* into `replay` rather
//! than saved — a weight is already bindable, and round-tripping it through the
//! host would be pure cost.

use std::collections::{HashMap, HashSet};

use rlx_ir::{Graph, NodeId, Op, Shape};

/// Prefix for the parameters carrying saved activations into `replay`.
/// Unlikely to collide with a model's own weight names.
pub const SAVED_PARAM_PREFIX: &str = "__rlx_vjp_saved.";

#[derive(Debug, thiserror::Error)]
pub enum SplitVjpError {
    #[error("no `d_output` input found; split_vjp expects a graph produced by grad_with_loss*")]
    NoCotangent,
}

/// One activation carried from `save` to `replay`.
#[derive(Debug, Clone)]
pub struct SavedActivation {
    /// Parameter name on `replay`; bind with `set_param`.
    pub name: String,
    /// Index into `save`'s outputs holding this activation.
    pub save_output: usize,
    pub shape: Shape,
}

/// A backward graph cut into a forward half and a gradient half.
#[derive(Debug)]
pub struct SplitVjp {
    /// Runs the forward once. Outputs: the original graph's forward-side
    /// outputs (see [`Self::save_output_indices`]), then one per
    /// [`SavedActivation`].
    pub save: Graph,
    /// Runs per cotangent. Outputs correspond to
    /// [`Self::replay_output_indices`].
    pub replay: Graph,
    /// Activations to carry across, in `save` output order.
    pub saved: Vec<SavedActivation>,
    /// For each leading output of `save`, its index in the original graph's
    /// outputs.
    pub save_output_indices: Vec<usize>,
    /// For each output of `replay`, its index in the original graph's outputs.
    pub replay_output_indices: Vec<usize>,
}

impl SplitVjp {
    /// Total elements carried from `save` to `replay` — the memory this trades
    /// for not recomputing the forward.
    pub fn saved_elements(&self) -> usize {
        self.saved
            .iter()
            .map(|s| s.shape.num_elements().unwrap_or(0))
            .sum()
    }
}

fn is_leaf(op: &Op) -> bool {
    matches!(
        op,
        Op::Param { .. } | Op::Input { .. } | Op::Constant { .. }
    )
}

/// Cut `bwd` into save + replay halves. See the module docs.
///
/// `bwd` must come from [`grad_with_loss`](crate::grad_with_loss) or a sibling —
/// the `d_output` input is what identifies the gradient half.
pub fn split_vjp(bwd: &Graph) -> Result<SplitVjp, SplitVjpError> {
    let d_output = bwd
        .nodes()
        .iter()
        .find_map(|n| match &n.op {
            Op::Input { name } if name == "d_output" => Some(n.id),
            _ => None,
        })
        .ok_or(SplitVjpError::NoCotangent)?;

    // Backward half = everything reachable from the cotangent. Node ids are
    // topologically ordered, so one forward sweep closes the set.
    let mut is_bwd = vec![false; bwd.len()];
    is_bwd[d_output.0 as usize] = true;
    for node in bwd.nodes() {
        if node.inputs.iter().any(|i| is_bwd[i.0 as usize]) {
            is_bwd[node.id.0 as usize] = true;
        }
    }

    // The cut: non-leaf forward nodes the backward half reads.
    let mut cut: Vec<NodeId> = Vec::new();
    let mut in_cut: HashSet<NodeId> = HashSet::new();
    for node in bwd.nodes() {
        if !is_bwd[node.id.0 as usize] {
            continue;
        }
        for &i in &node.inputs {
            if is_bwd[i.0 as usize] || is_leaf(&bwd.node(i).op) {
                continue;
            }
            if in_cut.insert(i) {
                cut.push(i);
            }
        }
    }

    // ── save: the forward half ──
    let mut save = Graph::new(format!("{}_save", bwd.name));
    let mut save_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in bwd.nodes() {
        if is_bwd[node.id.0 as usize] {
            continue;
        }
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| save_map[i]).collect();
        let id = save.add_node(node.op.clone(), inputs, node.shape.clone());
        save_map.insert(node.id, id);
    }

    let mut save_outputs: Vec<NodeId> = Vec::new();
    let mut save_output_indices: Vec<usize> = Vec::new();
    for (idx, &out) in bwd.outputs.iter().enumerate() {
        if !is_bwd[out.0 as usize] {
            save_outputs.push(save_map[&out]);
            save_output_indices.push(idx);
        }
    }
    let n_leading = save_outputs.len();
    for c in &cut {
        save_outputs.push(save_map[c]);
    }
    save.set_outputs(save_outputs);

    // ── replay: the gradient half ──
    let mut replay = Graph::new(format!("{}_replay", bwd.name));
    let mut replay_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut saved: Vec<SavedActivation> = Vec::with_capacity(cut.len());
    for (k, &c) in cut.iter().enumerate() {
        let name = format!("{SAVED_PARAM_PREFIX}{k}");
        let shape = bwd.node(c).shape.clone();
        let id = replay.param(&name, shape.clone());
        replay_map.insert(c, id);
        saved.push(SavedActivation {
            name,
            save_output: n_leading + k,
            shape,
        });
    }

    for node in bwd.nodes() {
        if !is_bwd[node.id.0 as usize] {
            continue;
        }
        let mut inputs: Vec<NodeId> = Vec::with_capacity(node.inputs.len());
        for &i in &node.inputs {
            if let Some(&mapped) = replay_map.get(&i) {
                inputs.push(mapped);
                continue;
            }
            // A forward leaf the backward half reads: copy it rather than
            // shipping its value through the host. Leaves have no inputs, so
            // copying on demand cannot break topological order.
            let leaf = bwd.node(i);
            debug_assert!(is_leaf(&leaf.op), "unmapped non-leaf {i} in backward half");
            let id = replay.add_node(leaf.op.clone(), vec![], leaf.shape.clone());
            replay_map.insert(i, id);
            inputs.push(id);
        }
        let id = replay.add_node(node.op.clone(), inputs, node.shape.clone());
        replay_map.insert(node.id, id);
    }

    let mut replay_outputs: Vec<NodeId> = Vec::new();
    let mut replay_output_indices: Vec<usize> = Vec::new();
    for (idx, &out) in bwd.outputs.iter().enumerate() {
        if is_bwd[out.0 as usize] {
            replay_outputs.push(replay_map[&out]);
            replay_output_indices.push(idx);
        }
    }
    replay.set_outputs(replay_outputs);

    Ok(SplitVjp {
        save,
        replay,
        saved,
        save_output_indices,
        replay_output_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape};

    fn simple_bwd() -> Graph {
        // y = (x·w)² so the backward genuinely needs a forward activation.
        let f = DType::F32;
        let s = Shape::new(&[2], f);
        let mut g = Graph::new("t");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let h = g.binary(BinaryOp::Mul, x, w, s.clone());
        let y = g.binary(BinaryOp::Mul, h, h, s);
        g.set_outputs(vec![y]);
        crate::grad_with_loss(&g, &[x, w])
    }

    #[test]
    fn missing_cotangent_is_an_error() {
        let mut g = Graph::new("no_dout");
        let s = Shape::new(&[2], DType::F32);
        let x = g.input("x", s.clone());
        g.set_outputs(vec![x]);
        assert!(matches!(split_vjp(&g), Err(SplitVjpError::NoCotangent)));
    }

    #[test]
    fn replay_takes_the_cotangent_and_save_does_not() {
        let split = split_vjp(&simple_bwd()).unwrap();
        let has_dout = |g: &Graph| {
            g.nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        };
        assert!(has_dout(&split.replay), "replay must take the cotangent");
        assert!(!has_dout(&split.save), "save must not take the cotangent");
    }

    /// Saved activations must be parameters: they are bound once and reused
    /// across every replay, which is the whole point of the split.
    #[test]
    fn saved_activations_are_parameters() {
        let split = split_vjp(&simple_bwd()).unwrap();
        assert!(!split.saved.is_empty(), "expected at least one saved value");
        for s in &split.saved {
            let node = split
                .replay
                .nodes()
                .iter()
                .find(|n| matches!(&n.op, Op::Param { name } if *name == s.name))
                .unwrap_or_else(|| panic!("{} missing from replay", s.name));
            assert_eq!(node.shape, s.shape);
            assert!(s.save_output < split.save.outputs.len());
        }
    }

    /// Weights are copied into replay, not shipped through the host.
    #[test]
    fn forward_leaves_are_copied_not_saved() {
        let split = split_vjp(&simple_bwd()).unwrap();
        assert!(
            split
                .replay
                .nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Param { name } if name == "w")),
            "replay should declare `w` itself"
        );
        assert!(
            !split
                .saved
                .iter()
                .any(|s| s.shape.num_elements() == Some(0)),
            "no zero-sized saves"
        );
    }

    #[test]
    fn output_indices_partition_the_original_outputs() {
        let bwd = simple_bwd();
        let split = split_vjp(&bwd).unwrap();
        let mut all: Vec<usize> = split
            .save_output_indices
            .iter()
            .chain(&split.replay_output_indices)
            .copied()
            .collect();
        all.sort_unstable();
        assert_eq!(
            all,
            (0..bwd.outputs.len()).collect::<Vec<_>>(),
            "every original output must be produced by exactly one half"
        );
    }
}
