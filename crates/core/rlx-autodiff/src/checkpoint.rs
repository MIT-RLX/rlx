// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gradient checkpointing: trade recomputation for activation memory.
//!
//! [`grad_with_loss`](crate::grad_with_loss) mirrors the whole forward into
//! the backward graph and lets gradient kernels reference those mirrored
//! nodes. Correct, but it means every activation stays live from the moment
//! it is produced until its backward consumer runs — so peak memory grows
//! with depth. On a 24-layer Qwen3.5 trunk with 6144-wide DeltaNet
//! intermediates that is the difference between a graph that compiles and one
//! that asks for tens of gigabytes.
//!
//! This rewrites the combined graph so only **boundary** activations survive:
//! everything else is recomputed, just before the backward node that needs
//! it, from the nearest boundary. Classic `O(sqrt(n))`-style checkpointing,
//! expressed as a graph transform rather than a runtime hook.
//!
//! ```text
//!  before   f0 f1 f2 f3 f4 f5 ────────────────► b5 b4 b3 b2 b1 b0
//!           └───────── all six live across the whole backward ────┘
//!
//!  after    f0 f1 f2 f3 f4 f5 ──► f4' f5' b5 b4 ──► f2' f3' b3 b2 ──► …
//!           └ boundaries kept ┘   └ one segment live at a time ┘
//! ```
//!
//! # When it helps — and when it does NOT
//!
//! It relies on the memory planner reusing buffers by liveness: a recomputed
//! value is produced next to its single consumer, so its slot is reclaimed
//! immediately. Nothing device-specific, so it works on every backend.
//!
//! But the saving depends on the graph's *shape*, and it can go the wrong way:
//!
//! | graph | 1 segment | best | result |
//! |---|---|---|---|
//! | 32-deep elementwise chain | 143 KB | 32 KB | **−77%** |
//! | Qwen3.5-0.8B training backward | 116 GB | 149 GB | **+28%** |
//!
//! A chain has one cross-segment edge per boundary, so almost everything is
//! recomputable. A residual transformer has skip connections and shared
//! parameters threading across segments, so most nodes feed a later segment
//! and must be kept anyway — and then the recomputed duplicates are pure
//! addition. Segmenting by node index makes that worse, because a skip spans
//! many indices.
//!
//! **So this is a tool, not a default.** Measure with the planner and keep the
//! better graph; [`checkpoint_backward`] is deliberately a pure function so a
//! caller can do exactly that. A cost model that segments along the model's
//! own layer structure, rather than node index, would be the way to make it
//! reliable on transformers — that is not implemented.
//!
//! # What it costs
//!
//! One extra forward evaluation of each non-boundary node. Segment count is
//! the knob, and memory is **U-shaped** in it, not monotone: with too few
//! segments almost nothing is recomputed, and with too many almost every node
//! straddles a segment border and so has to be kept anyway. Measured on a
//! 32-deep chain the planned arena went 110 KB (1 segment) → 32 KB (8) → 47 KB
//! (16). [`suggest_segments`] picks the usual `sqrt(n)`, which is near the
//! bottom of that curve.

use std::collections::HashMap;

use rlx_ir::{Graph, NodeId, Op};

/// The conventional `sqrt(n)` segment count for an `n`-node forward graph.
///
/// Memory is U-shaped in the segment count (see the module docs), and
/// `sqrt(n)` sits near the minimum: it balances the number of boundaries kept
/// against the size of the one segment being recomputed.
pub fn suggest_segments(forward_nodes: usize) -> usize {
    (forward_nodes as f64).sqrt().round().max(1.0) as usize
}

/// Whether a node is a leaf — cheap to reference and never worth recomputing
/// (parameters are resident anyway, inputs are fed).
fn is_leaf(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
    )
}

/// Rewrite `bwd` so non-boundary forward activations are recomputed.
///
/// `forward` is the graph that was differentiated; `bwd` is the result of
/// [`grad_with_loss`](crate::grad_with_loss) on it, whose first
/// `forward.nodes().len()` nodes mirror the forward one-for-one and in order.
/// That correspondence is the contract this pass depends on.
///
/// `segments` is the number of checkpoint segments. `0` or `1` returns the
/// graph unchanged.
pub fn checkpoint_backward(forward: &Graph, bwd: &Graph, segments: usize) -> Graph {
    let n = forward.nodes().len();
    if segments <= 1 || n == 0 || bwd.nodes().len() <= n {
        return bwd.clone();
    }
    assert!(
        bwd.nodes().len() >= n,
        "backward graph is shorter than the forward it mirrors"
    );

    let seg_of = |i: usize| -> usize { i * segments / n };

    // A forward node must be KEPT when it is a leaf, a graph output, or feeds
    // a *later* segment — anything else can be rebuilt inside its own segment.
    // Note backward consumers deliberately do not force a keep: those are
    // exactly the references this pass exists to break.
    let mut keep = vec![false; n];
    for (i, node) in bwd.nodes().iter().take(n).enumerate() {
        if is_leaf(&node.op) {
            keep[i] = true;
        }
        for input in &node.inputs {
            let j = input.0 as usize;
            if j < n && seg_of(j) < seg_of(i) {
                keep[j] = true;
            }
        }
    }
    for o in &bwd.outputs {
        let j = o.0 as usize;
        if j < n {
            keep[j] = true;
        }
    }

    let mut out = Graph::new(bwd.name.clone());
    // Original forward nodes: still computed once, in order — the forward
    // pass has to run regardless. Non-kept ones simply die early now.
    let mut orig: Vec<NodeId> = Vec::with_capacity(n);
    for node in bwd.nodes().iter().take(n) {
        let inputs: Vec<NodeId> = node.inputs.iter().map(|i| orig[i.0 as usize]).collect();
        orig.push(out.add_node(node.op.clone(), inputs, node.shape.clone()));
    }

    // Lazily materialize a recomputed copy of forward node `i`, emitting its
    // whole dependency chain back to the nearest kept node.
    fn recompute(
        i: usize,
        bwd: &Graph,
        out: &mut Graph,
        orig: &[NodeId],
        keep: &[bool],
        cache: &mut HashMap<usize, NodeId>,
    ) -> NodeId {
        if keep[i] {
            return orig[i];
        }
        if let Some(id) = cache.get(&i) {
            return *id;
        }
        let node = &bwd.nodes()[i];
        let inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|d| recompute(d.0 as usize, bwd, out, orig, keep, cache))
            .collect();
        let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
        cache.insert(i, id);
        id
    }

    // Backward nodes, in their original order. A forward reference is
    // redirected to a recompute emitted right here, so its live range is the
    // few nodes between production and use instead of the whole graph.
    let mut mapping: HashMap<usize, NodeId> = HashMap::new();
    let mut cache: HashMap<usize, NodeId> = HashMap::new();
    let mut cur_seg = usize::MAX;
    for (idx, node) in bwd.nodes().iter().enumerate().skip(n) {
        // The backward walks the forward in reverse, so the segment it needs
        // moves monotonically downward. Dropping the cache on each move is
        // what actually frees the previous segment's recomputes.
        let needed = node
            .inputs
            .iter()
            .map(|d| d.0 as usize)
            .filter(|j| *j < n)
            .map(seg_of)
            .min();
        if let Some(s) = needed
            && s != cur_seg
        {
            cache.clear();
            cur_seg = s;
        }
        let inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|d| {
                let j = d.0 as usize;
                if j < n {
                    recompute(j, bwd, &mut out, &orig, &keep, &mut cache)
                } else {
                    mapping[&j]
                }
            })
            .collect();
        let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
        mapping.insert(idx, id);
    }

    let outputs: Vec<NodeId> = bwd
        .outputs
        .iter()
        .map(|o| {
            let j = o.0 as usize;
            if j < n { orig[j] } else { mapping[&j] }
        })
        .collect();
    out.set_outputs(outputs);
    out
}
