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

//! PLAN: Schedule splitting for the Metal MPSGraph path.
//!
//! Background: today's Metal backend is all-or-nothing — either every op
//! lowers to MPSGraph (one big compile, one dispatch) or every op runs
//! as a thunk (N dispatches, ~50µs overhead each). The MPSGraph path
//! gives a 1.4-2.2× speedup at large batches BUT breaks parity on
//! transformer attention because MPSGraph's optimizer mishandles
//! slice-views of computed tensors (see
//! `tests/mps_attention_parity.rs::bisect_full_qkv_to_attention`).
//!
//! Schedule splitting is the structural workaround: split the topological
//! schedule at attention boundaries. Each contiguous run of
//! MPSGraph-compatible ops becomes one `MpsGraphSegment` (a sub-graph
//! compiled to MPSGraph with arena-bound input/output buffers); each
//! attention node stays as a `ThunkRange` (handled by the existing,
//! parity-correct thunk path). At run-time the executor walks the
//! segment list in order, mixing MPSGraph dispatches with thunk
//! dispatches.
//!
//! For a 6-layer BERT (the bench workload):
//! - All-thunks: 65 dispatches, ~3.25ms launch overhead floor
//! - Schedule-split: ~7 MPSGraph segments + 6 attention thunks = 13
//!   dispatches, ~0.65ms launch overhead — 5× reduction
//! - All-MPSGraph (if the bug were fixed): ~1 dispatch, ~0.05ms
//!
//! So splitting unlocks roughly 30% of the full MPSGraph win — a
//! meaningful step even before the underlying MPSGraph bug is fixed.
//!
//! Status: scaffolding only. The data model + segmenter are defined
//! here; the full executor wiring (reading boundary buffers from the
//! arena, dispatching segments mixed with thunks) is the next chunk
//! of work.

use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};

/// Which kind of segment this is.
#[derive(Debug)]
pub enum Segment {
    /// Range of `ThunkSchedule.thunks` indices to dispatch via the
    /// existing thunk path. Inclusive start, exclusive end.
    Thunks { start: usize, end: usize },
    /// One MPSGraph-compiled sub-graph with arena-bound boundary
    /// buffers. Dispatched as a single `runWithMTLCommandQueue:` call.
    MpsGraph(MpsGraphSegment),
}

/// A sub-graph compiled to MPSGraph. Holds the compiled plan plus the
/// bookkeeping needed to bind the boundary buffers (which arena slots
/// feed in, which arena slots receive the outputs) at run-time.
#[derive(Debug)]
pub struct MpsGraphSegment {
    /// The set of NodeIds this segment owns. Stored for diagnostics
    /// and verification; the actual MPSGraph plan references them by
    /// the placeholder/result tensors below.
    pub nodes: Vec<NodeId>,
    /// Boundary inputs — IR nodes that feed into this segment from
    /// outside. At run-time their arena slots are bound to the
    /// corresponding MPSGraph placeholders before dispatch.
    pub boundary_inputs: Vec<NodeId>,
    /// Boundary outputs — IR nodes inside this segment whose values
    /// are read by later segments (or by the final graph output).
    /// MPSGraph writes their values back to the arena slots.
    pub boundary_outputs: Vec<NodeId>,
    // TODO(plan): the actual `MpsGraphPlan` once `try_lower` learns
    // to lower a sub-graph against an explicit input set rather than
    // against `Op::Input { name }` declarations. The current
    // `try_lower` only handles whole graphs.
}

/// Predicate: true when the op is supported by MPSGraph lowering AND
/// safe to include in an MPSGraph segment. Today's known-broken case
/// is `Op::Attention` whose Q/K/V come from narrows of a computed
/// QKV (see `mg.attention`'s KNOWN BUG comment) — we exclude all
/// `Op::Attention` nodes conservatively to dodge the bug entirely.
fn mps_segment_eligible(op: &Op) -> bool {
    use rlx_ir::op::Activation;
    match op {
        // Boundary ops never go in any segment — they're either
        // graph-level inputs/params (handled separately as
        // placeholders) or stay on the thunk path.
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } => false,
        // Known-broken on MPSGraph; always thunk.
        Op::Attention { .. } => false,
        // The set MPSGraph lowering currently supports (mirrors the
        // match in `mps_graph_lower::try_lower`). Anything not here
        // falls into a thunk segment by default.
        Op::MatMul
        | Op::FusedMatMulBiasAct { .. }
        | Op::Activation(Activation::Gelu)
        | Op::Activation(Activation::Silu)
        | Op::Binary(_)
        | Op::LayerNorm { .. }
        | Op::FusedResidualLN { .. }
        | Op::Reshape { .. }
        | Op::Expand { .. }
        | Op::Cast { .. }
        | Op::Gather { .. }
        | Op::Narrow { .. }
        | Op::FusedSwiGLU { .. }
        | Op::Concat { .. }
        | Op::Rope { .. } => true,
        _ => false,
    }
}

/// Walk the graph in topo order and produce a segment list. Each segment
/// is either a maximal contiguous run of `mps_segment_eligible` ops
/// (becomes an MpsGraph segment) or a single non-eligible op (becomes
/// a single-thunk segment).
///
/// Returns the segment list. Segments are in dispatch order. The thunk
/// indices match the order of nodes in `graph.nodes()` — so the caller
/// (which has the matching `ThunkSchedule`) can reuse the same indices.
pub fn segment(graph: &Graph) -> Vec<Segment> {
    let nodes = graph.nodes();
    let mut segments: Vec<Segment> = Vec::new();

    // Pass 1: compute, for each node, the set of NodeIds it's read by
    // (used to identify boundary outputs of MPSGraph segments).
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in nodes {
        for &input in &node.inputs {
            consumers.entry(input).or_default().push(node.id);
        }
    }
    let graph_outputs: HashSet<NodeId> = graph.outputs.iter().copied().collect();

    // Pass 2: walk in node order. Group consecutive eligible ops into
    // a pending segment; flush whenever we hit an ineligible op or end.
    let mut pending_nodes: Vec<NodeId> = Vec::new();
    let mut idx_in_schedule: usize = 0;

    let flush_mps = |pending: &mut Vec<NodeId>, segs: &mut Vec<Segment>| {
        if pending.is_empty() {
            return;
        }
        // Determine boundary inputs: any input ID consumed by a
        // pending node that is NOT itself in `pending`.
        let pending_set: HashSet<NodeId> = pending.iter().copied().collect();
        let mut bin_set: HashSet<NodeId> = HashSet::new();
        for &id in pending.iter() {
            for &input in &graph_node(graph, id).inputs {
                if !pending_set.contains(&input) {
                    bin_set.insert(input);
                }
            }
        }
        // Boundary outputs: any pending node read by a non-pending
        // consumer OR by `graph.outputs`.
        let mut bout_set: HashSet<NodeId> = HashSet::new();
        for &id in pending.iter() {
            let used_outside = consumers
                .get(&id)
                .map(|cs| cs.iter().any(|c| !pending_set.contains(c)))
                .unwrap_or(false);
            if used_outside || graph_outputs.contains(&id) {
                bout_set.insert(id);
            }
        }
        let mut boundary_inputs: Vec<NodeId> = bin_set.into_iter().collect();
        boundary_inputs.sort_by_key(|id| id.0);
        let mut boundary_outputs: Vec<NodeId> = bout_set.into_iter().collect();
        boundary_outputs.sort_by_key(|id| id.0);
        segs.push(Segment::MpsGraph(MpsGraphSegment {
            nodes: pending.clone(),
            boundary_inputs,
            boundary_outputs,
        }));
        pending.clear();
    };

    for node in nodes {
        // Skip leaf nodes — they're handled as placeholders / arena
        // pre-population, not as schedule entries.
        if matches!(
            node.op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        ) {
            idx_in_schedule += 1;
            continue;
        }
        if mps_segment_eligible(&node.op) {
            pending_nodes.push(node.id);
        } else {
            // Flush any pending MPSGraph segment, then emit a
            // single-op thunk segment for this op.
            flush_mps(&mut pending_nodes, &mut segments);
            segments.push(Segment::Thunks {
                start: idx_in_schedule,
                end: idx_in_schedule + 1,
            });
        }
        idx_in_schedule += 1;
    }
    flush_mps(&mut pending_nodes, &mut segments);

    // Coalesce adjacent thunk segments into one range to amortize
    // dispatch loop overhead.
    let mut coalesced: Vec<Segment> = Vec::new();
    for seg in segments {
        match (coalesced.last_mut(), seg) {
            (Some(Segment::Thunks { end, .. }), Segment::Thunks { start: ns, end: ne })
                if *end == ns =>
            {
                *end = ne;
            }
            (_, s) => coalesced.push(s),
        }
    }
    coalesced
}

#[inline]
fn graph_node(graph: &Graph, id: NodeId) -> &rlx_ir::Node {
    graph.node(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp, MaskKind};
    use rlx_ir::{DType, Op, Shape};

    fn count_segments(segs: &[Segment]) -> (usize, usize) {
        let mut mps = 0;
        let mut th = 0;
        for s in segs {
            match s {
                Segment::MpsGraph(_) => mps += 1,
                Segment::Thunks { .. } => th += 1,
            }
        }
        (mps, th)
    }

    #[test]
    fn segment_pure_mlp_yields_one_mps_segment() {
        // A graph with no Attention nodes should be one big MPSGraph
        // segment.
        let f = DType::F32;
        let mut g = Graph::new("mlp");
        let x = g.input("x", Shape::new(&[1, 4], f));
        let w = g.param("w", Shape::new(&[4, 4], f));
        let mm = g.matmul(x, w, Shape::new(&[1, 4], f));
        let r = g.activation(Activation::Gelu, mm, Shape::new(&[1, 4], f));
        g.set_outputs(vec![r]);

        let segs = segment(&g);
        let (mps, th) = count_segments(&segs);
        assert_eq!(
            (mps, th),
            (1, 0),
            "pure MLP should be 1 MPSGraph segment + 0 thunk segments, got mps={mps} th={th}"
        );
    }

    #[test]
    fn segment_around_attention_splits_correctly() {
        // matmul → attention → matmul: should split into
        // [MPSGraph: matmul] + [Thunks: attention] + [MPSGraph: matmul].
        let f = DType::F32;
        let mut g = Graph::new("attn_split");
        let x = g.input("x", Shape::new(&[1, 4, 8], f));
        let mask = g.input("mask", Shape::new(&[1, 4], f));
        let w1 = g.param("w1", Shape::new(&[8, 24], f));
        let qkv = g.matmul(x, w1, Shape::new(&[1, 4, 24], f));
        let q = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 0,
                len: 8,
            },
            vec![qkv],
            Shape::new(&[1, 4, 8], f),
        );
        let k = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 8,
                len: 8,
            },
            vec![qkv],
            Shape::new(&[1, 4, 8], f),
        );
        let v = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 16,
                len: 8,
            },
            vec![qkv],
            Shape::new(&[1, 4, 8], f),
        );
        let attn = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 4,
                mask_kind: MaskKind::Custom,
            },
            vec![q, k, v, mask],
            Shape::new(&[1, 4, 8], f),
        );
        let w2 = g.param("w2", Shape::new(&[8, 8], f));
        let out = g.matmul(attn, w2, Shape::new(&[1, 4, 8], f));
        g.set_outputs(vec![out]);

        let segs = segment(&g);
        let (mps, th) = count_segments(&segs);
        // Expected layout: [MPS: mm + 3 narrows] [Thunk: attn] [MPS: mm]
        assert_eq!(
            (mps, th),
            (2, 1),
            "expected 2 MPSGraph segments + 1 thunk segment around attention, got mps={mps} th={th}"
        );
    }

    #[test]
    fn segment_attention_boundary_inputs_correct() {
        // Verify that the post-attention MPSGraph segment correctly
        // lists the attention output as a boundary input.
        let f = DType::F32;
        let mut g = Graph::new("attn_boundary");
        let x = g.input("x", Shape::new(&[1, 4, 8], f));
        let mask = g.input("mask", Shape::new(&[1, 4], f));
        let w1 = g.param("w1", Shape::new(&[8, 24], f));
        let qkv = g.matmul(x, w1, Shape::new(&[1, 4, 24], f));
        let q = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 0,
                len: 8,
            },
            vec![qkv],
            Shape::new(&[1, 4, 8], f),
        );
        let k = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 8,
                len: 8,
            },
            vec![qkv],
            Shape::new(&[1, 4, 8], f),
        );
        let v = g.add_node(
            Op::Narrow {
                axis: 2,
                start: 16,
                len: 8,
            },
            vec![qkv],
            Shape::new(&[1, 4, 8], f),
        );
        let attn_id = g.add_node(
            Op::Attention {
                num_heads: 2,
                head_dim: 4,
                mask_kind: MaskKind::Custom,
            },
            vec![q, k, v, mask],
            Shape::new(&[1, 4, 8], f),
        );
        let w2 = g.param("w2", Shape::new(&[8, 8], f));
        let _ = g.add_node(
            Op::Binary(BinaryOp::Add),
            vec![attn_id, w2],
            Shape::new(&[1, 4, 8], f),
        );
        g.set_outputs(vec![attn_id]);

        let segs = segment(&g);
        // Find the post-attention MPSGraph segment (if any) and
        // verify it lists `attn_id` as a boundary input.
        let mut found_post = false;
        let mut saw_attn_thunk = false;
        for s in &segs {
            match s {
                Segment::Thunks { .. } => saw_attn_thunk = true,
                Segment::MpsGraph(seg) if saw_attn_thunk => {
                    found_post = seg.boundary_inputs.contains(&attn_id);
                    break;
                }
                _ => {}
            }
        }
        // Either there's a post-attn segment with attn as input, OR
        // the `Add` consumes attn directly via a thunk segment.
        // The current segmenter sees Binary(Add) as eligible, so it
        // builds a post segment.
        assert!(
            found_post,
            "post-attention MPSGraph segment should list attention output as boundary input"
        );
    }
}
