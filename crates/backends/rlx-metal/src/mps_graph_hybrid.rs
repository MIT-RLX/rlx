// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Split MPSGraph lowering at GatedDeltaNet / Attention / DequantMatMul boundaries.
//!
//! Qwen3.5 decode graphs mix MPSGraph-eligible matmul/norm ops with host-only
//! GDN scans. Transformer AR graphs (Zonos, …) mix matmul/norm with attention
//! whose Q/K/V are slice-views of computed RoPE/GQA tensors — whole-graph
//! MPSGraph SDPA returns wrong values for that pattern. Whole-graph
//! `try_lower` bails on unsafe attention; this module builds alternating MPS
//! sub-graph plans + thunk ranges (attention / GDN run as thunks).

use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::mps_graph_lower::{MpsGraphPlan, try_lower_with_constants};

/// One step in a hybrid forward: either an MPS sub-graph or a thunk slice.
pub enum HybridStep {
    SubGraph {
        plan: MpsGraphPlan,
        /// Placeholder name → parent-graph node for arena binding.
        boundary_parent_ids: HashMap<String, NodeId>,
        /// Sub-graph output → parent-graph node for arena write-back.
        output_parent_ids: Vec<(NodeId, NodeId)>,
        /// Thunk indices in the parent schedule covered by this sub-graph.
        thunk_skip: Range<usize>,
    },
    Thunks(Range<usize>),
}

pub struct ExtractedSubgraph {
    pub graph: Graph,
    pub boundaries: HashMap<String, NodeId>,
    /// Sub-graph output id → parent-graph node id (arena binding).
    pub output_parent_ids: Vec<(NodeId, NodeId)>,
}

/// Build a lowerable sub-graph for `segment_nodes` (topo subset).
pub fn extract_subgraph(full: &Graph, segment_nodes: &[NodeId]) -> ExtractedSubgraph {
    let seg_set: HashSet<NodeId> = segment_nodes.iter().copied().collect();
    let mut boundary_parent: HashMap<String, NodeId> = HashMap::new();
    for &nid in segment_nodes {
        for &inp in &full.node(nid).inputs {
            if !seg_set.contains(&inp) {
                boundary_parent
                    .entry(format!("__boundary_{}", inp.0))
                    .or_insert(inp);
            }
        }
    }

    let mut sub = Graph::new(format!("{}_hybrid", full.name));
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();

    let mut boundary_names: Vec<String> = boundary_parent.keys().cloned().collect();
    boundary_names.sort();
    for name in &boundary_names {
        let parent_id = boundary_parent[name];
        let bn = full.node(parent_id);
        let new_id = match &bn.op {
            Op::Input { name: n } => sub.input(n.clone(), bn.shape.clone()),
            Op::Param { name: n } => sub.param(n.clone(), bn.shape.clone()),
            Op::Constant { data } => sub.add_node(
                Op::Constant { data: data.clone() },
                vec![],
                bn.shape.clone(),
            ),
            _ => sub.input(name.clone(), bn.shape.clone()),
        };
        map.insert(parent_id, new_id);
    }

    for &nid in segment_nodes {
        if map.contains_key(&nid) {
            continue;
        }
        let n = full.node(nid);
        let new_inputs: Vec<NodeId> = n
            .inputs
            .iter()
            .map(|&i| *map.get(&i).expect("dependency mapped"))
            .collect();
        let new_id = sub.add_node(n.op.clone(), new_inputs, n.shape.clone());
        map.insert(nid, new_id);
    }

    let graph_outputs: HashSet<NodeId> = full.outputs.iter().copied().collect();
    let mut outs = Vec::new();
    let mut output_parent_ids = Vec::new();
    for &nid in segment_nodes {
        let used_outside = full.users(nid).iter().any(|u| !seg_set.contains(u));
        if used_outside || graph_outputs.contains(&nid) {
            let sub_out = *map.get(&nid).unwrap();
            outs.push(sub_out);
            output_parent_ids.push((sub_out, nid));
        }
    }
    if outs.is_empty() {
        if let Some(&last) = segment_nodes.last() {
            let sub_out = *map.get(&last).unwrap();
            outs.push(sub_out);
            output_parent_ids.push((sub_out, last));
        }
    }
    sub.set_outputs(outs);

    ExtractedSubgraph {
        graph: sub,
        boundaries: boundary_parent,
        output_parent_ids,
    }
}

fn can_lower_dequant_in_mps(
    _graph: &Graph,
    _node_id: NodeId,
    _params_as_constants: Option<&HashMap<String, Vec<u8>>>,
) -> bool {
    // Never lower GGUF packed weights through MPSGraph: `mps_graph_lower` only
    // supports pre-dequantized F32 (`w_bytes.len() == k*n*4`), not K-quant U8.
    // Packed `Op::DequantMatMul` must use `Thunk::DequantMatMulGguf` (fused MSL
    // dequant+matmul on Metal when enabled).
    false
}

/// Build a hybrid plan when whole-graph lowering fails (typical Qwen3.5 decode).
///
/// Thunk ranges MUST index `ThunkSchedule.thunks`, which is one entry per
/// `graph.nodes()` in order (Input/Param/Constant → `Thunk::Nop`). Counting
/// only compute nodes (the old approach) desyncs Attention thunks from the
/// schedule and produces garbage on large transformer graphs (Zonos).
pub fn build_hybrid_plan(
    graph: &Graph,
    params_as_constants: Option<&HashMap<String, Vec<u8>>>,
) -> Option<Vec<HybridStep>> {
    // The hybrid MPSGraph→native handoff miscompiles a subgraph that produces
    // MULTIPLE boundary outputs feeding a single native op — CONFIRMED on both
    // GatedDeltaNet (q,k,v,g,beta) and Attention (q,k,v): at real dims the boundary
    // values come back finite-but-wrong (Kimi-K3 KDA drifts ~400/element; the MLA
    // layer's hidden state diverges grossly — different experts routed, wrong
    // token). These three ops are exactly where `is_split_boundary` cuts, so any
    // graph containing one goes entirely down the bit-exact native-thunk path
    // until the boundary handoff itself is fixed. (Correctness over the MPSGraph
    // fusion speedup; the native op was already a thunk regardless.)
    if graph.nodes().iter().any(|n| {
        matches!(
            n.op,
            Op::GatedDeltaNet { .. } | Op::Attention { .. } | Op::Lstm { .. }
        )
    }) {
        return None;
    }
    let mut steps: Vec<HybridStep> = Vec::new();
    let mut pending: Vec<NodeId> = Vec::new();
    let mut pending_idxs: Vec<usize> = Vec::new();

    let flush_mps =
        |pending: &mut Vec<NodeId>, pending_idxs: &mut Vec<usize>, steps: &mut Vec<HybridStep>| {
            if pending.is_empty() {
                return;
            }
            let start = *pending_idxs.iter().min().expect("pending idx");
            let end = *pending_idxs.iter().max().expect("pending idx") + 1;
            let extracted = extract_subgraph(graph, pending);
            match try_lower_with_constants(&extracted.graph, params_as_constants) {
                Some(plan) => {
                    steps.push(HybridStep::SubGraph {
                        plan,
                        boundary_parent_ids: extracted.boundaries,
                        output_parent_ids: extracted.output_parent_ids,
                        thunk_skip: start..end,
                    });
                }
                None => {
                    // Segment has an unsupported op — run it as thunks and keep
                    // trying MPSGraph on later segments (don't abort the hybrid).
                    steps.push(HybridStep::Thunks(start..end));
                }
            }
            pending.clear();
            pending_idxs.clear();
        };

    for (thunk_idx, node) in graph.nodes().iter().enumerate() {
        let id = node.id;
        let op = &node.op;
        if matches!(
            op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        ) {
            // Schedule slot is Nop — not part of MPS pending / attention splits.
            continue;
        }
        if matches!(
            op,
            Op::GatedDeltaNet { .. } | Op::Lstm { .. } | Op::Attention { .. }
        ) || (matches!(op, Op::DequantMatMul { .. })
            && !can_lower_dequant_in_mps(graph, id, params_as_constants))
        {
            flush_mps(&mut pending, &mut pending_idxs, &mut steps);
            steps.push(HybridStep::Thunks(thunk_idx..thunk_idx + 1));
        } else {
            pending.push(id);
            pending_idxs.push(thunk_idx);
        }
    }
    flush_mps(&mut pending, &mut pending_idxs, &mut steps);

    if steps.iter().all(|s| matches!(s, HybridStep::Thunks(_))) {
        return None;
    }
    Some(steps)
}

/// True when any step is an MPS sub-graph (worth the hybrid dispatch path).
pub fn hybrid_has_mps(steps: &[HybridStep]) -> bool {
    steps
        .iter()
        .any(|s| matches!(s, HybridStep::SubGraph { .. }))
}
