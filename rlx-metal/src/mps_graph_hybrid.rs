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
//! Split MPSGraph lowering at GatedDeltaNet / DequantMatMul boundaries.
//!
//! Qwen3.5 decode graphs mix MPSGraph-eligible matmul/norm/attn ops with
//! host-only GDN scans. Whole-graph `try_lower` bails on the first GDN;
//! this module builds alternating MPS sub-graph plans + thunk ranges.

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
                Op::Constant {
                    data: data.clone(),
                },
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
    graph: &Graph,
    node_id: NodeId,
    params_as_constants: Option<&HashMap<String, Vec<u8>>>,
) -> bool {
    let Some(params) = params_as_constants else {
        return false;
    };
    let node = graph.node(node_id);
    let Op::DequantMatMul { .. } = &node.op else {
        return false;
    };
    let w_id = node.inputs[1];
    let Op::Param { name } = &graph.node(w_id).op else {
        return false;
    };
    params.contains_key(name)
}

/// Build a hybrid plan when whole-graph lowering fails (typical Qwen3.5 decode).
pub fn build_hybrid_plan(
    graph: &Graph,
    params_as_constants: Option<&HashMap<String, Vec<u8>>>,
) -> Option<Vec<HybridStep>> {
    let schedulable: Vec<NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| {
            !matches!(
                n.op,
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
            )
        })
        .map(|n| n.id)
        .collect();

    let mut steps: Vec<HybridStep> = Vec::new();
    let mut pending: Vec<NodeId> = Vec::new();
    let mut thunk_idx = 0usize;

    let flush_mps = |pending: &mut Vec<NodeId>,
                         steps: &mut Vec<HybridStep>,
                         thunk_idx: usize|
     -> Option<usize> {
        if pending.is_empty() {
            return Some(thunk_idx);
        }
        let n_thunks = pending.len();
        let extracted = extract_subgraph(graph, pending);
        let plan = try_lower_with_constants(&extracted.graph, params_as_constants)?;
        steps.push(HybridStep::SubGraph {
            plan,
            boundary_parent_ids: extracted.boundaries,
            output_parent_ids: extracted.output_parent_ids,
            thunk_skip: thunk_idx..thunk_idx + n_thunks,
        });
        pending.clear();
        Some(thunk_idx + n_thunks)
    };

    for &id in &schedulable {
        let op = &graph.node(id).op;
        if matches!(op, Op::GatedDeltaNet { .. }) {
            thunk_idx = flush_mps(&mut pending, &mut steps, thunk_idx)?;
            steps.push(HybridStep::Thunks(thunk_idx..thunk_idx + 1));
            thunk_idx += 1;
        } else if matches!(op, Op::DequantMatMul { .. })
            && !can_lower_dequant_in_mps(graph, id, params_as_constants)
        {
            thunk_idx = flush_mps(&mut pending, &mut steps, thunk_idx)?;
            steps.push(HybridStep::Thunks(thunk_idx..thunk_idx + 1));
            thunk_idx += 1;
        } else {
            pending.push(id);
        }
    }
    thunk_idx = flush_mps(&mut pending, &mut steps, thunk_idx)?;

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
