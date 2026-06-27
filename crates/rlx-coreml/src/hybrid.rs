// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Split mixed host/CoreML graphs into alternating host and MIL segments.

use std::collections::{HashMap, HashSet};

use rlx_ir::{Graph, NodeId, Op, Shape};

use crate::host_exec::is_host_op;
use crate::{CoremlError, Result};

/// One MIL subgraph plus synthetic inputs wired from host tensors.
#[derive(Debug, Clone)]
pub struct MilSegment {
    pub graph: Graph,
    pub extra_inputs: Vec<(String, Shape)>,
}

/// One execution step — host ops or a CoreML-compilable subgraph.
#[derive(Debug, Clone)]
pub enum Segment {
    Host(Vec<NodeId>),
    Mil(MilSegment),
}

/// How to run a graph that mixes host ops with MIL-lowerable compute.
#[derive(Debug, Clone)]
pub enum ExecutionPlan {
    /// Entire graph lowers to one CoreML model.
    MilOnly,
    /// Alternating host / CoreML segments in topological order.
    Segmented(Vec<Segment>),
}

/// True when the CoreML body has no compute (host-only graphs like `Sample`).
pub fn mil_body_is_trivial(graph: &Graph) -> bool {
    !graph.nodes().iter().any(|n| {
        !matches!(
            n.op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        )
    })
}

/// Classify `graph` for hybrid execution.
pub fn plan_execution(graph: &Graph) -> Result<ExecutionPlan> {
    let segments = build_segments(graph)?;
    if segments.len() == 1 {
        if let Segment::Mil(ref m) = segments[0] {
            if mil_body_is_trivial(&m.graph) {
                return Ok(ExecutionPlan::Segmented(segments));
            }
            return Ok(ExecutionPlan::MilOnly);
        }
    }
    if segments.is_empty() {
        return Ok(ExecutionPlan::MilOnly);
    }
    Ok(ExecutionPlan::Segmented(segments))
}

fn build_segments(graph: &Graph) -> Result<Vec<Segment>> {
    let order: Vec<NodeId> = graph.topo_order().collect();
    let mut segments: Vec<Segment> = Vec::new();
    let mut mil_nodes: Vec<NodeId> = Vec::new();
    let mut host_chain: Vec<NodeId> = Vec::new();

    for id in order {
        if is_host_op(&graph.node(id).op) {
            flush_mil(graph, &mut segments, &mut mil_nodes)?;
            host_chain.push(id);
        } else {
            flush_host(&mut segments, &mut host_chain);
            mil_nodes.push(id);
        }
    }
    flush_host(&mut segments, &mut host_chain);
    flush_mil(graph, &mut segments, &mut mil_nodes)?;
    Ok(segments)
}

fn flush_host(segments: &mut Vec<Segment>, host_chain: &mut Vec<NodeId>) {
    if !host_chain.is_empty() {
        segments.push(Segment::Host(std::mem::take(host_chain)));
    }
}

fn flush_mil(
    graph: &Graph,
    segments: &mut Vec<Segment>,
    mil_nodes: &mut Vec<NodeId>,
) -> Result<()> {
    if mil_has_compute(graph, mil_nodes) {
        let (g, extra) = build_mil_subgraph(graph, mil_nodes)?;
        segments.push(Segment::Mil(MilSegment {
            graph: g,
            extra_inputs: extra,
        }));
    }
    mil_nodes.clear();
    Ok(())
}

fn mil_has_compute(graph: &Graph, mil_nodes: &[NodeId]) -> bool {
    mil_nodes.iter().any(|&id| {
        !matches!(
            graph.node(id).op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        )
    })
}

fn build_mil_subgraph(
    graph: &Graph,
    mil_nodes: &[NodeId],
) -> Result<(Graph, Vec<(String, Shape)>)> {
    let mil_set: HashSet<NodeId> = mil_nodes.iter().copied().collect();
    let mut g = Graph::new(format!("{}_coreml", graph.name));
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut extra_inputs: Vec<(String, Shape)> = Vec::new();

    let clone_leaf = |g: &mut Graph, map: &mut HashMap<NodeId, NodeId>, old: NodeId| -> NodeId {
        if let Some(&n) = map.get(&old) {
            return n;
        }
        let node = graph.node(old);
        let new_id = match &node.op {
            Op::Input { name } => g.input(name, node.shape.clone()),
            Op::Param { name } => g.param(name, node.shape.clone()),
            Op::Constant { data } => g.add_node(
                Op::Constant { data: data.clone() },
                vec![],
                node.shape.clone(),
            ),
            _ => unreachable!("clone_leaf on non-leaf"),
        };
        map.insert(old, new_id);
        new_id
    };

    for &id in mil_nodes {
        let node = graph.node(id);
        let mut new_inputs = Vec::with_capacity(node.inputs.len());
        for &inp in &node.inputs {
            if mil_set.contains(&inp) {
                new_inputs.push(
                    *map.get(&inp)
                        .ok_or_else(|| CoremlError::Runtime(format!("mil map missing {inp:?}")))?,
                );
            } else if is_host_op(&graph.node(inp).op) {
                let name = format!("host_v{}", inp.0);
                if let std::collections::hash_map::Entry::Vacant(e) = map.entry(inp) {
                    let shape = graph.shape(inp).clone();
                    let nid = g.input(&name, shape.clone());
                    e.insert(nid);
                    extra_inputs.push((name, shape));
                }
                new_inputs.push(map[&inp]);
            } else {
                new_inputs.push(clone_leaf(&mut g, &mut map, inp));
            }
        }
        let new_id = g.append_node(node.op.clone(), new_inputs, node.shape.clone(), None);
        map.insert(id, new_id);
    }

    let outs = mil_segment_outputs(graph, &mil_set);
    let mapped: Vec<NodeId> = outs
        .iter()
        .map(|&oid| {
            map.get(&oid)
                .copied()
                .ok_or_else(|| CoremlError::Runtime(format!("missing mil output map for {oid:?}")))
        })
        .collect::<Result<Vec<_>>>()?;
    if mapped.is_empty() {
        return Err(CoremlError::Unsupported(
            "empty MIL segment (no outputs)".into(),
        ));
    }
    g.set_outputs(mapped);
    Ok((g, extra_inputs))
}

fn mil_segment_outputs(graph: &Graph, mil_set: &HashSet<NodeId>) -> Vec<NodeId> {
    let mut outs: Vec<NodeId> = graph
        .outputs
        .iter()
        .filter(|o| mil_set.contains(o))
        .copied()
        .collect();
    if !outs.is_empty() {
        return outs;
    }
    for &id in mil_set {
        for user in graph.users(id) {
            if !mil_set.contains(&user) {
                outs.push(id);
                break;
            }
        }
    }
    outs.sort_by_key(|id| id.0);
    outs.dedup();
    outs
}
