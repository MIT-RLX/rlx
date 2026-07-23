// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Split mixed host/CoreML graphs into alternating host and MIL segments.

use std::collections::{HashMap, HashSet};

use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::host_exec::is_host_node;
use crate::{CoremlError, Result};

/// One MIL subgraph plus synthetic inputs wired from host tensors.
#[derive(Debug, Clone)]
pub struct MilSegment {
    pub graph: Graph,
    pub extra_inputs: Vec<(String, Shape)>,
    /// Maps each MIL-local output node id → the ORIGINAL global node id. The MIL
    /// subgraph renumbers nodes locally, so a global output `v80` may become a
    /// local `v10`; without this map the runner would write the buffer to
    /// `env[10]` instead of `env[80]`, losing the graph output (only the first
    /// segment escapes this because its local ids coincide with the global ones).
    pub out_local_to_global: HashMap<u32, u32>,
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
    if std::env::var("RLX_COREML_SEG_REPORT").as_deref() == Ok("1") {
        let nh = segments
            .iter()
            .filter(|s| matches!(s, Segment::Host(_)))
            .count();
        let nm = segments
            .iter()
            .filter(|s| matches!(s, Segment::Mil(_)))
            .count();
        eprintln!(
            "[coreml] hybrid segments: {nh} host + {nm} MIL ({nm} CoreML models to compile+load)"
        );
    }
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
        if is_host_node(graph, id) {
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
        let (g, extra, out_map) = build_mil_subgraph(graph, mil_nodes)?;
        segments.push(Segment::Mil(MilSegment {
            graph: g,
            extra_inputs: extra,
            out_local_to_global: out_map,
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
) -> Result<(Graph, Vec<(String, Shape)>, HashMap<u32, u32>)> {
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
            let is_leaf = matches!(
                graph.node(inp).op,
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
            );
            if mil_set.contains(&inp) {
                new_inputs.push(
                    *map.get(&inp)
                        .ok_or_else(|| CoremlError::Runtime(format!("mil map missing {inp:?}")))?,
                );
            } else if is_leaf {
                new_inputs.push(clone_leaf(&mut g, &mut map, inp));
            } else {
                // Any external NON-leaf value — a host op's output OR a prior MIL
                // segment's output — is fed into this MIL model as a boundary
                // input, sourced from the global `env` (which every segment
                // writes its outputs to) via the `host_v{id}` naming the runner
                // resolves. Previously only `is_host_op` inputs took this path,
                // so a later MIL segment consuming an earlier MIL segment's
                // output hit `clone_leaf on non-leaf`.
                let name = format!("host_v{}", inp.0);
                if let std::collections::hash_map::Entry::Vacant(e) = map.entry(inp) {
                    let shape = graph.shape(inp).clone();
                    // The global `env` (and the runner) carry every value as f32,
                    // and MIL feature I/O only accepts F32/F16/I32/F64 — never Bool
                    // or I64. Declare the boundary input F32 and, when the true
                    // dtype differs, cast it back inside the MIL graph so
                    // downstream ops see the right type. (moss's cross-segment
                    // masks are Bool; its indices small enough to be exact in f32.)
                    if shape.dtype() == DType::F32 {
                        let nid = g.input(&name, shape.clone());
                        e.insert(nid);
                        extra_inputs.push((name, shape));
                    } else {
                        let f32_shape = shape.clone().with_dtype(DType::F32);
                        let in_nid = g.input(&name, f32_shape.clone());
                        let cast_nid =
                            g.add_node(Op::Cast { to: shape.dtype() }, vec![in_nid], shape);
                        e.insert(cast_nid);
                        extra_inputs.push((name, f32_shape));
                    }
                }
                new_inputs.push(map[&inp]);
            }
        }
        let new_id = g.append_node(node.op.clone(), new_inputs, node.shape.clone(), None);
        map.insert(id, new_id);
    }

    let outs = mil_segment_outputs(graph, &mil_set);
    let mapped: Vec<NodeId> = outs
        .iter()
        .map(|&oid| {
            let local = map.get(&oid).copied().ok_or_else(|| {
                CoremlError::Runtime(format!("missing mil output map for {oid:?}"))
            })?;
            // MIL feature outputs must be F32/F16/I32/F64, and the runner reads
            // them back into the f32 `env` — so cast any Bool/I64 output to F32.
            let d = graph.shape(oid).dtype();
            if d == DType::F32 {
                Ok(local)
            } else {
                let f32_shape = graph.shape(oid).clone().with_dtype(DType::F32);
                Ok(g.add_node(Op::Cast { to: DType::F32 }, vec![local], f32_shape))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    if mapped.is_empty() {
        return Err(CoremlError::Unsupported(
            "empty MIL segment (no outputs)".into(),
        ));
    }
    // Reverse map: MIL-local output id → original global id, so the runner writes
    // each output buffer back to the correct global `env` slot.
    let out_local_to_global: HashMap<u32, u32> =
        outs.iter().zip(&mapped).map(|(g, l)| (l.0, g.0)).collect();
    g.set_outputs(mapped);
    Ok((g, extra_inputs, out_local_to_global))
}

fn mil_segment_outputs(graph: &Graph, mil_set: &HashSet<NodeId>) -> Vec<NodeId> {
    // A MIL segment must export both the final graph outputs it owns AND any of
    // its nodes consumed by a *later* segment (host or another MIL segment) — the
    // latter become that segment's boundary inputs, resolved through the global
    // `env`. (Previously an early return after collecting graph outputs dropped
    // the cross-segment-consumed nodes, so a later segment's `host_v{id}` input
    // had no producer in `env`.)
    //
    // Skip Input/Param/Constant leaves: `seed_leaf_env` already places them in
    // `env`, and exporting an Input as a MIL output reuses its feature name
    // (e.g. `speed`) for both the model input and output — CoreML then fails
    // load with "Error in declaring input <name> with error -1".
    let is_leaf = |id: NodeId| {
        matches!(
            graph.node(id).op,
            Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
        )
    };
    let mut outs: Vec<NodeId> = graph
        .outputs
        .iter()
        .filter(|o| mil_set.contains(o) && !is_leaf(**o))
        .copied()
        .collect();
    for &id in mil_set {
        if is_leaf(id) {
            continue;
        }
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

/// Expand fused ops claimed for coverage that CPU HostOp would Nop:
/// `FusedConvBiasAct`, `PartitionedConv`, `FusedTransformerLayer`.
pub fn lower_cpu_nop_fused_for_coreml(g: Graph) -> Graph {
    let needs = g.nodes().iter().any(|n| {
        matches!(
            n.op,
            Op::FusedConvBiasAct { .. }
                | Op::PartitionedConv { .. }
                | Op::FusedTransformerLayer { .. }
        )
    });
    if !needs {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = match &node.op {
            Op::FusedConvBiasAct { .. }
            | Op::PartitionedConv { .. }
            | Op::FusedTransformerLayer { .. } => {
                inline_unfused_fused_op(&mut out, &node.op, &new_inputs, &node.shape)
            }
            _ => out.add_node(node.op.clone(), new_inputs, node.shape.clone()),
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

fn inline_unfused_fused_op(out: &mut Graph, op: &Op, inputs: &[NodeId], shape: &Shape) -> NodeId {
    let mut mini = Graph::new("coreml_unfuse");
    let mut mini_ins = Vec::with_capacity(inputs.len());
    for (i, &src) in inputs.iter().enumerate() {
        let sh = out.node(src).shape.clone();
        mini_ins.push(mini.append_node(
            Op::Input {
                name: format!("in{i}"),
            },
            vec![],
            sh,
            None,
        ));
    }
    let out_id = mini.append_node(op.clone(), mini_ins, shape.clone(), None);
    mini.set_outputs(vec![out_id]);
    let expanded = rlx_opt::unfuse_fused_for_autodiff(mini);
    let mut map: HashMap<NodeId, NodeId> = HashMap::new();
    for n in expanded.nodes() {
        if let Op::Input { name } = &n.op {
            if let Some(rest) = name.strip_prefix("in") {
                if let Ok(i) = rest.parse::<usize>() {
                    map.insert(n.id, inputs[i]);
                    continue;
                }
            }
        }
        let mapped_ins: Vec<NodeId> = n.inputs.iter().map(|i| map[i]).collect();
        let id = out.add_node(n.op.clone(), mapped_ins, n.shape.clone());
        map.insert(n.id, id);
    }
    map[&expanded.outputs[0]]
}
