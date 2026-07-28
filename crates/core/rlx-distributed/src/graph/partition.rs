// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Partition a computation [`Graph`] into `N` contiguous **pipeline stages** for
//! multi-node execution. Each stage is a self-contained subgraph that a
//! different machine compiles and runs; activations cross stage boundaries as
//! named tensors (see [`crate::transport`]).
//!
//! The point is **RAM pooling**: a model too large for any single node is split
//! so each node materializes only *its* stage's parameters (`Op::Param` leaves
//! are copied into the stage that consumes them, never shared), plus the small
//! activation tensors flowing through. Compute nodes are assigned to stages by
//! contiguous topological order (nodes are stored in topo order); a cross-stage
//! edge between two compute nodes becomes a named **boundary** tensor —
//! `Op::Input` on the consumer side, an extra graph output on the producer side.

use rlx_ir::{Dim, Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};

/// Concrete static dims of a node's shape (dynamic dims → 0; static graphs have none).
fn static_dims(g: &Graph, id: NodeId) -> Vec<usize> {
    g.shape(id)
        .dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            Dim::Dynamic(_) => 0,
        })
        .collect()
}

/// One pipeline stage: a compilable subgraph plus the boundary tensor names it
/// consumes / produces and the parameter names its worker must load.
///
/// `Serialize`/`Deserialize` so a coordinator can partition once and ship each
/// stage to a remote worker (the `Graph` serializes via rlx-ir's `serialize`
/// feature). Weights are NOT part of a `Stage` — the worker pulls them from a
/// [`crate::source::ParamSource`] by `params` name.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Stage {
    pub index: usize,
    /// Self-contained subgraph (leaves materialized, boundaries as `Op::Input`).
    pub graph: Graph,
    /// `Op::Input` names this stage is fed: original model inputs materialized
    /// here + cross-stage boundary tensors from earlier stages.
    pub inputs: Vec<String>,
    /// Boundary tensor names this stage emits (order matches `graph.outputs`),
    /// consumed by later stages or, for the final stage, the model outputs.
    pub outputs: Vec<String>,
    /// Shapes of `outputs` (order matches `outputs` / `graph.outputs`).
    pub output_shapes: Vec<Vec<usize>>,
    /// `Op::Param` names in `graph` — the weights this stage's worker loads.
    pub params: Vec<String>,
}

/// Canonical boundary-tensor name for the value produced by original node `id`.
fn boundary_name(id: NodeId) -> String {
    format!("__stage_boundary_{}", id.0)
}

fn is_leaf(g: &Graph, id: NodeId) -> bool {
    matches!(
        g.node(id).op,
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
    )
}

/// Assign each compute node (index `i` of `m` in topo order) to a stage by
/// balanced contiguous ranges. Exposed so callers can check the split.
pub fn balanced_stage_of(i: usize, m: usize, n_stages: usize) -> usize {
    if m == 0 {
        return 0;
    }
    ((i * n_stages) / m).min(n_stages - 1)
}

/// Partition `graph` into `n_stages` pipeline stages. Panics if `n_stages == 0`.
/// With `n_stages == 1` the single stage is the whole graph (leaves materialized).
pub fn partition(graph: &Graph, n_stages: usize) -> Vec<Stage> {
    assert!(n_stages >= 1, "n_stages must be >= 1");
    partition_with(graph, n_stages, |i, m| balanced_stage_of(i, m, n_stages))
}

/// Partition with a custom compute-node→stage assignment (`(i, m) -> stage`,
/// where `i` is the node's index among the `m` compute nodes in topo order).
/// Lets callers cut on layer boundaries or balance by weight bytes rather than
/// node count. The assignment must be monotonic non-decreasing in `i` (stages
/// are contiguous in topo order) — this is asserted.
pub fn partition_with(
    graph: &Graph,
    n_stages: usize,
    assign: impl Fn(usize, usize) -> usize,
) -> Vec<Stage> {
    assert!(n_stages >= 1, "n_stages must be >= 1");
    let n = graph.len();
    let compute: Vec<NodeId> = (0..n as u32)
        .map(NodeId)
        .filter(|&id| !is_leaf(graph, id))
        .collect();
    let m = compute.len();

    // stage of each compute node (monotonic in topo index).
    let mut stage_of: HashMap<NodeId, usize> = HashMap::with_capacity(m);
    let mut last = 0usize;
    for (i, &id) in compute.iter().enumerate() {
        let s = assign(i, m).min(n_stages - 1);
        assert!(
            s >= last,
            "stage assignment must be non-decreasing in topo order"
        );
        last = s;
        stage_of.insert(id, s);
    }

    let graph_outputs: HashSet<NodeId> = graph.outputs.iter().copied().collect();
    let mut stages = Vec::with_capacity(n_stages);

    for s in 0..n_stages {
        let mut g = Graph::new(format!("{}__stage{s}", graph.name));
        let mut map: HashMap<NodeId, NodeId> = HashMap::new(); // orig compute → new
        let mut leaf_seen: HashMap<NodeId, NodeId> = HashMap::new();
        let mut boundary_seen: HashMap<NodeId, NodeId> = HashMap::new();
        let mut inputs: Vec<String> = Vec::new();
        let mut params: Vec<String> = Vec::new();

        for &cid in compute.iter().filter(|&&id| stage_of[&id] == s) {
            let node = graph.node(cid);
            let mut new_inputs = Vec::with_capacity(node.inputs.len());
            for &inp in &node.inputs {
                let new_in = if is_leaf(graph, inp) {
                    // Materialize the leaf into this stage (once). Params/Inputs
                    // are recorded; Constants carry their bytes in the Op.
                    *leaf_seen.entry(inp).or_insert_with(|| {
                        let ln = graph.node(inp);
                        match &ln.op {
                            Op::Input { name } => inputs.push(name.clone()),
                            Op::Param { name } => params.push(name.clone()),
                            _ => {}
                        }
                        g.append_node(ln.op.clone(), vec![], ln.shape.clone(), ln.name.clone())
                    })
                } else if stage_of[&inp] == s {
                    map[&inp]
                } else {
                    // Earlier stage → boundary input.
                    *boundary_seen.entry(inp).or_insert_with(|| {
                        let bn = boundary_name(inp);
                        inputs.push(bn.clone());
                        g.append_node(
                            Op::Input { name: bn.clone() },
                            vec![],
                            graph.shape(inp).clone(),
                            Some(bn),
                        )
                    })
                };
                new_inputs.push(new_in);
            }
            let nid = g.append_node(
                node.op.clone(),
                new_inputs,
                node.shape.clone(),
                node.name.clone(),
            );
            map.insert(cid, nid);
        }

        // Outputs: stage-s compute nodes consumed by a later stage, or model outputs.
        let mut out_ids = Vec::new();
        let mut out_names = Vec::new();
        let mut out_shapes = Vec::new();
        for &cid in compute.iter().filter(|&&id| stage_of[&id] == s) {
            let final_out = graph_outputs.contains(&cid);
            let used_later = graph
                .users(cid)
                .iter()
                .any(|u| stage_of.get(u).is_some_and(|&us| us > s));
            if final_out || used_later {
                out_ids.push(map[&cid]);
                out_names.push(boundary_name(cid));
                out_shapes.push(static_dims(graph, cid));
            }
        }
        g.set_outputs(out_ids);
        inputs.sort();
        inputs.dedup();
        params.sort();
        params.dedup();
        stages.push(Stage {
            index: s,
            graph: g,
            inputs,
            outputs: out_names,
            output_shapes: out_shapes,
            params,
        });
    }
    stages
}
