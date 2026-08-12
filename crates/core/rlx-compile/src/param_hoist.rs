// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Param-invariant subgraph hoisting.
//!
//! Splits a graph into a **prepare** graph (the "param-invariant closure" —
//! everything that depends only on `Op::Param` weights + `Op::Constant`, never on
//! an `Op::Input` activation) and a **main** graph (the rest). The runtime runs
//! `prepare` ONCE and feeds its outputs into `main` as persistent bound inputs,
//! so weight-derived tensors (transposed / dequantized / merged weights, RoPE
//! tables, …) are computed once instead of every forward.
//!
//! This is the runtime-side complement to compile-time `specialize_params` +
//! [`crate::ConstantFolding`]: that folds weight-derived compute away when the
//! weight VALUES are known at compile (`param_bindings`); this hoists it when
//! they are only known at run time. When `param_bindings` already folded the
//! invariant compute away, [`split_param_invariant`] finds no boundary and
//! returns `None`, so the two never conflict.

use crate::DeadCodeElimination;
use rlx_fusion::analysis::{Analysis, UseCounts};
use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};

/// Ops whose output is NOT a deterministic function of their inputs (RNG /
/// sampling). Treated as dynamic roots so their cone is never hoisted.
fn is_nondeterministic(op: &Op) -> bool {
    op.is_nondeterministic()
}

fn is_leaf(op: &Op) -> bool {
    op.is_leaf()
}

/// Result of a param-invariant split.
pub struct HoistSplit {
    /// Weight-only graph; its outputs are the boundary tensors. Runs once.
    pub prepare: Graph,
    /// The rest; each boundary tensor is a named `Op::Input` fed from `prepare`.
    pub main: Graph,
    /// Names of the boundary inputs, in `prepare`-output order.
    pub boundary: Vec<String>,
    /// `Op::Param` names present in `prepare` (for set_param routing).
    pub prepare_params: HashSet<String>,
    /// `Op::Param` names present in `main`.
    pub main_params: HashSet<String>,
}

/// Split `graph` so its param-invariant closure runs once. Returns `None` when
/// there is nothing worth hoisting (no COMPUTED weight-only tensor is consumed
/// by the dynamic part) — the common case for already-const-folded graphs.
pub fn split_param_invariant(graph: &Graph) -> Option<HoistSplit> {
    // 1. Mark dynamic nodes = reachable from an Input or a nondeterministic op.
    //    (graph.nodes() is topologically ordered, so a single forward pass.)
    let mut dynamic: HashSet<NodeId> = HashSet::new();
    for node in graph.nodes() {
        let dyn_here = matches!(node.op, Op::Input { .. })
            || is_nondeterministic(&node.op)
            || node.inputs.iter().any(|i| dynamic.contains(i));
        if dyn_here {
            dynamic.insert(node.id);
        }
    }

    // 2. Boundary = COMPUTED invariant node (not a leaf) consumed by a dynamic
    //    node, or that is itself a graph output.
    let out_set: HashSet<NodeId> = graph.outputs.iter().copied().collect();
    // `Graph::users` scans every node to answer for one node, and this filter
    // asks for *every* node — O(n²). Build the relation once instead.
    let uses = UseCounts::compute(graph);
    let boundary_ids: Vec<NodeId> = graph
        .nodes()
        .iter()
        .filter(|n| !dynamic.contains(&n.id) && !is_leaf(&n.op))
        .filter(|n| out_set.contains(&n.id) || uses.users(n.id).iter().any(|u| dynamic.contains(u)))
        .map(|n| n.id)
        .collect();
    if boundary_ids.is_empty() {
        return None;
    }
    let boundary: Vec<String> = (0..boundary_ids.len())
        .map(|i| format!("__rlx_prep_{i}"))
        .collect();

    // 3. prepare = graph with outputs replaced by the boundary; DCE drops the
    //    entire dynamic cone (and all Inputs), leaving a weight-only graph.
    let mut prepare = graph.clone();
    prepare.set_outputs(boundary_ids.clone());
    let prepare = DeadCodeElimination.run(prepare);

    // 4. main = graph with each boundary node replaced by a same-shape Input;
    //    DCE drops the now-dead invariant cone feeding those nodes.
    let boundary_name: HashMap<NodeId, &String> =
        boundary_ids.iter().copied().zip(boundary.iter()).collect();
    let mut main = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    for node in graph.nodes() {
        let new_id = if let Some(name) = boundary_name.get(&node.id) {
            main.input(name.as_str(), node.shape.clone())
        } else {
            let ins: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let nid = main.add_node(node.op.clone(), ins, node.shape.clone());
            let src = graph.node(node.id);
            let dst = main.node_mut(nid);
            dst.name = src.name.clone();
            dst.origin = src.origin.clone();
            nid
        };
        id_map.insert(node.id, new_id);
    }
    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|o| id_map[o]).collect();
    main.set_outputs(new_outputs);
    let main = DeadCodeElimination.run(main);

    let param_names = |g: &Graph| -> HashSet<String> {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Param { name } => Some(name.clone()),
                _ => None,
            })
            .collect()
    };
    let prepare_params = param_names(&prepare);
    let main_params = param_names(&main);

    Some(HoistSplit {
        prepare,
        main,
        boundary,
        prepare_params,
        main_params,
    })
}
