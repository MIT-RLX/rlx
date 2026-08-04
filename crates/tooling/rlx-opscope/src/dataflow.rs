// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dataflow repeated-pattern mining. Where `motifs` walks *linear* single-
//! consumer op chains, this canonicalizes each node's **input cone** (the
//! branching sub-DAG that produces it, up to a depth) into a structural
//! signature and counts recurrences. A sub-DAG that recurs across the graph is
//! a unit of repeated computation — a candidate to be **decomposed once and
//! shared**, or **fused** into a specialized op. This is the "track the data
//! flow that can be additionally decomposed, especially repeated patterns" seam.
//!
//! The signature is a merkle hash of the cone: `sig(n, d)` = the op name of `n`
//! joined with `sig` of each operand at depth `d-1`; at `d = 0` or a leaf the
//! cone is truncated to just the op name. Two nodes share a signature iff their
//! depth-`d` input cones are op-isomorphic. Cheap (memoized), exact for DAGs.
//!
//! The miner runs on a plain node list `(op_name, input_indices)` so it works on
//! both an in-memory [`Graph`] and a graph dumped from another workspace (see
//! [`repeated_flow_patterns_on`] and the `opscope-graph` bin).

use rlx_ir::{Graph, Op};
use std::collections::{HashMap, HashSet};

/// A recurring rooted dataflow pattern.
#[derive(Clone, Debug)]
pub struct FlowPattern {
    /// Cone depth this pattern was found at.
    pub depth: usize,
    /// Number of distinct roots whose depth-`depth` cone matches.
    pub count: usize,
    /// Human-readable cone, e.g. `Relu(Add(MatMul(·,w),w))`.
    pub tree: String,
    /// Root node indices (a few examples).
    pub sites: Vec<u32>,
}

impl FlowPattern {
    /// Decomposition payoff proxy: recurrence × cone size.
    pub fn score(&self) -> usize {
        self.count * (self.tree.matches('(').count() + 1)
    }
}

/// A readable op label that keeps the discriminating detail (activation kind,
/// binary op) instead of collapsing everything to `Activation`/`Binary`.
pub fn op_name(op: &Op) -> String {
    match op {
        Op::Activation(a) => format!("{a:?}"),
        Op::Binary(b) => format!("{b:?}"),
        Op::Compare(c) => format!("Cmp{c:?}"),
        Op::Input { .. } => "in".into(),
        Op::Param { .. } => "w".into(),
        Op::Constant { .. } => "k".into(),
        other => format!("{:?}", other.kind()),
    }
}

fn sig(
    ops: &[String],
    inputs: &[Vec<usize>],
    node: usize,
    depth: usize,
    memo: &mut HashMap<(usize, usize), String>,
) -> String {
    if let Some(s) = memo.get(&(node, depth)) {
        return s.clone();
    }
    let name = ops[node].clone();
    let s = if depth == 0 || inputs[node].is_empty() {
        name
    } else {
        let kids: Vec<String> = inputs[node]
            .iter()
            .map(|&i| sig(ops, inputs, i, depth - 1, memo))
            .collect();
        format!("{name}({})", kids.join(","))
    };
    memo.insert((node, depth), s.clone());
    s
}

/// Mine repeated input-cone patterns over a plain node list. `ops[i]` is node
/// `i`'s op label, `inputs[i]` its operand indices. Keeps cones recurring at
/// least `min_count` times, ranked by payoff; bare leaves are dropped.
pub fn repeated_flow_patterns_on(
    ops: &[String],
    inputs: &[Vec<usize>],
    min_depth: usize,
    max_depth: usize,
    min_count: usize,
) -> Vec<FlowPattern> {
    let mut memo: HashMap<(usize, usize), String> = HashMap::new();
    let mut out: Vec<FlowPattern> = Vec::new();
    let mut seen_tree: HashSet<String> = HashSet::new();

    for depth in min_depth..=max_depth {
        let mut groups: HashMap<String, Vec<u32>> = HashMap::new();
        for node in 0..ops.len() {
            if inputs[node].is_empty() {
                continue; // root cones only at compute nodes
            }
            let s = sig(ops, inputs, node, depth, &mut memo);
            if !s.contains('(') {
                continue; // must actually branch/recurse
            }
            groups.entry(s).or_default().push(node as u32);
        }
        for (tree, sites) in groups {
            if sites.len() >= min_count && seen_tree.insert(tree.clone()) {
                out.push(FlowPattern {
                    depth,
                    count: sites.len(),
                    sites: sites.into_iter().take(6).collect(),
                    tree,
                });
            }
        }
    }

    out.sort_by(|a, b| b.score().cmp(&a.score()).then(b.count.cmp(&a.count)));
    out
}

/// Convenience over an in-memory [`Graph`]. Node ids are dense insertion order,
/// so `NodeId.0` doubles as the list index.
pub fn repeated_flow_patterns(
    g: &Graph,
    min_depth: usize,
    max_depth: usize,
    min_count: usize,
) -> Vec<FlowPattern> {
    let ops: Vec<String> = g.nodes().iter().map(|n| op_name(&n.op)).collect();
    let inputs: Vec<Vec<usize>> = g
        .nodes()
        .iter()
        .map(|n| n.inputs.iter().map(|i| i.0 as usize).collect())
        .collect();
    repeated_flow_patterns_on(&ops, &inputs, min_depth, max_depth, min_count)
}

/// A short "how to exploit this repeat" hint for a mined cone.
pub fn decomposition_hint(p: &FlowPattern) -> &'static str {
    let t = &p.tree;
    let has_act = t.contains("Relu") || t.contains("Silu") || t.contains("Gelu");
    let has_attn = t.contains("Attention") || t.contains("Softmax") || t.contains("Rope");
    let has_moe = t.contains("GroupedMatMul") || t.contains("TopK");
    if has_moe {
        "→ MoE expert block: fuse grouped-matmul + gating"
    } else if has_attn {
        "→ attention block: fuse (FusedAttentionBlock)"
    } else if t.contains("MatMul(") && t.contains("Add(") && has_act {
        if t.starts_with("Add(") {
            "→ residual linear+bias+act: fuse (FusedMatMulBiasAct + residual)"
        } else {
            "→ linear+bias+act: fuse into one kernel (FusedMatMulBiasAct)"
        }
    } else if t.contains("MatMul(") {
        "→ recurring matmul cone: share/prepack"
    } else {
        "→ recurring sub-DAG: decompose once + reuse"
    }
}
