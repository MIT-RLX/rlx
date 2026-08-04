// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural op-subsequence mining — the "patterns in the *sequences of ops*"
//! reading. Pure graph analysis, no runtime data: walk single-consumer chains
//! (the ones a fusion pass could collapse) and count recurring op-kind n-grams.
//! A motif that recurs across many layers, weighted by its length, is a
//! fusion-kernel candidate.

use rlx_ir::{Graph, NodeId};
use std::collections::HashMap;

/// A recurring linear op-subsequence.
#[derive(Clone, Debug)]
pub struct Motif {
    /// Op-kind names in order, e.g. `["MatMul", "Binary", "Activation"]`.
    pub seq: Vec<String>,
    /// How many times this exact chain appears.
    pub count: usize,
    /// A few node ids where the chain starts (for inspection).
    pub examples: Vec<u32>,
}

impl Motif {
    /// Fusion payoff proxy: recurrence × chain length.
    pub fn score(&self) -> usize {
        self.count * self.seq.len()
    }
}

fn kind_name(g: &Graph, id: NodeId) -> String {
    format!("{:?}", g.node(id).op.kind())
}

/// Mine linear (single-consumer) op-kind n-grams of length `min_len..=max_len`.
/// Returns motifs recurring at least `min_count` times, ranked by payoff.
pub fn linear_op_motifs(
    graph: &Graph,
    min_len: usize,
    max_len: usize,
    min_count: usize,
) -> Vec<Motif> {
    // consumers[u] = nodes that read u.
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for node in graph.nodes() {
        for &inp in &node.inputs {
            consumers.entry(inp).or_default().push(node.id);
        }
    }

    // For each node, walk forward along single-consumer edges, emitting every
    // window that starts here. `key` is the joined kind sequence.
    let mut table: HashMap<String, Motif> = HashMap::new();
    for node in graph.nodes() {
        // Start chains only at compute nodes (skip Input/Param/Constant leaves).
        if node.inputs.is_empty() {
            continue;
        }
        let mut chain: Vec<NodeId> = vec![node.id];
        let mut cur = node.id;
        while chain.len() < max_len {
            match consumers.get(&cur) {
                Some(cs) if cs.len() == 1 => {
                    cur = cs[0];
                    chain.push(cur);
                }
                _ => break, // fan-out or terminal: chain ends (fusion boundary)
            }
        }
        for len in min_len..=chain.len() {
            let seq: Vec<String> = chain[..len]
                .iter()
                .map(|&id| kind_name(graph, id))
                .collect();
            let key = seq.join("→");
            let entry = table.entry(key).or_insert_with(|| Motif {
                seq,
                count: 0,
                examples: Vec::new(),
            });
            entry.count += 1;
            if entry.examples.len() < 4 {
                entry.examples.push(node.id.0);
            }
        }
    }

    let mut motifs: Vec<Motif> = table
        .into_values()
        .filter(|m| m.count >= min_count)
        .collect();
    motifs.sort_by(|a, b| b.score().cmp(&a.score()).then(b.count.cmp(&a.count)));
    motifs
}
