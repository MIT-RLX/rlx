// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Loading / memory layout for baked artifacts.
//!
//! By default v2 files could store the same weight bytes twice (graph
//! [`Op::Constant`] + weight table). These modes and helpers cut disk and peak
//! RAM at load / compile time.

use crate::format::{RlxFile, RlxWeight};
use anyhow::{Result, bail};
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// Where weight **bytes** live in the artifact after bake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryMode {
    /// Bytes in both graph Constants and the weight table (legacy / easy inspect).
    Duplex,
    /// Bytes only in graph Constants; table keeps name/shape/encoding (empty data).
    /// Best when you compile `file.graph` directly and rarely inspect the table.
    Runtime,
    /// Bytes only in the weight table; graph Constants are emptied by name.
    /// Call [`RlxFile::materialize_weights`] (or [`RlxFile::into_runtime_graph`])
    /// before `Session::compile`. Smallest on-disk footprint.
    #[default]
    Compact,
}

impl MemoryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Duplex => "duplex",
            Self::Runtime => "runtime",
            Self::Compact => "compact",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Duplex => "weight bytes in graph and table (duplicate)",
            Self::Runtime => "weight bytes in graph only; table is metadata",
            Self::Compact => "weight bytes in table only; materialize before compile",
        }
    }
}

impl fmt::Display for MemoryMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "duplex" | "both" | "full" => Ok(Self::Duplex),
            "runtime" | "graph" => Ok(Self::Runtime),
            "compact" | "table" | "lean" => Ok(Self::Compact),
            other => Err(format!(
                "unknown memory mode {other:?}; expected duplex, runtime, or compact"
            )),
        }
    }
}

/// Counters for memory / layout passes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryStats {
    pub mode: String,
    /// Identical Constant nodes merged (CSE).
    pub constants_deduped: usize,
    /// Bytes cleared from graph Constants (`compact`).
    pub graph_bytes_stripped: usize,
    /// Bytes cleared from the weight table (`runtime`).
    pub table_bytes_stripped: usize,
    /// Source bindings omitted because they were folded away.
    pub folded_bindings_dropped: usize,
}

/// Share identical `Op::Constant` payloads: remaps consumers to the first copy.
pub fn dedupe_identical_constants(graph: &Graph) -> (Graph, usize) {
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
    // key = (dtype tag via shape debug, data bytes) — use raw data + shape dims + dtype
    let mut blob_to_id: HashMap<Vec<u8>, NodeId> = HashMap::new();
    let mut merged = 0usize;

    for node in graph.nodes() {
        let new_id = match &node.op {
            Op::Constant { data } => {
                let mut key = Vec::with_capacity(16 + data.len());
                key.extend_from_slice(format!("{:?}", node.shape).as_bytes());
                key.extend_from_slice(data);
                if let Some(&existing) = blob_to_id.get(&key) {
                    merged += 1;
                    existing
                } else {
                    let id = out.add_node(node.op.clone(), vec![], node.shape.clone());
                    if let Some(name) = &node.name {
                        out.node_mut(id).name = Some(name.clone());
                    }
                    blob_to_id.insert(key, id);
                    id
                }
            }
            _ => {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
                if let Some(name) = &node.name {
                    out.node_mut(id).name = Some(name.clone());
                }
                id
            }
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
    (out, merged)
}

/// Clear Constant payloads whose names appear in the weight table.
pub fn strip_graph_weight_bytes(graph: &mut Graph, weights: &[RlxWeight]) -> usize {
    let names: std::collections::HashSet<&str> = weights.iter().map(|w| w.name.as_str()).collect();
    let mut stripped = 0usize;
    for node in graph.nodes_mut() {
        let Some(name) = node.name.as_deref() else {
            continue;
        };
        if !names.contains(name) {
            continue;
        }
        if let Op::Constant { data } = &mut node.op {
            stripped += data.len();
            data.clear();
        }
    }
    stripped
}

/// Clear weight-table payloads; keep name / shape / encoding for inspection.
pub fn strip_table_weight_bytes(weights: &mut [RlxWeight]) -> usize {
    let mut stripped = 0usize;
    for w in weights.iter_mut() {
        stripped += w.data.len();
        w.data.clear();
        if !w.note.contains("metadata-only") {
            w.note = format!("{} [metadata-only]", w.note);
        }
    }
    stripped
}

impl RlxFile {
    /// True when any named weight Constant has empty data while the table has bytes.
    pub fn needs_materialize(&self) -> bool {
        for w in &self.weights {
            if w.data.is_empty() {
                continue;
            }
            for n in self.graph.nodes() {
                if n.name.as_deref() != Some(w.name.as_str()) {
                    continue;
                }
                if let Op::Constant { data } = &n.op {
                    if data.is_empty() {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Copy weight-table bytes into matching named `Op::Constant` nodes.
    ///
    /// Required after [`MemoryMode::Compact`] before `Session::compile`.
    pub fn materialize_weights(&mut self) -> Result<usize> {
        let mut filled = 0usize;
        for w in &self.weights {
            if w.data.is_empty() {
                continue;
            }
            let mut hit = false;
            for n in self.graph.nodes_mut() {
                if n.name.as_deref() != Some(w.name.as_str()) {
                    continue;
                }
                if let Op::Constant { data } = &mut n.op {
                    *data = w.data.clone();
                    filled += w.data.len();
                    hit = true;
                }
            }
            if !hit {
                // Bias / folded bindings may exist only in the table — ok.
                continue;
            }
        }
        Ok(filled)
    }

    /// Drop table payloads after materialize to free RAM (graph keeps the bytes).
    pub fn drop_weight_table_data(&mut self) -> usize {
        strip_table_weight_bytes(&mut self.weights)
    }

    /// Materialize compact weights and return the graph for `Session::compile`.
    pub fn into_runtime_graph(mut self) -> Result<Graph> {
        if self.needs_materialize() {
            self.materialize_weights()?;
        }
        Ok(self.graph)
    }

    /// Apply a memory layout to an already-built file (used at end of bake).
    pub fn apply_memory_mode(&mut self, mode: MemoryMode) -> MemoryStats {
        let mut stats = MemoryStats {
            mode: mode.as_str().to_string(),
            ..Default::default()
        };
        match mode {
            MemoryMode::Duplex => {}
            MemoryMode::Runtime => {
                stats.table_bytes_stripped = strip_table_weight_bytes(&mut self.weights);
                self.meta.weight_bytes = self.weights.iter().map(|w| w.data.len()).sum();
            }
            MemoryMode::Compact => {
                stats.graph_bytes_stripped =
                    strip_graph_weight_bytes(&mut self.graph, &self.weights);
                self.meta.constant_bytes = self
                    .graph
                    .nodes()
                    .iter()
                    .map(|n| match &n.op {
                        Op::Constant { data } => data.len(),
                        _ => 0,
                    })
                    .sum();
            }
        }
        stats
    }
}

/// Ensure a compact artifact is ready to compile; errors if table bytes missing.
pub fn ensure_runtime_ready(file: &mut RlxFile) -> Result<()> {
    if !file.needs_materialize() {
        return Ok(());
    }
    let n = file.materialize_weights()?;
    if file.needs_materialize() {
        bail!(
            "compact *.rlx still has empty Constants after materialize \
             (filled {n} bytes) — weight table may be incomplete"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BakeOptions, BakeProfile, bake};
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Shape};
    use std::collections::HashMap;

    fn tiny() -> (rlx_ir::Graph, HashMap<String, Vec<f32>>) {
        let s = Shape::new(&[4], DType::F32);
        let mut g = Graph::new("m");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let y = g.binary(BinaryOp::Mul, x, w, s);
        g.set_outputs(vec![y]);
        let mut b = HashMap::new();
        b.insert("w".into(), vec![1.0, 2.0, 3.0, 4.0]);
        (g, b)
    }

    #[test]
    fn compact_strips_and_materializes() {
        let (g, b) = tiny();
        let mut opts = BakeOptions::from_profile(BakeProfile::Exact);
        opts.memory = MemoryMode::Compact;
        let (mut file, _) = bake(&g, &b, &opts);
        assert!(file.needs_materialize());
        let table_bytes: usize = file.weights.iter().map(|w| w.data.len()).sum();
        assert!(table_bytes > 0);
        let const_bytes: usize = file
            .graph
            .nodes()
            .iter()
            .map(|n| match &n.op {
                Op::Constant { data } => data.len(),
                _ => 0,
            })
            .sum();
        assert_eq!(const_bytes, 0);
        file.materialize_weights().unwrap();
        assert!(!file.needs_materialize());
    }

    #[test]
    fn runtime_strips_table() {
        let (g, b) = tiny();
        let mut opts = BakeOptions::from_profile(BakeProfile::Exact);
        opts.memory = MemoryMode::Runtime;
        let (file, report) = bake(&g, &b, &opts);
        assert!(!file.needs_materialize());
        assert!(file.weights.iter().all(|w| w.data.is_empty()));
        assert!(report.memory.table_bytes_stripped > 0);
    }
}
