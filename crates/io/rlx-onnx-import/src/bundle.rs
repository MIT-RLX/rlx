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

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorMeta {
    pub shape: Vec<serde_json::Value>,
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IoMeta {
    pub name: String,
    #[serde(flatten)]
    pub meta: TensorMeta,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BundleManifest {
    pub source_onnx: String,
    pub inputs: Vec<IoMeta>,
    pub outputs: Vec<IoMeta>,
    pub node_count: usize,
    pub initializer_count: usize,
    pub op_histogram: HashMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BundleNode {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, serde_json::Value>,
    pub output_meta: Vec<serde_json::Value>,
}

pub struct RlxBundle {
    pub dir: PathBuf,
    pub manifest: BundleManifest,
    pub nodes: Vec<BundleNode>,
    pub weight_bytes: Vec<u8>,
}

impl RlxBundle {
    pub fn weights(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.weight_bytes).context("parse safetensors")
    }
}

/// Order nodes so every tensor is produced before it is consumed.
pub fn topo_sort_nodes(nodes: Vec<BundleNode>) -> Vec<BundleNode> {
    let n = nodes.len();
    let mut producer: HashMap<&str, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for o in &node.outputs {
            if !o.is_empty() {
                producer.insert(o.as_str(), i);
            }
        }
    }
    let mut indegree = vec![0usize; n];
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate() {
        for inp in &node.inputs {
            if inp.is_empty() {
                continue;
            }
            if let Some(&p) = producer.get(inp.as_str()) {
                if p != i {
                    edges[p].push(i);
                    indegree[i] += 1;
                }
            }
        }
    }
    let mut queue: std::collections::VecDeque<usize> =
        (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &edges[u] {
            indegree[v] -= 1;
            if indegree[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    if order.len() != n {
        return nodes;
    }
    // Permute the owned nodes into topo order by MOVING each out of its slot
    // (`Option::take`) — a `nodes[i].clone()` here copied every BundleNode,
    // including its `HashMap<String, serde_json::Value>` attrs (KBs/node), on
    // every import. `order` is a permutation so each `take` fires exactly once.
    let mut slots: Vec<Option<BundleNode>> = nodes.into_iter().map(Some).collect();
    order
        .into_iter()
        .map(|i| slots[i].take().expect("topo order is a permutation"))
        .collect()
}

/// Optional ONNX RLX bundle directory for integration tests and tools.
///
/// Set `RLX_ONNX_BUNDLE` to point at an exported bundle directory.
pub fn onnx_bundle_dir() -> std::path::PathBuf {
    std::env::var("RLX_ONNX_BUNDLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp/rlx-onnx-bundle"))
}

pub fn load_bundle(dir: &Path) -> Result<RlxBundle> {
    let manifest: BundleManifest = serde_json::from_str(
        &std::fs::read_to_string(dir.join("manifest.json"))
            .with_context(|| format!("read {}", dir.join("manifest.json").display()))?,
    )?;
    let nodes: Vec<BundleNode> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("graph.json"))
            .with_context(|| format!("read {}", dir.join("graph.json").display()))?,
    )?;
    let weight_bytes = std::fs::read(dir.join("weights.safetensors"))
        .with_context(|| format!("read {}", dir.join("weights.safetensors").display()))?;
    Ok(RlxBundle {
        dir: dir.to_path_buf(),
        manifest,
        nodes,
        weight_bytes,
    })
}
