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

//! Load / save COMPAS-style networks from jax_fdm `data/json/*.json`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::mesh::edges_from_faces;
use crate::network::Network;

use super::mesh::MeshDocument;

#[derive(Deserialize, Serialize)]
struct CompasJson {
    #[serde(default)]
    attributes: Option<serde_json::Value>,
    node: Option<std::collections::HashMap<String, NodeAttrs>>,
    edge: Option<std::collections::HashMap<String, EdgeAdj>>,
    #[serde(default)]
    dna: Option<Dna>,
    #[serde(default)]
    dea: Option<serde_json::Value>,
    /// Mesh faces (RLX extension; also accepted on import).
    #[serde(default)]
    faces: Vec<Vec<usize>>,
    #[serde(default)]
    faces_load: Vec<[f64; 3]>,
    #[serde(default)]
    faces_load_local: bool,
    #[serde(default)]
    q: Option<Vec<f64>>,
}

#[derive(Deserialize, Serialize, Default)]
struct Dna {
    #[serde(default)]
    is_support: bool,
    #[serde(default)]
    px: f64,
    #[serde(default)]
    py: f64,
    #[serde(default)]
    pz: f64,
}

#[derive(Deserialize, Serialize, Default)]
struct NodeAttrs {
    #[serde(default)]
    x: f64,
    #[serde(default)]
    y: f64,
    #[serde(default)]
    z: f64,
    #[serde(default)]
    is_support: bool,
    #[serde(default)]
    px: f64,
    #[serde(default)]
    py: f64,
    #[serde(default)]
    pz: f64,
}

#[derive(Deserialize, Serialize)]
struct EdgeAdj {
    #[serde(flatten)]
    neighbors: std::collections::HashMap<String, serde_json::Value>,
}

/// Parse a jax_fdm / COMPAS JSON network (topology + optional mesh fields).
pub fn from_json_str(s: &str) -> Result<Network, String> {
    let doc: CompasJson = serde_json::from_str(s).map_err(|e| e.to_string())?;
    build_network(doc)
}

/// Read a JSON network from disk.
pub fn from_json_path(path: impl AsRef<Path>) -> Result<Network, String> {
    let s = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
    from_json_str(&s)
}

/// Parse mesh sidecar JSON (`MeshDocument`).
pub fn mesh_from_json_str(s: &str) -> Result<MeshDocument, String> {
    serde_json::from_str(s).map_err(|e| e.to_string())
}

/// Read mesh sidecar from disk.
pub fn mesh_from_json_path(path: impl AsRef<Path>) -> Result<MeshDocument, String> {
    let s = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
    mesh_from_json_str(&s)
}

/// Merge a topology network with a [`MeshDocument`].
pub fn merge_mesh(net: &mut Network, mesh: &MeshDocument) {
    mesh.apply_to_network(net);
}

/// Serialize a network to COMPAS-compatible JSON (includes mesh when set).
pub fn to_json_str(net: &Network) -> Result<String, String> {
    let doc = network_to_compas(net)?;
    serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())
}

/// Write network JSON to disk.
pub fn to_json_path(net: &Network, path: impl AsRef<Path>) -> Result<(), String> {
    let s = to_json_str(net)?;
    std::fs::write(path.as_ref(), s).map_err(|e| e.to_string())
}

fn build_network(doc: CompasJson) -> Result<Network, String> {
    let node_map = doc.node.unwrap_or_default();
    let edge_map = doc.edge.unwrap_or_default();

    let max_key = node_map
        .keys()
        .filter_map(|k| k.parse::<usize>().ok())
        .chain(edge_map.keys().filter_map(|k| k.parse::<usize>().ok()))
        .max()
        .unwrap_or(0);
    let n = max_key + 1;

    let dna = doc.dna.unwrap_or_default();
    let mut xyz = vec![0.0; n * 3];
    let mut is_support = vec![dna.is_support; n];
    let mut loads = vec![0.0; n * 3];
    for (key, attrs) in &node_map {
        let i: usize = key.parse().map_err(|_| format!("bad node key {key}"))?;
        if i >= n {
            continue;
        }
        xyz[3 * i] = attrs.x;
        xyz[3 * i + 1] = attrs.y;
        xyz[3 * i + 2] = attrs.z;
        if attrs.is_support {
            is_support[i] = true;
        }
        loads[3 * i] = attrs.px;
        loads[3 * i + 1] = attrs.py;
        loads[3 * i + 2] = attrs.pz;
    }

    let mut edges = Vec::new();
    for (ek, adj) in &edge_map {
        let u: usize = ek.parse().map_err(|_| format!("bad edge key {ek}"))?;
        for vk in adj.neighbors.keys() {
            let v: usize = vk.parse().map_err(|_| format!("bad edge neighbor {vk}"))?;
            if u < v {
                edges.push((u, v));
            }
        }
    }
    if edges.is_empty() && !doc.faces.is_empty() {
        edges = edges_from_faces(&doc.faces);
    }
    edges.sort_unstable();
    edges.dedup();

    let q = doc
        .q
        .filter(|q| q.len() == edges.len())
        .unwrap_or_else(|| vec![-1.0; edges.len()]);

    Ok(Network {
        xyz,
        is_support,
        loads,
        edges,
        q,
        edges_load: None,
        faces: if doc.faces.is_empty() {
            None
        } else {
            Some(doc.faces)
        },
        faces_load: if doc.faces_load.is_empty() {
            None
        } else {
            Some(doc.faces_load)
        },
        faces_load_local: doc.faces_load_local,
    })
}

fn network_to_compas(net: &Network) -> Result<CompasJson, String> {
    let n = net.num_nodes();
    let mut node = std::collections::HashMap::new();
    for i in 0..n {
        node.insert(
            i.to_string(),
            NodeAttrs {
                x: net.xyz[3 * i],
                y: net.xyz[3 * i + 1],
                z: net.xyz[3 * i + 2],
                is_support: net.is_support[i],
                px: net.loads[3 * i],
                py: net.loads[3 * i + 1],
                pz: net.loads[3 * i + 2],
            },
        );
    }
    let mut edge = std::collections::HashMap::new();
    for (ei, &(u, v)) in net.edges.iter().enumerate() {
        let mut neighbors = std::collections::HashMap::new();
        neighbors.insert(v.to_string(), serde_json::json!({}));
        edge.insert(ei.to_string(), EdgeAdj { neighbors });
        let _ = u;
    }
    Ok(CompasJson {
        attributes: Some(serde_json::json!({"name": "Network"})),
        node: Some(node),
        edge: Some(edge),
        dna: Some(Dna::default()),
        dea: Some(serde_json::json!({"q": 0.0})),
        faces: net.faces.clone().unwrap_or_default(),
        faces_load: net.faces_load.clone().unwrap_or_default(),
        faces_load_local: net.faces_load_local,
        q: Some(net.q.clone()),
    })
}
