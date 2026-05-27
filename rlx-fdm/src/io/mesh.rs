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

//! Mesh-augmented network JSON (faces + tributary loads).

use serde::{Deserialize, Serialize};

use crate::mesh::edges_from_faces;
use crate::network::Network;

/// RLX / jax_fdm mesh sidecar (faces + per-face loads).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshDocument {
    /// Face vertex loops (≥ 3 indices per face).
    pub faces: Vec<Vec<usize>>,
    /// Per-face load vectors in global or local axes.
    #[serde(default)]
    pub faces_load: Vec<[f64; 3]>,
    /// When true, [`Self::faces_load`] are transformed with face LCS.
    #[serde(default)]
    pub faces_load_local: bool,
    /// Optional per-edge force densities (overrides default `-1` on import).
    #[serde(default)]
    pub q: Option<Vec<f64>>,
}

impl MeshDocument {
    pub fn apply_to_network(&self, net: &mut Network) {
        net.faces = Some(self.faces.clone());
        if !self.faces_load.is_empty() {
            net.faces_load = Some(self.faces_load.clone());
        }
        net.faces_load_local = self.faces_load_local;
        if let Some(ref q) = self.q {
            if q.len() == net.num_edges() {
                net.q.clone_from(q);
            }
        }
    }

    /// Build a pin-jointed network from faces (edges deduced from face loops).
    pub fn to_network(&self, xyz: Vec<f64>, is_support: Vec<bool>, loads: Vec<f64>) -> Network {
        let edges = edges_from_faces(&self.faces);
        let q = self.q.clone().unwrap_or_else(|| vec![-1.0; edges.len()]);
        let net = Network {
            xyz,
            is_support,
            loads,
            edges,
            q,
            edges_load: None,
            faces: Some(self.faces.clone()),
            faces_load: if self.faces_load.is_empty() {
                None
            } else {
                Some(self.faces_load.clone())
            },
            faces_load_local: self.faces_load_local,
        };
        net
    }
}

/// Combined topology + mesh document for file interchange.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkMeshJson {
    #[serde(flatten)]
    pub topology: serde_json::Value,
    #[serde(default)]
    pub faces: Vec<Vec<usize>>,
    #[serde(default)]
    pub faces_load: Vec<[f64; 3]>,
    #[serde(default)]
    pub faces_load_local: bool,
}
