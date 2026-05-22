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

//! Pin-jointed network topology and attributes (jax_fdm `FDNetwork`).

/// 3D vector type used throughout FDM.
pub type Vec3 = [f64; 3];

/// A pin-jointed bar network in force-density form.
///
/// Nodes carry coordinates, support flags, and applied loads `(px, py, pz)`.
/// Edges carry force densities `q` (signed; compression vs tension convention
/// matches jax_fdm: negative `q` is typical for arches in compression).
#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    /// Node positions and attributes, row-major `[n, 3]` flattened as `xyz[i*3+c]`.
    pub xyz: Vec<f64>,
    /// `true` if the node is fixed (support / anchor).
    pub is_support: Vec<bool>,
    /// Nodal loads `[n, 3]` flattened like `xyz`.
    pub loads: Vec<f64>,
    /// Edge list as `(u, v)` node indices (0-based, contiguous nodes).
    pub edges: Vec<(usize, usize)>,
    /// Force density per edge.
    pub q: Vec<f64>,
    /// Optional per-edge load intensity (global XYZ), scaled by edge length when iterating.
    pub edges_load: Option<Vec<[f64; 3]>>,
    /// Triangle / polygon faces for tributary area loads (mesh form-finding).
    pub faces: Option<Vec<Vec<usize>>>,
    /// Per-face load vectors (paired with [`Self::faces`]).
    pub faces_load: Option<Vec<[f64; 3]>>,
    pub faces_load_local: bool,
}

impl Network {
    /// Build [`LoadState`](crate::loads::LoadState) for iterative solves.
    pub fn load_state(&self) -> crate::loads::LoadState {
        let mut ls = crate::loads::LoadState::from_network(&self.loads);
        if let Some(ref e) = self.edges_load {
            ls = ls.with_edge_loads(e.clone());
        }
        if let Some(ref f) = self.faces_load {
            ls = ls.with_face_loads(f.clone(), self.faces_load_local);
        }
        ls
    }

    pub fn mesh_structure(&self) -> Option<crate::mesh::MeshStructure> {
        self.faces
            .as_ref()
            .map(|faces| crate::mesh::MeshStructure::from_network_and_faces(self, faces.clone()))
    }

    /// Uniform edge follower load (e.g. self-weight direction).
    pub fn edges_load_uniform(&mut self, load: [f64; 3]) {
        self.edges_load = Some(vec![load; self.num_edges()]);
    }

    pub fn num_nodes(&self) -> usize {
        self.is_support.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Node coordinate slice.
    pub fn node_xyz(&self, i: usize) -> Vec3 {
        [
            self.xyz[3 * i],
            self.xyz[3 * i + 1],
            self.xyz[3 * i + 2],
        ]
    }

    pub fn set_node_xyz(&mut self, i: usize, p: Vec3) {
        self.xyz[3 * i] = p[0];
        self.xyz[3 * i + 1] = p[1];
        self.xyz[3 * i + 2] = p[2];
    }

    pub fn node_load(&self, i: usize) -> Vec3 {
        [
            self.loads[3 * i],
            self.loads[3 * i + 1],
            self.loads[3 * i + 2],
        ]
    }

    pub fn set_node_load(&mut self, i: usize, p: Vec3) {
        self.loads[3 * i] = p[0];
        self.loads[3 * i + 1] = p[1];
        self.loads[3 * i + 2] = p[2];
    }

    /// Chain polyline arch like jax_fdm `examples/arch/arch.py`.
    pub fn arch_chain(arch_length: f64, num_segments: usize, q_init: f64, pz: f64) -> Self {
        assert!(num_segments >= 1);
        let n = num_segments + 1;
        let mut xyz = vec![0.0; n * 3];
        for i in 0..n {
            let t = i as f64 / num_segments as f64;
            xyz[3 * i] = -arch_length / 2.0 + t * arch_length;
        }
        let mut is_support = vec![false; n];
        is_support[0] = true;
        is_support[n - 1] = true;
        let mut loads = vec![0.0; n * 3];
        for i in 0..n {
            if !is_support[i] {
                loads[3 * i + 2] = pz;
            }
        }
        let edges: Vec<_> = (0..num_segments).map(|i| (i, i + 1)).collect();
        let q = vec![q_init; num_segments];
        Self {
            xyz,
            is_support,
            loads,
            edges,
            q,
            edges_load: None,
            faces: None,
            faces_load: None,
            faces_load_local: false,
        }
    }

    /// Open polyline from a list of 3D points (jax_fdm `FDNetwork::from_lines`).
    pub fn from_polyline(points: &[Vec3], q_init: f64) -> Self {
        assert!(points.len() >= 2);
        let n = points.len();
        let mut xyz = Vec::with_capacity(n * 3);
        for p in points {
            xyz.extend_from_slice(p);
        }
        let is_support = vec![false; n];
        let loads = vec![0.0; n * 3];
        let edges: Vec<_> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let q = vec![q_init; n - 1];
        Self {
            xyz,
            is_support,
            loads,
            edges,
            q,
            edges_load: None,
            faces: None,
            faces_load: None,
            faces_load_local: false,
        }
    }

    /// Pin supports at the given node indices.
    pub fn anchor_nodes(&mut self, keys: &[usize]) {
        for &k in keys {
            assert!(k < self.num_nodes());
            self.is_support[k] = true;
        }
    }

    /// Apply the same load vector to all free nodes.
    pub fn loads_on_free(&mut self, load: Vec3) {
        for i in 0..self.num_nodes() {
            if !self.is_support[i] {
                self.set_node_load(i, load);
            }
        }
    }

    /// jax_fdm `datastructure_validate` (subset).
    pub fn validate(&self) -> Result<(), String> {
        let n = self.num_nodes();
        if n == 0 {
            return Err("network has no nodes".into());
        }
        if self.num_edges() == 0 {
            return Err("network has no edges".into());
        }
        if !self.is_support.iter().any(|&s| s) {
            return Err("network has no supports".into());
        }
        if self.q.iter().any(|&qi| qi == 0.0) {
            return Err("network has edges with zero force density".into());
        }
        if self.xyz.len() != n * 3 || self.loads.len() != n * 3 {
            return Err("xyz/loads length mismatch".into());
        }
        for &(u, v) in &self.edges {
            if u >= n || v >= n {
                return Err(format!("edge ({u}, {v}) out of range for {n} nodes"));
            }
        }
        Ok(())
    }
}
