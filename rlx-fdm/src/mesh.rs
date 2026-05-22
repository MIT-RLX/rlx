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

//! Mesh structures for face tributary loads (jax_fdm `equilibrium.structures.meshes`).

use crate::network::{Network, Vec3};
use crate::structure::Structure;

/// Mesh topology layered on a pin-jointed [`Structure`].
#[derive(Clone, Debug)]
pub struct MeshStructure {
    pub base: Structure,
    /// Face vertex index lists (≥ 3 vertices each).
    pub faces: Vec<Vec<usize>>,
    /// Up to two incident face indices per edge (`None` = boundary).
    pub edges_faces: Vec<[Option<usize>; 2]>,
    /// Row-normalized face–vertex matrix flattened `[num_faces, num_nodes]`.
    pub face_vertex_weights: Vec<f64>,
}

impl MeshStructure {
    pub fn from_network_and_faces(network: &Network, faces: Vec<Vec<usize>>) -> Self {
        let base = Structure::from_network(network);
        let edges_faces = edges_faces_from_mesh(&faces, &network.edges);
        let face_vertex_weights =
            face_vertex_matrix(&faces, base.num_nodes);
        Self {
            base,
            faces,
            edges_faces,
            face_vertex_weights,
        }
    }

    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Face centroids `C_fv @ xyz` (jax_fdm `connectivity_faces_vertices @ xyz`).
    pub fn face_centroids(&self, xyz: &[f64]) -> Vec<Vec3> {
        let n = self.base.num_nodes;
        let nf = self.num_faces();
        let mut out = vec![0.0; nf * 3];
        for f in 0..nf {
            for v in 0..n {
                let w = self.face_vertex_weights[f * n + v];
                if w == 0.0 {
                    continue;
                }
                out[f * 3] += w * xyz[v * 3];
                out[f * 3 + 1] += w * xyz[v * 3 + 1];
                out[f * 3 + 2] += w * xyz[v * 3 + 2];
            }
        }
        (0..nf)
            .map(|f| [out[f * 3], out[f * 3 + 1], out[f * 3 + 2]])
            .collect()
    }
}

fn face_vertex_matrix(faces: &[Vec<usize>], num_nodes: usize) -> Vec<f64> {
    let nf = faces.len();
    let mut f = vec![0.0; nf * num_nodes];
    for (fi, face) in faces.iter().enumerate() {
        let valid: Vec<usize> = face.iter().copied().filter(|&v| v < num_nodes).collect();
        if valid.is_empty() {
            continue;
        }
        let inv = 1.0 / valid.len() as f64;
        for v in valid {
            f[fi * num_nodes + v] = inv;
        }
    }
    f
}

fn edges_faces_from_mesh(faces: &[Vec<usize>], edges: &[(usize, usize)]) -> Vec<[Option<usize>; 2]> {
    let mut out = vec![[None, None]; edges.len()];
    for (fi, face) in faces.iter().enumerate() {
        let loopv: Vec<usize> = face
            .iter()
            .copied()
            .filter(|&v| v < usize::MAX / 2)
            .collect();
        if loopv.len() < 3 {
            continue;
        }
        for i in 0..loopv.len() {
            let u = loopv[i];
            let v = loopv[(i + 1) % loopv.len()];
            for (ei, &(a, b)) in edges.iter().enumerate() {
                if (a == u && b == v) || (a == v && b == u) {
                    if out[ei][0].is_none() {
                        out[ei][0] = Some(fi);
                    } else if out[ei][1].is_none() && out[ei][0] != Some(fi) {
                        out[ei][1] = Some(fi);
                    }
                }
            }
        }
    }
    out
}

/// Unique edges from face loops (jax_fdm `Mesh._edges_from_faces`).
pub fn edges_from_faces(faces: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut half = Vec::new();
    for face in faces {
        if face.len() < 3 {
            continue;
        }
        for i in 0..face.len() {
            half.push((face[i], face[(i + 1) % face.len()]));
        }
    }
    let mut edges = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (u, v) in half {
        let key = if u < v { (u, v) } else { (v, u) };
        if seen.insert(key) {
            edges.push(key);
        }
    }
    edges
}
