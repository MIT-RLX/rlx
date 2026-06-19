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

//! Shape-dependent loads for nonlinear FDM (jax_fdm `equilibrium.loads`).

use crate::equilibrium::EquilibriumModel;
use crate::geometry;
use crate::mesh::MeshStructure;
use crate::network::Vec3;
use crate::structure::Structure;

/// Nodal + optional edge / face follower loads (jax_fdm `EquilibriumParametersState`).
#[derive(Clone, Debug, Default)]
pub struct LoadState {
    /// Applied nodal loads `[n, 3]` (constant unless updated externally).
    pub nodes: Vec<f64>,
    /// Per-edge load intensity in global XYZ, scaled by current edge length each iteration.
    pub edges: Option<Vec<[f64; 3]>>,
    /// Per-face load in global XYZ (or local if `faces_load_local`), scaled by tributary areas.
    pub faces: Option<Vec<[f64; 3]>>,
    pub faces_load_local: bool,
}

impl LoadState {
    pub fn from_network(nodes_load: &[f64]) -> Self {
        Self {
            nodes: nodes_load.to_vec(),
            edges: None,
            faces: None,
            faces_load_local: false,
        }
    }

    pub fn with_edge_loads(mut self, edges: Vec<[f64; 3]>) -> Self {
        self.edges = Some(edges);
        self
    }

    pub fn with_face_loads(mut self, faces: Vec<[f64; 3]>, local: bool) -> Self {
        self.faces = Some(faces);
        self.faces_load_local = local;
        self
    }

    pub fn has_shape_dependent(&self) -> bool {
        self.edges.as_ref().is_some_and(|e| !e.is_empty())
            || self.faces.as_ref().is_some_and(|f| !f.is_empty())
    }
}

/// Total nodal loads including tributary edge / face loads at current geometry.
pub fn nodes_load_at(
    xyz: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
) -> Vec<f64> {
    nodes_load_at_mesh(xyz, load_state, structure, edges, None)
}

/// Like [`nodes_load_at`] with optional mesh for face tributaries.
pub fn nodes_load_at_mesh(
    xyz: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    mesh: Option<&MeshStructure>,
) -> Vec<f64> {
    let mut loads = load_state.nodes.clone();
    if let Some(edge_intensity) = load_state.edges.as_ref() {
        let edge_vecs = edge_load_vectors(xyz, edge_intensity, structure, edges);
        accumulate_edge_loads_to_nodes(&mut loads, &edge_vecs, structure);
    }
    if let (Some(mesh), Some(face_loads)) = (mesh, load_state.faces.as_ref()) {
        let face_global = face_loads_global(xyz, mesh, face_loads, load_state.faces_load_local);
        let edge_from_faces = edges_tributary_faces_load(xyz, &face_global, mesh, edges);
        accumulate_edge_loads_to_nodes(&mut loads, &edge_from_faces, structure);
    }
    loads
}

/// `edges_load * length` (jax_fdm `edges_tributary_edges_load`).
fn edge_load_vectors(
    xyz: &[f64],
    edge_intensity: &[[f64; 3]],
    structure: &Structure,
    edges: &[(usize, usize)],
) -> Vec<f64> {
    let vectors = EquilibriumModel::edges_vectors(xyz, structure, edges);
    let lengths = EquilibriumModel::edges_lengths(&vectors, structure.num_edges);
    let mut out = vec![0.0; structure.num_edges * 3];
    for e in 0..structure.num_edges {
        let scale = lengths[e];
        let i = edge_intensity.get(e).copied().unwrap_or([0.0; 3]);
        out[e * 3] = i[0] * scale;
        out[e * 3 + 1] = i[1] * scale;
        out[e * 3 + 2] = i[2] * scale;
    }
    out
}

/// Global face intensity per face (LCS rotation when `local`).
fn face_loads_global(
    xyz: &[f64],
    mesh: &MeshStructure,
    face_loads: &[[f64; 3]],
    local: bool,
) -> Vec<[f64; 3]> {
    let nf = mesh.num_faces();
    let mut out = vec![[0.0; 3]; nf];
    for f in 0..nf {
        let verts: Vec<Vec3> = mesh.faces[f]
            .iter()
            .map(|&vi| [xyz[vi * 3], xyz[vi * 3 + 1], xyz[vi * 3 + 2]])
            .collect();
        let load = face_loads.get(f).copied().unwrap_or([0.0; 3]);
        out[f] = if local {
            geometry::face_load_global(&verts, load)
        } else {
            load
        };
    }
    out
}

/// Face area load per edge (jax_fdm `edges_tributary_faces_load`).
fn edges_tributary_faces_load(
    xyz: &[f64],
    face_loads: &[[f64; 3]],
    mesh: &MeshStructure,
    edges: &[(usize, usize)],
) -> Vec<f64> {
    let centroids = mesh.face_centroids(xyz);
    let mut edge_loads = vec![0.0; edges.len() * 3];
    for (e, &(u, v)) in edges.iter().enumerate() {
        let line = [
            [xyz[u * 3], xyz[u * 3 + 1], xyz[u * 3 + 2]],
            [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]],
        ];
        let mut vec = [0.0; 3];
        for slot in &mesh.edges_faces[e] {
            let Some(fi) = *slot else { continue };
            let c = centroids[fi];
            let tri = [line[0], line[1], c];
            let area = geometry::area_triangle(tri);
            let fl = face_loads[fi];
            vec[0] += area * fl[0];
            vec[1] += area * fl[1];
            vec[2] += area * fl[2];
        }
        edge_loads[e * 3] = vec[0];
        edge_loads[e * 3 + 1] = vec[1];
        edge_loads[e * 3 + 2] = vec[2];
    }
    edge_loads
}

/// `(∂P/∂x_f)ᵀ λ` for edge follower loads (jax_fdm fixed-point adjoint load term).
///
/// `lambda` is the adjoint on packed free RHS `P` (`nf × 3`, same layout as equilibrium).
pub fn transpose_edge_loads_jacobian(
    xyz: &[f64],
    edge_intensity: &[[f64; 3]],
    structure: &Structure,
    edges: &[(usize, usize)],
    lambda: &[f64],
) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; nf * 3];
    for e in 0..edges.len().min(edge_intensity.len()) {
        let (u, v) = edges[e];
        let intensity = edge_intensity[e];
        let mut dx = [0.0; 3];
        for c in 0..3 {
            dx[c] = xyz[u * 3 + c] - xyz[v * 3 + c];
        }
        let l = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2])
            .sqrt()
            .max(1e-12);
        let cu = structure.c(e, u).abs();
        let cv = structure.c(e, v).abs();
        let pu = free_index(structure, u);
        let pv = free_index(structure, v);
        for c in 0..3 {
            for d in 0..3 {
                let dlen = dx[d] / l;
                let s = 0.5 * intensity[c] * dlen;
                if let Some(pu) = pu {
                    g[pu * 3 + d] += lambda[pu * 3 + c] * cu * s;
                }
                if let Some(pv) = pv {
                    g[pv * 3 + d] += lambda[pv * 3 + c] * cv * (-s);
                }
                if let (Some(pu), Some(pv)) = (pu, pv) {
                    g[pv * 3 + d] += lambda[pv * 3 + c] * cv * s;
                    g[pu * 3 + d] += lambda[pu * 3 + c] * cu * (-s);
                }
            }
        }
    }
    g
}

/// `(∂P/∂x_f)ᵀ λ` for face tributary loads (global or local LCS intensity).
pub fn transpose_face_loads_jacobian(
    xyz: &[f64],
    face_loads: &[[f64; 3]],
    mesh: &MeshStructure,
    structure: &Structure,
    edges: &[(usize, usize)],
    lambda: &[f64],
    faces_load_local: bool,
) -> Vec<f64> {
    let nf = structure.num_free();
    let n = structure.num_nodes;
    let mut g = vec![0.0; nf * 3];
    let face_global = face_loads_global(xyz, mesh, face_loads, faces_load_local);
    let centroids = mesh.face_centroids(xyz);

    let lcs_jac: Vec<Option<Vec<[[f64; 3]; 3]>>> = if faces_load_local {
        (0..mesh.num_faces())
            .map(|fi| {
                let verts: Vec<Vec3> = mesh.faces[fi]
                    .iter()
                    .map(|&vi| [xyz[vi * 3], xyz[vi * 3 + 1], xyz[vi * 3 + 2]])
                    .collect();
                if verts.len() < 3 {
                    return None;
                }
                let load = face_loads.get(fi).copied().unwrap_or([0.0; 3]);
                Some(geometry::jacobian_face_load_global_wrt_polygon(
                    &verts, load,
                ))
            })
            .collect()
    } else {
        vec![None; mesh.num_faces()]
    };

    for (e, &(u, v)) in edges.iter().enumerate() {
        let a = [xyz[u * 3], xyz[u * 3 + 1], xyz[u * 3 + 2]];
        let b = [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]];
        let cu = structure.c(e, u).abs();
        let cv = structure.c(e, v).abs();
        let pu = free_index(structure, u);
        let pv = free_index(structure, v);

        for slot in &mesh.edges_faces[e] {
            let Some(fi) = *slot else { continue };
            if fi >= face_global.len() {
                continue;
            }
            let c = centroids[fi];
            let area = geometry::area_triangle([a, b, c]);
            let (ga, gb, gc) = geometry::grad_area_triangle(a, b, c);
            let fl = face_global[fi];
            let face = &mesh.faces[fi];

            for comp in 0..3 {
                let flc = fl[comp];

                if let Some(pu) = pu {
                    for d in 0..3 {
                        g[pu * 3 + d] += lambda[pu * 3 + comp] * 0.5 * cu * flc * ga[d];
                    }
                }
                if let Some(pv) = pv {
                    for d in 0..3 {
                        g[pv * 3 + d] += lambda[pv * 3 + comp] * 0.5 * cv * flc * gb[d];
                    }
                }
                for &vk in face {
                    if vk >= n {
                        continue;
                    }
                    let w = mesh.face_vertex_weights[fi * n + vk];
                    if w == 0.0 {
                        continue;
                    }
                    let Some(pk) = free_index(structure, vk) else {
                        continue;
                    };
                    for d in 0..3 {
                        let gcw = flc * w * gc[d];
                        if let Some(pu) = pu {
                            g[pk * 3 + d] += lambda[pu * 3 + comp] * 0.5 * cu * gcw;
                        }
                        if let Some(pv) = pv {
                            g[pk * 3 + d] += lambda[pv * 3 + comp] * 0.5 * cv * gcw;
                        }
                    }
                }

                if let Some(ref jac) = lcs_jac[fi] {
                    for (vi, &vk) in face.iter().enumerate() {
                        if vi >= jac.len() {
                            break;
                        }
                        let Some(pk) = free_index(structure, vk) else {
                            continue;
                        };
                        for d in 0..3 {
                            let dfl = jac[vi][comp][d];
                            let mut acc = 0.0;
                            if let Some(pu) = pu {
                                acc += lambda[pu * 3 + comp] * 0.5 * cu;
                            }
                            if let Some(pv) = pv {
                                acc += lambda[pv * 3 + comp] * 0.5 * cv;
                            }
                            g[pk * 3 + d] += acc * area * dfl;
                        }
                    }
                }
            }
        }
    }
    g
}

/// FD reference for [`transpose_face_loads_jacobian`] (tests / local fallback).
pub fn transpose_face_loads_jacobian_fd(
    xyz: &[f64],
    face_loads: &[[f64; 3]],
    mesh: &MeshStructure,
    structure: &Structure,
    edges: &[(usize, usize)],
    lambda: &[f64],
    local: bool,
) -> Vec<f64> {
    let nf = structure.num_free();
    let dim = nf * 3;
    let mut g = vec![0.0; dim];
    let eps = 1e-7;
    let load_state = LoadState {
        nodes: vec![0.0; structure.num_nodes * 3],
        edges: None,
        faces: Some(face_loads.to_vec()),
        faces_load_local: local,
    };

    for a in 0..nf {
        for d in 0..3 {
            let node = structure.indices_free[a];
            let mut xp = xyz.to_vec();
            let mut xm = xyz.to_vec();
            xp[node * 3 + d] += eps;
            xm[node * 3 + d] -= eps;
            let pp = load_matrix_free(&xp, &load_state, structure, edges, mesh);
            let pm = load_matrix_free(&xm, &load_state, structure, edges, mesh);
            for c in 0..3 {
                g[a * 3 + d] += lambda[a * 3 + c] * (pp[a * 3 + c] - pm[a * 3 + c]) / (2.0 * eps);
            }
        }
    }
    g
}

fn load_matrix_free(
    xyz: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    mesh: &MeshStructure,
) -> Vec<f64> {
    let loads = nodes_load_at_mesh(xyz, load_state, structure, edges, Some(mesh));
    let nf = structure.num_free();
    let mut p = vec![0.0; nf * 3];
    for a in 0..nf {
        let node = structure.indices_free[a];
        for c in 0..3 {
            p[a * 3 + c] = loads[node * 3 + c];
        }
    }
    p
}

fn free_index(structure: &Structure, node: usize) -> Option<usize> {
    structure.indices_free.iter().position(|&n| n == node)
}

fn accumulate_edge_loads_to_nodes(
    nodes_load: &mut [f64],
    edge_loads: &[f64],
    structure: &Structure,
) {
    let n = structure.num_nodes;
    let ne = structure.num_edges;
    for e in 0..ne {
        for node in 0..n {
            let c = structure.c(e, node).abs();
            if c == 0.0 {
                continue;
            }
            for comp in 0..3 {
                nodes_load[node * 3 + comp] += 0.5 * c * edge_loads[e * 3 + comp];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::edges_from_faces;
    use crate::network::Network;

    #[test]
    fn local_and_global_face_loads_same_on_horizontal_quad() {
        let faces = vec![vec![0, 1, 2], vec![0, 2, 3]];
        let edges = edges_from_faces(&faces);
        let xyz = vec![
            0.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            1.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
        ];
        for local in [false, true] {
            let net = Network {
                xyz: xyz.clone(),
                is_support: vec![true, false, true, false],
                loads: vec![0.0; 12],
                edges: edges.clone(),
                q: vec![-1.0; edges.len()],
                edges_load: None,
                faces: Some(faces.clone()),
                faces_load: Some(vec![[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]),
                faces_load_local: local,
            };
            let s = Structure::from_network(&net);
            let mesh = net.mesh_structure().expect("mesh");
            let ls = net.load_state();
            let loads = nodes_load_at_mesh(&xyz, &ls, &s, &edges, Some(&mesh));
            assert!(
                loads[3 + 2] < -0.1,
                "local={local} nodal z load {}",
                loads[3 + 2]
            );
        }
    }
}
