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

//! Scalar objectives for inverse form-finding (jax_fdm `goals` / `losses` subset).

use crate::geometry::{
    area_triangle, grad_area_triangle, grad_polygon_planarity_wrt_nodes,
    grad_polygon_rectangular_wrt_nodes, polygon_planarity, polygon_rectangular_deviation,
};
use crate::mesh::MeshStructure;
use crate::network::Vec3;
use crate::state::EquilibriumState;
use crate::structure::Structure;

/// Edge index in the network edge list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeIndex(pub usize);

/// Goal index in an objective list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GoalIndex(pub usize);

/// Packed free-coordinate dimension `3 × num_free`.
pub fn packed_free_dim(num_free: usize) -> usize {
    num_free * 3
}

/// Total structural load path: `Σ |force| · length` (jax_fdm `NetworkLoadPathGoal` prediction).
pub fn network_loadpath(state: &EquilibriumState) -> f64 {
    state
        .forces
        .iter()
        .zip(state.lengths.iter())
        .map(|(&f, &l)| f.abs() * l)
        .sum()
}

/// Mean edge length (jax_fdm length helper).
pub fn mean_edge_length(state: &EquilibriumState) -> f64 {
    if state.lengths.is_empty() {
        return 0.0;
    }
    state.lengths.iter().sum::<f64>() / state.lengths.len() as f64
}

/// Alias kept for older call sites.
pub fn total_loadpath_proxy(state: &EquilibriumState) -> f64 {
    network_loadpath(state)
}

/// Squared error of edge `e` length vs target (jax_fdm `EdgeLengthGoal` flavor).
pub fn edge_length_error(state: &EquilibriumState, edge: usize, target: f64) -> f64 {
    let d = state.lengths.get(edge).copied().unwrap_or(0.0) - target;
    d * d
}

/// `∂L/∂x_f` for [`mean_edge_length`](Self::mean_edge_length), packed free layout.
pub fn grad_mean_edge_length_wrt_xyz_free(
    state: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
) -> Vec<f64> {
    grad_edge_lengths_wrt_xyz_free(state, structure, edges, 1.0 / edges.len() as f64)
}

/// `∂L/∂x_f` for [`edge_length_error`](Self::edge_length_error) w.r.t. one edge.
pub fn grad_edge_length_error_wrt_xyz_free(
    state: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
    edge: usize,
    target: f64,
) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; packed_free_dim(nf)];
    if edge < edges.len() {
        let scale = 2.0 * (state.lengths[edge] - target);
        accumulate_edge_length_grad(&mut g, state, structure, edges, edge, scale);
    }
    g
}

pub(crate) fn grad_edge_lengths_wrt_xyz_free(
    state: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
    scale_all: f64,
) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; packed_free_dim(nf)];
    if scale_all == 0.0 {
        return g;
    }
    for e in 0..edges.len() {
        accumulate_edge_length_grad(&mut g, state, structure, edges, e, scale_all);
    }
    g
}

pub(crate) fn accumulate_edge_length_grad(
    g: &mut [f64],
    state: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
    e: usize,
    scale: f64,
) {
    let (u, v) = edges[e];
    let l = state.lengths[e].max(1e-12);
    let inv_l = scale / l;
    for c in 0..3 {
        let dc = state.xyz[u * 3 + c] - state.xyz[v * 3 + c];
        if let Some(pu) = packed_free_index(structure, u) {
            g[pu * 3 + c] += dc * inv_l;
        }
        if let Some(pv) = packed_free_index(structure, v) {
            g[pv * 3 + c] -= dc * inv_l;
        }
    }
}

pub(crate) fn packed_free_index(structure: &Structure, node: usize) -> Option<usize> {
    structure.indices_free.iter().position(|&n| n == node)
}

/// Mean axial force magnitude.
pub fn mean_edge_force(state: &EquilibriumState) -> f64 {
    if state.forces.is_empty() {
        return 0.0;
    }
    state.forces.iter().map(|f| f.abs()).sum::<f64>() / state.forces.len() as f64
}

/// Force on edge `e` (signed axial force `q · l`).
pub fn edge_force(state: &EquilibriumState, edge: usize) -> f64 {
    state.forces.get(edge).copied().unwrap_or(0.0)
}

/// Coordinate of node `n` along axis `c` (0=x, 1=y, 2=z).
pub fn node_coord(state: &EquilibriumState, node: usize, component: usize) -> f64 {
    state.xyz.get(node * 3 + component).copied().unwrap_or(0.0)
}

/// Minimum `z` among free nodes.
pub fn min_free_z(state: &EquilibriumState, structure: &Structure) -> f64 {
    structure
        .indices_free
        .iter()
        .map(|&n| state.xyz[n * 3 + 2])
        .fold(f64::INFINITY, f64::min)
}

/// `∂L/∂x_f` for a linear functional on one free-node coordinate.
pub fn grad_node_coord_wrt_xyz_free(
    structure: &Structure,
    node: usize,
    component: usize,
    scale: f64,
) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; packed_free_dim(nf)];
    if let Some(p) = packed_free_index(structure, node) {
        if component < 3 {
            g[p * 3 + component] = scale;
        }
    }
    g
}

/// `∂L/∂x_f` for [`min_free_z`](Self::min_free_z) (subgradient at minimizer).
pub fn grad_min_free_z_wrt_xyz_free(state: &EquilibriumState, structure: &Structure) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; packed_free_dim(nf)];
    let mut min_z = f64::INFINITY;
    let mut argmin = Vec::new();
    for &n in &structure.indices_free {
        let z = state.xyz[n * 3 + 2];
        if z < min_z - 1e-12 {
            min_z = z;
            argmin.clear();
            argmin.push(n);
        } else if (z - min_z).abs() <= 1e-12 {
            argmin.push(n);
        }
    }
    if argmin.is_empty() {
        return g;
    }
    let scale = 1.0 / argmin.len() as f64;
    for n in argmin {
        if let Some(p) = packed_free_index(structure, n) {
            g[p * 3 + 2] = scale;
        }
    }
    g
}

/// `∂L/∂x_f` for [`edge_force`](Self::edge_force) w.r.t. positions (via `q·l`).
pub fn grad_edge_force_wrt_xyz_free(
    state: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
    edge: usize,
    scale: f64,
) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; packed_free_dim(nf)];
    if edge >= state.forces.len() {
        return g;
    }
    let q_sign = state.forces[edge].signum();
    accumulate_edge_length_grad(&mut g, state, structure, edges, edge, q_sign * scale);
    g
}

/// Total mesh face area (sum of triangle fan areas per face).
pub fn mesh_total_area(mesh: &MeshStructure, xyz: &[f64]) -> f64 {
    let mut total = 0.0;
    for face in &mesh.faces {
        if face.len() < 3 {
            continue;
        }
        let pts: Vec<Vec3> = face
            .iter()
            .map(|&v| [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]])
            .collect();
        for i in 1..pts.len().saturating_sub(1) {
            total += area_triangle([pts[0], pts[i], pts[i + 1]]);
        }
    }
    total
}

/// `∂(mesh area)/∂x_f` for [`mesh_total_area`](Self::mesh_total_area).
pub fn grad_mesh_total_area_wrt_xyz_free(
    mesh: &MeshStructure,
    xyz: &[f64],
    structure: &Structure,
) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; packed_free_dim(nf)];
    for face in &mesh.faces {
        if face.len() < 3 {
            continue;
        }
        let pts: Vec<Vec3> = face
            .iter()
            .map(|&v| [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]])
            .collect();
        for i in 1..pts.len().saturating_sub(1) {
            let (ga, gb, gc) = grad_area_triangle(pts[0], pts[i], pts[i + 1]);
            for (local, grad) in [(0, ga), (i, gb), (i + 1, gc)] {
                let node = face[local];
                if let Some(p) = packed_free_index(structure, node) {
                    for c in 0..3 {
                        g[p * 3 + c] += grad[c];
                    }
                }
            }
        }
    }
    g
}

/// Mean face planarity (jax_fdm `FacePlanarityGoal` proxy).
pub fn mesh_mean_planarity(mesh: &MeshStructure, xyz: &[f64]) -> f64 {
    let nf = mesh.num_faces();
    if nf == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    for face in &mesh.faces {
        let pts: Vec<Vec3> = face
            .iter()
            .map(|&v| [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]])
            .collect();
        s += polygon_planarity(&pts);
    }
    s / nf as f64
}

/// `∂(mean planarity)/∂x_f`.
pub fn grad_mesh_mean_planarity_wrt_xyz_free(
    mesh: &MeshStructure,
    xyz: &[f64],
    structure: &Structure,
) -> Vec<f64> {
    let nf = mesh.num_faces();
    if nf == 0 {
        return vec![0.0; packed_free_dim(structure.num_free())];
    }
    let scale = 1.0 / nf as f64;
    let mut g = vec![0.0; packed_free_dim(structure.num_free())];
    for face in &mesh.faces {
        let pts: Vec<Vec3> = face
            .iter()
            .map(|&v| [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]])
            .collect();
        let node_grads = grad_polygon_planarity_wrt_nodes(&pts, face);
        for (node, grad) in node_grads {
            if let Some(p) = packed_free_index(structure, node) {
                for c in 0..3 {
                    g[p * 3 + c] += scale * grad[c];
                }
            }
        }
    }
    g
}

/// Mean face rectangularity deviation (jax_fdm `FaceRectangularGoal` proxy).
pub fn mesh_mean_face_rectangular(mesh: &MeshStructure, xyz: &[f64]) -> f64 {
    let nf = mesh.num_faces();
    if nf == 0 {
        return 0.0;
    }
    let mut s = 0.0;
    for face in &mesh.faces {
        let pts: Vec<Vec3> = face
            .iter()
            .map(|&v| [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]])
            .collect();
        s += polygon_rectangular_deviation(&pts);
    }
    s / nf as f64
}

/// `∂(mean rectangular deviation)/∂x_f`.
pub fn grad_mesh_mean_face_rectangular_wrt_xyz_free(
    mesh: &MeshStructure,
    xyz: &[f64],
    structure: &Structure,
) -> Vec<f64> {
    let nf = mesh.num_faces();
    if nf == 0 {
        return vec![0.0; packed_free_dim(structure.num_free())];
    }
    let scale = 1.0 / nf as f64;
    let mut g = vec![0.0; packed_free_dim(structure.num_free())];
    for face in &mesh.faces {
        let pts: Vec<Vec3> = face
            .iter()
            .map(|&v| [xyz[v * 3], xyz[v * 3 + 1], xyz[v * 3 + 2]])
            .collect();
        let node_grads = grad_polygon_rectangular_wrt_nodes(&pts, face);
        for (node, grad) in node_grads {
            if let Some(p) = packed_free_index(structure, node) {
                for c in 0..3 {
                    g[p * 3 + c] += scale * grad[c];
                }
            }
        }
    }
    g
}

/// Mesh-edge spring energy `Σ ‖x_u − x_v‖²` over face boundary edges (smoothing).
pub fn mesh_laplacian_energy(mesh: &MeshStructure, xyz: &[f64]) -> f64 {
    let mut seen = std::collections::HashSet::new();
    let mut s = 0.0;
    for face in &mesh.faces {
        let n = face.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let u = face[i];
            let v = face[(i + 1) % n];
            let key = if u < v { (u, v) } else { (v, u) };
            if !seen.insert(key) {
                continue;
            }
            let mut d = [0.0; 3];
            for c in 0..3 {
                d[c] = xyz[u * 3 + c] - xyz[v * 3 + c];
            }
            s += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        }
    }
    s
}

/// `∂(mesh laplacian energy)/∂x_f`.
pub fn grad_mesh_laplacian_wrt_xyz_free(
    mesh: &MeshStructure,
    xyz: &[f64],
    structure: &Structure,
) -> Vec<f64> {
    let mut g = vec![0.0; packed_free_dim(structure.num_free())];
    let mut seen = std::collections::HashSet::new();
    for face in &mesh.faces {
        let n = face.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let u = face[i];
            let v = face[(i + 1) % n];
            let key = if u < v { (u, v) } else { (v, u) };
            if !seen.insert(key) {
                continue;
            }
            for c in 0..3 {
                let dc = xyz[u * 3 + c] - xyz[v * 3 + c];
                if let Some(pu) = packed_free_index(structure, u) {
                    g[pu * 3 + c] += 2.0 * dc;
                }
                if let Some(pv) = packed_free_index(structure, v) {
                    g[pv * 3 + c] -= 2.0 * dc;
                }
            }
        }
    }
    g
}

/// `∂(½‖r‖²)/∂x_f` for [`residual_loss`].
pub fn grad_residual_wrt_xyz_free(state: &EquilibriumState, structure: &Structure) -> Vec<f64> {
    let nf = structure.num_free();
    let mut g = vec![0.0; nf * 3];
    for (a, &node) in structure.indices_free.iter().enumerate() {
        for c in 0..3 {
            g[a * 3 + c] = state.residuals[node * 3 + c];
        }
    }
    g
}

/// Sum of squared residuals on free nodes (equilibrium misfit).
pub fn residual_loss(state: &EquilibriumState, is_support: &[bool]) -> f64 {
    let mut s = 0.0;
    for i in 0..state.num_nodes() {
        if is_support[i] {
            continue;
        }
        for c in 0..3 {
            let r = state.residuals[3 * i + c];
            s += r * r;
        }
    }
    s
}
