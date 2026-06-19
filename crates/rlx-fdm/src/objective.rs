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

//! Form-finding objectives (jax_fdm `goals` + `losses.errors`).

use crate::goals::{
    EdgeIndex, GoalIndex, accumulate_edge_length_grad, edge_force, grad_edge_force_wrt_xyz_free,
    grad_edge_lengths_wrt_xyz_free, grad_mesh_laplacian_wrt_xyz_free,
    grad_mesh_mean_face_rectangular_wrt_xyz_free, grad_mesh_mean_planarity_wrt_xyz_free,
    grad_mesh_total_area_wrt_xyz_free, grad_min_free_z_wrt_xyz_free, grad_node_coord_wrt_xyz_free,
    grad_residual_wrt_xyz_free, mean_edge_force, mean_edge_length, mesh_laplacian_energy,
    mesh_mean_face_rectangular, mesh_mean_planarity, mesh_total_area, min_free_z, network_loadpath,
    node_coord, packed_free_dim, residual_loss,
};
use crate::mesh::MeshStructure;
use crate::state::EquilibriumState;
use crate::structure::Structure;

/// Axis index for [`Goal::NodeCoord`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

/// A weighted objective term (jax_fdm `Goal` + squared error).
#[derive(Clone, Debug)]
pub enum Goal {
    /// Minimize `weight · (loadpath − target)²` (jax_fdm `NetworkLoadPathGoal`).
    NetworkLoadpath { target: f64, weight: f64 },
    /// Minimize `weight · (length[edge] − target)²` (jax_fdm `EdgeLengthGoal`).
    EdgeLength {
        edge: EdgeIndex,
        target: f64,
        weight: f64,
    },
    /// Minimize `weight · (mean length − target)²`.
    MeanEdgeLength { target: f64, weight: f64 },
    /// Minimize `weight · (force[edge] − target)²` (jax_fdm `EdgeForceGoal`).
    EdgeForce {
        edge: EdgeIndex,
        target: f64,
        weight: f64,
    },
    /// Minimize `weight · (mean |force| − target)²`.
    MeanEdgeForce { target: f64, weight: f64 },
    /// Minimize `weight · (node coord − target)²`.
    NodeCoord {
        node: usize,
        axis: CoordAxis,
        target: f64,
        weight: f64,
    },
    /// Minimize `weight · (min free-node z − target)²` (sag control).
    MinFreeZ { target: f64, weight: f64 },
    /// Penalize equilibrium residual norm (useful as soft constraint).
    Residual { weight: f64 },
    /// Minimize `weight · (total mesh area − target)²` (requires mesh on network).
    MeshArea { target: f64, weight: f64 },
    /// Minimize `weight · (mean face planarity − target)²`.
    MeshPlanarity { target: f64, weight: f64 },
    /// Minimize `weight · (mean face rectangular deviation − target)²` (jax_fdm `FaceRectangularGoal`).
    MeshFaceRectangular { target: f64, weight: f64 },
    /// Minimize `weight · (mesh edge spring energy − target)²` (smoothing).
    MeshLaplacian { target: f64, weight: f64 },
}

impl Goal {
    pub fn network_loadpath(target: f64, weight: f64) -> Self {
        Self::NetworkLoadpath { target, weight }
    }

    pub fn edge_length(edge: usize, target: f64, weight: f64) -> Self {
        Self::EdgeLength {
            edge: EdgeIndex(edge),
            target,
            weight,
        }
    }

    pub fn mean_edge_length(target: f64, weight: f64) -> Self {
        Self::MeanEdgeLength { target, weight }
    }

    pub fn edge_force(edge: usize, target: f64, weight: f64) -> Self {
        Self::EdgeForce {
            edge: EdgeIndex(edge),
            target,
            weight,
        }
    }

    pub fn mean_edge_force(target: f64, weight: f64) -> Self {
        Self::MeanEdgeForce { target, weight }
    }

    pub fn node_coord(node: usize, axis: CoordAxis, target: f64, weight: f64) -> Self {
        Self::NodeCoord {
            node,
            axis,
            target,
            weight,
        }
    }

    pub fn node_z(node: usize, target: f64, weight: f64) -> Self {
        Self::node_coord(node, CoordAxis::Z, target, weight)
    }

    pub fn min_free_z(target: f64, weight: f64) -> Self {
        Self::MinFreeZ { target, weight }
    }

    pub fn residual(weight: f64) -> Self {
        Self::Residual { weight }
    }

    pub fn mesh_area(target: f64, weight: f64) -> Self {
        Self::MeshArea { target, weight }
    }

    pub fn mesh_planarity(target: f64, weight: f64) -> Self {
        Self::MeshPlanarity { target, weight }
    }

    pub fn mesh_face_rectangular(target: f64, weight: f64) -> Self {
        Self::MeshFaceRectangular { target, weight }
    }

    pub fn mesh_laplacian(target: f64, weight: f64) -> Self {
        Self::MeshLaplacian { target, weight }
    }

    pub fn prediction(&self, state: &EquilibriumState, is_support: &[bool]) -> f64 {
        match self {
            Self::NetworkLoadpath { .. } => network_loadpath(state),
            Self::EdgeLength { edge, .. } => {
                let e = edge.0;
                state.lengths.get(e).copied().unwrap_or(0.0)
            }
            Self::MeanEdgeLength { .. } => mean_edge_length(state),
            Self::EdgeForce { edge, .. } => edge_force(state, edge.0),
            Self::MeanEdgeForce { .. } => mean_edge_force(state),
            Self::NodeCoord { node, axis, .. } => node_coord(state, *node, *axis as usize),
            Self::MinFreeZ { .. } => {
                let mut min_z = f64::INFINITY;
                for (i, &sup) in is_support.iter().enumerate() {
                    if !sup {
                        min_z = min_z.min(state.xyz[i * 3 + 2]);
                    }
                }
                min_z
            }
            Self::Residual { .. } => residual_loss(state, is_support).sqrt(),
            Self::MeshArea { .. }
            | Self::MeshPlanarity { .. }
            | Self::MeshFaceRectangular { .. }
            | Self::MeshLaplacian { .. } => 0.0,
        }
    }

    /// Like [`Self::prediction`] but with structure for mesh-aware goals.
    pub fn prediction_with_structure(
        &self,
        state: &EquilibriumState,
        structure: &Structure,
        is_support: &[bool],
        mesh: Option<&MeshStructure>,
    ) -> f64 {
        match self {
            Self::MinFreeZ { .. } => min_free_z(state, structure),
            Self::MeshArea { .. } => mesh.map(|m| mesh_total_area(m, &state.xyz)).unwrap_or(0.0),
            Self::MeshPlanarity { .. } => mesh
                .map(|m| mesh_mean_planarity(m, &state.xyz))
                .unwrap_or(0.0),
            Self::MeshFaceRectangular { .. } => mesh
                .map(|m| mesh_mean_face_rectangular(m, &state.xyz))
                .unwrap_or(0.0),
            Self::MeshLaplacian { .. } => mesh
                .map(|m| mesh_laplacian_energy(m, &state.xyz))
                .unwrap_or(0.0),
            _ => self.prediction(state, is_support),
        }
    }

    pub fn target(&self) -> f64 {
        match self {
            Self::NetworkLoadpath { target, .. }
            | Self::EdgeLength { target, .. }
            | Self::MeanEdgeLength { target, .. }
            | Self::EdgeForce { target, .. }
            | Self::MeanEdgeForce { target, .. }
            | Self::NodeCoord { target, .. }
            | Self::MinFreeZ { target, .. }
            | Self::MeshArea { target, .. }
            | Self::MeshPlanarity { target, .. }
            | Self::MeshFaceRectangular { target, .. }
            | Self::MeshLaplacian { target, .. } => *target,
            Self::Residual { .. } => 0.0,
        }
    }

    pub fn weight(&self) -> f64 {
        match self {
            Self::NetworkLoadpath { weight, .. }
            | Self::EdgeLength { weight, .. }
            | Self::MeanEdgeLength { weight, .. }
            | Self::EdgeForce { weight, .. }
            | Self::MeanEdgeForce { weight, .. }
            | Self::NodeCoord { weight, .. }
            | Self::MinFreeZ { weight, .. }
            | Self::Residual { weight, .. }
            | Self::MeshArea { weight, .. }
            | Self::MeshPlanarity { weight, .. }
            | Self::MeshFaceRectangular { weight, .. }
            | Self::MeshLaplacian { weight, .. } => *weight,
        }
    }

    /// Squared-error contribution (jax_fdm `SquaredError`).
    pub fn loss(&self, state: &EquilibriumState, is_support: &[bool]) -> f64 {
        let d = self.prediction(state, is_support) - self.target();
        self.weight() * d * d
    }

    pub fn loss_with_structure(
        &self,
        state: &EquilibriumState,
        structure: &Structure,
        is_support: &[bool],
        mesh: Option<&MeshStructure>,
    ) -> f64 {
        let d = self.prediction_with_structure(state, structure, is_support, mesh) - self.target();
        self.weight() * d * d
    }

    /// `∂L/∂x_f` for this goal (positions only; chain to `q` via implicit adjoint).
    pub fn grad_xyz_free(
        &self,
        state: &EquilibriumState,
        structure: &Structure,
        edges: &[(usize, usize)],
        is_support: &[bool],
        mesh: Option<&MeshStructure>,
    ) -> Vec<f64> {
        let nf = structure.num_free();
        let dim = packed_free_dim(nf);
        let pred = self.prediction_with_structure(state, structure, is_support, mesh);
        let scale = 2.0 * self.weight() * (pred - self.target());
        let mut g = vec![0.0; dim];
        match self {
            Self::NetworkLoadpath { .. } => {
                for e in 0..edges.len() {
                    let fe = state.forces[e].abs();
                    accumulate_edge_length_grad(&mut g, state, structure, edges, e, fe * scale);
                }
            }
            Self::EdgeLength { edge, .. } => {
                let e = edge.0;
                if e < edges.len() {
                    accumulate_edge_length_grad(&mut g, state, structure, edges, e, scale);
                }
            }
            Self::MeanEdgeLength { .. } => {
                let gg = grad_edge_lengths_wrt_xyz_free(
                    state,
                    structure,
                    edges,
                    scale / edges.len() as f64,
                );
                for (a, b) in g.iter_mut().zip(gg.iter()) {
                    *a += *b;
                }
            }
            Self::EdgeForce { edge, .. } => {
                let e = edge.0;
                let qeff = state.forces.get(e).copied().unwrap_or(0.0)
                    / state.lengths.get(e).copied().unwrap_or(1.0).max(1e-12);
                let gg = grad_edge_force_wrt_xyz_free(state, structure, edges, e, scale * qeff);
                for (a, b) in g.iter_mut().zip(gg.iter()) {
                    *a += *b;
                }
            }
            Self::MeanEdgeForce { .. } => {
                for e in 0..edges.len() {
                    let qeff = state.forces[e] / state.lengths[e].max(1e-12);
                    let gg = grad_edge_force_wrt_xyz_free(
                        state,
                        structure,
                        edges,
                        e,
                        scale * qeff / edges.len() as f64,
                    );
                    for (a, b) in g.iter_mut().zip(gg.iter()) {
                        *a += *b;
                    }
                }
            }
            Self::NodeCoord { node, axis, .. } => {
                let gg = grad_node_coord_wrt_xyz_free(structure, *node, *axis as usize, scale);
                for (a, b) in g.iter_mut().zip(gg.iter()) {
                    *a += *b;
                }
            }
            Self::MinFreeZ { .. } => {
                let gg = grad_min_free_z_wrt_xyz_free(state, structure);
                for (a, b) in g.iter_mut().zip(gg.iter()) {
                    *a += *b * scale;
                }
            }
            Self::Residual { .. } => {
                let gg = grad_residual_wrt_xyz_free(state, structure);
                for (a, b) in g.iter_mut().zip(gg.iter()) {
                    *a += *b * scale;
                }
            }
            Self::MeshArea { .. } => {
                if let Some(m) = mesh {
                    let gg = grad_mesh_total_area_wrt_xyz_free(m, &state.xyz, structure);
                    for (a, b) in g.iter_mut().zip(gg.iter()) {
                        *a += *b * scale;
                    }
                }
            }
            Self::MeshPlanarity { .. } => {
                if let Some(m) = mesh {
                    let gg = grad_mesh_mean_planarity_wrt_xyz_free(m, &state.xyz, structure);
                    for (a, b) in g.iter_mut().zip(gg.iter()) {
                        *a += *b * scale;
                    }
                }
            }
            Self::MeshFaceRectangular { .. } => {
                if let Some(m) = mesh {
                    let gg = grad_mesh_mean_face_rectangular_wrt_xyz_free(m, &state.xyz, structure);
                    for (a, b) in g.iter_mut().zip(gg.iter()) {
                        *a += *b * scale;
                    }
                }
            }
            Self::MeshLaplacian { .. } => {
                if let Some(m) = mesh {
                    let gg = grad_mesh_laplacian_wrt_xyz_free(m, &state.xyz, structure);
                    for (a, b) in g.iter_mut().zip(gg.iter()) {
                        *a += *b * scale;
                    }
                }
            }
        }
        g
    }
}

/// Sum of goal losses.
pub fn goals_loss(goals: &[Goal], state: &EquilibriumState, is_support: &[bool]) -> f64 {
    goals.iter().map(|g| g.loss(state, is_support)).sum()
}

pub fn goals_loss_with_structure(
    goals: &[Goal],
    state: &EquilibriumState,
    structure: &Structure,
    is_support: &[bool],
    mesh: Option<&MeshStructure>,
) -> f64 {
    goals
        .iter()
        .map(|g| g.loss_with_structure(state, structure, is_support, mesh))
        .sum()
}

/// Sum of goal `∂L/∂x_f` vectors.
pub fn goals_grad_xyz_free(
    goals: &[Goal],
    state: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
    is_support: &[bool],
    mesh: Option<&MeshStructure>,
) -> Vec<f64> {
    let dim = packed_free_dim(structure.num_free());
    let mut g = vec![0.0; dim];
    for goal in goals {
        let gg = goal.grad_xyz_free(state, structure, edges, is_support, mesh);
        for (a, b) in g.iter_mut().zip(gg.iter()) {
            *a += *b;
        }
    }
    g
}

/// Per-goal diagnostics after a solve.
#[derive(Clone, Debug)]
pub struct GoalReport {
    pub index: GoalIndex,
    pub prediction: f64,
    pub target: f64,
    pub loss: f64,
}

pub fn goals_report(
    goals: &[Goal],
    state: &EquilibriumState,
    structure: &Structure,
    is_support: &[bool],
    mesh: Option<&MeshStructure>,
) -> Vec<GoalReport> {
    goals
        .iter()
        .enumerate()
        .map(|(i, g)| GoalReport {
            index: GoalIndex(i),
            prediction: g.prediction_with_structure(state, structure, is_support, mesh),
            target: g.target(),
            loss: g.loss_with_structure(state, structure, is_support, mesh),
        })
        .collect()
}
