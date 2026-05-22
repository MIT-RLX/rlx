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

//! Soft constraints for constrained form-finding (jax_fdm `constraints` + `LogMaxError`).

use crate::geometry::{angle_vectors, cross, normalize};
use crate::goals::{accumulate_edge_length_grad, packed_free_dim, EdgeIndex};
use crate::mesh::MeshStructure;
use crate::network::Vec3;
use crate::state::EquilibriumState;
use crate::structure::Structure;

/// Penalized constraint on equilibrium quantities.
#[derive(Clone, Debug)]
pub enum Constraint {
    /// Keep force density `q[edge]` in `[low, up]` (projection also applied each step).
    EdgeQ {
        edge: Option<EdgeIndex>,
        low: f64,
        up: f64,
        weight: f64,
    },
    /// Upper bound on edge length: `log1p(max(0, l − up))` (jax_fdm `LogMaxError`).
    EdgeLengthMax {
        edge: EdgeIndex,
        up: f64,
        weight: f64,
    },
    /// Lower bound on edge length: `log1p(max(0, low − l))`.
    EdgeLengthMin {
        edge: EdgeIndex,
        low: f64,
        weight: f64,
    },
    /// Upper bound on axial force `|q·l|` (jax_fdm `EdgeForceConstraint`).
    EdgeForceMax {
        edge: EdgeIndex,
        up: f64,
        weight: f64,
    },
    /// Angle between edge vector and `vector` in `[low, up]` rad (jax_fdm `EdgeAngleConstraint`).
    EdgeAngle {
        edge: EdgeIndex,
        vector: [f64; 3],
        low: f64,
        up: f64,
        weight: f64,
    },
    /// Angle between mean incident-edge direction at `node` and `vector`.
    NodeTangent {
        node: usize,
        vector: [f64; 3],
        low: f64,
        up: f64,
        weight: f64,
    },
    /// Angle between mean face normal at `node` and `vector` (needs mesh).
    NodeNormalAngle {
        node: usize,
        vector: [f64; 3],
        low: f64,
        up: f64,
        weight: f64,
    },
}

impl Constraint {
    pub fn all_edge_q(low: f64, up: f64, weight: f64) -> Self {
        Self::EdgeQ {
            edge: None,
            low,
            up,
            weight,
        }
    }

    pub fn edge_length_max(edge: usize, up: f64, weight: f64) -> Self {
        Self::EdgeLengthMax {
            edge: EdgeIndex(edge),
            up,
            weight,
        }
    }

    pub fn edge_length_min(edge: usize, low: f64, weight: f64) -> Self {
        Self::EdgeLengthMin {
            edge: EdgeIndex(edge),
            low,
            weight,
        }
    }

    pub fn edge_angle(edge: usize, vector: [f64; 3], low: f64, up: f64, weight: f64) -> Self {
        Self::EdgeAngle {
            edge: EdgeIndex(edge),
            vector,
            low,
            up,
            weight,
        }
    }

    pub fn node_tangent(node: usize, vector: [f64; 3], low: f64, up: f64, weight: f64) -> Self {
        Self::NodeTangent {
            node,
            vector,
            low,
            up,
            weight,
        }
    }

    pub fn is_nonlinear(&self) -> bool {
        matches!(
            self,
            Self::EdgeAngle { .. } | Self::NodeTangent { .. } | Self::NodeNormalAngle { .. }
        )
    }

    /// Soft penalty value (zero when feasible).
    pub fn penalty(&self, state: &EquilibriumState, q: &[f64]) -> f64 {
        match self {
            Self::EdgeQ {
                edge,
                low,
                up,
                weight,
            } => {
                let mut s = 0.0;
                let edge_indices: Vec<usize> = match edge {
                    Some(EdgeIndex(e)) => vec![*e],
                    None => (0..q.len()).collect(),
                };
                for &e in &edge_indices {
                    if e >= q.len() {
                        continue;
                    }
                    let v = q[e];
                    s += log_barrier_below(v, *low) + log_barrier_above(v, *up);
                }
                weight * s
            }
            Self::EdgeLengthMax { edge, up, weight } => {
                let e = edge.0;
                if e >= state.lengths.len() {
                    return 0.0;
                }
                weight * log1p_violation_above(state.lengths[e], *up)
            }
            Self::EdgeLengthMin { edge, low, weight } => {
                let e = edge.0;
                if e >= state.lengths.len() {
                    return 0.0;
                }
                weight * log1p_violation_above(*low, state.lengths[e])
            }
            Self::EdgeForceMax { edge, up, weight } => {
                let e = edge.0;
                if e >= state.forces.len() {
                    return 0.0;
                }
                weight * log1p_violation_above(state.forces[e].abs(), *up)
            }
            Self::EdgeAngle { low, up, weight, .. }
            | Self::NodeTangent { low, up, weight, .. }
            | Self::NodeNormalAngle { low, up, weight, .. } => {
                let ang = self.constraint_angle(state, None, &[]);
                weight * (log1p_violation_above(ang, *up) + log1p_violation_above(*low, ang))
            }
        }
    }

    fn constraint_angle(
        &self,
        state: &EquilibriumState,
        mesh: Option<&MeshStructure>,
        edges: &[(usize, usize)],
    ) -> f64 {
        match self {
            Self::EdgeAngle { edge, vector, .. } => {
                let e = edge.0;
                if e >= state.vectors.len() / 3 {
                    return 0.0;
                }
                let v = [
                    state.vectors[e * 3],
                    state.vectors[e * 3 + 1],
                    state.vectors[e * 3 + 2],
                ];
                angle_vectors(v, *vector)
            }
            Self::NodeTangent { node, vector, .. } => {
                let t = node_tangent_vector(state, edges, *node);
                angle_vectors(t, *vector)
            }
            Self::NodeNormalAngle { node, vector, .. } => {
                let n = node_mean_normal(state, mesh, *node);
                angle_vectors(n, *vector)
            }
            _ => 0.0,
        }
    }

    /// Nonlinear constraint values `g(x)` for SLSQP (`g ≤ 0` feasible).
    pub fn nonlinear_ineq(&self, state: &EquilibriumState, mesh: Option<&MeshStructure>, edges: &[(usize, usize)]) -> Vec<f64> {
        if !self.is_nonlinear() {
            return vec![];
        }
        let ang = self.constraint_angle(state, mesh, edges);
        match self {
            Self::EdgeAngle { low, up, .. }
            | Self::NodeTangent { low, up, .. }
            | Self::NodeNormalAngle { low, up, .. } => {
                vec![*low - ang, ang - *up]
            }
            _ => vec![],
        }
    }

    /// `∂penalty/∂x_f` for chaining through equilibrium (edge-length constraints only).
    pub fn grad_xyz_free(
        &self,
        state: &EquilibriumState,
        structure: &Structure,
        edges: &[(usize, usize)],
    ) -> Vec<f64> {
        let nf = structure.num_free();
        let dim = packed_free_dim(nf);
        let mut g = vec![0.0; dim];
        match self {
            Self::EdgeLengthMax { edge, up, weight } => {
                let e = edge.0;
                if e < state.lengths.len() && state.lengths[e] > *up {
                    let scale = weight / (1.0 + (state.lengths[e] - up).max(0.0));
                    accumulate_edge_length_grad(&mut g, state, structure, edges, e, scale);
                }
            }
            Self::EdgeLengthMin { edge, low, weight } => {
                let e = edge.0;
                if e < state.lengths.len() && state.lengths[e] < *low {
                    let scale = -weight / (1.0 + (low - state.lengths[e]).max(0.0));
                    accumulate_edge_length_grad(&mut g, state, structure, edges, e, scale);
                }
            }
            Self::EdgeQ { .. } | Self::EdgeForceMax { .. } | Self::EdgeAngle { .. } | Self::NodeTangent { .. } | Self::NodeNormalAngle { .. } => {}
        }
        g
    }

    /// Project `q` onto hard `EdgeQ` bounds (always applied after a step).
    pub fn project_q(&self, q: &mut [f64]) {
        if let Self::EdgeQ { edge, low, up, .. } = self {
            match edge {
                Some(EdgeIndex(e)) => {
                    if *e < q.len() {
                        q[*e] = q[*e].clamp(*low, *up);
                    }
                }
                None => {
                    for qi in q.iter_mut() {
                        *qi = qi.clamp(*low, *up);
                    }
                }
            }
        }
    }
}

fn log1p_violation_above(value: f64, bound: f64) -> f64 {
    let v = (value - bound).max(0.0);
    (1.0 + v).ln()
}

fn log_barrier_below(v: f64, low: f64) -> f64 {
    log1p_violation_above(low, v)
}

fn log_barrier_above(v: f64, up: f64) -> f64 {
    log1p_violation_above(v, up)
}

pub fn constraints_penalty(constraints: &[Constraint], state: &EquilibriumState, q: &[f64]) -> f64 {
    constraints
        .iter()
        .map(|c| c.penalty(state, q))
        .sum()
}

pub fn constraints_have_nonlinear(constraints: &[Constraint]) -> bool {
    constraints.iter().any(|c| c.is_nonlinear())
}

pub fn nonlinear_ineq_values(
    constraints: &[Constraint],
    state: &EquilibriumState,
    mesh: Option<&MeshStructure>,
    edges: &[(usize, usize)],
) -> Vec<f64> {
    constraints
        .iter()
        .flat_map(|c| c.nonlinear_ineq(state, mesh, edges))
        .collect()
}

pub fn constraints_grad_xyz_free(
    constraints: &[Constraint],
    state: &EquilibriumState,
    structure: &Structure,
    edges: &[(usize, usize)],
) -> Vec<f64> {
    let dim = packed_free_dim(structure.num_free());
    let mut g = vec![0.0; dim];
    for c in constraints {
        let cg = c.grad_xyz_free(state, structure, edges);
        for (a, b) in g.iter_mut().zip(cg.iter()) {
            *a += *b;
        }
    }
    g
}

fn node_tangent_vector(state: &EquilibriumState, edges: &[(usize, usize)], node: usize) -> Vec3 {
    let mut sum = [0.0f64; 3];
    let mut count = 0usize;
    for (e, &(u, v)) in edges.iter().enumerate() {
        if u != node && v != node {
            continue;
        }
        let mut d = [
            state.vectors[e * 3],
            state.vectors[e * 3 + 1],
            state.vectors[e * 3 + 2],
        ];
        if v == node {
            d[0] = -d[0];
            d[1] = -d[1];
            d[2] = -d[2];
        }
        let udir = normalize(d);
        sum[0] += udir[0];
        sum[1] += udir[1];
        sum[2] += udir[2];
        count += 1;
    }
    if count == 0 {
        return [0.0, 1.0, 0.0];
    }
    normalize(sum)
}

fn node_mean_normal(state: &EquilibriumState, mesh: Option<&MeshStructure>, node: usize) -> Vec3 {
    let Some(mesh) = mesh else {
        return [0.0, 0.0, 1.0];
    };
    let xyz = &state.xyz;
    let mut acc = [0.0f64; 3];
    let mut n = 0usize;
    for (fi, face) in mesh.faces.iter().enumerate() {
        if !face.contains(&node) {
            continue;
        }
        let pts: Vec<Vec3> = face
            .iter()
            .map(|&vi| [xyz[vi * 3], xyz[vi * 3 + 1], xyz[vi * 3 + 2]])
            .collect();
        if pts.len() < 3 {
            continue;
        }
        let c = mesh.face_centroids(&state.xyz)[fi];
        let a = pts[0];
        let mut normal = [0.0; 3];
        for i in 0..pts.len() {
            let b = pts[i];
            let d = pts[(i + 1) % pts.len()];
            let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let e2 = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
            let cr = cross(e1, e2);
            normal[0] += cr[0];
            normal[1] += cr[1];
            normal[2] += cr[2];
        }
        let _ = c;
        let nn = normalize(normal);
        acc[0] += nn[0];
        acc[1] += nn[1];
        acc[2] += nn[2];
        n += 1;
    }
    if n == 0 {
        return [0.0, 0.0, 1.0];
    }
    normalize(acc)
}
