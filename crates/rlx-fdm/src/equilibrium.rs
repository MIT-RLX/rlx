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

//! FDM equilibrium linear algebra (jax_fdm `EquilibriumModel`).

use crate::iterative::{IterativeConfig, equilibrium_iterative};
use crate::loads::{LoadState, nodes_load_at_mesh};
use crate::mesh::MeshStructure;
use crate::solve::solve_columns_dense;
use crate::sparse::nodes_free_positions_auto;
use crate::state::EquilibriumState;
use crate::structure::Structure;

#[derive(Debug, Clone, PartialEq)]
pub enum FdmError {
    SingularStiffness,
    Dimension(String),
    Validation(String),
}

/// Linear FDM step (`tmax = 1` in jax_fdm): solve for free node coordinates.
#[derive(Clone, Debug, Default)]
pub struct EquilibriumModel;

impl EquilibriumModel {
    /// `K = C_fᵀ diag(q) C_f` (dense).
    pub fn stiffness_matrix(q: &[f64], structure: &Structure) -> Vec<f64> {
        let nf = structure.num_free();
        let c_free = structure.connectivity_free();
        let ne = structure.num_edges;
        let mut k = vec![0.0; nf * nf];
        for i in 0..ne {
            // jax_fdm uses signed force densities (compression arches often q < 0).
            let qi = -q[i];
            for a in 0..nf {
                let cia = c_free[i * nf + a];
                if cia == 0.0 {
                    continue;
                }
                for b in 0..nf {
                    let cib = c_free[i * nf + b];
                    if cib == 0.0 {
                        continue;
                    }
                    k[a * nf + b] += qi * cia * cib;
                }
            }
        }
        k
    }

    /// Fixed-node contribution to the load vector: `R_fixed = C_fᵀ diag(q) C_a X_a`.
    pub fn residual_fixed_matrix(q: &[f64], xyz_fixed: &[f64], structure: &Structure) -> Vec<f64> {
        let nf = structure.num_free();
        let na = structure.num_fixed();
        let ne = structure.num_edges;
        let c_free = structure.connectivity_free();
        let c_fixed = structure.connectivity_fixed();
        let mut r = vec![0.0; nf * 3];
        for i in 0..ne {
            // jax_fdm uses signed force densities (compression arches often q < 0).
            let qi = -q[i];
            for comp in 0..3 {
                let mut sum_fixed = 0.0;
                for j in 0..na {
                    sum_fixed += c_fixed[i * na + j] * xyz_fixed[j * 3 + comp];
                }
                let contrib = qi * sum_fixed;
                for a in 0..nf {
                    r[a * 3 + comp] += c_free[i * nf + a] * contrib;
                }
            }
        }
        r
    }

    /// `P = L_f − R_fixed` where `L_f` are loads on free nodes.
    pub fn load_matrix(
        q: &[f64],
        xyz_fixed: &[f64],
        loads_nodes: &[f64],
        structure: &Structure,
    ) -> Vec<f64> {
        let nf = structure.num_free();
        let r_fixed = Self::residual_fixed_matrix(q, xyz_fixed, structure);
        let mut p = vec![0.0; nf * 3];
        for a in 0..nf {
            let node = structure.indices_free[a];
            for comp in 0..3 {
                p[a * 3 + comp] = loads_nodes[node * 3 + comp] - r_fixed[a * 3 + comp];
            }
        }
        p
    }

    /// Solve `K X_f = P` for each coordinate column (dense).
    pub fn nodes_free_positions(
        q: &[f64],
        xyz_fixed: &[f64],
        loads_nodes: &[f64],
        structure: &Structure,
    ) -> Result<Vec<f64>, FdmError> {
        let k = Self::stiffness_matrix(q, structure);
        let p = Self::load_matrix(q, xyz_fixed, loads_nodes, structure);
        let n = structure.num_free();
        solve_columns_dense(&k, &p, n, 3)
    }

    /// Linear or fixed-point equilibrium with optional sparse solver.
    pub fn equilibrium_with_config(
        q: &[f64],
        xyz_anchor: &[f64],
        load_state: &LoadState,
        structure: &Structure,
        edges: &[(usize, usize)],
        config: &IterativeConfig,
        mesh: Option<&MeshStructure>,
    ) -> Result<EquilibriumState, FdmError> {
        let na = structure.num_fixed();
        let mut xyz_fixed = vec![0.0; na * 3];
        for (j, &node) in structure.indices_fixed.iter().enumerate() {
            for c in 0..3 {
                xyz_fixed[j * 3 + c] = xyz_anchor[node * 3 + c];
            }
        }
        let xyz_free = if config.tmax <= 1 && !load_state.has_shape_dependent() {
            let loads = nodes_load_at_mesh(xyz_anchor, load_state, structure, edges, mesh);
            nodes_free_positions_auto(
                q,
                &xyz_fixed,
                &loads,
                structure,
                config.use_sparse,
                config.pcg_max_iter,
                config.pcg_tol,
            )?
        } else {
            equilibrium_iterative(
                q, &xyz_fixed, load_state, structure, edges, xyz_anchor, config, mesh,
            )?
        };
        let xyz = Self::nodes_positions(&xyz_free, &xyz_fixed, structure);
        let loads = nodes_load_at_mesh(&xyz, load_state, structure, edges, mesh);
        Ok(Self::equilibrium_state(q, &xyz, &loads, structure, edges))
    }

    /// Extract packed free coordinates from a full node position vector.
    pub fn pack_xyz_free(xyz: &[f64], structure: &Structure) -> Vec<f64> {
        let nf = structure.num_free();
        let mut out = vec![0.0; nf * 3];
        for (a, &node) in structure.indices_free.iter().enumerate() {
            for c in 0..3 {
                out[a * 3 + c] = xyz[node * 3 + c];
            }
        }
        out
    }

    /// Pack free then fixed coordinates and permute to natural node order.
    pub fn nodes_positions(xyz_free: &[f64], xyz_fixed: &[f64], structure: &Structure) -> Vec<f64> {
        let n = structure.num_nodes;
        let mut packed = vec![0.0; (structure.num_free() + structure.num_fixed()) * 3];
        packed[..xyz_free.len()].copy_from_slice(xyz_free);
        packed[xyz_free.len()..].copy_from_slice(xyz_fixed);
        let mut xyz = vec![0.0; n * 3];
        for node in 0..n {
            let row = structure.indices_freefixed[node];
            for c in 0..3 {
                xyz[node * 3 + c] = packed[row * 3 + c];
            }
        }
        xyz
    }

    pub fn edges_vectors(xyz: &[f64], structure: &Structure, edges: &[(usize, usize)]) -> Vec<f64> {
        let ne = structure.num_edges;
        let mut v = vec![0.0; ne * 3];
        for (e, &(u, v_)) in edges.iter().enumerate() {
            for c in 0..3 {
                v[e * 3 + c] = xyz[u * 3 + c] - xyz[v_ * 3 + c];
            }
        }
        v
    }

    pub fn edges_lengths(vectors: &[f64], num_edges: usize) -> Vec<f64> {
        let mut len = vec![0.0; num_edges];
        for e in 0..num_edges {
            let x = vectors[e * 3];
            let y = vectors[e * 3 + 1];
            let z = vectors[e * 3 + 2];
            len[e] = (x * x + y * y + z * z).sqrt();
        }
        len
    }

    pub fn edges_forces(q: &[f64], lengths: &[f64]) -> Vec<f64> {
        q.iter()
            .zip(lengths.iter())
            .map(|(&qi, &li)| qi * li)
            .collect()
    }

    /// `r = L − Cᵀ diag(q) C X` (jax_fdm `nodes_residuals`).
    pub fn nodes_residuals(
        q: &[f64],
        loads_nodes: &[f64],
        vectors: &[f64],
        structure: &Structure,
    ) -> Vec<f64> {
        let n = structure.num_nodes;
        let ne = structure.num_edges;
        let mut r = loads_nodes.to_vec();
        for e in 0..ne {
            let qi = q[e];
            for node in 0..n {
                let c = structure.c(e, node);
                if c == 0.0 {
                    continue;
                }
                for comp in 0..3 {
                    r[node * 3 + comp] -= c * qi * vectors[e * 3 + comp];
                }
            }
        }
        r
    }

    pub fn equilibrium_state(
        q: &[f64],
        xyz: &[f64],
        loads_nodes: &[f64],
        structure: &Structure,
        edges: &[(usize, usize)],
    ) -> EquilibriumState {
        let vectors = Self::edges_vectors(xyz, structure, edges);
        let lengths = Self::edges_lengths(&vectors, structure.num_edges);
        let forces = Self::edges_forces(q, &lengths);
        let residuals = Self::nodes_residuals(q, loads_nodes, &vectors, structure);
        EquilibriumState {
            xyz: xyz.to_vec(),
            lengths,
            forces,
            residuals,
            loads: loads_nodes.to_vec(),
            vectors,
        }
    }

    /// One linear FDM equilibrium solve (jax_fdm `EquilibriumModel.__call__` with `tmax=1`).
    pub fn equilibrium(
        q: &[f64],
        xyz_anchor: &[f64],
        loads_nodes: &[f64],
        structure: &Structure,
        edges: &[(usize, usize)],
    ) -> Result<EquilibriumState, FdmError> {
        let load_state = LoadState::from_network(loads_nodes);
        Self::equilibrium_with_config(
            q,
            xyz_anchor,
            &load_state,
            structure,
            edges,
            &IterativeConfig::linear(),
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::Network;

    #[test]
    fn three_node_chain_interior_sags_negative_z() {
        let mut net =
            Network::from_polyline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], -1.0);
        net.anchor_nodes(&[0, 2]);
        net.loads_on_free([0.0, 0.0, -1.0]);
        let s = Structure::from_network(&net);
        let eq = super::EquilibriumModel::equilibrium(&net.q, &net.xyz, &net.loads, &s, &net.edges)
            .expect("eq");
        assert!(eq.xyz[3 + 2] < -0.01, "z={}", eq.xyz[3 + 2]);
    }

    #[test]
    fn pack_xyz_free_roundtrip() {
        let net = Network::arch_chain(5.0, 4, -1.0, -0.1);
        let s = Structure::from_network(&net);
        let eq = EquilibriumModel::equilibrium(&net.q, &net.xyz, &net.loads, &s, &net.edges)
            .expect("eq");
        let packed = EquilibriumModel::pack_xyz_free(&eq.xyz, &s);
        let na = s.num_fixed();
        let mut xf = vec![0.0; na * 3];
        for (j, &node) in s.indices_fixed.iter().enumerate() {
            for c in 0..3 {
                xf[j * 3 + c] = net.xyz[node * 3 + c];
            }
        }
        let round = EquilibriumModel::nodes_positions(&packed, &xf, &s);
        for (a, b) in eq.xyz.iter().zip(round.iter()) {
            assert!((a - b).abs() < 1e-12, "roundtrip mismatch {a} vs {b}");
        }
    }

    #[test]
    fn stiffness_is_symmetric_for_chain() {
        let net = Network::arch_chain(5.0, 4, -1.0, -0.1);
        let s = Structure::from_network(&net);
        let k = EquilibriumModel::stiffness_matrix(&net.q, &s);
        let nf = s.num_free();
        for i in 0..nf {
            for j in 0..nf {
                assert!((k[i * nf + j] - k[j * nf + i]).abs() < 1e-12);
            }
        }
    }
}
