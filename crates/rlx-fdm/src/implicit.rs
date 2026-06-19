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

//! Implicit / adjoint sensitivities (jax_fdm `solver_fixedpoint_implicit`).
//!
//! Linear equilibrium `K(q) x = P(q)` with `K = C_fᵀ diag(−q) C_f` and
//! `P = L_f − C_fᵀ diag(−q) C_a X_a` uses the implicit-function rule:
//!
//! ```text
//! λ = K⁻ᵀ (∂L/∂x)   (K symmetric → K⁻¹ ∂L/∂x per coordinate column)
//! ∂L/∂q_e = λᵀ (∂K/∂q_e) x + λᵀ (∂P/∂q_e)
//! ```
//!
//! Nonlinear fixed-point maps use an unrolled adjoint through
//! `x_{k+1} = K⁻¹ P(x_k)` ([`grad_loss_wrt_q_fixedpoint`]), including edge and
//! global and local (LCS) face tributary loads.

use crate::equilibrium::{EquilibriumModel, FdmError};
use crate::iterative::{
    IterativeConfig, config_for_implicit_adjoint, equilibrium_iterative,
    equilibrium_iterative_trajectory,
};
use crate::loads::{
    LoadState, nodes_load_at_mesh, transpose_edge_loads_jacobian, transpose_face_loads_jacobian,
};
use crate::mesh::MeshStructure;
use crate::solve::solve_columns_dense;
use crate::sparse::pattern_fast;
use crate::structure::Structure;

/// Gradient of a loss that depends on packed free coordinates w.r.t. `q`.
#[derive(Clone, Debug)]
pub struct QGradient {
    pub dq: Vec<f64>,
}

/// Gradient w.r.t. packed fixed support coordinates (`na × 3`, same layout as `xyz_fixed`).
#[derive(Clone, Debug)]
pub struct XFixedGradient {
    pub dxf: Vec<f64>,
}

/// Adjoint linear solve backend (dense LU vs CSR PCG).
#[derive(Clone, Debug)]
pub struct AdjointSolveConfig {
    pub use_sparse: bool,
    pub pcg_max_iter: u32,
    pub pcg_tol: f64,
    /// Use PCG when `num_free ≥ sparse_min_free`.
    pub sparse_min_free: usize,
}

impl Default for AdjointSolveConfig {
    fn default() -> Self {
        Self {
            use_sparse: false,
            pcg_max_iter: 4000,
            pcg_tol: 1e-10,
            sparse_min_free: 32,
        }
    }
}

impl From<&IterativeConfig> for AdjointSolveConfig {
    fn from(c: &IterativeConfig) -> Self {
        Self {
            use_sparse: c.use_sparse,
            pcg_max_iter: c.pcg_max_iter,
            pcg_tol: c.pcg_tol,
            sparse_min_free: 32,
        }
    }
}

impl AdjointSolveConfig {
    pub fn with_sparse(mut self, sparse: bool) -> Self {
        self.use_sparse = sparse;
        self
    }
}

/// `λ = K⁻¹ v` for packed `v` (`nf × 3`), using dense or PCG to match forward [`nodes_free_positions_auto`].
pub fn solve_adjoint_columns(
    q: &[f64],
    structure: &Structure,
    rhs_xyz: &[f64],
    config: &AdjointSolveConfig,
) -> Result<Vec<f64>, FdmError> {
    let nf = structure.num_free();
    if rhs_xyz.len() != nf * 3 {
        return Err(FdmError::Dimension(format!(
            "adjoint rhs len {} vs {nf}*3",
            rhs_xyz.len()
        )));
    }
    if config.use_sparse && nf >= config.sparse_min_free {
        let pat = pattern_fast(structure);
        return pat.solve_xyz(q, rhs_xyz, config.pcg_max_iter, config.pcg_tol);
    }
    let k = EquilibriumModel::stiffness_matrix(q, structure);
    solve_columns_dense(&k, rhs_xyz, nf, 3)
}

/// `dL/dq` via implicit adjoint (linear FDM, constant nodal loads).
///
/// `loss_grad_xyz_free` is `∂L/∂x_f` in packed free order `[x₀,y₀,z₀, x₁,…]`
/// (same layout as [`EquilibriumModel::nodes_free_positions`]).
pub fn grad_loss_wrt_q_linear(
    q: &[f64],
    xyz_fixed: &[f64],
    _loads_nodes: &[f64],
    structure: &Structure,
    xyz_star: &[f64],
    loss_grad_xyz_free: &[f64],
) -> Result<QGradient, FdmError> {
    grad_loss_wrt_q_linear_with_solver(
        q,
        xyz_fixed,
        structure,
        xyz_star,
        loss_grad_xyz_free,
        &AdjointSolveConfig::default(),
    )
}

/// Like [`grad_loss_wrt_q_linear`] with explicit adjoint solver (PCG when sparse).
pub fn grad_loss_wrt_q_linear_with_solver(
    q: &[f64],
    xyz_fixed: &[f64],
    structure: &Structure,
    xyz_star: &[f64],
    loss_grad_xyz_free: &[f64],
    adjoint: &AdjointSolveConfig,
) -> Result<QGradient, FdmError> {
    let nf = structure.num_free();
    let ne = structure.num_edges;
    let expected = nf * 3;
    if xyz_star.len() != expected || loss_grad_xyz_free.len() != expected {
        return Err(FdmError::Dimension(format!(
            "xyz_star/loss_grad len {} vs expected {expected}",
            xyz_star.len()
        )));
    }

    let mut dq = vec![0.0; ne];
    let lambda_xyz = solve_adjoint_columns(q, structure, loss_grad_xyz_free, adjoint)?;
    accumulate_dq_from_lambda(&mut dq, &lambda_xyz, q, xyz_fixed, structure, xyz_star);
    Ok(QGradient { dq })
}

/// `dL/dX_a` (packed fixed coords) for linear FDM with constant nodal loads.
pub fn grad_loss_wrt_xyz_fixed_linear(
    q: &[f64],
    structure: &Structure,
    _xyz_star: &[f64],
    loss_grad_xyz_free: &[f64],
    adjoint: &AdjointSolveConfig,
) -> Result<XFixedGradient, FdmError> {
    let na = structure.num_fixed();
    let lambda_xyz = solve_adjoint_columns(q, structure, loss_grad_xyz_free, adjoint)?;
    let mut dxf = vec![0.0; na * 3];
    accumulate_dxf_from_lambda(&mut dxf, &lambda_xyz, q, structure);
    Ok(XFixedGradient { dxf })
}

/// Add `∂L/∂q` from adjoint `λ = K⁻ᵀ v` at equilibrium `x`.
pub fn accumulate_dq_from_lambda(
    dq: &mut [f64],
    lambda_xyz: &[f64],
    _q: &[f64],
    xyz_fixed: &[f64],
    structure: &Structure,
    xyz_star: &[f64],
) {
    let nf = structure.num_free();
    let na = structure.num_fixed();
    let ne = structure.num_edges;
    let c_free = structure.connectivity_free();
    let c_fixed = structure.connectivity_fixed();
    for comp in 0..3 {
        let lambda: Vec<f64> = (0..nf).map(|a| lambda_xyz[a * 3 + comp]).collect();
        for e in 0..ne {
            let mut fixed_sum = 0.0;
            for j in 0..na {
                fixed_sum += c_fixed[e * na + j] * xyz_fixed[j * 3 + comp];
            }
            for a in 0..nf {
                let cia = c_free[e * nf + a];
                if cia == 0.0 {
                    continue;
                }
                dq[e] -= lambda[a] * cia * fixed_sum;
                for b in 0..nf {
                    let cib = c_free[e * nf + b];
                    if cib == 0.0 {
                        continue;
                    }
                    dq[e] += lambda[a] * cia * cib * xyz_star[b * 3 + comp];
                }
            }
        }
    }
}

/// Add `∂L/∂X_a` from adjoint `λ` (`P = L_f − C_fᵀ diag(q) C_a X_a`).
pub fn accumulate_dxf_from_lambda(
    dxf: &mut [f64],
    lambda_xyz: &[f64],
    q: &[f64],
    structure: &Structure,
) {
    let nf = structure.num_free();
    let na = structure.num_fixed();
    let ne = structure.num_edges;
    let c_free = structure.connectivity_free();
    let c_fixed = structure.connectivity_fixed();
    for comp in 0..3 {
        let lambda: Vec<f64> = (0..nf).map(|a| lambda_xyz[a * 3 + comp]).collect();
        for e in 0..ne {
            for j in 0..na {
                let caj = c_fixed[e * na + j];
                if caj == 0.0 {
                    continue;
                }
                for a in 0..nf {
                    let cia = c_free[e * nf + a];
                    if cia == 0.0 {
                        continue;
                    }
                    dxf[j * 3 + comp] += lambda[a] * q[e] * cia * caj;
                }
            }
        }
    }
}

/// `dL/dq` through nonlinear fixed-point equilibrium (edge follower loads).
///
/// Backpropagates through `x_{k+1} = K⁻¹ P(x_k)` with analytic `∂P/∂x` for edge
/// and global face loads.
pub fn grad_loss_wrt_q_fixedpoint(
    q: &[f64],
    xyz_fixed: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    xyz_anchor: &[f64],
    config: &IterativeConfig,
    mesh: Option<&MeshStructure>,
    loss_grad_xyz_free: &[f64],
) -> Result<QGradient, FdmError> {
    let has_edge = load_state.edges.as_ref().is_some_and(|e| !e.is_empty());
    let has_face = load_state.faces.as_ref().is_some_and(|f| !f.is_empty()) && mesh.is_some();
    if !has_edge && !has_face {
        if config.tmax <= 1 {
            return grad_loss_wrt_q_linear(
                q,
                xyz_fixed,
                &load_state.nodes,
                structure,
                &equilibrium_iterative(
                    q, xyz_fixed, load_state, structure, edges, xyz_anchor, config, mesh,
                )?,
                loss_grad_xyz_free,
            );
        }
        return grad_loss_wrt_q_fd(
            q,
            xyz_fixed,
            load_state,
            structure,
            edges,
            xyz_anchor,
            config,
            mesh,
            loss_grad_xyz_free,
            1e-7,
        );
    }

    let traj_cfg = config_for_implicit_adjoint(config);
    let traj = equilibrium_iterative_trajectory(
        q, xyz_fixed, load_state, structure, edges, xyz_anchor, &traj_cfg, mesh,
    )?;
    let adjoint = AdjointSolveConfig::from(config);
    let ne = structure.num_edges;
    let mut dq = vec![0.0; ne];
    let mut v = loss_grad_xyz_free.to_vec();

    for t in (0..traj.len()).rev() {
        let x_t = &traj[t];
        let lambda_xyz = solve_adjoint_columns(q, structure, &v, &adjoint)?;
        accumulate_dq_from_lambda(&mut dq, &lambda_xyz, q, xyz_fixed, structure, x_t);
        if t == 0 {
            break;
        }
        let mut xyz_full = xyz_anchor.to_vec();
        merge_free_into_full(&mut xyz_full, x_t, structure, xyz_fixed);
        let mut v_load = vec![0.0; v.len()];
        if let Some(edge_intensity) = load_state.edges.as_ref() {
            if !edge_intensity.is_empty() {
                v_load = transpose_edge_loads_jacobian(
                    &xyz_full,
                    edge_intensity,
                    structure,
                    edges,
                    &lambda_xyz,
                );
            }
        }
        if let (Some(mesh), Some(face_loads)) = (mesh, load_state.faces.as_ref()) {
            if !face_loads.is_empty() {
                let vf = transpose_face_loads_jacobian(
                    &xyz_full,
                    face_loads,
                    mesh,
                    structure,
                    edges,
                    &lambda_xyz,
                    load_state.faces_load_local,
                );
                for (a, b) in v_load.iter_mut().zip(vf.iter()) {
                    *a += *b;
                }
            }
        }
        v = v_load;
    }

    Ok(QGradient { dq })
}

fn merge_free_into_full(
    xyz_full: &mut [f64],
    xyz_free: &[f64],
    structure: &Structure,
    xyz_fixed: &[f64],
) {
    let pos = EquilibriumModel::nodes_positions(xyz_free, xyz_fixed, structure);
    xyz_full.copy_from_slice(&pos);
}

/// `dL/dq` through equilibrium: analytic when linear + constant loads, else FD.
pub fn grad_loss_wrt_q(
    q: &[f64],
    xyz_fixed: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    xyz_anchor: &[f64],
    config: &IterativeConfig,
    mesh: Option<&MeshStructure>,
    xyz_star: &[f64],
    loss_grad_xyz_free: &[f64],
    eps: f64,
) -> Result<QGradient, FdmError> {
    let adj_cfg = config_for_implicit_adjoint(config);
    if adj_cfg.tmax <= 1 && !load_state.has_shape_dependent() {
        return grad_loss_wrt_q_linear_with_solver(
            q,
            xyz_fixed,
            structure,
            xyz_star,
            loss_grad_xyz_free,
            &AdjointSolveConfig::from(config),
        );
    }
    if load_state.has_shape_dependent() || adj_cfg.tmax > 1 {
        return grad_loss_wrt_q_fixedpoint(
            q,
            xyz_fixed,
            load_state,
            structure,
            edges,
            xyz_anchor,
            config,
            mesh,
            loss_grad_xyz_free,
        );
    }
    grad_loss_wrt_q_fd(
        q,
        xyz_fixed,
        load_state,
        structure,
        edges,
        xyz_anchor,
        config,
        mesh,
        loss_grad_xyz_free,
        eps,
    )
}

/// Central-difference `dL/dq` (nonlinear / shape-dependent loads).
pub fn grad_loss_wrt_q_fd(
    q: &[f64],
    xyz_fixed: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    xyz_anchor: &[f64],
    config: &IterativeConfig,
    mesh: Option<&MeshStructure>,
    loss_grad_xyz_free: &[f64],
    eps: f64,
) -> Result<QGradient, FdmError> {
    let mut dq = vec![0.0; q.len()];
    for i in 0..q.len() {
        let mut qp = q.to_vec();
        let mut qm = q.to_vec();
        qp[i] += eps;
        qm[i] -= eps;
        let xp = solve_equilibrium(
            &qp, xyz_fixed, load_state, structure, edges, xyz_anchor, config, mesh,
        )?;
        let xm = solve_equilibrium(
            &qm, xyz_fixed, load_state, structure, edges, xyz_anchor, config, mesh,
        )?;
        let mut g = 0.0;
        for j in 0..loss_grad_xyz_free.len() {
            g += loss_grad_xyz_free[j] * (xp[j] - xm[j]) / (2.0 * eps);
        }
        dq[i] = g;
    }
    Ok(QGradient { dq })
}

fn solve_equilibrium(
    q: &[f64],
    xyz_fixed: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    xyz_anchor: &[f64],
    config: &IterativeConfig,
    mesh: Option<&MeshStructure>,
) -> Result<Vec<f64>, FdmError> {
    let adj_cfg = config_for_implicit_adjoint(config);
    if adj_cfg.tmax <= 1 && !load_state.has_shape_dependent() {
        let loads = nodes_load_at_mesh(xyz_anchor, load_state, structure, edges, mesh);
        return crate::sparse::nodes_free_positions_auto(
            q,
            xyz_fixed,
            &loads,
            structure,
            adj_cfg.use_sparse,
            adj_cfg.pcg_max_iter,
            adj_cfg.pcg_tol,
        );
    }
    equilibrium_iterative(
        q, xyz_fixed, load_state, structure, edges, xyz_anchor, &adj_cfg, mesh,
    )
}
