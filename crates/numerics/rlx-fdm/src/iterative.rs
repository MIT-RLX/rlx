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

//! Nonlinear equilibrium iteration (jax_fdm `solver_forward` / `equilibrium_iterative_xyz`).

use crate::equilibrium::{EquilibriumModel, FdmError};
use crate::loads::{LoadState, nodes_load_at_mesh};
use crate::mesh::MeshStructure;
use crate::sparse::{SparseStiffnessFast, nodes_free_positions_auto};
use crate::structure::Structure;

/// Fixed-point iteration controls (`tmax`, `eta` in jax_fdm).
#[derive(Clone, Debug)]
pub struct IterativeConfig {
    /// Maximum iterations (jax_fdm default model `tmax=100`; `fdm()` uses `1`).
    pub tmax: u32,
    /// Mean coordinate change tolerance per step.
    pub eta: f64,
    pub use_sparse: bool,
    pub pcg_max_iter: u32,
    pub pcg_tol: f64,
    /// Anderson acceleration depth (`0` = plain fixed-point; jax_fdm-style `anderson`).
    pub anderson_depth: u32,
}

impl Default for IterativeConfig {
    fn default() -> Self {
        Self {
            tmax: 100,
            eta: 1e-6,
            use_sparse: false,
            pcg_max_iter: 4000,
            pcg_tol: 1e-10,
            anderson_depth: 0,
        }
    }
}

impl IterativeConfig {
    pub fn linear() -> Self {
        Self {
            tmax: 1,
            ..Default::default()
        }
    }
}

fn mean_step_norm(prev: &[f64], curr: &[f64]) -> f64 {
    let n = prev.len() / 3;
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let dx = curr[i * 3] - prev[i * 3];
        let dy = curr[i * 3 + 1] - prev[i * 3 + 1];
        let dz = curr[i * 3 + 2] - prev[i * 3 + 2];
        sum += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    sum / n as f64
}

/// Config for implicit adjoint: full `tmax` unroll (no early `eta` stop).
pub fn config_for_implicit_adjoint(config: &IterativeConfig) -> IterativeConfig {
    let mut c = config.clone();
    c.eta = 0.0;
    c
}

/// Anderson acceleration: `x_{k+1} = g + Σ_j γ_j (x_j − g)` with γ from secant residuals.
fn anderson_mix(x: &[f64], g: &[f64], x_hist: &[Vec<f64>], f_hist: &[Vec<f64>]) -> Vec<f64> {
    let n = x.len();
    let mut f_cur = vec![0.0; n];
    for i in 0..n {
        f_cur[i] = g[i] - x[i];
    }
    let m = f_hist.len();
    if m == 0 {
        return g.to_vec();
    }
    let mut df: Vec<Vec<f64>> = Vec::with_capacity(m);
    for j in 0..m {
        let mut col = vec![0.0; n];
        for i in 0..n {
            col[i] = f_hist[j][i] - f_cur[i];
        }
        df.push(col);
    }
    let mut ata = vec![0.0; m * m];
    let mut atb = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            atb[j] -= df[j][i] * f_cur[i];
        }
        for k in 0..m {
            let mut s = 0.0;
            for i in 0..n {
                s += df[j][i] * df[k][i];
            }
            ata[j * m + k] = s;
        }
    }
    let gamma = solve_gamma_small(&ata, &atb, m);
    if gamma.iter().any(|v| !v.is_finite()) {
        return g.to_vec();
    }
    let mut out = g.to_vec();
    for j in 0..m {
        let gj = gamma[j];
        for i in 0..n {
            out[i] += gj * (x_hist[j][i] - g[i]);
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return g.to_vec();
    }
    out
}

fn solve_gamma_small(ata: &[f64], atb: &[f64], m: usize) -> Vec<f64> {
    let mut a = ata.to_vec();
    let mut b = atb.to_vec();
    for k in 0..m {
        let mut piv = k;
        let mut best = a[k * m + k].abs();
        for i in k + 1..m {
            let v = a[i * m + k].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-14 {
            return vec![0.0; m];
        }
        if piv != k {
            for j in 0..m {
                a.swap(k * m + j, piv * m + j);
            }
            b.swap(k, piv);
        }
        for i in k + 1..m {
            let factor = a[i * m + k] / a[k * m + k];
            for j in k..m {
                a[i * m + j] -= factor * a[k * m + j];
            }
            b[i] -= factor * b[k];
        }
    }
    for k in (0..m).rev() {
        for j in k + 1..m {
            b[k] -= a[k * m + j] * b[j];
        }
        let diag = a[k * m + k];
        if diag.abs() < 1e-14 {
            return vec![0.0; m];
        }
        b[k] /= diag;
    }
    b
}

fn push_anderson_history(
    x_hist: &mut Vec<Vec<f64>>,
    f_hist: &mut Vec<Vec<f64>>,
    x: &[f64],
    g: &[f64],
    depth: usize,
) {
    let n = x.len();
    let mut f = vec![0.0; n];
    for i in 0..n {
        f[i] = g[i] - x[i];
    }
    x_hist.push(x.to_vec());
    f_hist.push(f);
    while x_hist.len() > depth {
        x_hist.remove(0);
        f_hist.remove(0);
    }
}

/// Fixed-point solve `x_{k+1} = K⁻¹ P(x_k)` (jax_fdm `equilibrium_iterative_xyz` + `solver_forward`).
pub fn equilibrium_iterative(
    q: &[f64],
    xyz_fixed: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    xyz_anchor: &[f64],
    config: &IterativeConfig,
    mesh: Option<&MeshStructure>,
) -> Result<Vec<f64>, FdmError> {
    let nf = structure.num_free();
    let sparse_pat = if config.use_sparse && nf >= 32 {
        Some(SparseStiffnessFast::pattern(structure))
    } else {
        None
    };

    let mut xyz_full = xyz_anchor.to_vec();
    let mut xyz_free = initial_free_guess(
        q,
        xyz_fixed,
        load_state,
        structure,
        edges,
        xyz_anchor,
        config,
        sparse_pat.as_ref(),
        mesh,
    )?;

    merge_free_into_full(&mut xyz_full, &xyz_free, structure, xyz_fixed);

    if config.tmax <= 1 && !load_state.has_shape_dependent() {
        return Ok(xyz_free);
    }

    let depth = config.anderson_depth as usize;
    let mut x_hist: Vec<Vec<f64>> = Vec::new();
    let mut f_hist: Vec<Vec<f64>> = Vec::new();

    for _ in 0..config.tmax.saturating_sub(1) {
        let prev = xyz_free.clone();
        let loads = nodes_load_at_mesh(&xyz_full, load_state, structure, edges, mesh);
        let g_step = solve_free_step(q, xyz_fixed, &loads, structure, config, sparse_pat.as_ref())?;
        xyz_free = if depth == 0 {
            g_step
        } else {
            let next = anderson_mix(&prev, &g_step, &x_hist, &f_hist);
            push_anderson_history(&mut x_hist, &mut f_hist, &prev, &g_step, depth);
            next
        };
        merge_free_into_full(&mut xyz_full, &xyz_free, structure, xyz_fixed);
        if config.eta > 0.0 && mean_step_norm(&prev, &xyz_free) <= config.eta {
            break;
        }
    }
    Ok(xyz_free)
}

/// Free-coordinate trajectory for backward mode (`solver_fixedpoint_implicit`).
///
/// Returns `x_0, …, x_T` where `x_T` is the converged state (same as
/// [`equilibrium_iterative`]).
pub fn equilibrium_iterative_trajectory(
    q: &[f64],
    xyz_fixed: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    xyz_anchor: &[f64],
    config: &IterativeConfig,
    mesh: Option<&MeshStructure>,
) -> Result<Vec<Vec<f64>>, FdmError> {
    let nf = structure.num_free();
    let sparse_pat = if config.use_sparse && nf >= 32 {
        Some(SparseStiffnessFast::pattern(structure))
    } else {
        None
    };

    let mut xyz_full = xyz_anchor.to_vec();
    let mut xyz_free = initial_free_guess(
        q,
        xyz_fixed,
        load_state,
        structure,
        edges,
        xyz_anchor,
        config,
        sparse_pat.as_ref(),
        mesh,
    )?;
    merge_free_into_full(&mut xyz_full, &xyz_free, structure, xyz_fixed);

    let mut traj = vec![xyz_free.clone()];
    if config.tmax <= 1 && !load_state.has_shape_dependent() {
        return Ok(traj);
    }

    let adj_cfg = config_for_implicit_adjoint(config);
    let depth = adj_cfg.anderson_depth as usize;
    let mut x_hist: Vec<Vec<f64>> = Vec::new();
    let mut f_hist: Vec<Vec<f64>> = Vec::new();

    for _ in 0..adj_cfg.tmax.saturating_sub(1) {
        let prev = xyz_free.clone();
        let loads = nodes_load_at_mesh(&xyz_full, load_state, structure, edges, mesh);
        let g_step = solve_free_step(
            q,
            xyz_fixed,
            &loads,
            structure,
            &adj_cfg,
            sparse_pat.as_ref(),
        )?;
        xyz_free = if depth == 0 {
            g_step
        } else {
            let next = anderson_mix(&prev, &g_step, &x_hist, &f_hist);
            push_anderson_history(&mut x_hist, &mut f_hist, &prev, &g_step, depth);
            next
        };
        merge_free_into_full(&mut xyz_full, &xyz_free, structure, xyz_fixed);
        traj.push(xyz_free.clone());
    }
    Ok(traj)
}

fn initial_free_guess(
    q: &[f64],
    xyz_fixed: &[f64],
    load_state: &LoadState,
    structure: &Structure,
    edges: &[(usize, usize)],
    xyz_anchor: &[f64],
    config: &IterativeConfig,
    sparse_pat: Option<&SparseStiffnessFast>,
    mesh: Option<&MeshStructure>,
) -> Result<Vec<f64>, FdmError> {
    let loads = nodes_load_at_mesh(xyz_anchor, load_state, structure, edges, mesh);
    solve_free_step(q, xyz_fixed, &loads, structure, config, sparse_pat)
}

fn solve_free_step(
    q: &[f64],
    xyz_fixed: &[f64],
    loads_nodes: &[f64],
    structure: &Structure,
    config: &IterativeConfig,
    sparse_pat: Option<&SparseStiffnessFast>,
) -> Result<Vec<f64>, FdmError> {
    if let Some(pat) = sparse_pat {
        return crate::sparse::nodes_free_positions_sparse_fast(
            pat,
            q,
            xyz_fixed,
            loads_nodes,
            structure,
            config.pcg_max_iter,
            config.pcg_tol,
        );
    }
    nodes_free_positions_auto(
        q,
        xyz_fixed,
        loads_nodes,
        structure,
        config.use_sparse,
        config.pcg_max_iter,
        config.pcg_tol,
    )
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
