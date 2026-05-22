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

//! CSR stiffness for large free-DOF sets (jax_fdm `EquilibriumModelSparse`).

use crate::equilibrium::{EquilibriumModel, FdmError};
use crate::solve::pcg_solve;
pub use crate::sparse_fast::SparseStiffnessFast;
use crate::structure::Structure;

/// Fixed sparsity pattern of `K = C_fᵀ diag(q) C_f` on free nodes.
#[derive(Clone, Debug)]
pub struct SparseStiffness {
    pub n: usize,
    pub row_ptr: Vec<i32>,
    pub col_idx: Vec<i32>,
}

impl SparseStiffness {
    /// Build CSR pattern from topology (independent of `q`).
    pub fn pattern(structure: &Structure) -> Self {
        let nf = structure.num_free();
        let ne = structure.num_edges;
        let c_free = structure.connectivity_free();
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nf];
        for i in 0..ne {
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
                    rows[a].push((b, cia * cib));
                }
            }
        }
        let mut row_ptr = vec![0i32; nf + 1];
        let mut col_idx = Vec::new();
        let mut values_template = Vec::new();
        for r in 0..nf {
            rows[r].sort_by_key(|&(c, _)| c);
            rows[r].dedup_by_key(|(c, _)| *c);
            for (c, coef) in &rows[r] {
                col_idx.push(*c as i32);
                values_template.push(*coef);
            }
            row_ptr[r + 1] = col_idx.len() as i32;
        }
        let _ = values_template;
        Self {
            n: nf,
            row_ptr,
            col_idx,
        }
    }

    /// Fill numeric values for signed force densities (`qi = -q[i]` in equilibrium).
    pub fn assemble(&self, q: &[f64], structure: &Structure) -> Vec<f64> {
        let nf = self.n;
        let ne = structure.num_edges;
        let c_free = structure.connectivity_free();
        let mut values = vec![0.0; self.col_idx.len()];
        for i in 0..ne {
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
                    let v = qi * cia * cib;
                    let row_start = self.row_ptr[a] as usize;
                    let row_end = self.row_ptr[a + 1] as usize;
                    for k in row_start..row_end {
                        if self.col_idx[k] as usize == b {
                            values[k] += v;
                            break;
                        }
                    }
                }
            }
        }
        values
    }

    /// `K x = b` with PCG (one RHS).
    pub fn solve_vec(
        &self,
        q: &[f64],
        structure: &Structure,
        b: &[f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<Vec<f64>, FdmError> {
        let values = self.assemble(q, structure);
        let mut x = vec![0.0; self.n];
        pcg_solve(
            &values,
            &self.col_idx,
            &self.row_ptr,
            b,
            &mut x,
            max_iter,
            tol,
        )?;
        Ok(x)
    }

    /// Three coordinate RHS columns (x, y, z).
    pub fn solve_xyz(
        &self,
        q: &[f64],
        structure: &Structure,
        p: &[f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<Vec<f64>, FdmError> {
        let nf = self.n;
        if p.len() != nf * 3 {
            return Err(FdmError::Dimension(format!("P len {} != {}*3", p.len(), nf)));
        }
        let mut out = vec![0.0; nf * 3];
        for comp in 0..3 {
            let rhs: Vec<f64> = (0..nf).map(|a| p[a * 3 + comp]).collect();
            let x = self.solve_vec(q, structure, &rhs, max_iter, tol)?;
            for (a, &xi) in x.iter().enumerate() {
                out[a * 3 + comp] = xi;
            }
        }
        Ok(out)
    }
}

/// Sparse free-node positions (jax_fdm `EquilibriumModelSparse` + `spsolve`).
pub fn nodes_free_positions_sparse(
    pattern: &SparseStiffness,
    q: &[f64],
    xyz_fixed: &[f64],
    loads_nodes: &[f64],
    structure: &Structure,
    pcg_max_iter: u32,
    pcg_tol: f64,
) -> Result<Vec<f64>, FdmError> {
    let p = EquilibriumModel::load_matrix(q, xyz_fixed, loads_nodes, structure);
    pattern.solve_xyz(q, structure, &p, pcg_max_iter, pcg_tol)
}

/// Dense/sparse parity helper for tests.
pub fn max_abs_diff_dense_sparse(
    q: &[f64],
    xyz_fixed: &[f64],
    loads_nodes: &[f64],
    structure: &Structure,
) -> Result<f64, FdmError> {
    let dense = EquilibriumModel::nodes_free_positions(q, xyz_fixed, loads_nodes, structure)?;
    let pat = SparseStiffness::pattern(structure);
    let sparse = nodes_free_positions_sparse(&pat, q, xyz_fixed, loads_nodes, structure, 4000, 1e-10)?;
    let mut m: f64 = 0.0;
    for (a, b) in dense.iter().zip(sparse.iter()) {
        m = m.max((a - b).abs());
    }
    Ok(m)
}

/// Fast-pattern sparse solve (jax_fdm `EquilibriumStructureSparse`).
pub fn nodes_free_positions_sparse_fast(
    pattern: &SparseStiffnessFast,
    q: &[f64],
    xyz_fixed: &[f64],
    loads_nodes: &[f64],
    structure: &Structure,
    pcg_max_iter: u32,
    pcg_tol: f64,
) -> Result<Vec<f64>, FdmError> {
    let p = EquilibriumModel::load_matrix(q, xyz_fixed, loads_nodes, structure);
    pattern.solve_xyz(q, &p, pcg_max_iter, pcg_tol)
}

/// Export CSR for [`crate::rlx_op::pcg_solve_graph`] (`feature rlx-sparse`).
pub fn export_csr(
    pattern: &crate::sparse_fast::SparseStiffnessFast,
    q: &[f64],
) -> (Vec<f64>, Vec<i32>, Vec<i32>, usize) {
    (
        pattern.assemble(q),
        pattern.col_idx.clone(),
        pattern.row_ptr.clone(),
        pattern.n,
    )
}

/// Fall back to dense when `n` is tiny (cheaper than PCG setup).
pub fn nodes_free_positions_auto(
    q: &[f64],
    xyz_fixed: &[f64],
    loads_nodes: &[f64],
    structure: &Structure,
    use_sparse: bool,
    pcg_max_iter: u32,
    pcg_tol: f64,
) -> Result<Vec<f64>, FdmError> {
    let nf = structure.num_free();
    if !use_sparse || nf < 32 {
        EquilibriumModel::nodes_free_positions(q, xyz_fixed, loads_nodes, structure)
    } else {
        let pat = SparseStiffnessFast::pattern(structure);
        nodes_free_positions_sparse_fast(&pat, q, xyz_fixed, loads_nodes, structure, pcg_max_iter, pcg_tol)
    }
}

/// Build [`SparseStiffnessFast`] pattern (preferred for repeated solves).
pub fn pattern_fast(structure: &Structure) -> SparseStiffnessFast {
    SparseStiffnessFast::pattern(structure)
}
