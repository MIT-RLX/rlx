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

//! Fast CSR assembly via precomputed index maps (jax_fdm `EquilibriumStructureSparse`).

use crate::equilibrium::FdmError;
use crate::solve::pcg_solve;
use crate::structure::Structure;

#[derive(Clone, Debug)]
pub struct NzTerm {
    pub edge: usize,
    pub coef: f64,
}

/// CSR stiffness with O(nnz) assembly (jax_fdm `index_array` + `diag_indices`).
#[derive(Clone, Debug)]
pub struct SparseStiffnessFast {
    pub n: usize,
    pub row_ptr: Vec<i32>,
    pub col_idx: Vec<i32>,
    /// Per-nnz contributions `Σ qi * coef`.
    pub nz_terms: Vec<Vec<NzTerm>>,
    /// Edges incident to each free node (for diagonal).
    pub diag_edges: Vec<Vec<usize>>,
}

impl SparseStiffnessFast {
    pub fn pattern(structure: &Structure) -> Self {
        let nf = structure.num_free();
        let ne = structure.num_edges;
        let c_free = structure.connectivity_free();

        let mut accum: Vec<std::collections::HashMap<usize, Vec<NzTerm>>> =
            (0..nf).map(|_| std::collections::HashMap::new()).collect();

        for e in 0..ne {
            for a in 0..nf {
                let cia = c_free[e * nf + a];
                if cia == 0.0 {
                    continue;
                }
                for b in 0..nf {
                    let cib = c_free[e * nf + b];
                    if cib == 0.0 {
                        continue;
                    }
                    let coef = cia * cib;
                    accum[a]
                        .entry(b)
                        .or_default()
                        .push(NzTerm { edge: e, coef });
                }
            }
        }

        let mut diag_edges = vec![Vec::new(); nf];
        for e in 0..ne {
            for a in 0..nf {
                if c_free[e * nf + a] != 0.0 {
                    diag_edges[a].push(e);
                }
            }
        }

        let mut row_ptr = vec![0i32; nf + 1];
        let mut col_idx = Vec::new();
        let mut nz_terms = Vec::new();
        for a in 0..nf {
            let mut cols: Vec<_> = accum[a].drain().collect();
            cols.sort_by_key(|(c, _)| *c);
            for (b, terms) in cols {
                col_idx.push(b as i32);
                nz_terms.push(terms);
            }
            row_ptr[a + 1] = col_idx.len() as i32;
        }

        Self {
            n: nf,
            row_ptr,
            col_idx,
            nz_terms,
            diag_edges,
        }
    }

    pub fn assemble(&self, q: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.col_idx.len()];
        for (k, terms) in self.nz_terms.iter().enumerate() {
            let mut v = 0.0;
            for t in terms {
                v += -q[t.edge] * t.coef;
            }
            values[k] = v;
        }
        for (row, edges) in self.diag_edges.iter().enumerate() {
            let mut d = 0.0;
            for &e in edges {
                d += -q[e];
            }
            let rs = self.row_ptr[row] as usize;
            let re = self.row_ptr[row + 1] as usize;
            for k in rs..re {
                if self.col_idx[k] as usize == row {
                    values[k] = d;
                    break;
                }
            }
        }
        values
    }

    pub fn solve_vec(
        &self,
        q: &[f64],
        b: &[f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<Vec<f64>, FdmError> {
        let values = self.assemble(q);
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

    pub fn solve_xyz(
        &self,
        q: &[f64],
        p: &[f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<Vec<f64>, FdmError> {
        let nf = self.n;
        let mut out = vec![0.0; nf * 3];
        for comp in 0..3 {
            let rhs: Vec<f64> = (0..nf).map(|a| p[a * 3 + comp]).collect();
            let x = self.solve_vec(q, &rhs, max_iter, tol)?;
            for (a, &xi) in x.iter().enumerate() {
                out[a * 3 + comp] = xi;
            }
        }
        Ok(out)
    }
}
