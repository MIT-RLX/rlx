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

//! Sparse linear algebra for RLX — CSR LU, mat-vec, Conjugate Gradient.
//!
//! Downstream package modeled on `jax.experimental.sparse`. Registers
//! against rlx's custom-op scaffold without requiring any edits to the
//! framework crates. Three ops + a `SparseTensor` boundary abstraction.
//!
//! ## Usage
//!
//! ```ignore
//! // At application startup, once.
//! rlx_sparse::register();
//!
//! // Build graph as usual.
//! let mut g = Graph::new("photonics");
//! let v  = g.input("values",  Shape::new(&[nnz], DType::F64));
//! let ci = ...; // I32 col_idx (Op::Constant or Op::Input)
//! let rp = ...; // I32 row_ptr
//! let b  = g.input("b", Shape::new(&[n], DType::F64));
//!
//! let a = rlx_sparse::SparseTensor::from_csr(v, ci, rp, n, n);
//! let x = a.solve(&mut g, b);                 // direct LU
//! let y = a.mat_vec(&mut g, x);               // sparse matvec
//! let z = a.cg_solve(&mut g, b, 200, 1e-12);  // iterative CG
//! ```
//!
//! ## What's registered
//!
//! - `rlx_sparse.lu_solve` — direct LU via host LAPACK dgesv.
//!   v1 densifies CSR before solving; performance not yet sparse-fast,
//!   semantics are correct. Swapping for SuiteSparse UMFPACK or KLU is
//!   a kernel-body change with zero IR diff.
//! - `rlx_sparse.mat_vec` — `y = A·x` over CSR.
//! - `rlx_sparse.cg_solve` — Conjugate Gradient for SPD systems with
//!   `max_iter` + `tol` baked into the op's `attrs` blob.
//!
//! ## Adjoint convention (v1)
//!
//! All three ops assume `A` is symmetric. The closed-form adjoint
//! `dL/db = solve(Aᵀ, dL/dx)` reuses the same CSR triplet as the
//! forward call. Non-symmetric `A` requires an explicit transpose
//! triplet — sketch in the `vjp` body of each op. `dL/dvalues` is
//! non-differentiable in v1; it's `gather(-(dL/db) ⊗ x)` and slots
//! in as a separate gather op.
//!
//! ## Backend support
//!
//! | Backend | Status |
//! |---|---|
//! | CPU    | Full forward + autodiff. Real LAPACK. |
//! | Metal  | Trait surface only — full executor dispatch is a follow-up. |
//! | MLX    | Trait surface only — full executor dispatch is a follow-up. |
//! | Others | `Op::Custom` rejected at legalize; pin graph to `Device::Cpu`. |

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]

use std::sync::Arc;

use rlx_ir::{DType, Graph, Node, NodeId, Op, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

pub const SPARSE_LU_SOLVE: &str = "rlx_sparse.lu_solve";
pub const SPARSE_MAT_VEC: &str = "rlx_sparse.mat_vec";
pub const SPARSE_CG_SOLVE: &str = "rlx_sparse.cg_solve";

/// Outer-product gather op (the `dL/dvalues` building block).
/// Computes `out[k] = u[row_of(k)] * v[col_idx[k]]` for each non-zero
/// position `k` in the CSR pattern. Used by `SparseLu`/`SparseMatVec`/
/// `SparseCg`/`SparseGmres` VJPs to gather the dense outer-product
/// `u ⊗ v` at the matrix's nonzero positions.
pub const SPARSE_VALUES_GRAD: &str = "rlx_sparse.values_grad";

/// Non-symmetric LU solve. Forward `x = A⁻¹·b` (uses A only).
/// VJP `dL/db = solve(Aᵀ, dL/dx)` — needs an explicit transpose
/// triplet, supplied as the last 3 inputs to keep the IR self-
/// contained. The 4-input `SPARSE_LU_SOLVE` is the symmetric
/// specialization.
pub const SPARSE_LU_SOLVE_GENERAL: &str = "rlx_sparse.lu_solve_general";

/// GMRES solve for non-symmetric systems. Iterative analog of CG
/// for the asymmetric Maxwell / advection-diffusion regime. Same
/// 7-input shape as `SPARSE_LU_SOLVE_GENERAL`: forward uses A,
/// VJP routes the adjoint through Aᵀ.
pub const SPARSE_GMRES_SOLVE: &str = "rlx_sparse.gmres_solve";

/// Permute a CSR `values` vector into the values vector of `Aᵀ`.
/// 5 inputs: `(values_A, col_idx_A, row_ptr_A, col_idx_AT, row_ptr_AT)`.
/// The transposed pattern (`col_idx_AT`, `row_ptr_AT`) is structural
/// — depends only on the original pattern — so it can be precomputed
/// once at graph-build time via [`csr_transpose_pattern`] and embedded
/// as constants. Only the values get permuted per call. Useful for
/// inverse-design Newton loops where the matrix entries change each
/// iteration but the sparsity pattern is fixed.
pub const SPARSE_TRANSPOSE_VALUES: &str = "rlx_sparse.transpose_values";

/// Jacobi-preconditioned CG. Same 4-input shape as `SPARSE_CG_SOLVE`
/// (values, col_idx, row_ptr, b). The kernel extracts `diag(A)`
/// internally and uses it as the diagonal preconditioner —
/// dramatically faster than plain CG on ill-conditioned circuit
/// matrices where the diagonal magnitudes vary by orders of
/// magnitude. Convergence requires SPD A like CG.
pub const SPARSE_PCG_SOLVE: &str = "rlx_sparse.pcg_solve";

/// BiCGSTAB iterative solver for general (non-symmetric) sparse A·x = b.
/// 4 inputs (values, col_idx, row_ptr, b) + attrs encoding
/// (max_iter: u32, tol: f64, transpose_a: u8). When `transpose_a` is
/// set, the kernel solves Aᵀ·x = b — used by VJPs for adjoint solves.
pub const SPARSE_BICGSTAB_SOLVE: &str = "rlx_sparse.bicgstab_solve";

/// ILU(0)-preconditioned CG. Factors A in-place over its existing
/// sparsity pattern (zero fill-in) and applies the LU triangular
/// solves as the preconditioner. Same 4-input shape as PCG; converges
/// faster than Jacobi-PCG on stiff systems where row-row coupling
/// dominates the off-diagonal.
pub const SPARSE_ILU_PCG_SOLVE: &str = "rlx_sparse.ilu_pcg_solve";

/// Direct sparse Cholesky for SPD A·x = b. Densifies A and uses
/// LAPACK `dpotrf` + triangular solves. Same I/O contract as
/// `SPARSE_LU_SOLVE` but only valid for SPD matrices — ½× factor cost
/// of LU and numerically more stable.
pub const SPARSE_CHOLESKY_SOLVE: &str = "rlx_sparse.cholesky_solve";

/// LSQR for sparse least-squares `min_x ||A·x - b||₂`. 4 inputs
/// (values, col_idx, row_ptr, b) + attrs encoding (max_iter, tol,
/// n_cols). Forward only in v1 — VJP returns empty (least-squares
/// adjoint requires either AᵀA solve or a recursive LSQR call which
/// is non-trivial; defer until a use case appears).
pub const SPARSE_LSQR_SOLVE: &str = "rlx_sparse.lsqr_solve";

/// Sparse-sparse matrix multiply (CSR × CSR → CSR). 6 inputs:
/// (a_values, a_col_idx, a_row_ptr, b_values, b_col_idx, b_row_ptr)
/// plus attrs encoding (k: u32 = inner dim = b's row count). Output
/// is a packed buffer `[c_values | c_col_idx_as_f64 | c_row_ptr_as_f64]`
/// with sizes encoded in attrs alongside k. v1 caps nnz output at
/// `max_nnz` (attrs); allocate generously for known patterns.
pub const SPARSE_SPGEMM: &str = "rlx_sparse.spgemm";

// ── Shared algorithms (CPU + Metal kernels both call these) ──────
//
// Each function takes the typed slices it needs and returns a
// `Result<(), String>`. The CpuKernel impls extract typed slices via
// `CpuTensorRef::expect_*`; the MetalKernel impls extract them by
// casting raw byte slices (after dtype-checking the accompanying
// Shape). Both backends end up calling the same arithmetic.

#[cfg(feature = "cpu")]
mod algos {
    pub fn lu_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let n = b.len();
        if out.len() != n {
            return Err(format!("sparse_lu: output len {} != b len {n}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "sparse_lu: row_ptr len {} != n+1 ({})",
                row_ptr.len(),
                n + 1
            ));
        }
        let mut a_dense = vec![0f64; n * n];
        for r in 0..n {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                a_dense[r * n + col_idx[k] as usize] = values[k];
            }
        }
        let mut b_copy = b.to_vec();
        let info = rlx_cpu::blas::dgesv(&mut a_dense, &mut b_copy, n, 1);
        if info != 0 {
            return Err(format!(
                "sparse_lu: dgesv returned info={info} (>0 → singular)"
            ));
        }
        out.copy_from_slice(&b_copy);
        Ok(())
    }

    pub fn mat_vec(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        x: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let n = x.len();
        if out.len() != n {
            return Err(format!("mat_vec: output len {} != x len {n}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "mat_vec: row_ptr len {} != n+1 ({})",
                row_ptr.len(),
                n + 1
            ));
        }
        for r in 0..n {
            let mut acc = 0f64;
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                acc += values[k] * x[col_idx[k] as usize];
            }
            out[r] = acc;
        }
        Ok(())
    }

    /// Outer-product gather at CSR non-zero positions:
    ///   `out[k] = u[row_of(k)] * v[col_idx[k]]`
    /// where `row_of(k)` is the row containing the `k`-th non-zero
    /// (looked up by walking row_ptr; cached as a parallel array
    /// for O(nnz) lookup time).
    pub fn values_grad(
        col_idx: &[i32],
        row_ptr: &[i32],
        u: &[f64],
        v: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let n = u.len();
        let nnz = col_idx.len();
        if out.len() != nnz {
            return Err(format!("values_grad: out len {} != nnz {nnz}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "values_grad: row_ptr len {} != n+1 ({})",
                row_ptr.len(),
                n + 1
            ));
        }
        // Build row_of_k by scanning row_ptr.
        let mut row_of_k = vec![0u32; nnz];
        for r in 0..n {
            let s = row_ptr[r] as usize;
            let e = row_ptr[r + 1] as usize;
            for k in s..e {
                row_of_k[k] = r as u32;
            }
        }
        for k in 0..nnz {
            let r = row_of_k[k] as usize;
            let c = col_idx[k] as usize;
            if r >= n || c >= v.len() {
                return Err(format!(
                    "values_grad: k={k} (row={r}, col={c}) out of bounds"
                ));
            }
            out[k] = u[r] * v[c];
        }
        Ok(())
    }

    /// GMRES with online Givens-rotation QR on the Hessenberg
    /// system. Standard textbook Saad-Schultz formulation:
    ///
    ///   x = 0; r = b;  β = ||r||;  v_1 = r/β
    ///   for j in 1..=m:
    ///     w = A·v_j; orthogonalize w against v_1..v_j (Modified
    ///     Gram-Schmidt) → h_{i,j} entries; w' = ||w||; v_{j+1} = w/w'
    ///     apply prior Givens rotations to column j of H, generate
    ///     new rotation that zeroes the subdiagonal of column j,
    ///     update transformed RHS β·e_1
    ///     if |residual| < tol: break
    ///   solve upper-triangular system → y; x = Σ_j y_j · v_j
    ///
    /// Restart not implemented — `max_iter` caps Krylov dimension
    /// directly. For ill-conditioned systems set `max_iter` to a
    /// moderate value and re-solve with restart in the application.
    pub fn gmres_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<(), String> {
        let n = b.len();
        if out.len() != n {
            return Err(format!("gmres_solve: out len {} != n {n}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "gmres_solve: row_ptr len {} != n+1 ({})",
                row_ptr.len(),
                n + 1
            ));
        }
        let m = max_iter.max(1) as usize;

        let matvec = |x: &[f64], y: &mut [f64]| {
            for r in 0..n {
                let mut acc = 0f64;
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    acc += values[k] * x[col_idx[k] as usize];
                }
                y[r] = acc;
            }
        };

        // x_0 = 0; r_0 = b; β = ||b||
        let beta_init = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        if beta_init < tol {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            return Ok(());
        }

        // Krylov basis: v_1..v_{m+1}, each length n.
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(b.iter().map(|x| x / beta_init).collect());

        // Hessenberg matrix H is (m+1)×m; stored as rows-vector for
        // online QR via Givens. We track c_j, s_j (Givens cos/sin)
        // and the transformed RHS g (length m+1, initially β·e_1).
        let mut h: Vec<Vec<f64>> = Vec::with_capacity(m); // h[j] = column j (length j+2)
        let mut cs: Vec<f64> = Vec::with_capacity(m);
        let mut sn: Vec<f64> = Vec::with_capacity(m);
        let mut g: Vec<f64> = vec![0.0; m + 1];
        g[0] = beta_init;

        let mut converged_at: Option<usize> = None;
        let mut w = vec![0f64; n];

        for j in 0..m {
            matvec(&v[j], &mut w);
            // MGS orthogonalization: build column j of H (entries 0..=j+1).
            let mut hcol = vec![0f64; j + 2];
            for i in 0..=j {
                hcol[i] = w.iter().zip(&v[i]).map(|(a, b)| a * b).sum();
                for k in 0..n {
                    w[k] -= hcol[i] * v[i][k];
                }
            }
            hcol[j + 1] = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            // Apply previous Givens rotations to column j.
            for i in 0..j {
                let temp = cs[i] * hcol[i] + sn[i] * hcol[i + 1];
                hcol[i + 1] = -sn[i] * hcol[i] + cs[i] * hcol[i + 1];
                hcol[i] = temp;
            }
            // Generate Givens rotation to zero hcol[j+1].
            let denom = (hcol[j] * hcol[j] + hcol[j + 1] * hcol[j + 1]).sqrt();
            if denom == 0.0 {
                return Err("gmres_solve: breakdown (denom = 0)".into());
            }
            let c = hcol[j] / denom;
            let s = hcol[j + 1] / denom;
            cs.push(c);
            sn.push(s);
            hcol[j] = c * hcol[j] + s * hcol[j + 1];
            hcol[j + 1] = 0.0;
            // Apply rotation to RHS g.
            let g_temp = c * g[j] + s * g[j + 1];
            g[j + 1] = -s * g[j] + c * g[j + 1];
            g[j] = g_temp;
            h.push(hcol);

            // Convergence: |g[j+1]| is the residual norm.
            if g[j + 1].abs() < tol {
                converged_at = Some(j);
                break;
            }
            if hcol_last_zero_check(&h[j]) {
                // Lucky breakdown: h_{j+1,j} was already 0 → exact solution.
                converged_at = Some(j);
                break;
            }
            if j + 1 < m {
                let inv = 1.0 / hcol_subdiag(&h[j], j + 1).max(f64::MIN_POSITIVE);
                let _ = inv;
                // Build v_{j+1} = w / h[j+1,j]_pre_rotation. Since we
                // overwrote h[j+1,j] with 0 above, recompute the
                // norm we used.
                let norm_w = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_w < f64::MIN_POSITIVE * 64.0 {
                    converged_at = Some(j);
                    break;
                }
                v.push(w.iter().map(|x| x / norm_w).collect());
            }
        }

        // Solve upper-triangular H (truncated to k×k) for y.
        let k = converged_at.map(|j| j + 1).unwrap_or(m);
        let mut y = vec![0f64; k];
        for i in (0..k).rev() {
            let mut s = g[i];
            for j in (i + 1)..k {
                s -= h[j][i] * y[j];
            }
            y[i] = s / h[i][i];
        }

        // x = Σ y_j · v_j.
        for r in 0..n {
            out[r] = 0.0;
        }
        for j in 0..k {
            for r in 0..n {
                out[r] += y[j] * v[j][r];
            }
        }
        Ok(())
    }

    /// Permute `values_A` into the values vector of `Aᵀ`. The
    /// transposed pattern `(col_idx_t, row_ptr_t)` is assumed
    /// already-computed (depends only on the original pattern; see
    /// [`crate::csr_transpose_pattern`] for the pure-Rust helper).
    pub fn transpose_values(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        _col_idx_t: &[i32],
        row_ptr_t: &[i32],
        out: &mut [f64],
    ) -> Result<(), String> {
        let n = row_ptr.len().saturating_sub(1);
        let nnz = values.len();
        if out.len() != nnz {
            return Err(format!(
                "transpose_values: out len {} != nnz {nnz}",
                out.len()
            ));
        }
        // Cursor into the transposed CSR; starts at each row's
        // row_ptr_t offset and walks forward as we fill.
        let mut cursor: Vec<usize> = row_ptr_t.iter().map(|&x| x as usize).collect();
        for r in 0..n {
            let s = row_ptr[r] as usize;
            let e = row_ptr[r + 1] as usize;
            for k in s..e {
                let c = col_idx[k] as usize;
                let pos = cursor[c];
                if pos >= nnz {
                    return Err(format!(
                        "transpose_values: cursor[{c}]={pos} ≥ nnz={nnz} \
                         (transposed pattern likely inconsistent with input)"
                    ));
                }
                out[pos] = values[k];
                cursor[c] += 1;
            }
        }
        Ok(())
    }

    /// Jacobi-preconditioned CG. Identical to `cg_solve` except each
    /// step applies `M⁻¹` (= 1/diag(A)) to the residual before
    /// search-direction updates. Converges in fewer iterations on
    /// ill-conditioned matrices where `diag(A)` captures most of the
    /// spectrum's magnitude variation (true for circuit MNA matrices
    /// with mixed-magnitude device parameters).
    pub fn pcg_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<(), String> {
        let n = b.len();
        if out.len() != n {
            return Err(format!("pcg_solve: out len {} != n {n}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "pcg_solve: row_ptr len {} != n+1 ({})",
                row_ptr.len(),
                n + 1
            ));
        }

        // Extract diag(A) from CSR — one O(nnz) pass. Missing
        // diagonals (zero or absent entries) get a 1.0 fallback so
        // the preconditioner is well-defined; for SPD A the diagonal
        // is strictly positive so this guard only matters for
        // pathological inputs.
        let mut diag = vec![1.0f64; n];
        for r in 0..n {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                if col_idx[k] as usize == r {
                    diag[r] = values[k].max(f64::MIN_POSITIVE);
                    break;
                }
            }
        }
        let inv_diag: Vec<f64> = diag.iter().map(|&d| 1.0 / d).collect();

        let matvec = |x: &[f64], y: &mut [f64]| {
            for r in 0..n {
                let mut acc = 0f64;
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    acc += values[k] * x[col_idx[k] as usize];
                }
                y[r] = acc;
            }
        };

        // PCG with x0 = 0: r0 = b, z0 = M⁻¹·r0
        let mut x = vec![0f64; n];
        let mut r = b.to_vec();
        let mut z: Vec<f64> = r.iter().zip(&inv_diag).map(|(rv, mi)| rv * mi).collect();
        let mut p = z.clone();
        let mut ap = vec![0f64; n];
        let mut rho_old: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();

        for _ in 0..max_iter {
            // Convergence on plain ‖r‖₂ (matches CG's contract).
            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < tol {
                break;
            }
            matvec(&p, &mut ap);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap == 0.0 {
                return Err("pcg_solve: pᵀ·A·p = 0 (A is singular or not SPD)".into());
            }
            let alpha = rho_old / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }
            for i in 0..n {
                z[i] = r[i] * inv_diag[i];
            }
            let rho_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rho_new / rho_old;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rho_old = rho_new;
        }

        out.copy_from_slice(&x);
        Ok(())
    }

    /// Direct sparse Cholesky for SPD A. Densifies to a dense buffer,
    /// factors via LAPACK `dpotrf`, then forward+back triangular solve
    /// via `dtrsm`. The mirror of [`lu_solve`] for SPD matrices —
    /// faster (factor cost ½× LU) and numerically more stable.
    pub fn cholesky_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        let n = b.len();
        if out.len() != n {
            return Err(format!("cholesky_solve: out len {} != n {n}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "cholesky_solve: row_ptr len {} != n+1",
                row_ptr.len()
            ));
        }
        let mut a_dense = vec![0f64; n * n];
        for r in 0..n {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                a_dense[r * n + col_idx[k] as usize] = values[k];
            }
        }
        // Factor: A = L·Lᵀ; L stored in lower triangle of a_dense.
        let info = rlx_cpu::blas::dpotrf(&mut a_dense, n, /*lower=*/ true);
        if info != 0 {
            return Err(format!("cholesky_solve: dpotrf info={info} (not SPD?)"));
        }
        // Solve L·y = b (forward).
        let mut x = b.to_vec();
        rlx_cpu::blas::dtrsm_lower_or_upper(
            &a_dense, &mut x, n, 1, /*lower=*/ true, /*trans=*/ false,
        );
        // Solve Lᵀ·x = y (back).
        rlx_cpu::blas::dtrsm_lower_or_upper(
            &a_dense, &mut x, n, 1, /*lower=*/ true, /*trans=*/ true,
        );
        out.copy_from_slice(&x);
        Ok(())
    }

    /// BiCGSTAB for general non-symmetric A. `transpose_a` lets a single
    /// op solve either A·x = b or Aᵀ·x = b without materializing the
    /// transpose CSR — used by VJPs for adjoint solves.
    pub fn bicgstab(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
        max_iter: u32,
        tol: f64,
        transpose_a: bool,
    ) -> Result<(), String> {
        let n = b.len();
        if out.len() != n {
            return Err(format!("bicgstab: out len {} != n {n}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "bicgstab: row_ptr len {} != n+1 ({})",
                row_ptr.len(),
                n + 1
            ));
        }
        let matvec = |x: &[f64], y: &mut [f64]| {
            if !transpose_a {
                for r in 0..n {
                    let mut acc = 0f64;
                    for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                        acc += values[k] * x[col_idx[k] as usize];
                    }
                    y[r] = acc;
                }
            } else {
                for v in y.iter_mut() {
                    *v = 0.0;
                }
                for r in 0..n {
                    for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                        y[col_idx[k] as usize] += values[k] * x[r];
                    }
                }
            }
        };
        let mut x = vec![0f64; n];
        let mut r = b.to_vec();
        let r_hat = r.clone();
        let mut p = r.clone();
        let mut v = vec![0f64; n];
        let mut s = vec![0f64; n];
        let mut t = vec![0f64; n];
        let mut rho_old: f64 = r_hat.iter().zip(&r).map(|(a, b)| a * b).sum();

        for _ in 0..max_iter {
            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < tol {
                break;
            }
            matvec(&p, &mut v);
            let rh_v: f64 = r_hat.iter().zip(&v).map(|(a, b)| a * b).sum();
            if rh_v == 0.0 {
                return Err("bicgstab: breakdown r̂·v = 0".into());
            }
            let alpha = rho_old / rh_v;
            for i in 0..n {
                s[i] = r[i] - alpha * v[i];
            }
            let s_norm: f64 = s.iter().map(|v| v * v).sum::<f64>().sqrt();
            if s_norm < tol {
                for i in 0..n {
                    x[i] += alpha * p[i];
                }
                r[..n].copy_from_slice(&s[..n]);
                break;
            }
            matvec(&s, &mut t);
            let tt: f64 = t.iter().map(|v| v * v).sum();
            if tt == 0.0 {
                return Err("bicgstab: breakdown t·t = 0".into());
            }
            let ts: f64 = t.iter().zip(&s).map(|(a, b)| a * b).sum();
            let omega = ts / tt;
            for i in 0..n {
                x[i] += alpha * p[i] + omega * s[i];
                r[i] = s[i] - omega * t[i];
            }
            if omega == 0.0 {
                return Err("bicgstab: ω = 0 (stagnation)".into());
            }
            let rho_new: f64 = r_hat.iter().zip(&r).map(|(a, b)| a * b).sum();
            if rho_old == 0.0 {
                return Err("bicgstab: ρ_old = 0".into());
            }
            let beta = (rho_new / rho_old) * (alpha / omega);
            for i in 0..n {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
            rho_old = rho_new;
        }
        out.copy_from_slice(&x);
        Ok(())
    }

    /// LSQR (Paige-Saunders 1982) for sparse least-squares
    /// `min_x ||A·x - b||₂`. Works for over-determined (m > n) and
    /// under-determined (m < n) systems; for the latter returns the
    /// minimum-norm solution. Numerically stable for ill-conditioned
    /// A — superior to forming the normal equations AᵀA·x = Aᵀ·b
    /// (which squares the condition number).
    ///
    /// Algorithm: Golub-Kahan bidiagonalization with online Givens
    /// rotations on the resulting bidiagonal least-squares problem.
    pub fn lsqr_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
        max_iter: u32,
        tol: f64,
        n_cols: usize,
    ) -> Result<(), String> {
        let m = b.len();
        let n = n_cols;
        if out.len() != n {
            return Err(format!("lsqr: out len {} != n {n}", out.len()));
        }
        if row_ptr.len() != m + 1 {
            return Err(format!(
                "lsqr: row_ptr len {} != m+1 ({})",
                row_ptr.len(),
                m + 1
            ));
        }

        // y = A·x  (gather over rows of A)
        let av = |x: &[f64], y: &mut [f64]| {
            for r in 0..m {
                let mut acc = 0f64;
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    acc += values[k] * x[col_idx[k] as usize];
                }
                y[r] = acc;
            }
        };
        // y = Aᵀ·u  (scatter over rows)
        let atv = |u: &[f64], y: &mut [f64]| {
            for v in y.iter_mut() {
                *v = 0.0;
            }
            for r in 0..m {
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    y[col_idx[k] as usize] += values[k] * u[r];
                }
            }
        };

        let mut u = b.to_vec();
        let mut beta: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        if beta == 0.0 {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            return Ok(());
        }
        for v in u.iter_mut() {
            *v /= beta;
        }

        let mut v = vec![0f64; n];
        atv(&u, &mut v);
        let mut alpha: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if alpha == 0.0 {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            return Ok(());
        }
        for x in v.iter_mut() {
            *x /= alpha;
        }

        let mut x = vec![0f64; n];
        let mut w = v.clone();
        let mut phi_bar = beta;
        let mut rho_bar = alpha;

        let mut tmp_u = vec![0f64; m];
        let mut tmp_v = vec![0f64; n];

        for _ in 0..max_iter {
            // Bidiagonalization step.
            // u_new = A·v - alpha·u; β = ||u_new||
            av(&v, &mut tmp_u);
            for i in 0..m {
                tmp_u[i] -= alpha * u[i];
            }
            beta = tmp_u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if beta != 0.0 {
                for i in 0..m {
                    u[i] = tmp_u[i] / beta;
                }
                // v_new = Aᵀ·u - β·v; α = ||v_new||
                atv(&u, &mut tmp_v);
                for i in 0..n {
                    tmp_v[i] -= beta * v[i];
                }
                alpha = tmp_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if alpha != 0.0 {
                    for i in 0..n {
                        v[i] = tmp_v[i] / alpha;
                    }
                }
            }

            // Givens rotation to eliminate β below ρ̄.
            let rho = (rho_bar * rho_bar + beta * beta).sqrt();
            let c = rho_bar / rho;
            let s = beta / rho;
            let theta = s * alpha;
            rho_bar = -c * alpha;
            let phi = c * phi_bar;
            phi_bar *= s;

            // Update x and w.
            let phi_over_rho = phi / rho;
            let theta_over_rho = theta / rho;
            for i in 0..n {
                x[i] += phi_over_rho * w[i];
                w[i] = v[i] - theta_over_rho * w[i];
            }

            if phi_bar.abs() < tol {
                break;
            }
            if alpha == 0.0 || beta == 0.0 {
                break;
            }
        }
        out.copy_from_slice(&x);
        Ok(())
    }

    /// In-place ILU(0): factor `values` over CSR sparsity pattern.
    /// Returns a new value buffer with L (strict lower) below diag and
    /// U (incl. diag) on/above. L's unit diagonal is implicit.
    pub fn ilu0_factor(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        n: usize,
        out_fact: &mut [f64],
    ) -> Result<(), String> {
        if out_fact.len() != values.len() {
            return Err(format!(
                "ilu0: out len {} != values len {}",
                out_fact.len(),
                values.len()
            ));
        }
        out_fact.copy_from_slice(values);
        for i in 0..n {
            let row_i_start = row_ptr[i] as usize;
            let row_i_end = row_ptr[i + 1] as usize;
            for k in row_i_start..row_i_end {
                let j = col_idx[k] as usize;
                if j >= i {
                    break;
                }
                // Find a[j,j] in row j.
                let row_j_start = row_ptr[j] as usize;
                let row_j_end = row_ptr[j + 1] as usize;
                let mut a_jj = 0f64;
                let mut found = false;
                for kj in row_j_start..row_j_end {
                    if col_idx[kj] as usize == j {
                        a_jj = out_fact[kj];
                        found = true;
                        break;
                    }
                }
                if !found || a_jj == 0.0 {
                    return Err(format!("ilu0: zero/missing diag at row {j}"));
                }
                out_fact[k] /= a_jj;
                let lij = out_fact[k];
                for kk in (k + 1)..row_i_end {
                    let m = col_idx[kk] as usize;
                    for kj in row_j_start..row_j_end {
                        if col_idx[kj] as usize == m {
                            out_fact[kk] -= lij * out_fact[kj];
                            break;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply ILU(0): solve `(L·U)·x = b` on the CSR pattern.
    /// Forward then back triangular sweep over the existing pattern.
    pub fn ilu0_apply(
        fact: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        n: usize,
        b: &[f64],
        out: &mut [f64],
    ) {
        // Forward: L·y = b (unit-diag L). y reuses out.
        for i in 0..n {
            let mut acc = b[i];
            for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                let j = col_idx[k] as usize;
                if j < i {
                    acc -= fact[k] * out[j];
                } else {
                    break;
                }
            }
            out[i] = acc;
        }
        // Back: U·x = y.
        for i in (0..n).rev() {
            let mut acc = out[i];
            let mut diag = 1f64;
            for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                let j = col_idx[k] as usize;
                if j > i {
                    acc -= fact[k] * out[j];
                } else if j == i {
                    diag = fact[k];
                }
            }
            out[i] = acc / diag;
        }
    }

    /// ILU(0)-preconditioned CG. Same convergence contract as PCG but
    /// uses incomplete LU as the preconditioner instead of Jacobi.
    pub fn ilu_pcg_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<(), String> {
        let n = b.len();
        if out.len() != n {
            return Err(format!("ilu_pcg: out len {} != n {n}", out.len()));
        }
        let mut fact = vec![0f64; values.len()];
        ilu0_factor(values, col_idx, row_ptr, n, &mut fact)?;
        let matvec = |x: &[f64], y: &mut [f64]| {
            for r in 0..n {
                let mut acc = 0f64;
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    acc += values[k] * x[col_idx[k] as usize];
                }
                y[r] = acc;
            }
        };
        let mut x = vec![0f64; n];
        let mut r = b.to_vec();
        let mut z = vec![0f64; n];
        ilu0_apply(&fact, col_idx, row_ptr, n, &r, &mut z);
        let mut p = z.clone();
        let mut ap = vec![0f64; n];
        let mut rho_old: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        for _ in 0..max_iter {
            let r_norm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r_norm < tol {
                break;
            }
            matvec(&p, &mut ap);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap == 0.0 {
                return Err("ilu_pcg: pᵀ·A·p = 0".into());
            }
            let alpha = rho_old / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }
            ilu0_apply(&fact, col_idx, row_ptr, n, &r, &mut z);
            let rho_new: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rho_new / rho_old;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rho_old = rho_new;
        }
        out.copy_from_slice(&x);
        Ok(())
    }

    /// CSR × CSR → CSR via Gustavson's algorithm. Two-pass:
    /// (1) symbolic — count nnz per output row using a row-marker;
    /// (2) numeric — accumulate values via a sparse-accumulator (SPA).
    /// Returns (c_values, c_col_idx, c_row_ptr).
    pub fn spgemm_csr(
        a_values: &[f64],
        a_col_idx: &[i32],
        a_row_ptr: &[i32],
        b_values: &[f64],
        b_col_idx: &[i32],
        b_row_ptr: &[i32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<i32>, Vec<i32>), String> {
        if a_row_ptr.len() != m + 1 {
            return Err(format!("spgemm: a_row_ptr len {} != m+1", a_row_ptr.len()));
        }
        if b_row_ptr.len() != k + 1 {
            return Err(format!("spgemm: b_row_ptr len {} != k+1", b_row_ptr.len()));
        }
        // Symbolic + numeric in one pass with SPA.
        let mut c_row_ptr = vec![0i32; m + 1];
        let mut c_col_idx: Vec<i32> = Vec::new();
        let mut c_values: Vec<f64> = Vec::new();

        // Sparse accumulator: marker[col] = row index where last touched.
        let mut marker = vec![-1i32; n];
        let mut spa_vals = vec![0f64; n];
        let mut spa_cols: Vec<usize> = Vec::with_capacity(n);

        for i in 0..m {
            spa_cols.clear();
            for ka in a_row_ptr[i] as usize..a_row_ptr[i + 1] as usize {
                let j = a_col_idx[ka] as usize;
                let aij = a_values[ka];
                for kb in b_row_ptr[j] as usize..b_row_ptr[j + 1] as usize {
                    let l = b_col_idx[kb] as usize;
                    let bjl = b_values[kb];
                    if marker[l] != i as i32 {
                        marker[l] = i as i32;
                        spa_vals[l] = aij * bjl;
                        spa_cols.push(l);
                    } else {
                        spa_vals[l] += aij * bjl;
                    }
                }
            }
            // Sort columns for canonical CSR ordering.
            spa_cols.sort_unstable();
            for &l in &spa_cols {
                c_col_idx.push(l as i32);
                c_values.push(spa_vals[l]);
            }
            c_row_ptr[i + 1] = c_col_idx.len() as i32;
        }
        Ok((c_values, c_col_idx, c_row_ptr))
    }

    fn hcol_last_zero_check(hcol: &[f64]) -> bool {
        // After Givens rotation hcol[j+1] is 0 by construction.
        // The lucky-breakdown signal is the *un-rotated* subdiagonal
        // having been ≈0 — we approximate by checking if every
        // h_i value is small. Conservative (slightly over-eager
        // termination on near-zero columns).
        hcol.iter().all(|v| v.abs() < f64::MIN_POSITIVE * 64.0)
    }
    fn hcol_subdiag(hcol: &[f64], i: usize) -> f64 {
        hcol.get(i).copied().unwrap_or(0.0)
    }

    pub fn cg_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<(), String> {
        let n = b.len();
        if out.len() != n {
            return Err(format!("cg_solve: output len {} != b len {n}", out.len()));
        }
        if row_ptr.len() != n + 1 {
            return Err(format!(
                "cg_solve: row_ptr len {} != n+1 ({})",
                row_ptr.len(),
                n + 1
            ));
        }
        let matvec = |x: &[f64], y: &mut [f64]| {
            for r in 0..n {
                let mut acc = 0f64;
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    acc += values[k] * x[col_idx[k] as usize];
                }
                y[r] = acc;
            }
        };
        let mut x = vec![0f64; n];
        let mut r = b.to_vec();
        let mut p = r.clone();
        let mut ap = vec![0f64; n];
        let mut rs_old: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..max_iter {
            if rs_old.sqrt() < tol {
                break;
            }
            matvec(&p, &mut ap);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap == 0.0 {
                return Err("cg_solve: pᵀ·A·p = 0 (A is singular or not SPD)".into());
            }
            let alpha = rs_old / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            for i in 0..n {
                r[i] -= alpha * ap[i];
            }
            let rs_new: f64 = r.iter().map(|v| v * v).sum();
            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }
        out.copy_from_slice(&x);
        Ok(())
    }
}

// ── Sparse LU Solve ───────────────────────────────────────────────

struct SparseLuExt;

impl OpExtension for SparseLuExt {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let values = inputs[0];
        let col_idx = inputs[1];
        let row_ptr = inputs[2];
        let b = inputs[3];
        assert_eq!(values.dtype(), DType::F64, "sparse_lu: values must be F64");
        assert_eq!(
            col_idx.dtype(),
            DType::I32,
            "sparse_lu: col_idx must be I32"
        );
        assert_eq!(
            row_ptr.dtype(),
            DType::I32,
            "sparse_lu: row_ptr must be I32"
        );
        assert_eq!(b.dtype(), DType::F64, "sparse_lu: b must be F64");
        b.clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // y = solve(A, b). Closed-form gradients:
        //   dL/db = solve(Aᵀ, dL/dy)         [v1 symmetric → reuse A]
        //   dL/dvalues[k] = -dL/db[row(k)] · y_fwd[col(k)]   gathered at nonzero k
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];

        let g_b = ctx.bwd.custom_op(
            SPARSE_LU_SOLVE,
            Vec::new(),
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );

        // y is the forward solve output, mirrored in the bwd graph
        // by `grad_with_loss`'s up-front fwd→bwd Node copy. Look it up
        // via ctx.fwd_map[&node.id].
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, g_b, y_fwd],
        );
        // The values gradient is `-dL/db ⊗ y`, so negate the gather.
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);

        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparseLuCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseLuCpu {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("sparse_lu values")?;
        let col_idx = inputs[1].expect_i32("sparse_lu col_idx")?;
        let row_ptr = inputs[2].expect_i32("sparse_lu row_ptr")?;
        let b = inputs[3].expect_f64("sparse_lu b")?;
        let out = output.expect_f64_mut("sparse_lu output")?;
        algos::lu_solve(values, col_idx, row_ptr, b, out)
    }
}

// ── Sparse Mat-Vec ────────────────────────────────────────────────

struct SparseMatVecExt;

impl OpExtension for SparseMatVecExt {
    fn name(&self) -> &str {
        SPARSE_MAT_VEC
    }
    fn num_inputs(&self) -> usize {
        4
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // y has the shape of x (A is n×n by convention).
        inputs[3].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // y = A·x. Closed-form gradients:
        //   dL/dx = Aᵀ · upstream                  [v1 symmetric → mat_vec(A, ...)]
        //   dL/dvalues[k] = upstream[row(k)] · x[col(k)]   gathered at nonzero k
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];
        let x_bwd = ctx.fwd_map[&node.inputs[3]];

        let g_x = ctx.bwd.custom_op(
            SPARSE_MAT_VEC,
            Vec::new(),
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );
        let g_vals = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, ctx.upstream, x_bwd],
        );
        vec![(0, g_vals), (3, g_x)]
    }
}

#[cfg(feature = "cpu")]
struct SparseMatVecCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseMatVecCpu {
    fn name(&self) -> &str {
        SPARSE_MAT_VEC
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("mat_vec values")?;
        let col_idx = inputs[1].expect_i32("mat_vec col_idx")?;
        let row_ptr = inputs[2].expect_i32("mat_vec row_ptr")?;
        let x = inputs[3].expect_f64("mat_vec x")?;
        let out = output.expect_f64_mut("mat_vec y")?;
        algos::mat_vec(values, col_idx, row_ptr, x, out)
    }
}

// ── Conjugate Gradient ────────────────────────────────────────────

/// Encode CG attrs into the opaque `Vec<u8>` blob carried on
/// `Op::Custom`. Layout: `[max_iter:u32 LE, tol:f64 LE]` — 12 bytes.
pub fn encode_cg_attrs(max_iter: u32, tol: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&max_iter.to_le_bytes());
    out.extend_from_slice(&tol.to_le_bytes());
    out
}

fn decode_cg_attrs(attrs: &[u8]) -> Result<(u32, f64), String> {
    if attrs.len() != 12 {
        return Err(format!(
            "cg_solve: attrs must be 12 bytes (u32 max_iter + f64 tol), got {}",
            attrs.len()
        ));
    }
    let max_iter = u32::from_le_bytes(attrs[0..4].try_into().unwrap());
    let tol = f64::from_le_bytes(attrs[4..12].try_into().unwrap());
    Ok((max_iter, tol))
}

struct SparseCgExt;

impl OpExtension for SparseCgExt {
    fn name(&self) -> &str {
        SPARSE_CG_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    }

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Same shape as SparseLuExt::vjp — iterative solver, same
        // closed-form adjoint. The adjoint solve recurses into
        // sparse_cg_solve with the forward's attrs (same tolerance
        // and iteration cap).
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_CG_SOLVE,
            attrs,
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);

        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparseCgCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SparseCgCpu {
    fn name(&self) -> &str {
        SPARSE_CG_SOLVE
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("cg_solve values")?;
        let col_idx = inputs[1].expect_i32("cg_solve col_idx")?;
        let row_ptr = inputs[2].expect_i32("cg_solve row_ptr")?;
        let b = inputs[3].expect_f64("cg_solve b")?;
        let out = output.expect_f64_mut("cg_solve x")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::cg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── Sparse Values Gradient (`dL/dvalues` building block) ─────────

struct SparseValuesGradExt;

impl OpExtension for SparseValuesGradExt {
    fn name(&self) -> &str {
        SPARSE_VALUES_GRAD
    }
    fn num_inputs(&self) -> usize {
        4
    } // col_idx, row_ptr, u, v

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // Output shape == col_idx shape (length nnz, F64).
        let col_idx = inputs[0];
        assert_eq!(
            col_idx.dtype(),
            DType::I32,
            "values_grad: col_idx must be I32"
        );
        let nnz = col_idx
            .num_elements()
            .expect("values_grad: col_idx must have static shape");
        Shape::new(&[nnz], DType::F64)
    }
    // Non-differentiable (it's itself a gradient kernel; second-order
    // derivatives are out of v1 scope).
}

#[cfg(feature = "cpu")]
struct SparseValuesGradCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseValuesGradCpu {
    fn name(&self) -> &str {
        SPARSE_VALUES_GRAD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let col_idx = inputs[0].expect_i32("values_grad col_idx")?;
        let row_ptr = inputs[1].expect_i32("values_grad row_ptr")?;
        let u = inputs[2].expect_f64("values_grad u")?;
        let v = inputs[3].expect_f64("values_grad v")?;
        let out = output.expect_f64_mut("values_grad out")?;
        algos::values_grad(col_idx, row_ptr, u, v, out)
    }
}

// ── Sparse LU Solve (general / non-symmetric) ─────────────────────
//
// 7-input variant of `sparse_lu_solve`. Forward identity to the
// symmetric version (kernel only reads inputs 0..=3). The last 3
// inputs are the transpose CSR triplet used by the VJP for the
// adjoint solve `dL/db = solve(Aᵀ, dL/dx)`. Use this for non-
// symmetric matrices where reusing the forward triplet for the
// adjoint would be wrong.

struct SparseLuGeneralExt;

impl OpExtension for SparseLuGeneralExt {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE_GENERAL
    }
    fn num_inputs(&self) -> usize {
        7
    }
    // values_A, col_idx_A, row_ptr_A, b, values_AT, col_idx_AT, row_ptr_AT
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let b = inputs[3];
        b.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/db = solve(Aᵀ, dL/dy) — emit ANOTHER lu_solve_general
        // call with the roles swapped: the adjoint's forward matrix
        // is Aᵀ (already provided in inputs 4..7), so the adjoint
        // call's transpose triplet is back to A (inputs 0..3). The
        // outer sparse-LU-general's adjoint is itself a sparse-LU-
        // general — same recursion as in CG.
        let vals_a = ctx.fwd_map[&node.inputs[0]];
        let cidx_a = ctx.fwd_map[&node.inputs[1]];
        let rptr_a = ctx.fwd_map[&node.inputs[2]];
        let vals_at = ctx.fwd_map[&node.inputs[4]];
        let cidx_at = ctx.fwd_map[&node.inputs[5]];
        let rptr_at = ctx.fwd_map[&node.inputs[6]];

        let g_b = ctx.bwd.custom_op(
            SPARSE_LU_SOLVE_GENERAL,
            Vec::new(),
            // forward A is now Aᵀ; transpose for *this* adjoint solve is A.
            vec![
                vals_at,
                cidx_at,
                rptr_at,
                ctx.upstream,
                vals_a,
                cidx_a,
                rptr_a,
            ],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_a, rptr_a, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparseLuGeneralCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseLuGeneralCpu {
    fn name(&self) -> &str {
        SPARSE_LU_SOLVE_GENERAL
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        // Forward only reads A; the AT triplet rides along for the
        // VJP and is unused here. Same algos::lu_solve as the
        // symmetric version.
        let values = inputs[0].expect_f64("lu_solve_general values")?;
        let col_idx = inputs[1].expect_i32("lu_solve_general col_idx")?;
        let row_ptr = inputs[2].expect_i32("lu_solve_general row_ptr")?;
        let b = inputs[3].expect_f64("lu_solve_general b")?;
        let out = output.expect_f64_mut("lu_solve_general out")?;
        algos::lu_solve(values, col_idx, row_ptr, b, out)
    }
}

// ── GMRES Solve (non-symmetric iterative) ────────────────────────
//
// Iterative analog of CG for non-symmetric A. Same 7-input shape as
// `sparse_lu_solve_general`. Attrs encode `(max_iter, tol)` exactly
// like CG.

struct SparseGmresExt;

impl OpExtension for SparseGmresExt {
    fn name(&self) -> &str {
        SPARSE_GMRES_SOLVE
    }
    fn num_inputs(&self) -> usize {
        7
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Same closed-form adjoint as LU general. Recurse into
        // gmres_solve against (Aᵀ, upstream) with the same attrs.
        let vals_a = ctx.fwd_map[&node.inputs[0]];
        let cidx_a = ctx.fwd_map[&node.inputs[1]];
        let rptr_a = ctx.fwd_map[&node.inputs[2]];
        let vals_at = ctx.fwd_map[&node.inputs[4]];
        let cidx_at = ctx.fwd_map[&node.inputs[5]];
        let rptr_at = ctx.fwd_map[&node.inputs[6]];
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_GMRES_SOLVE,
            attrs,
            vec![
                vals_at,
                cidx_at,
                rptr_at,
                ctx.upstream,
                vals_a,
                cidx_a,
                rptr_a,
            ],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_a, rptr_a, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparseGmresCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseGmresCpu {
    fn name(&self) -> &str {
        SPARSE_GMRES_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("gmres values")?;
        let col_idx = inputs[1].expect_i32("gmres col_idx")?;
        let row_ptr = inputs[2].expect_i32("gmres row_ptr")?;
        let b = inputs[3].expect_f64("gmres b")?;
        let out = output.expect_f64_mut("gmres out")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::gmres_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── Sparse Transpose Values ───────────────────────────────────────

struct SparseTransposeValuesExt;

impl OpExtension for SparseTransposeValuesExt {
    fn name(&self) -> &str {
        SPARSE_TRANSPOSE_VALUES
    }
    fn num_inputs(&self) -> usize {
        5
    } // values, col_idx, row_ptr, col_idx_T, row_ptr_T

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        // Output shape = same as values (length nnz).
        inputs[0].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // VJP of transpose is itself transpose (with patterns swapped).
        // dL/d(values_A) = transpose_values(upstream, col_idx_T, row_ptr_T,
        //                                    col_idx, row_ptr).
        let cidx_a = ctx.fwd_map[&node.inputs[1]];
        let rptr_a = ctx.fwd_map[&node.inputs[2]];
        let cidx_at = ctx.fwd_map[&node.inputs[3]];
        let rptr_at = ctx.fwd_map[&node.inputs[4]];
        let g_vals = ctx.bwd.custom_op(
            SPARSE_TRANSPOSE_VALUES,
            Vec::new(),
            vec![ctx.upstream, cidx_at, rptr_at, cidx_a, rptr_a],
        );
        vec![(0, g_vals)]
    }
}

#[cfg(feature = "cpu")]
struct SparseTransposeValuesCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseTransposeValuesCpu {
    fn name(&self) -> &str {
        SPARSE_TRANSPOSE_VALUES
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("transpose_values values")?;
        let col_idx = inputs[1].expect_i32("transpose_values col_idx")?;
        let row_ptr = inputs[2].expect_i32("transpose_values row_ptr")?;
        let col_idx_t = inputs[3].expect_i32("transpose_values col_idx_T")?;
        let row_ptr_t = inputs[4].expect_i32("transpose_values row_ptr_T")?;
        let out = output.expect_f64_mut("transpose_values out")?;
        algos::transpose_values(values, col_idx, row_ptr, col_idx_t, row_ptr_t, out)
    }
}

// ── PCG (Jacobi preconditioner) ───────────────────────────────────

struct SparsePcgExt;

impl OpExtension for SparsePcgExt {
    fn name(&self) -> &str {
        SPARSE_PCG_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    } // values, col_idx, row_ptr, b

    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Closed-form adjoint identical to CG (same SPD assumption).
        // Recurse into pcg_solve with the same attrs.
        let vals_b = ctx.fwd_map[&node.inputs[0]];
        let cidx_b = ctx.fwd_map[&node.inputs[1]];
        let rptr_b = ctx.fwd_map[&node.inputs[2]];
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_PCG_SOLVE,
            attrs,
            vec![vals_b, cidx_b, rptr_b, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad = ctx.bwd.custom_op(
            SPARSE_VALUES_GRAD,
            Vec::new(),
            vec![cidx_b, rptr_b, g_b, y_fwd],
        );
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparsePcgCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparsePcgCpu {
    fn name(&self) -> &str {
        SPARSE_PCG_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("pcg values")?;
        let col_idx = inputs[1].expect_i32("pcg col_idx")?;
        let row_ptr = inputs[2].expect_i32("pcg row_ptr")?;
        let b = inputs[3].expect_f64("pcg b")?;
        let out = output.expect_f64_mut("pcg x")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::pcg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── BiCGSTAB (non-symmetric) ──────────────────────────────────────

fn decode_bicgstab_attrs(attrs: &[u8]) -> Result<(u32, f64, bool), String> {
    if attrs.len() < 13 {
        return Err(format!("bicgstab: attrs len {} < 13", attrs.len()));
    }
    let max_iter = u32::from_le_bytes(attrs[0..4].try_into().unwrap());
    let tol = f64::from_le_bytes(attrs[4..12].try_into().unwrap());
    let trans = attrs[12] != 0;
    Ok((max_iter, tol, trans))
}

struct SparseBicgstabExt;

impl OpExtension for SparseBicgstabExt {
    fn name(&self) -> &str {
        SPARSE_BICGSTAB_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    } // values, col_idx, row_ptr, b
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Forward solves A·x = b (or Aᵀ·x = b if transpose_a flag set).
        // VJP: dL/db = A⁻ᵀ·G via bicgstab with transpose flag flipped.
        //      dL/dvalues = -y_adj ⊗ x  (gathered on CSR pattern).
        let vals = ctx.fwd_map[&node.inputs[0]];
        let cidx = ctx.fwd_map[&node.inputs[1]];
        let rptr = ctx.fwd_map[&node.inputs[2]];
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let (max_iter, tol, trans) = decode_bicgstab_attrs(&attrs).unwrap_or((1, 1e-9, false));
        let mut adj_attrs = Vec::with_capacity(13);
        adj_attrs.extend_from_slice(&max_iter.to_le_bytes());
        adj_attrs.extend_from_slice(&tol.to_le_bytes());
        adj_attrs.push(if !trans { 1 } else { 0 });
        let g_b = ctx.bwd.custom_op(
            SPARSE_BICGSTAB_SOLVE,
            adj_attrs,
            vec![vals, cidx, rptr, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad =
            ctx.bwd
                .custom_op(SPARSE_VALUES_GRAD, Vec::new(), vec![cidx, rptr, g_b, y_fwd]);
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparseBicgstabCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseBicgstabCpu {
    fn name(&self) -> &str {
        SPARSE_BICGSTAB_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("bicgstab values")?;
        let col_idx = inputs[1].expect_i32("bicgstab col_idx")?;
        let row_ptr = inputs[2].expect_i32("bicgstab row_ptr")?;
        let b = inputs[3].expect_f64("bicgstab b")?;
        let out = output.expect_f64_mut("bicgstab x")?;
        let (max_iter, tol, trans) = decode_bicgstab_attrs(attrs)?;
        algos::bicgstab(values, col_idx, row_ptr, b, out, max_iter, tol, trans)
    }
}

// ── ILU(0)-preconditioned CG ──────────────────────────────────────

struct SparseIluPcgExt;

impl OpExtension for SparseIluPcgExt {
    fn name(&self) -> &str {
        SPARSE_ILU_PCG_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Same SPD adjoint shape as plain CG/PCG — reuse the kernel.
        let vals = ctx.fwd_map[&node.inputs[0]];
        let cidx = ctx.fwd_map[&node.inputs[1]];
        let rptr = ctx.fwd_map[&node.inputs[2]];
        let attrs = match &node.op {
            Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_b = ctx.bwd.custom_op(
            SPARSE_ILU_PCG_SOLVE,
            attrs,
            vec![vals, cidx, rptr, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad =
            ctx.bwd
                .custom_op(SPARSE_VALUES_GRAD, Vec::new(), vec![cidx, rptr, g_b, y_fwd]);
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparseIluPcgCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseIluPcgCpu {
    fn name(&self) -> &str {
        SPARSE_ILU_PCG_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("ilu_pcg values")?;
        let col_idx = inputs[1].expect_i32("ilu_pcg col_idx")?;
        let row_ptr = inputs[2].expect_i32("ilu_pcg row_ptr")?;
        let b = inputs[3].expect_f64("ilu_pcg b")?;
        let out = output.expect_f64_mut("ilu_pcg x")?;
        let (max_iter, tol) = decode_cg_attrs(attrs)?;
        algos::ilu_pcg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
    }
}

// ── Sparse Cholesky (direct) ──────────────────────────────────────

struct SparseCholeskyExt;

impl OpExtension for SparseCholeskyExt {
    fn name(&self) -> &str {
        SPARSE_CHOLESKY_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    } // values, col_idx, row_ptr, b
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        inputs[3].clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Same closed form as sparse_lu_solve (SPD ⇒ Aᵀ = A so the
        // adjoint solve uses the same kernel).
        let vals = ctx.fwd_map[&node.inputs[0]];
        let cidx = ctx.fwd_map[&node.inputs[1]];
        let rptr = ctx.fwd_map[&node.inputs[2]];
        let g_b = ctx.bwd.custom_op(
            SPARSE_CHOLESKY_SOLVE,
            Vec::new(),
            vec![vals, cidx, rptr, ctx.upstream],
        );
        let y_fwd = ctx.fwd_map[&node.id];
        let raw_grad =
            ctx.bwd
                .custom_op(SPARSE_VALUES_GRAD, Vec::new(), vec![cidx, rptr, g_b, y_fwd]);
        let raw_shape = ctx.bwd.node(raw_grad).shape.clone();
        let g_vals = ctx
            .bwd
            .activation(rlx_ir::op::Activation::Neg, raw_grad, raw_shape);
        vec![(0, g_vals), (3, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct SparseCholeskyCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseCholeskyCpu {
    fn name(&self) -> &str {
        SPARSE_CHOLESKY_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("chol values")?;
        let col_idx = inputs[1].expect_i32("chol col_idx")?;
        let row_ptr = inputs[2].expect_i32("chol row_ptr")?;
        let b = inputs[3].expect_f64("chol b")?;
        let out = output.expect_f64_mut("chol x")?;
        algos::cholesky_solve(values, col_idx, row_ptr, b, out)
    }
}

// ── LSQR ──────────────────────────────────────────────────────────

fn decode_lsqr_attrs(attrs: &[u8]) -> Result<(u32, f64, u32), String> {
    if attrs.len() < 16 {
        return Err(format!("lsqr: attrs len {} < 16", attrs.len()));
    }
    let max_iter = u32::from_le_bytes(attrs[0..4].try_into().unwrap());
    let tol = f64::from_le_bytes(attrs[4..12].try_into().unwrap());
    let n_cols = u32::from_le_bytes(attrs[12..16].try_into().unwrap());
    Ok((max_iter, tol, n_cols))
}

struct SparseLsqrExt;

impl OpExtension for SparseLsqrExt {
    fn name(&self) -> &str {
        SPARSE_LSQR_SOLVE
    }
    fn num_inputs(&self) -> usize {
        4
    } // values, col_idx, row_ptr, b
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let (_, _, n_cols) =
            decode_lsqr_attrs(attrs).expect("lsqr: attrs must encode (max_iter, tol, n_cols)");
        Shape::new(&[n_cols as usize], inputs[3].dtype())
    }
    // VJP deferred — see SPARSE_LSQR_SOLVE doc.
}

#[cfg(feature = "cpu")]
struct SparseLsqrCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SparseLsqrCpu {
    fn name(&self) -> &str {
        SPARSE_LSQR_SOLVE
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let values = inputs[0].expect_f64("lsqr values")?;
        let col_idx = inputs[1].expect_i32("lsqr col_idx")?;
        let row_ptr = inputs[2].expect_i32("lsqr row_ptr")?;
        let b = inputs[3].expect_f64("lsqr b")?;
        let out = output.expect_f64_mut("lsqr x")?;
        let (max_iter, tol, n_cols) = decode_lsqr_attrs(attrs)?;
        algos::lsqr_solve(
            values,
            col_idx,
            row_ptr,
            b,
            out,
            max_iter,
            tol,
            n_cols as usize,
        )
    }
}

// ── Pure-Rust SpGEMM (CSR × CSR → CSR) ────────────────────────────

/// CSR × CSR → CSR via Gustavson's algorithm. Pure-Rust convenience
/// wrapper around `algos::spgemm_csr` — exposed outside the IR
/// because sparsity patterns are structural (and typically static
/// across the differentiable training loop), so a one-shot multiply
/// at graph-build time is the natural shape. Returns
/// `(c_values, c_col_idx, c_row_ptr)` for the output CSR.
#[cfg(feature = "cpu")]
pub fn spgemm_csr(
    a_values: &[f64],
    a_col_idx: &[i32],
    a_row_ptr: &[i32],
    b_values: &[f64],
    b_col_idx: &[i32],
    b_row_ptr: &[i32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(Vec<f64>, Vec<i32>, Vec<i32>), String> {
    algos::spgemm_csr(
        a_values, a_col_idx, a_row_ptr, b_values, b_col_idx, b_row_ptr, m, k, n,
    )
}

// ── Pure-Rust helper for the structural CSR transpose pattern ─────

/// Compute `(col_idx_T, row_ptr_T)` — the sparsity pattern of `Aᵀ`
/// — from `A`'s pattern. This is the structural step that must
/// happen before [`SPARSE_TRANSPOSE_VALUES`] can permute the values
/// per Newton iteration. Result is independent of the values, so
/// downstream callers compute it once and embed as `Op::Constant`.
pub fn csr_transpose_pattern(
    col_idx: &[i32],
    row_ptr: &[i32],
    n_rows: usize,
    n_cols: usize,
) -> (Vec<i32>, Vec<i32>) {
    let nnz = col_idx.len();
    // Count entries per output-row (= input column).
    let mut t_count = vec![0i32; n_cols];
    for &c in col_idx {
        t_count[c as usize] += 1;
    }
    let mut t_row_ptr = vec![0i32; n_cols + 1];
    for r in 0..n_cols {
        t_row_ptr[r + 1] = t_row_ptr[r] + t_count[r];
    }
    let mut t_col_idx = vec![0i32; nnz];
    let mut cursor = t_row_ptr.clone();
    for r in 0..n_rows {
        for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
            let c = col_idx[k] as usize;
            let pos = cursor[c] as usize;
            t_col_idx[pos] = r as i32;
            cursor[c] += 1;
        }
    }
    (t_col_idx, t_row_ptr)
}

// ── SparseTensor: the boundary abstraction ────────────────────────

/// CSR-format sparse matrix at the IR level. Bundles the three
/// CSR `NodeId`s with structural shape info known at graph-build time.
///
/// Modeled on `jax.experimental.sparse.BCOO` — a wrapper around
/// `(data, indices)` arrays plus a `shape` tuple, with methods that
/// expand to the right subgraph for each operation.
#[derive(Clone, Copy, Debug)]
pub struct SparseTensor {
    /// Non-zero values in row-major CSR order. F64.
    pub values: NodeId,
    /// Column index per non-zero. I32.
    pub col_idx: NodeId,
    /// Start index in `values` / `col_idx` per row, length `n_rows + 1`. I32.
    pub row_ptr: NodeId,
    /// Logical row count of A.
    pub n_rows: usize,
    /// Logical column count of A (`n_rows == n_cols` for square / SPD).
    pub n_cols: usize,
}

impl SparseTensor {
    /// Build from existing CSR `NodeId`s. Caller is responsible for
    /// the layout invariants (sortedness within rows,
    /// `row_ptr.len() == n_rows + 1`, etc.).
    pub fn from_csr(
        values: NodeId,
        col_idx: NodeId,
        row_ptr: NodeId,
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        Self {
            values,
            col_idx,
            row_ptr,
            n_rows,
            n_cols,
        }
    }

    /// `y = A · x` for a length-`n_cols` dense vector.
    pub fn mat_vec(&self, g: &mut Graph, x: NodeId) -> NodeId {
        g.custom_op(
            SPARSE_MAT_VEC,
            Vec::new(),
            vec![self.values, self.col_idx, self.row_ptr, x],
        )
    }

    /// `x = A⁻¹ · b` via direct LU.
    pub fn solve(&self, g: &mut Graph, b: NodeId) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::solve requires a square matrix"
        );
        g.custom_op(
            SPARSE_LU_SOLVE,
            Vec::new(),
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

    /// `x = A⁻¹ · b` via Conjugate Gradient. SPD only. `tol` is the
    /// absolute residual threshold; `max_iter` caps iteration count.
    pub fn cg_solve(&self, g: &mut Graph, b: NodeId, max_iter: u32, tol: f64) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::cg_solve requires a square matrix"
        );
        g.custom_op(
            SPARSE_CG_SOLVE,
            encode_cg_attrs(max_iter, tol),
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

    /// `x = A⁻¹ · b` via direct LU for **non-symmetric** A. The
    /// caller supplies an explicit transpose `adjoint` (CSR of `Aᵀ`)
    /// — for square non-symmetric matrices, `Aᵀ` has the same
    /// nnz pattern as `A` after a CSR↔CSC swap. The forward solve
    /// uses only `self`; the VJP routes the adjoint solve through
    /// `adjoint`. Use this in place of `solve` when the assumption
    /// `Aᵀ = A` would be wrong.
    pub fn solve_general(&self, g: &mut Graph, b: NodeId, adjoint: &SparseTensor) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::solve_general requires a square matrix"
        );
        assert_eq!(
            adjoint.n_rows, self.n_cols,
            "adjoint shape mismatch: A is {}×{}, Aᵀ should be {}×{}",
            self.n_rows, self.n_cols, self.n_cols, self.n_rows
        );
        g.custom_op(
            SPARSE_LU_SOLVE_GENERAL,
            Vec::new(),
            vec![
                self.values,
                self.col_idx,
                self.row_ptr,
                b,
                adjoint.values,
                adjoint.col_idx,
                adjoint.row_ptr,
            ],
        )
    }

    /// `x = A⁻¹ · b` via Jacobi-preconditioned CG. SPD only.
    /// Convergence dramatically faster than plain CG on ill-
    /// conditioned matrices where `diag(A)` captures most of the
    /// magnitude variation — typical for circuit MNA matrices with
    /// mixed-magnitude device parameters. The preconditioner is
    /// extracted from the CSR by the kernel; no separate input.
    pub fn pcg_solve(&self, g: &mut Graph, b: NodeId, max_iter: u32, tol: f64) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::pcg_solve requires a square matrix"
        );
        g.custom_op(
            SPARSE_PCG_SOLVE,
            encode_cg_attrs(max_iter, tol),
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

    /// Permute this tensor's `values` into the values vector of
    /// `Aᵀ`. The transposed pattern `(col_idx_t, row_ptr_t)` is
    /// supplied as `NodeId`s — typically computed once via
    /// [`crate::csr_transpose_pattern`] and embedded as `Op::Constant`
    /// since the pattern is fixed across Newton iterations.
    pub fn transpose_values(&self, g: &mut Graph, col_idx_t: NodeId, row_ptr_t: NodeId) -> NodeId {
        g.custom_op(
            SPARSE_TRANSPOSE_VALUES,
            Vec::new(),
            vec![
                self.values,
                self.col_idx,
                self.row_ptr,
                col_idx_t,
                row_ptr_t,
            ],
        )
    }

    /// `x = A⁻¹ · b` via direct sparse Cholesky for SPD A. Densifies
    /// into a dense buffer and calls LAPACK `dpotrf` + triangular
    /// solves. Mirror of `solve` (LU-based) but ½× factor cost and
    /// numerically more stable; only valid when A is SPD.
    pub fn cholesky_solve(&self, g: &mut Graph, b: NodeId) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::cholesky_solve requires a square matrix"
        );
        g.custom_op(
            SPARSE_CHOLESKY_SOLVE,
            Vec::new(),
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

    /// `x = argmin ||A·x - b||₂` via LSQR (Paige-Saunders 1982).
    /// Works for any A (square / over-determined / under-determined);
    /// returns the minimum-norm solution when A is rank-deficient or
    /// under-determined. VJP not implemented in v1.
    pub fn lsqr_solve(&self, g: &mut Graph, b: NodeId, max_iter: u32, tol: f64) -> NodeId {
        let mut attrs = Vec::with_capacity(16);
        attrs.extend_from_slice(&max_iter.to_le_bytes());
        attrs.extend_from_slice(&tol.to_le_bytes());
        attrs.extend_from_slice(&(self.n_cols as u32).to_le_bytes());
        g.custom_op(
            SPARSE_LSQR_SOLVE,
            attrs,
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

    /// `x = A⁻¹ · b` via BiCGSTAB for **non-symmetric** A. Same
    /// shape contract as CG/PCG (no explicit adjoint pattern needed —
    /// the kernel itself can solve Aᵀ·x = b via a flag, used by VJPs).
    pub fn bicgstab_solve(&self, g: &mut Graph, b: NodeId, max_iter: u32, tol: f64) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::bicgstab_solve requires a square matrix"
        );
        let mut attrs = Vec::with_capacity(13);
        attrs.extend_from_slice(&max_iter.to_le_bytes());
        attrs.extend_from_slice(&tol.to_le_bytes());
        attrs.push(0); // transpose_a = false
        g.custom_op(
            SPARSE_BICGSTAB_SOLVE,
            attrs,
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

    /// `x = A⁻¹ · b` via ILU(0)-preconditioned CG. SPD A required
    /// (same contract as CG/PCG). ILU is factored on each call —
    /// for static-pattern Newton loops the cost amortizes against the
    /// faster convergence vs. Jacobi-PCG.
    pub fn ilu_pcg_solve(&self, g: &mut Graph, b: NodeId, max_iter: u32, tol: f64) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::ilu_pcg_solve requires a square matrix"
        );
        g.custom_op(
            SPARSE_ILU_PCG_SOLVE,
            encode_cg_attrs(max_iter, tol),
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

    /// `x = A⁻¹ · b` via GMRES for **non-symmetric** A. Same
    /// transpose-triplet contract as `solve_general`. `max_iter`
    /// caps Krylov dimension; `tol` is the residual norm threshold.
    pub fn gmres_solve(
        &self,
        g: &mut Graph,
        b: NodeId,
        max_iter: u32,
        tol: f64,
        adjoint: &SparseTensor,
    ) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::gmres_solve requires a square matrix"
        );
        assert_eq!(adjoint.n_rows, self.n_cols, "adjoint shape mismatch");
        g.custom_op(
            SPARSE_GMRES_SOLVE,
            encode_cg_attrs(max_iter, tol),
            vec![
                self.values,
                self.col_idx,
                self.row_ptr,
                b,
                adjoint.values,
                adjoint.col_idx,
                adjoint.row_ptr,
            ],
        )
    }
}

// ── Metal kernels ─────────────────────────────────────────────────
//
// Active with the `metal` feature. The MetalKernel trait gives us
// raw `(&[u8], &Shape)` pairs; we cast each to its declared dtype
// and call the same `algos::*` body the CpuKernel impls use. Apple
// Silicon's unified memory means `Buffer::contents()` is host-
// accessible — running these kernels on the Metal backend is no
// slower than running on CPU, *provided* the rlx-metal executor's
// segment-at-CustomOp dispatch is wired (which it is, as of the
// owned-encoder refactor).

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_kernels {
    use super::*;
    use rlx_ir::DType;
    use rlx_metal::op_registry::MetalKernel;

    /// Cast `&[u8]` → `&[T]` after dtype-checking the accompanying Shape.
    /// Length is taken from the Shape's element count (must match the
    /// byte-slice length). Caller asserts contiguous + aligned data,
    /// which the rlx-metal arena delivers.
    unsafe fn typed<'a, T: Copy>(
        bytes: &'a [u8],
        shape: &rlx_ir::Shape,
        want: DType,
        role: &str,
    ) -> Result<&'a [T], String> {
        if shape.dtype() != want {
            return Err(format!(
                "{role}: expected {want:?}, got {:?}",
                shape.dtype()
            ));
        }
        let n = shape
            .num_elements()
            .ok_or_else(|| format!("{role}: dynamic shape not supported"))?;
        let need = n * std::mem::size_of::<T>();
        if bytes.len() < need {
            return Err(format!("{role}: bytes {} < need {need}", bytes.len()));
        }
        Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n) })
    }

    unsafe fn typed_mut<'a, T: Copy>(
        bytes: &'a mut [u8],
        shape: &rlx_ir::Shape,
        want: DType,
        role: &str,
    ) -> Result<&'a mut [T], String> {
        if shape.dtype() != want {
            return Err(format!(
                "{role}: expected {want:?}, got {:?}",
                shape.dtype()
            ));
        }
        let n = shape
            .num_elements()
            .ok_or_else(|| format!("{role}: dynamic shape not supported"))?;
        let need = n * std::mem::size_of::<T>();
        if bytes.len() < need {
            return Err(format!("{role}: bytes {} < need {need}", bytes.len()));
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, n) })
    }

    #[derive(Debug)]
    pub(super) struct SparseLuMetal;
    impl MetalKernel for SparseLuMetal {
        fn name(&self) -> &str {
            SPARSE_LU_SOLVE
        }
        fn execute(
            &self,
            inputs: &[(&[u8], &rlx_ir::Shape)],
            output: (&mut [u8], &rlx_ir::Shape),
            _attrs: &[u8],
        ) -> Result<(), String> {
            unsafe {
                let values = typed::<f64>(inputs[0].0, inputs[0].1, DType::F64, "values")?;
                let col_idx = typed::<i32>(inputs[1].0, inputs[1].1, DType::I32, "col_idx")?;
                let row_ptr = typed::<i32>(inputs[2].0, inputs[2].1, DType::I32, "row_ptr")?;
                let b = typed::<f64>(inputs[3].0, inputs[3].1, DType::F64, "b")?;
                let out = typed_mut::<f64>(output.0, output.1, DType::F64, "out")?;
                algos::lu_solve(values, col_idx, row_ptr, b, out)
            }
        }
    }

    #[derive(Debug)]
    pub(super) struct SparseMatVecMetal;
    impl MetalKernel for SparseMatVecMetal {
        fn name(&self) -> &str {
            SPARSE_MAT_VEC
        }
        fn execute(
            &self,
            inputs: &[(&[u8], &rlx_ir::Shape)],
            output: (&mut [u8], &rlx_ir::Shape),
            _attrs: &[u8],
        ) -> Result<(), String> {
            unsafe {
                let values = typed::<f64>(inputs[0].0, inputs[0].1, DType::F64, "values")?;
                let col_idx = typed::<i32>(inputs[1].0, inputs[1].1, DType::I32, "col_idx")?;
                let row_ptr = typed::<i32>(inputs[2].0, inputs[2].1, DType::I32, "row_ptr")?;
                let x = typed::<f64>(inputs[3].0, inputs[3].1, DType::F64, "x")?;
                let out = typed_mut::<f64>(output.0, output.1, DType::F64, "out")?;
                algos::mat_vec(values, col_idx, row_ptr, x, out)
            }
        }
    }

    #[derive(Debug)]
    pub(super) struct SparseCgMetal;
    impl MetalKernel for SparseCgMetal {
        fn name(&self) -> &str {
            SPARSE_CG_SOLVE
        }
        fn execute(
            &self,
            inputs: &[(&[u8], &rlx_ir::Shape)],
            output: (&mut [u8], &rlx_ir::Shape),
            attrs: &[u8],
        ) -> Result<(), String> {
            let (max_iter, tol) = decode_cg_attrs(attrs)?;
            unsafe {
                let values = typed::<f64>(inputs[0].0, inputs[0].1, DType::F64, "values")?;
                let col_idx = typed::<i32>(inputs[1].0, inputs[1].1, DType::I32, "col_idx")?;
                let row_ptr = typed::<i32>(inputs[2].0, inputs[2].1, DType::I32, "row_ptr")?;
                let b = typed::<f64>(inputs[3].0, inputs[3].1, DType::F64, "b")?;
                let out = typed_mut::<f64>(output.0, output.1, DType::F64, "out")?;
                algos::cg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
            }
        }
    }

    #[derive(Debug)]
    pub(super) struct SparseValuesGradMetal;
    impl MetalKernel for SparseValuesGradMetal {
        fn name(&self) -> &str {
            SPARSE_VALUES_GRAD
        }
        fn execute(
            &self,
            inputs: &[(&[u8], &rlx_ir::Shape)],
            output: (&mut [u8], &rlx_ir::Shape),
            _attrs: &[u8],
        ) -> Result<(), String> {
            unsafe {
                let col_idx = typed::<i32>(inputs[0].0, inputs[0].1, DType::I32, "col_idx")?;
                let row_ptr = typed::<i32>(inputs[1].0, inputs[1].1, DType::I32, "row_ptr")?;
                let u = typed::<f64>(inputs[2].0, inputs[2].1, DType::F64, "u")?;
                let v = typed::<f64>(inputs[3].0, inputs[3].1, DType::F64, "v")?;
                let out = typed_mut::<f64>(output.0, output.1, DType::F64, "out")?;
                algos::values_grad(col_idx, row_ptr, u, v, out)
            }
        }
    }

    #[derive(Debug)]
    pub(super) struct SparseLuGeneralMetal;
    impl MetalKernel for SparseLuGeneralMetal {
        fn name(&self) -> &str {
            SPARSE_LU_SOLVE_GENERAL
        }
        fn execute(
            &self,
            inputs: &[(&[u8], &rlx_ir::Shape)],
            output: (&mut [u8], &rlx_ir::Shape),
            _attrs: &[u8],
        ) -> Result<(), String> {
            // Forward only reads A; AT triplet (inputs 4..=6) rides
            // along for the VJP path and is unused here.
            unsafe {
                let values = typed::<f64>(inputs[0].0, inputs[0].1, DType::F64, "values")?;
                let col_idx = typed::<i32>(inputs[1].0, inputs[1].1, DType::I32, "col_idx")?;
                let row_ptr = typed::<i32>(inputs[2].0, inputs[2].1, DType::I32, "row_ptr")?;
                let b = typed::<f64>(inputs[3].0, inputs[3].1, DType::F64, "b")?;
                let out = typed_mut::<f64>(output.0, output.1, DType::F64, "out")?;
                algos::lu_solve(values, col_idx, row_ptr, b, out)
            }
        }
    }

    #[derive(Debug)]
    pub(super) struct SparseGmresMetal;
    impl MetalKernel for SparseGmresMetal {
        fn name(&self) -> &str {
            SPARSE_GMRES_SOLVE
        }
        fn execute(
            &self,
            inputs: &[(&[u8], &rlx_ir::Shape)],
            output: (&mut [u8], &rlx_ir::Shape),
            attrs: &[u8],
        ) -> Result<(), String> {
            let (max_iter, tol) = decode_cg_attrs(attrs)?;
            unsafe {
                let values = typed::<f64>(inputs[0].0, inputs[0].1, DType::F64, "values")?;
                let col_idx = typed::<i32>(inputs[1].0, inputs[1].1, DType::I32, "col_idx")?;
                let row_ptr = typed::<i32>(inputs[2].0, inputs[2].1, DType::I32, "row_ptr")?;
                let b = typed::<f64>(inputs[3].0, inputs[3].1, DType::F64, "b")?;
                let out = typed_mut::<f64>(output.0, output.1, DType::F64, "out")?;
                algos::gmres_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
            }
        }
    }
}

// ── MLX kernels ───────────────────────────────────────────────────
//
// Active with the `mlx` feature. The `MlxKernel` trait gives us
// MLX `Array` handles (lazy graph nodes); we read each input's
// bytes, run the same `algos::*` body the CPU + Metal kernels use,
// and build a new `Array` of the output shape from the result
// bytes. The lazy graph absorbs the new `Array` as the value for
// this `Op::Custom` node, so consumers downstream see it as just
// another operand.
//
// Same caveat as the Metal kernels: this is a host-callback that
// runs f64 LAPACK on the ARM cores, not GPU compute. The point is
// that an MLX-shaped graph that *contains* sparse-LU still routes
// correctly through MLX's pipeline; surrounding ops still benefit
// from MLX's lazy graph optimizer.

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod mlx_kernels {
    use super::*;
    use rlx_ir::DType;
    use rlx_mlx::array::{Array, MlxError};
    use rlx_mlx::op_registry::MlxKernel;

    fn shape_dims_static(s: &rlx_ir::Shape) -> Result<Vec<usize>, MlxError> {
        s.dims()
            .iter()
            .map(|d| match d {
                rlx_ir::Dim::Static(n) => Ok(*n),
                _ => Err(MlxError(
                    "rlx-sparse mlx kernel: dynamic shape not supported".into(),
                )),
            })
            .collect()
    }

    /// Reinterpret a byte buffer as f64. Length is byte-count / 8.
    fn bytes_to_f64(b: &[u8]) -> Vec<f64> {
        b.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn bytes_to_i32(b: &[u8]) -> Vec<i32> {
        b.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
    fn f64_to_bytes(xs: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(xs.len() * 8);
        for x in xs {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn run_lu(inputs: &[&Array], output_shape: &rlx_ir::Shape) -> Result<Array, MlxError> {
        let values = bytes_to_f64(&inputs[0].to_bytes()?);
        let col_idx = bytes_to_i32(&inputs[1].to_bytes()?);
        let row_ptr = bytes_to_i32(&inputs[2].to_bytes()?);
        let b = bytes_to_f64(&inputs[3].to_bytes()?);
        let mut out = vec![0f64; b.len()];
        algos::lu_solve(&values, &col_idx, &row_ptr, &b, &mut out).map_err(MlxError)?;
        let dims = shape_dims_static(output_shape)?;
        Array::from_bytes(&f64_to_bytes(&out), &dims, DType::F64)
    }

    fn run_mat_vec(inputs: &[&Array], output_shape: &rlx_ir::Shape) -> Result<Array, MlxError> {
        let values = bytes_to_f64(&inputs[0].to_bytes()?);
        let col_idx = bytes_to_i32(&inputs[1].to_bytes()?);
        let row_ptr = bytes_to_i32(&inputs[2].to_bytes()?);
        let x = bytes_to_f64(&inputs[3].to_bytes()?);
        let mut out = vec![0f64; x.len()];
        algos::mat_vec(&values, &col_idx, &row_ptr, &x, &mut out).map_err(MlxError)?;
        let dims = shape_dims_static(output_shape)?;
        Array::from_bytes(&f64_to_bytes(&out), &dims, DType::F64)
    }

    fn run_cg(
        inputs: &[&Array],
        output_shape: &rlx_ir::Shape,
        attrs: &[u8],
    ) -> Result<Array, MlxError> {
        let (max_iter, tol) = decode_cg_attrs(attrs).map_err(MlxError)?;
        let values = bytes_to_f64(&inputs[0].to_bytes()?);
        let col_idx = bytes_to_i32(&inputs[1].to_bytes()?);
        let row_ptr = bytes_to_i32(&inputs[2].to_bytes()?);
        let b = bytes_to_f64(&inputs[3].to_bytes()?);
        let mut out = vec![0f64; b.len()];
        algos::cg_solve(&values, &col_idx, &row_ptr, &b, &mut out, max_iter, tol)
            .map_err(MlxError)?;
        let dims = shape_dims_static(output_shape)?;
        Array::from_bytes(&f64_to_bytes(&out), &dims, DType::F64)
    }

    pub(super) struct SparseLuMlx;
    impl MlxKernel for SparseLuMlx {
        fn name(&self) -> &str {
            SPARSE_LU_SOLVE
        }
        fn execute(
            &self,
            inputs: &[&Array],
            out_shape: &rlx_ir::Shape,
            _attrs: &[u8],
        ) -> Result<Array, MlxError> {
            run_lu(inputs, out_shape)
        }
    }
    pub(super) struct SparseMatVecMlx;
    impl MlxKernel for SparseMatVecMlx {
        fn name(&self) -> &str {
            SPARSE_MAT_VEC
        }
        fn execute(
            &self,
            inputs: &[&Array],
            out_shape: &rlx_ir::Shape,
            _attrs: &[u8],
        ) -> Result<Array, MlxError> {
            run_mat_vec(inputs, out_shape)
        }
    }
    pub(super) struct SparseCgMlx;
    impl MlxKernel for SparseCgMlx {
        fn name(&self) -> &str {
            SPARSE_CG_SOLVE
        }
        fn execute(
            &self,
            inputs: &[&Array],
            out_shape: &rlx_ir::Shape,
            attrs: &[u8],
        ) -> Result<Array, MlxError> {
            run_cg(inputs, out_shape, attrs)
        }
    }

    fn run_values_grad(inputs: &[&Array], output_shape: &rlx_ir::Shape) -> Result<Array, MlxError> {
        let col_idx = bytes_to_i32(&inputs[0].to_bytes()?);
        let row_ptr = bytes_to_i32(&inputs[1].to_bytes()?);
        let u = bytes_to_f64(&inputs[2].to_bytes()?);
        let v = bytes_to_f64(&inputs[3].to_bytes()?);
        let mut out = vec![0f64; col_idx.len()];
        algos::values_grad(&col_idx, &row_ptr, &u, &v, &mut out).map_err(MlxError)?;
        let dims = shape_dims_static(output_shape)?;
        Array::from_bytes(&f64_to_bytes(&out), &dims, DType::F64)
    }

    fn run_lu_general(inputs: &[&Array], output_shape: &rlx_ir::Shape) -> Result<Array, MlxError> {
        // Forward reads only inputs 0..=3 (values_A, col_idx_A,
        // row_ptr_A, b); inputs 4..=6 are the AT triplet for the VJP.
        let values = bytes_to_f64(&inputs[0].to_bytes()?);
        let col_idx = bytes_to_i32(&inputs[1].to_bytes()?);
        let row_ptr = bytes_to_i32(&inputs[2].to_bytes()?);
        let b = bytes_to_f64(&inputs[3].to_bytes()?);
        let mut out = vec![0f64; b.len()];
        algos::lu_solve(&values, &col_idx, &row_ptr, &b, &mut out).map_err(MlxError)?;
        let dims = shape_dims_static(output_shape)?;
        Array::from_bytes(&f64_to_bytes(&out), &dims, DType::F64)
    }

    fn run_gmres(
        inputs: &[&Array],
        output_shape: &rlx_ir::Shape,
        attrs: &[u8],
    ) -> Result<Array, MlxError> {
        let (max_iter, tol) = decode_cg_attrs(attrs).map_err(MlxError)?;
        let values = bytes_to_f64(&inputs[0].to_bytes()?);
        let col_idx = bytes_to_i32(&inputs[1].to_bytes()?);
        let row_ptr = bytes_to_i32(&inputs[2].to_bytes()?);
        let b = bytes_to_f64(&inputs[3].to_bytes()?);
        let mut out = vec![0f64; b.len()];
        algos::gmres_solve(&values, &col_idx, &row_ptr, &b, &mut out, max_iter, tol)
            .map_err(MlxError)?;
        let dims = shape_dims_static(output_shape)?;
        Array::from_bytes(&f64_to_bytes(&out), &dims, DType::F64)
    }

    pub(super) struct SparseValuesGradMlx;
    impl MlxKernel for SparseValuesGradMlx {
        fn name(&self) -> &str {
            SPARSE_VALUES_GRAD
        }
        fn execute(
            &self,
            inputs: &[&Array],
            out_shape: &rlx_ir::Shape,
            _attrs: &[u8],
        ) -> Result<Array, MlxError> {
            run_values_grad(inputs, out_shape)
        }
    }
    pub(super) struct SparseLuGeneralMlx;
    impl MlxKernel for SparseLuGeneralMlx {
        fn name(&self) -> &str {
            SPARSE_LU_SOLVE_GENERAL
        }
        fn execute(
            &self,
            inputs: &[&Array],
            out_shape: &rlx_ir::Shape,
            _attrs: &[u8],
        ) -> Result<Array, MlxError> {
            run_lu_general(inputs, out_shape)
        }
    }
    pub(super) struct SparseGmresMlx;
    impl MlxKernel for SparseGmresMlx {
        fn name(&self) -> &str {
            SPARSE_GMRES_SOLVE
        }
        fn execute(
            &self,
            inputs: &[&Array],
            out_shape: &rlx_ir::Shape,
            attrs: &[u8],
        ) -> Result<Array, MlxError> {
            run_gmres(inputs, out_shape, attrs)
        }
    }
}

// ── Registration ──────────────────────────────────────────────────

/// Register every sparse op's IR-level extension and per-backend
/// kernels enabled at compile time. Idempotent — the underlying
/// registries already warn on overwrite. Call once at application
/// startup.
/// Host CG for SPD `A·x = b` given CSR `(values, col_idx, row_ptr)`.
pub fn cg_solve(
    values: &[f64],
    col_idx: &[i32],
    row_ptr: &[i32],
    b: &[f64],
    out: &mut [f64],
    max_iter: u32,
    tol: f64,
) -> Result<(), String> {
    algos::cg_solve(values, col_idx, row_ptr, b, out, max_iter, tol)
}

pub fn register() {
    register_op(Arc::new(SparseLuExt));
    register_op(Arc::new(SparseMatVecExt));
    register_op(Arc::new(SparseCgExt));
    register_op(Arc::new(SparseValuesGradExt));
    register_op(Arc::new(SparseLuGeneralExt));
    register_op(Arc::new(SparseGmresExt));
    register_op(Arc::new(SparseTransposeValuesExt));
    register_op(Arc::new(SparsePcgExt));
    register_op(Arc::new(SparseBicgstabExt));
    register_op(Arc::new(SparseIluPcgExt));
    register_op(Arc::new(SparseCholeskyExt));
    register_op(Arc::new(SparseLsqrExt));

    #[cfg(feature = "cpu")]
    {
        register_cpu_kernel(Arc::new(SparseLuCpu));
        register_cpu_kernel(Arc::new(SparseMatVecCpu));
        register_cpu_kernel(Arc::new(SparseCgCpu));
        register_cpu_kernel(Arc::new(SparseValuesGradCpu));
        register_cpu_kernel(Arc::new(SparseLuGeneralCpu));
        register_cpu_kernel(Arc::new(SparseGmresCpu));
        register_cpu_kernel(Arc::new(SparseTransposeValuesCpu));
        register_cpu_kernel(Arc::new(SparsePcgCpu));
        register_cpu_kernel(Arc::new(SparseBicgstabCpu));
        register_cpu_kernel(Arc::new(SparseIluPcgCpu));
        register_cpu_kernel(Arc::new(SparseCholeskyCpu));
        register_cpu_kernel(Arc::new(SparseLsqrCpu));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use rlx_metal::op_registry::register_metal_kernel;
        register_metal_kernel(Arc::new(metal_kernels::SparseLuMetal));
        register_metal_kernel(Arc::new(metal_kernels::SparseMatVecMetal));
        register_metal_kernel(Arc::new(metal_kernels::SparseCgMetal));
        register_metal_kernel(Arc::new(metal_kernels::SparseValuesGradMetal));
        register_metal_kernel(Arc::new(metal_kernels::SparseLuGeneralMetal));
        register_metal_kernel(Arc::new(metal_kernels::SparseGmresMetal));
    }

    #[cfg(all(feature = "mlx", target_os = "macos"))]
    {
        use rlx_mlx::op_registry::register_mlx_kernel;
        register_mlx_kernel(Arc::new(mlx_kernels::SparseLuMlx));
        register_mlx_kernel(Arc::new(mlx_kernels::SparseMatVecMlx));
        register_mlx_kernel(Arc::new(mlx_kernels::SparseCgMlx));
        register_mlx_kernel(Arc::new(mlx_kernels::SparseValuesGradMlx));
        register_mlx_kernel(Arc::new(mlx_kernels::SparseLuGeneralMlx));
        register_mlx_kernel(Arc::new(mlx_kernels::SparseGmresMlx));
    }
}
