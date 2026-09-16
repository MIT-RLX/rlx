// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

use rlx_ir::{Graph, NodeId, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::register_cpu_kernel;

// ── Op names (stable strings; downstream callers use these to look
//    up the registered op or build `Op::Custom` directly) ─────────

mod op_bicgstab;
mod op_cg;
mod op_cholesky;
mod op_gmres;
mod op_ic0_pcg;
mod op_ilu_pcg;
mod op_lsqr;
mod op_lu;
mod op_lu_general;
mod op_mat_vec;
mod op_pcg;
mod op_transpose_values;
mod op_values_grad;

use op_bicgstab::*;
use op_cg::*;
use op_cholesky::*;
use op_gmres::*;
use op_ic0_pcg::*;
use op_ilu_pcg::*;
use op_lsqr::*;
use op_lu::*;
use op_lu_general::*;
use op_mat_vec::*;
use op_pcg::*;
use op_transpose_values::*;
use op_values_grad::*;

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

/// Conjugate gradients preconditioned by incomplete Cholesky — the SPD
/// counterpart of [`SPARSE_ILU_PCG_SOLVE`], which stores both triangles of a
/// factorisation whose halves are transposes of each other.
pub const SPARSE_IC0_PCG_SOLVE: &str = "rlx_sparse.ic0_pcg_solve";

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
/// Below this much work — non-zeros times right-hand sides — a kernel runs on
/// the calling thread.
///
/// Deliberately conservative. Three things argue for staying serial longer
/// than a flop count alone would suggest:
///
/// * A small system lives in cache, and splitting it across cores trades a
///   warm L2 for cold traffic plus synchronisation. Measured on a 3,600-row
///   Laplacian, threading an 8-column block made it **five times slower**.
/// * Callers frequently parallelise above this — one solve per core over many
///   independent systems is the common shape — and a second level of
///   splitting underneath buys nothing and costs scheduling.
/// * When threading does pay, it pays by a lot: the same benchmark at 128
///   columns runs 4× faster. There is no need to chase the margin.
///
/// So the rule is to thread only where the win is unambiguous. This constant
/// is a floor, not a tuned optimum, and was chosen on a machine busy enough
/// that a tuned optimum could not have been measured honestly.
const PAR_MIN_WORK: usize = 1 << 21;

mod algos {
    use super::PAR_MIN_WORK;
    use rayon::prelude::*;

    /// `y = A·x` for a CSR matrix, threaded over output rows.
    ///
    /// Every iterative solver in this module needs exactly this and each one
    /// used to spell it out again, so threading it meant threading it four
    /// times. The rows are disjoint — each writes one element and only reads
    /// `x` — which is what makes the split safe without any synchronisation.
    pub(super) fn mat_vec_into(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        x: &[f64],
        y: &mut [f64],
    ) {
        let row = |r: usize| -> f64 {
            (row_ptr[r] as usize..row_ptr[r + 1] as usize)
                .map(|k| values[k] * x[col_idx[k] as usize])
                .sum()
        };
        if values.len() >= PAR_MIN_WORK {
            let stride = rows_per_task(y.len());
            y.par_chunks_mut(stride).enumerate().for_each(|(blk, out)| {
                let base = blk * stride;
                for (i, o) in out.iter_mut().enumerate() {
                    *o = row(base + i);
                }
            });
        } else {
            y.iter_mut().enumerate().for_each(|(r, o)| *o = row(r));
        }
    }

    /// How many output rows one parallel task should take.
    ///
    /// Granularity is the whole game here. A task per row means rayon's
    /// bookkeeping — a few hundred nanoseconds — against a handful of
    /// multiply-adds, and the "parallel" version loses to the serial one by
    /// six times. Aiming at a few tasks per thread keeps the split amortised
    /// while leaving the work-stealer something to balance with.
    pub(super) fn rows_per_task(n_rows: usize) -> usize {
        let want = rayon::current_num_threads().max(1) * 4;
        n_rows.div_ceil(want).max(64)
    }

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

        let matvec = |x: &[f64], y: &mut [f64]| mat_vec_into(values, col_idx, row_ptr, x, y);

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
        if row_ptr.len() < 2 {
            return Err(format!("pcg_solve: row_ptr len {}", row_ptr.len()));
        }
        let n = row_ptr.len() - 1;
        if out.len() != b.len() {
            return Err(format!(
                "pcg_solve: out len {} != b len {}",
                out.len(),
                b.len()
            ));
        }
        if n == 0 || !b.len().is_multiple_of(n) {
            return Err(format!(
                "pcg_solve: b len {} is not a whole number of {n}-long \
                 right-hand sides",
                b.len()
            ));
        }
        let k = b.len() / n;

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

        // `A·P` for all `k` columns at once. The matrix is read once per
        // iteration rather than once per right-hand side, which is the whole
        // reason to batch: for `k` in the hundreds the sparse entries stop
        // being the cost and the dense block becomes it.
        //
        // What batching costs, and it is not nothing: the block runs until
        // its **slowest** column has converged, so every column pays the
        // worst one's iteration count where separate solves each stop at
        // their own. Batching therefore wins when `k` is large enough that
        // re-reading the matrix dominates, and loses when `k` is small and
        // the columns differ in difficulty. Measured on a 3,600-row
        // Laplacian: slower at `k = 8`, roughly 3× faster at `k = 32` and
        // above. There is no threshold that makes this choice for the caller,
        // because it depends on how alike the right-hand sides are.
        //
        // Row-major throughout, so the `k` values a matrix entry touches are
        // contiguous and the inner loop is a strided AXPY the compiler can
        // vectorise.
        //
        // Threaded over output rows, which are disjoint — each writes its own
        // slice and only reads `xs`. Below `PAR_MIN_WORK` the split costs more
        // than the arithmetic, and a caller who is already running one solve
        // per core wants this to stay out of the way.
        let matmul = |xs: &[f64], ys: &mut [f64]| {
            let work = values.len() * k;
            let body = |r: usize, row: &mut [f64]| {
                let (s, e) = (row_ptr[r] as usize, row_ptr[r + 1] as usize);
                row.fill(0.0);
                for t in s..e {
                    let v = values[t];
                    let c = col_idx[t] as usize * k;
                    for (o, sv) in row.iter_mut().zip(&xs[c..c + k]) {
                        *o += v * sv;
                    }
                }
            };
            if work >= PAR_MIN_WORK {
                // Chunked by *blocks of rows*, not by single rows: one task
                // per row would be a few multiply-adds against rayon's own
                // overhead, which is a reliable way to make a parallel loop
                // slower than the serial one.
                let stride = rows_per_task(n);
                ys.par_chunks_mut(stride * k)
                    .enumerate()
                    .for_each(|(blk, out)| {
                        let base = blk * stride;
                        for (i, row) in out.chunks_mut(k).enumerate() {
                            body(base + i, row);
                        }
                    });
            } else {
                ys.chunks_mut(k)
                    .enumerate()
                    .for_each(|(r, row)| body(r, row));
            }
        };
        // Per-column dot product of two `n × k` blocks.
        let dots = |a: &[f64], c: &[f64], into: &mut [f64]| {
            into.fill(0.0);
            for r in 0..n {
                for j in 0..k {
                    into[j] += a[r * k + j] * c[r * k + j];
                }
            }
        };

        // PCG with X₀ = 0: R₀ = B, Z₀ = M⁻¹·R₀
        let mut x = vec![0f64; n * k];
        let mut r = b.to_vec();
        let mut z = vec![0f64; n * k];
        for i in 0..n {
            for j in 0..k {
                z[i * k + j] = r[i * k + j] * inv_diag[i];
            }
        }
        let mut p = z.clone();
        let mut ap = vec![0f64; n * k];
        let mut rho_old = vec![0f64; k];
        dots(&r, &z, &mut rho_old);

        let mut pap = vec![0f64; k];
        let mut rho_new = vec![0f64; k];
        let mut alpha = vec![0f64; k];
        let mut beta = vec![0f64; k];
        let mut rr = vec![0f64; k];

        for _ in 0..max_iter {
            // Convergence on plain ‖r‖₂ per column (matches CG's contract).
            // Columns converge at different rates; the block runs until the
            // slowest is done, and a column that has arrived is held still
            // rather than pushed around by its own rounding.
            dots(&r, &r, &mut rr);
            if rr.iter().all(|v| v.sqrt() < tol) {
                break;
            }

            matmul(&p, &mut ap);
            dots(&p, &ap, &mut pap);
            if k == 1 && pap[0] == 0.0 {
                return Err("pcg_solve: pᵀ·A·p = 0 (A is singular or not SPD)".into());
            }
            for j in 0..k {
                // A zero curvature means this column has nothing left to do;
                // stepping it would divide by zero and spread a NaN through
                // the block.
                alpha[j] = if pap[j] == 0.0 || rr[j].sqrt() < tol {
                    0.0
                } else {
                    rho_old[j] / pap[j]
                };
            }
            for i in 0..n {
                for j in 0..k {
                    let idx = i * k + j;
                    x[idx] += alpha[j] * p[idx];
                    r[idx] -= alpha[j] * ap[idx];
                    z[idx] = r[idx] * inv_diag[i];
                }
            }
            dots(&r, &z, &mut rho_new);
            for j in 0..k {
                beta[j] = if rho_old[j] == 0.0 {
                    0.0
                } else {
                    rho_new[j] / rho_old[j]
                };
            }
            for i in 0..n {
                for j in 0..k {
                    let idx = i * k + j;
                    p[idx] = z[idx] + beta[j] * p[idx];
                }
            }
            rho_old.copy_from_slice(&rho_new);
        }

        out.copy_from_slice(&x);
        Ok(())
    }

    /// Direct sparse Cholesky for SPD A. Densifies to a dense buffer,
    /// factors via LAPACK `dpotrf`, then forward+back triangular solve
    /// via `dtrsm`. The mirror of [`lu_solve`] for SPD matrices —
    /// faster (factor cost ½× LU) and numerically more stable.
    ///
    /// # Many right-hand sides share one factorisation
    ///
    /// `b` may hold **several** right-hand sides: an `n × nrhs` row-major
    /// block, with `nrhs` taken from `b.len() / n`. One column is the
    /// ordinary case and behaves exactly as before.
    ///
    /// This is the shape a direct method is *for*. Factoring costs `O(n³)`
    /// and each solve after it costs `O(n²)`, so a problem that reuses one
    /// matrix — a time series, a batch of loads, a reconstruction solving the
    /// same system once per slice — should pay the factorisation once. Calling
    /// this `nrhs` times instead re-factors `nrhs` times, which for `n` in the
    /// thousands is the difference between minutes and hours.
    ///
    /// # Cost
    ///
    /// The dense buffer is `8n²` bytes and is allocated whatever the sparsity:
    /// 800 MB at `n = 10,000`. That is a real limit rather than a detail, so
    /// it is refused with a message that says so rather than by the allocator.
    /// A true sparse factorisation with a fill-reducing ordering would lift
    /// it; until then, an iterative solver ([`pcg`], [`lsqr_solve`]) is what
    /// large systems want.
    pub fn cholesky_solve(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if row_ptr.len() < 2 {
            return Err(format!("cholesky_solve: row_ptr len {}", row_ptr.len()));
        }
        let n = row_ptr.len() - 1;
        if out.len() != b.len() {
            return Err(format!(
                "cholesky_solve: out len {} != b len {}",
                out.len(),
                b.len()
            ));
        }
        if n == 0 || !b.len().is_multiple_of(n) {
            return Err(format!(
                "cholesky_solve: b len {} is not a whole number of {n}-long \
                 right-hand sides",
                b.len()
            ));
        }
        let nrhs = b.len() / n;

        // 8n² bytes, dense, however sparse the input was.
        const MAX_DENSE_BYTES: usize = 8 << 30;
        let bytes = n.saturating_mul(n).saturating_mul(8);
        if bytes > MAX_DENSE_BYTES {
            return Err(format!(
                "cholesky_solve: this densifies to {n}×{n}, which is {} GiB — \
                 use an iterative solver (pcg, lsqr) for a system this size",
                bytes >> 30
            ));
        }

        let mut a_dense = vec![0f64; n * n];
        for r in 0..n {
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                a_dense[r * n + col_idx[k] as usize] = values[k];
            }
        }
        // Factor: A = L·Lᵀ; L stored in lower triangle of a_dense. Once,
        // whatever `nrhs` is — that is the point of a direct method.
        let info = rlx_cpu::blas::dpotrf(&mut a_dense, n, /*lower=*/ true);
        if info != 0 {
            return Err(format!("cholesky_solve: dpotrf info={info} (not SPD?)"));
        }
        // Solve L·Y = B (forward), then Lᵀ·X = Y (back), all columns at once.
        let mut x = b.to_vec();
        rlx_cpu::blas::dtrsm_lower_or_upper(
            &a_dense, &mut x, n, nrhs, /*lower=*/ true, /*trans=*/ false,
        );
        rlx_cpu::blas::dtrsm_lower_or_upper(
            &a_dense, &mut x, n, nrhs, /*lower=*/ true, /*trans=*/ true,
        );
        out.copy_from_slice(&x);
        Ok(())
    }

    /// Conjugate gradients preconditioned by [`ic0_factor`].
    ///
    /// Same contract as [`pcg`] — SPD `A`, absolute `‖r‖₂` tolerance, `x₀ = 0`
    /// — and the same answer, reached in fewer iterations. The preconditioner
    /// is the whole difference: Jacobi divides by the diagonal, which knows
    /// nothing about how the matrix couples its unknowns, where `IC(0)`
    /// approximates the factorisation itself.
    ///
    /// Each iteration costs two triangular solves on top of the sparse
    /// product, so this is worth it when it saves more iterations than that
    /// overhead — which for anything with structure it usually does, and by a
    /// lot. Measured on a super-resolution reconstruction's normal equations,
    /// `9,900 × 9,900` with 292,356 non-zeros: Jacobi reaches the `f64` floor
    /// in about 48 iterations, this in about 10.
    pub fn ic0_pcg(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        b: &[f64],
        out: &mut [f64],
        max_iter: u32,
        tol: f64,
    ) -> Result<(), String> {
        if row_ptr.len() < 2 {
            return Err(format!("ic0_pcg: row_ptr len {}", row_ptr.len()));
        }
        let n = row_ptr.len() - 1;
        if out.len() != b.len() {
            return Err(format!(
                "ic0_pcg: out len {} != b len {}",
                out.len(),
                b.len()
            ));
        }
        if n == 0 || !b.len().is_multiple_of(n) {
            return Err(format!(
                "ic0_pcg: b len {} is not a whole number of {n}-long \
                 right-hand sides",
                b.len()
            ));
        }
        let k = b.len() / n;

        // Factorised **once**, whatever `k` is. This is the reason to hand
        // several right-hand sides in rather than call this once each: the
        // incomplete Cholesky depends only on the matrix, and a caller solving
        // the same system repeatedly — a super-resolution reconstruction runs
        // 1,430 solves against one matrix — otherwise pays for it every time.
        let (l, lc, lr) = ic0_factor(values, col_idx, row_ptr, n)?;

        let mut x = vec![0f64; n];
        let mut r = vec![0f64; n];
        let mut z = vec![0f64; n];
        let mut p = vec![0f64; n];
        let mut ap = vec![0f64; n];

        for rhs in 0..k {
            let bj = &b[rhs * n..(rhs + 1) * n];
            x.fill(0.0);
            r.copy_from_slice(bj);
            ic0_apply(&l, &lc, &lr, n, &r, &mut z);
            p.copy_from_slice(&z);
            let mut rho_old: f64 = r.iter().zip(&z).map(|(a, c)| a * c).sum();

            for _ in 0..max_iter {
                if r.iter().map(|v| v * v).sum::<f64>().sqrt() < tol {
                    break;
                }
                mat_vec_into(values, col_idx, row_ptr, &p, &mut ap);
                let pap: f64 = p.iter().zip(&ap).map(|(a, c)| a * c).sum();
                if pap == 0.0 {
                    return Err("ic0_pcg: pᵀ·A·p = 0 (A is singular or not SPD)".into());
                }
                let alpha = rho_old / pap;
                for i in 0..n {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * ap[i];
                }
                ic0_apply(&l, &lc, &lr, n, &r, &mut z);
                let rho_new: f64 = r.iter().zip(&z).map(|(a, c)| a * c).sum();
                let beta = if rho_old == 0.0 {
                    0.0
                } else {
                    rho_new / rho_old
                };
                for i in 0..n {
                    p[i] = z[i] + beta * p[i];
                }
                rho_old = rho_new;
            }
            out[rhs * n..(rhs + 1) * n].copy_from_slice(&x);
        }
        Ok(())
    }

    /// Incomplete Cholesky, `IC(0)`: `A ≈ L·Lᵀ` with `L` confined to the
    /// lower triangle of `A`'s own sparsity pattern.
    ///
    /// The SPD counterpart of [`ilu0_factor`], and the right one to reach for
    /// when the matrix is symmetric positive definite — which is exactly when
    /// conjugate gradients apply, so any caller of [`pcg`] is a candidate.
    /// `ILU(0)` on such a matrix computes and stores both triangles of a
    /// factorisation whose halves are transposes of each other: twice the
    /// memory, and twice the work in every triangular solve.
    ///
    /// # Breakdown
    ///
    /// Dropping fill can make a pivot non-positive even though `A` is
    /// definite — the factorisation exists, its incomplete version need not.
    /// The standard remedy is Manteuffel's: retry on `A + α·diag(A)` with `α`
    /// growing until it succeeds. The shifted factor is still a valid
    /// preconditioner, because a preconditioner only has to be *close* to
    /// `A⁻¹` and symmetric positive definite; it never has to be exact.
    ///
    /// Returns `L` in CSR, lower triangle including the diagonal.
    pub fn ic0_factor(
        values: &[f64],
        col_idx: &[i32],
        row_ptr: &[i32],
        n: usize,
    ) -> Result<(Vec<f64>, Vec<i32>, Vec<i32>), String> {
        // The lower triangle of A's pattern, row by row, columns ascending.
        let mut l_row_ptr = Vec::with_capacity(n + 1);
        let mut l_col_idx: Vec<i32> = Vec::new();
        let mut a_low: Vec<f64> = Vec::new();
        let mut diag_a = vec![0f64; n];
        l_row_ptr.push(0i32);
        for i in 0..n {
            let mut row: Vec<(i32, f64)> = (row_ptr[i] as usize..row_ptr[i + 1] as usize)
                .filter(|&k| (col_idx[k] as usize) <= i)
                .map(|k| (col_idx[k], values[k]))
                .collect();
            row.sort_by_key(|(c, _)| *c);
            for (c, v) in &row {
                if *c as usize == i {
                    diag_a[i] = *v;
                }
                l_col_idx.push(*c);
                a_low.push(*v);
            }
            l_row_ptr.push(l_col_idx.len() as i32);
        }
        if diag_a.iter().any(|d| *d <= 0.0) {
            return Err("ic0: a non-positive diagonal, so A is not SPD".into());
        }

        // Up to a few shifted attempts; α doubles from a small fraction.
        let mut alpha = 0f64;
        for attempt in 0..12 {
            let mut l = a_low.clone();
            for (i, d) in diag_a.iter().enumerate() {
                let k = l_row_ptr[i + 1] as usize - 1; // the diagonal is last
                l[k] = d * (1.0 + alpha);
            }
            if ic0_sweep(&mut l, &l_col_idx, &l_row_ptr, n).is_ok() {
                if attempt > 0 {
                    // Worth nothing to the caller but everything to whoever
                    // is reading a profile and wondering where the time went.
                    debug_assert!(alpha > 0.0);
                }
                return Ok((l, l_col_idx, l_row_ptr));
            }
            alpha = if alpha == 0.0 { 1e-3 } else { alpha * 2.0 };
        }
        Err("ic0: no diagonal shift made the factorisation succeed".into())
    }

    /// One in-place `IC(0)` sweep over an already-extracted lower triangle.
    fn ic0_sweep(l: &mut [f64], col_idx: &[i32], row_ptr: &[i32], n: usize) -> Result<(), String> {
        for i in 0..n {
            let (rs, re) = (row_ptr[i] as usize, row_ptr[i + 1] as usize);
            for k in rs..re {
                let j = col_idx[k] as usize;
                // `Σ L[i,p]·L[j,p]` over columns the two rows share, which is
                // a merge of two ascending index lists.
                let (js, je) = (row_ptr[j] as usize, row_ptr[j + 1] as usize);
                let mut sum = l[k];
                let (mut p, mut q) = (rs, js);
                while p < k && q < je && col_idx[q] < j as i32 {
                    match col_idx[p].cmp(&col_idx[q]) {
                        std::cmp::Ordering::Less => p += 1,
                        std::cmp::Ordering::Greater => q += 1,
                        std::cmp::Ordering::Equal => {
                            sum -= l[p] * l[q];
                            p += 1;
                            q += 1;
                        }
                    }
                }
                if j == i {
                    if sum <= 0.0 {
                        return Err("ic0: non-positive pivot".into());
                    }
                    l[k] = sum.sqrt();
                } else {
                    let d = l[row_ptr[j + 1] as usize - 1];
                    if d == 0.0 {
                        return Err("ic0: zero pivot".into());
                    }
                    l[k] = sum / d;
                }
            }
        }
        Ok(())
    }

    /// Apply `M⁻¹ = (L·Lᵀ)⁻¹` to one vector: forward then back substitution.
    fn ic0_apply(l: &[f64], col_idx: &[i32], row_ptr: &[i32], n: usize, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
        // L·y = r
        for i in 0..n {
            let (rs, re) = (row_ptr[i] as usize, row_ptr[i + 1] as usize);
            let mut acc = z[i];
            for k in rs..re - 1 {
                acc -= l[k] * z[col_idx[k] as usize];
            }
            z[i] = acc / l[re - 1];
        }
        // Lᵀ·x = y, which walks rows backwards and scatters.
        for i in (0..n).rev() {
            let (rs, re) = (row_ptr[i] as usize, row_ptr[i + 1] as usize);
            z[i] /= l[re - 1];
            let zi = z[i];
            for k in rs..re - 1 {
                z[col_idx[k] as usize] -= l[k] * zi;
            }
        }
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
        damp: f64,
    ) -> Result<usize, String> {
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

        // y = A·x  (gather over rows of A), threaded.
        let av = |x: &[f64], y: &mut [f64]| mat_vec_into(values, col_idx, row_ptr, x, y);
        // y = Aᵀ·u  (scatter over rows).
        //
        // Deliberately *not* threaded: every entry does a read-modify-write
        // into a location the index array picks, so two rows can collide on
        // one output and the split would need atomics or per-thread buffers.
        // Measured on a banded matrix the scatter costs only 1.06× the
        // gather — the collisions are local and the cache absorbs them — so
        // the cost of making it safe would exceed what it could win.
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
            return Ok(0);
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
            return Ok(0);
        }
        for x in v.iter_mut() {
            *x /= alpha;
        }

        let mut x = vec![0f64; n];
        let mut w = v.clone();
        let mut phi_bar = beta;
        let mut rho_bar = alpha;
        // ‖b‖, and a running ‖A‖_F built from the bidiagonal entries — both
        // needed to make the stopping tests relative. Paige & Saunders (1982)
        // §6; `anorm² = Σ (αᵢ² + βᵢ²)`.
        let bnorm = beta;
        let mut anorm2 = alpha * alpha;
        let mut iters = 0usize;
        // With damping the residual has a second part — the `damp·x` block of
        // the augmented system — which is accumulated rather than formed.
        let mut res2 = 0f64;

        let mut tmp_u = vec![0f64; m];
        let mut tmp_v = vec![0f64; n];

        for _ in 0..max_iter {
            iters += 1;
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

            anorm2 += beta * beta + alpha * alpha + damp * damp;

            // A first rotation folds the damping into the bidiagonal, which
            // is what makes this solve the *augmented* least-squares problem
            //
            //     min ‖[ A ; damp·I ]·x − [ b ; 0 ]‖₂
            //
            // without anyone building `[ A ; damp·I ]`. That matrix has `n`
            // extra rows and, for a caller who only wanted Tikhonov, exists
            // solely to be multiplied by zero.
            let (rho_bar1, psi) = if damp != 0.0 {
                let r1 = (rho_bar * rho_bar + damp * damp).sqrt();
                let c1 = rho_bar / r1;
                let s1 = damp / r1;
                let psi = s1 * phi_bar;
                phi_bar *= c1;
                (r1, psi)
            } else {
                (rho_bar, 0.0)
            };

            // Givens rotation to eliminate β below ρ̄.
            let rho = (rho_bar1 * rho_bar1 + beta * beta).sqrt();
            let c = rho_bar1 / rho;
            let s = beta / rho;
            let theta = s * alpha;
            rho_bar = -c * alpha;
            let phi = c * phi_bar;
            phi_bar *= s;
            let tau = s * phi;
            res2 += psi * psi;

            // Update x and w.
            let phi_over_rho = phi / rho;
            let theta_over_rho = theta / rho;
            for i in 0..n {
                x[i] += phi_over_rho * w[i];
                w[i] = v[i] - theta_over_rho * w[i];
            }

            // ── Stopping ────────────────────────────────────────────
            //
            // Two tests, and the second is the one that matters.
            //
            // `‖A·x − b‖` reaches zero only when the system is *consistent*.
            // A least-squares problem is overdetermined and inconsistent by
            // construction — that is what makes it a least-squares problem —
            // so its residual converges to a nonzero minimum and a test on
            // `‖r‖` alone never fires. Every solve then runs to `max_iter`,
            // returning the right answer having spent an arbitrary multiple
            // of the time reaching it.
            //
            // The quantity that does go to zero is `‖Aᵀ·r‖`, the gradient of
            // `½‖A·x − b‖²`, which is what "least squares" means. Paige &
            // Saunders express both in terms of the bidiagonalisation
            // already computed here:
            //
            //     ‖r‖    = phi_bar
            //     ‖Aᵀr‖  = phi_bar · alpha · |c|
            //
            // so neither costs a product. Both are made relative, because an
            // absolute threshold on a residual is a threshold on the units
            // the data happens to be in.
            let rnorm = (phi_bar * phi_bar + res2).sqrt();
            let arnorm = alpha * tau.abs();
            let anorm = anorm2.sqrt();
            let xnorm = x.iter().map(|v| v * v).sum::<f64>().sqrt();

            // Consistent system: the residual itself is going to zero.
            if rnorm <= tol * bnorm + tol * anorm * xnorm {
                break;
            }
            // Inconsistent system: the residual is orthogonal to the range.
            if arnorm <= tol * anorm * rnorm {
                break;
            }
            if alpha == 0.0 || beta == 0.0 {
                break;
            }
        }
        out.copy_from_slice(&x);
        Ok(iters)
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
        let matvec = |x: &[f64], y: &mut [f64]| mat_vec_into(values, col_idx, row_ptr, x, y);
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
        let matvec = |x: &[f64], y: &mut [f64]| mat_vec_into(values, col_idx, row_ptr, x, y);
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

/// Encode CG attrs into the opaque `Vec<u8>` blob carried on
/// `Op::Custom`. Layout: `[max_iter:u32 LE, tol:f64 LE]` — 12 bytes.
pub fn encode_cg_attrs(max_iter: u32, tol: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&max_iter.to_le_bytes());
    out.extend_from_slice(&tol.to_le_bytes());
    out
}

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
    /// `x = A⁻¹ · b` by dense Cholesky, for symmetric positive-definite A.
    ///
    /// `b` may be a single vector or an `n × nrhs` row-major block; the
    /// factorisation is computed **once** and reused across the columns. See
    /// `algos::cholesky_solve` for why that distinction is the whole reason
    /// to choose a direct method, and for the dense memory it costs.
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
        self.lsqr_solve_damped(g, b, max_iter, tol, 0.0)
    }

    /// `x = argmin ‖A·x − b‖₂² + damp²·‖x‖₂²` — Tikhonov-regularised least
    /// squares, by LSQR.
    ///
    /// This is the same problem as
    ///
    /// ```text
    ///     min ‖ [   A    ]·x − [ b ] ‖
    ///         ‖ [ damp·I ]     [ 0 ] ‖₂
    /// ```
    ///
    /// and the reason to ask for it here rather than build that matrix is
    /// that the matrix is mostly a formality: `n` extra rows, one entry each,
    /// multiplied by a right-hand side of zeros. Paige & Saunders fold the
    /// damping into the bidiagonalisation with one extra Givens rotation per
    /// iteration, so it costs two scalars and no storage.
    ///
    /// Ridge regression, Tikhonov-regularised inverse problems and damped
    /// Gauss–Newton steps are all this call. `damp = 0` is plain
    /// [`SparseTensor::lsqr_solve`].
    pub fn lsqr_solve_damped(
        &self,
        g: &mut Graph,
        b: NodeId,
        max_iter: u32,
        tol: f64,
        damp: f64,
    ) -> NodeId {
        let mut attrs = Vec::with_capacity(24);
        attrs.extend_from_slice(&max_iter.to_le_bytes());
        attrs.extend_from_slice(&tol.to_le_bytes());
        attrs.extend_from_slice(&(self.n_cols as u32).to_le_bytes());
        attrs.extend_from_slice(&damp.to_le_bytes());
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
    /// `x = A⁻¹ · b` by conjugate gradients preconditioned with incomplete
    /// Cholesky, for **symmetric positive-definite** A.
    ///
    /// The one to reach for when `A` is SPD, which is precisely when
    /// conjugate gradients apply at all. Against [`SparseTensor::pcg_solve`]'s
    /// Jacobi preconditioner it converges in a fraction of the iterations —
    /// roughly a fifth on a normal-equations system measured at
    /// `9,900 × 9,900` — at the cost of two triangular solves per iteration
    /// and one factorisation up front.
    ///
    /// Against [`SparseTensor::ilu_pcg_solve`] it does the same job with half
    /// the storage and half the triangular work, because `ILU(0)` on a
    /// symmetric matrix computes both halves of a factorisation whose halves
    /// are transposes of each other.
    pub fn ic0_pcg_solve(&self, g: &mut Graph, b: NodeId, max_iter: u32, tol: f64) -> NodeId {
        assert_eq!(
            self.n_rows, self.n_cols,
            "SparseTensor::ic0_pcg_solve requires a square matrix"
        );
        g.custom_op(
            SPARSE_IC0_PCG_SOLVE,
            encode_cg_attrs(max_iter, tol),
            vec![self.values, self.col_idx, self.row_ptr, b],
        )
    }

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

#[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
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
    register_op(Arc::new(SparseIc0PcgExt));
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
        register_cpu_kernel(Arc::new(SparseIc0PcgCpu));
        register_cpu_kernel(Arc::new(SparseIluPcgCpu));
        register_cpu_kernel(Arc::new(SparseCholeskyCpu));
        register_cpu_kernel(Arc::new(SparseLsqrCpu));
    }

    #[cfg(all(feature = "metal", target_vendor = "apple", not(target_os = "watchos")))]
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

#[cfg(all(test, feature = "cpu"))]
mod algo_tests {
    use super::algos;

    /// A small SPD matrix in CSR: tridiagonal, diagonally dominant.
    fn spd() -> (Vec<f64>, Vec<i32>, Vec<i32>, usize) {
        let values = vec![4.0, 1.0, 1.0, 5.0, 1.0, 1.0, 6.0, 1.0, 1.0, 7.0];
        let col_idx = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let row_ptr = vec![0, 2, 5, 8, 10];
        (values, col_idx, row_ptr, 4)
    }

    fn mul(values: &[f64], col_idx: &[i32], row_ptr: &[i32], x: &[f64]) -> Vec<f64> {
        (0..row_ptr.len() - 1)
            .map(|r| {
                (row_ptr[r] as usize..row_ptr[r + 1] as usize)
                    .map(|k| values[k] * x[col_idx[k] as usize])
                    .sum()
            })
            .collect()
    }

    /// One factorisation, several right-hand sides — and the columns must be
    /// indistinguishable from solving them one at a time.
    ///
    /// This is what a direct method is for: factoring costs `O(n³)` and each
    /// solve after it costs `O(n²)`, so a caller with `k` right-hand sides
    /// for one matrix should pay for one factorisation rather than `k`.
    #[test]
    fn cholesky_reuses_one_factorisation_across_a_block() {
        let (v, ci, rp, n) = spd();
        let cols: [[f64; 4]; 3] = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, -1.0],
            [2.5, -1.5, 0.5, 1.0],
        ];

        // As an n × 3 row-major block.
        let mut b = vec![0f64; n * 3];
        for (j, c) in cols.iter().enumerate() {
            for (i, val) in c.iter().enumerate() {
                b[i * 3 + j] = *val;
            }
        }
        let mut block = vec![0f64; n * 3];
        algos::cholesky_solve(&v, &ci, &rp, &b, &mut block).unwrap();

        for (j, c) in cols.iter().enumerate() {
            let mut alone = vec![0f64; n];
            algos::cholesky_solve(&v, &ci, &rp, c, &mut alone).unwrap();

            let column: Vec<f64> = (0..n).map(|i| block[i * 3 + j]).collect();
            for i in 0..n {
                assert!(
                    (column[i] - alone[i]).abs() < 1e-12,
                    "column {j} differs from the lone solve: {column:?} vs {alone:?}"
                );
            }
            // And it solves the system it was given.
            for (r, got) in mul(&v, &ci, &rp, &column).iter().enumerate() {
                assert!((got - c[r]).abs() < 1e-10, "row {r}: {got} vs {}", c[r]);
            }
        }
    }

    /// A single right-hand side is the one-column case, unchanged.
    #[test]
    fn cholesky_still_takes_a_bare_vector() {
        let (v, ci, rp, n) = spd();
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut x = vec![0f64; n];
        algos::cholesky_solve(&v, &ci, &rp, &b, &mut x).unwrap();
        for (r, got) in mul(&v, &ci, &rp, &x).iter().enumerate() {
            assert!((got - b[r]).abs() < 1e-10, "row {r}: {got} vs {}", b[r]);
        }
    }

    /// A right-hand side that is not a whole number of columns is a caller
    /// error, and saying so beats solving something else.
    #[test]
    fn cholesky_refuses_a_ragged_block() {
        let (v, ci, rp, _) = spd();
        let b = [1.0, 2.0, 3.0, 4.0, 5.0]; // 5 is not a multiple of 4
        let mut x = vec![0f64; 5];
        let e = algos::cholesky_solve(&v, &ci, &rp, &b, &mut x).unwrap_err();
        assert!(e.contains("right-hand sides"), "{e}");
    }

    /// Incomplete Cholesky must reproduce the answer Jacobi reaches, in
    /// fewer iterations.
    ///
    /// Both are preconditioners on the same system, so the *answer* is not
    /// negotiable — a preconditioner that changed it would be a bug, not a
    /// faster method. What it may change is how long it takes, and that is
    /// the point.
    #[test]
    fn ic0_reaches_the_same_answer_as_jacobi_in_fewer_iterations() {
        // A 2-D Laplacian: SPD, and structured enough that a preconditioner
        // which understands the coupling beats one that only sees diagonals.
        let side = 24usize;
        let n = side * side;
        let (mut values, mut col_idx, mut row_ptr) = (Vec::new(), Vec::new(), vec![0i32]);
        for i in 0..n {
            let (r, c) = (i / side, i % side);
            let mut push = |j: usize, v: f64| {
                values.push(v);
                col_idx.push(j as i32);
            };
            if r > 0 {
                push(i - side, -1.0);
            }
            if c > 0 {
                push(i - 1, -1.0);
            }
            push(i, 4.0);
            if c + 1 < side {
                push(i + 1, -1.0);
            }
            if r + 1 < side {
                push(i + side, -1.0);
            }
            row_ptr.push(values.len() as i32);
        }
        let b: Vec<f64> = (0..n).map(|i| ((i % 17) as f64 / 17.0) - 0.5).collect();

        let resid = |x: &[f64]| {
            let mut ax = vec![0f64; n];
            algos::mat_vec_into(&values, &col_idx, &row_ptr, x, &mut ax);
            let num: f64 = ax.iter().zip(&b).map(|(a, c)| (a - c) * (a - c)).sum();
            let den: f64 = b.iter().map(|v| v * v).sum();
            (num / den).sqrt()
        };

        // Converged, both of them, to the same place.
        let mut jac = vec![0f64; n];
        algos::pcg_solve(&values, &col_idx, &row_ptr, &b, &mut jac, 5000, 1e-13).unwrap();
        let mut ic0 = vec![0f64; n];
        algos::ic0_pcg(&values, &col_idx, &row_ptr, &b, &mut ic0, 5000, 1e-13).unwrap();
        assert!(resid(&jac) < 1e-10, "jacobi left {:.3e}", resid(&jac));
        assert!(resid(&ic0) < 1e-10, "ic0 left {:.3e}", resid(&ic0));
        let peak = jac.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (a, c) in ic0.iter().zip(&jac) {
            assert!(
                (a - c).abs() < 1e-8 * peak.max(1e-30),
                "the preconditioner changed the answer"
            );
        }

        // And at a budget too small for Jacobi, IC(0) is already there.
        let capped =
            |f: fn(&[f64], &[i32], &[i32], &[f64], &mut [f64], u32, f64) -> Result<(), String>| {
                let mut x = vec![0f64; n];
                f(&values, &col_idx, &row_ptr, &b, &mut x, 12, 1e-30).unwrap();
                resid(&x)
            };
        let (j12, i12) = (capped(algos::pcg_solve), capped(algos::ic0_pcg));
        assert!(
            i12 < j12 / 10.0,
            "after 12 iterations ic0 is at {i12:.3e} and jacobi at {j12:.3e}, which is \
             not the improvement a real preconditioner gives"
        );
    }

    /// The shifted retry must actually rescue a matrix that breaks IC(0).
    ///
    /// Dropping fill can make a pivot non-positive even where `A` is
    /// definite, and a preconditioner that gives up there is one the caller
    /// has to write a fallback for.
    #[test]
    fn ic0_recovers_from_a_breakdown_by_shifting() {
        // Weak diagonal relative to the off-diagonals: definite, but only
        // just, which is where the incomplete factorisation struggles.
        let n = 40usize;
        let (mut values, mut col_idx, mut row_ptr) = (Vec::new(), Vec::new(), vec![0i32]);
        for i in 0..n {
            if i > 0 {
                values.push(-1.0);
                col_idx.push(i as i32 - 1);
            }
            values.push(2.0001);
            col_idx.push(i as i32);
            if i + 1 < n {
                values.push(-1.0);
                col_idx.push(i as i32 + 1);
            }
            row_ptr.push(values.len() as i32);
        }
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let mut x = vec![0f64; n];
        algos::ic0_pcg(&values, &col_idx, &row_ptr, &b, &mut x, 500, 1e-12).unwrap();

        let mut ax = vec![0f64; n];
        algos::mat_vec_into(&values, &col_idx, &row_ptr, &x, &mut ax);
        for (r, got) in ax.iter().enumerate() {
            assert!((got - b[r]).abs() < 1e-8, "row {r}: {got} vs {}", b[r]);
        }
    }

    /// A block of right-hand sides must be indistinguishable from solving
    /// them one at a time.
    ///
    /// Batching exists so the matrix is read once per iteration rather than
    /// once per right-hand side. What it must not do is let the columns see
    /// each other: a solver that mixed them would still converge and still
    /// return plausible numbers, and only a comparison against the lone
    /// solves would notice.
    #[test]
    fn pcg_solves_a_block_the_same_way_it_solves_the_columns() {
        let (v, ci, rp, n) = spd();
        let cols: [[f64; 4]; 3] = [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, -1.0],
            [2.5, -1.5, 0.5, 1.0],
        ];
        let mut b = vec![0f64; n * 3];
        for (j, c) in cols.iter().enumerate() {
            for (i, val) in c.iter().enumerate() {
                b[i * 3 + j] = *val;
            }
        }

        let mut block = vec![0f64; n * 3];
        algos::pcg_solve(&v, &ci, &rp, &b, &mut block, 500, 1e-14).unwrap();

        for (j, c) in cols.iter().enumerate() {
            let mut alone = vec![0f64; n];
            algos::pcg_solve(&v, &ci, &rp, c, &mut alone, 500, 1e-14).unwrap();
            let column: Vec<f64> = (0..n).map(|i| block[i * 3 + j]).collect();
            for i in 0..n {
                assert!(
                    (column[i] - alone[i]).abs() < 1e-10,
                    "column {j}: {column:?} vs {alone:?}"
                );
            }
            for (r, got) in mul(&v, &ci, &rp, &column).iter().enumerate() {
                assert!((got - c[r]).abs() < 1e-10, "row {r} of column {j}");
            }
        }
    }

    /// A column that has already converged must not be pushed around by the
    /// ones that have not.
    ///
    /// The zero right-hand side is the sharp case: its solution is exactly
    /// zero, it converges at iteration nought, and it then sits in the block
    /// while its neighbours iterate. Anything that leaks across columns shows
    /// up here as a non-zero.
    #[test]
    fn pcg_holds_a_converged_column_still() {
        let (v, ci, rp, n) = spd();
        let mut b = vec![0f64; n * 2];
        for i in 0..n {
            b[i * 2] = (i + 1) as f64; // column 0 has work to do
            b[i * 2 + 1] = 0.0; // column 1 is already solved
        }
        let mut x = vec![0f64; n * 2];
        algos::pcg_solve(&v, &ci, &rp, &b, &mut x, 500, 1e-14).unwrap();

        for i in 0..n {
            assert_eq!(
                x[i * 2 + 1],
                0.0,
                "the zero column moved: {:?}",
                (0..n).map(|r| x[r * 2 + 1]).collect::<Vec<_>>()
            );
        }
        let solved: Vec<f64> = (0..n).map(|i| x[i * 2]).collect();
        for (r, got) in mul(&v, &ci, &rp, &solved).iter().enumerate() {
            assert!((got - (r + 1) as f64).abs() < 1e-10, "row {r}");
        }
    }

    /// Damped LSQR must agree with plain LSQR on the matrix it stands in for.
    ///
    /// `min ‖A·x − b‖² + damp²‖x‖²` is the same problem as
    /// `min ‖[A; damp·I]·x − [b; 0]‖²`, and the only reason to have the
    /// damped form is to avoid building the second matrix. So the check is
    /// that they land on the same answer: this constructs the augmented
    /// system explicitly and solves it undamped.
    #[test]
    fn damped_lsqr_matches_the_augmented_system_it_replaces() {
        // 5×3, inconsistent.
        let values = vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let col_idx = vec![0, 1, 1, 2, 0, 2, 0, 1, 1, 2];
        let row_ptr = vec![0, 2, 4, 6, 8, 10];
        let b = [1.0, 2.0, 3.0, 0.5, 1.5];
        let (m, n) = (5usize, 3usize);

        for damp in [0.1f64, 1.0, 7.5] {
            let mut damped = vec![0f64; n];
            algos::lsqr_solve(
                &values,
                &col_idx,
                &row_ptr,
                &b,
                &mut damped,
                5000,
                1e-14,
                n,
                damp,
            )
            .unwrap();

            // The same thing spelled out: `n` extra rows of `damp` on the
            // diagonal, and `n` extra zeros on the right-hand side.
            let mut av = values.clone();
            let mut ac = col_idx.clone();
            let mut ar = row_ptr.clone();
            for j in 0..n {
                av.push(damp);
                ac.push(j as i32);
                ar.push(av.len() as i32);
            }
            let mut ab = b.to_vec();
            ab.resize(m + n, 0.0);

            let mut augmented = vec![0f64; n];
            algos::lsqr_solve(&av, &ac, &ar, &ab, &mut augmented, 5000, 1e-14, n, 0.0).unwrap();

            for i in 0..n {
                assert!(
                    (damped[i] - augmented[i]).abs() < 1e-9,
                    "damp = {damp}: {damped:?} vs {augmented:?}"
                );
            }
        }
    }

    /// Damping must actually regularise: more of it, smaller answer.
    ///
    /// Agreement with the augmented system would still hold if both were
    /// wrong in the same way, so this checks the property damping is *for*.
    #[test]
    fn more_damping_gives_a_smaller_solution() {
        let values = vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let col_idx = vec![0, 1, 1, 2, 0, 2, 0, 1, 1, 2];
        let row_ptr = vec![0, 2, 4, 6, 8, 10];
        let b = [1.0, 2.0, 3.0, 0.5, 1.5];

        let norm_at = |damp: f64| {
            let mut x = vec![0f64; 3];
            algos::lsqr_solve(
                &values, &col_idx, &row_ptr, &b, &mut x, 5000, 1e-14, 3, damp,
            )
            .unwrap();
            x.iter().map(|v| v * v).sum::<f64>().sqrt()
        };

        let norms: Vec<f64> = [0.0, 0.5, 2.0, 10.0].iter().map(|d| norm_at(*d)).collect();
        for w in norms.windows(2) {
            assert!(w[1] < w[0], "‖x‖ did not shrink with damping: {norms:?}");
        }
        // And in the limit it goes to zero rather than somewhere arbitrary.
        assert!(norm_at(1e6) < 1e-6, "{}", norm_at(1e6));
    }

    /// Threading must not change the answer.
    ///
    /// The split is over output rows, which are disjoint, so this should hold
    /// exactly rather than approximately — every row accumulates in the same
    /// order whichever thread runs it. A reduction that had been split
    /// differently would show up here as a last-bit difference.
    #[test]
    fn threading_the_matvec_changes_nothing() {
        // Big enough to cross `PAR_MIN_WORK`, so the threaded branch is the
        // one under test. The assertion below says so rather than trusting
        // it: raising the threshold once left this problem on the serial
        // path, where it proved nothing.
        let side = 700usize;
        let n = side * side;
        let (mut values, mut col_idx, mut row_ptr) = (Vec::new(), Vec::new(), vec![0i32]);
        for i in 0..n {
            let (r, c) = (i / side, i % side);
            let mut push = |j: usize, v: f64| {
                values.push(v);
                col_idx.push(j as i32);
            };
            if r > 0 {
                push(i - side, -1.0);
            }
            if c > 0 {
                push(i - 1, -1.0);
            }
            push(i, 4.5);
            if c + 1 < side {
                push(i + 1, -1.0);
            }
            if r + 1 < side {
                push(i + side, -1.0);
            }
            row_ptr.push(values.len() as i32);
        }
        assert!(
            values.len() >= super::PAR_MIN_WORK,
            "the test problem is too small to thread"
        );

        let x: Vec<f64> = (0..n).map(|i| ((i % 71) as f64 / 71.0) - 0.5).collect();
        let mut threaded = vec![0f64; n];
        algos::mat_vec_into(&values, &col_idx, &row_ptr, &x, &mut threaded);

        // The same thing, spelled out serially.
        let mut serial = vec![0f64; n];
        for r in 0..n {
            let mut acc = 0f64;
            for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                acc += values[k] * x[col_idx[k] as usize];
            }
            serial[r] = acc;
        }
        assert_eq!(threaded, serial, "threading moved a value");
    }

    /// How much of an `IC(0)` solve is the factorisation?
    ///
    /// It is recomputed on every call, and a caller with one matrix and many
    /// right-hand sides pays for it every time. Whether that matters depends
    /// on its share of the whole, which is what this prints.
    #[test]
    #[ignore = "timing, not correctness"]
    fn bench_ic0_factor_share() {
        let side = 100usize;
        let n = side * side;
        let (mut values, mut col_idx, mut row_ptr) = (Vec::new(), Vec::new(), vec![0i32]);
        for i in 0..n {
            let (r, c) = (i / side, i % side);
            let mut push = |j: usize, v: f64| {
                values.push(v);
                col_idx.push(j as i32);
            };
            if r > 0 {
                push(i - side, -1.0);
            }
            if c > 0 {
                push(i - 1, -1.0);
            }
            push(i, 4.0);
            if c + 1 < side {
                push(i + 1, -1.0);
            }
            if r + 1 < side {
                push(i + side, -1.0);
            }
            row_ptr.push(values.len() as i32);
        }
        let b: Vec<f64> = (0..n).map(|i| ((i % 17) as f64 / 17.0) - 0.5).collect();
        let reps = 30;

        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            algos::ic0_factor(&values, &col_idx, &row_ptr, n).unwrap();
        }
        let factor = t0.elapsed() / reps;

        let mut x = vec![0f64; n];
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            algos::ic0_pcg(&values, &col_idx, &row_ptr, &b, &mut x, 5000, 1e-12).unwrap();
        }
        let ic0 = t1.elapsed() / reps;

        let t2 = std::time::Instant::now();
        for _ in 0..reps {
            algos::pcg_solve(&values, &col_idx, &row_ptr, &b, &mut x, 5000, 1e-12).unwrap();
        }
        let jac = t2.elapsed() / reps;

        eprintln!(
            "  n={n} nnz={}  factor {factor:>9.2?}  ic0 total {ic0:>9.2?} \
             ({:.0}% is the factorisation)  jacobi {jac:>9.2?}  speedup {:.2}x",
            values.len(),
            100.0 * factor.as_secs_f64() / ic0.as_secs_f64(),
            jac.as_secs_f64() / ic0.as_secs_f64()
        );
    }

    /// Where does a sparse iteration actually spend its time?
    ///
    /// `A·v` is a gather: each output element reads a contiguous run and
    /// accumulates into a register. `Aᵀ·u` over the same CSR is a scatter:
    /// every entry does a read-modify-write into a location the index array
    /// chooses, which cannot be vectorised and cannot be threaded without
    /// atomics. LSQR does one of each per iteration, so the ratio between
    /// them is the ceiling on what any tuning could win.
    #[test]
    #[ignore = "timing, not correctness"]
    fn bench_gather_versus_scatter() {
        let side = 100usize;
        let n = side * side;
        let (mut values, mut col_idx, mut row_ptr) = (Vec::new(), Vec::new(), vec![0i32]);
        for i in 0..n {
            let (r, c) = (i / side, i % side);
            let mut push = |j: usize, v: f64| {
                values.push(v);
                col_idx.push(j as i32);
            };
            if r > 0 {
                push(i - side, -1.0);
            }
            if c > 0 {
                push(i - 1, -1.0);
            }
            push(i, 4.5);
            if c + 1 < side {
                push(i + 1, -1.0);
            }
            if r + 1 < side {
                push(i + side, -1.0);
            }
            row_ptr.push(values.len() as i32);
        }
        let x: Vec<f64> = (0..n).map(|i| (i % 31) as f64).collect();
        let reps = 200;

        let mut y = vec![0f64; n];
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            for r in 0..n {
                let mut acc = 0f64;
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    acc += values[k] * x[col_idx[k] as usize];
                }
                y[r] = acc;
            }
        }
        let gather = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            y.fill(0.0);
            for r in 0..n {
                for k in row_ptr[r] as usize..row_ptr[r + 1] as usize {
                    y[col_idx[k] as usize] += values[k] * x[r];
                }
            }
        }
        let scatter = t1.elapsed();

        eprintln!(
            "  nnz={} gather {gather:>9.2?}  scatter {scatter:>9.2?}  \
             scatter is {:.2}x the gather",
            values.len(),
            scatter.as_secs_f64() / gather.as_secs_f64()
        );
        assert!(y.iter().any(|v| *v != 0.0));
    }

    /// Not an assertion — a measurement, printed for whoever is deciding
    /// whether batching is worth it on their problem.
    ///
    /// Read the `worst diff` column first: it is the one that means something
    /// on any machine, and it should be exactly zero. The speedups are
    /// whatever the box was doing at the time, and a run that is not
    /// monotonic in `k` is measuring the load rather than the code.
    #[test]
    #[ignore = "timing, not correctness"]
    fn bench_block_versus_separate() {
        // A 2-D Laplacian, the shape most sparse SPD systems have.
        let side = 60usize;
        let n = side * side;
        let (mut values, mut col_idx, mut row_ptr) = (Vec::new(), Vec::new(), vec![0i32]);
        for i in 0..n {
            let (r, c) = (i / side, i % side);
            let mut push = |j: usize, v: f64| {
                values.push(v);
                col_idx.push(j as i32);
            };
            if r > 0 {
                push(i - side, -1.0);
            }
            if c > 0 {
                push(i - 1, -1.0);
            }
            push(i, 4.5);
            if c + 1 < side {
                push(i + 1, -1.0);
            }
            if r + 1 < side {
                push(i + side, -1.0);
            }
            row_ptr.push(values.len() as i32);
        }

        for k in [1usize, 8, 32, 128] {
            let b: Vec<f64> = (0..n * k).map(|i| ((i % 97) as f64 / 97.0) - 0.5).collect();

            let t0 = std::time::Instant::now();
            let mut separate = vec![0f64; n * k];
            for j in 0..k {
                let col: Vec<f64> = (0..n).map(|i| b[i * k + j]).collect();
                let mut x = vec![0f64; n];
                algos::pcg_solve(&values, &col_idx, &row_ptr, &col, &mut x, 5000, 1e-10).unwrap();
                for i in 0..n {
                    separate[i * k + j] = x[i];
                }
            }
            let t_sep = t0.elapsed();

            let t1 = std::time::Instant::now();
            let mut block = vec![0f64; n * k];
            algos::pcg_solve(&values, &col_idx, &row_ptr, &b, &mut block, 5000, 1e-10).unwrap();
            let t_blk = t1.elapsed();

            let worst = separate
                .iter()
                .zip(&block)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0, f64::max);
            eprintln!(
                "  n={n} k={k:4}  separate {t_sep:>10.2?}  block {t_blk:>10.2?}  \
                 speedup {:.2}x  worst diff {worst:.2e}",
                t_sep.as_secs_f64() / t_blk.as_secs_f64()
            );
        }
    }

    /// LSQR must stop when the gradient vanishes, not when the budget does.
    ///
    /// `‖A·x − b‖` reaches zero only for a consistent system; an
    /// overdetermined least-squares problem has a nonzero minimum residual,
    /// so a test on `‖r‖` alone never fires and the solve runs to `max_iter`
    /// every time. The quantity that does vanish is `‖Aᵀ·r‖`.
    #[test]
    fn lsqr_stops_early_on_an_inconsistent_system() {
        // 5×3, no exact solution.
        let values = vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let col_idx = vec![0, 1, 1, 2, 0, 2, 0, 1, 1, 2];
        let row_ptr = vec![0, 2, 4, 6, 8, 10];
        let b = [1.0, 2.0, 3.0, 0.5, 1.5];

        let mut small = vec![0f64; 3];
        let n_small = algos::lsqr_solve(
            &values, &col_idx, &row_ptr, &b, &mut small, 50, 1e-12, 3, 0.0,
        )
        .unwrap();
        let mut large = vec![0f64; 3];
        let n_large = algos::lsqr_solve(
            &values, &col_idx, &row_ptr, &b, &mut large, 50_000, 1e-12, 3, 0.0,
        )
        .unwrap();

        // A thousandfold budget must not mean a thousandfold cost.
        assert!(
            n_large <= 50,
            "raising the cap from 50 to 50,000 took {n_large} iterations, so the \
             solve is running to the cap rather than converging"
        );
        assert_eq!(n_small, n_large, "the cap changed where it stopped");
        for (a, c) in small.iter().zip(&large) {
            assert!((a - c).abs() < 1e-12, "{small:?} vs {large:?}");
        }
    }

    /// The residual really is orthogonal to the range of A at the answer —
    /// that is the property the new stopping test asserts, so it is worth
    /// checking against the matrix rather than trusting the counter.
    #[test]
    fn lsqr_leaves_the_residual_orthogonal_to_the_columns() {
        let values = vec![1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let col_idx = vec![0, 1, 1, 2, 0, 2, 0, 1, 1, 2];
        let row_ptr = vec![0, 2, 4, 6, 8, 10];
        let b = [1.0, 2.0, 3.0, 0.5, 1.5];
        let (m, n) = (5usize, 3usize);

        let mut x = vec![0f64; n];
        algos::lsqr_solve(&values, &col_idx, &row_ptr, &b, &mut x, 1000, 1e-14, n, 0.0).unwrap();

        // r = A·x − b, then Aᵀ·r, which is the gradient of ½‖A·x − b‖².
        let mut r = vec![0f64; m];
        for row in 0..m {
            let mut acc = 0f64;
            for k in row_ptr[row] as usize..row_ptr[row + 1] as usize {
                acc += values[k] * x[col_idx[k] as usize];
            }
            r[row] = acc - b[row];
        }
        let mut atr = vec![0f64; n];
        for row in 0..m {
            for k in row_ptr[row] as usize..row_ptr[row + 1] as usize {
                atr[col_idx[k] as usize] += values[k] * r[row];
            }
        }
        let g = atr.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            g < 1e-10,
            "‖Aᵀr‖ = {g:.3e}, so this is not a least-squares solution"
        );
        // And the residual itself is *not* zero — the system is inconsistent,
        // which is what makes the first stopping test useless here.
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            rn > 1e-3,
            "the test problem is consistent, so it proves nothing"
        );
    }
}

#[cfg(test)]
mod multi_rhs_tests {
    use super::*;

    /// `ic0_pcg` with `k` right-hand sides gives what `k` separate calls give.
    ///
    /// The incomplete Cholesky depends only on the matrix, so handing several
    /// right-hand sides in at once should factorise once and solve each — not
    /// change any of the answers. A caller that solves the same system repeatedly
    /// otherwise pays for the factorisation every time; a super-resolution
    /// reconstruction runs 1,430 solves against one matrix.
    #[test]
    fn ic0_pcg_solves_several_right_hand_sides_exactly_as_it_solves_one() {
        // A small SPD system with off-diagonal structure, so IC(0) has something
        // to do and the triangular solves are not trivial.
        let n = 40usize;
        let mut vals: Vec<f64> = Vec::new();
        let mut cidx: Vec<i32> = Vec::new();
        let mut rptr: Vec<i32> = vec![0];
        for i in 0..n {
            for j in i.saturating_sub(2)..(i + 3).min(n) {
                let v = if i == j {
                    8.0 + (i % 5) as f64
                } else {
                    -1.0 / (1 + i.abs_diff(j)) as f64
                };
                cidx.push(j as i32);
                vals.push(v);
            }
            rptr.push(cidx.len() as i32);
        }

        let k = 5usize;
        let b: Vec<f64> = (0..n * k)
            .map(|i| ((i * 37 % 23) as f64 - 11.0) / 3.0)
            .collect();

        let mut together = vec![0.0f64; n * k];
        algos::ic0_pcg(&vals, &cidx, &rptr, &b, &mut together, 500, 1e-13).expect("batched solve");

        for j in 0..k {
            let mut one = vec![0.0f64; n];
            algos::ic0_pcg(
                &vals,
                &cidx,
                &rptr,
                &b[j * n..(j + 1) * n],
                &mut one,
                500,
                1e-13,
            )
            .expect("single solve");
            for i in 0..n {
                let (a, c) = (together[j * n + i], one[i]);
                assert!(
                    (a - c).abs() <= 1e-12 * (1.0 + c.abs()),
                    "rhs {j}, entry {i}: batched {a} against separate {c}"
                );
            }
        }

        // And each column really solves its own system.
        for j in 0..k {
            for i in 0..n {
                let ax: f64 = (rptr[i] as usize..rptr[i + 1] as usize)
                    .map(|t| vals[t] * together[j * n + cidx[t] as usize])
                    .sum();
                let want = b[j * n + i];
                assert!(
                    (ax - want).abs() < 1e-8 * (1.0 + want.abs()),
                    "rhs {j}, row {i}: A·x = {ax}, b = {want}"
                );
            }
        }
    }
}
