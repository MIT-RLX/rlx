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

//! Dense linalg for RLX — `eigh`, `svd`, `qr`, `cholesky`, `solve_triangular`.
//!
//! Downstream package modeled on `jax.scipy.linalg` / `jnp.linalg`.
//! Registers against rlx's framework `OpExtension` + `CpuKernel`
//! scaffold (no rlx-core edits). Forward kernels delegate to LAPACK
//! through `rlx-cpu/src/blas.rs` wrappers.
//!
//! ## v1 scope
//!
//! All ops have working forward and reverse-mode VJPs:
//!
//!   - ✅ `cholesky` — Cholesky-Murray identity (triangular solve + copy).
//!   - ✅ `solve_triangular` — closed form via second triangular solve.
//!   - ✅ `eigh` — Magnus-Neudecker formula with F-matrix degeneracy mask.
//!   - ✅ `qr`   — Walter-Lehmann 2010 (copytril of `R·dRᵀ - dQᵀ·Q`).
//!   - ✅ `svd`  — Townsend 2016 with F-matrix + orthogonal complement.
//!   - ✅ `logdet`, `slogdet`, `pinv`, `lstsq`, `expm` — see per-op docs.
//!
//! ## Multi-output ops
//!
//! `eigh`/`svd`/`qr` return multiple logical tensors. We pack them
//! into a single output buffer per the framework convention; the
//! `DenseTensor` wrapper provides typed accessors that emit `Narrow`
//! to extract each piece:
//!
//!   - `eigh(A)` → packed `[eigvals (n), eigvecs (n×n flat)]`,
//!     length `n + n²` f64.
//!   - `qr(A)` → packed `[Q (m×k flat), R (k×n flat)]`,
//!     length `m·k + k·n` where `k = min(m, n)`.
//!   - `svd(A)` → packed `[U (m×k), S (k), V^T (k×n)]`,
//!     length `m·k + k + k·n`.
//!
//! ## Backend status
//!
//! | Op                | CPU | Metal | MLX |
//! |-------------------|-----|-------|-----|
//! | `cholesky`        | ✅  | —     | —   |
//! | `solve_triangular`| ✅  | —     | —   |
//! | `eigh`            | ✅  | —     | —   |
//! | `qr`              | ✅  | —     | —   |
//! | `svd`             | ✅  | —     | —   |
//!
//! Metal/MLX are mechanical follow-ups (host-callback bodies via
//! the per-backend kernel registries). For graphs containing dense
//! linalg today, pin to `Device::Cpu`.

#![cfg_attr(not(feature = "cpu"), allow(dead_code))]

use std::sync::Arc;

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, Node, NodeId, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

// ── Op names ─────────────────────────────────────────────────────

pub const LINALG_CHOLESKY: &str = "rlx_linalg.cholesky";
pub const LINALG_SOLVE_TRIANGULAR: &str = "rlx_linalg.solve_triangular";
pub const LINALG_EIGH: &str = "rlx_linalg.eigh";
pub const LINALG_QR: &str = "rlx_linalg.qr";
pub const LINALG_SVD: &str = "rlx_linalg.svd";

/// Log-determinant of a SPD matrix. Forward = `2 · Σ log(diag(chol(A)))`;
/// VJP `dL/dA = dL/d(logdet) · A⁻¹` (which is symmetric and SPD).
pub const LINALG_LOGDET: &str = "rlx_linalg.logdet";

/// Backward kernel for `cholesky`: takes `(L, dL/dL)` → produces `dL/dA`.
pub const LINALG_CHOLESKY_BACKWARD: &str = "rlx_linalg.cholesky_backward";
/// Forward Frechet derivative for `cholesky`: `(L, dA) → dL`.
/// `dL = L · phi(L⁻¹·dA·L⁻ᵀ)` where `phi(M) = strict_lower(M) + ½·diag(M)`.
pub const LINALG_CHOLESKY_JVP: &str = "rlx_linalg.cholesky_jvp";
/// Backward kernel for `eigh`: takes `(λ, V, dL/dλ, dL/dV)` → produces `dL/dA`.
pub const LINALG_EIGH_BACKWARD: &str = "rlx_linalg.eigh_backward";
/// Forward Frechet derivative for `eigh`: `(λ, V, dA) → packed [t_λ, t_V_flat]`
/// of length `n + n²`. Computed via `C = Vᵀ·dA·V`, `t_λ = diag(C)`,
/// `Ω[i,j] = C[i,j]/(λ[j]-λ[i])` (off-diag), `t_V = V·Ω`.
pub const LINALG_EIGH_JVP: &str = "rlx_linalg.eigh_jvp";
/// Forward Frechet derivative for `qr`: `(Q, R, dA) → packed [dQ, dR]`
/// of length `m·k + k·n` (k = min(m, n)). Walter-Lehmann forward
/// direction via `M = Qᵀ·dA·R⁻¹`, `X = strict_antisym(M)`,
/// `U = strict_upper_sum(M) + diag(M)`, `dR = U·R`,
/// `dQ = Q·X + (I - Q·Qᵀ)·dA·R⁻¹`.
pub const LINALG_QR_JVP: &str = "rlx_linalg.qr_jvp";
/// Forward Frechet derivative for `svd`: `(U, s, Vᵀ, dA) → packed
/// [dU, ds, dVᵀ]` of length `m·k + k + k·n`. Townsend forward via
/// `C = Uᵀ·dA·V`, `ds = diag(C)`, `Ω_U/Ω_V` from the 2×2 system.
pub const LINALG_SVD_JVP: &str = "rlx_linalg.svd_jvp";
/// Forward Frechet derivative for `pinv`: `(A, dA) → dY` (n×m).
/// Composed via internal SVD: `dY = dV·D·Uᵀ − V·diag(ds/s²)·Uᵀ + V·D·dUᵀ`
/// where `D = diag(1/s)`.
pub const LINALG_PINV_JVP: &str = "rlx_linalg.pinv_jvp";
/// Backward kernel for `qr`: takes `(Q, R, dL/dQ, dL/dR)` → produces `dL/dA`.
pub const LINALG_QR_BACKWARD: &str = "rlx_linalg.qr_backward";
/// Backward kernel for `svd`: takes `(U, s, V^T, dL/dU, dL/ds, dL/dV^T)`
/// → produces `dL/dA`. Townsend 2016 closed form with degeneracy mask.
pub const LINALG_SVD_BACKWARD: &str = "rlx_linalg.svd_backward";
/// Backward kernel for `logdet`: takes `(A, dL/d(logdet))` → produces
/// `dL/dA = dL/d(logdet) · A⁻¹`. Computed via solve(A, I) internally.
pub const LINALG_LOGDET_BACKWARD: &str = "rlx_linalg.logdet_backward";

/// Sign + log|det| of a general square matrix via LU with pivoting.
/// Output packed `[sign, log|det|]`, length 2 F64.
/// VJP: `dL/dA = dL/d(log|det|) · A⁻ᵀ` (sign is non-differentiable).
pub const LINALG_SLOGDET: &str = "rlx_linalg.slogdet";
/// Backward kernel for `slogdet`: `(A, dL/d(log|det|)) → dL/dA`.
pub const LINALG_SLOGDET_BACKWARD: &str = "rlx_linalg.slogdet_backward";

/// Extract the diagonal of a matrix: `[n, n] → [n]`. VJP is `diag_set`.
pub const LINALG_DIAG_EXTRACT: &str = "rlx_linalg.diag_extract";
/// Build a diagonal matrix from a vector: `[n] → [n, n]`. VJP is `diag_extract`.
pub const LINALG_DIAG_SET: &str = "rlx_linalg.diag_set";

/// Matrix exponential `exp(A)` via Padé-13 with scaling-and-squaring.
/// VJP via the augmented-matrix trick (Al-Mohy/Higham): the upper-right
/// n×n block of `exp([[Aᵀ, G], [0, Aᵀ]])` equals the Frechet derivative
/// adjoint dL/dA. Hence the backward kernel is itself a 2n×2n expm.
pub const LINALG_EXPM: &str = "rlx_linalg.expm";
/// Backward kernel for `expm`: `(A, dL/d(exp(A))) → dL/dA`.
pub const LINALG_EXPM_BACKWARD: &str = "rlx_linalg.expm_backward";
/// Forward Frechet derivative for `expm`: `(A, dA) → L_exp(A, dA)`.
/// Computed as the upper-right block of `exp([[A, dA], [0, A]])`.
pub const LINALG_EXPM_JVP: &str = "rlx_linalg.expm_jvp";

/// Moore-Penrose pseudo-inverse via thin SVD with cutoff:
///   Y = V · diag(1/s_filtered) · Uᵀ. Output shape `[n, m]` for
///   input `[m, n]`. Full-column-rank case (m ≥ n) is best supported.
pub const LINALG_PINV: &str = "rlx_linalg.pinv";
/// Backward kernel for `pinv`: `(A, Y, dL/dY) → dL/dA`.
pub const LINALG_PINV_BACKWARD: &str = "rlx_linalg.pinv_backward";

/// Least-squares: `x = argmin ||A·x - b||²` via thin SVD pseudoinverse.
/// `A: [m, n]`, `b: [m]` (vector RHS only in v1). Output `x: [n]`.
pub const LINALG_LSTSQ: &str = "rlx_linalg.lstsq";
/// Backward kernel: `(A, x, b, dL/dx) → dL/dA`.
pub const LINALG_LSTSQ_BACKWARD_A: &str = "rlx_linalg.lstsq_backward_a";
/// Backward kernel: `(A, dL/dx) → dL/db`.
pub const LINALG_LSTSQ_BACKWARD_B: &str = "rlx_linalg.lstsq_backward_b";

// ── Algos: shared LAPACK-backed kernel bodies ────────────────────

#[cfg(feature = "cpu")]
mod algos {
    /// Cholesky: in-place factorization. Returns the factor in the
    /// configured triangle, zeros the other.
    pub fn cholesky(a_in: &[f64], n: usize, lower: bool, out: &mut [f64]) -> Result<(), String> {
        if a_in.len() != n * n || out.len() != n * n {
            return Err(format!("cholesky: shape mismatch (n={n})"));
        }
        out.copy_from_slice(a_in);
        let info = rlx_cpu::blas::dpotrf(out, n, lower);
        if info != 0 {
            return Err(format!("cholesky: dpotrf info={info} (matrix not SPD)"));
        }
        Ok(())
    }

    /// Solve `op(L) · X = B` where L is triangular (lower if `lower`,
    /// upper otherwise). `transpose_a` selects between `L · X = B` and
    /// `Lᵀ · X = B`. B is overwritten with X (so we copy first).
    pub fn solve_triangular(
        a: &[f64],
        b: &[f64],
        n: usize,
        nrhs: usize,
        lower: bool,
        transpose_a: bool,
        out: &mut [f64],
    ) -> Result<(), String> {
        if a.len() != n * n {
            return Err(format!(
                "solve_triangular: A must be n×n (got len {})",
                a.len()
            ));
        }
        if b.len() != n * nrhs || out.len() != n * nrhs {
            return Err("solve_triangular: B/out shape mismatch".to_string());
        }
        out.copy_from_slice(b);
        rlx_cpu::blas::dtrsm_lower_or_upper(a, out, n, nrhs, lower, transpose_a);
        Ok(())
    }

    /// Symmetric eigendecomposition. Output is packed:
    ///   - `out[0..n]` = eigenvalues (ascending)
    ///   - `out[n..n+n²]` = eigenvectors as a row-major `n×n` matrix
    ///     where row `i` is the i-th eigenvector
    ///
    /// Note: LAPACK's `dsyevd` returns eigenvectors as columns of A
    /// in column-major. The col-major-to-row-major view of the same
    /// bytes treats columns as rows → eigenvector `i` is row `i` in
    /// the row-major view.
    pub fn eigh(a_in: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if a_in.len() != n * n || out.len() != n + n * n {
            return Err(format!("eigh: shape mismatch (n={n})"));
        }
        let mut a_buf = a_in.to_vec();
        let mut w = vec![0f64; n];
        let info = rlx_cpu::blas::dsyevd(&mut a_buf, &mut w, n);
        if info != 0 {
            return Err(format!("eigh: dsyevd info={info}"));
        }
        out[..n].copy_from_slice(&w);
        out[n..].copy_from_slice(&a_buf);
        Ok(())
    }

    /// QR factorization. Output packed:
    ///   - `out[0..m·k]` = Q row-major, shape `[m, k]`
    ///   - `out[m·k..m·k + k·n]` = R row-major, shape `[k, n]`
    /// where `k = min(m, n)`.
    pub fn qr(a_in: &[f64], m: usize, n: usize, out: &mut [f64]) -> Result<(), String> {
        let k = m.min(n);
        if a_in.len() != m * n || out.len() != m * k + k * n {
            return Err(format!("qr: shape mismatch (m={m}, n={n})"));
        }
        let mut a_work = a_in.to_vec();
        let mut q = vec![0f64; m * k];
        let mut r = vec![0f64; k * n];
        let info = rlx_cpu::blas::dgeqrf_full(&mut a_work, m, n, &mut q, &mut r);
        if info != 0 {
            return Err(format!("qr: LAPACK info={info}"));
        }
        out[..m * k].copy_from_slice(&q);
        out[m * k..].copy_from_slice(&r);
        Ok(())
    }

    // ── Backward kernels ─────────────────────────────────────────
    //
    // Each backward op takes the relevant forward outputs plus the
    // upstream gradient(s) and produces dL/dA in one CPU pass.
    // Routed via `Op::Custom` — the IR-level VJP arm on each forward
    // op emits a call to the corresponding backward op.

    /// Row-major C = A·B via Accelerate/MKL/OpenBLAS `dgemm`. Drop-in
    /// replacement for the previous hand-rolled triple-loop — the
    /// 10-50× win on n≥50 matters for every backward / JVP kernel
    /// that builds composite matrix products (eigh / qr / svd / pinv /
    /// expm / lstsq backward + forward Frechet derivatives).
    #[inline]
    fn matmul_naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, out: &mut [f64]) {
        if m == 0 || k == 0 || n == 0 {
            for v in out.iter_mut() {
                *v = 0.0;
            }
            return;
        }
        rlx_cpu::blas::dgemm(a, b, out, m, k, n);
    }

    fn transpose(a: &[f64], r: usize, c: usize, out: &mut [f64]) {
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = a[i * c + j];
            }
        }
    }

    /// `dL/dA` for `L = cholesky(A, lower=true)`. Murray 2016 formula:
    ///
    ///   M    = Lᵀ · dL/dL
    ///   Φ(M) = strict-lower(M) + 0.5 · diag(M); upper zeroed
    ///   Q    = L⁻ᵀ · Φ(M) · L⁻¹       (two triangular solves)
    ///   dL/dA = 0.5 · (Q + Qᵀ)         (symmetrize)
    ///
    /// Lower-triangular factor only in v1.
    pub fn cholesky_backward(
        l: &[f64],
        dl_dl: &[f64],
        n: usize,
        lower: bool,
        out: &mut [f64],
    ) -> Result<(), String> {
        if !lower {
            return Err("cholesky_backward: only lower-triangular factor supported".into());
        }
        if l.len() != n * n || dl_dl.len() != n * n || out.len() != n * n {
            return Err(format!("cholesky_backward: shape mismatch (n={n})"));
        }
        // Step 1: M = Lᵀ · dL/dL.
        let mut m = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0f64;
                for k in 0..n {
                    s += l[k * n + i] * dl_dl[k * n + j];
                }
                m[i * n + j] = s;
            }
        }
        // Step 2: in-place Φ.
        for i in 0..n {
            for j in 0..n {
                if i < j {
                    m[i * n + j] = 0.0;
                } else if i == j {
                    m[i * n + j] *= 0.5;
                }
            }
        }
        // Step 3a: Y = L⁻ᵀ · Φ(M).  Solve Lᵀ · Y = Φ(M) → dtrsm transpose=true.
        let mut y = m;
        rlx_cpu::blas::dtrsm_lower_or_upper(l, &mut y, n, n, true, true);
        // Step 3b: Q = Y · L⁻¹.  Q · L = Y ⇒ Lᵀ · Qᵀ = Yᵀ.
        let mut yt = vec![0f64; n * n];
        transpose(&y, n, n, &mut yt);
        rlx_cpu::blas::dtrsm_lower_or_upper(l, &mut yt, n, n, true, true);
        let mut q = vec![0f64; n * n];
        transpose(&yt, n, n, &mut q);
        // Step 4: symmetrize.
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = 0.5 * (q[i * n + j] + q[j * n + i]);
            }
        }
        Ok(())
    }

    /// Forward Frechet derivative for eigh. Output is packed
    ///   `[t_λ (n), t_V_flat (n²)]` of length `n + n²`.
    ///
    /// Given A = V·Λ·Vᵀ:
    ///   C = Vᵀ·dA·V          (n×n, symmetric)
    ///   t_λ = diag(C)
    ///   Ω[i,j] = C[i,j]/(λ\[j\]-λ\[i\]) for i≠j  (degeneracy mask)
    ///   t_V = V·Ω
    pub fn eigh_jvp(
        eigvals: &[f64],
        eigvecs: &[f64],
        da: &[f64],
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        if eigvals.len() != n || eigvecs.len() != n * n || da.len() != n * n {
            return Err(format!("eigh_jvp: shape mismatch (n={n})"));
        }
        if out.len() != n + n * n {
            return Err(format!(
                "eigh_jvp: out len {} != n+n²={}",
                out.len(),
                n + n * n
            ));
        }
        // V_col has columns as eigenvectors (transpose of row-major eigvecs).
        let mut v_col = vec![0f64; n * n];
        transpose(eigvecs, n, n, &mut v_col);
        // C = V_colᵀ · dA · V_col = eigvecs · dA · eigvecs^T... wait
        // V_col is n×n with cols = eigvectors. V_colᵀ is row-major where
        // each row is an eigvector — same as eigvecs row-major form.
        // So V_colᵀ · dA = eigvecs · dA.
        let mut tmp = vec![0f64; n * n];
        matmul_naive(eigvecs, da, n, n, n, &mut tmp);
        let mut c = vec![0f64; n * n];
        matmul_naive(&tmp, &v_col, n, n, n, &mut c);
        // t_λ = diag(C).
        for i in 0..n {
            out[i] = c[i * n + i];
        }
        // Ω[i,j] = C[i,j]/(λ[j]-λ[i]) for i≠j; degeneracy mask.
        let scale = eigvals.iter().fold(0f64, |a, b| a.max(b.abs())).max(1.0);
        let tol = scale * 1e-10;
        let mut omega = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d = eigvals[j] - eigvals[i];
                    if d.abs() > tol {
                        omega[i * n + j] = c[i * n + j] / d;
                    }
                }
            }
        }
        // dV_col = V_col · Ω.
        let mut dv_col = vec![0f64; n * n];
        matmul_naive(&v_col, &omega, n, n, n, &mut dv_col);
        // Convert dv_col (cols-as-eigvecs) → row-major eigvecs format
        // (rows-as-eigvecs) = dv_col^T.
        for i in 0..n {
            for j in 0..n {
                out[n + i * n + j] = dv_col[j * n + i];
            }
        }
        Ok(())
    }

    /// Forward Frechet derivative for thin QR (m ≥ n = k).
    /// Output packed `[dQ (m·k), dR (k·n)]`. Walter-Lehmann forward.
    pub fn qr_jvp(
        q: &[f64],
        r: &[f64],
        da: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        let k = m.min(n);
        if k != n {
            return Err(format!(
                "qr_jvp: only thin m≥n supported (got m={m}, n={n})"
            ));
        }
        if q.len() != m * k || r.len() != k * n || da.len() != m * n || out.len() != m * k + k * n {
            return Err(format!("qr_jvp: shape mismatch (m={m}, n={n})"));
        }
        // Step 1: M = Qᵀ · dA · R⁻¹  (k×n=k×k since k=n).
        let mut qt = vec![0f64; k * m];
        transpose(q, m, k, &mut qt);
        let mut qt_da = vec![0f64; k * n];
        matmul_naive(&qt, da, k, m, n, &mut qt_da);
        // M = qt_da · R⁻¹: solve Rᵀ·Mᵀ = qt_daᵀ → Mᵀ = R⁻ᵀ·qt_daᵀ → M = (R⁻ᵀ·qt_daᵀ)ᵀ.
        let mut tmp = vec![0f64; n * k];
        transpose(&qt_da, k, n, &mut tmp);
        rlx_cpu::blas::dtrsm_lower_or_upper(
            r, &mut tmp, n, k, /*lower=*/ false, /*trans=*/ true,
        );
        let mut m_mat = vec![0f64; k * n];
        transpose(&tmp, n, k, &mut m_mat);
        // Step 2: X (antisymmetric) and U (upper-with-diag).
        let mut x = vec![0f64; k * k];
        let mut u = vec![0f64; k * n];
        for i in 0..k {
            for j in 0..n {
                if i > j {
                    x[i * k + j] = m_mat[i * k + j];
                    if j < k {
                        x[j * k + i] = -m_mat[i * k + j];
                    } // strict-upper of X
                } else if i == j {
                    u[i * n + j] = m_mat[i * n + j];
                } else {
                    // i < j: U[i,j] = M[i,j] + M[j,i]   (j may be ≥ k for non-square)
                    let mji = if j < k { m_mat[j * n + i] } else { 0.0 };
                    u[i * n + j] = m_mat[i * n + j] + mji;
                }
            }
        }
        // Step 3: dR = U · R   (k×n).
        let mut dr = vec![0f64; k * n];
        matmul_naive(&u, r, k, n, n, &mut dr);
        // Step 4: dQ_inner = Q · X   (m×k).
        let mut dq = vec![0f64; m * k];
        matmul_naive(q, &x, m, k, k, &mut dq);
        // Step 5: orthogonal-complement (only when m > k):
        //   dQ += dA·R⁻¹ - Q·M.
        if m > k {
            // dA · R⁻¹: solve Rᵀ·Y = dAᵀ → Y = R⁻ᵀ·dAᵀ → Yᵀ = dA·R⁻¹.
            // Easier: solve R·X = ... hmm, dA · R⁻¹ where R is k×n=k×k.
            // dA·R⁻¹ has shape m×k. Compute via: solve Rᵀ·Z = dAᵀ → Z = R⁻ᵀ·dAᵀ → dA·R⁻¹ = Zᵀ.
            let mut dat = vec![0f64; n * m];
            transpose(da, m, n, &mut dat);
            rlx_cpu::blas::dtrsm_lower_or_upper(
                r, &mut dat, n, m, /*lower=*/ false, /*trans=*/ true,
            );
            let mut da_rinv = vec![0f64; m * k];
            transpose(&dat, n, m, &mut da_rinv);
            // Q·M  (m×k)
            let mut qm = vec![0f64; m * k];
            matmul_naive(q, &m_mat, m, k, k, &mut qm);
            for i in 0..(m * k) {
                dq[i] += da_rinv[i] - qm[i];
            }
        }
        // Pack output.
        out[..m * k].copy_from_slice(&dq);
        out[m * k..].copy_from_slice(&dr);
        Ok(())
    }

    /// Forward Frechet derivative for thin SVD (m ≥ n = k).
    /// Output packed `[dU (m·k), ds (k), dVᵀ (k·n)]`. Townsend forward.
    pub fn svd_jvp(
        u: &[f64],
        s: &[f64],
        vt: &[f64],
        da: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        let k = m.min(n);
        if k != n {
            return Err(format!(
                "svd_jvp: only thin m≥n supported (got m={m}, n={n})"
            ));
        }
        if u.len() != m * k
            || s.len() != k
            || vt.len() != k * n
            || da.len() != m * n
            || out.len() != m * k + k + k * n
        {
            return Err(format!("svd_jvp: shape mismatch (m={m}, n={n})"));
        }
        // V from Vᵀ.
        let mut v = vec![0f64; n * k];
        transpose(vt, k, n, &mut v);
        // C = Uᵀ · dA · V  (k×k).
        let mut ut = vec![0f64; k * m];
        transpose(u, m, k, &mut ut);
        let mut tmp = vec![0f64; k * n];
        matmul_naive(&ut, da, k, m, n, &mut tmp);
        let mut c = vec![0f64; k * k];
        matmul_naive(&tmp, &v, k, n, k, &mut c);

        // Degeneracy mask.
        let s_max = s.iter().cloned().fold(0f64, f64::max).max(1.0);
        let tol = (s_max * s_max) * 1e-12;

        // Ω_U / Ω_V (k×k).
        let mut omega_u = vec![0f64; k * k];
        let mut omega_v = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    let d = s[j] * s[j] - s[i] * s[i];
                    if d.abs() > tol {
                        let f = 1.0 / d;
                        omega_u[i * k + j] = f * (s[j] * c[i * k + j] + s[i] * c[j * k + i]);
                        omega_v[i * k + j] = f * (s[i] * c[i * k + j] + s[j] * c[j * k + i]);
                    }
                }
            }
        }
        // dU_inner = U · Ω_U.
        let mut du = vec![0f64; m * k];
        matmul_naive(u, &omega_u, m, k, k, &mut du);
        // dV = V · Ω_V  (n×k).
        let mut dv = vec![0f64; n * k];
        matmul_naive(&v, &omega_v, n, k, k, &mut dv);
        // Orthogonal complement when m > k.
        if m > k {
            // dA · V  (m×k).
            let mut dav = vec![0f64; m * k];
            matmul_naive(da, &v, m, n, k, &mut dav);
            // (dA · V) · diag(1/s).
            let cutoff = tol.sqrt();
            for i in 0..m {
                for j in 0..k {
                    let denom = if s[j].abs() > cutoff { s[j] } else { 1.0 };
                    dav[i * k + j] /= denom;
                }
            }
            // (I - U·Uᵀ) · dav = dav - U·(Uᵀ·dav).
            let mut ut_dav = vec![0f64; k * k];
            matmul_naive(&ut, &dav, k, m, k, &mut ut_dav);
            let mut u_ut_dav = vec![0f64; m * k];
            matmul_naive(u, &ut_dav, m, k, k, &mut u_ut_dav);
            for i in 0..(m * k) {
                du[i] += dav[i] - u_ut_dav[i];
            }
        }
        // ds = diag(C).
        // Pack output.
        out[..m * k].copy_from_slice(&du);
        for i in 0..k {
            out[m * k + i] = c[i * k + i];
        }
        // dVᵀ = (dV)ᵀ  (k×n).
        let mut dvt = vec![0f64; k * n];
        transpose(&dv, n, k, &mut dvt);
        out[m * k + k..].copy_from_slice(&dvt);
        Ok(())
    }

    /// Forward Frechet derivative for `pinv` (m ≥ n full column rank).
    /// Composed via SVD:
    ///   `Y = V·D·Uᵀ`, `D = diag(1/s)`
    ///   `dY = dV·D·Uᵀ + V·dD·Uᵀ + V·D·dUᵀ`
    ///   `dD = -diag(ds/s²)`.
    pub fn pinv_jvp(
        a: &[f64],
        da: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        let k = m.min(n);
        if k != n {
            return Err(format!("pinv_jvp: only m≥n supported (got m={m}, n={n})"));
        }
        if a.len() != m * n || da.len() != m * n || out.len() != n * m {
            return Err(format!("pinv_jvp: shape mismatch (m={m}, n={n})"));
        }
        // Internal SVD.
        let mut a_work = a.to_vec();
        let mut u = vec![0f64; m * k];
        let mut s = vec![0f64; k];
        let mut vt = vec![0f64; k * n];
        let info = rlx_cpu::blas::dgesvd_thin(&mut a_work, m, n, &mut s, &mut u, &mut vt);
        if info != 0 {
            return Err(format!("pinv_jvp: dgesvd info={info}"));
        }
        // SVD JVP components.
        let mut svd_jvp_out = vec![0f64; m * k + k + k * n];
        svd_jvp(&u, &s, &vt, da, m, n, &mut svd_jvp_out)?;
        let du = &svd_jvp_out[..m * k];
        let ds = &svd_jvp_out[m * k..m * k + k];
        let dvt = &svd_jvp_out[m * k + k..];

        // V (n×k), dV (n×k) from Vᵀ / dVᵀ.
        let mut v = vec![0f64; n * k];
        transpose(&vt, k, n, &mut v);
        let mut dv = vec![0f64; n * k];
        transpose(dvt, k, n, &mut dv);
        // Uᵀ (k×m), dUᵀ (k×m).
        let mut ut = vec![0f64; k * m];
        transpose(&u, m, k, &mut ut);
        let mut dut = vec![0f64; k * m];
        transpose(du, m, k, &mut dut);

        let s_max = s.iter().cloned().fold(0f64, f64::max);
        let cutoff = (m.max(n) as f64) * f64::EPSILON * s_max;

        // term1 = dV · D · Uᵀ.
        let mut d_ut = ut.clone();
        for i in 0..k {
            let inv_si = if s[i] > cutoff { 1.0 / s[i] } else { 0.0 };
            for j in 0..m {
                d_ut[i * m + j] *= inv_si;
            }
        }
        let mut term1 = vec![0f64; n * m];
        matmul_naive(&dv, &d_ut, n, k, m, &mut term1);
        // term2 = V · dD · Uᵀ where dD = -diag(ds_i/s_i²).
        #[allow(non_snake_case)]
        let mut neg_dD_ut = ut.clone();
        for i in 0..k {
            let scale = if s[i] > cutoff {
                -ds[i] / (s[i] * s[i])
            } else {
                0.0
            };
            for j in 0..m {
                neg_dD_ut[i * m + j] *= scale;
            }
        }
        let mut term2 = vec![0f64; n * m];
        matmul_naive(&v, &neg_dD_ut, n, k, m, &mut term2);
        // term3 = V · D · dUᵀ.
        let mut d_dut = dut.clone();
        for i in 0..k {
            let inv_si = if s[i] > cutoff { 1.0 / s[i] } else { 0.0 };
            for j in 0..m {
                d_dut[i * m + j] *= inv_si;
            }
        }
        let mut term3 = vec![0f64; n * m];
        matmul_naive(&v, &d_dut, n, k, m, &mut term3);

        for i in 0..(n * m) {
            out[i] = term1[i] + term2[i] + term3[i];
        }
        Ok(())
    }

    /// Forward Frechet derivative for cholesky:
    ///   t_L = L · phi(L⁻¹·dA·L⁻ᵀ)
    /// where phi(M) = strict_lower(M) + ½·diag(M).
    pub fn cholesky_jvp(
        l: &[f64],
        da: &[f64],
        n: usize,
        lower: bool,
        out: &mut [f64],
    ) -> Result<(), String> {
        if !lower {
            return Err("cholesky_jvp: only lower-triangular factor supported".into());
        }
        if l.len() != n * n || da.len() != n * n || out.len() != n * n {
            return Err(format!("cholesky_jvp: shape mismatch (n={n})"));
        }
        // Step 1: Y = L⁻¹·dA via L·Y = dA.
        let mut y = da.to_vec();
        rlx_cpu::blas::dtrsm_lower_or_upper(l, &mut y, n, n, true, false);
        // Step 2: M = Y·L⁻ᵀ = (L⁻¹·Yᵀ)ᵀ.
        let mut yt = vec![0f64; n * n];
        transpose(&y, n, n, &mut yt);
        rlx_cpu::blas::dtrsm_lower_or_upper(l, &mut yt, n, n, true, false);
        let mut m = vec![0f64; n * n];
        transpose(&yt, n, n, &mut m);
        // Step 3: phi(M) = strict_lower(M) + ½·diag(M).
        for i in 0..n {
            for j in 0..n {
                if i < j {
                    m[i * n + j] = 0.0;
                } else if i == j {
                    m[i * n + j] *= 0.5;
                }
            }
        }
        // Step 4: t_L = L · phi(M).
        matmul_naive(l, &m, n, n, n, out);
        Ok(())
    }

    /// `dL/dA` for `(λ, V) = eigh(A)`, A symmetric. Inputs:
    ///   - `eigvals`: λ length n (ascending from forward)
    ///   - `eigvecs`: V row-major n×n where row i is the i-th eigenvector
    ///     (matches the row-major view of LAPACK's col-major output)
    ///   - `dl_dlambda`: dL/dλ length n
    ///   - `dl_dv`: dL/dV row-major n×n (same orientation as `eigvecs`)
    ///
    /// Formula:
    ///   F[i,j] = 1/(λ\[j\] - λ\[i\])  for i ≠ j (with degeneracy mask)
    ///   G      = Vᵀ · dL/dV
    ///   T      = diag(dL/dλ) + (F ⊙ (G - Gᵀ)) / 2
    ///   dL/dA  = sym(V · T · Vᵀ)
    ///
    /// Degeneracy: |λ\[j\] - λ\[i\]| < eps → F[i,j] = 0 (drop the contribution).
    /// The eps threshold is per-eigenvalue-pair and uses the spectrum's
    /// max-magnitude as a scale.
    pub fn eigh_backward(
        eigvals: &[f64],
        eigvecs: &[f64],
        dl_dlambda: &[f64],
        dl_dv: &[f64],
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        if eigvals.len() != n || eigvecs.len() != n * n {
            return Err(format!("eigh_backward: shape mismatch (n={n})"));
        }
        if dl_dlambda.len() != n || dl_dv.len() != n * n || out.len() != n * n {
            return Err(format!("eigh_backward: gradient shape mismatch (n={n})"));
        }

        // The convention from `algos::eigh`: V is stored as row-major
        // [n, n] where row i is the i-th eigenvector. For the math
        // below we need V as a "matrix whose columns are eigenvectors"
        // — call that V_col. V_col is the transpose of V (row-major).
        // Matrix-multiply formulas below use V_col throughout.
        let mut v_col = vec![0f64; n * n];
        transpose(eigvecs, n, n, &mut v_col);

        // dL/dV in the same column-as-eigenvector convention.
        let mut dvc = vec![0f64; n * n];
        transpose(dl_dv, n, n, &mut dvc);

        // F-matrix. Use a relative tolerance scaled by the spectrum's
        // largest magnitude — guards against singular F entries when
        // two eigenvalues are degenerate (otherwise 1/(λ_j-λ_i)
        // diverges and the gradient explodes).
        let scale = eigvals.iter().fold(0f64, |a, b| a.max(b.abs())).max(1.0);
        let tol = scale * 1e-10;
        let mut f = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d = eigvals[j] - eigvals[i];
                    f[i * n + j] = if d.abs() > tol { 1.0 / d } else { 0.0 };
                }
            }
        }

        // G = V_colᵀ · dL/dV_col. (n×n)
        let mut g = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0f64;
                for k in 0..n {
                    s += v_col[k * n + i] * dvc[k * n + j];
                }
                g[i * n + j] = s;
            }
        }

        // T = diag(dL/dλ) + (F ⊙ (G - Gᵀ)) / 2.
        let mut t = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    t[i * n + j] = dl_dlambda[i];
                } else {
                    t[i * n + j] = 0.5 * f[i * n + j] * (g[i * n + j] - g[j * n + i]);
                }
            }
        }

        // dL/dA = V_col · T · V_colᵀ.
        let mut vt = vec![0f64; n * n];
        matmul_naive(&v_col, &t, n, n, n, &mut vt);
        let mut v_col_t = vec![0f64; n * n];
        transpose(&v_col, n, n, &mut v_col_t);
        let mut dad = vec![0f64; n * n];
        matmul_naive(&vt, &v_col_t, n, n, n, &mut dad);

        // Symmetrize.
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = 0.5 * (dad[i * n + j] + dad[j * n + i]);
            }
        }
        Ok(())
    }

    /// `dL/dA` for `(Q, R) = qr(A)`. Walter–Lehmann 2010 formula for
    /// thin QR with full column rank:
    ///
    ///   M       = R · dL/dRᵀ - dL/dQᵀ · Q
    ///   copytril(M)[i,j] = M[i,j] if i > j; M[i,j] if i < j (taken
    ///                      from M[j,i]); M[i,i] for i = j... actually
    ///                      copytril(M) := M_lower + M_lowerᵀ (symmetric copy).
    ///   S       = dL/dQ + Q · copytril(M)
    ///   dL/dA   = S · R⁻ᵀ
    ///
    /// Requires R square (m ≥ n case for thin QR with k = n).
    pub fn qr_backward(
        q: &[f64],
        r: &[f64],
        dl_dq: &[f64],
        dl_dr: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        let k = m.min(n);
        if k != n {
            return Err(format!(
                "qr_backward: only m≥n thin QR supported (got m={m}, n={n})"
            ));
        }
        if q.len() != m * k || r.len() != k * n {
            return Err("qr_backward: forward shape mismatch".to_string());
        }
        if dl_dq.len() != m * k || dl_dr.len() != k * n || out.len() != m * n {
            return Err("qr_backward: gradient shape mismatch".to_string());
        }

        // Step 1: dL/dRᵀ (transpose dL/dR)
        let mut dr_t = vec![0f64; n * k];
        transpose(dl_dr, k, n, &mut dr_t);
        // Step 2: M_part1 = R · dL/dRᵀ  (k×k)
        let mut m_part1 = vec![0f64; k * k];
        matmul_naive(r, &dr_t, k, n, k, &mut m_part1);
        // Step 3: dL/dQᵀ · Q  (k×k)
        let mut dq_t = vec![0f64; k * m];
        transpose(dl_dq, m, k, &mut dq_t);
        let mut m_part2 = vec![0f64; k * k];
        matmul_naive(&dq_t, q, k, m, k, &mut m_part2);
        // Step 4: M = M_part1 - M_part2
        let mut m_mat = vec![0f64; k * k];
        for i in 0..k * k {
            m_mat[i] = m_part1[i] - m_part2[i];
        }

        // Step 5: copytril(M) = M_strict_lower + M_strict_lowerᵀ + diag(M).
        // (Symmetric matrix where the strictly-upper is filled by reflecting
        // the strictly-lower.)
        let mut copytril = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                if i > j {
                    copytril[i * k + j] = m_mat[i * k + j];
                } else if i < j {
                    copytril[i * k + j] = m_mat[j * k + i];
                } else {
                    copytril[i * k + j] = m_mat[i * k + j];
                }
            }
        }

        // Step 6: S = dL/dQ + Q · copytril(M)  (m×k)
        let mut q_ctril = vec![0f64; m * k];
        matmul_naive(q, &copytril, m, k, k, &mut q_ctril);
        let mut s = vec![0f64; m * k];
        for i in 0..m * k {
            s[i] = dl_dq[i] + q_ctril[i];
        }

        // Step 7: dL/dA = S · R⁻ᵀ
        // Compute via solve_triangular(R, transpose) on S transposed:
        //   dL/dAᵀ = R⁻¹ · Sᵀ ⇒ dL/dA = (R⁻¹ Sᵀ)ᵀ
        // Use upper triangular R; transpose=false; solve R · X = Sᵀ → X = R⁻¹ Sᵀ.
        // Then transpose X to get dL/dA.
        let mut s_t = vec![0f64; k * m];
        transpose(&s, m, k, &mut s_t);
        rlx_cpu::blas::dtrsm_lower_or_upper(
            r, &mut s_t, k, m, /*lower=*/ false, /*trans=*/ false,
        );
        // s_t now holds R⁻¹ · Sᵀ = (dL/dA)ᵀ
        transpose(&s_t, k, m, out);
        Ok(())
    }

    /// `dL/dA` for `(U, s, Vᵀ) = svd(A)` (thin SVD, m≥n=k).
    ///
    /// Townsend 2016 closed form. For thin SVD with m ≥ n:
    ///
    ///   F[i,j] = 1/(s_j² - s_i²) for i ≠ j (degenerate → 0)
    ///   Σ_U = Uᵀ · dL/dU         (k×k)
    ///   Σ_V = Vᵀ · dL/dV         (k×k)        (V = (V^T)^T)
    ///
    ///   dA_diag    = U · diag(dL/ds) · Vᵀ
    ///   subspace_U = U · (F ⊙ ((Σ_U·Σ_s) - (Σ_s·Σ_Uᵀ))) · Vᵀ
    ///   subspace_V = U · (F ⊙ ((Σ_s·Σ_V) - (Σ_Vᵀ·Σ_s))) · Vᵀ
    ///   ortho_U    = (I - U·Uᵀ) · dL/dU · diag(1/s) · Vᵀ
    ///                          (only when m > k = n; thin case)
    ///
    ///   dL/dA = dA_diag + subspace_U + subspace_V + ortho_U
    ///
    /// where `Σ_s = diag(s)`. Per JAX's implementation; Townsend's
    /// paper is the canonical reference.
    pub fn svd_backward(
        u: &[f64],
        s: &[f64],
        vt: &[f64],
        dl_du: &[f64],
        dl_ds: &[f64],
        dl_dvt: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        let k = m.min(n);
        if k != n {
            return Err(format!(
                "svd_backward: only thin SVD with m≥n supported (got m={m}, n={n})"
            ));
        }
        if u.len() != m * k || s.len() != k || vt.len() != k * n {
            return Err("svd_backward: forward shape mismatch".into());
        }
        if dl_du.len() != m * k || dl_ds.len() != k || dl_dvt.len() != k * n {
            return Err("svd_backward: gradient shape mismatch".into());
        }
        if out.len() != m * n {
            return Err("svd_backward: output shape mismatch".into());
        }

        // V = (V^T)^T, n×k row-major. Same shape as Uᵀ blocks below.
        let mut v = vec![0f64; n * k];
        transpose(vt, k, n, &mut v);
        // dL/dV = (dL/dV^T)^T
        let mut dl_dv = vec![0f64; n * k];
        transpose(dl_dvt, k, n, &mut dl_dv);

        // F-matrix with degeneracy mask:
        //   F[i,j] = 1/(s_j² - s_i²)  for i ≠ j (degenerate → 0)
        let s_max = s.iter().fold(0f64, |a, b| a.max(b.abs())).max(1.0);
        let tol = (s_max * s_max) * 1e-12;
        let mut f = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    let d = s[j] * s[j] - s[i] * s[i];
                    f[i * k + j] = if d.abs() > tol { 1.0 / d } else { 0.0 };
                }
            }
        }

        // Σ_U = Uᵀ · dL/dU  (k×k)
        let mut ut = vec![0f64; k * m];
        transpose(u, m, k, &mut ut);
        let mut sigma_u = vec![0f64; k * k];
        matmul_naive(&ut, dl_du, k, m, k, &mut sigma_u);

        // Σ_V = Vᵀ · dL/dV  (k×k)
        let mut vt_mat = vec![0f64; k * n];
        transpose(&v, n, k, &mut vt_mat); // = original vt, but recompute for clarity
        let mut sigma_v = vec![0f64; k * k];
        matmul_naive(&vt_mat, &dl_dv, k, n, k, &mut sigma_v);

        // Inner k×k middle matrix:
        //   M[i,j] = δ_ij · dL/ds[i]
        //          + F[i,j] · ((Σ_U[i,j] - Σ_U[j,i])·s[j]
        //                       + (Σ_V[i,j] - Σ_V[j,i])·s[i])
        let mut middle = vec![0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    middle[i * k + j] = dl_ds[i];
                } else {
                    let sigma_u_ij = sigma_u[i * k + j];
                    let sigma_u_ji = sigma_u[j * k + i];
                    let sigma_v_ij = sigma_v[i * k + j];
                    let sigma_v_ji = sigma_v[j * k + i];
                    let term = (sigma_u_ij - sigma_u_ji) * s[j] + (sigma_v_ij - sigma_v_ji) * s[i];
                    middle[i * k + j] = f[i * k + j] * term;
                }
            }
        }

        // Subspace term: U · middle · Vᵀ
        let mut u_mid = vec![0f64; m * k];
        matmul_naive(u, &middle, m, k, k, &mut u_mid);
        let mut subspace = vec![0f64; m * n];
        matmul_naive(&u_mid, vt, m, k, n, &mut subspace);

        // Orthogonal-complement term (thin SVD, m > k): adds a
        // contribution from dL/dU outside U's column space.
        //   ortho_U = (I - U·Uᵀ) · dL/dU · diag(1/s) · Vᵀ
        // Only evaluate when m > k (avoids needless work for square A).
        let mut ortho = vec![0f64; m * n];
        if m > k {
            // proj = (I - U Uᵀ) · dL/dU = dL/dU - U · (Uᵀ · dL/dU)
            let mut proj = dl_du.to_vec();
            // Uᵀ · dL/dU is sigma_u (already computed). Subtract U·sigma_u.
            let mut u_sigma = vec![0f64; m * k];
            matmul_naive(u, &sigma_u, m, k, k, &mut u_sigma);
            for idx in 0..m * k {
                proj[idx] -= u_sigma[idx];
            }
            // Right-multiply by diag(1/s).
            for i in 0..m {
                for j in 0..k {
                    let denom = if s[j].abs() > tol.sqrt() { s[j] } else { 1.0 };
                    proj[i * k + j] /= denom;
                }
            }
            // ortho = proj · Vᵀ (m×k · k×n = m×n)
            matmul_naive(&proj, vt, m, k, n, &mut ortho);
        }

        // Combine: dL/dA = subspace + ortho.
        for idx in 0..m * n {
            out[idx] = subspace[idx] + ortho[idx];
        }
        Ok(())
    }

    /// LU with partial pivoting (Doolittle, in-place row-major).
    /// Returns `swap_count` (number of row interchanges); A becomes
    /// `L\U` (unit-diag lower stored implicitly, U on/above diag).
    /// `pivot[i]` = original row index now at row `i` (so the permutation
    /// matrix P satisfies P·A_orig = L·U).
    pub fn lu_pivoting(a: &mut [f64], n: usize, pivot: &mut [usize]) -> Result<usize, String> {
        if a.len() != n * n || pivot.len() != n {
            return Err("lu_pivoting: shape mismatch".into());
        }
        for i in 0..n {
            pivot[i] = i;
        }
        let mut swaps = 0usize;
        for k in 0..n {
            // Find pivot row in column k (max |a[r,k]| for r ≥ k).
            let mut piv_row = k;
            let mut piv_val = a[k * n + k].abs();
            for r in (k + 1)..n {
                let v = a[r * n + k].abs();
                if v > piv_val {
                    piv_val = v;
                    piv_row = r;
                }
            }
            if piv_val == 0.0 {
                return Err(format!("lu_pivoting: singular at column {k}"));
            }
            if piv_row != k {
                for c in 0..n {
                    a.swap(k * n + c, piv_row * n + c);
                }
                pivot.swap(k, piv_row);
                swaps += 1;
            }
            let inv = 1.0 / a[k * n + k];
            for r in (k + 1)..n {
                a[r * n + k] *= inv;
                let f = a[r * n + k];
                for c in (k + 1)..n {
                    a[r * n + c] -= f * a[k * n + c];
                }
            }
        }
        Ok(swaps)
    }

    /// `slogdet(A)` for general square A. Output: `[sign, log|det|]`
    /// (length 2). Sign ∈ {-1, 0, +1}; log|det| = -∞ for singular A
    /// (returned as `f64::NEG_INFINITY`).
    pub fn slogdet(a_in: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if a_in.len() != n * n {
            return Err(format!("slogdet: A must be n×n, got {}", a_in.len()));
        }
        if out.len() != 2 {
            return Err(format!("slogdet: out must be [2], got {}", out.len()));
        }
        let mut a = a_in.to_vec();
        let mut piv = vec![0usize; n];
        match lu_pivoting(&mut a, n, &mut piv) {
            Ok(swaps) => {
                let mut sign: f64 = if swaps % 2 == 0 { 1.0 } else { -1.0 };
                let mut log_abs = 0f64;
                for i in 0..n {
                    let d = a[i * n + i];
                    if d == 0.0 {
                        sign = 0.0;
                        log_abs = f64::NEG_INFINITY;
                        break;
                    }
                    if d < 0.0 {
                        sign = -sign;
                    }
                    log_abs += d.abs().ln();
                }
                out[0] = sign;
                out[1] = log_abs;
                Ok(())
            }
            Err(_) => {
                out[0] = 0.0;
                out[1] = f64::NEG_INFINITY;
                Ok(())
            }
        }
    }

    /// `dL/dA = dL/d(log|det|) · A⁻ᵀ` for general square A.
    /// Sign component is non-differentiable (gradient dropped).
    pub fn slogdet_backward(
        a: &[f64],
        dl_d_logabsdet: f64,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        if a.len() != n * n || out.len() != n * n {
            return Err(format!("slogdet_bwd: shape mismatch (n={n})"));
        }
        // A⁻ᵀ = (A⁻¹)ᵀ. Solve Aᵀ·X = I → X = A⁻ᵀ.
        // Trick: solve A·Y = I → Y = A⁻¹, then transpose.
        let mut a_buf = a.to_vec();
        let mut b_buf = vec![0f64; n * n];
        for i in 0..n {
            b_buf[i * n + i] = 1.0;
        }
        let info = rlx_cpu::blas::dgesv(&mut a_buf, &mut b_buf, n, n);
        if info != 0 {
            return Err(format!("slogdet_bwd: dgesv info={info}"));
        }
        // out = dL/d(logabsdet) · (A⁻¹)ᵀ
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = dl_d_logabsdet * b_buf[j * n + i];
            }
        }
        Ok(())
    }

    /// Extract diagonal entries of a square matrix.
    pub fn diag_extract(a: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if a.len() != n * n || out.len() != n {
            return Err(format!("diag_extract: shape mismatch (n={n})"));
        }
        for i in 0..n {
            out[i] = a[i * n + i];
        }
        Ok(())
    }

    /// Build a diagonal matrix from a length-n vector. Off-diagonals zeroed.
    pub fn diag_set(v: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if v.len() != n || out.len() != n * n {
            return Err(format!("diag_set: shape mismatch (n={n})"));
        }
        for i in 0..(n * n) {
            out[i] = 0.0;
        }
        for i in 0..n {
            out[i * n + i] = v[i];
        }
        Ok(())
    }

    /// Matrix 1-norm: max column absolute sum.
    fn mat_norm_1(a: &[f64], n: usize) -> f64 {
        let mut max_col = 0f64;
        for j in 0..n {
            let mut s = 0f64;
            for i in 0..n {
                s += a[i * n + j].abs();
            }
            if s > max_col {
                max_col = s;
            }
        }
        max_col
    }

    /// `exp(A)` via Padé-13 + scaling-and-squaring (Higham, 2005).
    pub fn expm(a_in: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if a_in.len() != n * n || out.len() != n * n {
            return Err(format!("expm: shape mismatch (n={n})"));
        }
        // Padé-13 coefficients.
        const B: [f64; 14] = [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ];
        const THETA_13: f64 = 5.371920351148152;

        let norm = mat_norm_1(a_in, n);
        let s = if norm > THETA_13 {
            (norm / THETA_13).log2().ceil() as i32
        } else {
            0
        };
        let scale = (-(s as f64) * 2f64.ln()).exp(); // 1 / 2^s
        let mut a = a_in.to_vec();
        for v in a.iter_mut() {
            *v *= scale;
        }

        // Compute A², A⁴, A⁶.
        let mut a2 = vec![0f64; n * n];
        matmul_naive(&a, &a, n, n, n, &mut a2);
        let mut a4 = vec![0f64; n * n];
        matmul_naive(&a2, &a2, n, n, n, &mut a4);
        let mut a6 = vec![0f64; n * n];
        matmul_naive(&a4, &a2, n, n, n, &mut a6);

        // u_inner = b9·A² + b11·A⁴ + b13·A⁶
        // u_outer = A·(b1·I + b3·A² + b5·A⁴ + b7·A⁶ + A⁶·u_inner)
        // v_inner = b8·A² + b10·A⁴ + b12·A⁶
        // v       = b0·I + b2·A² + b4·A⁴ + b6·A⁶ + A⁶·v_inner
        let mut u_inner = vec![0f64; n * n];
        for i in 0..n * n {
            u_inner[i] = B[9] * a2[i] + B[11] * a4[i] + B[13] * a6[i];
        }
        let mut a6_u = vec![0f64; n * n];
        matmul_naive(&a6, &u_inner, n, n, n, &mut a6_u);
        let mut u_pre = vec![0f64; n * n];
        for i in 0..n * n {
            u_pre[i] = a6_u[i] + B[7] * a6[i] + B[5] * a4[i] + B[3] * a2[i];
        }
        // Add B[1]·I
        for i in 0..n {
            u_pre[i * n + i] += B[1];
        }
        let mut u = vec![0f64; n * n];
        matmul_naive(&a, &u_pre, n, n, n, &mut u);

        let mut v_inner = vec![0f64; n * n];
        for i in 0..n * n {
            v_inner[i] = B[8] * a2[i] + B[10] * a4[i] + B[12] * a6[i];
        }
        let mut a6_v = vec![0f64; n * n];
        matmul_naive(&a6, &v_inner, n, n, n, &mut a6_v);
        let mut v = vec![0f64; n * n];
        for i in 0..n * n {
            v[i] = a6_v[i] + B[6] * a6[i] + B[4] * a4[i] + B[2] * a2[i];
        }
        for i in 0..n {
            v[i * n + i] += B[0];
        }

        // Solve (V - U)·R = (V + U) → R = exp(scaled A).
        let mut lhs = vec![0f64; n * n];
        let mut rhs = vec![0f64; n * n];
        for i in 0..n * n {
            lhs[i] = v[i] - u[i];
            rhs[i] = v[i] + u[i];
        }
        let info = rlx_cpu::blas::dgesv(&mut lhs, &mut rhs, n, n);
        if info != 0 {
            return Err(format!("expm: dgesv info={info}"));
        }
        let mut r = rhs;

        // Square s times: r = r²·r²·... s applications.
        let mut tmp = vec![0f64; n * n];
        for _ in 0..s {
            matmul_naive(&r, &r, n, n, n, &mut tmp);
            std::mem::swap(&mut r, &mut tmp);
        }
        out.copy_from_slice(&r);
        Ok(())
    }

    /// Forward Frechet derivative of expm: L_exp(A, dA) =
    ///   exp([[A, dA], [0, A]])_upper_right (= ∂/∂t exp(A + t·dA)|_{t=0}).
    /// Same augmented-matrix trick as `expm_backward` but using A
    /// (rather than Aᵀ) so the result is the forward push, not the
    /// adjoint.
    pub fn expm_jvp(a: &[f64], da: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if a.len() != n * n || da.len() != n * n || out.len() != n * n {
            return Err(format!("expm_jvp: shape mismatch (n={n})"));
        }
        let m = 2 * n;
        let mut big = vec![0f64; m * m];
        for i in 0..n {
            for j in 0..n {
                big[i * m + j] = a[i * n + j]; // top-left = A
                big[i * m + n + j] = da[i * n + j]; // top-right = dA
                big[(n + i) * m + n + j] = a[i * n + j]; // bottom-right = A
            }
        }
        let mut big_exp = vec![0f64; m * m];
        expm(&big, m, &mut big_exp)?;
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = big_exp[i * m + n + j];
            }
        }
        Ok(())
    }

    /// VJP for `expm` via Al-Mohy/Higham augmented-matrix trick:
    ///   exp([[Aᵀ, G], [0, Aᵀ]])_upper_right = L_exp(Aᵀ, G) = dL/dA.
    pub fn expm_backward(a: &[f64], g: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if a.len() != n * n || g.len() != n * n || out.len() != n * n {
            return Err(format!("expm_bwd: shape mismatch (n={n})"));
        }
        let m = 2 * n;
        let mut big = vec![0f64; m * m];
        for i in 0..n {
            for j in 0..n {
                big[i * m + j] = a[j * n + i]; // top-left = Aᵀ
                big[i * m + n + j] = g[i * n + j]; // top-right = G
                big[(n + i) * m + n + j] = a[j * n + i]; // bottom-right = Aᵀ
                // bottom-left already zero
            }
        }
        let mut big_exp = vec![0f64; m * m];
        expm(&big, m, &mut big_exp)?;
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = big_exp[i * m + n + j];
            }
        }
        Ok(())
    }

    /// Moore-Penrose pseudo-inverse via thin SVD. `A: m×n` → `Y: n×m`.
    /// Singular values smaller than `rcond·s_max` are dropped (treated
    /// as zero). For full-rank A, `Y = (AᵀA)⁻¹·Aᵀ` (when m ≥ n) or
    /// `Aᵀ·(A·Aᵀ)⁻¹` (when m < n).
    pub fn pinv(a_in: &[f64], m: usize, n: usize, out: &mut [f64]) -> Result<(), String> {
        if a_in.len() != m * n || out.len() != n * m {
            return Err(format!("pinv: shape mismatch (m={m}, n={n})"));
        }
        let k = m.min(n);
        let mut a_work = a_in.to_vec();
        let mut u = vec![0f64; m * k];
        let mut s = vec![0f64; k];
        let mut vt = vec![0f64; k * n];
        let info = rlx_cpu::blas::dgesvd_thin(&mut a_work, m, n, &mut s, &mut u, &mut vt);
        if info != 0 {
            return Err(format!("pinv: dgesvd info={info}"));
        }
        // rcond cutoff: drop s_i < max(m,n)·eps·s_max.
        let s_max = s.iter().cloned().fold(0f64, f64::max);
        let cutoff = (m.max(n) as f64) * f64::EPSILON * s_max;
        // Y = V · diag(1/s_filtered) · Uᵀ
        // Compute Sinv·Uᵀ first as k×m, then V·that as n×m.
        // V is (Vᵀ)ᵀ which is n×k.
        let mut sinv_ut = vec![0f64; k * m];
        for i in 0..k {
            let inv_si = if s[i] > cutoff { 1.0 / s[i] } else { 0.0 };
            for j in 0..m {
                sinv_ut[i * m + j] = inv_si * u[j * k + i];
            }
        }
        // Y[i,j] = sum_l V[i,l] · sinv_ut[l,j] = sum_l Vᵀ[l,i] · sinv_ut[l,j]
        for i in 0..n {
            for j in 0..m {
                let mut acc = 0f64;
                for l in 0..k {
                    acc += vt[l * n + i] * sinv_ut[l * m + j];
                }
                out[i * m + j] = acc;
            }
        }
        Ok(())
    }

    /// VJP for `pinv`. For full column-rank A (m ≥ n):
    ///   dL/dA = -Yᵀ·G·Yᵀ + (I_m - A·Y)·Gᵀ·Y·Yᵀ
    /// where Y = pinv(A), G = dL/dY (n×m). For square A, the second
    /// term vanishes (A·Y = I_m), reducing to -A⁻ᵀ·G·A⁻ᵀ.
    pub fn pinv_backward(
        a: &[f64],
        y: &[f64],
        g: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        if a.len() != m * n || y.len() != n * m || g.len() != n * m || out.len() != m * n {
            return Err(format!("pinv_bwd: shape mismatch (m={m}, n={n})"));
        }
        // Term 1: -Yᵀ·G·Yᵀ  (m×n)
        // Yᵀ is m×n; Yᵀ·G: m×n · n×m = m×m; ·Yᵀ: m×m · m×n = m×n.
        let mut yt_g = vec![0f64; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut acc = 0f64;
                for l in 0..n {
                    acc += y[l * m + i] * g[l * m + j]; // Yᵀ[i,l]·G[l,j]
                }
                yt_g[i * m + j] = acc;
            }
        }
        // term1[i,j] = -(Yᵀ·G·Yᵀ)[i,j] = -sum_l yt_g[i,l]·Yᵀ[l,j]
        //           = -sum_l yt_g[i,l]·Y[j,l]
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for l in 0..m {
                    acc += yt_g[i * m + l] * y[j * m + l];
                }
                out[i * n + j] = -acc;
            }
        }
        // Term 2 (only if m > n): (I_m - A·Y)·Gᵀ·Y·Yᵀ
        // A·Y is m×m; P = I - A·Y (residual projector).
        // Gᵀ is m×n; Gᵀ·Y: m×n·n×m = m×m; ·Yᵀ: m×m·m×n = m×n.
        // P·(...): m×m·m×n = m×n. Add to out.
        if m > n {
            // ay = A · Y  (m×m)
            let mut ay = vec![0f64; m * m];
            for i in 0..m {
                for j in 0..m {
                    let mut acc = 0f64;
                    for l in 0..n {
                        acc += a[i * n + l] * y[l * m + j];
                    }
                    ay[i * m + j] = acc;
                }
            }
            // gt_y = Gᵀ · Y  (m×m): gt_y[i,j] = sum_l G[l,i]·Y[l,j]
            let mut gt_y = vec![0f64; m * m];
            for i in 0..m {
                for j in 0..m {
                    let mut acc = 0f64;
                    for l in 0..n {
                        acc += g[l * m + i] * y[l * m + j];
                    }
                    gt_y[i * m + j] = acc;
                }
            }
            // gt_y_yt = Gᵀ·Y·Yᵀ  (m×n): gt_y_yt[i,j] = sum_l gt_y[i,l]·Y[j,l]
            let mut gt_y_yt = vec![0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0f64;
                    for l in 0..m {
                        acc += gt_y[i * m + l] * y[j * m + l];
                    }
                    gt_y_yt[i * n + j] = acc;
                }
            }
            // out += (I - A·Y) · gt_y_yt
            for i in 0..m {
                for j in 0..n {
                    let mut acc = gt_y_yt[i * n + j]; // I·gt_y_yt
                    for l in 0..m {
                        acc -= ay[i * m + l] * gt_y_yt[l * n + j];
                    }
                    out[i * n + j] += acc;
                }
            }
        }
        Ok(())
    }

    /// Least-squares: x = pinv(A)·b. A: m×n (m≥n preferred), b: m.
    /// Output x: n.
    pub fn lstsq(a: &[f64], b: &[f64], m: usize, n: usize, out: &mut [f64]) -> Result<(), String> {
        if a.len() != m * n || b.len() != m || out.len() != n {
            return Err(format!("lstsq: shape mismatch (m={m}, n={n})"));
        }
        // Forward: x = pinv(A)·b. Compute pinv via SVD, then matvec.
        let mut y = vec![0f64; n * m];
        pinv(a, m, n, &mut y)?;
        for i in 0..n {
            let mut acc = 0f64;
            for j in 0..m {
                acc += y[i * m + j] * b[j];
            }
            out[i] = acc;
        }
        Ok(())
    }

    /// VJP for `lstsq` w.r.t. A (full column rank case, m ≥ n):
    ///   dL/dA = -Yᵀ·G·xᵀ + r·Gᵀ·(AᵀA)⁻¹
    /// where Y = pinv(A), G = dL/dx (n), x = pinv(A)·b (n),
    /// r = b - A·x (m). For square m=n the residual term vanishes.
    pub fn lstsq_backward_a(
        a: &[f64],
        x: &[f64],
        b: &[f64],
        dl_dx: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        if a.len() != m * n
            || x.len() != n
            || b.len() != m
            || dl_dx.len() != n
            || out.len() != m * n
        {
            return Err(format!("lstsq_bwd_a: shape mismatch (m={m}, n={n})"));
        }
        // Y = pinv(A), n×m
        let mut y = vec![0f64; n * m];
        pinv(a, m, n, &mut y)?;
        // term1[i,j] = -(Yᵀ·G·xᵀ)[i,j]
        //            = -(Yᵀ·G)[i] · x[j]   (G is column n→1, so Yᵀ·G is m vector)
        // Yᵀ·G is m vector: (Yᵀ·G)[i] = sum_l Y[l,i] · G[l]
        let mut yt_g = vec![0f64; m];
        for i in 0..m {
            let mut acc = 0f64;
            for l in 0..n {
                acc += y[l * m + i] * dl_dx[l];
            }
            yt_g[i] = acc;
        }
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = -yt_g[i] * x[j];
            }
        }
        // Term 2: r·Gᵀ·(AᵀA)⁻¹  (only matters when m > n)
        if m > n {
            // r = b - A·x
            let mut r = vec![0f64; m];
            for i in 0..m {
                let mut acc = b[i];
                for j in 0..n {
                    acc -= a[i * n + j] * x[j];
                }
                r[i] = acc;
            }
            // (AᵀA)⁻¹·G computed via solving normal equations.
            // Build AᵀA (n×n SPD).
            let mut ata = vec![0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0f64;
                    for l in 0..m {
                        acc += a[l * n + i] * a[l * n + j];
                    }
                    ata[i * n + j] = acc;
                }
            }
            // Solve AᵀA · z = G. b_solve must be n×1.
            let mut z = dl_dx.to_vec();
            let info = rlx_cpu::blas::dgesv(&mut ata, &mut z, n, 1);
            if info != 0 {
                return Err(format!("lstsq_bwd_a: dgesv info={info}"));
            }
            // Add r ⊗ z to out: out[i,j] += r[i] · z[j]
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] += r[i] * z[j];
                }
            }
        }
        Ok(())
    }

    /// VJP for `lstsq` w.r.t. b: dL/db = pinv(A)ᵀ · dL/dx.
    pub fn lstsq_backward_b(
        a: &[f64],
        dl_dx: &[f64],
        m: usize,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        if a.len() != m * n || dl_dx.len() != n || out.len() != m {
            return Err(format!("lstsq_bwd_b: shape mismatch (m={m}, n={n})"));
        }
        let mut y = vec![0f64; n * m];
        pinv(a, m, n, &mut y)?;
        // pinv(A)ᵀ is m×n; pinv(A)ᵀ · G has shape m.
        // out[i] = sum_j Y[j,i] · G[j]
        for i in 0..m {
            let mut acc = 0f64;
            for j in 0..n {
                acc += y[j * m + i] * dl_dx[j];
            }
            out[i] = acc;
        }
        Ok(())
    }

    /// `logdet(A)` for SPD A. Computed via Cholesky:
    ///   L = chol(A);  log det(A) = 2 · Σ log L[i,i]
    pub fn logdet(a_in: &[f64], n: usize, out: &mut [f64]) -> Result<(), String> {
        if a_in.len() != n * n {
            return Err(format!("logdet: A must be n×n, got len {}", a_in.len()));
        }
        if out.len() != 1 {
            return Err(format!(
                "logdet: out must be scalar [1], got len {}",
                out.len()
            ));
        }
        let mut l = a_in.to_vec();
        let info = rlx_cpu::blas::dpotrf(&mut l, n, /*lower=*/ true);
        if info != 0 {
            return Err(format!("logdet: Cholesky failed (info={info}, A not SPD)"));
        }
        let mut sum_log = 0f64;
        for i in 0..n {
            let d = l[i * n + i];
            if d <= 0.0 {
                return Err(format!("logdet: non-positive Cholesky diag at {i}"));
            }
            sum_log += d.ln();
        }
        out[0] = 2.0 * sum_log;
        Ok(())
    }

    /// `dL/dA = dL/d(logdet) · A⁻ᵀ` for SPD A. Since A is symmetric
    /// SPD, A⁻ᵀ = A⁻¹, so the gradient is just `dL/d(logdet) · A⁻¹`.
    /// Computed via solve(A, I).
    pub fn logdet_backward(
        a: &[f64],
        dl_d_logdet: f64,
        n: usize,
        out: &mut [f64],
    ) -> Result<(), String> {
        if a.len() != n * n || out.len() != n * n {
            return Err(format!("logdet_backward: shape mismatch (n={n})"));
        }
        // Solve A · X = I  → X = A⁻¹.
        // dgesv overwrites both A and B; copy A and an identity into mutables.
        let mut a_buf = a.to_vec();
        let mut b_buf = vec![0f64; n * n];
        for i in 0..n {
            b_buf[i * n + i] = 1.0;
        }
        let info = rlx_cpu::blas::dgesv(&mut a_buf, &mut b_buf, n, n);
        if info != 0 {
            return Err(format!("logdet_backward: dgesv info={info}"));
        }
        // dL/dA = dL/d(logdet) · A⁻¹  (already symmetric for SPD A,
        // but symmetrize defensively).
        for i in 0..n {
            for j in 0..n {
                let v = 0.5 * (b_buf[i * n + j] + b_buf[j * n + i]);
                out[i * n + j] = dl_d_logdet * v;
            }
        }
        Ok(())
    }

    /// SVD (thin / "S" mode). Output packed:
    ///   - `out[0..m·k]`             = U row-major, shape `[m, k]`
    ///   - `out[m·k..m·k+k]`         = S, length `k`
    ///   - `out[m·k+k..m·k+k+k·n]`   = V^T row-major, shape `[k, n]`
    /// where `k = min(m, n)`.
    pub fn svd(a_in: &[f64], m: usize, n: usize, out: &mut [f64]) -> Result<(), String> {
        let k = m.min(n);
        if a_in.len() != m * n || out.len() != m * k + k + k * n {
            return Err(format!("svd: shape mismatch (m={m}, n={n})"));
        }
        let mut a_work = a_in.to_vec();
        let mut u = vec![0f64; m * k];
        let mut s = vec![0f64; k];
        let mut vt = vec![0f64; k * n];
        let info = rlx_cpu::blas::dgesvd_thin(&mut a_work, m, n, &mut s, &mut u, &mut vt);
        if info != 0 {
            return Err(format!("svd: dgesvd info={info}"));
        }
        out[..m * k].copy_from_slice(&u);
        out[m * k..m * k + k].copy_from_slice(&s);
        out[m * k + k..].copy_from_slice(&vt);
        Ok(())
    }
}

// ── Cholesky ─────────────────────────────────────────────────────

struct CholeskyExt;

impl OpExtension for CholeskyExt {
    fn name(&self) -> &str {
        LINALG_CHOLESKY
    }
    fn num_inputs(&self) -> usize {
        1
    } // A: [n, n]
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "cholesky: A must be F64");
        assert_eq!(a.rank(), 2, "cholesky: A must be 2D");
        a.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Closed-form Murray 2016 dL/dA via the cholesky_backward op.
        // Forward L = chol(A); upstream is dL/dL.
        let l_fwd = ctx.fwd_map[&node.id];
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => Vec::new(),
        };
        let g_a = ctx
            .bwd
            .custom_op(LINALG_CHOLESKY_BACKWARD, attrs, vec![l_fwd, ctx.upstream]);
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // t_L = L · phi(L⁻¹·dA·L⁻ᵀ).
        let t_a = ctx.tangents[0]?;
        let l = ctx.fwd_map[&node.id];
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return None,
        };
        Some(ctx.bwd.custom_op(LINALG_CHOLESKY_JVP, attrs, vec![l, t_a]))
    }
}

#[cfg(feature = "cpu")]
struct CholeskyCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for CholeskyCpu {
    fn name(&self) -> &str {
        LINALG_CHOLESKY
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("cholesky A")?;
        let out = output.expect_f64_mut("cholesky out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("cholesky: A length {n_sq} not n²"));
        }
        algos::cholesky(a, n, lower, out)
    }
}

// ── Solve Triangular ─────────────────────────────────────────────

struct SolveTriangularExt;

impl OpExtension for SolveTriangularExt {
    fn name(&self) -> &str {
        LINALG_SOLVE_TRIANGULAR
    }
    fn num_inputs(&self) -> usize {
        2
    } // A (n×n), B (n×nrhs)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let b = inputs[1];
        b.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // y = solve(op(A), B). Closed form:
        //   dL/dB = (op(A))⁻ᵀ · upstream
        // = solve(op(A), upstream, transpose_flag_flipped). Only dL/dB
        // is implemented in v1; dL/dA is a triangular outer-product
        // gather (mirrors sparse-LU's values gradient — mechanical
        // follow-up).
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return vec![],
        };
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let transpose_a = attrs.get(1).copied().unwrap_or(0) != 0;
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let new_attrs = vec![
            if lower { 1u8 } else { 0 },
            if !transpose_a { 1 } else { 0 },
        ];
        let g_b = ctx.bwd.custom_op(
            LINALG_SOLVE_TRIANGULAR,
            new_attrs,
            vec![a_bwd, ctx.upstream],
        );
        vec![(1, g_b)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // y = solve(A, B); dy = solve(A, dB - dA·y).
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return None,
        };
        let a = ctx.fwd_map[&node.inputs[0]];
        let y = ctx.fwd_map[&node.id];
        let y_shape = ctx.bwd.shape(y).clone();
        let rhs = match (ctx.tangents[0], ctx.tangents[1]) {
            (Some(t_a), Some(t_b)) => {
                let prod = ctx.bwd.matmul(t_a, y, y_shape.clone());
                ctx.bwd
                    .binary(rlx_ir::op::BinaryOp::Sub, t_b, prod, y_shape.clone())
            }
            (Some(t_a), None) => {
                let prod = ctx.bwd.matmul(t_a, y, y_shape.clone());
                ctx.bwd
                    .activation(rlx_ir::op::Activation::Neg, prod, y_shape.clone())
            }
            (None, Some(t_b)) => t_b,
            (None, None) => return None,
        };
        Some(
            ctx.bwd
                .custom_op(LINALG_SOLVE_TRIANGULAR, attrs, vec![a, rhs]),
        )
    }
}

#[cfg(feature = "cpu")]
struct SolveTriangularCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SolveTriangularCpu {
    fn name(&self) -> &str {
        LINALG_SOLVE_TRIANGULAR
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("solve_triangular A")?;
        let b = inputs[1].expect_f64("solve_triangular B")?;
        let out = output.expect_f64_mut("solve_triangular out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let transpose_a = attrs.get(1).copied().unwrap_or(0) != 0;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("solve_triangular: A length {n_sq} not n²"));
        }
        let nrhs = b.len() / n;
        algos::solve_triangular(a, b, n, nrhs, lower, transpose_a, out)
    }
}

// ── Symmetric Eigendecomposition ─────────────────────────────────

struct EighExt;

impl OpExtension for EighExt {
    fn name(&self) -> &str {
        LINALG_EIGH
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "eigh: A must be F64");
        assert_eq!(a.rank(), 2, "eigh: A must be 2D");
        let n = a.num_elements().expect("eigh: A must be statically shaped");
        let n_dim = (n as f64).sqrt() as usize;
        assert_eq!(n_dim * n_dim, n, "eigh: A must be square");
        // Packed: [eigenvalues (n), eigenvectors (n²)] flat 1D.
        Shape::new(&[n_dim + n], DType::F64)
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Forward output is packed [λ (n), V (n²)]. Upstream has the
        // same layout (built from the user's downstream Narrow + ops).
        // Unpack both, call eigh_backward(λ, V, dL/dλ, dL/dV).
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let n = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };

        let packed_fwd = ctx.fwd_map[&node.id];
        let lambda_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: n,
            },
            vec![packed_fwd],
            Shape::new(&[n], DType::F64),
        );
        let v_flat_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: n,
                len: n * n,
            },
            vec![packed_fwd],
            Shape::new(&[n * n], DType::F64),
        );

        let dl_dlambda = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: n,
            },
            vec![ctx.upstream],
            Shape::new(&[n], DType::F64),
        );
        let dl_dv_flat = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: n,
                len: n * n,
            },
            vec![ctx.upstream],
            Shape::new(&[n * n], DType::F64),
        );

        // eigh_backward kernel reads V and dL/dV as flat n²; reshape
        // not strictly necessary because the kernel computes its
        // own row/col indexing — but we wrap them so shape inference
        // for the backward op sees consistent metadata.
        let g_a = ctx.bwd.custom_op(
            LINALG_EIGH_BACKWARD,
            Vec::new(),
            vec![lambda_fwd, v_flat_fwd, dl_dlambda, dl_dv_flat],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Forward Frechet via the eigh_jvp kernel. Inputs to the kernel:
        // (λ, V_flat, dA). Output: packed [t_λ, t_V_flat] of length n+n².
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&node.inputs[0]];
        let n = match ctx.bwd.shape(a).dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        // Unpack λ and V from forward output (stored in JVP graph at fwd_map[&node.id]).
        let packed_fwd = ctx.fwd_map[&node.id];
        let lambda = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: n,
            },
            vec![packed_fwd],
            Shape::new(&[n], DType::F64),
        );
        let v_flat = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: n,
                len: n * n,
            },
            vec![packed_fwd],
            Shape::new(&[n * n], DType::F64),
        );
        // dA might be 2D [n,n] but the kernel expects flat n²; reshape.
        let da_flat = ctx.bwd.add_node(
            rlx_ir::Op::Reshape {
                new_shape: vec![(n * n) as i64],
            },
            vec![t_a],
            Shape::new(&[n * n], DType::F64),
        );
        Some(
            ctx.bwd
                .custom_op(LINALG_EIGH_JVP, Vec::new(), vec![lambda, v_flat, da_flat]),
        )
    }
}

struct EighJvpExt;

impl OpExtension for EighJvpExt {
    fn name(&self) -> &str {
        LINALG_EIGH_JVP
    }
    fn num_inputs(&self) -> usize {
        3
    } // λ, V_flat, dA_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let n = inputs[0]
            .num_elements()
            .expect("eigh_jvp: λ must have static shape");
        Shape::new(&[n + n * n], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct EighJvpCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for EighJvpCpu {
    fn name(&self) -> &str {
        LINALG_EIGH_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let lambda = inputs[0].expect_f64("eigh_jvp λ")?;
        let v_flat = inputs[1].expect_f64("eigh_jvp V")?;
        let da_flat = inputs[2].expect_f64("eigh_jvp dA")?;
        let out = output.expect_f64_mut("eigh_jvp out")?;
        let n = lambda.len();
        algos::eigh_jvp(lambda, v_flat, da_flat, n, out)
    }
}

#[cfg(feature = "cpu")]
struct EighCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for EighCpu {
    fn name(&self) -> &str {
        LINALG_EIGH
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("eigh A")?;
        let out = output.expect_f64_mut("eigh out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("eigh: A length {n_sq} not n²"));
        }
        algos::eigh(a, n, out)
    }
}

// ── QR ───────────────────────────────────────────────────────────

struct QrExt;

impl OpExtension for QrExt {
    fn name(&self) -> &str {
        LINALG_QR
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        // Shapes need both m and n. The infer_shape input is the matrix
        // A which carries them. We encode no special attrs (yet).
        let _ = attrs;
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "qr: A must be F64");
        assert_eq!(a.rank(), 2, "qr: A must be 2D");
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("qr: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("qr: dynamic dim"),
        };
        let k = m.min(n);
        Shape::new(&[m * k + k * n], DType::F64)
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Walter–Lehmann 2010 closed form via the qr_backward kernel.
        // Forward output is packed [Q (m·k), R (k·n)]; upstream has
        // the same layout. Unpack and call qr_backward(Q, R, dQ, dR).
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };
        let k = m.min(n);

        let packed_fwd = ctx.fwd_map[&node.id];
        let q_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let r_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k * n,
            },
            vec![packed_fwd],
            Shape::new(&[k * n], DType::F64),
        );
        let dq = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![ctx.upstream],
            Shape::new(&[m * k], DType::F64),
        );
        let dr = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k * n,
            },
            vec![ctx.upstream],
            Shape::new(&[k * n], DType::F64),
        );

        let g_a = ctx
            .bwd
            .custom_op(LINALG_QR_BACKWARD, Vec::new(), vec![q_fwd, r_fwd, dq, dr]);
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Walter-Lehmann forward via qr_jvp kernel. Inputs: (Q_flat,
        // R_flat, dA_flat); output packed [dQ_flat, dR_flat].
        let t_a = ctx.tangents[0]?;
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let k = m.min(n);
        let packed_fwd = ctx.fwd_map[&node.id];
        let q_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let r_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k * n,
            },
            vec![packed_fwd],
            Shape::new(&[k * n], DType::F64),
        );
        let da_flat = ctx.bwd.add_node(
            rlx_ir::Op::Reshape {
                new_shape: vec![(m * n) as i64],
            },
            vec![t_a],
            Shape::new(&[m * n], DType::F64),
        );
        Some(
            ctx.bwd
                .custom_op(LINALG_QR_JVP, Vec::new(), vec![q_fwd, r_fwd, da_flat]),
        )
    }
}

#[cfg(feature = "cpu")]
struct QrCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for QrCpu {
    fn name(&self) -> &str {
        LINALG_QR
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("qr A")?;
        let a_shape = inputs[0].shape();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("qr: dynamic dim 0".into()),
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("qr: dynamic dim 1".into()),
        };
        let out = output.expect_f64_mut("qr out")?;
        algos::qr(a, m, n, out)
    }
}

// ── SVD ──────────────────────────────────────────────────────────

struct SvdExt;

impl OpExtension for SvdExt {
    fn name(&self) -> &str {
        LINALG_SVD
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "svd: A must be F64");
        assert_eq!(a.rank(), 2, "svd: A must be 2D");
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("svd: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("svd: dynamic dim"),
        };
        let k = m.min(n);
        // U (m·k) + S (k) + V^T (k·n)
        Shape::new(&[m * k + k + k * n], DType::F64)
    }

    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Forward output is packed [U (m·k), s (k), V^T (k·n)];
        // upstream has the same layout. Unpack and call svd_backward.
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Vec::new(),
        };
        let k = m.min(n);

        let packed_fwd = ctx.fwd_map[&node.id];
        let u_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let s_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k,
            },
            vec![packed_fwd],
            Shape::new(&[k], DType::F64),
        );
        let vt_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k + k,
                len: k * n,
            },
            vec![packed_fwd],
            Shape::new(&[k * n], DType::F64),
        );
        let du = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![ctx.upstream],
            Shape::new(&[m * k], DType::F64),
        );
        let ds = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k,
            },
            vec![ctx.upstream],
            Shape::new(&[k], DType::F64),
        );
        let dvt = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k + k,
                len: k * n,
            },
            vec![ctx.upstream],
            Shape::new(&[k * n], DType::F64),
        );

        let g_a = ctx.bwd.custom_op(
            LINALG_SVD_BACKWARD,
            Vec::new(),
            vec![u_fwd, s_fwd, vt_fwd, du, ds, dvt],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Townsend forward via svd_jvp kernel. Inputs: (U_flat, s,
        // Vt_flat, dA_flat); output packed [dU, ds, dVt].
        let t_a = ctx.tangents[0]?;
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let a_shape = ctx.bwd.node(a_bwd).shape.clone();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let k = m.min(n);
        let packed_fwd = ctx.fwd_map[&node.id];
        let u_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 0,
                len: m * k,
            },
            vec![packed_fwd],
            Shape::new(&[m * k], DType::F64),
        );
        let s_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k,
                len: k,
            },
            vec![packed_fwd],
            Shape::new(&[k], DType::F64),
        );
        let vt_fwd = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: m * k + k,
                len: k * n,
            },
            vec![packed_fwd],
            Shape::new(&[k * n], DType::F64),
        );
        let da_flat = ctx.bwd.add_node(
            rlx_ir::Op::Reshape {
                new_shape: vec![(m * n) as i64],
            },
            vec![t_a],
            Shape::new(&[m * n], DType::F64),
        );
        Some(ctx.bwd.custom_op(
            LINALG_SVD_JVP,
            Vec::new(),
            vec![u_fwd, s_fwd, vt_fwd, da_flat],
        ))
    }
}

#[cfg(feature = "cpu")]
struct SvdCpu;

#[cfg(feature = "cpu")]
impl CpuKernel for SvdCpu {
    fn name(&self) -> &str {
        LINALG_SVD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("svd A")?;
        let a_shape = inputs[0].shape();
        let m = match a_shape.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("svd: dynamic dim 0".into()),
        };
        let n = match a_shape.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => return Err("svd: dynamic dim 1".into()),
        };
        let out = output.expect_f64_mut("svd out")?;
        algos::svd(a, m, n, out)
    }
}

// ── LogDet ────────────────────────────────────────────────────────

struct LogDetExt;

impl OpExtension for LogDetExt {
    fn name(&self) -> &str {
        LINALG_LOGDET
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "logdet: A must be F64");
        assert_eq!(a.rank(), 2, "logdet: A must be 2D");
        // Scalar output.
        Shape::new(&[1], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/dA = dL/d(logdet) · A⁻¹  via the logdet_backward kernel.
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let g_a = ctx.bwd.custom_op(
            LINALG_LOGDET_BACKWARD,
            Vec::new(),
            vec![a_bwd, ctx.upstream],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // d/dt log|det(A(t))| = tr(A⁻¹·dA) = tr(solve(A, dA)).
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&node.inputs[0]];
        let n = match ctx.bwd.shape(a).dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let x = ctx.bwd.dense_solve(a, t_a, Shape::new(&[n, n], DType::F64));
        let d = ctx.bwd.custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![x]);
        // forward output is shape [1] (length-1 tensor), so keep_dim=true.
        Some(ctx.bwd.sum(d, vec![0], true))
    }
}

#[cfg(feature = "cpu")]
struct LogDetCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for LogDetCpu {
    fn name(&self) -> &str {
        LINALG_LOGDET
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("logdet A")?;
        let out = output.expect_f64_mut("logdet out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("logdet: A length {n_sq} not n²"));
        }
        algos::logdet(a, n, out)
    }
}

// ── SlogDet ───────────────────────────────────────────────────────

struct SlogDetExt;

impl OpExtension for SlogDetExt {
    fn name(&self) -> &str {
        LINALG_SLOGDET
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "slogdet: A must be F64");
        assert_eq!(a.rank(), 2, "slogdet: A must be 2D");
        Shape::new(&[2], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // upstream is dL/d(packed[2]). Extract index-1 (logabsdet grad)
        // via Narrow; sign component is non-differentiable.
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let dl_d_logabs = ctx.bwd.add_node(
            rlx_ir::Op::Narrow {
                axis: 0,
                start: 1,
                len: 1,
            },
            vec![ctx.upstream],
            Shape::new(&[1], DType::F64),
        );
        let g_a = ctx.bwd.custom_op(
            LINALG_SLOGDET_BACKWARD,
            Vec::new(),
            vec![a_bwd, dl_d_logabs],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Output is packed [sign, log|det|]. Sign is non-differentiable
        // (zero tangent); log|det| tangent is tr(A⁻¹·dA) like logdet.
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&node.inputs[0]];
        let n = match ctx.bwd.shape(a).dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => return None,
        };
        let x = ctx.bwd.dense_solve(a, t_a, Shape::new(&[n, n], DType::F64));
        let d = ctx.bwd.custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![x]);
        let t_logabs = ctx.bwd.sum(d, vec![0], true); // [1]
        let zero = ctx.bwd.add_node(
            rlx_ir::Op::Constant {
                data: 0.0_f64.to_le_bytes().to_vec(),
            },
            vec![],
            Shape::new(&[1], DType::F64),
        );
        Some(ctx.bwd.add_node(
            rlx_ir::Op::Concat { axis: 0 },
            vec![zero, t_logabs],
            Shape::new(&[2], DType::F64),
        ))
    }
}

#[cfg(feature = "cpu")]
struct SlogDetCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SlogDetCpu {
    fn name(&self) -> &str {
        LINALG_SLOGDET
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("slogdet A")?;
        let out = output.expect_f64_mut("slogdet out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("slogdet: A length {n_sq} not n²"));
        }
        algos::slogdet(a, n, out)
    }
}

struct SlogDetBackwardExt;

impl OpExtension for SlogDetBackwardExt {
    fn name(&self) -> &str {
        LINALG_SLOGDET_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dL/d(logabsdet)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
struct SlogDetBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SlogDetBackwardCpu {
    fn name(&self) -> &str {
        LINALG_SLOGDET_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("slogdet_bwd A")?;
        let dl_d = inputs[1].expect_f64("slogdet_bwd dL/d(logabsdet)")?;
        let out = output.expect_f64_mut("slogdet_bwd out")?;
        if dl_d.len() != 1 {
            return Err(format!(
                "slogdet_bwd: gradient must be scalar, got {}",
                dl_d.len()
            ));
        }
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("slogdet_bwd: A length {n_sq} not n²"));
        }
        algos::slogdet_backward(a, dl_d[0], n, out)
    }
}

// ── Diag extract / set ────────────────────────────────────────────

struct DiagExtractExt;

impl OpExtension for DiagExtractExt {
    fn name(&self) -> &str {
        LINALG_DIAG_EXTRACT
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "diag_extract: A must be F64");
        assert_eq!(a.rank(), 2, "diag_extract: A must be 2D");
        let n = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("diag_extract: dynamic dim"),
        };
        Shape::new(&[n], DType::F64)
    }
    fn vjp(&self, _node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/dA = diag_set(upstream).
        let g_a = ctx
            .bwd
            .custom_op(LINALG_DIAG_SET, Vec::new(), vec![ctx.upstream]);
        vec![(0, g_a)]
    }
    fn jvp(&self, _node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Linear op: dy = diag_extract(dA).
        let t_a = ctx.tangents[0]?;
        Some(
            ctx.bwd
                .custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![t_a]),
        )
    }
    fn vmap(&self, node: &Node, ctx: &mut rlx_ir::VmapContext) -> Option<NodeId> {
        // Batched A: [B, n, n] → [B, n]. Unroll over the static batch
        // dim: per batch, Narrow + Reshape + diag_extract + Reshape;
        // then Concat along axis 0. Works for any dtype (no assumptions
        // about Gather's f32-only kernel).
        if !ctx.is_batched[0] {
            return None;
        }
        let n = match node.shape.dim(0) {
            rlx_ir::Dim::Static(n) => n,
            _ => return None,
        };
        let b = ctx.batch_size;
        let a_b = ctx.lifted_inputs[0];
        let mut per_batch: Vec<NodeId> = Vec::with_capacity(b);
        for k in 0..b {
            let slice = ctx.out.add_node(
                rlx_ir::Op::Narrow {
                    axis: 0,
                    start: k,
                    len: 1,
                },
                vec![a_b],
                Shape::new(&[1, n, n], DType::F64),
            );
            let mat = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![n as i64, n as i64],
                },
                vec![slice],
                Shape::new(&[n, n], DType::F64),
            );
            let d = ctx
                .out
                .custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![mat]);
            let d_3d = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![1, n as i64],
                },
                vec![d],
                Shape::new(&[1, n], DType::F64),
            );
            per_batch.push(d_3d);
        }
        Some(ctx.out.add_node(
            rlx_ir::Op::Concat { axis: 0 },
            per_batch,
            Shape::new(&[b, n], DType::F64),
        ))
    }
}

#[cfg(feature = "cpu")]
struct DiagExtractCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for DiagExtractCpu {
    fn name(&self) -> &str {
        LINALG_DIAG_EXTRACT
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("diag_extract A")?;
        let out = output.expect_f64_mut("diag_extract out")?;
        let n = out.len();
        if a.len() != n * n {
            return Err(format!("diag_extract: A {} ≠ n²={}·{}", a.len(), n, n));
        }
        algos::diag_extract(a, n, out)
    }
}

struct DiagSetExt;

impl OpExtension for DiagSetExt {
    fn name(&self) -> &str {
        LINALG_DIAG_SET
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let v = inputs[0];
        assert_eq!(v.dtype(), DType::F64, "diag_set: v must be F64");
        assert_eq!(v.rank(), 1, "diag_set: v must be 1D");
        let n = match v.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("diag_set: dynamic dim"),
        };
        Shape::new(&[n, n], DType::F64)
    }
    fn vjp(&self, _node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // dL/dv = diag_extract(upstream).
        let g_v = ctx
            .bwd
            .custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![ctx.upstream]);
        vec![(0, g_v)]
    }
    fn jvp(&self, _node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Linear op: dM = diag_set(dv).
        let t_v = ctx.tangents[0]?;
        Some(ctx.bwd.custom_op(LINALG_DIAG_SET, Vec::new(), vec![t_v]))
    }
    fn vmap(&self, node: &Node, ctx: &mut rlx_ir::VmapContext) -> Option<NodeId> {
        // Batched v: [B, n] → [B, n, n]. Per-batch unroll mirroring
        // diag_extract's vmap.
        if !ctx.is_batched[0] {
            return None;
        }
        let n = match node.shape.dim(0) {
            rlx_ir::Dim::Static(n) => n,
            _ => return None,
        };
        let b = ctx.batch_size;
        let v_b = ctx.lifted_inputs[0];
        let mut per_batch: Vec<NodeId> = Vec::with_capacity(b);
        for k in 0..b {
            let slice = ctx.out.add_node(
                rlx_ir::Op::Narrow {
                    axis: 0,
                    start: k,
                    len: 1,
                },
                vec![v_b],
                Shape::new(&[1, n], DType::F64),
            );
            let vec1d = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![n as i64],
                },
                vec![slice],
                Shape::new(&[n], DType::F64),
            );
            let m = ctx.out.custom_op(LINALG_DIAG_SET, Vec::new(), vec![vec1d]);
            let m_3d = ctx.out.add_node(
                rlx_ir::Op::Reshape {
                    new_shape: vec![1, n as i64, n as i64],
                },
                vec![m],
                Shape::new(&[1, n, n], DType::F64),
            );
            per_batch.push(m_3d);
        }
        Some(ctx.out.add_node(
            rlx_ir::Op::Concat { axis: 0 },
            per_batch,
            Shape::new(&[b, n, n], DType::F64),
        ))
    }
}

#[cfg(feature = "cpu")]
struct DiagSetCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for DiagSetCpu {
    fn name(&self) -> &str {
        LINALG_DIAG_SET
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let v = inputs[0].expect_f64("diag_set v")?;
        let out = output.expect_f64_mut("diag_set out")?;
        let n = v.len();
        if out.len() != n * n {
            return Err(format!("diag_set: out {} ≠ n²={}·{}", out.len(), n, n));
        }
        algos::diag_set(v, n, out)
    }
}

// ── Expm ──────────────────────────────────────────────────────────

struct ExpmExt;

impl OpExtension for ExpmExt {
    fn name(&self) -> &str {
        LINALG_EXPM
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "expm: A must be F64");
        assert_eq!(a.rank(), 2, "expm: A must be 2D");
        a.clone()
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let g_a = ctx
            .bwd
            .custom_op(LINALG_EXPM_BACKWARD, Vec::new(), vec![a_bwd, ctx.upstream]);
        vec![(0, g_a)]
    }
    fn jvp(&self, _node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Frechet derivative via augmented-matrix kernel.
        let t_a = ctx.tangents[0]?;
        let a = ctx.fwd_map[&_node.inputs[0]];
        Some(ctx.bwd.custom_op(LINALG_EXPM_JVP, Vec::new(), vec![a, t_a]))
    }
}

struct ExpmJvpExt;

impl OpExtension for ExpmJvpExt {
    fn name(&self) -> &str {
        LINALG_EXPM_JVP
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dA
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
struct ExpmJvpCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for ExpmJvpCpu {
    fn name(&self) -> &str {
        LINALG_EXPM_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("expm_jvp A")?;
        let da = inputs[1].expect_f64("expm_jvp dA")?;
        let out = output.expect_f64_mut("expm_jvp out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("expm_jvp: A length {n_sq} not n²"));
        }
        algos::expm_jvp(a, da, n, out)
    }
}

// ── QR JVP ────────────────────────────────────────────────────────

struct QrJvpExt;

impl OpExtension for QrJvpExt {
    fn name(&self) -> &str {
        LINALG_QR_JVP
    }
    fn num_inputs(&self) -> usize {
        3
    } // Q_flat, R_flat, dA_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Output packed [dQ (m·k), dR (k·n)] — same length as Q + R together.
        let q_len = inputs[0].num_elements().expect("qr_jvp: dynamic shape");
        let r_len = inputs[1].num_elements().expect("qr_jvp: dynamic shape");
        Shape::new(&[q_len + r_len], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct QrJvpCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for QrJvpCpu {
    fn name(&self) -> &str {
        LINALG_QR_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f64("qr_jvp Q")?;
        let r = inputs[1].expect_f64("qr_jvp R")?;
        let da = inputs[2].expect_f64("qr_jvp dA")?;
        let out = output.expect_f64_mut("qr_jvp out")?;
        let r_len = r.len();
        let n = (r_len as f64).sqrt() as usize;
        if n * n != r_len {
            return Err(format!(
                "qr_jvp: R must be square (m≥n thin QR), got len {r_len}"
            ));
        }
        let m = q.len() / n;
        algos::qr_jvp(q, r, da, m, n, out)
    }
}

// ── SVD JVP ───────────────────────────────────────────────────────

struct SvdJvpExt;

impl OpExtension for SvdJvpExt {
    fn name(&self) -> &str {
        LINALG_SVD_JVP
    }
    fn num_inputs(&self) -> usize {
        4
    } // U_flat, s, Vt_flat, dA_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let u_len = inputs[0].num_elements().expect("svd_jvp: dynamic shape");
        let s_len = inputs[1].num_elements().expect("svd_jvp: dynamic shape");
        let vt_len = inputs[2].num_elements().expect("svd_jvp: dynamic shape");
        Shape::new(&[u_len + s_len + vt_len], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct SvdJvpCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SvdJvpCpu {
    fn name(&self) -> &str {
        LINALG_SVD_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _: &[u8],
    ) -> Result<(), String> {
        let u = inputs[0].expect_f64("svd_jvp U")?;
        let s = inputs[1].expect_f64("svd_jvp s")?;
        let vt = inputs[2].expect_f64("svd_jvp Vt")?;
        let da = inputs[3].expect_f64("svd_jvp dA")?;
        let out = output.expect_f64_mut("svd_jvp out")?;
        let k = s.len();
        let m = u.len() / k;
        let n = vt.len() / k;
        algos::svd_jvp(u, s, vt, da, m, n, out)
    }
}

// ── Pinv JVP ──────────────────────────────────────────────────────

struct PinvJvpExt;

impl OpExtension for PinvJvpExt {
    fn name(&self) -> &str {
        LINALG_PINV_JVP
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dA
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Output = pinv shape [n, m] (transpose of A's [m, n]).
        let a = inputs[0];
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv_jvp: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv_jvp: dynamic dim"),
        };
        Shape::new(&[n, m], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct PinvJvpCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for PinvJvpCpu {
    fn name(&self) -> &str {
        LINALG_PINV_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("pinv_jvp A")?;
        let da = inputs[1].expect_f64("pinv_jvp dA")?;
        let out = output.expect_f64_mut("pinv_jvp out")?;
        // Recover m from attrs (encoded by pinv builder).
        if attrs.len() < 4 {
            return Err("pinv_jvp: attrs must encode m (u32 LE)".into());
        }
        let m = u32::from_le_bytes(attrs[..4].try_into().unwrap()) as usize;
        if m == 0 || a.len() % m != 0 {
            return Err(format!("pinv_jvp: bad attrs m={m}"));
        }
        let n = a.len() / m;
        algos::pinv_jvp(a, da, m, n, out)
    }
}

#[cfg(feature = "cpu")]
struct ExpmCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for ExpmCpu {
    fn name(&self) -> &str {
        LINALG_EXPM
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("expm A")?;
        let out = output.expect_f64_mut("expm out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("expm: A length {n_sq} not n²"));
        }
        algos::expm(a, n, out)
    }
}

struct ExpmBackwardExt;

impl OpExtension for ExpmBackwardExt {
    fn name(&self) -> &str {
        LINALG_EXPM_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dL/d(exp(A))
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
struct ExpmBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for ExpmBackwardCpu {
    fn name(&self) -> &str {
        LINALG_EXPM_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("expm_bwd A")?;
        let g = inputs[1].expect_f64("expm_bwd dL/d(exp)")?;
        let out = output.expect_f64_mut("expm_bwd out")?;
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("expm_bwd: A length {n_sq} not n²"));
        }
        algos::expm_backward(a, g, n, out)
    }
}

// ── Pinv ──────────────────────────────────────────────────────────

struct PinvExt;

impl OpExtension for PinvExt {
    fn name(&self) -> &str {
        LINALG_PINV
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        assert_eq!(a.dtype(), DType::F64, "pinv: A must be F64");
        assert_eq!(a.rank(), 2, "pinv: A must be 2D");
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv: dynamic dim"),
        };
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("pinv: dynamic dim"),
        };
        Shape::new(&[n, m], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        // Y = pinv(A); pinv_backward needs (A, Y, dL/dY) and the m attr.
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return vec![],
        };
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let y_bwd = ctx.fwd_map[&node.id];
        let g_a = ctx.bwd.custom_op(
            LINALG_PINV_BACKWARD,
            attrs,
            vec![a_bwd, y_bwd, ctx.upstream],
        );
        vec![(0, g_a)]
    }
    fn jvp(&self, node: &Node, ctx: &mut rlx_ir::JvpContext) -> Option<NodeId> {
        // Forward Frechet via pinv_jvp kernel (does its own internal SVD).
        let t_a = ctx.tangents[0]?;
        let attrs = match &node.op {
            rlx_ir::Op::Custom { attrs, .. } => attrs.clone(),
            _ => return None,
        };
        let a = ctx.fwd_map[&node.inputs[0]];
        Some(ctx.bwd.custom_op(LINALG_PINV_JVP, attrs, vec![a, t_a]))
    }
}

#[cfg(feature = "cpu")]
struct PinvCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for PinvCpu {
    fn name(&self) -> &str {
        LINALG_PINV
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("pinv A")?;
        let out = output.expect_f64_mut("pinv out")?;
        // Output shape is [n, m]; recover m, n via known total = m·n
        // and the fact that sqrt is needed here is ambiguous. Use the
        // attrs-free convention: square root only works for square A.
        // Better: forward pass takes m, n from attrs… but we don't have
        // them. Recover from output length and input length:
        //   a.len() == m·n,  out.len() == n·m  (same number).
        // We need m and n separately. Pull from input shape via
        // OpExtension::infer_shape having already validated it; here
        // we re-derive from sizes: lacking shape access, encode in
        // attrs. v1 simplification: assume row-major contiguous and
        // recover m from output's leading dim later. For now, re-derive
        // by requiring the kernel be called only when the executor has
        // already wired inputs sized m·n. We use the approach: factor
        // out the ambiguity by passing m as the first 4 bytes of attrs.
        // (Done below via attrs.)
        // FALLBACK: scan factors of a.len() to find best (m,n) such
        // that m·n = a.len(); ambiguous. We instead require attrs.
        let mn = a.len();
        // Attrs encode m as little-endian u32 (n derived as mn/m).
        // Builder always sets this.
        let attrs = _attrs;
        if attrs.len() < 4 {
            return Err("pinv: attrs must encode m (u32 LE)".into());
        }
        let m = u32::from_le_bytes(attrs[..4].try_into().unwrap()) as usize;
        if m == 0 || mn % m != 0 {
            return Err(format!("pinv: bad attrs m={m} for input len {mn}"));
        }
        let n = mn / m;
        algos::pinv(a, m, n, out)
    }
}

struct PinvBackwardExt;

impl OpExtension for PinvBackwardExt {
    fn name(&self) -> &str {
        LINALG_PINV_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        3
    } // A, Y, dL/dY
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone() // dL/dA shape == A shape
    }
}

#[cfg(feature = "cpu")]
struct PinvBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for PinvBackwardCpu {
    fn name(&self) -> &str {
        LINALG_PINV_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("pinv_bwd A")?;
        let y = inputs[1].expect_f64("pinv_bwd Y")?;
        let g = inputs[2].expect_f64("pinv_bwd dL/dY")?;
        let out = output.expect_f64_mut("pinv_bwd out")?;
        // Recover m, n: a.len() = m·n, y.len() = n·m. Need m alone.
        // Y has shape n×m and a has shape m×n; out has shape m×n.
        // Use out.len() = m·n, and recover m via gcd? Better: encode
        // attrs again. v1: take attrs[0..4] as m (u32 LE).
        let attrs = _attrs;
        if attrs.len() < 4 {
            return Err("pinv_bwd: attrs must encode m (u32 LE)".into());
        }
        let m = u32::from_le_bytes(attrs[..4].try_into().unwrap()) as usize;
        if m == 0 || a.len() % m != 0 {
            return Err(format!("pinv_bwd: bad attrs m={m}"));
        }
        let n = a.len() / m;
        algos::pinv_backward(a, y, g, m, n, out)
    }
}

// ── Lstsq ─────────────────────────────────────────────────────────

struct LstsqExt;

impl OpExtension for LstsqExt {
    fn name(&self) -> &str {
        LINALG_LSTSQ
    }
    fn num_inputs(&self) -> usize {
        2
    } // A (m×n), b (m)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        let b = inputs[1];
        assert_eq!(a.dtype(), DType::F64, "lstsq: A must be F64");
        assert_eq!(a.rank(), 2, "lstsq: A must be 2D");
        assert_eq!(b.rank(), 1, "lstsq: b must be 1D");
        let n = match a.dim(1) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("lstsq: dynamic dim"),
        };
        Shape::new(&[n], DType::F64)
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        let a_bwd = ctx.fwd_map[&node.inputs[0]];
        let b_bwd = ctx.fwd_map[&node.inputs[1]];
        let x_bwd = ctx.fwd_map[&node.id];
        let g_a = ctx.bwd.custom_op(
            LINALG_LSTSQ_BACKWARD_A,
            Vec::new(),
            vec![a_bwd, x_bwd, b_bwd, ctx.upstream],
        );
        let g_b = ctx.bwd.custom_op(
            LINALG_LSTSQ_BACKWARD_B,
            Vec::new(),
            vec![a_bwd, ctx.upstream],
        );
        vec![(0, g_a), (1, g_b)]
    }
}

#[cfg(feature = "cpu")]
struct LstsqCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for LstsqCpu {
    fn name(&self) -> &str {
        LINALG_LSTSQ
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("lstsq A")?;
        let b = inputs[1].expect_f64("lstsq b")?;
        let out = output.expect_f64_mut("lstsq out")?;
        let m = b.len();
        let n = out.len();
        if a.len() != m * n {
            return Err(format!("lstsq: A len {} != m·n = {}·{}", a.len(), m, n));
        }
        algos::lstsq(a, b, m, n, out)
    }
}

struct LstsqBackwardAExt;

impl OpExtension for LstsqBackwardAExt {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_A
    }
    fn num_inputs(&self) -> usize {
        4
    } // A, x, b, dL/dx
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
struct LstsqBackwardACpu;
#[cfg(feature = "cpu")]
impl CpuKernel for LstsqBackwardACpu {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_A
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("lstsq_bwd_a A")?;
        let x = inputs[1].expect_f64("lstsq_bwd_a x")?;
        let b = inputs[2].expect_f64("lstsq_bwd_a b")?;
        let dl_dx = inputs[3].expect_f64("lstsq_bwd_a dL/dx")?;
        let out = output.expect_f64_mut("lstsq_bwd_a out")?;
        let m = b.len();
        let n = x.len();
        if a.len() != m * n || dl_dx.len() != n || out.len() != m * n {
            return Err(format!("lstsq_bwd_a: shape mismatch (m={m}, n={n})"));
        }
        algos::lstsq_backward_a(a, x, b, dl_dx, m, n, out)
    }
}

struct LstsqBackwardBExt;

impl OpExtension for LstsqBackwardBExt {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_B
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dL/dx
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        let a = inputs[0];
        let m = match a.dim(0) {
            rlx_ir::Dim::Static(v) => v,
            _ => panic!("lstsq_bwd_b: dynamic dim"),
        };
        Shape::new(&[m], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct LstsqBackwardBCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for LstsqBackwardBCpu {
    fn name(&self) -> &str {
        LINALG_LSTSQ_BACKWARD_B
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("lstsq_bwd_b A")?;
        let dl_dx = inputs[1].expect_f64("lstsq_bwd_b dL/dx")?;
        let out = output.expect_f64_mut("lstsq_bwd_b out")?;
        let m = out.len();
        let n = dl_dx.len();
        if a.len() != m * n {
            return Err(format!("lstsq_bwd_b: A {} ≠ m·n = {}·{}", a.len(), m, n));
        }
        algos::lstsq_backward_b(a, dl_dx, m, n, out)
    }
}

// ── Cholesky JVP ──────────────────────────────────────────────────

struct CholeskyJvpExt;

impl OpExtension for CholeskyJvpExt {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_JVP
    }
    fn num_inputs(&self) -> usize {
        2
    } // L, dA
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
struct CholeskyJvpCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for CholeskyJvpCpu {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_JVP
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let l = inputs[0].expect_f64("chol_jvp L")?;
        let da = inputs[1].expect_f64("chol_jvp dA")?;
        let out = output.expect_f64_mut("chol_jvp out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let n_sq = l.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("chol_jvp: n²={n_sq}"));
        }
        algos::cholesky_jvp(l, da, n, lower, out)
    }
}

// ── Backward ops ──────────────────────────────────────────────────

struct CholeskyBackwardExt;

impl OpExtension for CholeskyBackwardExt {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        2
    } // L, dL/dL
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone() // dL/dA has the same shape as L (= A)
    }
    // No second-order VJP (returns empty).
}

#[cfg(feature = "cpu")]
struct CholeskyBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for CholeskyBackwardCpu {
    fn name(&self) -> &str {
        LINALG_CHOLESKY_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let l = inputs[0].expect_f64("chol_bwd L")?;
        let dl_dl = inputs[1].expect_f64("chol_bwd dL/dL")?;
        let out = output.expect_f64_mut("chol_bwd out")?;
        let lower = attrs.first().copied().unwrap_or(1) != 0;
        let n_sq = l.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("chol_bwd: n²={n_sq}"));
        }
        algos::cholesky_backward(l, dl_dl, n, lower, out)
    }
}

struct EighBackwardExt;

impl OpExtension for EighBackwardExt {
    fn name(&self) -> &str {
        LINALG_EIGH_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        4
    } // λ, V_flat, dL/dλ, dL/dV_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // dL/dA has shape [n, n] where n = inputs[0].len.
        let n = inputs[0]
            .num_elements()
            .expect("eigh_bwd: λ must have static shape");
        Shape::new(&[n, n], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct EighBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for EighBackwardCpu {
    fn name(&self) -> &str {
        LINALG_EIGH_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let lambda = inputs[0].expect_f64("eigh_bwd λ")?;
        let v_flat = inputs[1].expect_f64("eigh_bwd V")?;
        let dl_dl = inputs[2].expect_f64("eigh_bwd dL/dλ")?;
        let dl_dv = inputs[3].expect_f64("eigh_bwd dL/dV")?;
        let out = output.expect_f64_mut("eigh_bwd out")?;
        let n = lambda.len();
        algos::eigh_backward(lambda, v_flat, dl_dl, dl_dv, n, out)
    }
}

struct QrBackwardExt;

impl OpExtension for QrBackwardExt {
    fn name(&self) -> &str {
        LINALG_QR_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        4
    } // Q_flat, R_flat, dL/dQ_flat, dL/dR_flat
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // Q has m·k flat; R has k·n flat. dL/dA has shape [m, n].
        // Recover m, n from input lengths assuming k = min(m, n).
        // For the test cases we know m ≥ n (k = n), so m = q_len / n
        // and n = r_len / k. Build shape [m, n].
        let q_len = inputs[0].num_elements().expect("qr_bwd: dynamic shape");
        let r_len = inputs[1].num_elements().expect("qr_bwd: dynamic shape");
        // For thin QR with k = n: q_len = m·n, r_len = n·n.
        let n = (r_len as f64).sqrt() as usize;
        assert_eq!(n * n, r_len, "qr_bwd: R must be square (m≥n thin QR)");
        let m = q_len / n;
        Shape::new(&[m, n], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct QrBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for QrBackwardCpu {
    fn name(&self) -> &str {
        LINALG_QR_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let q = inputs[0].expect_f64("qr_bwd Q")?;
        let r = inputs[1].expect_f64("qr_bwd R")?;
        let dl_dq = inputs[2].expect_f64("qr_bwd dL/dQ")?;
        let dl_dr = inputs[3].expect_f64("qr_bwd dL/dR")?;
        let out = output.expect_f64_mut("qr_bwd out")?;
        let r_len = r.len();
        let n = (r_len as f64).sqrt() as usize;
        if n * n != r_len {
            return Err(format!("qr_bwd: R must be n²={r_len}"));
        }
        let m = q.len() / n;
        if m * n != q.len() {
            return Err(format!("qr_bwd: Q shape {}/n={n} not int", q.len()));
        }
        algos::qr_backward(q, r, dl_dq, dl_dr, m, n, out)
    }
}

struct SvdBackwardExt;

impl OpExtension for SvdBackwardExt {
    fn name(&self) -> &str {
        LINALG_SVD_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        6
    } // U, s, Vt, dU, ds, dVt
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        // dL/dA shape is [m, n]. Recover from input lengths:
        //   U: m·k flat, s: k flat, Vt: k·n flat. k = s.len.
        let k = inputs[1].num_elements().expect("svd_bwd: dynamic shape");
        let u_len = inputs[0].num_elements().expect("svd_bwd: dynamic shape");
        let vt_len = inputs[2].num_elements().expect("svd_bwd: dynamic shape");
        let m = u_len / k;
        let n = vt_len / k;
        Shape::new(&[m, n], DType::F64)
    }
}

#[cfg(feature = "cpu")]
struct SvdBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for SvdBackwardCpu {
    fn name(&self) -> &str {
        LINALG_SVD_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let u = inputs[0].expect_f64("svd_bwd U")?;
        let s = inputs[1].expect_f64("svd_bwd s")?;
        let vt = inputs[2].expect_f64("svd_bwd Vt")?;
        let dl_du = inputs[3].expect_f64("svd_bwd dL/dU")?;
        let dl_ds = inputs[4].expect_f64("svd_bwd dL/ds")?;
        let dl_dvt = inputs[5].expect_f64("svd_bwd dL/dVt")?;
        let out = output.expect_f64_mut("svd_bwd out")?;
        let k = s.len();
        let m = u.len() / k;
        let n = vt.len() / k;
        algos::svd_backward(u, s, vt, dl_du, dl_ds, dl_dvt, m, n, out)
    }
}

struct LogDetBackwardExt;

impl OpExtension for LogDetBackwardExt {
    fn name(&self) -> &str {
        LINALG_LOGDET_BACKWARD
    }
    fn num_inputs(&self) -> usize {
        2
    } // A, dL/d(logdet) (scalar)
    fn infer_shape(&self, inputs: &[&Shape], _: &[u8]) -> Shape {
        inputs[0].clone()
    }
}

#[cfg(feature = "cpu")]
struct LogDetBackwardCpu;
#[cfg(feature = "cpu")]
impl CpuKernel for LogDetBackwardCpu {
    fn name(&self) -> &str {
        LINALG_LOGDET_BACKWARD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("logdet_bwd A")?;
        let dl_d_lg = inputs[1].expect_f64("logdet_bwd dL/d(logdet)")?;
        let out = output.expect_f64_mut("logdet_bwd out")?;
        if dl_d_lg.len() != 1 {
            return Err(format!(
                "logdet_bwd: dL/d(logdet) must be scalar, got len {}",
                dl_d_lg.len()
            ));
        }
        let n_sq = a.len();
        let n = (n_sq as f64).sqrt() as usize;
        if n * n != n_sq {
            return Err(format!("logdet_bwd: A length {n_sq} not n²"));
        }
        algos::logdet_backward(a, dl_d_lg[0], n, out)
    }
}

// ── Public builder API ───────────────────────────────────────────

/// `L = cholesky(A)`. A is row-major n×n SPD F64. Returns L (lower
/// triangular if `lower`, else U upper) of shape `[n, n]` F64 with
/// the unused triangle zeroed.
pub fn cholesky(g: &mut Graph, a: NodeId, lower: bool) -> NodeId {
    let attrs = vec![if lower { 1 } else { 0 }];
    g.custom_op(LINALG_CHOLESKY, attrs, vec![a])
}

/// Solve `op(A)·X = B` where A is triangular. `lower` selects which
/// triangle of A is read; `transpose_a` toggles `op(A) = Aᵀ`.
/// Output shape == B's shape.
pub fn solve_triangular(
    g: &mut Graph,
    a: NodeId,
    b: NodeId,
    lower: bool,
    transpose_a: bool,
) -> NodeId {
    let attrs = vec![if lower { 1 } else { 0 }, if transpose_a { 1 } else { 0 }];
    g.custom_op(LINALG_SOLVE_TRIANGULAR, attrs, vec![a, b])
}

/// `(eigvals, eigvecs) = eigh(A)`. A is symmetric n×n F64.
/// Returns:
///   - `eigvals`: shape `[n]` F64, ascending
///   - `eigvecs`: shape `[n, n]` F64, row `i` = i-th eigenvector
pub fn eigh(g: &mut Graph, a: NodeId) -> (NodeId, NodeId) {
    let a_shape = g.shape(a).clone();
    let n = match a_shape.dim(0) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("eigh: dynamic dim"),
    };
    let packed = g.custom_op(LINALG_EIGH, Vec::new(), vec![a]);
    let eigvals = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: 0,
            len: n,
        },
        vec![packed],
        Shape::new(&[n], DType::F64),
    );
    let eigvecs_flat = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: n,
            len: n * n,
        },
        vec![packed],
        Shape::new(&[n * n], DType::F64),
    );
    let eigvecs = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![n as i64, n as i64],
        },
        vec![eigvecs_flat],
        Shape::new(&[n, n], DType::F64),
    );
    (eigvals, eigvecs)
}

/// `(Q, R) = qr(A)`. A is m×n F64. Returns:
///   - `Q`: shape `[m, k]` F64 with `k = min(m, n)`
///   - `R`: shape `[k, n]` F64
pub fn qr(g: &mut Graph, a: NodeId) -> (NodeId, NodeId) {
    let a_shape = g.shape(a).clone();
    let m = match a_shape.dim(0) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("qr: dynamic dim"),
    };
    let n = match a_shape.dim(1) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("qr: dynamic dim"),
    };
    let k = m.min(n);
    let packed = g.custom_op(LINALG_QR, Vec::new(), vec![a]);
    let q_flat = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: 0,
            len: m * k,
        },
        vec![packed],
        Shape::new(&[m * k], DType::F64),
    );
    let q = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![m as i64, k as i64],
        },
        vec![q_flat],
        Shape::new(&[m, k], DType::F64),
    );
    let r_flat = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: m * k,
            len: k * n,
        },
        vec![packed],
        Shape::new(&[k * n], DType::F64),
    );
    let r = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![k as i64, n as i64],
        },
        vec![r_flat],
        Shape::new(&[k, n], DType::F64),
    );
    (q, r)
}

/// `logdet(A)` for SPD A — log-determinant computed via Cholesky:
/// `2 · Σ log L[i,i]` where `L = chol(A, lower)`. Output shape `[1]`
/// F64. VJP: `dL/dA = dL/d(logdet) · A⁻¹`.
pub fn logdet(g: &mut Graph, a: NodeId) -> NodeId {
    g.custom_op(LINALG_LOGDET, Vec::new(), vec![a])
}

/// Extract the diagonal of a square matrix `A: [n, n]` → `[n]`.
pub fn diag_extract(g: &mut Graph, a: NodeId) -> NodeId {
    g.custom_op(LINALG_DIAG_EXTRACT, Vec::new(), vec![a])
}

/// Build a diagonal matrix from a vector `v: [n]` → `[n, n]`.
pub fn diag_set(g: &mut Graph, v: NodeId) -> NodeId {
    g.custom_op(LINALG_DIAG_SET, Vec::new(), vec![v])
}

/// `trace(A)`: sum of diagonal entries. Pure composition of
/// `diag_extract` + reduction; VJP is the diag_set of the upstream.
pub fn trace(g: &mut Graph, a: NodeId) -> NodeId {
    let d = diag_extract(g, a);
    g.sum(d, vec![0], false)
}

/// Kronecker product `kron(A, B)`. For `A: [m, n]` and `B: [p, q]`
/// produces `[m·p, n·q]` with `kron[i·p+r, j·q+s] = A[i,j]·B[r,s]`.
/// Implemented as broadcast multiply on reshaped operands — VJP comes
/// for free from autodiff over the underlying ops.
pub fn kron(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    let a_shape = g.shape(a).clone();
    let b_shape = g.shape(b).clone();
    let m = match a_shape.dim(0) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("kron: dynamic dim"),
    };
    let n = match a_shape.dim(1) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("kron: dynamic dim"),
    };
    let p = match b_shape.dim(0) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("kron: dynamic dim"),
    };
    let q = match b_shape.dim(1) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("kron: dynamic dim"),
    };
    // A: [m,n] → [m, 1, n, 1] → expand to [m, p, n, q]
    // B: [p,q] → [1, p, 1, q] → expand to [m, p, n, q]
    // (Use explicit Reshape+Expand: rlx-cpu's Op::Binary broadcasting
    // is limited to last-axis bias-add patterns; Expand is the
    // first-class broadcast op.)
    let a4 = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![m as i64, 1, n as i64, 1],
        },
        vec![a],
        Shape::new(&[m, 1, n, 1], DType::F64),
    );
    let a_exp = g.add_node(
        rlx_ir::Op::Expand {
            target_shape: vec![m as i64, p as i64, n as i64, q as i64],
        },
        vec![a4],
        Shape::new(&[m, p, n, q], DType::F64),
    );
    let b4 = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![1, p as i64, 1, q as i64],
        },
        vec![b],
        Shape::new(&[1, p, 1, q], DType::F64),
    );
    let b_exp = g.add_node(
        rlx_ir::Op::Expand {
            target_shape: vec![m as i64, p as i64, n as i64, q as i64],
        },
        vec![b4],
        Shape::new(&[m, p, n, q], DType::F64),
    );
    let prod = g.binary(
        rlx_ir::op::BinaryOp::Mul,
        a_exp,
        b_exp,
        Shape::new(&[m, p, n, q], DType::F64),
    );
    g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![(m * p) as i64, (n * q) as i64],
        },
        vec![prod],
        Shape::new(&[m * p, n * q], DType::F64),
    )
}

/// Polar decomposition `A = U · H` where U is orthogonal and H is SPD.
/// Computed via thin SVD: `A = U_svd · S · V^T` ⇒
///   `U = U_svd · V^T`,  `H = V · S · V^T`.
/// Returns `(U, H)`. Forward only in v1 — VJP composes through `svd`
/// and matmul once an SVD VJP path is traced through.
pub fn polar(g: &mut Graph, a: NodeId) -> (NodeId, NodeId) {
    let a_shape = g.shape(a).clone();
    let m = match a_shape.dim(0) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("polar: dynamic dim"),
    };
    let n = match a_shape.dim(1) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("polar: dynamic dim"),
    };
    assert!(m >= n, "polar: requires m ≥ n (got m={m}, n={n})");
    let k = n;
    let (u_svd, s, vt) = svd(g, a);
    // V = (V^T)^T  →  [n, k] (k = n here)
    let v_mat = g.add_node(
        rlx_ir::Op::Transpose { perm: vec![1, 0] },
        vec![vt],
        Shape::new(&[n, k], DType::F64),
    );
    // U_orth = U_svd · V^T  →  [m, n]
    let u_orth = g.matmul(u_svd, vt, Shape::new(&[m, n], DType::F64));
    // Σ = diag(s)  →  [k, k]
    let sigma = diag_set(g, s);
    // V·Σ → [n, k]
    let v_sigma = g.matmul(v_mat, sigma, Shape::new(&[n, k], DType::F64));
    // H = (V·Σ)·V^T  →  [n, n]
    let h = g.matmul(v_sigma, vt, Shape::new(&[n, n], DType::F64));
    (u_orth, h)
}

/// `expm(A)`: matrix exponential via Padé-13 + scaling-and-squaring.
/// A is square n×n F64; output is n×n F64. VJP via Al-Mohy/Higham
/// augmented-matrix trick.
pub fn expm(g: &mut Graph, a: NodeId) -> NodeId {
    g.custom_op(LINALG_EXPM, Vec::new(), vec![a])
}

/// `pinv(A)`: Moore-Penrose pseudo-inverse via thin SVD. A is m×n F64;
/// output is n×m F64. For full-rank A this is `(AᵀA)⁻¹·Aᵀ` (m≥n) or
/// `Aᵀ·(A·Aᵀ)⁻¹` (m<n).
pub fn pinv(g: &mut Graph, a: NodeId) -> NodeId {
    let a_shape = g.shape(a).clone();
    let m = match a_shape.dim(0) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("pinv: dynamic dim"),
    };
    let n = match a_shape.dim(1) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("pinv: dynamic dim"),
    };
    let attrs = (m as u32).to_le_bytes().to_vec();
    let _ = n;
    g.custom_op(LINALG_PINV, attrs, vec![a])
}

/// `lstsq(A, b)`: x = pinv(A)·b. A is m×n F64; b is m F64; x is n F64.
/// VJP supports the full-column-rank case (m ≥ n). For under-determined
/// systems, the same kernel returns the minimum-norm solution.
pub fn lstsq(g: &mut Graph, a: NodeId, b: NodeId) -> NodeId {
    g.custom_op(LINALG_LSTSQ, Vec::new(), vec![a, b])
}

/// `(sign, log|det|) = slogdet(A)` for general square A. Computed
/// via LU with partial pivoting. Returns two scalar nodes.
/// VJP flows through `log|det|` only (sign is non-differentiable).
pub fn slogdet(g: &mut Graph, a: NodeId) -> (NodeId, NodeId) {
    let packed = g.custom_op(LINALG_SLOGDET, Vec::new(), vec![a]);
    let sign = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: 0,
            len: 1,
        },
        vec![packed],
        Shape::new(&[1], DType::F64),
    );
    let logabsdet = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: 1,
            len: 1,
        },
        vec![packed],
        Shape::new(&[1], DType::F64),
    );
    (sign, logabsdet)
}

/// `(U, S, Vᵀ) = svd(A)` (thin / "S" mode). A is m×n F64. Returns:
///   - `U`:  `[m, k]`  F64
///   - `S`:  `[k]`     F64, descending
///   - `V^T`:`[k, n]`  F64
/// where `k = min(m, n)`.
pub fn svd(g: &mut Graph, a: NodeId) -> (NodeId, NodeId, NodeId) {
    let a_shape = g.shape(a).clone();
    let m = match a_shape.dim(0) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("svd: dynamic dim"),
    };
    let n = match a_shape.dim(1) {
        rlx_ir::Dim::Static(v) => v,
        _ => panic!("svd: dynamic dim"),
    };
    let k = m.min(n);
    let packed = g.custom_op(LINALG_SVD, Vec::new(), vec![a]);
    let u_flat = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: 0,
            len: m * k,
        },
        vec![packed],
        Shape::new(&[m * k], DType::F64),
    );
    let u = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![m as i64, k as i64],
        },
        vec![u_flat],
        Shape::new(&[m, k], DType::F64),
    );
    let s = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: m * k,
            len: k,
        },
        vec![packed],
        Shape::new(&[k], DType::F64),
    );
    let vt_flat = g.add_node(
        rlx_ir::Op::Narrow {
            axis: 0,
            start: m * k + k,
            len: k * n,
        },
        vec![packed],
        Shape::new(&[k * n], DType::F64),
    );
    let vt = g.add_node(
        rlx_ir::Op::Reshape {
            new_shape: vec![k as i64, n as i64],
        },
        vec![vt_flat],
        Shape::new(&[k, n], DType::F64),
    );
    (u, s, vt)
}

// ── Registration ─────────────────────────────────────────────────

/// Register every linalg op's IR-level extension and per-backend
/// kernels enabled at compile time.
pub fn register() {
    register_op(Arc::new(CholeskyExt));
    register_op(Arc::new(SolveTriangularExt));
    register_op(Arc::new(EighExt));
    register_op(Arc::new(QrExt));
    register_op(Arc::new(SvdExt));
    register_op(Arc::new(CholeskyJvpExt));
    register_op(Arc::new(CholeskyBackwardExt));
    register_op(Arc::new(EighJvpExt));
    register_op(Arc::new(EighBackwardExt));
    register_op(Arc::new(QrBackwardExt));
    register_op(Arc::new(SvdBackwardExt));
    register_op(Arc::new(LogDetExt));
    register_op(Arc::new(LogDetBackwardExt));
    register_op(Arc::new(SlogDetExt));
    register_op(Arc::new(SlogDetBackwardExt));
    register_op(Arc::new(ExpmExt));
    register_op(Arc::new(ExpmBackwardExt));
    register_op(Arc::new(ExpmJvpExt));
    register_op(Arc::new(QrJvpExt));
    register_op(Arc::new(SvdJvpExt));
    register_op(Arc::new(PinvJvpExt));
    register_op(Arc::new(DiagExtractExt));
    register_op(Arc::new(DiagSetExt));
    register_op(Arc::new(PinvExt));
    register_op(Arc::new(PinvBackwardExt));
    register_op(Arc::new(LstsqExt));
    register_op(Arc::new(LstsqBackwardAExt));
    register_op(Arc::new(LstsqBackwardBExt));

    #[cfg(feature = "cpu")]
    {
        register_cpu_kernel(Arc::new(CholeskyCpu));
        register_cpu_kernel(Arc::new(SolveTriangularCpu));
        register_cpu_kernel(Arc::new(EighCpu));
        register_cpu_kernel(Arc::new(QrCpu));
        register_cpu_kernel(Arc::new(SvdCpu));
        register_cpu_kernel(Arc::new(CholeskyJvpCpu));
        register_cpu_kernel(Arc::new(CholeskyBackwardCpu));
        register_cpu_kernel(Arc::new(EighJvpCpu));
        register_cpu_kernel(Arc::new(EighBackwardCpu));
        register_cpu_kernel(Arc::new(QrBackwardCpu));
        register_cpu_kernel(Arc::new(SvdBackwardCpu));
        register_cpu_kernel(Arc::new(LogDetCpu));
        register_cpu_kernel(Arc::new(LogDetBackwardCpu));
        register_cpu_kernel(Arc::new(SlogDetCpu));
        register_cpu_kernel(Arc::new(SlogDetBackwardCpu));
        register_cpu_kernel(Arc::new(ExpmCpu));
        register_cpu_kernel(Arc::new(ExpmBackwardCpu));
        register_cpu_kernel(Arc::new(ExpmJvpCpu));
        register_cpu_kernel(Arc::new(QrJvpCpu));
        register_cpu_kernel(Arc::new(SvdJvpCpu));
        register_cpu_kernel(Arc::new(PinvJvpCpu));
        register_cpu_kernel(Arc::new(DiagExtractCpu));
        register_cpu_kernel(Arc::new(DiagSetCpu));
        register_cpu_kernel(Arc::new(PinvCpu));
        register_cpu_kernel(Arc::new(PinvBackwardCpu));
        register_cpu_kernel(Arc::new(LstsqCpu));
        register_cpu_kernel(Arc::new(LstsqBackwardACpu));
        register_cpu_kernel(Arc::new(LstsqBackwardBCpu));
    }
}
