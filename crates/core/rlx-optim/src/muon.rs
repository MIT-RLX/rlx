// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Muon — MomentUm Orthogonalized by Newton–Schulz (Jordan, Bernstein,
//! Vyas, Hubara, et al., 2024).
//!
//! # Idea
//!
//! For a 2-D parameter, replace the momentum buffer with its **closest
//! semi-orthogonal matrix** before applying it as an update. The SVD
//! `M = U·Σ·Vᵀ` has closest semi-orthogonal matrix `U·Vᵀ` — but the
//! SVD is expensive. A *Newton–Schulz cubic iteration* approximates
//! `U·Vᵀ` in only 5 small matrix products per step. Empirically this
//! gives a step-size-invariant update that punches above its weight on
//! transformer training.
//!
//! # Update rule (2-D parameter `W ∈ ℝ^{m×n}`)
//!
//! ```text
//! m_t = μ·m_{t-1} + g_t                              // Polyak momentum
//! M   = m_t                  if !nesterov
//!     = g_t + μ·m_t          if  nesterov
//! M̂   = M / ‖M‖_F                                    // normalize for NS
//! repeat ns_steps times:                              // ns_steps = 5
//!     A = M̂ · M̂ᵀ
//!     M̂ ← a·M̂ + b·A·M̂ + c·A²·M̂                       // cubic NS iter
//! U   = √max(m, n) · M̂                                // RMS-of-cols scaling
//! θ_t = θ_{t-1} − lr · ( U + λ·θ_{t-1} )
//! ```
//!
//! The (a, b, c) coefficients are chosen so the cubic polynomial maps
//! singular values in (0, √3] toward 1; defaults
//! `(3.4445, −4.7750, 2.0315)` are from the original release.
//!
//! Non-2-D parameters fall back to SGD-with-momentum (the original
//! recipe routes them to AdamW; this crate stays dependency-free).
//!
//! # When to use
//!
//! Pre-training transformer matrix-shaped weights (Q/K/V/FFN
//! projections). Often paired with AdamW for embeddings and biases.
//! State cost: one momentum buffer per matrix.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// Muon — Momentum-Orthogonalized-by-Newton-Schulz.
///
/// Per-tensor state: **one** momentum buffer per matrix (half of
/// Adam's footprint, like Lion).
#[derive(Debug, Clone)]
pub struct Muon {
    /// Learning rate. The Newton–Schulz update has roughly unit
    /// Frobenius norm per column, so this is on the same scale as
    /// SGD's lr — typically `2e-2` to `5e-2`.
    pub lr: f32,
    /// Polyak momentum coefficient. Default `0.95`.
    pub momentum: f32,
    /// Use Nesterov lookahead inside the matrix being orthogonalized.
    /// Default `true`.
    pub nesterov: bool,
    /// Decoupled weight-decay coefficient λ. Default `0.0`.
    pub weight_decay: f32,
    /// Newton–Schulz iteration count. `5` is the published default;
    /// `3` is enough for most well-conditioned matrices.
    pub ns_steps: u32,
    /// `(a, b, c)` coefficients of the cubic Newton–Schulz iteration
    /// `X ← a·X + b·(XXᵀ)X + c·(XXᵀ)²X`. Defaults match Jordan et al.
    pub ns_coeffs: (f32, f32, f32),
    m: HashMap<String, Vec<f32>>,
}

impl Muon {
    /// Construct with `(μ, nesterov, λ, ns_steps) = (0.95, true, 0.0, 5)`
    /// and the published NS coefficients.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.95,
            nesterov: true,
            weight_decay: 0.0,
            ns_steps: 5,
            ns_coeffs: (3.4445, -4.7750, 2.0315),
            m: HashMap::new(),
        }
    }

    /// Override the Polyak momentum coefficient.
    pub fn with_momentum(mut self, mu: f32) -> Self {
        self.momentum = mu;
        self
    }

    /// Override the decoupled-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Override the Newton–Schulz iteration count.
    pub fn with_ns_steps(mut self, n: u32) -> Self {
        self.ns_steps = n;
        self
    }
}

impl Optimizer for Muon {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let mu = self.momentum;
        let wd = self.weight_decay;
        let lr = self.lr;
        let m = zeros_entry(&mut self.m, name, param.len());
        // EMA buffer (classical Polyak momentum: `m ← μ·m + g`).
        for i in 0..param.len() {
            m[i] = mu * m[i] + grad[i];
        }
        if shape.len() != 2 {
            // Non-matrix: SGD-with-momentum update.
            for i in 0..param.len() {
                let g = if self.nesterov {
                    grad[i] + mu * m[i]
                } else {
                    m[i]
                };
                param[i] -= lr * (g + wd * param[i]);
            }
            return;
        }
        let (rows, cols) = (shape[0], shape[1]);
        debug_assert_eq!(rows * cols, param.len());
        // Build the matrix to orthogonalize. With Nesterov:
        //   G = grad + μ·m   (m has already been updated above)
        let mut g_mat = vec![0.0f32; rows * cols];
        if self.nesterov {
            for i in 0..rows * cols {
                g_mat[i] = grad[i] + mu * m[i];
            }
        } else {
            g_mat.copy_from_slice(m);
        }
        let ortho = newton_schulz_orth(&g_mat, rows, cols, self.ns_steps, self.ns_coeffs);
        // The Muon paper scales the update by sqrt(max(rows, cols)) so
        // its effective magnitude matches a unit-norm column.
        let s = (rows.max(cols) as f32).sqrt();
        for i in 0..param.len() {
            param[i] -= lr * (s * ortho[i] + wd * param[i]);
        }
    }
}

/// Map `f(row_index, &mut row)` over the `n_rows × cols` row-major matrix
/// `out`, parallelized across output rows. With the `parallel` feature this uses
/// rayon's **persistent global pool** (no per-call thread spawn — the dominant
/// overhead when the Newton–Schulz iteration fires this many small matmuls per
/// step); without it, dependency-free scoped threads. Serial below 64 rows.
/// `f` reads only shared (immutable) state and writes its own disjoint row.
/// (Dead on macOS, where the Newton–Schulz path uses Accelerate BLAS instead.)
#[allow(dead_code)]
fn par_rows<F: Fn(usize, &mut [f32]) + Sync>(out: &mut [f32], cols: usize, f: F) {
    let n = out.len() / cols;
    if n < 64 {
        for (i, row) in out.chunks_mut(cols).enumerate() {
            f(i, row);
        }
        return;
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(i, row)| f(i, row));
    }
    #[cfg(not(feature = "parallel"))]
    {
        let threads = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .min(n);
        let per = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (t, chunk) in out.chunks_mut(per * cols).enumerate() {
                let f = &f;
                s.spawn(move || {
                    let base = t * per;
                    for (ii, row) in chunk.chunks_mut(cols).enumerate() {
                        f(base + ii, row);
                    }
                });
            }
        });
    }
}

/// Accelerate (Apple AMX/SME) BLAS shims for the Newton–Schulz matmuls. Linked
/// via `build.rs` (`framework=Accelerate`). Row-major `cblas_sgemm`; the weight
/// matrices are never empty so no zero-dim guard is needed.
#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod accel {
    unsafe extern "C" {
        #[link_name = "cblas_sgemm"]
        fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    /// `C[m×n] = A[m×k] · B[k×n]`.
    #[inline]
    pub fn gemm_nn(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }

    /// `C[m×n] = A[m×k] · B[n×k]ᵀ`.
    #[inline]
    pub fn gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                k as i32,
                0.0,
                c.as_mut_ptr(),
                n as i32,
            );
        }
    }
}

/// Newton–Schulz semi-orthogonalization. Operates on a row-major
/// `rows × cols` matrix and returns its closest semi-orthogonal matrix
/// (up to the polynomial truncation). The input is first scaled by its
/// Frobenius norm to stay inside the polynomial's region of convergence.
/// The per-iteration `X·Xᵀ`, `A²` and `X`-update products are parallelized
/// over rows, so it stays cheap even for the largest transformer matrices.
pub fn newton_schulz_orth(
    g: &[f32],
    rows: usize,
    cols: usize,
    steps: u32,
    c: (f32, f32, f32),
) -> Vec<f32> {
    let mut x = g.to_vec();
    // Frobenius normalization.
    let mut fro = 0.0f64;
    for &xi in &x {
        fro += xi as f64 * xi as f64;
    }
    let fro = (fro.sqrt() as f32).max(1e-12);
    for xi in &mut x {
        *xi /= fro;
    }
    // Operate with the SMALLER dimension as rows, so the `XXᵀ` Gram (`r×r`) is
    // `min×min` — the cubic NS iteration's cost is dominated by that `r×r`
    // matrix, and `U·Vᵀ` is identical in either orientation. (Transposing to
    // `min×max`, as the canonical Muon reference does, is up to (max/min)²
    // cheaper on the `A²` product than working on the `max×max` Gram.)
    let (mut x_mat, r, k, transposed) = if rows > cols {
        // transpose rows×cols → cols×rows so r = cols = min
        let mut t = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                t[j * rows + i] = x[i * cols + j];
            }
        }
        (t, cols, rows, true)
    } else {
        (x, rows, cols, false)
    };
    let (a, b, cc) = c;
    let mut tmp = vec![0.0f32; r * k]; // XXᵀ X has shape r × k
    let mut a_mat = vec![0.0f32; r * r];
    let mut a2 = vec![0.0f32; r * r];
    for _ in 0..steps {
        // On macOS the three per-iteration products (`X·Xᵀ`, `A²`, and the
        // `X`-update contraction) are the whole cost of Muon on large models —
        // route them through Accelerate's `cblas_sgemm`, which dispatches to the
        // AMX / SME matrix coprocessor and is ~10-50× the hand-rolled loop.
        #[cfg(target_os = "macos")]
        {
            accel::gemm_nt(&x_mat, &x_mat, &mut a_mat, r, k, r); // A = X · Xᵀ
            accel::gemm_nn(&a_mat, &a_mat, &mut a2, r, r, r); // A² = A · A
            for i in 0..r * r {
                a2[i] = b * a_mat[i] + cc * a2[i]; // reuse a2 as (b·A + cc·A²)
            }
            accel::gemm_nn(&a2, &x_mat, &mut tmp, r, r, k); // tmp = (b·A + cc·A²)·X
            for i in 0..r * k {
                tmp[i] += a * x_mat[i]; // tmp += a·X
            }
        }
        // Portable parallel fallback (non-macOS): cache-friendly `ikj` matmuls.
        #[cfg(not(target_os = "macos"))]
        {
            // A = X · Xᵀ — dot products of contiguous rows.
            par_rows(&mut a_mat, r, |i, arow| {
                let xi = &x_mat[i * k..i * k + k];
                for (j, aij) in arow.iter_mut().enumerate() {
                    let xj = &x_mat[j * k..j * k + k];
                    let mut s = 0.0f32;
                    for p in 0..k {
                        s += xi[p] * xj[p];
                    }
                    *aij = s;
                }
            });
            // A² = A · A.
            par_rows(&mut a2, r, |i, a2row| {
                a2row.fill(0.0);
                let ai = &a_mat[i * r..i * r + r];
                for p in 0..r {
                    let aip = ai[p];
                    let ap = &a_mat[p * r..p * r + r];
                    for j in 0..r {
                        a2row[j] += aip * ap[j];
                    }
                }
            });
            // X ← a·X + b·A·X + cc·A²·X.
            par_rows(&mut tmp, k, |i, trow| {
                let xi = &x_mat[i * k..i * k + k];
                for j in 0..k {
                    trow[j] = a * xi[j];
                }
                let ai = &a_mat[i * r..i * r + r];
                let a2i = &a2[i * r..i * r + r];
                for p in 0..r {
                    let coef = b * ai[p] + cc * a2i[p];
                    let xp = &x_mat[p * k..p * k + k];
                    for j in 0..k {
                        trow[j] += coef * xp[j];
                    }
                }
            });
        }
        std::mem::swap(&mut x_mat, &mut tmp);
    }
    if transposed {
        // Transpose back to rows × cols.
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..r {
            for j in 0..k {
                out[j * r + i] = x_mat[i * k + j];
            }
        }
        out
    } else {
        x_mat
    }
}
