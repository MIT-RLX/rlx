// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stiefel — Riemannian SGD-with-momentum on the (compact) Stiefel
//! manifold `St(m, n) = { W ∈ ℝ^{m×n} : W·Wᵀ = I_m }` (row-orthonormal,
//! `m ≤ n`).
//!
//! # Why
//!
//! SPDNet's BiMap layer weights are constrained to be semi-orthogonal
//! (`W·Wᵀ = I_m`). A plain Euclidean optimizer ignores that constraint
//! and lets `W` drift off the manifold within a few steps, so the layer
//! stops being a valid bilinear map on SPD matrices. This optimizer
//! keeps every step *on* the manifold: it projects the Euclidean
//! gradient to the tangent space, moves along the tangent, then
//! **retracts** back to `St(m, n)`.
//!
//! # Update rule (2-D parameter `W ∈ ℝ^{m×n}`, `m ≤ n`)
//!
//! ```text
//! sym(A) = ½·(A + Aᵀ)
//! G_riem = G − W · sym(Wᵀ·G)          // project Euclidean grad to T_W St
//! v_t    = μ·v_{t-1} + G_riem          // (optional) tangent momentum
//! Y      = W − lr · v_t                // Euclidean step off the manifold
//! W_new  = qf(Y)                        // QR retraction back onto St(m,n)
//! ```
//!
//! where `qf(Y)` is the row-orthonormal factor of `Y`: we take the thin
//! QR of `Yᵀ` (shape `n×m`, tall) and return `Qᵀ` (shape `m×n`) so that
//! `W_new·W_newᵀ = I_m`. The QR is a dependency-free modified
//! Gram–Schmidt on the rows of `Y`. The sign convention of `qf`
//! (diagonal of `R` forced positive) keeps the retraction a smooth map
//! agreeing with the exponential to first order — the standard QR
//! retraction of Absil, Mahony & Sepulchre, *Optimization Algorithms on
//! Matrix Manifolds* (Princeton, 2008), §4.1.
//!
//! # Non-Stiefel parameters
//!
//! Only 2-D parameters with `rows ≤ cols` are treated as Stiefel
//! points. Any other shape (bias vectors, `rows > cols`, higher-rank
//! tensors) falls back to plain SGD-with-momentum so a single `Stiefel`
//! instance can still drive an entire SPDNet (BiMap weights on the
//! manifold, everything else Euclidean).
//!
//! # State
//!
//! One tangent-momentum buffer per parameter (only allocated / used when
//! `momentum > 0`), same footprint as [`crate::Sgd`].

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// Riemannian SGD-with-momentum on the Stiefel manifold.
///
/// All hyperparameters are public so callers can hot-swap them between
/// iterations. State is keyed by parameter name.
#[derive(Debug, Clone)]
pub struct Stiefel {
    /// Learning rate (tangent step size). No default — pass to
    /// [`Stiefel::new`].
    pub lr: f32,
    /// Polyak momentum coefficient μ ∈ \[0, 1). `0.0` disables the
    /// tangent-momentum buffer entirely. Default `0.0`.
    pub momentum: f32,
    m: HashMap<String, Vec<f32>>,
}

impl Stiefel {
    /// Construct with the given learning rate and momentum disabled.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            m: HashMap::new(),
        }
    }

    /// Enable Polyak momentum on the tangent update.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for Stiefel {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let lr = self.lr;
        let mu = self.momentum;

        // Non-Stiefel shapes: plain SGD-with-momentum fallback.
        let is_stiefel = shape.len() == 2 && shape[0] <= shape[1];
        if !is_stiefel {
            if mu == 0.0 {
                for i in 0..param.len() {
                    param[i] -= lr * grad[i];
                }
            } else {
                let m = zeros_entry(&mut self.m, name, param.len());
                for i in 0..param.len() {
                    m[i] = mu * m[i] + grad[i];
                    param[i] -= lr * m[i];
                }
            }
            return;
        }

        let (rows, cols) = (shape[0], shape[1]);
        debug_assert_eq!(rows * cols, param.len());

        // ── Riemannian gradient ─────────────────────────────────────
        // sym = ½(WᵀG + GᵀW) is m×m via Wᵀ·G symmetrized; but we need
        // W·sym(Wᵀ·G) which is m×n. Compute S = Wᵀ·G? No: for the
        // row-orthonormal convention the projection is
        //   G_riem = G − W · sym(Wᵀ·G_no)   where sym acts on the m×m
        // matrix  M = W·Gᵀ  (since Wᵀ·G would be n×n).
        //
        // Tangent space of St(m,n) at W (rows orthonormal):
        //   T_W = { Z : W·Zᵀ + Z·Wᵀ = 0 }.
        // The orthogonal projection of an ambient G onto T_W is
        //   P_W(G) = G − W · sym(W·Gᵀ),   sym(A)=½(A+Aᵀ),  A = W·Gᵀ (m×m).
        // (Edelman, Arias & Smith 1998; ASM 2008 §3.6.1, transposed to
        // the wide/row-orthonormal layout.)
        let a = mat_mul_bt(param, grad, rows, cols); // A = W·Gᵀ  (m×m)
        let mut symm = vec![0.0f32; rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                symm[i * rows + j] = 0.5 * (a[i * rows + j] + a[j * rows + i]);
            }
        }
        // W·sym  (m×n)
        let wsym = mat_mul(&symm, param, rows, rows, cols);
        let mut g_riem = vec![0.0f32; rows * cols];
        for i in 0..rows * cols {
            g_riem[i] = grad[i] - wsym[i];
        }

        // ── Tangent step (+ optional momentum) ──────────────────────
        let mut y = vec![0.0f32; rows * cols];
        if mu == 0.0 {
            for i in 0..rows * cols {
                y[i] = param[i] - lr * g_riem[i];
            }
        } else {
            let m = zeros_entry(&mut self.m, name, rows * cols);
            for i in 0..rows * cols {
                m[i] = mu * m[i] + g_riem[i];
                y[i] = param[i] - lr * m[i];
            }
        }

        // ── QR retraction back onto St(m,n) ─────────────────────────
        // qf(Y) = row-orthonormal factor. Orthonormalize the rows of Y
        // (modified Gram–Schmidt); result satisfies W_new·W_newᵀ = I_m.
        qf_rows_in_place(&mut y, rows, cols);
        param.copy_from_slice(&y);
    }
}

/// Row-major `A · B` with `A: m×k`, `B: k×n` → `m×n`. Local copy so
/// this module stays self-contained (the crate's `common::matmul`
/// writes into a caller buffer; here an owned return reads cleaner).
fn mat_mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Row-major `A · Bᵀ` with `A: m×n`, `B: r×n` → `m×r`.
fn mat_mul_bt(a: &[f32], b: &[f32], m: usize, n: usize) -> Vec<f32> {
    // Here both A and B are m×n (W and G); result is m×m.
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(b.len(), m * n);
    let mut c = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0f32;
            for p in 0..n {
                s += a[i * n + p] * b[j * n + p];
            }
            c[i * m + j] = s;
        }
    }
    c
}

/// In-place `qf` retraction: orthonormalize the `rows` rows of the
/// row-major `rows × cols` matrix `y` (each row a length-`cols`
/// vector) via **modified Gram–Schmidt**, so afterwards
/// `Y·Yᵀ = I_rows`. This is exactly the thin-QR `Q` factor of `Yᵀ`
/// transposed back, with the diagonal of `R` implicitly forced
/// non-negative (each pivot row is normalized by its own positive
/// norm), matching the standard QR-retraction sign convention. Requires
/// `rows ≤ cols` and the rows of `Y` to be (numerically) linearly
/// independent — true for any `Y` close to a Stiefel point.
fn qf_rows_in_place(y: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(y.len(), rows * cols);
    for i in 0..rows {
        // Subtract projections onto the already-orthonormal rows 0..i.
        for j in 0..i {
            let mut dot = 0.0f32;
            for c in 0..cols {
                dot += y[i * cols + c] * y[j * cols + c];
            }
            for c in 0..cols {
                y[i * cols + c] -= dot * y[j * cols + c];
            }
        }
        // Normalize row i.
        let mut nrm = 0.0f64;
        for c in 0..cols {
            let v = y[i * cols + c] as f64;
            nrm += v * v;
        }
        let nrm = (nrm.sqrt() as f32).max(1e-12);
        let inv = 1.0 / nrm;
        for c in 0..cols {
            y[i * cols + c] *= inv;
        }
    }
}
