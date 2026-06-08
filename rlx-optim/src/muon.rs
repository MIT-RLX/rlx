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

/// Newton–Schulz semi-orthogonalization. Operates on a row-major
/// `rows × cols` matrix and returns its closest semi-orthogonal matrix
/// (up to the polynomial truncation). The input is first scaled by its
/// Frobenius norm to stay inside the polynomial's region of convergence.
fn newton_schulz_orth(
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
    // The cubic iteration is more efficient on the "thin" side; we
    // transpose internally if rows < cols so that the inner products
    // are over the longer axis.
    let (mut x_mat, r, k, transposed) = if rows < cols {
        // transpose
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
        // A = X · Xᵀ  (r × r)
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += x_mat[i * k + p] * x_mat[j * k + p];
                }
                a_mat[i * r + j] = s;
            }
        }
        // A² = A · A
        for i in 0..r {
            for j in 0..r {
                let mut s = 0.0f32;
                for p in 0..r {
                    s += a_mat[i * r + p] * a_mat[p * r + j];
                }
                a2[i * r + j] = s;
            }
        }
        // X ← a·X + b·A·X + cc·A²·X
        for i in 0..r {
            for j in 0..k {
                let mut s = a * x_mat[i * k + j];
                for p in 0..r {
                    s += b * a_mat[i * r + p] * x_mat[p * k + j];
                    s += cc * a2[i * r + p] * x_mat[p * k + j];
                }
                tmp[i * k + j] = s;
            }
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
