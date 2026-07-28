// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Kron-PSGD — Preconditioned SGD with a Kronecker-factored
//! preconditioner (Li, 2018; "Preconditioned Stochastic Gradient
//! Descent").
//!
//! # Idea
//!
//! Approximate the inverse Hessian as a Kronecker product
//! `P ≈ P_L ⊗ P_R` where `P_L = Q_LᵀQ_L` and `P_R = Q_RᵀQ_R` for two
//! upper-triangular factors. The factors are updated by a *Lie-group
//! descent* on a whitening criterion — no eigendecomposition needed,
//! and updates are stable by construction (the upper-triangular
//! manifold).
//!
//! # Update rule
//!
//! For a 2-D parameter `W ∈ ℝ^{m×n}`:
//!
//! ```text
//! A = Q_L · G · Q_Rᵀ                          // m×n
//! B = Q_L⁻ᵀ · G · Q_R⁻¹                       // m×n (triangular solves)
//! dQ_L ∝ tril(A·Aᵀ − B·Bᵀ);  Q_L ← Q_L − η_p · Q_L · dQ_L
//! dQ_R ∝ tril(Aᵀ·A − Bᵀ·B);  Q_R ← Q_R − η_p · Q_R · dQ_R
//! P_L = Q_LᵀQ_L;   P_R = Q_RᵀQ_R
//! p_g = P_L · G · P_R                         // preconditioned grad
//! [spectral-clip to ‖·‖_∞ ≤ clip, then SGD+momentum on p_g]
//! ```
//!
//! Li (2018) Algorithm 1 uses an HVP probe `v` and its perturbed
//! gradient to update `Q_L, Q_R`. This crate has no HVP oracle, so we
//! use the gradient itself as the probe — the "PSGD-Affine"
//! approximation — which is cheap and still gives strong empirical
//! preconditioning on convex and mildly non-convex problems.
//! Non-2-D parameters fall back to plain SGD-with-momentum.
//!
//! # When to use
//!
//! Ill-conditioned problems where Adam's coordinate-wise
//! preconditioner is too weak (RNNs, deep MLPs, certain inverse
//! problems). State cost per matrix: `m² + n²` plus a velocity buffer.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::{matmul, zeros_entry};

#[derive(Debug, Clone)]
struct KronState {
    ql: Vec<f32>, // m × m upper-triangular
    qr: Vec<f32>, // n × n upper-triangular
}

/// Kron-PSGD — Kronecker-factored preconditioned SGD.
#[derive(Debug, Clone)]
pub struct KronPsgd {
    /// Learning rate.
    pub lr: f32,
    /// Learning rate for the **preconditioner** update (Lie-group
    /// descent on Q_L / Q_R). Default `0.1`. Too high ⇒ Q drifts;
    /// too low ⇒ preconditioner lags.
    pub precond_lr: f32,
    /// Polyak momentum for the preconditioned-gradient SGD step.
    /// Default `0.9`.
    pub momentum: f32,
    /// L2 weight-decay coefficient (folded into the gradient).
    /// Default `0.0`.
    pub weight_decay: f32,
    /// Numerical floor on the preconditioner-update normalizer.
    /// Default `1e-8`.
    pub eps: f32,
    /// Cap the per-coordinate magnitude of the preconditioned update
    /// (defensive — early Q estimates can be ill-conditioned). Default `1.0`.
    pub clip: f32,
    state: HashMap<String, KronState>,
    mom: HashMap<String, Vec<f32>>,
}

impl KronPsgd {
    /// Construct with `(precond_lr, μ, λ, ε, clip) = (0.1, 0.9, 0.0, 1e-8, 1.0)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            precond_lr: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
            eps: 1e-8,
            clip: 1.0,
            state: HashMap::new(),
            mom: HashMap::new(),
        }
    }

    /// Override the Polyak momentum.
    pub fn with_momentum(mut self, mu: f32) -> Self {
        self.momentum = mu;
        self
    }

    /// Override the weight-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for KronPsgd {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let lr = self.lr;
        let wd = self.weight_decay;

        if shape.len() != 2 {
            // Non-matrix: SGD + momentum fallback.
            let v = zeros_entry(&mut self.mom, name, param.len());
            let mu = self.momentum;
            for i in 0..param.len() {
                v[i] = mu * v[i] + grad[i] + wd * param[i];
                param[i] -= lr * v[i];
            }
            return;
        }
        let (m, n) = (shape[0], shape[1]);
        debug_assert_eq!(m * n, param.len());
        let st = self
            .state
            .entry(name.to_owned())
            .or_insert_with(|| KronState {
                ql: identity_triangular(m),
                qr: identity_triangular(n),
            });

        // ── 1. Update Q_L, Q_R via Li (2018) Lie-group rule. ──────
        // Use g itself as the probe; the affine variant requires:
        //   A = Q_L · g · Q_Rᵀ        (m × n)
        //   B = Q_L⁻ᵀ · g · Q_R⁻¹     (m × n; cheap because Q is triangular)
        // dQ_L ∝ tril(A·Aᵀ − B·Bᵀ); dQ_R ∝ tril(Aᵀ·A − Bᵀ·B).
        let a = matmul_3(&st.ql, grad, &st.qr, m, n, /*trans_q_r=*/ true);
        let b = matmul_3_inv(&st.ql, grad, &st.qr, m, n);
        update_factor(&mut st.ql, &a, &b, m, n, true, self.precond_lr, self.eps);
        update_factor(&mut st.qr, &a, &b, m, n, false, self.precond_lr, self.eps);

        // ── 2. Preconditioned gradient: p_g = Q_Lᵀ · Q_L · g · Q_R · Q_Rᵀ ──
        // Build Q_Lᵀ Q_L (m×m, symmetric)
        let mut ql_t_ql = vec![0.0f32; m * m];
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0f32;
                for p in 0..m {
                    s += st.ql[p * m + i] * st.ql[p * m + j];
                }
                ql_t_ql[i * m + j] = s;
            }
        }
        let mut qr_qr_t = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..n {
                    s += st.qr[i * n + p] * st.qr[j * n + p];
                }
                qr_qr_t[i * n + j] = s;
            }
        }
        // p_g = (Q_Lᵀ Q_L) · g · (Q_R Q_Rᵀ)
        let mut tmp = vec![0.0f32; m * n];
        matmul(&ql_t_ql, grad, m, m, n, &mut tmp);
        let mut p_g = vec![0.0f32; m * n];
        matmul(&tmp, &qr_qr_t, m, n, n, &mut p_g);

        // ── 3. Spectral clip + momentum + apply. ─────────────────
        let mut max_abs = 0.0f32;
        for &x in &p_g {
            if x.abs() > max_abs {
                max_abs = x.abs();
            }
        }
        let scale = if max_abs > self.clip {
            self.clip / max_abs
        } else {
            1.0
        };
        let v = zeros_entry(&mut self.mom, name, param.len());
        let mu = self.momentum;
        for i in 0..param.len() {
            let g = scale * p_g[i] + wd * param[i];
            v[i] = mu * v[i] + g;
            param[i] -= lr * v[i];
        }
    }
}

fn identity_triangular(n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

/// Compute `Q_L · G · Q_Rᵀ` (or `Q_L · G · Q_R` if `trans_q_r=false`).
fn matmul_3(ql: &[f32], g: &[f32], qr: &[f32], m: usize, n: usize, trans_q_r: bool) -> Vec<f32> {
    let mut t1 = vec![0.0f32; m * n];
    matmul(ql, g, m, m, n, &mut t1);
    let mut out = vec![0.0f32; m * n];
    if trans_q_r {
        // out = t1 · Q_Rᵀ  ⇒  out[i,j] = sum_p t1[i,p] · Q_R[j,p]
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..n {
                    s += t1[i * n + p] * qr[j * n + p];
                }
                out[i * n + j] = s;
            }
        }
    } else {
        matmul(&t1, qr, m, n, n, &mut out);
    }
    out
}

/// Compute `Q_L⁻ᵀ · G · Q_R⁻¹` for upper-triangular Q's via two
/// triangular solves on `G`.
fn matmul_3_inv(ql: &[f32], g: &[f32], qr: &[f32], m: usize, n: usize) -> Vec<f32> {
    // First solve Q_Lᵀ · X = G column-by-column. Q_Lᵀ is lower-triangular.
    let mut x = g.to_vec();
    for j in 0..n {
        // Forward-substitute one column.
        for i in 0..m {
            let mut s = x[i * n + j];
            for p in 0..i {
                s -= ql[p * m + i] * x[p * n + j];
            }
            let d = ql[i * m + i];
            x[i * n + j] = if d.abs() > 1e-12 { s / d } else { 0.0 };
        }
    }
    // Then solve Y · Q_R = X for Y row-by-row (Q_R upper-triangular).
    // Equivalently: for each row i, back-substitute Y[i,:] · Q_R = X[i,:].
    let mut y = x;
    for i in 0..m {
        for j in 0..n {
            let mut s = y[i * n + j];
            for p in 0..j {
                s -= y[i * n + p] * qr[p * n + j];
            }
            let d = qr[j * n + j];
            y[i * n + j] = if d.abs() > 1e-12 { s / d } else { 0.0 };
        }
    }
    y
}

/// Lie-group update of a triangular factor. `which = true` updates Q_L
/// using `A·Aᵀ − B·Bᵀ` (m×m), `which = false` updates Q_R using
/// `Aᵀ·A − Bᵀ·B` (n×n). The descent direction is then projected onto
/// the upper-triangular tangent space.
fn update_factor(
    q: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    which: bool,
    plr: f32,
    eps: f32,
) {
    let dim = if which { m } else { n };
    let mut grad_q = vec![0.0f32; dim * dim];
    // Build A·Aᵀ − B·Bᵀ  or  Aᵀ·A − Bᵀ·B.
    let mut norm = 0.0f64;
    for i in 0..dim {
        for j in 0..dim {
            let mut a_term = 0.0f32;
            let mut b_term = 0.0f32;
            if which {
                for p in 0..n {
                    a_term += a[i * n + p] * a[j * n + p];
                    b_term += b[i * n + p] * b[j * n + p];
                }
            } else {
                for p in 0..m {
                    a_term += a[p * n + i] * a[p * n + j];
                    b_term += b[p * n + i] * b[p * n + j];
                }
            }
            let d = a_term - b_term;
            grad_q[i * dim + j] = d;
            norm += d as f64 * d as f64;
        }
    }
    let scale = plr / ((norm.sqrt() as f32) + eps);
    // Project onto upper-triangular: Q ← Q · (I − 0.5·scale·tril(grad_q + grad_qᵀ))
    // (Simplified Lie-group projection; full version solves a tiny matrix
    // exponential, but a single linearized step is the standard choice.)
    for i in 0..dim {
        for j in 0..dim {
            if j < i {
                grad_q[i * dim + j] = 0.0; // upper-triangular projection
            }
        }
    }
    let mut q_new = vec![0.0f32; dim * dim];
    matmul(q, &grad_q, dim, dim, dim, &mut q_new);
    for k in 0..dim * dim {
        q[k] -= scale * q_new[k];
    }
}
