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

//! SOAP — ShampoO with Adam in the Preconditioner's eigenbasis
//! (Vyas, Morwani, Anil, et al., 2024).
//!
//! # Idea
//!
//! Shampoo (Gupta et al. 2018) preconditions a 2-D parameter's
//! gradient by `L⁻¹ᐟ⁴ · G · R⁻¹ᐟ⁴`, where `L = E[G·Gᵀ]` and
//! `R = E[Gᵀ·G]` are Kronecker-factor covariances. SOAP observes that
//! the same preconditioner is equivalent to **running Adam in the
//! eigenbasis** of `L` and `R` — and that you only need to recompute
//! the eigenbasis every K steps. This delivers Shampoo's quality with
//! Adam's per-step cost (between recompiles).
//!
//! # Update rule (for a 2-D parameter `W ∈ ℝ^{m×n}`)
//!
//! ```text
//! L_t = sb·L_{t-1} + (1−sb)·G·Gᵀ           // m×m
//! R_t = sb·R_{t-1} + (1−sb)·Gᵀ·G           // n×n
//! every K steps:                            // K = precond_freq
//!   Q_L, _ = eigh(L_t)                     // m×m eigenbasis
//!   Q_R, _ = eigh(R_t)                     // n×n eigenbasis
//! G' = Q_Lᵀ · G · Q_R                       // rotated gradient
//! [per-element Adam on G' → U']
//! U  = Q_L · U' · Q_Rᵀ                      // rotate back
//! θ_t = θ_{t-1} − lr · ( U + λ·θ_{t-1} )
//! ```
//!
//! For non-2-D parameters we fall back to plain AdamW.
//!
//! # When to use
//!
//! When you want Shampoo's quality on transformers / large dense
//! models and can afford the eigendecomposition cost amortized over
//! `precond_freq` steps. State cost per matrix:
//! `L (m²) + R (n²) + Q_L (m²) + Q_R (n²) + m_rot (m·n) + v_rot (m·n)`.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::{jacobi_eigh_sym, matmul, zeros_entry};

#[derive(Debug, Clone)]
struct SoapState {
    l: Vec<f32>,     // m × m left covariance
    r: Vec<f32>,     // n × n right covariance
    ql: Vec<f32>,    // m × m eigenbasis (row-major: row i = eigvec i)
    qr: Vec<f32>,    // n × n eigenbasis
    m_rot: Vec<f32>, // first moment in rotated basis (m·n)
    v_rot: Vec<f32>, // second moment in rotated basis (m·n)
    initialized_basis: bool,
}

/// SOAP — Shampoo-in-Adam-basis optimizer.
#[derive(Debug, Clone)]
pub struct Soap {
    /// Learning rate.
    pub lr: f32,
    /// First-moment EMA decay (in the *rotated* basis). Default `0.95`.
    pub beta1: f32,
    /// Second-moment EMA decay (in the *rotated* basis). Default `0.95`.
    pub beta2: f32,
    /// Decay for the L/R covariance EMAs. Often equal to β₂.
    pub shampoo_beta: f32,
    /// Denominator stability constant. Default `1e-8`.
    pub eps: f32,
    /// Decoupled weight-decay coefficient λ. Default `0.01`.
    pub weight_decay: f32,
    /// Recompute the eigenbasis every `precond_freq` steps. Larger
    /// values amortize the Jacobi cost but lag the preconditioner.
    /// Default `10`.
    pub precond_freq: u64,
    /// Max Jacobi sweeps per rediagonalization. Default `30`.
    pub jacobi_sweeps: u32,
    step: u64,
    state: HashMap<String, SoapState>,
    // Fallback Adam state for non-2D parameters.
    fb_m: HashMap<String, Vec<f32>>,
    fb_v: HashMap<String, Vec<f32>>,
}

impl Soap {
    /// Construct with `(β₁, β₂, sb, ε, λ, freq, sweeps) =
    /// (0.95, 0.95, 0.95, 1e-8, 0.01, 10, 30)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.95,
            beta2: 0.95,
            shampoo_beta: 0.95,
            eps: 1e-8,
            weight_decay: 0.01,
            precond_freq: 10,
            jacobi_sweeps: 30,
            step: 0,
            state: HashMap::new(),
            fb_m: HashMap::new(),
            fb_v: HashMap::new(),
        }
    }

    /// Override the decoupled-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Soap {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        if shape.len() != 2 {
            // Fallback: plain AdamW for non-matrix parameters.
            adamw_fallback(self, name, param, grad);
            return;
        }
        let (m, n) = (shape[0], shape[1]);
        debug_assert_eq!(m * n, param.len());
        let t = (self.step + 1) as f64;
        let b1 = self.beta1 as f64;
        let b2 = self.beta2 as f64;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let sb = self.shampoo_beta as f64;
        let eps = self.eps;
        let lr = self.lr;
        let wd = self.weight_decay;

        let st = self
            .state
            .entry(name.to_owned())
            .or_insert_with(|| SoapState {
                l: vec![0.0; m * m],
                r: vec![0.0; n * n],
                ql: identity(m),
                qr: identity(n),
                m_rot: vec![0.0; m * n],
                v_rot: vec![0.0; m * n],
                initialized_basis: false,
            });

        // ── 1. Update L, R covariances ─────────────────────────────
        // L += (1-sb)·G·Gᵀ; R += (1-sb)·Gᵀ·G  (with β2 decay).
        for i in 0..m {
            for j in 0..m {
                let mut s = 0.0f64;
                for p in 0..n {
                    s += grad[i * n + p] as f64 * grad[j * n + p] as f64;
                }
                let lij = sb * st.l[i * m + j] as f64 + (1.0 - sb) * s;
                st.l[i * m + j] = lij as f32;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f64;
                for p in 0..m {
                    s += grad[p * n + i] as f64 * grad[p * n + j] as f64;
                }
                let rij = sb * st.r[i * n + j] as f64 + (1.0 - sb) * s;
                st.r[i * n + j] = rij as f32;
            }
        }

        // ── 2. Rediagonalize periodically (and once on the first step). ──
        let need_rediag = !st.initialized_basis || self.step.is_multiple_of(self.precond_freq);
        if need_rediag {
            let mut l_copy = st.l.clone();
            let mut r_copy = st.r.clone();
            jacobi_eigh_sym(&mut l_copy, m, &mut st.ql, self.jacobi_sweeps, 1e-6);
            jacobi_eigh_sym(&mut r_copy, n, &mut st.qr, self.jacobi_sweeps, 1e-6);
            st.initialized_basis = true;
        }

        // ── 3. Rotate gradient: G' = Qₗᵀ · G · Q_r ────────────────
        let mut tmp = vec![0.0f32; m * n];
        // tmp = Qₗᵀ · G  ⇒ tmp[i,j] = sum_p Qₗ[p,i] · G[p,j]
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..m {
                    s += st.ql[p * m + i] * grad[p * n + j];
                }
                tmp[i * n + j] = s;
            }
        }
        let mut g_rot = vec![0.0f32; m * n];
        matmul(&tmp, &st.qr, m, n, n, &mut g_rot);

        // ── 4. Per-element Adam on rotated grad ──────────────────
        let mut u_rot = vec![0.0f32; m * n];
        for k in 0..m * n {
            let g = g_rot[k] as f64;
            let mi = b1 * st.m_rot[k] as f64 + (1.0 - b1) * g;
            let vi = b2 * st.v_rot[k] as f64 + (1.0 - b2) * g * g;
            st.m_rot[k] = mi as f32;
            st.v_rot[k] = vi as f32;
            let m_hat = mi / bc1;
            let v_hat = vi / bc2;
            u_rot[k] = (m_hat / (v_hat.sqrt() + eps as f64)) as f32;
        }

        // ── 5. Rotate back: U = Qₗ · U' · Q_rᵀ ───────────────────
        // tmp = Qₗ · U'
        matmul(&st.ql, &u_rot, m, m, n, &mut tmp);
        // U = tmp · Q_rᵀ ⇒ U[i,j] = sum_p tmp[i,p] · Q_r[j,p]
        let mut u = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..n {
                    s += tmp[i * n + p] * st.qr[j * n + p];
                }
                u[i * n + j] = s;
            }
        }

        // ── 6. Decoupled weight decay + parameter update ─────────
        for i in 0..m * n {
            param[i] -= lr * (u[i] + wd * param[i]);
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}

fn identity(n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

// Plain AdamW for the non-matrix fallback path.
fn adamw_fallback(opt: &mut Soap, name: &str, param: &mut [f32], grad: &[f32]) {
    let t = (opt.step + 1) as f64;
    let b1 = opt.beta1 as f64;
    let b2 = opt.beta2 as f64;
    let bc1 = 1.0 - b1.powf(t);
    let bc2 = 1.0 - b2.powf(t);
    let m = zeros_entry(&mut opt.fb_m, name, param.len());
    let v = zeros_entry(&mut opt.fb_v, name, param.len());
    let eps = opt.eps as f64;
    let lr = opt.lr as f64;
    let wd = opt.weight_decay as f64;
    for i in 0..param.len() {
        let g = grad[i] as f64;
        let mi = b1 * m[i] as f64 + (1.0 - b1) * g;
        let vi = b2 * v[i] as f64 + (1.0 - b2) * g * g;
        m[i] = mi as f32;
        v[i] = vi as f32;
        let p = param[i] as f64;
        param[i] = (p - lr * (mi / bc1 / ((vi / bc2).sqrt() + eps) + wd * p)) as f32;
    }
}
