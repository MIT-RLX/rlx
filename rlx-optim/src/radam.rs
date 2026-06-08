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

//! RAdam — Rectified Adam (Liu et al., 2019, "On the Variance of the
//! Adaptive Learning Rate and Beyond").
//!
//! # Motivation
//!
//! Early in training, `v_t` is built from very few samples and its
//! variance is huge — which makes Adam's effective learning rate
//! noisy at the same iterations where stability matters most.
//! Practitioners "fix" this with an LR warm-up; RAdam derives a
//! *closed-form* warm-up from the variance of the inverse-square-root
//! of `v̂_t`.
//!
//! # Update rule
//!
//! Let `ρ_∞ = 2/(1−β₂) − 1` and the "SMA length"
//! `ρ_t = ρ_∞ − 2t·β₂ᵗ / (1 − β₂ᵗ)`. Define the rectification term
//!
//! ```text
//! r_t = √( ((ρ_t − 4)(ρ_t − 2)·ρ_∞) / ((ρ_∞ − 4)(ρ_∞ − 2)·ρ_t) )
//! ```
//!
//! When `ρ_t > 4` the second moment is "stable enough" — use the
//! corrected Adam step with `r_t` scaling:
//!
//! ```text
//! θ_t = θ_{t-1} − lr · r_t · m̂_t / (√v̂_t + ε)
//! ```
//!
//! Otherwise (`ρ_t ≤ 4`, early steps), fall back to SGD-with-momentum
//! using the first moment alone:
//!
//! ```text
//! θ_t = θ_{t-1} − lr · m̂_t
//! ```
//!
//! # When to use
//!
//! Drop-in replacement for Adam when you don't want to hand-tune a
//! warm-up schedule. Same memory cost as Adam.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// Rectified Adam. Per-tensor state: two `f32` buffers.
#[derive(Debug, Clone)]
pub struct RAdam {
    /// Learning rate.
    pub lr: f32,
    /// First-moment EMA decay β₁. Default `0.9`.
    pub beta1: f32,
    /// Second-moment EMA decay β₂. Default `0.999`.
    pub beta2: f32,
    /// Denominator stability constant. Default `1e-8`.
    pub eps: f32,
    /// L2 weight-decay coefficient (folded into the gradient — like
    /// classical Adam, **not** decoupled). Default `0.0`.
    pub weight_decay: f32,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl RAdam {
    /// Construct with `(β₁, β₂, ε, λ) = (0.9, 0.999, 1e-8, 0.0)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Override (β₁, β₂).
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    /// Override the L2 weight-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for RAdam {
    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let t = (self.step + 1) as f64;
        let b1 = self.beta1 as f64;
        let b2 = self.beta2 as f64;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let rho_inf = 2.0 / (1.0 - b2) - 1.0;
        let rho_t = rho_inf - 2.0 * t * b2.powf(t) / bc2;
        let eps = self.eps as f64;
        let lr = self.lr as f64;
        let wd = self.weight_decay;
        // Variance-rectification term `r_t` (Liu et al. eq. 14).
        let r_t = if rho_t > 4.0 {
            (((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                .sqrt()
        } else {
            0.0
        };
        let m = zeros_entry(&mut self.m, name, param.len());
        let v = zeros_entry(&mut self.v, name, param.len());
        for i in 0..param.len() {
            let g = (grad[i] + wd * param[i]) as f64;
            let mi = b1 * m[i] as f64 + (1.0 - b1) * g;
            let vi = b2 * v[i] as f64 + (1.0 - b2) * g * g;
            m[i] = mi as f32;
            v[i] = vi as f32;
            let m_hat = mi / bc1;
            let update = if rho_t > 4.0 {
                let v_hat = (vi / bc2).sqrt();
                r_t * m_hat / (v_hat + eps)
            } else {
                m_hat
            };
            param[i] -= (lr * update) as f32;
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
