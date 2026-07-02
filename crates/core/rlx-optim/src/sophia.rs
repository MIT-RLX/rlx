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

//! Sophia-H — Second-order Clipped Stochastic Optimization (Liu, Xie,
//! Zhang, Ma, 2023).
//!
//! # Idea
//!
//! Adam preconditions by `1/√v_t` (a noisy proxy for the inverse
//! Hessian *diagonal*); Sophia preconditions by the **actual Hessian
//! diagonal**, computed periodically via a Hutchinson estimator or a
//! Gauss–Newton approximation. The crucial trick is a *per-coordinate
//! clip* of the resulting update — even with a noisy Hessian, the
//! clip caps each coordinate's step at `ρ`, so adversarial curvature
//! estimates can never blow up the trajectory.
//!
//! # Update rule
//!
//! ```text
//! m_t = β₁·m_{t-1} + (1 − β₁)·g_t                  // first moment EMA
//! [every K steps the caller updates h via Sophia::update_hessian:]
//!   h ← β₂·h + (1 − β₂)·diag(H_t)                  // Hessian-diag EMA
//! u_i = m_{t,i} / max(γ · h_i, ε)
//! u_i = clip(u_i, −ρ, +ρ)
//! θ_t = θ_{t-1} − lr · ( u + λ·θ_{t-1} )
//! ```
//!
//! # HVP oracle
//!
//! This crate doesn't ship an HVP oracle (it lives in `rlx-autodiff`
//! as [`rlx_autodiff::hvp`](../../rlx_autodiff/fn.hvp.html)). Call
//! [`Sophia::update_hessian`] yourself whenever you have a fresh
//! diagonal estimate (Hutchinson: `H_diag ≈ u ⊙ (∇²L · u)` with random
//! Rademacher `u`; or Gauss–Newton: `H_diag ≈ g_t²` from a held-out
//! micro-batch). If you never update it, Sophia degenerates to a
//! magnitude-clipped first-moment step.
//!
//! # When to use
//!
//! Curvature-aware optimization for LLM pre-training; the original
//! paper reports ~2× wall-clock speedup vs AdamW at the same loss.
//! State cost: two buffers per parameter (`m`, `h`).

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// Sophia-H — Hessian-diagonal second-order optimizer.
#[derive(Debug, Clone)]
pub struct Sophia {
    /// Learning rate. Typically slightly *larger* than the AdamW LR
    /// you'd use on the same model, because the clip bounds the step.
    pub lr: f32,
    /// First-moment EMA decay β₁. Default `0.965`.
    pub beta1: f32,
    /// Hessian-diagonal EMA decay β₂. Default `0.99`.
    pub beta2: f32,
    /// Hessian scale γ (Liu et al. default `0.01`). Multiplies the
    /// Hessian estimate before forming the denominator.
    pub gamma: f32,
    /// Per-coordinate clip threshold ρ. Default `0.04` — the
    /// dimensionless cap on each step's magnitude.
    pub rho: f32,
    /// Denominator floor. Default `1e-12`.
    pub eps: f32,
    /// Decoupled weight-decay coefficient λ. Default `0.1` (large by
    /// AdamW standards — Sophia tolerates more decay).
    pub weight_decay: f32,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    h: HashMap<String, Vec<f32>>,
}

impl Sophia {
    /// Construct with `(β₁, β₂, γ, ρ, ε, λ) = (0.965, 0.99, 0.01, 0.04, 1e-12, 0.1)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.965,
            beta2: 0.99,
            gamma: 0.01,
            rho: 0.04,
            eps: 1e-12,
            weight_decay: 0.1,
            step: 0,
            m: HashMap::new(),
            h: HashMap::new(),
        }
    }

    /// Override (β₁, β₂).
    pub fn with_betas(mut self, b1: f32, b2: f32) -> Self {
        self.beta1 = b1;
        self.beta2 = b2;
        self
    }

    /// Override the decoupled-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Update the diagonal-Hessian estimate for parameter `name`.
    /// `h_hat` should be a fresh estimate (typically `H_diag` from a
    /// Hutchinson estimator or `g²` from a Gauss-Newton approximation).
    pub fn update_hessian(&mut self, name: &str, h_hat: &[f32]) {
        let h = zeros_entry(&mut self.h, name, h_hat.len());
        let b2 = self.beta2;
        for i in 0..h.len() {
            h[i] = b2 * h[i] + (1.0 - b2) * h_hat[i];
        }
    }
}

impl Optimizer for Sophia {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let b1 = self.beta1;
        let gamma = self.gamma.max(self.eps);
        let rho = self.rho;
        let eps = self.eps;
        let lr = self.lr;
        let wd = self.weight_decay;
        let m = zeros_entry(&mut self.m, name, param.len());
        for i in 0..param.len() {
            m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
        }
        // Snapshot h (zero if not yet populated).
        let h_default = vec![0.0f32; param.len()];
        let h = self.h.get(name).unwrap_or(&h_default);
        for i in 0..param.len() {
            let denom = (gamma * h[i]).max(eps);
            let mut u = m[i] / denom;
            // Per-coordinate clip to [-rho, rho].
            if u > rho {
                u = rho;
            } else if u < -rho {
                u = -rho;
            }
            // Decoupled decay.
            param[i] -= lr * (u + wd * param[i]);
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
