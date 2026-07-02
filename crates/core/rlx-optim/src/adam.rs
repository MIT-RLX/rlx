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

//! Adam — Adaptive Moment Estimation (Kingma & Ba, 2014).
//!
//! # Update rule
//!
//! For each parameter, with `t` the 1-based iteration index:
//!
//! ```text
//! g_t   = ∇L(θ_{t-1}) + λ·θ_{t-1}            // L2 decay folded in
//! m_t   = β₁·m_{t-1} + (1 − β₁)·g_t
//! v_t   = β₂·v_{t-1} + (1 − β₂)·g_t²
//! m̂_t   = m_t / (1 − β₁ᵗ)                    // bias correction
//! v̂_t   = v_t / (1 − β₂ᵗ)
//! θ_t   = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
//! ```
//!
//! # When to use
//!
//! Reliable default for transformer pre-training and most non-vision
//! workloads. Per-parameter memory is **2×** the parameter size (one
//! `m`, one `v`). If you need decoupled weight decay (recommended for
//! transformers) use [`crate::AdamW`] instead.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::{zeros_entry, zip4_for_each};

/// Bias-corrected first/second moment optimizer.
///
/// Per-tensor state: two `f32` buffers (`m`, `v`) of the same shape as
/// the parameter.
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate. Typical: `1e-3` for from-scratch CNNs, `1e-4`
    /// for transformer fine-tuning.
    pub lr: f32,
    /// First-moment EMA decay β₁ ∈ \[0, 1). Default `0.9`.
    pub beta1: f32,
    /// Second-moment EMA decay β₂ ∈ \[0, 1). Default `0.999`.
    pub beta2: f32,
    /// Stability constant in the denominator. Default `1e-8`.
    pub eps: f32,
    /// L2 weight decay coefficient. **Folded into the gradient**
    /// (the "classic Adam" rule); use [`crate::AdamW`] for decoupled
    /// decay. Default `0.0`.
    pub weight_decay: f32,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl Adam {
    /// Construct with the given learning rate and the standard
    /// (β₁, β₂, ε) = (0.9, 0.999, 1e-8) defaults.
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

    /// Override the denominator stability constant ε.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Override the L2 weight-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// 1-based iteration counter. Starts at 1 (so the first call to
    /// `step()` sees `t=1`), advances on [`Optimizer::end_iteration`].
    pub fn current_step(&self) -> u64 {
        self.step + 1
    }
}

impl Optimizer for Adam {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let t = (self.step + 1) as f64;
        let b1 = self.beta1 as f64;
        let b2 = self.beta2 as f64;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let eps = self.eps as f64;
        let lr = self.lr as f64;
        let wd = self.weight_decay;
        let n = param.len();
        // `self.m` / `self.v` are distinct fields, so the two
        // `zeros_entry` calls borrow disjoint regions of `self` and
        // their results can coexist.
        let m = zeros_entry(&mut self.m, name, n);
        let v = zeros_entry(&mut self.v, name, n);
        zip4_for_each(param, m, v, grad, |p, mi, vi, gi| {
            let g = (gi + wd * *p) as f64;
            let new_m = b1 * *mi as f64 + (1.0 - b1) * g;
            let new_v = b2 * *vi as f64 + (1.0 - b2) * g * g;
            *mi = new_m as f32;
            *vi = new_v as f32;
            let m_hat = new_m / bc1;
            let v_hat = new_v / bc2;
            *p -= (lr * m_hat / (v_hat.sqrt() + eps)) as f32;
        });
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
