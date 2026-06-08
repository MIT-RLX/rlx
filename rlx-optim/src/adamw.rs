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

//! AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
//!
//! # Why "decoupled"?
//!
//! In classical Adam, an L2 penalty `λ·θ` is folded into the gradient
//! before the EMAs — but the second-moment estimator `v_t` then scales
//! the decay term, so parameters with large gradients get *less* L2
//! regularization. AdamW separates the two: decay multiplies the
//! parameter directly, *outside* the adaptive step.
//!
//! # Update rule
//!
//! ```text
//! m_t = β₁·m_{t-1} + (1 − β₁)·g_t            // g_t is the raw grad
//! v_t = β₂·v_{t-1} + (1 − β₂)·g_t²
//! m̂_t = m_t / (1 − β₁ᵗ)
//! v̂_t = v_t / (1 − β₂ᵗ)
//! θ_t = θ_{t-1} − lr · ( m̂_t/(√v̂_t + ε) + λ·θ_{t-1} )
//! ```
//!
//! # When to use
//!
//! The default for transformer pre-training. Pair with a cosine /
//! linear LR schedule and `weight_decay = 0.1` for LLMs, `0.01–0.05`
//! for vision transformers.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::{zeros_entry, zip4_for_each};

/// Adam with decoupled weight decay.
///
/// Per-tensor state identical to [`crate::Adam`] (two `f32` buffers).
#[derive(Debug, Clone)]
pub struct AdamW {
    /// Learning rate. Typical LLM pre-training value: `1e-4` to `3e-4`.
    pub lr: f32,
    /// First-moment EMA decay. Default `0.9`.
    pub beta1: f32,
    /// Second-moment EMA decay. Default `0.999` (matches Adam);
    /// `0.95` is common for very long pre-training runs.
    pub beta2: f32,
    /// Denominator stability constant. Default `1e-8`.
    pub eps: f32,
    /// **Decoupled** weight-decay coefficient λ. Multiplies the
    /// parameter directly inside the update; `0.01–0.1` typical.
    /// Defaults to `0.01`.
    pub weight_decay: f32,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl AdamW {
    /// Construct with the given learning rate and the standard
    /// (β₁, β₂, ε, λ) = (0.9, 0.999, 1e-8, 0.01) defaults.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
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

    /// Override the decoupled-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Override the denominator ε.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let t = (self.step + 1) as f64;
        let b1 = self.beta1 as f64;
        let b2 = self.beta2 as f64;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let eps = self.eps as f64;
        let lr = self.lr as f64;
        let wd = self.weight_decay as f64;
        let m = zeros_entry(&mut self.m, name, param.len());
        let v = zeros_entry(&mut self.v, name, param.len());
        zip4_for_each(param, m, v, grad, |p, mi, vi, gi| {
            let g = gi as f64;
            let new_m = b1 * *mi as f64 + (1.0 - b1) * g;
            let new_v = b2 * *vi as f64 + (1.0 - b2) * g * g;
            *mi = new_m as f32;
            *vi = new_v as f32;
            let m_hat = new_m / bc1;
            let v_hat = new_v / bc2;
            // Decoupled decay: applied to the parameter, then the
            // adaptive step is subtracted.
            let pf = *p as f64;
            *p = (pf - lr * (m_hat / (v_hat.sqrt() + eps) + wd * pf)) as f32;
        });
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
