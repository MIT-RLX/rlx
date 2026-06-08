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

//! QHAdamW — Quasi-Hyperbolic Adam (Ma & Yarats, 2019) with decoupled
//! weight decay.
//!
//! # Idea
//!
//! Adam can be viewed as "the second moment EMA scales the gradient,
//! the first moment EMA *replaces* the gradient." The quasi-hyperbolic
//! family says: don't *replace* — *interpolate*. Mix the EMA with the
//! raw current gradient, controlled by per-moment scalars `ν₁, ν₂`.
//!
//! # Update rule
//!
//! ```text
//! m_t = β₁·m_{t-1} + (1 − β₁)·g_t
//! v_t = β₂·v_{t-1} + (1 − β₂)·g_t²
//! num = (1 − ν₁)·g_t   + ν₁·m̂_t
//! den = √((1 − ν₂)·g_t² + ν₂·v̂_t) + ε
//! θ_t = θ_{t-1} − lr · ( num / den + λ·θ_{t-1} )
//! ```
//!
//! Setting `ν₁ = ν₂ = 1` recovers standard AdamW; `ν₁ = β₁` recovers
//! Nesterov-style behavior in the limit; `ν₁ < 1` injects more of the
//! current gradient and tends to be more robust on noisy losses.
//!
//! # When to use
//!
//! When you've found AdamW too sluggish on noisy / heavy-tail
//! gradients (RL, GAN training) and an LR sweep didn't help.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// Quasi-hyperbolic AdamW. Per-tensor state: two `f32` buffers.
#[derive(Debug, Clone)]
pub struct QHAdamW {
    /// Learning rate.
    pub lr: f32,
    /// First-moment EMA decay β₁. Ma & Yarats recommend `0.995` (much
    /// closer to 1 than vanilla Adam) — the QH interpolation already
    /// keeps current-gradient weight in the numerator.
    pub beta1: f32,
    /// Second-moment EMA decay β₂. Default `0.999`.
    pub beta2: f32,
    /// First-moment QH interpolation coefficient ν₁ ∈ \[0, 1\].
    /// `1.0` = pure EMA (standard Adam first moment); `0.0` = pure
    /// current gradient (no momentum). Default `0.7`.
    pub nu1: f32,
    /// Second-moment QH interpolation coefficient ν₂. `1.0` = standard
    /// Adam denominator. Default `1.0`.
    pub nu2: f32,
    /// Denominator stability constant. Default `1e-8`.
    pub eps: f32,
    /// Decoupled weight-decay coefficient λ. Default `0.01`.
    pub weight_decay: f32,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl QHAdamW {
    /// Construct with `(β₁, β₂, ν₁, ν₂, ε, λ) = (0.995, 0.999, 0.7, 1.0, 1e-8, 0.01)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.995,
            beta2: 0.999,
            nu1: 0.7,
            nu2: 1.0,
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

    /// Override the quasi-hyperbolic coefficients (ν₁, ν₂).
    pub fn with_nus(mut self, n1: f32, n2: f32) -> Self {
        self.nu1 = n1;
        self.nu2 = n2;
        self
    }

    /// Override the decoupled-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for QHAdamW {
    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let t = (self.step + 1) as f64;
        let b1 = self.beta1 as f64;
        let b2 = self.beta2 as f64;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);
        let n1 = self.nu1 as f64;
        let n2 = self.nu2 as f64;
        let eps = self.eps as f64;
        let lr = self.lr as f64;
        let wd = self.weight_decay as f64;
        let m = zeros_entry(&mut self.m, name, param.len());
        let v = zeros_entry(&mut self.v, name, param.len());
        for i in 0..param.len() {
            let g = grad[i] as f64;
            let mi = b1 * m[i] as f64 + (1.0 - b1) * g;
            let vi = b2 * v[i] as f64 + (1.0 - b2) * g * g;
            m[i] = mi as f32;
            v[i] = vi as f32;
            let m_hat = mi / bc1;
            let v_hat = vi / bc2;
            // Quasi-hyperbolic numerator & denominator (Ma & Yarats Alg. 2).
            let num = (1.0 - n1) * g + n1 * m_hat;
            let den = ((1.0 - n2) * g * g + n2 * v_hat).sqrt() + eps;
            let p = param[i] as f64;
            param[i] = (p - lr * (num / den + wd * p)) as f32;
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
