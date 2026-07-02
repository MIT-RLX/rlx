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

//! MARS — Make vAriance Reduction Shine (Yuan, Liu, Wu, Su, Gu, 2024).
//!
//! # Idea
//!
//! Variance reduction (SVRG, SARAH) lowers gradient noise by mixing in
//! a *previous* gradient — at the cost of an extra forward/backward
//! pass per snapshot. MARS shows that you don't need a snapshot:
//! using just `g_{t−1}` (the previous mini-batch's gradient) as the
//! "control variate" gives most of the benefit of full variance
//! reduction, for free.
//!
//! # Update rule
//!
//! ```text
//! c_t = g_t + γ · β₁/(1−β₁) · (g_t − g_{t−1})      // VR-corrected grad
//! [optional: clip c_t to unit norm per tensor]
//! m_t = β₁·m_{t-1} + (1 − β₁)·c_t
//! v_t = β₂·v_{t-1} + (1 − β₂)·c_t²
//! θ_t = θ_{t-1} − lr · ( m̂_t/(√v̂_t + ε) + λ·θ_{t-1} )    // AdamW-style
//! ```
//!
//! γ = 0 collapses MARS to AdamW. γ = 1 is the "full" Yuan et al.
//! prescription; the recommended sweet spot is `γ ≈ 0.025`.
//!
//! # When to use
//!
//! Anywhere variance-reduced SGD/Adam variants would help — noisy
//! gradients, small batches, RL-style on-policy training. State cost
//! per parameter: three buffers (`m`, `v`, previous-gradient cache).

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// MARS — variance-reduced AdamW. Per-tensor state: three `f32`
/// buffers (`m`, `v`, previous-gradient cache).
#[derive(Debug, Clone)]
pub struct Mars {
    /// Learning rate.
    pub lr: f32,
    /// First-moment EMA decay β₁. Default `0.95`.
    pub beta1: f32,
    /// Second-moment EMA decay β₂. Default `0.99`.
    pub beta2: f32,
    /// Denominator stability constant. Default `1e-8`.
    pub eps: f32,
    /// Decoupled weight-decay coefficient λ. Default `0.0`.
    pub weight_decay: f32,
    /// Variance-reduction strength γ (Yuan et al. eq. 7). `0.0`
    /// collapses MARS to plain AdamW; `1.0` is the full prescription.
    /// Default `0.025`.
    pub gamma: f32,
    /// If `true`, clip the variance-reduced surrogate `c_t` to unit
    /// norm per tensor (matches the "MARS-AdamW" recipe and keeps
    /// the VR kick from exploding when `g_{t-1}` is unrelated noise
    /// on early steps).
    pub clip_c: bool,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
    prev_g: HashMap<String, Vec<f32>>,
    /// Reusable scratch for the variance-reduced surrogate `c_t`.
    scratch: HashMap<String, Vec<f32>>,
}

impl Mars {
    /// Construct with `(β₁, β₂, ε, λ, γ, clip_c) = (0.95, 0.99, 1e-8, 0.0, 0.025, true)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.95,
            beta2: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            gamma: 0.025,
            clip_c: true,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
            prev_g: HashMap::new(),
            scratch: HashMap::new(),
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
}

impl Optimizer for Mars {
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
        let scale = self.gamma as f64 * b1 / (1.0 - b1);
        let eps = self.eps as f64;
        let lr = self.lr as f64;
        let wd = self.weight_decay as f64;
        // Four distinct fields ⇒ borrows can coexist.
        let prev = zeros_entry(&mut self.prev_g, name, param.len());
        let c = zeros_entry(&mut self.scratch, name, param.len());
        let m = zeros_entry(&mut self.m, name, param.len());
        let v = zeros_entry(&mut self.v, name, param.len());
        // c_t = g_t + scale * (g_t - g_{t-1})
        let mut c_sq_norm = 0.0f64;
        for i in 0..param.len() {
            let g = grad[i] as f64;
            let pg = prev[i] as f64;
            let ci = g + scale * (g - pg);
            c[i] = ci as f32;
            c_sq_norm += ci * ci;
            prev[i] = grad[i];
        }
        // Optional per-tensor norm clip on c (keeps the VR kick from
        // exploding on early steps when g_{t-1} is unrelated noise).
        if self.clip_c && c_sq_norm > 1.0 {
            let s = (1.0 / c_sq_norm.sqrt()) as f32;
            for ci in c.iter_mut() {
                *ci *= s;
            }
        }
        for i in 0..param.len() {
            let ci = c[i] as f64;
            let mi = b1 * m[i] as f64 + (1.0 - b1) * ci;
            let vi = b2 * v[i] as f64 + (1.0 - b2) * ci * ci;
            m[i] = mi as f32;
            v[i] = vi as f32;
            let m_hat = mi / bc1;
            let v_hat = vi / bc2;
            let p = param[i] as f64;
            param[i] = (p - lr * (m_hat / (v_hat.sqrt() + eps) + wd * p)) as f32;
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
