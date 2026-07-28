// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! NAdamW — Nesterov-accelerated AdamW.
//!
//! Combines Dozat (2016, "Incorporating Nesterov Momentum into Adam")
//! with the decoupled-decay rule of [`crate::AdamW`]. The Nesterov
//! trick is a *lookahead*: instead of using the bias-corrected first
//! moment `m̂_t` directly, we use a convex combination of `m̂_{t+1}`
//! (predicted) and the current bias-corrected gradient.
//!
//! # Update rule
//!
//! ```text
//! m_t   = β₁·m_{t-1} + (1 − β₁)·g_t
//! v_t   = β₂·v_{t-1} + (1 − β₂)·g_t²
//! m̄_t   = β₁ · m_t / (1 − β₁^{t+1}) + (1 − β₁) · g_t / (1 − β₁ᵗ)
//! v̂_t   = v_t / (1 − β₂ᵗ)
//! θ_t   = θ_{t-1} − lr · ( m̄_t/(√v̂_t + ε) + λ·θ_{t-1} )
//! ```
//!
//! # When to use
//!
//! Slightly more aggressive than AdamW on the early steps; the
//! lookahead first-moment occasionally helps escape flat regions.
//! Same state cost as AdamW.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// Nesterov AdamW. Per-tensor state: two `f32` buffers.
#[derive(Debug, Clone)]
pub struct NAdamW {
    /// Learning rate.
    pub lr: f32,
    /// First-moment EMA decay β₁. Default `0.9`.
    pub beta1: f32,
    /// Second-moment EMA decay β₂. Default `0.999`.
    pub beta2: f32,
    /// Denominator stability constant. Default `1e-8`.
    pub eps: f32,
    /// Decoupled weight-decay coefficient λ. Default `0.01`.
    pub weight_decay: f32,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
}

impl NAdamW {
    /// Construct with `(β₁, β₂, ε, λ) = (0.9, 0.999, 1e-8, 0.01)`.
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
}

impl Optimizer for NAdamW {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let t = (self.step + 1) as f64;
        let b1 = self.beta1 as f64;
        let b2 = self.beta2 as f64;
        let bc1 = 1.0 - b1.powf(t);
        let bc1_next = 1.0 - b1.powf(t + 1.0);
        let bc2 = 1.0 - b2.powf(t);
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
            // Nesterov-corrected first moment (Dozat eq. 6):
            //   m_bar = b1 * m_hat_{t+1} + (1-b1)/bc1 * g
            let m_hat = mi / bc1_next;
            let m_bar = b1 * m_hat + (1.0 - b1) * g / bc1;
            let v_hat = vi / bc2;
            let p = param[i] as f64;
            param[i] = (p - lr * (m_bar / (v_hat.sqrt() + eps) + wd * p)) as f32;
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
