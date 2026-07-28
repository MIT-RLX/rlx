// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lion — EvoLved Sign Momentum (Chen et al., 2023, "Symbolic
//! Discovery of Optimization Algorithms").
//!
//! # Idea
//!
//! Lion was *discovered* by a program-synthesis search over candidate
//! optimizer expressions. The found rule is shockingly simple — one
//! momentum buffer, and the update is the **sign** of an
//! interpolation between the momentum and the gradient.
//!
//! # Update rule
//!
//! ```text
//! c_t   = β₁·m_{t-1} + (1 − β₁)·g_t
//! θ_t   = θ_{t-1} − lr · ( sign(c_t) + λ·θ_{t-1} )
//! m_t   = β₂·m_{t-1} + (1 − β₂)·g_t          // note: different β₂!
//! ```
//!
//! Two distinct betas: `β₁` shapes the *update direction* (faster
//! adaptation), `β₂` shapes the *carried momentum* (slower memory).
//!
//! # When to use
//!
//! Half the memory of Adam (one buffer instead of two), often
//! converges to similar quality on transformers when the LR is
//! tuned 3–10× lower than the corresponding AdamW LR. Sign updates
//! get coarse on tiny problems — favor large-batch / large-model
//! regimes.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::{zeros_entry, zip3_for_each};

/// EvoLved sign-momentum optimizer.
///
/// Per-tensor state: **one** `f32` buffer (half of Adam's footprint).
#[derive(Debug, Clone)]
pub struct Lion {
    /// Learning rate. **Critical**: typically 3–10× smaller than the
    /// AdamW LR you'd use on the same model (because the update has
    /// unit `‖sign(·)‖` per coordinate).
    pub lr: f32,
    /// Interpolation coefficient for the *update direction* (β₁ in
    /// Chen et al.). Default `0.9`.
    pub beta1: f32,
    /// EMA coefficient for the *carried momentum* (β₂). Default `0.99`.
    pub beta2: f32,
    /// Decoupled weight-decay coefficient λ. Tune ~3–10× higher than
    /// the AdamW λ you'd pair with the same model. Default `0.0`.
    pub weight_decay: f32,
    m: HashMap<String, Vec<f32>>,
}

impl Lion {
    /// Construct with `(β₁, β₂, λ) = (0.9, 0.99, 0.0)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
            m: HashMap::new(),
        }
    }

    /// Override (β₁, β₂). They serve different roles — see the
    /// struct-level docs.
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

impl Optimizer for Lion {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let b1 = self.beta1;
        let b2 = self.beta2;
        let lr = self.lr;
        let wd = self.weight_decay;
        let m = zeros_entry(&mut self.m, name, param.len());
        zip3_for_each(param, m, grad, |p, mi, gi| {
            // Update direction = sign(b1*m + (1-b1)*g)
            let c = b1 * *mi + (1.0 - b1) * gi;
            let sign = if c > 0.0 {
                1.0
            } else if c < 0.0 {
                -1.0
            } else {
                0.0
            };
            // Decoupled weight decay (matches Chen et al. eq. 1).
            *p -= lr * (sign + wd * *p);
            // Then update the momentum with a different β₂.
            *mi = b2 * *mi + (1.0 - b2) * gi;
        });
    }
}
