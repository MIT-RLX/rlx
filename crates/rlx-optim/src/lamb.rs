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

//! LAMB — Layer-wise Adaptive Moments for Batch training (You et al.,
//! 2019, "Large Batch Optimization for Deep Learning: Training BERT
//! in 76 minutes").
//!
//! # Idea
//!
//! Naïve large-batch training stalls because the per-coordinate Adam
//! step doesn't account for the magnitude difference between
//! different layers' weights. LAMB rescales each tensor's Adam-style
//! update by the **trust ratio** `‖θ‖ / ‖u‖`, so that the per-step
//! relative change `‖Δθ‖ / ‖θ‖` is bounded and identical across layers.
//!
//! # Update rule
//!
//! For each tensor (and its flat parameter vector θ):
//!
//! ```text
//! m_t = β₁·m_{t-1} + (1 − β₁)·g_t
//! v_t = β₂·v_{t-1} + (1 − β₂)·g_t²
//! u_t = m̂_t / (√v̂_t + ε) + λ·θ_{t-1}        // raw update
//! r_t = ‖θ_{t-1}‖₂ / ‖u_t‖₂                  // trust ratio
//! θ_t = θ_{t-1} − lr · r_t · u_t
//! ```
//!
//! `r_t` is clamped to `1.0` when either norm is zero (warm-up edge
//! case). LAMB's headline result is that this rescaling makes very
//! large batch sizes (32k–64k) viable without quality loss.
//!
//! # When to use
//!
//! Large-batch pre-training — BERT/ViT/Llama-scale data-parallel
//! runs. State cost = Adam (two buffers).

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::{l2_norm, zeros_entry};

/// Layer-wise Adaptive Moments for Batch training.
///
/// Per-tensor state: two `f32` buffers + a per-call scratch buffer
/// for the trust-ratio numerator (allocated inside [`Optimizer::step`]).
#[derive(Debug, Clone)]
pub struct Lamb {
    /// Learning rate.
    pub lr: f32,
    /// First-moment EMA decay β₁. Default `0.9`.
    pub beta1: f32,
    /// Second-moment EMA decay β₂. Default `0.999`.
    pub beta2: f32,
    /// Denominator stability constant. Default `1e-6` (looser than
    /// Adam's `1e-8` — matches NVIDIA's reference).
    pub eps: f32,
    /// Decoupled weight-decay coefficient λ. Default `0.01`.
    pub weight_decay: f32,
    /// If `true`, divide by bias-corrected moments. Defaults to `true`
    /// (matches NVIDIA's reference impl); the original paper omits it.
    pub bias_correction: bool,
    step: u64,
    m: HashMap<String, Vec<f32>>,
    v: HashMap<String, Vec<f32>>,
    /// Reusable per-tensor scratch buffer for the trust-ratio
    /// numerator. Cached so we don't allocate every step.
    scratch: HashMap<String, Vec<f32>>,
}

impl Lamb {
    /// Construct with `(β₁, β₂, ε, λ) = (0.9, 0.999, 1e-6, 0.01)`.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-6,
            weight_decay: 0.01,
            bias_correction: true,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
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

impl Optimizer for Lamb {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let t = (self.step + 1) as f64;
        let b1 = self.beta1 as f64;
        let b2 = self.beta2 as f64;
        let (bc1, bc2) = if self.bias_correction {
            (1.0 - b1.powf(t), 1.0 - b2.powf(t))
        } else {
            (1.0, 1.0)
        };
        let eps = self.eps as f64;
        let lr = self.lr;
        let wd = self.weight_decay as f64;
        let m = zeros_entry(&mut self.m, name, param.len());
        let v = zeros_entry(&mut self.v, name, param.len());
        let update = zeros_entry(&mut self.scratch, name, param.len());
        // First pass: update m/v, build `r_i = m_hat / (sqrt(v_hat) + eps) + wd * w`.
        for i in 0..param.len() {
            let g = grad[i] as f64;
            let mi = b1 * m[i] as f64 + (1.0 - b1) * g;
            let vi = b2 * v[i] as f64 + (1.0 - b2) * g * g;
            m[i] = mi as f32;
            v[i] = vi as f32;
            let m_hat = mi / bc1;
            let v_hat = vi / bc2;
            update[i] = (m_hat / (v_hat.sqrt() + eps) + wd * param[i] as f64) as f32;
        }
        let w_norm = l2_norm(param);
        let r_norm = l2_norm(update);
        let trust = if w_norm > 0.0 && r_norm > 0.0 {
            w_norm / r_norm
        } else {
            1.0
        };
        let step_size = lr * trust;
        for i in 0..param.len() {
            param[i] -= step_size * update[i];
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
