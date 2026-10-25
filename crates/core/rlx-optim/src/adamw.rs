// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    /// When true, moment/parameter updates stay in pure `f32` (PyTorch AdamW /
    /// `foreach=False` default). Default `false` uses `f64` intermediates
    /// (slightly more accurate, but drifts vs PyTorch shared-init checks).
    pub f32_math: bool,
    /// When true, the exact operation sequence of PyTorch's single-tensor
    /// `torch.optim.AdamW` (the CPU default) — see [`AdamW::with_torch_math`].
    /// Takes precedence over `f32_math`.
    pub torch_math: bool,
    /// The learning rate as the Python float torch keeps it; `torch_math`
    /// uses it when set ([`AdamW::set_lr_f64`]). [`Optimizer::set_lr`] clears it.
    pub lr_f64: Option<f64>,
    /// `(β₁, β₂, weight_decay)` as the Python floats torch holds; `torch_math`
    /// uses them when set ([`AdamW::torch`]). The f32 fields round `0.9` to
    /// `0.899999976…`, which shifts `1 − β₁ᵗ` and `1 − lr·wd`.
    pub torch_hp: Option<(f64, f64, f64)>,
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
            f32_math: false,
            torch_math: false,
            lr_f64: None,
            torch_hp: None,
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

    /// Pure-`f32` AdamW arithmetic (match PyTorch training trajectories).
    pub fn with_f32_math(mut self, on: bool) -> Self {
        self.f32_math = on;
        self
    }

    /// PyTorch's single-tensor AdamW, operation for operation:
    ///
    /// ```text
    /// p.mul_(1 − lr·wd)                 # the factor a Python float, rounded to f32
    /// m.lerp_(g, 1 − β₁)                # weight < 0.5: m + w·(g − m)
    /// v.mul_(β₂).addcmul_(g, g, 1 − β₂) # v·β₂ + (w·g)·g
    /// bc1, bc2 = 1 − β₁ᵗ, 1 − β₂ᵗ       # Python floats (f64)
    /// denom = sqrt(v) / sqrt(bc2) + eps # NOT sqrt(v / bc2) + eps
    /// p.addcdiv_(m, denom, −lr/bc1)
    /// ```
    ///
    /// The `f32_math` mode differs in all four marked places, each by a last-
    /// bit rounding that compounds over a training run.
    pub fn with_torch_math(mut self, on: bool) -> Self {
        self.torch_math = on;
        self
    }

    /// PyTorch's `torch.optim.AdamW(params, lr, betas, weight_decay)` (eps
    /// `1e-8`), exactly: [`Self::with_torch_math`] with the hyperparameters
    /// kept at f64, as torch keeps them.
    pub fn torch(lr: f64, betas: (f64, f64), weight_decay: f64) -> Self {
        let mut o = Self::new(lr as f32)
            .with_betas(betas.0 as f32, betas.1 as f32)
            .with_weight_decay(weight_decay as f32)
            .with_torch_math(true);
        o.lr_f64 = Some(lr);
        o.torch_hp = Some((betas.0, betas.1, weight_decay));
        o
    }

    /// Set the learning rate at f64 precision. torch holds `lr` as a Python
    /// float, so a scheduled value (`5e-5 · step / warmup`) rounded to f32
    /// first would change `lr / bc1` in the last bit.
    pub fn set_lr_f64(&mut self, lr: f64) {
        self.lr = lr as f32;
        self.lr_f64 = Some(lr);
    }
}

// ── checkpointing ───────────────────────────────────────────

impl AdamW {
    /// Named accumulators plus the step counter — see
    /// [`crate::OptimizerState`].
    pub(crate) fn snapshot(&self) -> crate::OptimizerState {
        let mut out = crate::OptimizerState {
            step: self.step,
            buffers: Vec::new(),
        };
        out.extend_slot("m", &self.m);
        out.extend_slot("v", &self.v);
        out
    }

    pub(crate) fn restore(&mut self, state: &crate::OptimizerState) {
        self.step = state.step;
        state.take_slot("m", &mut self.m);
        state.take_slot("v", &mut self.v);
    }
}

impl Optimizer for AdamW {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        self.lr_f64 = None;
    }

    fn state_dict(&self) -> Option<crate::OptimizerState> {
        Some(self.snapshot())
    }

    fn load_state_dict(&mut self, state: &crate::OptimizerState) -> bool {
        self.restore(state);
        true
    }

    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let m = zeros_entry(&mut self.m, name, param.len());
        let v = zeros_entry(&mut self.v, name, param.len());
        if self.torch_math {
            let t = (self.step + 1) as i32;
            let lr = self.lr_f64.unwrap_or(self.lr as f64);
            let (b1, b2, wd) = self.torch_hp.unwrap_or((
                self.beta1 as f64,
                self.beta2 as f64,
                self.weight_decay as f64,
            ));
            let decay = (1.0 - lr * wd) as f32;
            let w1 = (1.0 - b1) as f32;
            let (b2f, w2) = (b2 as f32, (1.0 - b2) as f32);
            let step_size = (lr / (1.0 - b1.powi(t))) as f32;
            let bc2_sqrt = (1.0 - b2.powi(t)).sqrt() as f32;
            let eps = self.eps;
            zip4_for_each(param, m, v, grad, |p, mi, vi, gi| {
                *p *= decay;
                *mi += w1 * (gi - *mi);
                *vi = *vi * b2f + w2 * gi * gi;
                let denom = vi.sqrt() / bc2_sqrt + eps;
                *p += -step_size * (*mi / denom);
            });
        } else if self.f32_math {
            let t = (self.step + 1) as i32;
            let b1 = self.beta1;
            let b2 = self.beta2;
            let bc1 = 1.0 - b1.powi(t);
            let bc2 = 1.0 - b2.powi(t);
            let eps = self.eps;
            let lr = self.lr;
            let wd = self.weight_decay;
            zip4_for_each(param, m, v, grad, |p, mi, vi, gi| {
                let new_m = b1 * *mi + (1.0 - b1) * gi;
                let new_v = b2 * *vi + (1.0 - b2) * (gi * gi);
                *mi = new_m;
                *vi = new_v;
                let m_hat = new_m / bc1;
                let v_hat = new_v / bc2;
                // PyTorch AdamW: param *= (1 - lr*wd); param += -lr * m_hat/(sqrt(v_hat)+eps)
                *p = *p * (1.0 - lr * wd) - lr * (m_hat / (v_hat.sqrt() + eps));
            });
        } else {
            let t = (self.step + 1) as f64;
            let b1 = self.beta1 as f64;
            let b2 = self.beta2 as f64;
            let bc1 = 1.0 - b1.powf(t);
            let bc2 = 1.0 - b2.powf(t);
            let eps = self.eps as f64;
            let lr = self.lr as f64;
            let wd = self.weight_decay as f64;
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
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}

#[cfg(test)]
mod torch_math_tests {
    use super::*;

    /// torch's single-tensor AdamW, transcribed independently, two steps.
    fn reference(p0: f32, g: [f32; 2], lr: f64, wd: f64, b1: f64, b2: f64, eps: f32) -> f32 {
        let (mut p, mut m, mut v) = (p0, 0f32, 0f32);
        for (i, gi) in g.iter().enumerate() {
            let t = (i + 1) as i32;
            p *= (1.0 - lr * wd) as f32;
            m += (1.0 - b1) as f32 * (gi - m);
            v = v * b2 as f32 + (1.0 - b2) as f32 * gi * gi;
            let denom = v.sqrt() / ((1.0 - b2.powi(t)).sqrt() as f32) + eps;
            p += -((lr / (1.0 - b1.powi(t))) as f32) * (m / denom);
        }
        p
    }

    #[test]
    fn torch_math_is_torchs_operation_sequence_with_an_f64_lr() {
        // lr = 1/7919 with these gradients: the f32-rounded lr ends one ULP
        // away (found by search; for many lr values the rounding vanishes)
        let (lr, wd, b1, b2) = (1.0 / 7919.0, 0.01, 0.9, 0.95);
        // p0 = 0: the update IS the value, so a one-ULP change of the step
        // size shows (next to a value of order 1 it is far below one ULP)
        let (p0, g) = (0.0_f32, [0.25_f32, -0.0625]);
        let mut opt = AdamW::torch(lr, (b1, b2), wd);
        let mut p = [p0];
        for gi in g {
            opt.step("p", &[1], &mut p, &[gi]);
            opt.end_iteration();
        }
        assert_eq!(
            p[0].to_bits(),
            reference(p0, g, lr, wd, b1, b2, 1e-8).to_bits()
        );
        // the f32-rounded lr gives a different trajectory, so this pins the f64 lr
        assert_ne!(
            p[0].to_bits(),
            reference(p0, g, (lr as f32) as f64, wd, b1, b2, 1e-8).to_bits()
        );
    }
}
