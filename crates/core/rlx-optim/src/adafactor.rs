// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adafactor (Shazeer & Stern, 2018, "Adafactor: Adaptive Learning
//! Rates with Sublinear Memory Cost").
//!
//! # Idea
//!
//! Adam's `v_t` is the same shape as θ — for a 70B-parameter model
//! that's 280 GB of optimizer state. Adafactor *factorizes* the
//! second-moment matrix: for a 2-D parameter of shape `m × n`, instead
//! of an `m·n` buffer it stores a row-statistic `R ∈ ℝᵐ` and a
//! column-statistic `C ∈ ℝⁿ`, then reconstructs
//! `V̂_{ij} ≈ R_i · C_j / Σ_k R_k`. State drops from `O(m·n)` to
//! `O(m + n)`.
//!
//! # Update rule (this impl: factored 2nd-moment, no 1st-moment)
//!
//! Let `β₂_t = 1 − t^{decay_rate}` (default decay_rate = −0.8). For a
//! 2-D parameter:
//!
//! ```text
//! R_i = β₂_t·R_i + (1−β₂_t)·mean_j(g_ij² + ε₁)
//! C_j = β₂_t·C_j + (1−β₂_t)·mean_i(g_ij² + ε₁)
//! V̂_{ij} = R_i · C_j / Σ_k R_k
//! u_{ij}  = g_{ij} / √V̂_{ij}
//! u ← u / max(1, RMS(u) / clip_threshold)        // RMS-of-update clip
//! lr_t    = manual_lr OR  min(1/√t, 1e-2) · max(ε₂, RMS(θ))   // relative step
//! θ_t     = θ_{t-1} − lr_t · ( u + λ·θ_{t-1} )
//! ```
//!
//! For non-2-D parameters (bias vectors, 4-D conv weights) we fall
//! back to a full per-element EMA — the savings are negligible there
//! anyway. The optional first-moment EMA is **not** implemented
//! (matches the recommended T5 configuration).
//!
//! # When to use
//!
//! When you don't have memory for Adam-style optimizer state — large
//! models, low-VRAM fine-tuning, sequence-length scaling experiments.
//! State cost per matrix = `m + n` floats vs Adam's `2·m·n`.

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::{l2_norm, zeros_entry};

/// Adafactor — factored-second-moment optimizer.
///
/// Per-tensor state: a `rows`-vector + a `cols`-vector for 2-D
/// parameters (sublinear in `rows·cols`), or a full EMA for non-2-D.
#[derive(Debug, Clone)]
pub struct Adafactor {
    /// Optional manual learning rate. `None` ⇒ use the "relative
    /// step" rule `min(1/√t, 1e-2) · max(ε₂, RMS(θ))` from the paper.
    /// Default `None`.
    pub lr: Option<f32>,
    /// β₂_t decay-rate exponent. `β₂_t = 1 − tˣ` with `x = -0.8`
    /// (default) means slow decay early, full decay asymptotically.
    pub beta2_decay: f32,
    /// Squared-gradient stability constant added before each row /
    /// column average. Default `1e-30`.
    pub eps1: f32,
    /// RMS-of-parameter floor for the relative-step rule. Default `1e-3`.
    pub eps2: f32,
    /// Update-RMS clipping threshold (Shazeer & Stern §6). Default `1.0`.
    pub clip_threshold: f32,
    /// Decoupled weight-decay coefficient λ. Default `0.0`.
    pub weight_decay: f32,
    step: u64,
    // Per-parameter state.
    r: HashMap<String, Vec<f32>>, // row factor (length rows) for 2D
    c: HashMap<String, Vec<f32>>, // col factor (length cols) for 2D
    v: HashMap<String, Vec<f32>>, // full EMA for non-2D
}

impl Adafactor {
    /// Construct with paper defaults (no manual lr ⇒ relative step,
    /// `decay_rate = -0.8`, `ε₁=1e-30, ε₂=1e-3, clip=1.0, λ=0.0`).
    pub fn new() -> Self {
        Self {
            lr: None,
            beta2_decay: -0.8,
            eps1: 1e-30,
            eps2: 1e-3,
            clip_threshold: 1.0,
            weight_decay: 0.0,
            step: 0,
            r: HashMap::new(),
            c: HashMap::new(),
            v: HashMap::new(),
        }
    }

    /// Switch from the relative-step rule to a manual learning rate.
    pub fn with_lr(mut self, lr: f32) -> Self {
        self.lr = Some(lr);
        self
    }

    /// Override the decoupled-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Default for Adafactor {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for Adafactor {
    fn step(&mut self, name: &str, shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let t = (self.step + 1) as f64;
        // β₂_t = 1 − t^{beta2_decay}, decay_rate ∈ (-1, 0).
        let beta2_t = 1.0 - t.powf(self.beta2_decay as f64);
        let eps1 = self.eps1 as f64;
        let clip = self.clip_threshold as f64;
        let n = param.len();

        // ── Update second-moment estimate ──────────────────────────
        let mut update = vec![0.0f32; n];
        if shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            debug_assert_eq!(rows * cols, n);
            let r = zeros_entry(&mut self.r, name, rows);
            // Row factor: average of g² across columns, then EMA.
            let mut row_buf = vec![0.0f64; rows];
            for i in 0..rows {
                let mut s = 0.0f64;
                for j in 0..cols {
                    let g = grad[i * cols + j] as f64;
                    s += g * g + eps1;
                }
                row_buf[i] = s / cols as f64;
            }
            for i in 0..rows {
                r[i] = (beta2_t * r[i] as f64 + (1.0 - beta2_t) * row_buf[i]) as f32;
            }
            let r_snapshot: Vec<f32> = r.clone();

            // Column factor: average of g² across rows, then EMA.
            let c = zeros_entry(&mut self.c, name, cols);
            let mut col_buf = vec![0.0f64; cols];
            for j in 0..cols {
                let mut s = 0.0f64;
                for i in 0..rows {
                    let g = grad[i * cols + j] as f64;
                    s += g * g + eps1;
                }
                col_buf[j] = s / rows as f64;
            }
            for j in 0..cols {
                c[j] = (beta2_t * c[j] as f64 + (1.0 - beta2_t) * col_buf[j]) as f32;
            }
            let r_sum: f64 = r_snapshot.iter().map(|&x| x as f64).sum();
            // v_ij = r_i * c_j / (sum_k r_k). Build update = g / sqrt(v).
            for i in 0..rows {
                for j in 0..cols {
                    let v_ij = r_snapshot[i] as f64 * c[j] as f64 / r_sum.max(eps1);
                    let g = grad[i * cols + j] as f64;
                    update[i * cols + j] = (g / v_ij.sqrt().max(eps1.sqrt())) as f32;
                }
            }
        } else {
            // Non-2D: full per-element EMA.
            let v = zeros_entry(&mut self.v, name, n);
            for i in 0..n {
                let g = grad[i] as f64;
                v[i] = (beta2_t * v[i] as f64 + (1.0 - beta2_t) * (g * g + eps1)) as f32;
                update[i] = (g / (v[i] as f64).sqrt().max(eps1.sqrt())) as f32;
            }
        }

        // RMS-of-update clipping (Shazeer & Stern §6).
        let u_rms = (l2_norm(&update) as f64 / (n as f64).sqrt()).max(1.0 / clip);
        let scale = (1.0 / (u_rms * clip)).min(1.0);
        for u in update.iter_mut() {
            *u = (*u as f64 * scale) as f32;
        }

        // Learning rate (relative-step or manual).
        let lr = match self.lr {
            Some(x) => x as f64,
            None => {
                let p_rms = (l2_norm(param) as f64 / (n as f64).sqrt()).max(self.eps2 as f64);
                (1.0 / t.sqrt()).min(1e-2) * p_rms
            }
        };
        let wd = self.weight_decay as f64;
        for i in 0..n {
            let p = param[i] as f64;
            param[i] = (p - lr * (update[i] as f64 + wd * p)) as f32;
        }
    }

    fn end_iteration(&mut self) {
        self.step += 1;
    }
}
