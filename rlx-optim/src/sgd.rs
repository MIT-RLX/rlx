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

//! Stochastic Gradient Descent with optional momentum and decoupled
//! L2 weight decay.
//!
//! # Update rules
//!
//! Vanilla SGD (`momentum = 0`):
//!
//! ```text
//! θ_{t+1} = θ_t − lr · (g_t + λ·θ_t)
//! ```
//!
//! Polyak momentum (`momentum = μ`, `nesterov = false`):
//!
//! ```text
//! v_{t+1} = μ·v_t + (g_t + λ·θ_t)
//! θ_{t+1} = θ_t − lr · v_{t+1}
//! ```
//!
//! Nesterov-accelerated SGD (`nesterov = true`):
//!
//! ```text
//! v_{t+1} = μ·v_t + (g_t + λ·θ_t)
//! θ_{t+1} = θ_t − lr · (g_t + λ·θ_t + μ·v_{t+1})
//! ```
//!
//! # When to use
//!
//! The default choice when training CNNs from scratch; with a
//! well-tuned `lr` schedule it still beats Adam on many vision
//! benchmarks. Cheap state (one buffer if `momentum > 0`).

use std::collections::HashMap;

use crate::Optimizer;
use crate::common::zeros_entry;

/// SGD with momentum / Nesterov / L2 weight decay.
///
/// All hyperparameters are public so callers can hot-swap them between
/// iterations (e.g. for a warm-up schedule). State is keyed by
/// parameter name; the same `Sgd` instance can drive every tensor in
/// a model.
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Learning rate. No default — pass it to [`Sgd::new`].
    pub lr: f32,
    /// Polyak momentum coefficient ∈ \[0, 1\). `0.0` disables momentum
    /// entirely (and the per-tensor velocity buffer is still allocated
    /// but unused — set via [`Sgd::with_momentum`] if you want it on).
    pub momentum: f32,
    /// Use Nesterov-accelerated momentum. Only meaningful when
    /// `momentum > 0`.
    pub nesterov: bool,
    /// L2 weight decay coefficient λ. Folded into the gradient
    /// *before* the momentum EMA (classical, **not** decoupled).
    /// Use [`crate::AdamW`]-style decoupling if you need that.
    pub weight_decay: f32,
    v: HashMap<String, Vec<f32>>,
}

impl Sgd {
    /// Construct with `lr` and momentum / decay disabled.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            nesterov: false,
            weight_decay: 0.0,
            v: HashMap::new(),
        }
    }

    /// Enable Polyak (or Nesterov) momentum.
    pub fn with_momentum(mut self, momentum: f32, nesterov: bool) -> Self {
        self.momentum = momentum;
        self.nesterov = nesterov;
        self
    }

    /// Set the L2 weight-decay coefficient.
    pub fn with_weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, name: &str, _shape: &[usize], param: &mut [f32], grad: &[f32]) {
        debug_assert_eq!(param.len(), grad.len());
        let v = zeros_entry(&mut self.v, name, param.len());
        let mu = self.momentum;
        let wd = self.weight_decay;
        let lr = self.lr;
        for i in 0..param.len() {
            let g = grad[i] + wd * param[i];
            if mu == 0.0 {
                param[i] -= lr * g;
            } else {
                v[i] = mu * v[i] + g;
                let update = if self.nesterov { g + mu * v[i] } else { v[i] };
                param[i] -= lr * update;
            }
        }
    }
}
