// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Learning-rate schedules for [`Func::train_step_at`](crate::Func::train_step_at).
//!
//! A schedule maps a global step to a learning rate; `train_step_at` calls
//! `Optimizer::set_lr` with that value before each step. Available with the
//! `optim` feature.

/// Common learning-rate schedules.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LrSchedule {
    /// Fixed learning rate.
    Constant(f32),
    /// Multiply `base` by `gamma` every `step_size` steps (staircase).
    Step {
        base: f32,
        step_size: usize,
        gamma: f32,
    },
    /// Cosine decay from `base` to `min` over `total` steps (then holds `min`).
    Cosine { base: f32, min: f32, total: usize },
    /// Linear warmup from 0 to `base` over `warmup` steps, then holds `base`.
    Warmup { base: f32, warmup: usize },
    /// Linear warmup to `base` over `warmup`, then cosine decay to `min` by
    /// `total` — the standard transformer schedule.
    WarmupCosine {
        base: f32,
        min: f32,
        warmup: usize,
        total: usize,
    },
}

impl LrSchedule {
    /// Learning rate at `step` (0-based).
    pub fn lr_at(&self, step: usize) -> f32 {
        match *self {
            LrSchedule::Constant(lr) => lr,
            LrSchedule::Step {
                base,
                step_size,
                gamma,
            } => {
                let k = step.checked_div(step_size).unwrap_or(0);
                base * gamma.powi(k as i32)
            }
            LrSchedule::Cosine { base, min, total } => cosine(base, min, step, total),
            LrSchedule::Warmup { base, warmup } => {
                if warmup > 0 && step < warmup {
                    base * (step as f32 + 1.0) / warmup as f32
                } else {
                    base
                }
            }
            LrSchedule::WarmupCosine {
                base,
                min,
                warmup,
                total,
            } => {
                if warmup > 0 && step < warmup {
                    base * (step as f32 + 1.0) / warmup as f32
                } else {
                    cosine(base, min, step - warmup, total.saturating_sub(warmup))
                }
            }
        }
    }
}

/// Cosine decay from `base` to `min` over `total` steps; holds `min` after.
fn cosine(base: f32, min: f32, step: usize, total: usize) -> f32 {
    if total == 0 {
        return min;
    }
    let t = (step.min(total)) as f32 / total as f32;
    let factor = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    min + (base - min) * factor
}
