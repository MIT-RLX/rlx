// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Learning-rate schedules: the factor on the base learning rate at each
//! update, as `torch.optim.lr_scheduler` computes it.
//!
//! `factor(i)` is for the update with 0-based index `i` (the first update is
//! 0), which is the rate PyTorch's optimizer uses after `i` calls to
//! `scheduler.step()`. A linear warmup over the first `warmup` updates (from
//! `warmup_start` of the rate to all of it) multiplies any schedule.

use std::f64::consts::PI;

/// The shape of a schedule.
#[derive(Clone, Debug, PartialEq)]
pub enum ScheduleKind {
    /// The base rate throughout.
    Constant,
    /// `LinearLR` after warmup: from 1 to `end_factor` over the updates
    /// between `warmup` and `total`.
    Linear { total: u64, end_factor: f64 },
    /// `CosineAnnealingLR` after warmup: from 1 to `min_factor` along a half
    /// cosine over the updates between `warmup` and `total`.
    Cosine { total: u64, min_factor: f64 },
    /// `StepLR`: × `gamma` every `step_size` updates.
    Step { step_size: u64, gamma: f64 },
    /// `MultiStepLR`: × `gamma` at each milestone.
    MultiStep { milestones: Vec<u64>, gamma: f64 },
    /// `ExponentialLR`: × `gamma` every update.
    Exponential { gamma: f64 },
    /// `PolynomialLR` after warmup: from 1 to `end_factor` as `(1 − t)^power`.
    Polynomial {
        total: u64,
        power: f64,
        end_factor: f64,
    },
    /// `OneCycleLR` (cosine annealing): from `1/div_factor` up to 1 over the
    /// first `pct_start` of `total`, then down to `1/(div_factor ×
    /// final_div_factor)`. The base rate is the peak; it carries its own
    /// warmup, so `warmup` should be 0.
    OneCycle {
        total: u64,
        pct_start: f64,
        div_factor: f64,
        final_div_factor: f64,
    },
}

/// A schedule with an optional linear warmup.
#[derive(Clone, Debug, PartialEq)]
pub struct LrSchedule {
    pub kind: ScheduleKind,
    pub warmup: u64,
    /// The warmup's first factor (`LinearLR`'s `start_factor`). Default 0.
    pub warmup_start: f64,
}

impl LrSchedule {
    pub fn new(kind: ScheduleKind) -> Self {
        Self {
            kind,
            warmup: 0,
            warmup_start: 0.0,
        }
    }

    pub fn with_warmup(mut self, updates: u64, start: f64) -> Self {
        self.warmup = updates;
        self.warmup_start = start;
        self
    }

    /// The factor on the base rate for update `i` (0-based).
    pub fn factor(&self, i: u64) -> f64 {
        let w = self.warmup;
        // The fraction of the decay phase done: after warmup, up to `total`.
        let phase = |total: u64| {
            if total <= w {
                1.0
            } else {
                ((i.saturating_sub(w)) as f64 / (total - w) as f64).clamp(0.0, 1.0)
            }
        };
        let f = match &self.kind {
            ScheduleKind::Constant => 1.0,
            ScheduleKind::Linear { total, end_factor } => 1.0 + (end_factor - 1.0) * phase(*total),
            ScheduleKind::Cosine { total, min_factor } => {
                min_factor + (1.0 - min_factor) * 0.5 * (1.0 + (PI * phase(*total)).cos())
            }
            ScheduleKind::Step { step_size, gamma } => gamma.powi((i / (*step_size).max(1)) as i32),
            ScheduleKind::MultiStep { milestones, gamma } => {
                gamma.powi(milestones.iter().filter(|m| **m <= i).count() as i32)
            }
            ScheduleKind::Exponential { gamma } => gamma.powi(i as i32),
            ScheduleKind::Polynomial {
                total,
                power,
                end_factor,
            } => end_factor + (1.0 - end_factor) * (1.0 - phase(*total)).powf(*power),
            ScheduleKind::OneCycle {
                total,
                pct_start,
                div_factor,
                final_div_factor,
            } => {
                let start = 1.0 / div_factor;
                let end = start / final_div_factor;
                // torch's phase ends: pct_start·total − 1, then total − 1.
                let up = (pct_start * *total as f64 - 1.0).max(1.0);
                let down = (*total as f64 - 1.0).max(up + 1.0);
                let cos =
                    |from: f64, to: f64, t: f64| to + (from - to) * 0.5 * (1.0 + (PI * t).cos());
                let x = i as f64;
                if x <= up {
                    cos(start, 1.0, x / up)
                } else {
                    cos(1.0, end, ((x - up) / (down - up)).min(1.0))
                }
            }
        };
        if i < w {
            f * (self.warmup_start + (1.0 - self.warmup_start) * i as f64 / w as f64)
        } else {
            f
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn schedules_follow_torch() {
        let cos = LrSchedule::new(ScheduleKind::Cosine {
            total: 100,
            min_factor: 0.0,
        });
        assert!(
            close(cos.factor(0), 1.0) && close(cos.factor(50), 0.5) && close(cos.factor(100), 0.0)
        );
        let warm = cos.clone().with_warmup(10, 0.0);
        assert!(
            close(warm.factor(0), 0.0) && close(warm.factor(5), 0.5) && close(warm.factor(10), 1.0)
        );
        assert!(
            close(warm.factor(55), 0.5),
            "the cosine runs over updates 10..100"
        );
        let step = LrSchedule::new(ScheduleKind::Step {
            step_size: 30,
            gamma: 0.1,
        });
        assert!(
            close(step.factor(29), 1.0)
                && close(step.factor(30), 0.1)
                && close(step.factor(60), 0.01)
        );
        let ms = LrSchedule::new(ScheduleKind::MultiStep {
            milestones: vec![10, 20],
            gamma: 0.5,
        });
        assert!(
            close(ms.factor(9), 1.0) && close(ms.factor(10), 0.5) && close(ms.factor(25), 0.25)
        );
        let lin = LrSchedule::new(ScheduleKind::Linear {
            total: 10,
            end_factor: 0.0,
        });
        assert!(close(lin.factor(5), 0.5) && close(lin.factor(20), 0.0));
        let poly = LrSchedule::new(ScheduleKind::Polynomial {
            total: 10,
            power: 2.0,
            end_factor: 0.0,
        });
        assert!(close(poly.factor(5), 0.25));
        assert!(close(
            LrSchedule::new(ScheduleKind::Exponential { gamma: 0.9 }).factor(2),
            0.81
        ));
    }

    #[test]
    fn one_cycle_starts_low_peaks_and_ends_lower() {
        let oc = LrSchedule::new(ScheduleKind::OneCycle {
            total: 1000,
            pct_start: 0.3,
            div_factor: 25.0,
            final_div_factor: 1e4,
        });
        assert!(close(oc.factor(0), 1.0 / 25.0));
        assert!(
            close(oc.factor(299), 1.0),
            "the peak at pct_start · total − 1"
        );
        assert!(close(oc.factor(999), 1.0 / 25.0 / 1e4));
        assert!(oc.factor(500) < 1.0 && oc.factor(500) > oc.factor(800));
    }
}
