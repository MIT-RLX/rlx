// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Acquisition functions for Bayesian optimisation.
//!
//! All acquisitions are formulated for **minimisation** of the underlying
//! objective and return a number to *maximise* (better candidates score
//! higher). `f_best` is the current incumbent value.

use crate::gp::GpPosterior;

/// Expected Improvement (Mockus 1978). `f_best` is the current best
/// (lowest) observed value; xi is the standard exploration trade-off
/// (typically 0.0–0.05 of the y-range).
pub fn ei(gp: &GpPosterior, x: &[f64], f_best: f64, xi: f64) -> f64 {
    let (mu, sigma) = gp.mean_std(x);
    if sigma < 1e-12 {
        return 0.0;
    }
    let z = (f_best - mu - xi) / sigma;
    sigma * (z * cdf(z) + pdf(z))
}

/// Log-Expected Improvement (Ament et al. 2023) — numerically stable for
/// very high f_best or very small sigma. Returns `ln(EI)`.
pub fn log_ei(gp: &GpPosterior, x: &[f64], f_best: f64, xi: f64) -> f64 {
    let raw = ei(gp, x, f_best, xi).max(1e-300);
    raw.ln()
}

/// Probability of Improvement (Kushner 1964).
pub fn pi(gp: &GpPosterior, x: &[f64], f_best: f64, xi: f64) -> f64 {
    let (mu, sigma) = gp.mean_std(x);
    if sigma < 1e-12 {
        return 0.0;
    }
    let z = (f_best - mu - xi) / sigma;
    cdf(z)
}

/// Upper Confidence Bound — for **minimisation** it's `−mu + beta·sigma`
/// (i.e. lower-confidence-bound of f, which is what we want to avoid).
/// Larger `beta` → more exploration.
pub fn ucb(gp: &GpPosterior, x: &[f64], beta: f64) -> f64 {
    let (mu, sigma) = gp.mean_std(x);
    -mu + beta * sigma
}

#[inline]
fn pdf(z: f64) -> f64 {
    let inv_sqrt_2pi = 0.398_942_280_401_432_7;
    inv_sqrt_2pi * (-0.5 * z * z).exp()
}

#[inline]
fn cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz–Stegun 7.1.26 — max abs error ~1.5e-7.
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::{GpPosterior, Kernel};

    #[test]
    fn ei_positive_in_unobserved_region() {
        let gp = GpPosterior::fit(
            vec![vec![0.0], vec![1.0]],
            vec![0.5, 0.6],
            Kernel::Rbf {
                length_scale: 0.5,
                variance: 1.0,
            },
            1e-3,
        )
        .unwrap();
        let f_best = 0.5;
        let v = ei(&gp, &[0.5], f_best, 0.0);
        assert!(v > 0.0, "EI in unobserved region should be > 0, got {v}");
    }

    #[test]
    fn ucb_increases_with_beta() {
        let gp = GpPosterior::fit(
            vec![vec![0.0]],
            vec![0.0],
            Kernel::Rbf {
                length_scale: 1.0,
                variance: 1.0,
            },
            1e-3,
        )
        .unwrap();
        let x = vec![5.0];
        let lo = ucb(&gp, &x, 0.5);
        let hi = ucb(&gp, &x, 2.0);
        assert!(hi > lo);
    }
}
