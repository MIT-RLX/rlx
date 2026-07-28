// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Top-level Bayesian optimisation loop.
//!
//! Mirrors the existing `cmaes<F>()` / `pso<F>()` signature:
//!   `bo<F>(bbox, cfg, seed, f) -> BboSolution`.
//!
//! Strategy:
//!   1. Latin-hypercube warm-start of `n_init` evaluations.
//!   2. At each subsequent step: fit a GP, then maximise an acquisition
//!      function over a random+LHS candidate pool (no inner-loop
//!      optimiser — cheap and contention-free for low dimensions).

use crate::acquisition::{ei, log_ei, pi, ucb};
use crate::gp::{GpPosterior, Kernel};
use crate::sampling::lhs;
use crate::{BboSolution, Bbox};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Clone, Copy, Debug)]
pub enum Acquisition {
    Ei { xi: f64 },
    LogEi { xi: f64 },
    Pi { xi: f64 },
    Ucb { beta: f64 },
}

impl Default for Acquisition {
    fn default() -> Self {
        Acquisition::Ei { xi: 0.01 }
    }
}

#[derive(Clone, Debug)]
pub struct BoConfig {
    pub n_init: usize,
    pub n_iters: usize,
    pub kernel: Kernel,
    pub noise: f64,
    pub acquisition: Acquisition,
    /// Candidate pool size for inner acquisition maximisation per step.
    pub n_candidates: usize,
}

impl Default for BoConfig {
    fn default() -> Self {
        Self {
            n_init: 10,
            n_iters: 40,
            kernel: Kernel::Matern52 {
                length_scale: 1.0,
                variance: 1.0,
            },
            noise: 1e-3,
            acquisition: Acquisition::default(),
            n_candidates: 512,
        }
    }
}

pub fn bo<F>(bbox: &Bbox, cfg: &BoConfig, seed: u64, mut f: F) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let dim = bbox.dim();
    let init = lhs(cfg.n_init, dim, seed);
    let mut xs: Vec<Vec<f64>> = init
        .into_iter()
        .map(|u| {
            u.iter()
                .enumerate()
                .map(|(d, &v)| bbox.bounds[d].0 + v * (bbox.bounds[d].1 - bbox.bounds[d].0))
                .collect()
        })
        .collect();
    let mut ys: Vec<f64> = xs.iter().map(|x| f(x)).collect();
    let mut best_idx = argmin(&ys);
    let mut trace: Vec<f64> = (0..ys.len())
        .scan(f64::INFINITY, |acc, i| {
            if ys[i] < *acc {
                *acc = ys[i];
            }
            Some(*acc)
        })
        .collect();
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xB07));

    for it in 0..cfg.n_iters {
        // Standardise y for kernel stability (zero-mean, unit-std).
        let (y_mean, y_std) = mean_std(&ys);
        let y_std_safe = y_std.max(1e-9);
        let ys_std: Vec<f64> = ys.iter().map(|y| (y - y_mean) / y_std_safe).collect();
        let f_best_std = (ys[best_idx] - y_mean) / y_std_safe;

        let Some(gp) = GpPosterior::fit(xs.clone(), ys_std, cfg.kernel.clone(), cfg.noise) else {
            // Cholesky breakdown — fall back to a random sample.
            let x = bbox.sample(&mut rng);
            let y = f(&x);
            push_obs(&mut xs, &mut ys, &mut trace, &mut best_idx, x, y);
            continue;
        };

        // Build candidate pool: LHS + random.
        let lhs_n = (cfg.n_candidates / 2).max(1);
        let mut candidates: Vec<Vec<f64>> = lhs(lhs_n, dim, seed.wrapping_add(it as u64 * 13))
            .into_iter()
            .map(|u| {
                u.iter()
                    .enumerate()
                    .map(|(d, &v)| bbox.bounds[d].0 + v * (bbox.bounds[d].1 - bbox.bounds[d].0))
                    .collect()
            })
            .collect();
        while candidates.len() < cfg.n_candidates {
            candidates.push(bbox.sample(&mut rng));
        }

        let mut best_x = candidates[0].clone();
        let mut best_a = f64::NEG_INFINITY;
        for c in &candidates {
            let a = match cfg.acquisition {
                Acquisition::Ei { xi } => ei(&gp, c, f_best_std, xi),
                Acquisition::LogEi { xi } => log_ei(&gp, c, f_best_std, xi),
                Acquisition::Pi { xi } => pi(&gp, c, f_best_std, xi),
                Acquisition::Ucb { beta } => ucb(&gp, c, beta),
            };
            if a > best_a {
                best_a = a;
                best_x = c.clone();
            }
        }
        let y_new = f(&best_x);
        push_obs(&mut xs, &mut ys, &mut trace, &mut best_idx, best_x, y_new);
    }

    BboSolution {
        x: xs[best_idx].clone(),
        value: ys[best_idx],
        trace,
        n_evals: xs.len(),
    }
}

fn push_obs(
    xs: &mut Vec<Vec<f64>>,
    ys: &mut Vec<f64>,
    trace: &mut Vec<f64>,
    best_idx: &mut usize,
    x: Vec<f64>,
    y: f64,
) {
    xs.push(x);
    ys.push(y);
    let cur_best = ys[*best_idx];
    let new = ys.len() - 1;
    if y < cur_best {
        *best_idx = new;
        trace.push(y);
    } else {
        trace.push(cur_best);
    }
}

fn argmin(v: &[f64]) -> usize {
    let mut bi = 0;
    let mut bv = v[0];
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x < bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}
