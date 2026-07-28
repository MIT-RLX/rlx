// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Central-difference + Adam gradient descent for scalar objectives.

use rlx_optim::{Adam, Optimizer};
use serde::{Deserialize, Serialize};

const PARAM_KEY: &str = "params";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamOptConfig {
    pub steps: u32,
    pub lr: f32,
    pub fd_rel: f64,
    pub fd_min: f64,
}

impl Default for AdamOptConfig {
    fn default() -> Self {
        Self {
            steps: 32,
            lr: 2_000.0,
            fd_rel: 0.02,
            fd_min: 50.0,
        }
    }
}

impl AdamOptConfig {
    #[must_use]
    pub fn from_evals(evals: usize) -> Self {
        Self {
            steps: evals.max(4) as u32,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamOptResult {
    pub params: Vec<f64>,
    pub final_loss: f64,
    pub history: Vec<f64>,
}

pub fn adam_opt_1d(
    x0: f64,
    bounds: (f64, f64),
    cfg: &AdamOptConfig,
    mut loss_at: impl FnMut(f64) -> f64,
) -> AdamOptResult {
    adam_opt_nd(&[x0], &[bounds], cfg, |x| loss_at(x[0]))
}

pub fn adam_opt_nd(
    x0: &[f64],
    bounds: &[(f64, f64)],
    cfg: &AdamOptConfig,
    mut loss_at: impl FnMut(&[f64]) -> f64,
) -> AdamOptResult {
    assert_eq!(x0.len(), bounds.len(), "adam_opt_nd: dim mismatch");
    let _n = x0.len();
    let mut x: Vec<f64> = x0.to_vec();
    let mut opt = Adam::new(cfg.lr);
    let mut history = Vec::with_capacity(cfg.steps as usize + 1);
    let l0 = loss_at(&x);
    history.push(l0);

    for _step in 1..=cfg.steps {
        let grads = central_fd_grad_nd(&mut loss_at, &x, cfg.fd_rel, cfg.fd_min);
        adam_step_f64(&mut x, &grads, &mut opt, cfg.lr);
        for (xi, &(lo, hi)) in x.iter_mut().zip(bounds) {
            *xi = xi.clamp(lo, hi);
        }
        history.push(loss_at(&x));
    }

    AdamOptResult {
        params: x,
        final_loss: *history.last().unwrap_or(&l0),
        history,
    }
}

/// One Adam step on f64 params via f32 cast (rlx-optim is f32-native).
fn adam_step_f64(x: &mut [f64], grads: &[f64], opt: &mut Adam, lr: f32) {
    let n = x.len();
    let mut xf: Vec<f32> = x.iter().map(|v| *v as f32).collect();
    let gf: Vec<f32> = grads.iter().map(|v| *v as f32).collect();
    opt.lr = lr;
    opt.step(PARAM_KEY, &[n], &mut xf, &gf);
    opt.end_iteration();
    for (xi, &v) in x.iter_mut().zip(xf.iter()) {
        *xi = f64::from(v);
    }
}

fn central_fd_grad_nd(
    loss_at: &mut impl FnMut(&[f64]) -> f64,
    x: &[f64],
    fd_rel: f64,
    fd_min: f64,
) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, &xi)| {
            let h = (xi.abs() * fd_rel).max(fd_min);
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += h;
            xm[i] -= h;
            (loss_at(&xp) - loss_at(&xm)) / (2.0 * h)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convex_2d_loss_decreases() {
        let cfg = AdamOptConfig {
            steps: 24,
            lr: 0.2,
            ..AdamOptConfig::default()
        };
        let res = adam_opt_nd(&[4.0, -3.0], &[(-10.0, 10.0), (-10.0, 10.0)], &cfg, |p| {
            (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2)
        });
        assert!(res.final_loss < res.history[0]);
    }
}
