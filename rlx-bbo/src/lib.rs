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
//! Black-box optimization + FMQ/QGBS search (domain-agnostic `f64` objectives).
//!
//! For compiled flow-map **policies** and RLX-graph FMQ training, use [`rlx-rl`](../rlx-rl/).

mod cmaes;
mod flow_map;
mod q_guidance;
mod surrogate;
mod trajectory;
mod twin;

pub use cmaes::{cmaes, CmaesConfig};
pub use flow_map::{
    fmq_surrogate_step, load_flow_map, save_flow_map, train_from_jsonl, LinearFlowMap,
};
pub use q_guidance::{
    eta_eff_twin, finite_diff_grad, q_guided_beam_search, q_steered_search,
    q_steered_search_with_grad, search_by_method, trust_region_q_step, QSteerConfig, QgbsConfig,
    DEFAULT_KAPPA,
};
pub use surrogate::{
    fit_from_trajectory_jsonl, fit_linear_surrogate, load_surrogate, save_surrogate,
    LinearSurrogate,
};
pub use trajectory::{append_jsonl, diagonal_flow_pairs, load_jsonl, TrajectoryRecord};
pub use twin::q_steered_search_twin;

use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BboSolution {
    pub x: Vec<f64>,
    pub value: f64,
    pub trace: Vec<f64>,
    pub n_evals: usize,
}

#[derive(Clone, Debug)]
pub struct Bbox {
    pub bounds: Vec<(f64, f64)>,
}

impl Bbox {
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        Self { bounds }
    }
    pub fn dim(&self) -> usize {
        self.bounds.len()
    }
    pub fn sample(&self, rng: &mut StdRng) -> Vec<f64> {
        self.bounds
            .iter()
            .map(|&(lo, hi)| rng.gen_range(lo..=hi))
            .collect()
    }
    pub fn clip(&self, x: &mut [f64]) {
        for (xi, &(lo, hi)) in x.iter_mut().zip(self.bounds.iter()) {
            if *xi < lo {
                *xi = lo;
            }
            if *xi > hi {
                *xi = hi;
            }
        }
    }
    pub fn width(&self, i: usize) -> f64 {
        self.bounds[i].1 - self.bounds[i].0
    }
}

pub fn random_search<F>(bbox: &Bbox, n_evals: usize, seed: u64, mut f: F) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best_x = bbox.sample(&mut rng);
    let mut best_v = f(&best_x);
    let mut trace = Vec::with_capacity(n_evals);
    trace.push(best_v);
    for _ in 1..n_evals {
        let x = bbox.sample(&mut rng);
        let v = f(&x);
        if v < best_v {
            best_v = v;
            best_x = x;
        }
        trace.push(best_v);
    }
    BboSolution {
        x: best_x,
        value: best_v,
        trace,
        n_evals,
    }
}

#[derive(Clone, Debug)]
pub struct PsoConfig {
    pub n_particles: usize,
    pub n_iters: usize,
    pub w: f64,
    pub c1: f64,
    pub c2: f64,
}

impl Default for PsoConfig {
    fn default() -> Self {
        Self {
            n_particles: 30,
            n_iters: 100,
            w: 0.729,
            c1: 1.494,
            c2: 1.494,
        }
    }
}

pub fn pso<F>(bbox: &Bbox, cfg: &PsoConfig, seed: u64, mut f: F) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let n = bbox.dim();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut positions: Vec<Vec<f64>> = (0..cfg.n_particles).map(|_| bbox.sample(&mut rng)).collect();
    let mut velocities: Vec<Vec<f64>> = (0..cfg.n_particles)
        .map(|_| {
            (0..n)
                .map(|i| rng.gen_range(-bbox.width(i) / 4.0..=bbox.width(i) / 4.0))
                .collect()
        })
        .collect();
    let mut pbests = positions.clone();
    let mut pbest_vals: Vec<f64> = positions.iter().map(|p| f(p)).collect();
    let (gbest_i, gbest_v) = argmin_with_value(&pbest_vals).expect("pso");
    let mut gbest = pbests[gbest_i].clone();
    let mut gbest_v = *gbest_v;
    let mut n_evals = cfg.n_particles;
    let mut trace = vec![gbest_v];
    for _ in 0..cfg.n_iters {
        for p_idx in 0..cfg.n_particles {
            for d in 0..n {
                let r1: f64 = rng.gen_range(0.0..1.0);
                let r2: f64 = rng.gen_range(0.0..1.0);
                velocities[p_idx][d] = cfg.w * velocities[p_idx][d]
                    + cfg.c1 * r1 * (pbests[p_idx][d] - positions[p_idx][d])
                    + cfg.c2 * r2 * (gbest[d] - positions[p_idx][d]);
                positions[p_idx][d] += velocities[p_idx][d];
            }
            bbox.clip(&mut positions[p_idx]);
            let v = f(&positions[p_idx]);
            n_evals += 1;
            if v < pbest_vals[p_idx] {
                pbest_vals[p_idx] = v;
                pbests[p_idx] = positions[p_idx].clone();
                if v < gbest_v {
                    gbest_v = v;
                    gbest = positions[p_idx].clone();
                }
            }
        }
        trace.push(gbest_v);
    }
    BboSolution {
        x: gbest,
        value: gbest_v,
        trace,
        n_evals,
    }
}

fn argmin_with_value(v: &[f64]) -> Option<(usize, &f64)> {
    let mut it = v.iter().enumerate();
    let (mut bi, mut bv) = it.next()?;
    for (i, val) in it {
        if val < bv {
            bi = i;
            bv = val;
        }
    }
    Some((bi, bv))
}

#[derive(Clone, Debug)]
pub struct EsConfig {
    pub n_iters: usize,
    pub sigma0_frac: f64,
    pub adapt_window: usize,
}

impl Default for EsConfig {
    fn default() -> Self {
        Self {
            n_iters: 200,
            sigma0_frac: 0.1,
            adapt_window: 10,
        }
    }
}

pub fn one_plus_one_es<F>(bbox: &Bbox, cfg: &EsConfig, seed: u64, mut f: F) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let n = bbox.dim();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = bbox.sample(&mut rng);
    let mut best_v = f(&x);
    let mut trace = vec![best_v];
    let mut sigmas: Vec<f64> = (0..n).map(|i| bbox.width(i) * cfg.sigma0_frac).collect();
    let mut window_successes = 0usize;
    let mut n_evals = 1usize;
    for k in 0..cfg.n_iters {
        let mut candidate = x.clone();
        for d in 0..n {
            let normal = Normal::new(0.0, sigmas[d]).unwrap();
            candidate[d] += normal.sample(&mut rng);
        }
        bbox.clip(&mut candidate);
        let v = f(&candidate);
        n_evals += 1;
        if v < best_v {
            best_v = v;
            x = candidate;
            window_successes += 1;
        }
        trace.push(best_v);
        if (k + 1) % cfg.adapt_window == 0 {
            let success_rate = window_successes as f64 / cfg.adapt_window as f64;
            let scale = if success_rate > 0.2 { 1.22 } else { 1.0 / 1.22 };
            for s in sigmas.iter_mut() {
                *s *= scale;
            }
            window_successes = 0;
        }
    }
    BboSolution {
        x,
        value: best_v,
        trace,
        n_evals,
    }
}
