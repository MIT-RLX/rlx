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

pub mod acquisition;
pub mod bo;
mod cmaes;
mod flow_map;
pub mod gp;
mod gradcheck;
mod gradient_descent;
mod graph_opt;
mod q_guidance;
pub mod sampling;
mod surrogate;
pub mod tpe;
mod trajectory;
mod twin;

pub use bo::{Acquisition, BoConfig, bo};
pub use cmaes::{CmaesConfig, cmaes};
pub use flow_map::{
    LinearFlowMap, fmq_surrogate_step, load_flow_map, save_flow_map, train_from_jsonl,
};
pub use gp::{GpPosterior, Kernel, cholesky};
pub use gradcheck::gradcheck_graph;
pub use gradient_descent::{AdamOptConfig, AdamOptResult, adam_opt_1d, adam_opt_nd};
pub use graph_opt::{
    GraphOptConfig, GraphOptError, GraphOptResult, GraphOptSpec, adam_opt_graph, find_param_node,
    find_param_nodes,
};
pub use q_guidance::{
    DEFAULT_KAPPA, QSteerConfig, QgbsConfig, eta_eff_twin, finite_diff_grad, q_guided_beam_search,
    q_steered_search, q_steered_search_with_grad, search_by_method, trust_region_q_step,
};
pub use surrogate::{
    LinearSurrogate, fit_from_trajectory_jsonl, fit_linear_surrogate, load_surrogate,
    save_surrogate,
};
pub use trajectory::{TrajectoryRecord, append_jsonl, diagonal_flow_pairs, load_jsonl};
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
    /// Optional early-stop: halt once the global best has not improved for this
    /// many consecutive iterations. `None` (the default) always runs the full
    /// `n_iters`. Useful when each evaluation is expensive (e.g. a full ADC
    /// modulator + FFT per call) and the swarm has visibly plateaued.
    pub patience: Option<usize>,
}

impl Default for PsoConfig {
    fn default() -> Self {
        Self {
            n_particles: 30,
            n_iters: 100,
            w: 0.729,
            c1: 1.494,
            c2: 1.494,
            patience: None,
        }
    }
}

pub fn pso<F>(bbox: &Bbox, cfg: &PsoConfig, seed: u64, mut f: F) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let n = bbox.dim();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut positions: Vec<Vec<f64>> = (0..cfg.n_particles)
        .map(|_| bbox.sample(&mut rng))
        .collect();
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
    let mut stall = 0usize;
    for _ in 0..cfg.n_iters {
        let gbest_before = gbest_v;
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
        // Opt-in early stop once the swarm stops improving the global best.
        if let Some(patience) = cfg.patience {
            if gbest_v < gbest_before {
                stall = 0;
            } else {
                stall += 1;
                if stall >= patience {
                    break;
                }
            }
        }
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

#[cfg(test)]
mod core_optimizer_tests {
    //! Convergence and invariant tests for the core black-box optimizers
    //! (`random_search`, `pso`, `one_plus_one_es`) on standard benchmarks.
    use super::*;

    /// Sphere: convex, minimum 0 at the origin (inside the symmetric box).
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    /// Rosenbrock (2-D): a curved, narrow valley — a harder test.
    fn rosenbrock(x: &[f64]) -> f64 {
        let (a, b) = (1.0, 100.0);
        (a - x[0]).powi(2) + b * (x[1] - x[0] * x[0]).powi(2)
    }

    fn box3() -> Bbox {
        Bbox::new(vec![(-5.0, 5.0); 3])
    }

    #[test]
    fn all_optimizers_minimize_the_sphere() {
        let rs = random_search(&box3(), 4000, 1, sphere);
        let pso_sol = pso(&box3(), &PsoConfig::default(), 1, sphere);
        let es = one_plus_one_es(&box3(), &EsConfig::default(), 1, sphere);
        assert!(rs.value < 3.0, "random_search {}", rs.value);
        assert!(pso_sol.value < 1e-2, "pso {}", pso_sol.value);
        assert!(es.value < 1e-1, "es {}", es.value);
        // The guided searches beat the blind one on a convex function.
        assert!(pso_sol.value < rs.value && es.value < rs.value);
    }

    #[test]
    fn solutions_stay_within_bounds() {
        let bbox = box3();
        for sol in [
            random_search(&bbox, 500, 2, sphere),
            pso(&bbox, &PsoConfig::default(), 2, sphere),
            one_plus_one_es(&bbox, &EsConfig::default(), 2, sphere),
        ] {
            for (&xi, &(lo, hi)) in sol.x.iter().zip(&bbox.bounds) {
                assert!(
                    xi >= lo - 1e-9 && xi <= hi + 1e-9,
                    "{xi} out of [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn trace_is_monotone_best_so_far() {
        for sol in [
            random_search(&box3(), 500, 3, sphere),
            pso(&box3(), &PsoConfig::default(), 3, sphere),
            one_plus_one_es(&box3(), &EsConfig::default(), 3, sphere),
        ] {
            assert!(!sol.trace.is_empty());
            assert!(
                sol.trace.windows(2).all(|w| w[1] <= w[0] + 1e-12),
                "trace not monotone"
            );
            assert!((sol.trace.last().unwrap() - sol.value).abs() < 1e-12);
        }
    }

    #[test]
    fn reported_value_matches_the_objective_at_the_solution() {
        for sol in [
            random_search(&box3(), 500, 4, sphere),
            pso(&box3(), &PsoConfig::default(), 4, sphere),
            one_plus_one_es(&box3(), &EsConfig::default(), 4, sphere),
        ] {
            assert!((sphere(&sol.x) - sol.value).abs() < 1e-12);
        }
    }

    #[test]
    fn pso_solves_the_rosenbrock_valley() {
        // Global minimum is (1, 1) with value 0; a decent optimizer gets close.
        let bbox = Bbox::new(vec![(-2.0, 2.0); 2]);
        let cfg = PsoConfig {
            n_particles: 40,
            n_iters: 300,
            ..PsoConfig::default()
        };
        let sol = pso(&bbox, &cfg, 1, rosenbrock);
        assert!(sol.value < 1e-2, "rosenbrock {}", sol.value);
        assert!(
            (sol.x[0] - 1.0).abs() < 0.1 && (sol.x[1] - 1.0).abs() < 0.1,
            "x = {:?}",
            sol.x
        );
    }

    #[test]
    fn more_evaluations_do_not_hurt() {
        let short = pso(
            &box3(),
            &PsoConfig {
                n_iters: 10,
                ..PsoConfig::default()
            },
            5,
            sphere,
        );
        let long = pso(
            &box3(),
            &PsoConfig {
                n_iters: 150,
                ..PsoConfig::default()
            },
            5,
            sphere,
        );
        assert!(
            long.value <= short.value + 1e-12,
            "long {} vs short {}",
            long.value,
            short.value
        );
    }

    #[test]
    fn early_stop_cuts_evaluations_without_hurting_quality() {
        // A tiny discrete objective plateaus fast: the swarm locks onto the
        // bin minimum, then the global best stops changing bit-for-bit.
        let step = |x: &[f64]| (x[0].round().powi(2) + x[1].round().powi(2)).max(0.0);
        let bbox = Bbox::new(vec![(-4.0, 4.0); 2]);
        let full = pso(
            &bbox,
            &PsoConfig {
                n_iters: 200,
                ..PsoConfig::default()
            },
            7,
            step,
        );
        let stopped = pso(
            &bbox,
            &PsoConfig {
                n_iters: 200,
                patience: Some(8),
                ..PsoConfig::default()
            },
            7,
            step,
        );
        // Same optimum reached, but with a shorter run and fewer evaluations.
        assert_eq!(stopped.value, full.value);
        assert!(
            stopped.trace.len() < full.trace.len(),
            "did not stop early: {}",
            stopped.trace.len()
        );
        assert!(stopped.n_evals < full.n_evals);
        // The tail it skipped was flat anyway (no improvement was left on the table).
        assert_eq!(*full.trace.last().unwrap(), *stopped.trace.last().unwrap());
    }

    #[test]
    fn generous_patience_is_identical_to_no_early_stop() {
        // Patience larger than n_iters can never trigger ⇒ bit-identical result.
        let a = pso(&box3(), &PsoConfig::default(), 9, sphere);
        let b = pso(
            &box3(),
            &PsoConfig {
                patience: Some(10_000),
                ..PsoConfig::default()
            },
            9,
            sphere,
        );
        assert_eq!(a.value, b.value);
        assert_eq!(a.trace.len(), b.trace.len());
        assert_eq!(a.n_evals, b.n_evals);
    }
}
