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
//! Separable CMA-ES (diagonal covariance) — Hansen-style sep-CMA-ES.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::{BboSolution, Bbox};

#[derive(Clone, Debug)]
pub struct CmaesConfig {
    /// Initial step size as a fraction of mean bbox width per dimension.
    pub sigma0_frac: f64,
    /// Maximum objective evaluations.
    pub max_evals: usize,
}

impl Default for CmaesConfig {
    fn default() -> Self {
        Self {
            sigma0_frac: 0.2,
            max_evals: 200,
        }
    }
}

fn chi_n(n: usize) -> f64 {
    let n = n as f64;
    n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
}

/// Diagonal CMA-ES. Minimizes `f`.
pub fn cmaes<F>(bbox: &Bbox, cfg: &CmaesConfig, seed: u64, mut f: F) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let n = bbox.dim();
    assert!(n >= 1, "cmaes: empty bbox");

    let mut rng = StdRng::seed_from_u64(seed);
    let lambda = (4.0 + 3.0 * (n as f64).ln()).floor() as usize;
    let lambda = lambda.max(4);
    let mu = lambda / 2;
    let mu = mu.max(1);

    let mut weights: Vec<f64> = (0..mu)
        .map(|i| ((mu + 1) as f64).ln() - ((i + 1) as f64).ln())
        .collect();
    let wsum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= wsum;
    }
    let mueff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();
    let cc = (4.0 + mueff / n as f64) / (n as f64 + 4.0);
    let cs = (mueff + 2.0) / (n as f64 + mueff + 5.0);
    let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mueff);
    let cmu = (1.0 - c1)
        .min(1.0 - c1 + 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n as f64 + 2.0).powi(2) + mueff));
    let damp = 1.0 + 2.0 * (0.0f64).max(((mueff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0) + cs;

    let mut mean: Vec<f64> = bbox
        .bounds
        .iter()
        .map(|&(lo, hi)| 0.5 * (lo + hi))
        .collect();
    let mean_width: f64 =
        bbox.bounds.iter().map(|&(lo, hi)| hi - lo).sum::<f64>() / n as f64;
    let mut sigma = mean_width * cfg.sigma0_frac;
    let mut diag: Vec<f64> = vec![1.0; n];
    let mut ps = vec![0.0; n];
    let mut pc = vec![0.0; n];

    let mut best_x = mean.clone();
    let mut best_v = f(&best_x);
    let mut trace = vec![best_v];
    let mut n_evals = 1usize;

    while n_evals < cfg.max_evals {
        let mut pop: Vec<(Vec<f64>, f64)> = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            if n_evals >= cfg.max_evals {
                break;
            }
            let mut x = vec![0.0; n];
            for d in 0..n {
                let normal = Normal::new(0.0, 1.0).unwrap();
                let z = normal.sample(&mut rng);
                x[d] = mean[d] + sigma * z * diag[d].sqrt();
            }
            bbox.clip(&mut x);
            let v = f(&x);
            n_evals += 1;
            pop.push((x, v));
            if v < best_v {
                best_v = v;
                best_x = pop.last().unwrap().0.clone();
            }
            trace.push(best_v);
        }
        if pop.len() < mu {
            break;
        }
        pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let old_mean = mean.clone();
        mean = vec![0.0; n];
        for (k, (xk, _)) in pop.iter().take(mu).enumerate() {
            for d in 0..n {
                mean[d] += weights[k] * xk[d];
            }
        }

        let inv_sigma = 1.0 / sigma.max(1e-30);
        for d in 0..n {
            ps[d] = (1.0 - cs) * ps[d]
                + (cs * (2.0 - cs) * mueff).sqrt() * (mean[d] - old_mean[d]) * inv_sigma;
        }
        let ps_norm: f64 = ps.iter().map(|p| p * p).sum::<f64>().sqrt();
        sigma *= ((cs / damp) * (ps_norm / chi_n(n) - 1.0)).exp();

        for d in 0..n {
            pc[d] = (1.0 - cc) * pc[d]
                + (cc * (2.0 - cc) * mueff).sqrt() * (mean[d] - old_mean[d]) * inv_sigma;
        }
        for d in 0..n {
            let mut v = 0.0;
            for (k, (xk, _)) in pop.iter().take(mu).enumerate() {
                let diff = (xk[d] - mean[d]) * inv_sigma;
                v += weights[k] * diff * diff;
            }
            diag[d] = (1.0 - c1 - cmu) * diag[d] + c1 * pc[d] * pc[d] + cmu * v;
            diag[d] = diag[d].max(1e-12);
        }

        sigma = sigma.clamp(mean_width * 1e-8, mean_width * 2.0);
    }

    BboSolution {
        x: best_x,
        value: best_v,
        trace,
        n_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bbox;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    #[test]
    fn cmaes_finds_sphere_origin() {
        let b = Bbox::new(vec![(-3.0, 3.0); 4]);
        let cfg = CmaesConfig {
            max_evals: 500,
            sigma0_frac: 0.3,
            ..CmaesConfig::default()
        };
        let sol = cmaes(&b, &cfg, 99, sphere);
        assert!(sol.value < 0.1, "got {}", sol.value);
    }

    #[test]
    fn trace_monotone() {
        let b = Bbox::new(vec![(-2.0, 2.0); 2]);
        let sol = cmaes(&b, &CmaesConfig::default(), 1, sphere);
        for w in sol.trace.windows(2) {
            assert!(w[1] <= w[0] + 1e-9);
        }
    }
}
