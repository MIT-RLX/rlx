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
//! Twin-critic trust region (prescreen vs expensive sign-off).

use rand::SeedableRng;

use crate::q_guidance::{eta_eff_twin, finite_diff_grad, trust_region_q_step};
use crate::{Bbox, BboSolution, QSteerConfig};

/// Q-steered search with twin objectives: `cheap` (e.g. prescreen) and `expensive` (e.g. ngspice).
///
/// Optimization uses `expensive` for the incumbent; `eta_eff` uses disagreement at `x_best`
/// (paper Eq. 13). Optional `cheap` finite-difference gradient saves expensive evals when
/// `prescreen_grad` is true.
pub fn q_steered_search_twin<Fc, Fe>(
    bbox: &Bbox,
    x_ref: &[f64],
    n_evals: usize,
    seed: u64,
    cfg: &QSteerConfig,
    prescreen_grad: bool,
    mut cheap: Fc,
    mut expensive: Fe,
) -> BboSolution
where
    Fc: FnMut(&[f64]) -> f64,
    Fe: FnMut(&[f64]) -> f64,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut x_best = x_ref.to_vec();
    bbox.clip(&mut x_best);
    let mut best_v = expensive(&x_best);
    let mut trace = vec![best_v];
    let mut n_evals_used = 1usize;

    if n_evals <= 1 {
        return BboSolution {
            x: x_best,
            value: best_v,
            trace,
            n_evals: n_evals_used,
        };
    }

    let remaining = n_evals - 1;
    let n_explore = ((remaining as f64) * cfg.explore_frac).floor() as usize;
    let n_steer = remaining.saturating_sub(n_explore);
    let beta = cfg.twin_beta.max(1e-6);

    for _ in 0..n_steer {
        if n_evals_used >= n_evals {
            break;
        }

        let grad = if prescreen_grad {
            finite_diff_grad(&mut cheap, &x_best, bbox, cfg.fd_eps_frac)
        } else {
            finite_diff_grad(&mut expensive, &x_best, bbox, cfg.fd_eps_frac)
        };
        if !prescreen_grad {
            n_evals_used += x_best.len().saturating_mul(2);
        } else {
            n_evals_used += x_best.len().saturating_mul(2);
        }

        let q_cheap = cheap(&x_best);
        let q_exp = expensive(&x_best);
        n_evals_used += 1;
        let mean_delta = (q_cheap - q_exp).abs();
        let eta = eta_eff_twin(q_cheap, q_exp, cfg.eta, beta, mean_delta, 1e-6);

        if n_evals_used >= n_evals {
            break;
        }
        let x_step = trust_region_q_step(
            &x_best,
            &grad,
            bbox,
            eta,
            cfg.eta_scale_width,
            cfg.kappa,
        );
        let v = expensive(&x_step);
        n_evals_used += 1;
        if v < best_v {
            best_v = v;
            x_best = x_step;
        }
        trace.push(best_v);
    }

    for _ in 0..n_explore {
        if n_evals_used >= n_evals {
            break;
        }
        let x = bbox.sample(&mut rng);
        let v = expensive(&x);
        n_evals_used += 1;
        if v < best_v {
            best_v = v;
            x_best = x;
        }
        trace.push(best_v);
    }

    BboSolution {
        x: x_best,
        value: best_v,
        trace,
        n_evals: n_evals_used.min(n_evals),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twin_search_improves_on_separable() {
        let b = Bbox::new(vec![(-1.0, 1.0); 2]);
        let x0 = vec![0.8, 0.7];
        let cfg = QSteerConfig {
            twin_beta: 0.5,
            explore_frac: 0.1,
            ..QSteerConfig::default()
        };
        let sol = q_steered_search_twin(
            &b,
            &x0,
            12,
            1,
            &cfg,
            true,
            |x| x.iter().map(|v| v * v).sum(),
            |x| x.iter().map(|v| (v - 0.1).powi(2)).sum(),
        );
        assert!(sol.value < 0.5);
    }
}
