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
//! Trust-region Q-guidance and Q-guided beam search (FMQ / QGBS).
//!
//! Reference: Ziakas et al., *Aligning Flow Map Policies with Optimal Q-Guidance*,
//! arXiv:2605.12416 — adapted for black-box **minimization** of `f(x)` with optional
//! offline reference `x_ref`.

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::{BboSolution, Bbox};

/// Stability constant for normalized gradient steps (paper κ₁).
pub const DEFAULT_KAPPA: f64 = 1e-8;

#[derive(Clone, Debug)]
pub struct QSteerConfig {
    /// Trust-region radius: multiplied by per-dimension bbox width when [`Self::eta_scale_width`] is true.
    pub eta: f64,
    pub eta_scale_width: bool,
    pub kappa: f64,
    /// Centered finite-difference step as a fraction of each dimension width.
    pub fd_eps_frac: f64,
    /// Fraction of the evaluation budget spent on uniform exploration (remainder on trust steps).
    pub explore_frac: f64,
    /// Twin-critic shrink factor β for [`eta_eff_twin`]; 0 disables adaptive η.
    pub twin_beta: f64,
}

impl Default for QSteerConfig {
    fn default() -> Self {
        Self {
            eta: 0.15,
            eta_scale_width: true,
            kappa: DEFAULT_KAPPA,
            fd_eps_frac: 1e-4,
            explore_frac: 0.25,
            twin_beta: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QgbsConfig {
    /// Beam width M (particles kept each round).
    pub beam: usize,
    /// Branches B per parent (renoise + optional steer).
    pub branches: usize,
    /// Renoise rounds K.
    pub rounds: usize,
    /// SNR for renoising interpolant t′ = SNR / (1 + SNR).
    pub renoise_snr: f64,
    pub eta: f64,
    pub eta_scale_width: bool,
    pub kappa: f64,
    pub fd_eps_frac: f64,
    /// Apply one trust-region step after each renoised sample.
    pub steer_after_renoise: bool,
}

impl Default for QgbsConfig {
    fn default() -> Self {
        Self {
            beam: 4,
            branches: 4,
            rounds: 1,
            renoise_snr: 2.0,
            eta: 0.12,
            eta_scale_width: true,
            kappa: DEFAULT_KAPPA,
            fd_eps_frac: 1e-4,
            steer_after_renoise: true,
        }
    }
}

impl QgbsConfig {
    /// Function evaluations per run: `beam * (1 + rounds * branches * (1 + steer))`.
    pub fn n_evals_upper_bound(&self) -> usize {
        let steer = if self.steer_after_renoise { 1 } else { 0 };
        self.beam * (1 + self.rounds * self.branches * (1 + steer))
    }
}

fn eta_dim(bbox: &Bbox, cfg_eta: f64, scale_width: bool, d: usize) -> f64 {
    if scale_width {
        cfg_eta * bbox.width(d)
    } else {
        cfg_eta
    }
}

/// One trust-region step toward lower `f` (minimization): `x_ref − η · ∇f / (‖∇f‖ + κ)`.
pub fn trust_region_q_step(
    x_ref: &[f64],
    grad: &[f64],
    bbox: &Bbox,
    eta: f64,
    eta_scale_width: bool,
    kappa: f64,
) -> Vec<f64> {
    let n = x_ref.len();
    assert_eq!(n, grad.len());
    let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt() + kappa;
    let mut x: Vec<f64> = x_ref.to_vec();
    for d in 0..n {
        let step = eta_dim(bbox, eta, eta_scale_width, d);
        x[d] -= step * grad[d] / gnorm;
    }
    bbox.clip(&mut x);
    x
}

/// Centered finite-difference gradient of `f` at `x`.
pub fn finite_diff_grad<F>(mut f: F, x: &[f64], bbox: &Bbox, eps_frac: f64) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    for d in 0..n {
        let h = (bbox.width(d) * eps_frac).max(1e-12);
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[d] = (x[d] + h).min(bbox.bounds[d].1);
        xm[d] = (x[d] - h).max(bbox.bounds[d].0);
        let fp = f(&xp);
        let fm = f(&xm);
        grad[d] = (fp - fm) / (xp[d] - xm[d]).max(1e-30);
    }
    grad
}

/// Adaptive trust radius from twin critic disagreement (paper eq. 13, batch-normalized δ).
pub fn eta_eff_twin(
    q1: f64,
    q2: f64,
    eta_base: f64,
    beta: f64,
    batch_mean_delta: f64,
    kappa2: f64,
) -> f64 {
    let delta = (q1 - q2).abs() / std::f64::consts::SQRT_2;
    let denom = batch_mean_delta + kappa2;
    let tilde = if denom > 0.0 { delta / denom } else { 0.0 };
    let eff = 1.0 / (1.0 + beta * tilde);
    (eta_base * eff).clamp(1e-6, eta_base)
}

fn renoise_sample(rng: &mut StdRng, x1: &[f64], bbox: &Bbox, snr: f64) -> Vec<f64> {
    let t_prime = snr / (1.0 + snr);
    let eps = bbox.sample(rng);
    let mut out = Vec::with_capacity(x1.len());
    for (a, e) in x1.iter().zip(eps.iter()) {
        out.push(t_prime * a + (1.0 - t_prime) * e);
    }
    bbox.clip(&mut out);
    out
}

/// Q-steered search with a caller-supplied gradient (e.g. rlx AD); skips finite differences.
pub fn q_steered_search_with_grad<F, G>(
    bbox: &Bbox,
    x_ref: &[f64],
    n_evals: usize,
    seed: u64,
    cfg: &QSteerConfig,
    mut f: F,
    mut grad_fn: G,
) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_best = x_ref.to_vec();
    bbox.clip(&mut x_best);
    let mut best_v = f(&x_best);
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

    for _ in 0..n_steer {
        if n_evals_used >= n_evals {
            break;
        }
        let grad = grad_fn(&x_best);
        let x_step = trust_region_q_step(
            &x_best,
            &grad,
            bbox,
            cfg.eta,
            cfg.eta_scale_width,
            cfg.kappa,
        );
        let v = f(&x_step);
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
        let v = f(&x);
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

/// Q-steered search: anchor at `x_ref`, trust-region gradient steps + random exploration.
pub fn q_steered_search<F>(
    bbox: &Bbox,
    x_ref: &[f64],
    n_evals: usize,
    seed: u64,
    cfg: &QSteerConfig,
    mut f: F,
) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x_best = x_ref.to_vec();
    bbox.clip(&mut x_best);
    let mut best_v = f(&x_best);
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

    for _ in 0..n_steer {
        if n_evals_used >= n_evals {
            break;
        }
        let grad = finite_diff_grad(&mut f, &x_best, bbox, cfg.fd_eps_frac);
        n_evals_used += x_best.len().saturating_mul(2);

        let eta = if cfg.twin_beta > 0.0 && n_evals_used < n_evals {
            let x_probe = trust_region_q_step(
                &x_best,
                &grad,
                bbox,
                cfg.eta,
                cfg.eta_scale_width,
                cfg.kappa,
            );
            let v_probe = f(&x_probe);
            n_evals_used += 1;
            let delta = (best_v - v_probe).abs();
            eta_eff_twin(best_v, v_probe, cfg.eta, cfg.twin_beta, delta, 1e-6)
        } else {
            cfg.eta
        };

        if n_evals_used >= n_evals {
            break;
        }
        let x_step = trust_region_q_step(&x_best, &grad, bbox, eta, cfg.eta_scale_width, cfg.kappa);
        let v = f(&x_step);
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
        let v = f(&x);
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

/// Q-guided beam search (QGBS): renoise → optional trust steer → keep top beam by `f`.
pub fn q_guided_beam_search<F>(
    bbox: &Bbox,
    x_ref: &[f64],
    cfg: &QgbsConfig,
    seed: u64,
    mut f: F,
) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    let mut beam: Vec<(Vec<f64>, f64)> = {
        let mut x = x_ref.to_vec();
        bbox.clip(&mut x);
        let v = f(&x);
        vec![(x, v)]
    };
    let mut trace = vec![beam[0].1];
    let mut n_evals = 1usize;

    for _round in 0..cfg.rounds {
        let mut candidates: Vec<(Vec<f64>, f64)> = Vec::new();
        beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let parents: Vec<_> = beam.iter().take(cfg.beam).cloned().collect();

        for (parent_x, _) in &parents {
            for b in 0..cfg.branches {
                let salt = seed.wrapping_add(n_evals as u64).wrapping_add(b as u64);
                let mut r2 = StdRng::seed_from_u64(salt);
                let mut x = renoise_sample(&mut r2, parent_x, bbox, cfg.renoise_snr);
                n_evals += 1;
                if cfg.steer_after_renoise {
                    let grad = finite_diff_grad(&mut f, &x, bbox, cfg.fd_eps_frac);
                    n_evals += x.len() * 2;
                    x = trust_region_q_step(
                        &x,
                        &grad,
                        bbox,
                        cfg.eta,
                        cfg.eta_scale_width,
                        cfg.kappa,
                    );
                }
                let v = f(&x);
                n_evals += 1;
                candidates.push((x, v));
                trace.push(
                    candidates
                        .iter()
                        .map(|(_, v)| *v)
                        .fold(f64::INFINITY, f64::min),
                );
            }
        }

        beam = parents;
        beam.extend(candidates);
        beam.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        beam.truncate(cfg.beam);
    }

    let (best_x, best_v) = beam
        .into_iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    BboSolution {
        x: best_x,
        value: best_v,
        trace,
        n_evals,
    }
}

/// Run random search, Q-steer, or QGBS from a string tag (`"bbo"`, `"qsteer"`, `"qgbs"`).
pub fn search_by_method<F>(
    method: &str,
    bbox: &Bbox,
    x_ref: &[f64],
    n_evals: usize,
    seed: u64,
    f: F,
) -> BboSolution
where
    F: FnMut(&[f64]) -> f64,
{
    match method {
        "qsteer" | "q-steer" | "fmq" => {
            q_steered_search(bbox, x_ref, n_evals, seed, &QSteerConfig::default(), f)
        }
        "qgbs" | "q-gbs" => {
            let cfg = QgbsConfig::default();
            q_guided_beam_search(bbox, x_ref, &cfg, seed, f)
        }
        _ => super::random_search(bbox, n_evals, seed, f),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    fn sphere_grad(x: &[f64]) -> Vec<f64> {
        x.iter().map(|v| 2.0 * v).collect()
    }

    #[test]
    fn trust_step_moves_toward_origin() {
        let b = Bbox::new(vec![(-2.0, 2.0); 2]);
        let x = vec![1.5, -1.0];
        let g = sphere_grad(&x);
        let step = trust_region_q_step(&x, &g, &b, 0.5, true, DEFAULT_KAPPA);
        assert!(sphere(&step) < sphere(&x));
    }

    #[test]
    fn qsteer_beats_random_on_sphere() {
        let b = Bbox::new(vec![(-2.0, 2.0); 3]);
        let x0 = vec![1.8, 1.5, 1.2];
        let rs = crate::random_search(&b, 40, 1, sphere);
        let qs = q_steered_search(
            &b,
            &x0,
            40,
            1,
            &QSteerConfig {
                explore_frac: 0.1,
                ..QSteerConfig::default()
            },
            sphere,
        );
        assert!(
            qs.value < rs.value,
            "qsteer {} vs random {}",
            qs.value,
            rs.value
        );
    }

    #[test]
    fn qgbs_finds_low_sphere() {
        let b = Bbox::new(vec![(-2.0, 2.0); 2]);
        let x0 = vec![1.9, 1.8];
        let sol = q_guided_beam_search(&b, &x0, &QgbsConfig::default(), 99, sphere);
        assert!(sol.value < 0.5, "got {}", sol.value);
    }

    #[test]
    fn finite_diff_matches_sphere_grad_direction() {
        let b = Bbox::new(vec![(-2.0, 2.0); 2]);
        let x = vec![0.5, -0.3];
        let fd = finite_diff_grad(sphere, &x, &b, 1e-3);
        let g = sphere_grad(&x);
        let dot: f64 = fd.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
        assert!(dot > 0.0);
    }

    #[test]
    fn eta_eff_shrinks_with_disagreement() {
        let e1 = eta_eff_twin(1.0, 1.0, 0.2, 2.0, 0.1, 1e-6);
        let e2 = eta_eff_twin(1.0, 5.0, 0.2, 2.0, 0.1, 1e-6);
        assert!(e1 > e2);
    }
}
