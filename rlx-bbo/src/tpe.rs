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

//! Tree-structured Parzen Estimator (Bergstra et al. 2011, Optuna).
//!
//! Splits observed `(x, y)` pairs into a "good" set (best γ-quantile) and
//! a "bad" set (the rest), models each with a kernel density, and
//! suggests new points by maximising `ℓ(x) / g(x)`.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Returned candidate plus its TPE score.
#[derive(Clone, Debug)]
pub struct TpeSuggestion {
    pub x: Vec<f64>,
    pub score: f64,
}

/// One TPE step: given observations `(xs, ys)` (lower-is-better), the
/// design bounds, the `gamma` quantile split, and a candidate budget,
/// return the best candidate by `ℓ(x)/g(x)`.
pub fn tpe_suggest(
    xs: &[Vec<f64>],
    ys: &[f64],
    bounds: &[(f64, f64)],
    gamma: f64,
    n_candidates: usize,
    seed: u64,
) -> TpeSuggestion {
    assert_eq!(xs.len(), ys.len());
    assert!(!xs.is_empty());
    assert!((0.0..1.0).contains(&gamma));
    let mut rng = StdRng::seed_from_u64(seed);

    // Sort by y ascending and split.
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| ys[a].partial_cmp(&ys[b]).unwrap());
    let n_good = ((xs.len() as f64 * gamma).ceil() as usize).max(1);
    let good: Vec<Vec<f64>> = idx[..n_good].iter().map(|&i| xs[i].clone()).collect();
    let bad: Vec<Vec<f64>> = idx[n_good..].iter().map(|&i| xs[i].clone()).collect();
    if bad.is_empty() {
        // Degenerate split: just return a perturbed best.
        let x = perturb_around(&good[0], bounds, &mut rng);
        return TpeSuggestion { x, score: 0.0 };
    }

    // Per-dimension kernel bandwidth — Bergstra's rule: use the gap to
    // the nearest neighbour, floored at 1% of the range.
    let dim = bounds.len();
    let bw_good = bandwidth(&good, bounds);
    let bw_bad = bandwidth(&bad, bounds);

    let mut best_x = vec![0.0; dim];
    let mut best_score = f64::NEG_INFINITY;
    for _ in 0..n_candidates {
        let cand = sample_from_kde(&good, &bw_good, bounds, &mut rng);
        let l = kde_density(&cand, &good, &bw_good);
        let g = kde_density(&cand, &bad, &bw_bad);
        let score = (l + 1e-300).ln() - (g + 1e-300).ln();
        if score > best_score {
            best_score = score;
            best_x = cand;
        }
    }
    TpeSuggestion {
        x: best_x,
        score: best_score,
    }
}

fn bandwidth(pts: &[Vec<f64>], bounds: &[(f64, f64)]) -> Vec<f64> {
    let dim = bounds.len();
    let mut bw = vec![0.0; dim];
    for d in 0..dim {
        let range = (bounds[d].1 - bounds[d].0).max(1e-12);
        let mut min_gap = range;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let g = (pts[i][d] - pts[j][d]).abs();
                if g > 0.0 && g < min_gap {
                    min_gap = g;
                }
            }
        }
        bw[d] = min_gap.max(0.01 * range);
    }
    bw
}

fn sample_from_kde(
    centers: &[Vec<f64>],
    bw: &[f64],
    bounds: &[(f64, f64)],
    rng: &mut StdRng,
) -> Vec<f64> {
    let i = rng.gen_range(0..centers.len());
    let c = &centers[i];
    let mut out = vec![0.0; c.len()];
    for d in 0..c.len() {
        let n = Normal::new(c[d], bw[d]).unwrap();
        let v = n.sample(rng).clamp(bounds[d].0, bounds[d].1);
        out[d] = v;
    }
    out
}

fn kde_density(x: &[f64], centers: &[Vec<f64>], bw: &[f64]) -> f64 {
    let n = centers.len() as f64;
    let mut acc = 0.0;
    for c in centers {
        let mut log_p = 0.0;
        for d in 0..x.len() {
            let z = (x[d] - c[d]) / bw[d];
            log_p += -0.5 * z * z - (bw[d] * (2.0 * std::f64::consts::PI).sqrt()).ln();
        }
        acc += log_p.exp();
    }
    acc / n
}

fn perturb_around(c: &[f64], bounds: &[(f64, f64)], rng: &mut StdRng) -> Vec<f64> {
    let mut out = vec![0.0; c.len()];
    for d in 0..c.len() {
        let range = bounds[d].1 - bounds[d].0;
        let n = Normal::new(c[d], 0.1 * range).unwrap();
        out[d] = n.sample(rng).clamp(bounds[d].0, bounds[d].1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tpe_finds_good_region_on_1d_quadratic() {
        // f(x) = (x − 0.3)² on [0, 1]. Seed with random observations,
        // then TPE should suggest near 0.3.
        let xs: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 / 30.0]).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x[0] - 0.3).powi(2)).collect();
        let s = tpe_suggest(&xs, &ys, &[(0.0, 1.0)], 0.2, 200, 7);
        let xv = s.x[0];
        assert!(
            xv > 0.05 && xv < 0.55,
            "TPE suggest fell outside good region: {xv}"
        );
    }
}
