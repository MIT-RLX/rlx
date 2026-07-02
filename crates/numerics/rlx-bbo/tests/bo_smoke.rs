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

//! End-to-end smoke tests for the BO module on classic global-opt benchmarks.

use rlx_bbo::{Acquisition, Bbox, BoConfig, Kernel, bo};

fn branin(x: &[f64]) -> f64 {
    // Standard form, global minima ≈ 0.397887 at three locations.
    let a = 1.0;
    let b = 5.1 / (4.0 * std::f64::consts::PI.powi(2));
    let c = 5.0 / std::f64::consts::PI;
    let r = 6.0;
    let s = 10.0;
    let t = 1.0 / (8.0 * std::f64::consts::PI);
    let (x1, x2) = (x[0], x[1]);
    a * (x2 - b * x1 * x1 + c * x1 - r).powi(2) + s * (1.0 - t) * x1.cos() + s
}

fn rosenbrock(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..(x.len() - 1) {
        let a = x[i + 1] - x[i] * x[i];
        let b = 1.0 - x[i];
        acc += 100.0 * a * a + b * b;
    }
    acc
}

#[test]
fn bo_branin_within_5_pct_of_optimum() {
    let bbox = Bbox::new(vec![(-5.0, 10.0), (0.0, 15.0)]);
    let cfg = BoConfig {
        n_init: 8,
        n_iters: 60,
        kernel: Kernel::Matern52 {
            length_scale: 2.0,
            variance: 1.0,
        },
        noise: 1e-3,
        acquisition: Acquisition::Ei { xi: 0.01 },
        n_candidates: 1024,
    };
    let sol = bo(&bbox, &cfg, 7, branin);
    let opt = 0.397_887;
    let gap = sol.value - opt;
    assert!(
        gap < 0.3,
        "Branin BO did not converge: best = {} (gap = {gap})",
        sol.value
    );
}

#[test]
fn bo_rosenbrock_2d_descends() {
    let bbox = Bbox::new(vec![(-2.0, 2.0), (-2.0, 2.0)]);
    let cfg = BoConfig {
        n_init: 10,
        n_iters: 80,
        kernel: Kernel::Matern52 {
            length_scale: 0.5,
            variance: 1.0,
        },
        noise: 1e-3,
        acquisition: Acquisition::Ei { xi: 0.0 },
        n_candidates: 2048,
    };
    let sol = bo(&bbox, &cfg, 17, rosenbrock);
    // Trace must descend (non-increasing) and finish well below the median
    // of a uniform random search budget (Rosenbrock is hard but a few
    // dozen samples should land well under 50).
    for w in sol.trace.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-12,
            "trace must be monotone-non-increasing"
        );
    }
    assert!(sol.value < 50.0, "BO Rosenbrock final = {}", sol.value);
}
