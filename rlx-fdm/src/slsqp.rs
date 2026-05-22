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

//! SLSQP-style constrained optimizer (jax_fdm `SLSQP` / penalty + L-BFGS).

use crate::lbfgs::Lbfgs;

/// Penalty + L-BFGS on box-constrained `x` with nonlinear inequalities `c(x) ≤ 0`.
pub struct Slsqp {
    pub lbfgs: Lbfgs,
    pub penalty_weight: f64,
    pub fd_eps: f64,
}

impl Default for Slsqp {
    fn default() -> Self {
        Self {
            lbfgs: Lbfgs::default(),
            penalty_weight: 50.0,
            fd_eps: 1e-7,
        }
    }
}

impl Slsqp {
    pub fn step<F, G, C>(
        &mut self,
        x: &mut [f64],
        low: &[f64],
        up: &[f64],
        mut objective: F,
        mut gradient: G,
        mut nonlinear: C,
    ) -> (f64, f64)
    where
        F: FnMut(&[f64]) -> f64,
        G: FnMut(&[f64], &mut [f64]),
        C: FnMut(&[f64]) -> Vec<f64>,
    {
        let f0 = augmented(&mut objective, &mut nonlinear, x, self.penalty_weight);
        let mut g = vec![0.0; x.len()];
        gradient(x, &mut g);
        nonlinear_fd_grad(x, &mut g, &mut nonlinear, self.penalty_weight, self.fd_eps);
        let dir = self.lbfgs.direction(&g);
        let eval = |trial: &[f64]| augmented(&mut objective, &mut nonlinear, trial, self.penalty_weight);
        let (x_new, _) = self.lbfgs.line_search(x, &dir, f0, &g, eval);
        x.copy_from_slice(&x_new);
        for (xi, (lo, hi)) in x.iter_mut().zip(low.iter().zip(up.iter())) {
            *xi = xi.clamp(*lo, *hi);
        }
        let f1 = augmented(&mut objective, &mut nonlinear, x, self.penalty_weight);
        let gn: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        (f1, gn)
    }
}

fn augmented<F, C>(objective: &mut F, nonlinear: &mut C, x: &[f64], mu: f64) -> f64
where
    F: FnMut(&[f64]) -> f64,
    C: FnMut(&[f64]) -> Vec<f64>,
{
    let mut f = objective(x);
    for v in nonlinear(x) {
        if v > 0.0 {
            f += mu * v * v;
        }
    }
    f
}

fn nonlinear_fd_grad<C>(x: &[f64], g: &mut [f64], nonlinear: &mut C, mu: f64, eps: f64)
where
    C: FnMut(&[f64]) -> Vec<f64>,
{
    let c0 = nonlinear(x);
    let pen0: f64 = c0.iter().map(|&v| if v > 0.0 { v * v } else { 0.0 }).sum();
    for i in 0..x.len() {
        let mut xp = x.to_vec();
        xp[i] += eps;
        let cp = nonlinear(&xp);
        let penp: f64 = cp.iter().map(|&v| if v > 0.0 { v * v } else { 0.0 }).sum();
        g[i] += mu * (penp - pen0) / eps;
    }
}
