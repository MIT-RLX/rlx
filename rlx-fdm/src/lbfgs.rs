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

//! Limited-memory BFGS with backtracking line search (jax_fdm `LBFGSB` subset).

/// L-BFGS optimizer for unconstrained or box-projected steps.
#[derive(Clone, Debug)]
pub struct Lbfgs {
    pub history: usize,
    pub max_line_iter: usize,
    pub c1: f64,
    pub tau: f64,
    pub min_step: f64,
    s_hist: Vec<Vec<f64>>,
    y_hist: Vec<Vec<f64>>,
    rho_hist: Vec<f64>,
}

impl Default for Lbfgs {
    fn default() -> Self {
        Self {
            history: 10,
            max_line_iter: 25,
            c1: 1e-4,
            tau: 0.5,
            min_step: 1e-12,
            s_hist: Vec::new(),
            y_hist: Vec::new(),
            rho_hist: Vec::new(),
        }
    }
}

impl Lbfgs {
    /// Search direction `d` (descent) from L-BFGS two-loop recursion.
    pub fn direction(&self, g: &[f64]) -> Vec<f64> {
        let m = self.s_hist.len();
        if m == 0 {
            return g.iter().map(|&gi| -gi).collect();
        }
        let mut q = g.to_vec();
        let mut alpha = vec![0.0; m];
        for i in (0..m).rev() {
            alpha[i] = self.rho_hist[i] * dot(&self.s_hist[i], &q);
            q = sub_scaled(&q, &self.y_hist[i], alpha[i]);
        }
        let mut r = scale(
            g,
            ys_dot(&self.y_hist[m - 1], &self.s_hist[m - 1]).max(1e-14),
        );
        for i in 0..m {
            let beta = self.rho_hist[i] * dot(&self.y_hist[i], &r);
            r = add_scaled(&r, &self.s_hist[i], alpha[i] - beta);
        }
        r.iter().map(|&x| -x).collect()
    }

    /// Backtracking line search on `f`, starting at `x` along direction `d`.
    pub fn line_search<F>(
        &self,
        x: &[f64],
        d: &[f64],
        f0: f64,
        g0: &[f64],
        mut f: F,
    ) -> (Vec<f64>, f64)
    where
        F: FnMut(&[f64]) -> f64,
    {
        let mut step = 1.0;
        let gdot = dot(g0, d);
        for _ in 0..self.max_line_iter {
            let x_try: Vec<f64> = x
                .iter()
                .zip(d.iter())
                .map(|(&xi, &di)| xi + step * di)
                .collect();
            let f_try = f(&x_try);
            if f_try <= f0 + self.c1 * step * gdot {
                return (x_try, f_try);
            }
            step *= self.tau;
            if step < self.min_step {
                break;
            }
        }
        let x_try: Vec<f64> = x
            .iter()
            .zip(d.iter())
            .map(|(&xi, &di)| xi + step * di)
            .collect();
        let loss = f(&x_try);
        (x_try, loss)
    }

    /// Store `(s, y)` pair after a successful step.
    pub fn update(&mut self, x_prev: &[f64], g_prev: &[f64], x_new: &[f64], g_new: &[f64]) {
        let s: Vec<f64> = x_new
            .iter()
            .zip(x_prev.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let y: Vec<f64> = g_new
            .iter()
            .zip(g_prev.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let ys = dot(&y, &s);
        if ys <= 1e-14 {
            return;
        }
        let rho = 1.0 / ys;
        if self.s_hist.len() >= self.history {
            self.s_hist.remove(0);
            self.y_hist.remove(0);
            self.rho_hist.remove(0);
        }
        self.s_hist.push(s);
        self.y_hist.push(y);
        self.rho_hist.push(rho);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn ys_dot(y: &[f64], s: &[f64]) -> f64 {
    dot(y, s) / dot(s, s).max(1e-14)
}

fn sub_scaled(a: &[f64], b: &[f64], c: f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - c * y).collect()
}

fn add_scaled(a: &[f64], b: &[f64], c: f64) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + c * y).collect()
}

fn scale(a: &[f64], c: f64) -> Vec<f64> {
    a.iter().map(|&x| c * x).collect()
}
