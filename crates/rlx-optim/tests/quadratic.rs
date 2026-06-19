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

//
// Sanity test: every optimizer must drive a tiny convex quadratic
// `f(x) = 0.5·‖x − x*‖²` to (near) the minimum within a reasonable
// budget. This catches sign / index / iteration-count bugs without
// asserting any specific convergence rate.

use rlx_optim::{
    Adafactor, Adam, AdamW, KronPsgd, Lamb, Lion, Mars, Muon, NAdamW, Optimizer, QHAdamW, RAdam,
    Sgd, Soap, Sophia,
};

fn target() -> Vec<f32> {
    (0..16).map(|i| (i as f32) * 0.1 - 0.8).collect()
}

fn grad(x: &[f32], t: &[f32]) -> Vec<f32> {
    x.iter().zip(t).map(|(xi, ti)| xi - ti).collect()
}

fn err(x: &[f32], t: &[f32]) -> f32 {
    x.iter()
        .zip(t)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn drive_elementwise<O: Optimizer>(mut opt: O, steps: usize) -> f32 {
    let t = target();
    let mut x = vec![0.0f32; t.len()];
    let shape = [t.len()];
    for _ in 0..steps {
        let g = grad(&x, &t);
        opt.step("p", &shape, &mut x, &g);
        opt.end_iteration();
    }
    err(&x, &t)
}

fn drive_matrix<O: Optimizer>(mut opt: O, steps: usize, rows: usize, cols: usize) -> f32 {
    let n = rows * cols;
    let t: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
    let mut x = vec![0.0f32; n];
    let shape = [rows, cols];
    for _ in 0..steps {
        let g = grad(&x, &t);
        opt.step("p", &shape, &mut x, &g);
        opt.end_iteration();
    }
    err(&x, &t)
}

#[test]
fn sgd_converges() {
    let e = drive_elementwise(Sgd::new(0.5), 200);
    assert!(e < 1e-4, "SGD residual {e}");
}

#[test]
fn sgd_momentum_converges() {
    let e = drive_elementwise(Sgd::new(0.1).with_momentum(0.9, true), 200);
    assert!(e < 1e-3, "SGD+Nesterov residual {e}");
}

#[test]
fn adam_converges() {
    let e = drive_elementwise(Adam::new(0.1), 400);
    assert!(e < 1e-3, "Adam residual {e}");
}

#[test]
fn adamw_converges() {
    let e = drive_elementwise(AdamW::new(0.1).with_weight_decay(0.0), 400);
    assert!(e < 1e-3, "AdamW residual {e}");
}

#[test]
fn nadamw_converges() {
    let e = drive_elementwise(NAdamW::new(0.1).with_weight_decay(0.0), 400);
    assert!(e < 1e-3, "NAdamW residual {e}");
}

#[test]
fn radam_converges() {
    let e = drive_elementwise(RAdam::new(0.1), 600);
    assert!(e < 1e-2, "RAdam residual {e}");
}

#[test]
fn qhadamw_converges() {
    let e = drive_elementwise(QHAdamW::new(0.1).with_weight_decay(0.0), 1500);
    assert!(e < 0.05, "QHAdamW residual {e}");
}

#[test]
fn lamb_converges() {
    // LAMB's trust ratio damps progress when ‖w‖ is initially tiny —
    // initialize away from zero so the ratio is well-defined.
    let mut opt = Lamb::new(0.1).with_weight_decay(0.0);
    let t = target();
    let mut x: Vec<f32> = (0..t.len()).map(|i| 0.5 + 0.01 * i as f32).collect();
    let shape = [t.len()];
    for _ in 0..1500 {
        let g = grad(&x, &t);
        opt.step("p", &shape, &mut x, &g);
        opt.end_iteration();
    }
    let e = err(&x, &t);
    assert!(e < 0.15, "LAMB residual {e}");
}

#[test]
fn adafactor_converges() {
    // Adafactor on a 1-D parameter falls back to full EMA. The
    // RMS-of-update clip caps the step at ‖update‖_rms = 1, so on
    // this tiny problem we need many iterations to chip down. Loose
    // bound — Adafactor is designed for the *matrix* case (see the
    // `adafactor_matrix_converges` test).
    let e = drive_elementwise(Adafactor::new().with_lr(0.1), 3000);
    assert!(e < 0.3, "Adafactor residual {e}");
}

#[test]
fn adafactor_matrix_converges() {
    // Adafactor's selling point: factored update on 2-D weights.
    let e = drive_matrix(Adafactor::new().with_lr(0.05), 800, 6, 4);
    assert!(e < 0.5, "Adafactor (matrix) residual {e}");
}

#[test]
fn lion_converges() {
    // Lion's sign-update has a coarse "lattice"; allow a looser bound.
    let e = drive_elementwise(Lion::new(0.01), 400);
    assert!(e < 0.05, "Lion residual {e}");
}

#[test]
fn sophia_converges() {
    let mut opt = Sophia::new(0.1).with_weight_decay(0.0);
    let t = target();
    let mut x = vec![0.0f32; t.len()];
    let shape = [t.len()];
    // Hessian of f = I, so a constant unit Hessian estimate is perfect.
    let h_hat = vec![1.0f32; t.len()];
    opt.update_hessian("p", &h_hat);
    for _ in 0..400 {
        let g = grad(&x, &t);
        opt.step("p", &shape, &mut x, &g);
        opt.end_iteration();
    }
    let e = err(&x, &t);
    assert!(e < 0.1, "Sophia residual {e}");
}

#[test]
fn mars_converges() {
    let e = drive_elementwise(Mars::new(0.1).with_weight_decay(0.0), 600);
    assert!(e < 1e-2, "MARS residual {e}");
}

#[test]
fn muon_matrix_converges() {
    let e = drive_matrix(Muon::new(0.02).with_weight_decay(0.0), 800, 6, 4);
    // Muon's update is a rescaled orthogonalization of the momentum,
    // so the residual on a quadratic stalls before reaching machine zero.
    assert!(e < 0.6, "Muon residual {e}");
}

#[test]
fn soap_matrix_converges() {
    let e = drive_matrix(Soap::new(0.1).with_weight_decay(0.0), 400, 6, 4);
    assert!(e < 0.5, "SOAP residual {e}");
}

#[test]
fn kron_psgd_matrix_runs() {
    // PSGD is most useful on ill-conditioned problems; we only check
    // it strictly *decreases* loss on a well-conditioned quadratic.
    let mut opt = KronPsgd::new(0.05).with_weight_decay(0.0);
    let rows = 6;
    let cols = 4;
    let n = rows * cols;
    let t: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.03).sin()).collect();
    let mut x = vec![0.0f32; n];
    let shape = [rows, cols];
    let g0 = grad(&x, &t);
    let e0 = (g0.iter().map(|v| v * v).sum::<f32>()).sqrt();
    for _ in 0..200 {
        let g = grad(&x, &t);
        opt.step("p", &shape, &mut x, &g);
    }
    let e1 = err(&x, &t);
    assert!(e1 < e0, "Kron-PSGD did not reduce loss: {e0} → {e1}");
}
