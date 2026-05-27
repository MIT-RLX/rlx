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
//! Linear surrogate critic fit from harness JSON / trajectory logs.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::trajectory::{TrajectoryRecord, load_jsonl};

/// Affine Q(x) ≈ b + w·x trained by ridge regression on logged trajectories.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearSurrogate {
    pub dim: usize,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub topology: String,
}

impl LinearSurrogate {
    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut s = self.bias;
        for (wi, xi) in self.weights.iter().zip(x.iter()) {
            s += wi * xi;
        }
        s
    }

    pub fn grad(&self, _x: &[f64]) -> Vec<f64> {
        self.weights.clone()
    }
}

/// Ridge fit: minimize ‖Xw − y‖² + λ‖w‖² with bias column.
pub fn fit_linear_surrogate(
    records: &[TrajectoryRecord],
    topology: &str,
    lambda: f64,
) -> Option<LinearSurrogate> {
    let rows: Vec<_> = records
        .iter()
        .filter(|r| r.topology == topology && !r.action.is_empty())
        .collect();
    if rows.is_empty() {
        return None;
    }
    let dim = rows[0].action.len();
    if !rows.iter().all(|r| r.action.len() == dim) {
        return None;
    }
    let n = rows.len();
    let mut w = vec![0.0; dim];
    let mut b = rows.iter().map(|r| r.loss).sum::<f64>() / n as f64;

    for _ in 0..64 {
        let mut gw = vec![0.0; dim];
        let mut gb = 0.0;
        for r in &rows {
            let pred = b + w
                .iter()
                .zip(r.action.iter())
                .map(|(a, x)| a * x)
                .sum::<f64>();
            let err = pred - r.loss;
            for d in 0..dim {
                gw[d] += 2.0 * err * r.action[d] / n as f64 + 2.0 * lambda * w[d] / n as f64;
            }
            gb += 2.0 * err / n as f64;
        }
        let step = 0.05;
        for d in 0..dim {
            w[d] -= step * gw[d];
        }
        b -= step * gb;
    }

    Some(LinearSurrogate {
        dim,
        weights: w,
        bias: b,
        topology: topology.to_string(),
    })
}

pub fn fit_from_trajectory_jsonl(
    path: &Path,
    topology: &str,
    lambda: f64,
) -> Option<LinearSurrogate> {
    let recs = load_jsonl(path).ok()?;
    fit_linear_surrogate(&recs, topology, lambda)
}

pub fn save_surrogate(path: &Path, s: &LinearSurrogate) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(s)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

pub fn load_surrogate(path: &Path) -> std::io::Result<LinearSurrogate> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
