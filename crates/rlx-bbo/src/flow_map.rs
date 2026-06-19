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
//! Simplified flow-map policy: affine one-step map + diagonal CFM training (§3.1–3.2).

use crate::Bbox;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::trajectory::{TrajectoryRecord, diagonal_flow_pairs, load_jsonl};

/// One-step flow map X_{0,1}(a₀) = a₀ + W·a₀ + b (linear MVP analogue).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearFlowMap {
    pub dim: usize,
    pub velocity_weights: Vec<f64>,
    pub velocity_bias: Vec<f64>,
    pub topology: String,
}

impl LinearFlowMap {
    pub fn one_step(&self, noise: &[f64]) -> Vec<f64> {
        noise
            .iter()
            .enumerate()
            .map(|(d, &a0)| {
                a0 + self.velocity_bias.get(d).copied().unwrap_or(0.0)
                    + a0 * self.velocity_weights.get(d).copied().unwrap_or(0.0)
            })
            .collect()
    }

    pub fn train_diagonal(records: &[TrajectoryRecord], topology: &str) -> Option<Self> {
        let pairs = diagonal_flow_pairs(records);
        if pairs.is_empty() {
            let actions: Vec<_> = records
                .iter()
                .filter(|r| r.topology == topology)
                .map(|r| r.action.clone())
                .collect();
            if actions.is_empty() {
                return None;
            }
            let dim = actions[0].len();
            return Some(Self {
                dim,
                velocity_weights: vec![0.0; dim],
                velocity_bias: mean_action(&actions, dim),
                topology: topology.to_string(),
            });
        }
        let dim = pairs[0].0.len();
        let mut vel_sum = vec![0.0; dim];
        let mut count = 0usize;
        for (_, v) in &pairs {
            if v.len() != dim {
                continue;
            }
            for d in 0..dim {
                vel_sum[d] += v[d];
            }
            count += 1;
        }
        if count == 0 {
            return None;
        }
        let velocity_bias: Vec<f64> = vel_sum.iter().map(|s| s / count as f64).collect();
        Some(Self {
            dim,
            velocity_weights: vec![0.0; dim],
            velocity_bias,
            topology: topology.to_string(),
        })
    }
}

fn mean_action(actions: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut s = vec![0.0; dim];
    let n = actions.len().max(1) as f64;
    for a in actions {
        for d in 0..dim.min(a.len()) {
            s[d] += a[d];
        }
    }
    s.iter().map(|x| x / n).collect()
}

/// Offline train from JSONL trajectories; returns flow map + training MSE.
pub fn train_from_jsonl(
    path: &Path,
    topology: &str,
) -> std::io::Result<Option<(LinearFlowMap, f64)>> {
    let recs = load_jsonl(path)?;
    let fm = LinearFlowMap::train_diagonal(&recs, topology);
    let Some(fm) = fm else {
        return Ok(None);
    };
    let pairs = diagonal_flow_pairs(&recs);
    let mse = if pairs.is_empty() {
        0.0
    } else {
        let mut err = 0.0;
        let mut n = 0usize;
        for (a1, v_star) in pairs {
            if let Some(a0) = recs
                .iter()
                .find(|r| r.action == a1)
                .and_then(|r| r.noise.clone())
            {
                let pred = fm.one_step(&a0);
                for d in 0..v_star.len().min(pred.len()).min(a0.len()) {
                    let v_pred = pred[d] - a0[d];
                    err += (v_pred - v_star[d]).powi(2);
                    n += 1;
                }
            }
        }
        if n > 0 { err / n as f64 } else { 0.0 }
    };
    Ok(Some((fm, mse)))
}

pub fn save_flow_map(path: &Path, fm: &LinearFlowMap) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(fm)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

pub fn load_flow_map(path: &Path) -> std::io::Result<LinearFlowMap> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// FMQ-style online step using surrogate grad: x ← x − η ∇Q / ‖∇Q‖.
pub fn fmq_surrogate_step(
    x: &[f64],
    x_ref: &[f64],
    grad_q: &[f64],
    bbox: &Bbox,
    eta: f64,
    kappa: f64,
) -> Vec<f64> {
    let _ = x_ref;
    crate::trust_region_q_step(x, grad_q, bbox, eta, true, kappa)
}
