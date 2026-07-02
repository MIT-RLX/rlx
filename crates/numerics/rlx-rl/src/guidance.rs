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
// RLX — FMQ trust-region Q projection.

use crate::spec::RlSpec;

/// Per-sample adaptive trust region from twin-critic disagreement (QGBS / legacy).
pub fn eta_effective(spec: &RlSpec, q1: f32, q2: f32, batch_delta_mean: f32) -> f32 {
    let delta = (q1 - q2).abs() / std::f32::consts::SQRT_2;
    let norm = delta / (batch_delta_mean + spec.eta_kappa);
    (1.0 / (1.0 + spec.eta_beta * norm)).clamp(0.0, 1.0) * spec.eta
}

/// FMQ Eq. 13: η_eff = 1 / (1 + β · δ̃) with δ̃ = std(Q) / mean(std(Q)).
pub fn fmq_eta_effective(spec: &RlSpec, q1: f32, q2: f32) -> f32 {
    if !spec.fmq_adaptive_eta {
        return spec.fmq_eta();
    }
    let q_std = (q1 - q2).abs();
    let q_std_rel = q_std / (q_std + spec.eta_kappa);
    (1.0 / (1.0 + spec.fmq_beta * q_std_rel)).clamp(0.0, 1.0)
}

pub fn clip_action(a: &[f32], clip: f32) -> Vec<f32> {
    a.iter().map(|x| x.clamp(-clip, clip)).collect()
}

/// Unit-normalize ∇Q (Python `fmq_normalize_grad`).
pub fn normalize_grad(grad: &[f32]) -> Vec<f32> {
    let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-8);
    grad.iter().map(|g| g / norm).collect()
}

/// \(a^* = a + \eta \nabla_a Q / \|\nabla_a Q\|\) (single sample).
pub fn q_guided_project(action: &[f32], grad_q: &[f32], eta: f32) -> Vec<f32> {
    let mut out = action.to_vec();
    let norm: f32 = grad_q.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-8);
    for (o, g) in out.iter_mut().zip(grad_q.iter()) {
        *o += eta * (*g / norm);
    }
    out
}

/// Batch projection for `batch` rows in row-major layout.
pub fn q_guided_project_batch(
    actions: &[f32],
    grad_q: &[f32],
    etas: &[f32],
    action_dim: usize,
) -> Vec<f32> {
    let batch = etas.len();
    let mut out = vec![0.0f32; batch * action_dim];
    for b in 0..batch {
        let a = &actions[b * action_dim..(b + 1) * action_dim];
        let g = &grad_q[b * action_dim..(b + 1) * action_dim];
        let proj = q_guided_project(a, g, etas[b]);
        out[b * action_dim..(b + 1) * action_dim].copy_from_slice(&proj);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_moves_along_gradient() {
        let a = [0.0f32, 0.0];
        let g = [3.0, 4.0];
        let p = q_guided_project(&a, &g, 0.1);
        assert!((p[0] - 0.06).abs() < 1e-5);
        assert!((p[1] - 0.08).abs() < 1e-5);
    }
}
