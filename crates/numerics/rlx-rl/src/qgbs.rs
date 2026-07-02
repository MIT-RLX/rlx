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
// RLX — Q-Guided Beam Search (host-side inference over compiled flow maps).

use crate::guidance::{clip_action, normalize_grad, q_guided_project};
use crate::spec::RlSpec;

#[cfg(feature = "compile")]
use crate::graph::{CompiledFlowMapAgent, CompiledTwinCritic};

/// QGBS hyperparameters (paper §3.4, Python `qgbs_eval.yaml`).
#[derive(Debug, Clone)]
pub struct QgbsConfig {
    /// K explore-exploit iterations.
    pub k_steps: usize,
    /// M beam width (defaults to [`RlSpec::actor_num_samples`] when building from spec).
    pub beam_width: usize,
    /// B branches per particle.
    pub branches: usize,
    /// ρ SNR: t' = ρ/(1+ρ).
    pub snr: f32,
    /// η trust-region step (Python `qgbs_eta`).
    pub eta: f32,
}

impl QgbsConfig {
    pub fn from_spec(spec: &RlSpec) -> Self {
        Self {
            k_steps: 1,
            beam_width: spec.actor_num_samples,
            branches: 4,
            snr: 1.5,
            eta: spec.qgbs_eta,
        }
    }

    pub fn nfe(&self) -> usize {
        self.beam_width * (1 + self.k_steps * self.branches)
    }
}

impl Default for QgbsConfig {
    fn default() -> Self {
        Self {
            k_steps: 1,
            beam_width: 32,
            branches: 4,
            snr: 1.5,
            eta: 0.3,
        }
    }
}

fn renoise_time(snr: f32) -> f32 {
    snr / (1.0 + snr)
}

/// QGBS (Algorithm 2) aligned with Python `agents/qgbs.py`.
#[cfg(feature = "compile")]
pub fn qgbs_select_action(
    agent: &mut CompiledFlowMapAgent,
    critic: &mut CompiledTwinCritic,
    spec: &RlSpec,
    state: &[f32],
    _a0: &[f32],
    cfg: &QgbsConfig,
) -> Vec<f32> {
    let ad = spec.action_dim;
    let clip = spec.action_clip;
    let m = cfg.beam_width;
    let b_mc = cfg.branches;
    let t_prime = renoise_time(cfg.snr);
    let sigma_tp = 1.0 - t_prime;
    let eta = cfg.eta;
    let norm_g = spec.fmq_normalize_grad;

    let mut seed = 0x5147_4253_u64;
    let mut beams = Vec::with_capacity(m);

    seed = crate::buffer::rand_like(seed);
    let eps0 = sample_noise(ad, &mut seed);
    beams.push(clip_action(&agent.one_step(state, &eps0), clip));

    for _ in 0..m.saturating_sub(1) {
        seed = crate::buffer::rand_like(seed);
        let eps = sample_noise(ad, &mut seed);
        beams.push(clip_action(&agent.one_step(state, &eps), clip));
    }

    for _ in 0..cfg.k_steps {
        let mk = m * b_mc;
        let mut candidates = Vec::with_capacity(mk);
        for a1 in &beams {
            for _ in 0..b_mc {
                seed = crate::buffer::rand_like(seed);
                let eps = sample_noise(ad, &mut seed);
                let x_tp: Vec<f32> = a1
                    .iter()
                    .zip(eps.iter())
                    .map(|(&a, &e)| t_prime * a + sigma_tp * e)
                    .collect();
                let v = agent.velocity(state, &x_tp, t_prime, t_prime);
                let comp = clip_action(
                    &x_tp
                        .iter()
                        .zip(v.iter())
                        .map(|(&x, &vel)| x + sigma_tp * vel)
                        .collect::<Vec<_>>(),
                    clip,
                );
                candidates.push(comp);
            }
        }

        candidates.sort_by(|a, b| {
            let (qa1, qa2) = critic.q_values(state, a);
            let (qb1, qb2) = critic.q_values(state, b);
            let qa = qa1.min(qa2);
            let qb = qb1.min(qb2);
            qb.partial_cmp(&qa).unwrap_or(std::cmp::Ordering::Equal)
        });
        beams = candidates.into_iter().take(m).collect();

        let mut projected = Vec::with_capacity(m);
        for a in &beams {
            let mut grad_q = critic.action_grad(state, a);
            if norm_g {
                grad_q = normalize_grad(&grad_q);
            }
            projected.push(clip_action(&q_guided_project(a, &grad_q, eta), clip));
        }
        beams = projected;
    }

    if cfg.k_steps == 0 && eta > 0.0 {
        let mut projected = Vec::with_capacity(m);
        for a in &beams {
            let mut grad_q = critic.action_grad(state, a);
            if norm_g {
                grad_q = normalize_grad(&grad_q);
            }
            projected.push(clip_action(&q_guided_project(a, &grad_q, eta), clip));
        }
        beams = projected;
    }

    let mut best = beams[0].clone();
    let mut best_q = f32::NEG_INFINITY;
    for a in &beams {
        let (q1, q2) = critic.q_values(state, a);
        let q = q1.min(q2);
        if q.is_finite() && q > best_q {
            best_q = q;
            best = a.clone();
        }
    }
    best
}

fn sample_noise(action_dim: usize, seed: &mut u64) -> Vec<f32> {
    (0..action_dim)
        .map(|d| {
            *seed = crate::buffer::rand_like(seed.wrapping_add(d as u64));
            box_muller(seed)
        })
        .collect()
}

fn box_muller(seed: &mut u64) -> f32 {
    *seed = crate::buffer::rand_like(*seed);
    let u1 = ((*seed >> 11) as f32 / (1u32 << 21) as f32).max(1e-7);
    *seed = crate::buffer::rand_like(*seed);
    let u2 = (*seed >> 11) as f32 / (1u32 << 21) as f32;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}
