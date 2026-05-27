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
// RLX — host-side action selection (one-step flow map or optional QGBS).

use crate::graph::{CompiledFlowMapAgent, CompiledTwinCritic};
use crate::qgbs::{QgbsConfig, qgbs_select_action};
use crate::spec::RlSpec;

/// Inference-time options. Default is a single flow-map step (no search).
#[derive(Debug, Clone, Default)]
pub struct EvalConfig {
    /// When `Some`, run QGBS over the compiled flow-map actor at eval.
    pub qgbs: Option<QgbsConfig>,
    /// Best-of-N over `one_step` samples (Python `actor_type=best-of-n`).
    pub best_of_n: Option<usize>,
}

impl EvalConfig {
    pub fn one_step() -> Self {
        Self::default()
    }

    pub fn with_qgbs(cfg: QgbsConfig) -> Self {
        Self {
            qgbs: Some(cfg),
            best_of_n: None,
        }
    }

    pub fn best_of_n(n: usize) -> Self {
        Self {
            qgbs: None,
            best_of_n: Some(n),
        }
    }
}

/// Select an action for `state` given initial noise `a0`.
///
/// Uses compiled CPU graphs only (`CompiledFlowMapAgent` + optional twin critic for QGBS).
#[cfg(feature = "compile")]
pub fn select_action(
    agent: &mut CompiledFlowMapAgent,
    critic: &mut CompiledTwinCritic,
    spec: &RlSpec,
    state: &[f32],
    a0: &[f32],
    eval: &EvalConfig,
) -> Vec<f32> {
    if let Some(cfg) = &eval.qgbs {
        return qgbs_select_action(agent, critic, spec, state, a0, cfg);
    }
    if let Some(n) = eval.best_of_n {
        return best_of_n_select(agent, critic, spec, state, n);
    }
    agent.one_step(state, a0)
}

#[cfg(feature = "compile")]
fn best_of_n_select(
    agent: &mut CompiledFlowMapAgent,
    critic: &mut CompiledTwinCritic,
    spec: &RlSpec,
    state: &[f32],
    n: usize,
) -> Vec<f32> {
    let ad = spec.action_dim;
    let clip = spec.action_clip;
    let mut seed = 0xB057_u64;
    let mut best_q = f32::NEG_INFINITY;
    let mut best_a = vec![0.0f32; ad];

    for i in 0..n {
        seed = crate::buffer::rand_like(seed.wrapping_add(i as u64));
        let mut noise_seed = seed;
        let eps: Vec<f32> = (0..ad)
            .map(|d| {
                noise_seed = crate::buffer::rand_like(noise_seed.wrapping_add(d as u64));
                box_muller(&mut noise_seed)
            })
            .collect();
        let a1 = crate::guidance::clip_action(&agent.one_step(state, &eps), clip);
        let (q1, q2) = critic.q_values(state, &a1);
        let q = q1.min(q2);
        if q.is_finite() && q > best_q {
            best_q = q;
            best_a = a1;
        }
    }
    best_a
}

/// i.i.d. standard normal noise for the flow-map base distribution \(a_0\).
pub fn sample_noise(action_dim: usize, seed: &mut u64) -> Vec<f32> {
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
