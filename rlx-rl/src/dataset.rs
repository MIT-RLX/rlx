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
// RLX — offline dataset batches.

use crate::buffer::{ReplayBuffer, Transition};
use crate::flow_curriculum::sample_r_t;
use crate::spec::RlSpec;

/// Static offline transitions for flow-matching pretraining.
#[derive(Debug, Clone, Default)]
pub struct OfflineDataset {
    pub transitions: Vec<Transition>,
}

impl OfflineDataset {
    pub fn from_transitions(transitions: Vec<Transition>) -> Self {
        Self { transitions }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn from_replay(buf: &ReplayBuffer) -> Self {
        Self {
            transitions: buf.iter().cloned().collect(),
        }
    }

    /// L_Diag batch (Eq. 3): `r = t`, target velocity `a₁ - a₀`.
    pub fn sample_diag_batch(
        &self,
        spec: &RlSpec,
        indices: &[usize],
        rng: &mut u64,
    ) -> ActorTrainBatch {
        let b = indices.len();
        let sd = spec.state_dim;
        let ad = spec.action_dim;
        let mut state = vec![0.0f32; b * sd];
        let mut a_r = vec![0.0f32; b * ad];
        let mut r = vec![0.0f32; b];
        let mut t = vec![0.0f32; b];
        let mut target_u = vec![0.0f32; b * ad];

        for (bi, &idx) in indices.iter().enumerate() {
            let tr = &self.transitions[idx % self.transitions.len()];
            state[bi * sd..(bi + 1) * sd].copy_from_slice(&tr.state);
            let mut x0 = vec![0.0f32; ad];
            for d in 0..ad {
                x0[d] = box_muller(rng);
            }
            let tc = uniform01(rng);
            r[bi] = tc;
            t[bi] = tc;
            for d in 0..ad {
                a_r[bi * ad + d] = (1.0 - tc) * x0[d] + tc * tr.action[d];
                target_u[bi * ad + d] = tr.action[d] - x0[d];
            }
        }
        ActorTrainBatch {
            state,
            a_r,
            r,
            t,
            target_u,
        }
    }

    /// Pack ESD inputs: states, bridge `a_r`, curriculum `(r,t)`, noise velocity `v_rt`.
    pub fn sample_esd_inputs(
        &self,
        spec: &RlSpec,
        indices: &[usize],
        step: usize,
        rng: &mut u64,
    ) -> EsdInputsBatch {
        let b = indices.len();
        let sd = spec.state_dim;
        let ad = spec.action_dim;
        let mut state = vec![0.0f32; b * sd];
        let mut a_r = vec![0.0f32; b * ad];
        let mut v_rt = vec![0.0f32; b * ad];
        let mut gamma = vec![0.0f32; b];

        let (rs, ts) = sample_r_t(
            b,
            step,
            spec.flow_map_warmup_steps,
            spec.flow_map_anneal_end_step,
            rng,
        );

        for (bi, &idx) in indices.iter().enumerate() {
            let tr = &self.transitions[idx % self.transitions.len()];
            state[bi * sd..(bi + 1) * sd].copy_from_slice(&tr.state);
            let mut x0 = vec![0.0f32; ad];
            for d in 0..ad {
                x0[d] = box_muller(rng);
            }
            let ri = rs[bi];
            for d in 0..ad {
                a_r[bi * ad + d] = (1.0 - ri) * x0[d] + ri * tr.action[d];
                v_rt[bi * ad + d] = tr.action[d] - x0[d];
            }
            gamma[bi] = uniform01(rng);
        }

        EsdInputsBatch {
            state,
            a_r,
            r: rs,
            t: ts,
            v_rt,
            gamma,
        }
    }

    /// Legacy CFM batch (diagonal only).
    pub fn sample_cfm_batch(&self, spec: &RlSpec, indices: &[usize], rng: &mut u64) -> CfmBatch {
        let batch = self.sample_diag_batch(spec, indices, rng);
        CfmBatch {
            state: batch.state,
            a0: vec![],
            a1: vec![],
            a_r: batch.a_r,
            r: batch.r,
            t: batch.t,
            target_u: batch.target_u,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActorTrainBatch {
    pub state: Vec<f32>,
    pub a_r: Vec<f32>,
    pub r: Vec<f32>,
    pub t: Vec<f32>,
    pub target_u: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct EsdInputsBatch {
    pub state: Vec<f32>,
    pub a_r: Vec<f32>,
    pub r: Vec<f32>,
    pub t: Vec<f32>,
    pub v_rt: Vec<f32>,
    pub gamma: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct CfmBatch {
    pub state: Vec<f32>,
    pub a0: Vec<f32>,
    pub a1: Vec<f32>,
    pub a_r: Vec<f32>,
    pub r: Vec<f32>,
    pub t: Vec<f32>,
    pub target_u: Vec<f32>,
}

fn uniform01(seed: &mut u64) -> f32 {
    *seed = crate::buffer::rand_like(*seed);
    (*seed >> 11) as f32 / (1u32 << 21) as f32
}

fn box_muller(seed: &mut u64) -> f32 {
    let u1 = uniform01(seed).max(1e-7);
    let u2 = uniform01(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}
