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
// RLX — FMQ offline / online training (CPU Session + autodiff).

use rlx_runtime::{Device, Session};

use crate::buffer::ReplayBuffer;
use crate::dataset::{ActorTrainBatch, EsdInputsBatch, OfflineDataset};
use crate::distillation::esd_regression_target;
use crate::env::RlEnv;
use crate::flow_curriculum::sample_r_t;
use crate::graph::{
    CompiledFlowMapAgent, CompiledTwinCritic, ParamSlot, WeightStore, build_actor_graphs,
    build_critic_graphs, init_actor_weights, init_critic_weights,
};
use crate::guidance::{clip_action, fmq_eta_effective, normalize_grad};
use crate::policy::{EvalConfig, sample_noise, select_action};
use crate::spec::RlSpec;

/// Flow-map FMQ trainer.
pub struct FmqTrainer {
    pub spec: RlSpec,
    pub actor_weights: WeightStore,
    actor_anchor_weights: WeightStore,
    pub critic_weights: WeightStore,
    target_critic_weights: WeightStore,
    pub agent_infer: CompiledFlowMapAgent,
    agent_anchor: CompiledFlowMapAgent,
    agent_offline: CompiledFlowMapAgent,
    agent_online: CompiledFlowMapAgent,
    critic: CompiledTwinCritic,
    target_critic: CompiledTwinCritic,
    pub replay: ReplayBuffer,
    /// Global training step (Python `network.step`).
    pub train_step: usize,
    rng: u64,
}

impl FmqTrainer {
    pub fn new(spec: RlSpec) -> Self {
        let session = Session::new(Device::Cpu);
        let infer_spec = spec.with_batch(1);
        let train_spec = spec.clone();

        let actor_weights = init_actor_weights(&train_spec, 42);
        let actor_anchor_weights = actor_weights.clone();
        let critic_weights = init_critic_weights(&train_spec, 43);

        let actor_bundle = build_actor_graphs(&train_spec);
        let infer_bundle = build_actor_graphs(&infer_spec);

        let mut agent_infer = CompiledFlowMapAgent::compile(&session, &infer_bundle);
        let mut agent_anchor = CompiledFlowMapAgent::compile(&session, &infer_bundle);
        let mut agent_offline = CompiledFlowMapAgent::compile(&session, &actor_bundle);
        let mut agent_online = CompiledFlowMapAgent::compile(&session, &actor_bundle);
        let critic_bundle = build_critic_graphs(&train_spec);
        let mut critic = CompiledTwinCritic::compile(&session, &critic_bundle);
        let mut target_critic = CompiledTwinCritic::compile(&session, &critic_bundle);

        agent_infer.set_weights(&actor_weights);
        agent_anchor.set_weights(&actor_anchor_weights);
        agent_offline.set_weights(&actor_weights);
        agent_online.set_weights(&actor_weights);
        critic.set_weights(&critic_weights);
        let target_critic_weights = critic_weights.clone();
        target_critic.set_weights(&target_critic_weights);

        Self {
            spec,
            actor_weights,
            actor_anchor_weights,
            critic_weights,
            target_critic_weights,
            agent_infer,
            agent_anchor,
            agent_offline,
            agent_online,
            critic,
            target_critic,
            replay: ReplayBuffer::with_capacity(50_000),
            train_step: 0,
            rng: 7,
        }
    }

    pub fn freeze_offline_anchor(&mut self) {
        self.actor_anchor_weights = self.actor_weights.clone();
        self.agent_anchor.set_weights(&self.actor_anchor_weights);
    }

    /// Offline L_Diag + L_ESD + critic (Python `FlowMapPolicy`).
    pub fn offline_pretrain(&mut self, dataset: &OfflineDataset, steps: usize) {
        let b = self.spec.batch;
        if dataset.is_empty() {
            return;
        }
        for _ in 0..steps {
            let indices: Vec<usize> = (0..b)
                .map(|_| {
                    self.rng = crate::buffer::rand_like(self.rng);
                    (self.rng as usize) % dataset.len()
                })
                .collect();

            let diag = dataset.sample_diag_batch(&self.spec, &indices, &mut self.rng);
            self.actor_offline_step(&diag, 1.0);

            let esd_weight = if self.train_step > self.spec.flow_map_warmup_steps {
                1.0
            } else {
                0.0
            };
            if esd_weight > 0.0 {
                let inputs =
                    dataset.sample_esd_inputs(&self.spec, &indices, self.train_step, &mut self.rng);
                let esd = self.build_esd_batch(&inputs);
                self.actor_offline_step(&esd, esd_weight);
            }

            self.train_critic_offline_batch(dataset, &indices);
            self.train_step += 1;
            self.sync_actor();

            if self.train_step.is_multiple_of(50) {
                eprintln!("offline step {}", self.train_step);
            }
        }
        self.freeze_offline_anchor();
    }

    pub fn online_finetune<E: RlEnv>(&mut self, env: &mut E, env_steps: usize) {
        let ad = self.spec.action_dim;
        let b = self.spec.batch;

        for _ in 0..env_steps {
            let state = env.reset();
            let a0 = sample_noise(ad, &mut self.rng);
            let a1 = self.select_action(&state, &a0, &EvalConfig::one_step());

            let data_action = if self.replay.is_empty() {
                a1.clone()
            } else {
                let idx = self.replay.sample_indices(1, &mut self.rng)[0];
                self.replay.get(idx).action.clone()
            };
            self.fmq_actor_update(&state, &data_action, b);

            if self.spec.esd_weight > 0.0 {
                let indices = self.replay.sample_indices(b, &mut self.rng);
                if !indices.is_empty() {
                    let esd = self.sample_replay_esd_batch(&indices);
                    self.actor_online_step(&esd, self.spec.esd_weight);
                }
            }
            if self.spec.diag_weight > 0.0 {
                let indices = self.replay.sample_indices(b, &mut self.rng);
                if !indices.is_empty() {
                    let diag = self.sample_replay_diag_batch(&indices);
                    self.actor_online_step(&diag, self.spec.diag_weight);
                }
            }

            let tr = env.step(&a1);
            self.replay.push(tr);

            if self.replay.len() >= b {
                self.train_critic_batch(b);
            }

            self.train_step += 1;
            if self.train_step.is_multiple_of(100) {
                eprintln!(
                    "online step {} replay={}",
                    self.train_step,
                    self.replay.len()
                );
            }
        }
    }

    pub fn online_step_from_transition(&mut self, tr: &crate::buffer::Transition) {
        let b = self.spec.batch;
        self.fmq_actor_update(&tr.state, &tr.action, b);
        self.replay.push(tr.clone());
        if self.replay.len() >= b {
            self.train_critic_batch(b);
        }
        self.train_step += 1;
    }

    pub fn eval_rollout<E: RlEnv>(&mut self, env: &mut E, eval: &EvalConfig) -> f32 {
        let ad = self.spec.action_dim;
        let mut state = env.reset();
        let mut total = 0.0f32;
        loop {
            let a0 = sample_noise(ad, &mut self.rng);
            let mut action = self.select_action(&state, &a0, eval);
            if !action.iter().all(|x| x.is_finite()) {
                action.fill(0.0);
            }
            let tr = env.step(&action);
            total += tr.reward;
            if tr.done {
                break;
            }
            state = tr.next_state;
        }
        total
    }

    pub fn sample_a0(&mut self) -> Vec<f32> {
        sample_noise(self.spec.action_dim, &mut self.rng)
    }

    pub fn select_action(&mut self, state: &[f32], a0: &[f32], eval: &EvalConfig) -> Vec<f32> {
        select_action(
            &mut self.agent_infer,
            &mut self.critic,
            &self.spec,
            state,
            a0,
            eval,
        )
    }

    fn build_esd_batch(&mut self, inputs: &EsdInputsBatch) -> ActorTrainBatch {
        let b = inputs.r.len();
        let sd = self.spec.state_dim;
        let ad = self.spec.action_dim;
        let mut state = vec![0.0f32; b * sd];
        let mut a_r = vec![0.0f32; b * ad];
        let mut r = vec![0.0f32; b];
        let mut t = vec![0.0f32; b];
        let mut target_u = vec![0.0f32; b * ad];

        for bi in 0..b {
            let st = &inputs.state[bi * sd..(bi + 1) * sd];
            let ar = &inputs.a_r[bi * ad..(bi + 1) * ad];
            let v = &inputs.v_rt[bi * ad..(bi + 1) * ad];
            let (ar2, ri, ti, tgt) = esd_regression_target(
                self.spec.distillation_type,
                &mut self.agent_infer,
                &self.spec,
                st,
                ar,
                inputs.r[bi],
                inputs.t[bi],
                v,
                inputs.gamma[bi],
            );
            state[bi * sd..(bi + 1) * sd].copy_from_slice(st);
            a_r[bi * ad..(bi + 1) * ad].copy_from_slice(&ar2);
            r[bi] = ri;
            t[bi] = ti;
            target_u[bi * ad..(bi + 1) * ad].copy_from_slice(&tgt);
        }
        ActorTrainBatch {
            state,
            a_r,
            r,
            t,
            target_u,
        }
    }

    fn sample_replay_esd_batch(&mut self, indices: &[usize]) -> ActorTrainBatch {
        let b = indices.len();
        let sd = self.spec.state_dim;
        let ad = self.spec.action_dim;
        let mut state = vec![0.0f32; b * sd];
        let mut a_r = vec![0.0f32; b * ad];
        let mut v_rt = vec![0.0f32; b * ad];
        let mut gamma = vec![0.0f32; b];

        let (rs, ts) = sample_r_t(
            b,
            self.train_step,
            self.spec.esd_warmup_steps,
            self.spec.esd_anneal_end_step,
            &mut self.rng,
        );

        for (bi, &idx) in indices.iter().enumerate() {
            let tr = self.replay.get(idx);
            state[bi * sd..(bi + 1) * sd].copy_from_slice(&tr.state);
            let x0 = sample_noise(ad, &mut self.rng);
            let ri = rs[bi];
            for d in 0..ad {
                a_r[bi * ad + d] = (1.0 - ri) * x0[d] + ri * tr.action[d];
                v_rt[bi * ad + d] = tr.action[d] - x0[d];
            }
            gamma[bi] = uniform01(&mut self.rng);
        }

        let inputs = EsdInputsBatch {
            state,
            a_r,
            r: rs,
            t: ts,
            v_rt,
            gamma,
        };
        self.build_esd_batch(&inputs)
    }

    fn sample_replay_diag_batch(&mut self, indices: &[usize]) -> ActorTrainBatch {
        let b = indices.len();
        let sd = self.spec.state_dim;
        let ad = self.spec.action_dim;
        let mut state = vec![0.0f32; b * sd];
        let mut a_r = vec![0.0f32; b * ad];
        let mut r = vec![0.0f32; b];
        let mut t = vec![0.0f32; b];
        let mut target_u = vec![0.0f32; b * ad];

        for (bi, &idx) in indices.iter().enumerate() {
            let tr = self.replay.get(idx);
            state[bi * sd..(bi + 1) * sd].copy_from_slice(&tr.state);
            let x0 = sample_noise(ad, &mut self.rng);
            let tc = uniform01(&mut self.rng);
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

    fn actor_offline_step(&mut self, batch: &ActorTrainBatch, scale: f32) {
        let params = &self.agent_offline.offline_meta.params;
        let exec = &mut self.agent_offline.offline;
        let lr = self.spec.actor_lr;
        let outs = exec.run(&[
            ("state", &batch.state),
            ("a_r", &batch.a_r),
            ("r", &batch.r),
            ("t", &batch.t),
            ("target_u", &batch.target_u),
            ("d_output", &[scale]),
        ]);
        sgd(params, exec, &mut self.actor_weights, lr, &outs);
        self.sync_actor();
    }

    fn actor_online_step(&mut self, batch: &ActorTrainBatch, scale: f32) {
        let params = &self.agent_online.online_meta.params;
        let exec = &mut self.agent_online.online;
        let lr = self.spec.actor_lr;
        let outs = exec.run(&[
            ("state", &batch.state),
            ("a_r", &batch.a_r),
            ("r", &batch.r),
            ("t", &batch.t),
            ("target_u", &batch.target_u),
            ("d_output", &[scale]),
        ]);
        sgd(params, exec, &mut self.actor_weights, lr, &outs);
        self.sync_actor();
    }

    fn fmq_actor_update(&mut self, state: &[f32], data_action: &[f32], batch: usize) {
        let sd = self.spec.state_dim;
        let ad = self.spec.action_dim;
        let clip = self.spec.action_clip;

        self.rng = crate::buffer::rand_like(self.rng);
        let r = uniform01(&mut self.rng);
        let omr = 1.0 - r;
        let eps = sample_noise(ad, &mut self.rng);
        let a_r: Vec<f32> = eps
            .iter()
            .zip(data_action.iter())
            .map(|(&e, &a)| omr * e + r * a)
            .collect();

        let u_off = self.agent_anchor.velocity(state, &a_r, r, 1.0);
        let a1_off = clip_action(
            &a_r.iter()
                .zip(u_off.iter())
                .map(|(&ar, &u)| ar + omr * u)
                .collect::<Vec<_>>(),
            clip,
        );

        let u_theta = self.agent_infer.velocity(state, &a_r, r, 1.0);
        let a1_online = clip_action(
            &a_r.iter()
                .zip(u_theta.iter())
                .map(|(&ar, &u)| ar + omr * u)
                .collect::<Vec<_>>(),
            clip,
        );

        let a_grad = if self.spec.fmq_grad_at_online {
            &a1_online
        } else {
            &a1_off
        };
        let mut grad_q = self.critic.action_grad(state, a_grad);
        if !grad_q.iter().all(|g| g.is_finite()) {
            return;
        }
        if self.spec.fmq_normalize_grad {
            grad_q = normalize_grad(&grad_q);
        }

        let (q1, q2) = self.critic.q_values(state, a_grad);
        let eta_eff = fmq_eta_effective(&self.spec, q1, q2);
        let target_u: Vec<f32> = u_off
            .iter()
            .zip(grad_q.iter())
            .map(|(&u, &g)| u + eta_eff * g)
            .collect();

        let batch_train = ActorTrainBatch {
            state: {
                let mut s = vec![0.0f32; batch * sd];
                for i in 0..batch {
                    s[i * sd..(i + 1) * sd].copy_from_slice(state);
                }
                s
            },
            a_r: {
                let mut a = vec![0.0f32; batch * ad];
                for i in 0..batch {
                    a[i * ad..(i + 1) * ad].copy_from_slice(&a_r);
                }
                a
            },
            r: vec![r; batch],
            t: vec![1.0f32; batch],
            target_u: {
                let mut t = vec![0.0f32; batch * ad];
                for i in 0..batch {
                    t[i * ad..(i + 1) * ad].copy_from_slice(&target_u);
                }
                t
            },
        };
        self.actor_online_step(&batch_train, 1.0);
    }

    fn train_critic_offline_batch(&mut self, dataset: &OfflineDataset, indices: &[usize]) {
        let sd = self.spec.state_dim;
        let ad = self.spec.action_dim;
        let b = indices.len();
        let mut states = vec![0.0f32; b * sd];
        let mut actions = vec![0.0f32; b * ad];
        let mut targets = vec![0.0f32; b];

        for (bi, &idx) in indices.iter().enumerate() {
            let tr = &dataset.transitions[idx % dataset.len()];
            states[bi * sd..(bi + 1) * sd].copy_from_slice(&tr.state);
            actions[bi * ad..(bi + 1) * ad].copy_from_slice(&tr.action);
            let a0 = sample_noise(ad, &mut self.rng);
            let a_next = self.agent_infer.one_step(&tr.next_state, &a0);
            let q_next = self.target_critic.q_values(&tr.next_state, &a_next).0;
            targets[bi] = if tr.done {
                tr.reward
            } else {
                tr.reward + self.spec.gamma * q_next
            };
        }

        let outs = self.critic.train.run(&[
            ("state", &states),
            ("action", &actions),
            ("target", &targets),
            ("d_output", &[1.0f32]),
        ]);
        sgd(
            &self.critic.train_meta.params,
            &mut self.critic.train,
            &mut self.critic_weights,
            self.spec.critic_lr,
            &outs,
        );
        self.critic.set_weights(&self.critic_weights);
        soft_update_weights(
            &mut self.target_critic_weights,
            &self.critic_weights,
            self.spec.tau,
        );
        self.target_critic.set_weights(&self.target_critic_weights);
    }

    fn train_critic_batch(&mut self, batch: usize) {
        let indices = self.replay.sample_indices(batch, &mut self.rng);
        if indices.is_empty() {
            return;
        }
        let sd = self.spec.state_dim;
        let ad = self.spec.action_dim;
        let mut states = vec![0.0f32; batch * sd];
        let mut actions = vec![0.0f32; batch * ad];
        let mut targets = vec![0.0f32; batch];

        for (bi, &idx) in indices.iter().enumerate() {
            let tr = self.replay.get(idx);
            states[bi * sd..(bi + 1) * sd].copy_from_slice(&tr.state);
            actions[bi * ad..(bi + 1) * ad].copy_from_slice(&tr.action);
            let a0 = sample_noise(ad, &mut self.rng);
            let a_next = self.agent_infer.one_step(&tr.next_state, &a0);
            let q_next = self.target_critic.q_values(&tr.next_state, &a_next).0;
            targets[bi] = if tr.done {
                tr.reward
            } else {
                tr.reward + self.spec.gamma * q_next
            };
        }

        let outs = self.critic.train.run(&[
            ("state", &states),
            ("action", &actions),
            ("target", &targets),
            ("d_output", &[1.0f32]),
        ]);
        sgd(
            &self.critic.train_meta.params,
            &mut self.critic.train,
            &mut self.critic_weights,
            self.spec.critic_lr,
            &outs,
        );
        self.critic.set_weights(&self.critic_weights);
        soft_update_weights(
            &mut self.target_critic_weights,
            &self.critic_weights,
            self.spec.tau,
        );
        self.target_critic.set_weights(&self.target_critic_weights);
    }

    fn sync_actor(&mut self) {
        self.agent_infer.set_weights(&self.actor_weights);
        self.agent_offline.set_weights(&self.actor_weights);
        self.agent_online.set_weights(&self.actor_weights);
    }
}

fn uniform01(seed: &mut u64) -> f32 {
    *seed = crate::buffer::rand_like(*seed);
    ((*seed >> 11) as f32) / ((1u32 << 21) as f32)
}

fn sgd(
    params: &[ParamSlot],
    exec: &mut rlx_runtime::CompiledGraph,
    weights: &mut WeightStore,
    lr: f32,
    outs: &[Vec<f32>],
) {
    let mut global_norm_sq = 0.0f32;
    for (i, _p) in params.iter().enumerate() {
        let grad = outs.get(1 + i).expect("grad output");
        for gi in grad {
            if gi.is_finite() {
                global_norm_sq += gi * gi;
            }
        }
    }
    let clip = 1.0f32;
    let scale = if global_norm_sq > clip * clip {
        clip / global_norm_sq.sqrt()
    } else {
        1.0
    };

    for (i, p) in params.iter().enumerate() {
        let grad = outs.get(1 + i).expect("grad output");
        let w = weights.0.get_mut(&p.name).expect("param");
        assert_eq!(w.len(), grad.len(), "grad size mismatch for {}", p.name);
        for (wi, gi) in w.iter_mut().zip(grad.iter()) {
            if gi.is_finite() {
                *wi -= lr * scale * gi;
            }
        }
        debug_assert!(w.iter().all(|x| x.is_finite()), "NaN in {}", p.name);
        exec.set_param(&p.name, w);
    }
}

fn soft_update_weights(target: &mut WeightStore, online: &WeightStore, tau: f32) {
    for (name, ow) in &online.0 {
        let t = target.0.entry(name.clone()).or_insert_with(|| ow.clone());
        for (ti, &oi) in t.iter_mut().zip(ow.iter()) {
            *ti = tau * oi + (1.0 - tau) * *ti;
        }
    }
}
