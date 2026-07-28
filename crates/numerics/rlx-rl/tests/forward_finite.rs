// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Sanity: initialized flow-map forward is finite.

use rlx_rl::FmqTrainer;
use rlx_rl::dataset::OfflineDataset;
use rlx_rl::env::RlEnv;
use rlx_rl::graph::{CompiledFlowMapAgent, build_actor_graphs, init_actor_weights};
use rlx_rl::policy::EvalConfig;
use rlx_rl::spec::RlSpec;
use rlx_rl::toy_goal::ToyGoalEnv;
use rlx_runtime::{Device, Session};

#[test]
fn forward_finite_after_init() {
    let spec = RlSpec::toy(1);
    let bundle = build_actor_graphs(&spec);
    let mut agent = CompiledFlowMapAgent::compile(&Session::new(Device::Cpu), &bundle);
    agent.set_weights(&init_actor_weights(&spec, 1));
    let state = [0.0, 0.0, 1.0, 1.0];
    let a0 = [0.1, -0.1];
    let a1 = agent.one_step(&state, &a0);
    assert_eq!(a1.len(), 2);
    assert!(a1.iter().all(|x| x.is_finite()), "action={a1:?}");
}

#[test]
fn offline_keeps_weights_finite() {
    let spec = RlSpec::toy(4);
    let demos = ToyGoalEnv::collect_expert_episodes(2, 20);
    let dataset = OfflineDataset::from_transitions(demos);
    let mut trainer = FmqTrainer::new(spec.clone());
    trainer.offline_pretrain(&dataset, 40);
    let state = [0.0, 0.0, 1.0, 1.0];
    let a1 = trainer.agent_infer.one_step(&state, &[0.0, 0.0]);
    assert!(a1.iter().all(|x| x.is_finite()), "weights corrupt: {a1:?}");
}

#[test]
fn eval_rollout_finite_after_offline() {
    let spec = RlSpec::toy(4);
    let demos = ToyGoalEnv::collect_expert_episodes(20, 40);
    let mut trainer = FmqTrainer::new(spec);
    trainer.offline_pretrain(&OfflineDataset::from_transitions(demos), 40);
    let mut env = ToyGoalEnv::default();
    let r = trainer.eval_rollout(&mut env, &EvalConfig::one_step());
    assert!(r.is_finite(), "eval return {r}");
}
