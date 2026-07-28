// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Flow-map + FMQ demo on the optional toy reach-goal MDP.

use rlx_rl::dataset::OfflineDataset;
use rlx_rl::policy::EvalConfig;
use rlx_rl::spec::RlSpec;
use rlx_rl::toy_goal::ToyGoalEnv;
use rlx_rl::{FmqTrainer, QgbsConfig};

fn main() {
    let spec = RlSpec::toy(8);
    let demos = ToyGoalEnv::collect_expert_episodes(40, 50);
    let dataset = OfflineDataset::from_transitions(demos);

    let mut trainer = FmqTrainer::new(spec);
    let mut env = ToyGoalEnv::default();

    eprintln!("offline CFM…");
    trainer.offline_pretrain(&dataset, 200);

    let before = trainer.eval_rollout(&mut env, &EvalConfig::one_step());
    eprintln!("return (one-step): {before:.3}");

    eprintln!("online FMQ…");
    trainer.online_finetune(&mut env, 500);

    let after = trainer.eval_rollout(&mut env, &EvalConfig::one_step());
    eprintln!("return after online (one-step): {after:.3}");

    let qgbs = EvalConfig::with_qgbs(QgbsConfig::default());
    let with_search = trainer.eval_rollout(&mut env, &qgbs);
    eprintln!(
        "return with QGBS (NFE≈{}): {with_search:.3}",
        QgbsConfig::default().nfe()
    );
}
