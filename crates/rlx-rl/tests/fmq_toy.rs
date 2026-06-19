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
//! End-to-end FMQ on the toy reach-goal MDP.

use rlx_rl::dataset::OfflineDataset;
use rlx_rl::env::RlEnv;
use rlx_rl::policy::EvalConfig;
use rlx_rl::spec::RlSpec;
use rlx_rl::toy_goal::ToyGoalEnv;
use rlx_rl::{FmqTrainer, QgbsConfig};

#[test]
fn fmq_toy_offline_then_online() {
    let spec = RlSpec::toy(4);
    let demos = ToyGoalEnv::collect_expert_episodes(20, 40);
    let dataset = OfflineDataset::from_transitions(demos);

    let mut trainer = FmqTrainer::new(spec);
    trainer.offline_pretrain(&dataset, 40);

    let mut env = ToyGoalEnv::default();
    let before = trainer.eval_rollout(&mut env, &EvalConfig::one_step());
    trainer.online_finetune(&mut env, 100);
    let after = trainer.eval_rollout(&mut env, &EvalConfig::one_step());

    eprintln!("toy return before={before:.3} after={after:.3}");
    assert!(
        before.is_finite() && after.is_finite(),
        "returns must be finite"
    );
}

#[test]
fn eval_qgbs_is_finite() {
    let spec = RlSpec::toy(1);
    let demos = ToyGoalEnv::collect_expert_episodes(5, 30);
    let mut trainer = FmqTrainer::new(spec);
    trainer.offline_pretrain(&OfflineDataset::from_transitions(demos), 20);

    let mut env = ToyGoalEnv::default();
    let eval = EvalConfig::with_qgbs(QgbsConfig::default());
    let r = trainer.eval_rollout(&mut env, &eval);
    assert!(r.is_finite(), "QGBS eval return {r}");
}
