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
