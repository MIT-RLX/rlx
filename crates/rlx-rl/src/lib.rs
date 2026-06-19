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
// RLX — Flow Map Q-Guidance (arxiv:2605.12416).
//
// Design:
// - MLP actor/critic in `rlx-ir` (not `rlx-flow` LLM builders).
// - CPU execution via `rlx-runtime` + `rlx-autodiff` + `legalize_broadcast` before AD.
// - No simulator bindings: implement [`env::RlEnv`] and push [`buffer::Transition`]s.
// - Optional QGBS at eval: [`policy::EvalConfig`] over [`CompiledFlowMapAgent`].

//! # rlx-rl
//!
//! Compiled flow-map policies and twin critics for offline-to-online RL on RLX.

pub mod buffer;
pub mod dataset;
pub mod distillation;
pub mod env;
pub mod flow_curriculum;
pub mod graph;
pub mod guidance;
pub mod policy;
pub mod qgbs;
pub mod spec;

#[cfg(feature = "toy")]
pub mod toy_goal;

#[cfg(feature = "compile")]
pub mod trainer;

pub use buffer::{ReplayBuffer, Transition};
pub use dataset::OfflineDataset;
pub use env::RlEnv;
pub use flow_curriculum::sample_r_t;
pub use graph::{
    ActorGraphBundle, CompiledFlowMapAgent, CompiledTwinCritic, CriticGraphBundle, ParamSlot,
    WeightStore, build_actor_graphs, build_critic_graphs, init_actor_weights, init_critic_weights,
};
pub use guidance::{eta_effective, q_guided_project};
pub use policy::{EvalConfig, sample_noise, select_action};
pub use qgbs::{QgbsConfig, qgbs_select_action};
pub use spec::{DistillationType, RlSpec};

#[cfg(feature = "compile")]
pub use trainer::FmqTrainer;
