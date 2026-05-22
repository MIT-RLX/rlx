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
// RLX — MLP flow-map actor + twin critic (compiled on CPU).
//
// Graphs are built with `rlx_ir::Graph` + [`rlx_ir::infer::GraphExt`], not `rlx-flow`.
// Training graphs call [`rlx_compile::legalize_broadcast::run_with_remap`] before
// [`rlx_autodiff::grad_with_loss`].

mod actor;
mod critic;
mod mlp;

pub use actor::{
    build_actor_graphs, init_actor_weights, ActorGraphBundle, ActorTrainGraph, CompiledFlowMapAgent,
    WeightStore,
};
pub use critic::{
    build_critic_graphs, init_critic_weights, CompiledTwinCritic, CriticGraphBundle, CriticQGradGraph,
    CriticTrainGraph,
};
pub use mlp::ParamSlot;
