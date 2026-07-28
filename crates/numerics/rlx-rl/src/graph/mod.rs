// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// RLX — MLP flow-map actor + twin critic (compiled on CPU).
//
// Graphs are built with `rlx_ir::Graph` + [`rlx_ir::infer::GraphExt`], not `rlx-flow`.
// Training graphs call [`rlx_compile::legalize_broadcast::run_with_remap`] before
// [`rlx_autodiff::grad_with_loss`].

mod actor;
mod critic;
mod mlp;

pub use actor::{
    ActorGraphBundle, ActorTrainGraph, CompiledFlowMapAgent, WeightStore, build_actor_graphs,
    init_actor_weights,
};
pub use critic::{
    CompiledTwinCritic, CriticGraphBundle, CriticQGradGraph, CriticTrainGraph, build_critic_graphs,
    init_critic_weights,
};
pub use mlp::ParamSlot;
