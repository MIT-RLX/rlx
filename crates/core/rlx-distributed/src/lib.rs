// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-distributed — model-agnostic multi-node inference
//!
//! Run a model across machines two ways, both **model-agnostic** (RLX provides
//! the API; model crates provide only the thin model-specific seam):
//!
//! - **Layer-block pipeline** (crate root) — a model implements a
//!   [`BlockRunner`] for a contiguous range of its transformer layers; the
//!   [`PipelineCoordinator`] relays hidden states rank→rank over an
//!   `rlx-driver` [`ProcessGroup`] (TCP / Thunderbolt / MLX). Reverse-rank like
//!   mlx-lm's `PipelineMixin`. [`config`] parses `hosts.json`, [`launch`] spawns
//!   a local N-rank cluster (torchrun-style). This is the idiomatic path for
//!   layer-structured LLMs (rlx-qwen3 ships a `BlockRunner`).
//! - **Graph-node pipeline** ([`graph`]) — partition an arbitrary compiled
//!   [`rlx_ir::Graph`] into weight-sharded subgraph stages joined by named
//!   boundary tensors ([`graph::partition`]), executed in-process
//!   ([`graph::run_pipeline_local`]) or over TCP ([`graph::serve_stage`] +
//!   [`graph::Pipeline::run_tcp`]). Use it when a model isn't a clean layer
//!   stack (e.g. the DeepSeek-V4 graph) or you want automatic weight sharding.
//!
//! Both seams share [`source::ParamSource`] — the model-agnostic weight seam:
//! RLX asks for params by name, a model crate returns dense f32 or packed bytes
//! (wrapping its own mlx / GGUF / safetensors loader). No model dependency leaks
//! into this crate.

pub mod cluster;
pub mod config;
pub mod experts;
pub mod graph;
pub mod launch;
pub mod partition;
pub mod pipeline;
pub mod source;

// ── Layer-block seam (public API used by rlx-qwen3 / rlx-protocol) ──
pub use config::{DistConfig, Hostfile, ParallelMode, TransportBackend};
pub use launch::{LocalCluster, WorkerArgs, free_loopback_ports, worker_args};
pub use partition::{BlockRole, block_role, pipeline_layer_range};
pub use pipeline::{BlockInput, BlockOutput, BlockRunner, PipelineCoordinator};

// ── MoE expert-parallel offload seam (model-agnostic; DeepSeek / Llama-4 / Kimi) ──
pub use experts::{
    ExpertProvider, ExpertShards, dispatch_experts, serve_expert_worker, shutdown_expert_workers,
};

// Re-export the transport primitives so model crates depend only on
// `rlx-distributed`, not `rlx-driver` directly.
pub use rlx_driver::{
    NetTransport, ProcessGroup, ReduceKind, TcpTransport, ThunderboltTransport, Transport,
};

// ── Graph-node seam + shared weight source ──
pub use graph::{NamedTensor, Pipeline, Stage, StageRunner};
pub use source::{Param, ParamSource};

// ── Cluster orchestration (config-driven, HW-aware placement + monitoring) ──
pub use cluster::{
    Assignment, Cluster, ClusterConfig, ClusterRun, KvPolicy, ModelCost, NodeCaps, NodeConfig,
    PlacementPolicy, probe_local,
};
