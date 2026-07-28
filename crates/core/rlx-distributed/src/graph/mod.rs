// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Graph-level** pipeline parallelism: partition a compiled [`rlx_ir::Graph`]
//! into weight-sharded subgraph stages connected by named boundary tensors. Use
//! this for models NOT expressed as clean layer blocks (arbitrary graphs), or
//! when you want automatic weight sharding without writing a `BlockRunner`.
//!
//! The complementary **layer-block** seam lives at the crate root
//! ([`crate::BlockRunner`] / [`crate::PipelineCoordinator`]): there a model
//! implements a runner for a contiguous range of transformer layers. Both seams
//! share [`crate::source::ParamSource`], the config/launch orchestration, and
//! (for the block seam) the `rlx-driver` transports.

pub mod facade;
pub mod partition;
pub mod pipeline;
pub mod transport;

pub use facade::Pipeline;
pub use partition::{Stage, balanced_stage_of, partition, partition_with};
pub use pipeline::{NamedTensor, StageRunner, run_pipeline_local};
pub use transport::{bind_stage, run_pipeline_tcp, serve_bound, serve_stage};
