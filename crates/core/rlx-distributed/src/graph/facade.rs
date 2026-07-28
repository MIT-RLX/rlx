// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ergonomic front door. [`Pipeline`] wraps the partition + run steps so the
//! common paths are one call, while the underlying [`crate::partition`],
//! [`crate::pipeline`] and [`crate::distributed`] pieces stay available for full
//! control (custom cuts, custom transports, streaming param sources).
//!
//! ```ignore
//! use rlx_distributed::{Pipeline, NamedTensor, Param};
//! use rlx_runtime::{CompileOptions, Device};
//!
//! // Single machine, any graph + any weight source:
//! let out = Pipeline::partition(&graph, 2)
//!     .run_local(&mut params, vec![NamedTensor::new("input_ids", vec![1, seq], ids)],
//!                Device::Cpu, &CompileOptions::default());
//!
//! // Cluster: split by whole layers, balance by weight bytes, ship + serve.
//! let pipe = Pipeline::partition_with(&graph, 2, |i, m| if i < m/2 {0} else {1});
//! let logits = pipe.run_tcp(&["host-b:9000".into(), "host-c:9000".into()], inputs)?;
//! ```

use super::partition::{Stage, partition, partition_with};
use super::pipeline::{NamedTensor, run_pipeline_local};
use crate::source::ParamSource;
use rlx_ir::Graph;
use rlx_runtime::{CompileOptions, Device};
use std::io;

/// An ordered set of pipeline [`Stage`]s ready to run in-process or on a cluster.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Pipeline {
    stages: Vec<Stage>,
}

impl Pipeline {
    /// Balanced split of `graph` into `n_stages` (by compute-node count).
    pub fn partition(graph: &Graph, n_stages: usize) -> Self {
        Self {
            stages: partition(graph, n_stages),
        }
    }

    /// Split with a custom compute-node→stage assignment (`(i, m) -> stage`),
    /// e.g. cut on layer boundaries or balance by per-stage weight bytes so an
    /// uneven cluster (say 61 GB + 44 GB) each gets a stage that fits.
    pub fn partition_with(
        graph: &Graph,
        n_stages: usize,
        assign: impl Fn(usize, usize) -> usize,
    ) -> Self {
        Self {
            stages: partition_with(graph, n_stages, assign),
        }
    }

    /// Wrap pre-built stages (e.g. deserialized on a worker).
    pub fn from_stages(stages: Vec<Stage>) -> Self {
        Self { stages }
    }

    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }
    pub fn len(&self) -> usize {
        self.stages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
    pub fn into_stages(self) -> Vec<Stage> {
        self.stages
    }

    /// Run all stages in this process (single machine). `source` supplies every
    /// param; peak RAM is still one stage's weights at a time.
    pub fn run_local(
        self,
        source: &mut dyn ParamSource,
        inputs: Vec<NamedTensor>,
        device: Device,
        opts: &CompileOptions,
    ) -> Vec<NamedTensor> {
        run_pipeline_local(self.stages, source, inputs, device, opts)
    }

    /// Drive a cluster where `worker_addrs[i]` serves stage `i`
    /// (see [`crate::distributed::serve_stage`]). Returns the final logits.
    pub fn run_tcp(
        &self,
        worker_addrs: &[String],
        inputs: Vec<NamedTensor>,
    ) -> io::Result<Vec<NamedTensor>> {
        super::transport::run_pipeline_tcp(&self.stages, worker_addrs, inputs)
    }
}
