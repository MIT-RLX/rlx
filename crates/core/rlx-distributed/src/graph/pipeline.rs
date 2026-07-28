// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Execute a partitioned model as a pipeline. [`run_pipeline_local`] runs every
//! stage in one process (single machine) — the correctness reference for the
//! multi-node path in [`crate::distributed`]. Each [`StageRunner`] compiles one
//! [`Stage`] and loads only that stage's parameters, so peak RAM is one stage's
//! weights, not the whole model.

use super::partition::Stage;
use crate::source::{Param, ParamSource};
use rlx_runtime::{CompileOptions, CompiledGraph, Device, Session};
use std::collections::HashMap;

/// A named activation tensor flowing between stages (row-major f32).
#[derive(Clone, Debug)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NamedTensor {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            shape,
            data,
        }
    }
}

/// A compiled stage plus its loaded parameter shard.
pub struct StageRunner {
    pub stage: Stage,
    compiled: CompiledGraph,
}

impl StageRunner {
    /// Compile `stage` on `device` and load ONLY its parameter shard from
    /// `source` (dense f32 or packed/typed). A param the source doesn't own is
    /// left unset (the runtime may stream it, or another shard owns it).
    pub fn compile(
        stage: Stage,
        source: &mut dyn ParamSource,
        device: Device,
        opts: &CompileOptions,
    ) -> Self {
        let mut compiled = Session::new(device).compile_with(stage.graph.clone(), opts);
        for p in &stage.params {
            match source.get(p) {
                Some(Param::F32(d)) => compiled.set_param(p, &d),
                Some(Param::Typed(bytes, dtype)) => compiled.set_param_typed(p, &bytes, dtype),
                None => {}
            }
        }
        Self { stage, compiled }
    }

    /// Names this stage must be fed (model inputs + upstream boundaries).
    pub fn input_names(&self) -> &[String] {
        &self.stage.inputs
    }

    /// Run the stage: gather its inputs from `pool` by name, return its outputs
    /// (order matches [`Stage::outputs`]).
    pub fn run(&mut self, pool: &HashMap<String, NamedTensor>) -> Vec<NamedTensor> {
        let feed: Vec<(&str, &[f32])> = self
            .stage
            .inputs
            .iter()
            .map(|n| {
                let t = pool.get(n).unwrap_or_else(|| {
                    panic!("stage {} missing input tensor `{n}`", self.stage.index)
                });
                (n.as_str(), t.data.as_slice())
            })
            .collect();
        let outs = self.compiled.run(&feed);
        self.stage
            .outputs
            .iter()
            .cloned()
            .zip(self.stage.output_shapes.iter().cloned())
            .zip(outs)
            .map(|((name, shape), data)| NamedTensor { name, shape, data })
            .collect()
    }
}

/// Run a partitioned model in one process, stage by stage. This is the
/// single-machine reference the distributed path must match bit-for-bit.
///
/// `source` supplies each stage's parameters (dense or packed); `inputs` seeds
/// the dataflow pool with the model's `Op::Input` tensors (`input_ids`, masks,
/// …) — a tensor needed by several stages stays in the pool and is reused.
/// Returns the final stage's outputs.
pub fn run_pipeline_local(
    stages: Vec<Stage>,
    source: &mut dyn ParamSource,
    inputs: Vec<NamedTensor>,
    device: Device,
    opts: &CompileOptions,
) -> Vec<NamedTensor> {
    let mut pool: HashMap<String, NamedTensor> =
        inputs.into_iter().map(|t| (t.name.clone(), t)).collect();
    let mut last = Vec::new();
    for stage in stages {
        let mut runner = StageRunner::compile(stage, source, device, opts);
        let outs = runner.run(&pool);
        for t in &outs {
            pool.insert(t.name.clone(), t.clone());
        }
        last = outs;
    }
    last
}
