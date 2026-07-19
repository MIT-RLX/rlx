//! Ship-graph **inference**: partition a model into stages, ship each worker a
//! self-describing `StageSpec`, and drive activations rank→rank. Shared weight
//! loading + device resolution live in the parent `dist` module.

use super::{WeightCache, WeightRef, resolve_device};
use crate::compiled::CompiledGraph;
use crate::{Device, Session};
use rlx_driver::ProcessGroup;
use rlx_ir::Graph;
use serde::{Deserialize, Serialize};

/// Wire tags for the ship-graph protocol (kept clear of the reserved
/// collective range in `rlx-driver`).
const TAG_SPEC: u32 = 40;
const TAG_ACT: u32 = 41;

/// A self-describing unit of work for a worker: everything needed to run one
/// stage of a model, with nothing implied by convention.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StageSpec {
    pub graph: Graph,
    /// Input node name the activation feeds.
    pub input: String,
    /// Output node name (informational; `run` returns the single output).
    pub output: String,
    pub weights: Vec<WeightRef>,
    /// Device directive: `auto` (fastest available) or a device name
    /// (`cuda`, `metal`, `cpu`, …). Honored against local availability.
    pub device: String,
}

impl StageSpec {
    /// A stage whose weights all come from `uri_for(param_name)`, on `device`.
    pub fn new(
        graph: Graph,
        input: impl Into<String>,
        output: impl Into<String>,
        device: impl Into<String>,
    ) -> Self {
        Self {
            graph,
            input: input.into(),
            output: output.into(),
            weights: Vec::new(),
            device: device.into(),
        }
    }
    /// Add an f32 weight (dequantized on load).
    pub fn weight(mut self, name: impl Into<String>, uri: impl Into<String>) -> Self {
        self.weights.push(WeightRef {
            name: name.into(),
            uri: uri.into(),
            packed: false,
        });
        self
    }
    /// Add a **quantized** weight loaded as packed bytes (for a `DequantMatMul`
    /// graph) — native quant memory, no dequant round-trip.
    pub fn weight_packed(mut self, name: impl Into<String>, uri: impl Into<String>) -> Self {
        self.weights.push(WeightRef {
            name: name.into(),
            uri: uri.into(),
            packed: true,
        });
        self
    }
}

/// Coordinator: ship a stage spec to worker `rank`.
pub fn ship_stage(group: &ProcessGroup, rank: u32, spec: &StageSpec) -> Result<(), String> {
    let bytes = serde_json::to_vec(spec).map_err(|e| e.to_string())?;
    group
        .transport()
        .send_bytes(rank, TAG_SPEC, &bytes)
        .map_err(|e| format!("ship_stage: {e}"))
}

/// Send an activation tensor to `to` (a worker, or the next pipeline stage).
pub fn send_activation(group: &ProcessGroup, to: u32, data: &[f32]) -> Result<(), String> {
    group.send_f32(to, TAG_ACT, data).map_err(|e| e.to_string())
}

/// Receive an activation tensor from `from`.
pub fn recv_activation(group: &ProcessGroup, from: u32) -> Result<Vec<f32>, String> {
    group.recv_f32(from, TAG_ACT).map_err(|e| e.to_string())
}

/// A compiled stage sitting on this worker, ready to run activations.
pub struct WorkerStage {
    /// The backend this worker resolved the stage onto.
    pub device: Device,
    /// Node count of the received graph (handy for logging).
    pub nodes: usize,
    input: String,
    compiled: CompiledGraph,
}

impl WorkerStage {
    /// Run one activation through the stage; returns the (single) output.
    pub fn run(&mut self, x: &[f32]) -> Vec<f32> {
        self.compiled
            .run(&[(self.input.as_str(), x)])
            .into_iter()
            .next()
            .unwrap_or_default()
    }
    pub fn input_name(&self) -> &str {
        &self.input
    }
}

/// Worker: receive a [`StageSpec`] from the coordinator (rank 0), resolve its
/// weights via `resolve` (`uri -> f32`), pick the directed device, and compile.
/// This is the generic worker's setup — the same code runs any model's stage.
///
/// `resolve` is where the caller plugs in weight loading (GGUF / safetensors /
/// HF), keeping rlx core model-agnostic. It runs on the worker, so the weights
/// never cross the wire.
pub fn recv_stage<F>(group: &ProcessGroup, mut fallback: F) -> Result<WorkerStage, String>
where
    F: FnMut(&str) -> Vec<f32>,
{
    let bytes = group
        .transport()
        .recv_bytes(0, TAG_SPEC)
        .map_err(|e| format!("recv_stage: {e}"))?;
    let spec: StageSpec = serde_json::from_slice(&bytes).map_err(|e| format!("StageSpec: {e}"))?;
    let device = resolve_device(&spec.device);
    let nodes = spec.graph.len();
    let mut compiled = Session::new(device).compile(spec.graph);
    // Parse-once, serve-many: one `WeightCache` across the stage's weights, so
    // many tensors from the same GGUF are read after a single parse.
    let mut cache = WeightCache::new();
    for w in &spec.weights {
        if w.packed {
            // Quantized-on-device: raw packed bytes → U8 param. The graph's
            // DequantMatMul decodes at matmul time (native quant memory).
            let bytes = cache
                .bytes(&w.uri)
                .map_err(|e| format!("weight {}: {e}", w.name))?;
            compiled.set_param_typed(&w.name, &bytes, rlx_ir::DType::U8);
        } else {
            // f32: built-in gguf:// / safetensors:// / file:// (cached), else
            // the caller's `fallback` resolver (custom schemes like seed://).
            let vals = cache.f32(&w.uri).unwrap_or_else(|_| fallback(&w.uri));
            compiled.set_param(&w.name, &vals);
        }
    }
    Ok(WorkerStage {
        device,
        nodes,
        input: spec.input,
        compiled,
    })
}

/// Worker convenience: set up the stage, then run one activation cycle —
/// receive from the previous rank (or an external feed for rank 1), run, and
/// forward to the next rank (or back to the coordinator if last). Returns the
/// device it ran on. For a serving loop, call [`recv_stage`] once and drive
/// [`WorkerStage::run`] + [`recv_activation`]/[`send_activation`] yourself.
pub fn serve_stage<F>(group: &ProcessGroup, resolve: F) -> Result<Device, String>
where
    F: FnMut(&str) -> Vec<f32>,
{
    let n = group.world_size();
    let rank = group.rank();
    let mut stage = recv_stage(group, resolve)?;
    let input = recv_activation(group, rank - 1)?;
    let out = stage.run(&input);
    let next = if rank + 1 < n { rank + 1 } else { 0 };
    send_activation(group, next, &out)?;
    Ok(stage.device)
}

/// Same as [`serve_stage`] but with no custom resolver — a worker binary that
/// serves GGUF / safetensors / raw-f32 models out of the box (`recv_stage`'s
/// built-in [`WeightCache`] handles those schemes; anything else warns).
pub fn serve_stage_uri(group: &ProcessGroup) -> Result<Device, String> {
    serve_stage(group, |uri| {
        eprintln!("rlx dist: no resolver for weight URI [{uri}]");
        Vec::new()
    })
}
