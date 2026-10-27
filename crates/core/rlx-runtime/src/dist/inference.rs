// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
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
    /// Identifies which synthesized datapath this stage corresponds to, for
    /// pre-synthesized fixed-function ranks (FPGA bitstreams). `None` for
    /// ordinary stages, which any worker can compile. See
    /// [`node::FixedFunction`](crate::dist::node::FixedFunction).
    #[serde(default)]
    pub stage_id: Option<String>,
    /// Number format the worker compiles the stage in (`f32` when unset; any
    /// [`rlx_ir::NumberFormat`] spelling: `bf16`, `f16`, `tf32`, `f8e4m3`,
    /// `f4/f8`, `f64`, …). The coordinator decides, so every stage agrees.
    #[serde(default)]
    pub number_format: Option<String>,
    /// Round explicit `Cast`s to f16 / bf16 (see [`crate::HalfCasts`]).
    #[serde(default)]
    pub half_casts: bool,
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
            stage_id: None,
            number_format: None,
            half_casts: false,
        }
    }

    /// Round explicit `Cast`s to f16 / bf16 in this stage.
    pub fn round_half_casts(mut self) -> Self {
        self.half_casts = true;
        self
    }

    /// Compile this stage in `format` (see [`Self::number_format`]).
    pub fn number_format(mut self, format: impl Into<String>) -> Self {
        self.number_format = Some(format.into());
        self
    }

    /// Tag this stage with the synthesized datapath it targets, so a
    /// fixed-function rank can accept (or refuse) it at placement time.
    pub fn stage_id(mut self, id: impl Into<String>) -> Self {
        self.stage_id = Some(id.into());
        self
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
    let options = crate::CompileOptions {
        number_format: spec
            .number_format
            .as_deref()
            .map(str::parse)
            .transpose()
            .map_err(|e| format!("StageSpec.number_format: {e}"))?,
        half_casts: if spec.half_casts {
            crate::HalfCasts::Round
        } else {
            crate::CompileOptions::default().half_casts
        },
        ..Default::default()
    };
    let mut compiled = Session::new(device).compile_with(spec.graph, &options);
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

// ── Micro-batched streaming pipeline ────────────────────────────────────────

/// Wire tag for activations sent with [`send_activation_as`] (a 1-byte dtype
/// header + payload), kept apart from the plain-f32 [`TAG_ACT`] stream.
const TAG_ACT_WIRE: u32 = 43;

/// Send an activation as `wire` (f32, or 16-bit to halve a bandwidth-bound
/// hop). The receiver decodes any of them with [`recv_activation_any`].
pub fn send_activation_as(
    group: &ProcessGroup,
    to: u32,
    data: &[f32],
    wire: super::WireDtype,
) -> Result<(), String> {
    let wire = wire.for_activation(data);
    let mut bytes = Vec::with_capacity(1 + data.len() * 4);
    bytes.push(wire as u8);
    match wire {
        super::WireDtype::Auto => unreachable!("for_activation resolves Auto"),
        super::WireDtype::F32 => {
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        super::WireDtype::Bf16 => {
            for v in data {
                bytes.extend_from_slice(&half::bf16::from_f32(*v).to_bits().to_le_bytes());
            }
        }
        super::WireDtype::F16 => {
            for v in data {
                bytes.extend_from_slice(&half::f16::from_f32(*v).to_bits().to_le_bytes());
            }
        }
    }
    group
        .transport()
        .send_bytes(to, TAG_ACT_WIRE, &bytes)
        .map_err(|e| format!("send activation → {to}: {e}"))
}

/// Receive an activation sent by [`send_activation_as`], in whatever width.
pub fn recv_activation_any(group: &ProcessGroup, from: u32) -> Result<Vec<f32>, String> {
    let b = group
        .transport()
        .recv_bytes(from, TAG_ACT_WIRE)
        .map_err(|e| format!("receive activation ← {from}: {e}"))?;
    let (&kind, body) = b
        .split_first()
        .ok_or_else(|| format!("empty activation from rank {from}"))?;
    Ok(match kind {
        0 => body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        1 => body
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)).to_f32())
            .collect(),
        2 => body
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_bits(u16::from_le_bytes(*c)).to_f32())
            .collect(),
        k => {
            return Err(format!(
                "activation from rank {from}: unknown wire kind {k}"
            ));
        }
    })
}

/// A background sender: activations queue here and go out on their own
/// thread, so a stage computes micro-batch `i+1` while `i` is on the wire.
struct Outbox<'s> {
    tx: Option<std::sync::mpsc::Sender<Vec<f32>>>,
    handle: Option<std::thread::ScopedJoinHandle<'s, Result<(), String>>>,
}

impl<'s> Outbox<'s> {
    fn new<'e>(
        scope: &'s std::thread::Scope<'s, 'e>,
        group: &'e ProcessGroup,
        to: u32,
        wire: super::WireDtype,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let handle = scope.spawn(move || {
            for a in rx {
                send_activation_as(group, to, &a, wire)?;
            }
            Ok(())
        });
        Self {
            tx: Some(tx),
            handle: Some(handle),
        }
    }
    fn push(&self, a: Vec<f32>) -> Result<(), String> {
        self.tx
            .as_ref()
            .and_then(|t| t.send(a).ok())
            .ok_or_else(|| "activation sender exited".to_string())
    }
    /// Flush and surface any send error.
    fn finish(mut self) -> Result<(), String> {
        drop(self.tx.take());
        self.handle
            .take()
            .map(|h| h.join().unwrap_or_else(|_| Err("sender panicked".into())))
            .unwrap_or(Ok(()))
    }
}

/// Worker side of a streaming pipeline: run `n_micro` micro-batches through
/// `stage` — receive from the previous rank, run, hand to the next rank (the
/// last stage returns results to rank 0). Receiving, compute and sending
/// overlap across micro-batches, so with `M` micro-batches over `S` stages the
/// stages are busy for `M` of the `M + S − 1` slots instead of one in `S`.
pub fn serve_stage_stream(
    group: &ProcessGroup,
    run: &mut dyn FnMut(&[f32]) -> Vec<f32>,
    n_micro: usize,
    wire: super::WireDtype,
) -> Result<(), String> {
    let (rank, n) = (group.rank(), group.world_size());
    if rank == 0 {
        return Err("serve_stage_stream: rank 0 drives the pipeline (pipeline_run)".into());
    }
    let next = if rank + 1 < n { rank + 1 } else { 0 };
    std::thread::scope(|s| {
        let out = Outbox::new(s, group, next, wire);
        for _ in 0..n_micro {
            let x = recv_activation_any(group, rank - 1)?;
            out.push(run(&x))?;
        }
        out.finish()
    })
}

/// Rank 0 side of a streaming pipeline: run each micro-batch through stage 0
/// (`first`), stream it down the ranks, and collect the last stage's outputs
/// in order. With one rank it is just `first` on every micro-batch.
pub fn pipeline_run(
    group: &ProcessGroup,
    first: &mut dyn FnMut(&[f32]) -> Vec<f32>,
    micro: &[Vec<f32>],
    wire: super::WireDtype,
) -> Result<Vec<Vec<f32>>, String> {
    let n = group.world_size();
    if group.rank() != 0 {
        return Err("pipeline_run: only rank 0 drives the pipeline".into());
    }
    if n == 1 {
        return Ok(micro.iter().map(|x| first(x)).collect());
    }
    std::thread::scope(|s| {
        // Results arrive in order on their own thread while we keep feeding.
        let collector = s.spawn(|| {
            (0..micro.len())
                .map(|_| recv_activation_any(group, n - 1))
                .collect::<Result<Vec<_>, _>>()
        });
        let out = Outbox::new(s, group, 1, wire);
        for x in micro {
            out.push(first(x))?;
        }
        out.finish()?;
        collector
            .join()
            .unwrap_or_else(|_| Err("collector panicked".into()))
    })
}
