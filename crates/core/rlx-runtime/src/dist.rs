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

//! Ship-graph distributed execution — **build the worker once, run any model.**
//!
//! A worker is a *generic* executor: it compiles and runs a graph the
//! coordinator ships at runtime, so no model is baked into the worker binary
//! and there is **no per-node / per-model recompile**. Build one worker binary
//! per architecture (`Node::from_env().connect()` + [`serve_stage`]), deploy it
//! to every node, and drive any model from a coordinator.
//!
//! The coordinator — which owns the model's graph builders (e.g. a crate in
//! `rlx-models`) — partitions the model into stages and ships each worker a
//! [`StageSpec`]: the serialized subgraph, its I/O node names, where to fetch
//! each weight, and a device directive. Weights are resolved **locally** by a
//! caller-provided closure (GGUF / safetensors / HF — rlx core stays
//! model-agnostic), so only specs (KB) and activations cross the wire — never
//! the weights.
//!
//! ```no_run
//! # use rlx_runtime::dist;
//! # use rlx_driver::ProcessGroup; use std::sync::Arc;
//! # fn demo(group: Arc<ProcessGroup>) -> Result<(), String> {
//! // Generic worker binary — the SAME build runs any model:
//! let mut stage = dist::recv_stage(&group, |uri| load_weights(uri))?;
//! let x = dist::recv_activation(&group, group.rank() - 1)?;
//! let y = stage.run(&x);
//! dist::send_activation(&group, /*next*/ 0, &y)?;
//! # Ok(()) }
//! # fn load_weights(_uri: &str) -> Vec<f32> { vec![] }
//! ```

use crate::compiled::CompiledGraph;
use crate::{Device, Session};
use rlx_driver::ProcessGroup;
use rlx_ir::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wire tags for the ship-graph protocol (kept clear of the reserved
/// collective range in `rlx-driver`).
const TAG_SPEC: u32 = 40;
const TAG_ACT: u32 = 41;

/// Where one parameter's data comes from. The worker resolves the `uri`
/// itself, so large weights stay node-local.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WeightRef {
    pub name: String,
    /// Opaque to rlx core — the caller's resolver interprets it (e.g.
    /// `gguf:///models/x.gguf#blk.0.ffn`, `file:///…`, `hf://…`).
    pub uri: String,
    /// If set, load the **quantized bytes as-is** (U8 param) instead of
    /// dequantizing to f32 — the graph's `DequantMatMul` decodes them at
    /// matmul time, so the weight stays at native quant memory (e.g. Q4_K)
    /// with no dequant round-trip. The coordinator sets this when it built a
    /// `DequantMatMul` graph for the stage.
    #[serde(default)]
    pub packed: bool,
}

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

fn resolve_device(spec: &str) -> Device {
    if spec.eq_ignore_ascii_case("auto") || spec.is_empty() {
        return crate::fastest_device();
    }
    match crate::parse_device(spec) {
        Ok(d) if crate::is_available(d) => d,
        _ => crate::fastest_device(),
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

// ── weight source URIs ─────────────────────────────────────────────────────
//
// A model-agnostic resolver for the common on-disk formats, so the generic
// worker can load its own shard from a path the coordinator names — weights
// stay node-local, only the URI crosses the wire.

fn split_frag(rest: &str) -> Result<(&str, &str), String> {
    rest.split_once('#')
        .ok_or_else(|| format!("weight URI needs a #<tensor> fragment: {rest}"))
}

/// Resolve a weight tensor to f32 from its source URI. Supported schemes:
///   `gguf://<path>#<tensor>`         — dequantize a GGUF tensor (any K-quant)
///   `safetensors://<path>#<tensor>`  — read a safetensors tensor (F32/F16/BF16)
///   `file://<path>`                  — raw little-endian f32 array
///
/// This is model-agnostic (formats, not models); callers can still pass their
/// own closure to [`recv_stage`]/[`serve_stage`] for other schemes.
pub fn resolve_weight_uri(uri: &str) -> Result<Vec<f32>, String> {
    if let Some(rest) = uri.strip_prefix("gguf://") {
        let (path, tensor) = split_frag(rest)?;
        let mut f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let gguf = rlx_gguf::GgufFile::from_reader(&mut f).map_err(|e| e.to_string())?;
        let (data, _dims) = gguf.dequant_f32(tensor).map_err(|e| e.to_string())?;
        Ok(data)
    } else if let Some(rest) = uri.strip_prefix("safetensors://") {
        let (path, tensor) = split_frag(rest)?;
        let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        read_safetensors_f32(&bytes, tensor)
    } else if let Some(path) = uri.strip_prefix("file://") {
        let b = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    } else {
        Err(format!("unsupported weight URI scheme: {uri}"))
    }
}

/// Load the **quantized bytes** of a weight as-is (no dequant), for the
/// on-device-quant path. `gguf://<path>#<tensor>` returns the packed block
/// bytes; `file://<path>` returns the raw file. Feed to `set_param_typed(_, _,
/// U8)` behind a `DequantMatMul` graph.
pub fn resolve_weight_bytes(uri: &str) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("gguf://") {
        let (path, tensor) = split_frag(rest)?;
        let mut f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let gguf = rlx_gguf::GgufFile::from_reader(&mut f).map_err(|e| e.to_string())?;
        let t = gguf
            .get(tensor)
            .ok_or_else(|| format!("tensor not found: {tensor}"))?;
        Ok(gguf.tensor_bytes(t).map_err(|e| e.to_string())?.to_vec())
    } else if let Some(path) = uri.strip_prefix("file://") {
        std::fs::read(path).map_err(|e| format!("read {path}: {e}"))
    } else {
        Err(format!("packed load unsupported for scheme: {uri}"))
    }
}

/// Parse-once, serve-many weight cache. A GGUF file is parsed a single time and
/// then serves any number of its tensors (f32 or packed) — the right shape for
/// a worker whose stage pulls many tensors (q/k/v/o/ffn) from one model file.
/// Created per stage inside [`recv_stage`]; also usable standalone.
#[derive(Default)]
pub struct WeightCache {
    gguf: HashMap<String, rlx_gguf::GgufFile>,
}

impl WeightCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn gguf_file(&mut self, path: &str) -> Result<&rlx_gguf::GgufFile, String> {
        if !self.gguf.contains_key(path) {
            let mut f = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
            let g = rlx_gguf::GgufFile::from_reader(&mut f).map_err(|e| e.to_string())?;
            self.gguf.insert(path.to_string(), g);
        }
        Ok(&self.gguf[path])
    }

    /// Dequantized f32 for `uri`, reusing the parsed file for `gguf://`.
    pub fn f32(&mut self, uri: &str) -> Result<Vec<f32>, String> {
        if let Some(rest) = uri.strip_prefix("gguf://") {
            let (path, tensor) = split_frag(rest)?;
            let (data, _dims) = self
                .gguf_file(path)?
                .dequant_f32(tensor)
                .map_err(|e| e.to_string())?;
            Ok(data)
        } else if uri.starts_with("safetensors://") || uri.starts_with("file://") {
            resolve_weight_uri(uri) // header parse is cheap / file is the tensor
        } else {
            Err(format!("unsupported weight URI scheme: {uri}"))
        }
    }

    /// Packed (quantized) bytes for `uri`, reusing the parsed file for `gguf://`.
    pub fn bytes(&mut self, uri: &str) -> Result<Vec<u8>, String> {
        if let Some(rest) = uri.strip_prefix("gguf://") {
            let (path, tensor) = split_frag(rest)?;
            let g = self.gguf_file(path)?;
            let t = g
                .get(tensor)
                .ok_or_else(|| format!("tensor not found: {tensor}"))?;
            Ok(g.tensor_bytes(t).map_err(|e| e.to_string())?.to_vec())
        } else {
            resolve_weight_bytes(uri)
        }
    }
}

/// Built-in resolver for [`serve_stage_uri`]: [`resolve_weight_uri`], logging
/// and returning empty on failure so a missing shard surfaces loudly rather
/// than panicking mid-collective.
pub fn uri_resolver(uri: &str) -> Vec<f32> {
    match resolve_weight_uri(uri) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("rlx dist: weight resolve failed [{uri}]: {e}");
            Vec::new()
        }
    }
}

// bf16 is the top 16 bits of an f32.
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// IEEE-754 half → f32 (handles subnormal / inf / nan).
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let v = match exp {
        0 => (mant as f32) * 2f32.powi(-24), // 0 or subnormal
        0x1f if mant == 0 => f32::INFINITY,
        0x1f => f32::NAN,
        _ => (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 { -v } else { v }
}

/// Minimal safetensors reader: `[u64 header_len][JSON header][data]`; returns
/// the named tensor as f32 (F32 / F16 / BF16). No external crate needed.
fn read_safetensors_f32(buf: &[u8], name: &str) -> Result<Vec<f32>, String> {
    if buf.len() < 8 {
        return Err("safetensors file too small".into());
    }
    let hlen = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    let header = buf
        .get(8..8 + hlen)
        .ok_or("safetensors: truncated header")?;
    let data = &buf[8 + hlen..];
    let v: serde_json::Value = serde_json::from_slice(header).map_err(|e| e.to_string())?;
    let t = v
        .get(name)
        .ok_or_else(|| format!("tensor not found: {name}"))?;
    let dtype = t.get("dtype").and_then(|d| d.as_str()).ok_or("no dtype")?;
    let off = t
        .get("data_offsets")
        .and_then(|o| o.as_array())
        .ok_or("no data_offsets")?;
    let start = off[0].as_u64().ok_or("bad data_offsets")? as usize;
    let end = off[1].as_u64().ok_or("bad data_offsets")? as usize;
    let raw = data
        .get(start..end)
        .ok_or("safetensors: data_offsets out of range")?;
    Ok(match dtype {
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => raw
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        other => return Err(format!("unsupported safetensors dtype: {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_safetensors_f32() {
        let vals = [1.0f32, 2.0, -3.5, 4.25];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = format!(
            r#"{{"w":{{"dtype":"F32","shape":[4],"data_offsets":[0,{}]}}}}"#,
            data.len()
        );
        let mut buf = (header.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&data);
        let path = std::env::temp_dir().join(format!("rlx_dist_{}.safetensors", std::process::id()));
        std::fs::write(&path, &buf).unwrap();
        let got = resolve_weight_uri(&format!("safetensors://{}#w", path.display())).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(got, vals);
    }

    #[test]
    fn resolve_gguf_f32() {
        use rlx_gguf::{GgmlType, GgufWriter};
        let vals = [1.0f32, 2.0, -3.5, 4.25];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut w = GgufWriter::new();
        w.add_tensor_bytes("w", vec![4], GgmlType::F32, bytes).unwrap();
        let path = std::env::temp_dir().join(format!("rlx_dist_{}.gguf", std::process::id()));
        w.write_to_path(&path).unwrap();
        let got = resolve_weight_uri(&format!("gguf://{}#w", path.display())).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(got, vals);
    }

    #[test]
    fn weight_cache_serves_many_tensors_from_one_file() {
        use rlx_gguf::{GgmlType, GgufWriter};
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0];
        let mut w = GgufWriter::new();
        w.add_tensor_bytes("a", vec![4], GgmlType::F32, a.iter().flat_map(|v| v.to_le_bytes()).collect())
            .unwrap();
        w.add_tensor_bytes("b", vec![2], GgmlType::F32, b.iter().flat_map(|v| v.to_le_bytes()).collect())
            .unwrap();
        let path = std::env::temp_dir().join(format!("rlx_cache_{}.gguf", std::process::id()));
        w.write_to_path(&path).unwrap();
        let p = path.display();

        // One cache, two tensors from the same parsed file.
        let mut cache = WeightCache::new();
        assert_eq!(cache.f32(&format!("gguf://{p}#a")).unwrap(), a);
        assert_eq!(cache.f32(&format!("gguf://{p}#b")).unwrap(), b);
        // packed bytes of the F32 tensor are exactly its little-endian f32 bytes.
        let raw = cache.bytes(&format!("gguf://{p}#a")).unwrap();
        let raw_f32: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(raw_f32, a);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn dequant_matmul_gguf_end_to_end() {
        // Quantize a weight to Q8_0, feed the PACKED bytes as a U8 param behind
        // a DequantMatMul graph, and check the output matches dequant-then-
        // matmul — the quantized-on-device path, weight at native quant memory.
        use rlx_ir::quant::QuantScheme;
        use rlx_ir::{DType, Graph, Shape};

        let (m, k, n) = (2usize, 32usize, 4usize); // K multiple of 32 for Q8_0
        let w_floats: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
        let packed = rlx_gguf::quantize(&w_floats, rlx_gguf::GgmlType::Q8_0).unwrap();
        let x: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.05).collect();

        let mut g = Graph::new("dq");
        let xin = g.input("x", Shape::new(&[m, k], DType::F32));
        let wp = g.param("W", Shape::new(&[packed.len()], DType::U8));
        let out =
            g.dequant_matmul_packed(xin, wp, QuantScheme::GgufQ8_0, Shape::new(&[m, n], DType::F32));
        g.set_outputs(vec![out]);

        let mut c = Session::new(Device::Cpu).compile(g);
        c.set_param_typed("W", &packed, DType::U8);
        let got = c.run(&[("x", x.as_slice())]).into_iter().next().unwrap();

        // Reference: dequant the SAME bytes, then BT-matmul (W is [n, k]).
        let w = rlx_gguf::dequant_q8_0(&packed, n * k).unwrap();
        let mut expect = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for c2 in 0..k {
                    acc += x[i * k + c2] * w[j * k + c2];
                }
                expect[i * n + j] = acc;
            }
        }
        let err = got
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-3, "max_err={err} got={got:?} expect={expect:?}");
    }
}
