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

//! Load a `.onnx` file directly into the bundle IR consumed by [`crate::lower`].

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use protobuf::Message;
use rlx_onnx_proto::onnx::{
    AttributeProto_AttributeType, ModelProto, TensorProto, TensorProto_DataType, TypeProto_Tensor,
};
use serde_json::json;

use crate::bundle::{BundleManifest, BundleNode, IoMeta, TensorMeta};
use crate::lower::{ImportOptions, ImportReport, build_hir_from_parts};
use crate::shape_propagate::propagate_shapes;

fn tensor_dtype(elem: TensorProto_DataType) -> String {
    match elem {
        TensorProto_DataType::FLOAT => "f32".to_string(),
        TensorProto_DataType::INT64 => "i64".to_string(),
        TensorProto_DataType::INT32 => "i32".to_string(),
        TensorProto_DataType::BOOL => "bool".to_string(),
        other => format!("type_{other:?}"),
    }
}

fn shape_from_tensor_type(tt: &TypeProto_Tensor) -> Vec<serde_json::Value> {
    if !tt.has_shape() {
        return vec![json!("?")];
    }
    tt.get_shape()
        .get_dim()
        .iter()
        .map(|d| {
            if d.has_dim_value() {
                let v = d.get_dim_value();
                if v > 0 { json!(v) } else { json!("?") }
            } else if d.has_dim_param() && !d.get_dim_param().is_empty() {
                // Preserve the symbolic NAME (e.g. `batch_size`, `text_length`) so
                // the resolver can tell a batch dim (→1) from a length dim (→seq).
                // Dropping it to "?" made every dynamic dim resolve to seq_len,
                // mangling batch (see resolve_dim_ir named-dim handling).
                json!(d.get_dim_param())
            } else {
                json!("?")
            }
        })
        .collect()
}

fn f32_from_raw(raw: &[u8], n: usize) -> Option<Vec<f32>> {
    if raw.len() != n * 4 {
        return None;
    }
    Some(
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn tensor_to_f32(name: &str, t: &TensorProto) -> Result<Vec<f32>> {
    if t.get_dims().contains(&0) {
        return Ok(Vec::new());
    }
    let n = t.get_dims().iter().product::<i64>().max(1) as usize;
    if !t.get_float_data().is_empty() {
        return Ok(t.get_float_data().to_vec());
    }
    if let Some(v) = f32_from_raw(t.get_raw_data(), n) {
        return Ok(v);
    }
    let dt = t.get_data_type();
    match dt {
        TensorProto_DataType::FLOAT | TensorProto_DataType::UNDEFINED => {
            if let Some(v) = f32_from_raw(t.get_raw_data(), n) {
                return Ok(v);
            }
        }
        TensorProto_DataType::INT64 => {
            let ints: Vec<i64> = if !t.get_int64_data().is_empty() {
                t.get_int64_data().to_vec()
            } else if t.get_raw_data().len() == n * 8 {
                t.get_raw_data()
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            } else {
                Vec::new()
            };
            return Ok(ints.into_iter().map(|x| x as f32).collect());
        }
        TensorProto_DataType::INT32 => {
            let raw = t.get_raw_data();
            if raw.len() == n * 4 {
                return Ok(raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                    .collect());
            }
        }
        TensorProto_DataType::UINT8 | TensorProto_DataType::INT8 => {
            let raw = t.get_raw_data();
            return Ok(raw.iter().map(|&b| b as f32).collect());
        }
        _ if dt as i32 == 10 => {
            // FLOAT16
            let raw = t.get_raw_data();
            if raw.len() == n * 2 {
                return Ok(raw
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect());
            }
        }
        _ if dt as i32 == 16 => {
            // BFLOAT16 (not in older protobuf enums)
            let raw = t.get_raw_data();
            if raw.len() == n * 2 {
                return Ok(raw
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u32::from(c[0]) | (u32::from(c[1]) << 8)) << 16))
                    .collect());
            }
        }
        _ => {}
    }
    anyhow::bail!("initializer {name}: unsupported dtype for native import")
}

fn tensor_to_i64(name: &str, t: &TensorProto) -> Result<Vec<i64>> {
    // A genuinely empty tensor (a dim of 0, e.g. a `Reshape`-to-scalar target `[]`)
    // has no data — return the empty vector rather than bailing on the byte check.
    if t.get_dims().contains(&0) {
        return Ok(Vec::new());
    }
    let n = t.get_dims().iter().product::<i64>().max(1) as usize;
    if !t.get_int64_data().is_empty() {
        return Ok(t.get_int64_data().to_vec());
    }
    if !t.get_int32_data().is_empty() {
        return Ok(t.get_int32_data().iter().map(|&x| x as i64).collect());
    }
    let raw = t.get_raw_data();
    match t.get_data_type() as i32 {
        7 if raw.len() == n * 8 => {
            return Ok(raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect());
        }
        6 if raw.len() == n * 4 => {
            return Ok(raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect());
        }
        9 => {
            return Ok(raw.iter().map(|&b| i64::from(b != 0)).collect());
        }
        _ => {}
    }
    anyhow::bail!("initializer {name}: unsupported i64 dtype for native import")
}

fn parse_attribute(a: &rlx_onnx_proto::onnx::AttributeProto) -> Option<serde_json::Value> {
    use AttributeProto_AttributeType::*;
    let name = a.get_name();
    match a.get_field_type() {
        INT => Some(json!(a.get_i())),
        INTS => Some(json!(a.get_ints().to_vec())),
        FLOAT => Some(json!(a.get_f())),
        FLOATS => Some(json!(a.get_floats().to_vec())),
        STRING => {
            let s = a.get_s();
            Some(json!(String::from_utf8_lossy(s).into_owned()))
        }
        STRINGS => Some(json!(
            a.get_strings()
                .iter()
                .map(|s| String::from_utf8_lossy(s).into_owned())
                .collect::<Vec<_>>()
        )),
        TENSOR => {
            let t = a.get_t();
            let dims: Vec<i64> = t.get_dims().to_vec();
            // Also capture the scalar value for small tensors. The metadata-only
            // form dropped the data, so `ConstantOfShape(value=1)` (all-ones mask
            // prefix) silently became 0. Cheap and additive — larger tensor attrs
            // still carry only shape/dtype.
            let scalar: Option<f64> = tensor_to_f32(name, t)
                .ok()
                .and_then(|v| v.first().map(|&x| x as f64))
                .or_else(|| {
                    tensor_to_i64(name, t)
                        .ok()
                        .and_then(|v| v.first().map(|&x| x as f64))
                });
            Some(json!({
                "tensor": {
                    "dims": dims,
                    "dtype": tensor_dtype(t.get_data_type()),
                    "scalar": scalar,
                }
            }))
        }
        GRAPH => Some(json!({ "graph": true, "name": name })),
        _ => None,
    }
}

// Fold a `Constant` op node's tensor into the param maps (mirrors an
// initializer), recording its shape + name. No-op for non-`Constant` nodes.
// Shared by the top-level graph and `If`-branch subgraphs (see `lower_if`).
thread_local! {
    /// Names of folded CONSTANT tensors that are rank-0 SCALARS. A scalar Gather
    /// index removes its axis (`Gather(x[1,10,17], scalar, axis=2) → [1,10]`), while
    /// a rank-1 `[1]` index keeps it (`→ [1,10,1]`). `fold_constant_node` records the
    /// original rank here so the lowerer's scalar-index squeeze fires for folded
    /// scalar constants (not just scalar graph INPUTS) — MOSS-TTS's embedding-mask
    /// `Gather(input_ids, 16)` was otherwise kept as `[1,10,1]`, injecting a spurious
    /// axis through the whole GPT-2 stack. Cleared per import in `prepare_onnx_file`.
    static SCALAR_CONSTS: std::cell::RefCell<HashSet<String>> =
        std::cell::RefCell::new(HashSet::new());
}

fn scalar_consts_clear() {
    SCALAR_CONSTS.with(|c| c.borrow_mut().clear());
}

/// Drain the folded-scalar-constant names collected during the current import.
pub fn take_scalar_consts() -> HashSet<String> {
    SCALAR_CONSTS.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

thread_local! {
    /// `If`-node name → (then-branch nodes, else-branch nodes). Lets `lower_if`
    /// INLINE a branch that COMPUTES its output (`Squeeze`/`Identity`/… of an
    /// outer-scope tensor) rather than only handling branches whose outputs are
    /// folded constants — the MOSS codec's final `If(shape[1]==1, squeeze(wave),
    /// wave)` reshaping otherwise fell to the zero stub and collapsed the audio.
    /// Cleared per import in `prepare_onnx_file`.
    static IF_BRANCHES: std::cell::RefCell<HashMap<String, (Vec<BundleNode>, Vec<BundleNode>)>> =
        std::cell::RefCell::new(HashMap::new());
}

fn if_branches_clear() {
    IF_BRANCHES.with(|c| c.borrow_mut().clear());
}

/// Drain the `If`-branch subgraph nodes collected during the current import.
pub fn take_if_branches() -> HashMap<String, (Vec<BundleNode>, Vec<BundleNode>)> {
    IF_BRANCHES.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

/// Install `If`-branch subgraphs before [`crate::build_hir_from_parts`] /
/// [`crate::build_hir_from_bundle`] when loading a pre-exported RLX graph dir
/// (no live `prepare_onnx_file` call on this thread).
pub fn install_if_branches(branches: HashMap<String, (Vec<BundleNode>, Vec<BundleNode>)>) {
    IF_BRANCHES.with(|c| *c.borrow_mut() = branches);
}

/// Install folded scalar-constant names before lowering a pre-exported graph.
pub fn install_scalar_consts(names: HashSet<String>) {
    SCALAR_CONSTS.with(|c| *c.borrow_mut() = names);
}

fn fold_constant_node(
    n: &rlx_onnx_proto::onnx::NodeProto,
    params: &mut HashMap<String, Vec<f32>>,
    i64_params: &mut HashMap<String, Vec<i64>>,
    const_shapes: &mut HashMap<String, Vec<usize>>,
    folded_constants: &mut HashSet<String>,
) {
    if n.get_op_type() != "Constant" || n.get_output().is_empty() {
        return;
    }
    let out = n.get_output()[0].to_string();
    let mut shape: Option<Vec<usize>> = None;
    for a in n.get_attribute() {
        // `value_int`/`value_float` are always rank-0 scalars.
        if matches!(a.get_name(), "value_int" | "value_float") {
            SCALAR_CONSTS.with(|c| {
                c.borrow_mut().insert(out.clone());
            });
        }
        match a.get_name() {
            "value" => {
                let t = a.get_t();
                let dims: Vec<usize> = t.get_dims().iter().map(|&d| d.max(1) as usize).collect();
                if t.get_dims().is_empty() {
                    SCALAR_CONSTS.with(|c| {
                        c.borrow_mut().insert(out.clone());
                    });
                }
                // Fold non-fatally: a Constant we cannot convert (external data,
                // exotic dtype) is left in the graph for `lower_constant` rather
                // than failing the whole import — folding is an optimization.
                let folded = match t.get_data_type() as i32 {
                    6 | 7 | 9 => match tensor_to_i64(&out, t) {
                        Ok(v) => {
                            i64_params.insert(out.clone(), v);
                            true
                        }
                        Err(_) => false,
                    },
                    _ => match tensor_to_f32(&out, t) {
                        Ok(v) => {
                            params.insert(out.clone(), v);
                            true
                        }
                        Err(_) => false,
                    },
                };
                if folded {
                    shape = Some(if dims.is_empty() { vec![1] } else { dims });
                }
            }
            "value_float" => {
                params.insert(out.clone(), vec![a.get_f()]);
                shape = Some(vec![1]);
            }
            "value_int" => {
                i64_params.insert(out.clone(), vec![a.get_i()]);
                shape = Some(vec![1]);
            }
            "value_floats" => {
                let v = a.get_floats().to_vec();
                shape = Some(vec![v.len().max(1)]);
                params.insert(out.clone(), v);
            }
            "value_ints" => {
                let v = a.get_ints().to_vec();
                shape = Some(vec![v.len().max(1)]);
                i64_params.insert(out.clone(), v);
            }
            _ => {}
        }
    }
    if let Some(sh) = shape {
        const_shapes.insert(out.clone(), sh);
        folded_constants.insert(out);
    }
}

fn output_meta_from_value_info(v: &rlx_onnx_proto::onnx::ValueInfoProto) -> serde_json::Value {
    if v.has_field_type() {
        let tp = v.get_field_type();
        if tp.has_tensor_type() {
            let tt = tp.get_tensor_type();
            return json!({
                "shape": shape_from_tensor_type(tt),
                "dtype": tensor_dtype(tt.get_elem_type()),
            });
        }
    }
    json!({"shape": ["?"], "dtype": "f32"})
}

fn build_value_info_map(
    graph: &rlx_onnx_proto::onnx::GraphProto,
) -> HashMap<String, serde_json::Value> {
    let mut map = HashMap::new();
    for v in graph.get_input() {
        map.insert(v.get_name().to_string(), output_meta_from_value_info(v));
    }
    for v in graph.get_output() {
        map.insert(v.get_name().to_string(), output_meta_from_value_info(v));
    }
    for v in graph.get_value_info() {
        map.insert(v.get_name().to_string(), output_meta_from_value_info(v));
    }
    for init in graph.get_initializer() {
        let shape: Vec<serde_json::Value> =
            init.get_dims().iter().map(|&d| json!(d.max(1))).collect();
        map.insert(
            init.get_name().to_string(),
            json!({"shape": shape, "dtype": tensor_dtype(init.get_data_type())}),
        );
    }
    map
}

fn io_meta_from_value_info(v: &rlx_onnx_proto::onnx::ValueInfoProto) -> IoMeta {
    let meta = output_meta_from_value_info(v);
    let shape = meta
        .get("shape")
        .and_then(|s| s.as_array())
        .cloned()
        .unwrap_or_else(|| vec![json!("?")]);
    let dtype = meta
        .get("dtype")
        .and_then(|d| d.as_str())
        .unwrap_or("f32")
        .to_string();
    IoMeta {
        name: v.get_name().to_string(),
        meta: TensorMeta { shape, dtype },
    }
}

/// Parse ONNX and return manifest + graph pieces (no HIR lowering).
/// Inline ONNX external-data tensors. `external_data` (proto field 13, repeated
/// `StringStringEntryProto` with keys `location`/`offset`/`length`) and
/// `data_location` (field 14, `EXTERNAL == 1`) are unknown to the pinned proto
/// crate, so they arrive in each tensor's `unknown_fields`. For every initializer
/// that carries them we read `<onnx_dir>/<location>[offset .. offset+length]` and
/// set it as `raw_data`, so the tensor becomes indistinguishable from an inline one.
fn resolve_external_initializers(model: &mut ModelProto, onnx_path: &Path) -> Result<()> {
    use rlx_onnx_proto::onnx::StringStringEntryProto;
    const EXTERNAL_DATA_FIELD: u32 = 13;
    const DATA_LOCATION_FIELD: u32 = 14;
    let dir = onnx_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| Path::new(".").to_path_buf());
    let mut cache: HashMap<String, Vec<u8>> = HashMap::new();
    for init in model.mut_graph().mut_initializer().iter_mut() {
        if !init.get_raw_data().is_empty() {
            continue; // already inline
        }
        let uf = init.get_unknown_fields();
        let entries: Vec<Vec<u8>> = uf
            .get(EXTERNAL_DATA_FIELD)
            .map(|v| v.length_delimited.clone())
            .unwrap_or_default();
        let is_external = uf
            .get(DATA_LOCATION_FIELD)
            .and_then(|v| v.varint.first().copied())
            == Some(1)
            || !entries.is_empty();
        if !is_external {
            continue;
        }
        let (mut location, mut offset, mut length) = (String::new(), 0usize, None::<usize>);
        for e in &entries {
            if let Ok(entry) = protobuf::parse_from_bytes::<StringStringEntryProto>(e) {
                match entry.get_key() {
                    "location" => location = entry.get_value().to_string(),
                    "offset" => offset = entry.get_value().trim().parse().unwrap_or(0),
                    "length" => length = entry.get_value().trim().parse().ok(),
                    _ => {}
                }
            }
        }
        if location.is_empty() {
            continue;
        }
        if !cache.contains_key(&location) {
            let data = std::fs::read(dir.join(&location))
                .with_context(|| format!("read external data file {location}"))?;
            cache.insert(location.clone(), data);
        }
        let data = &cache[&location];
        let end = length
            .map(|l| offset.saturating_add(l))
            .unwrap_or(data.len());
        let slice = data
            .get(offset..end.min(data.len()))
            .unwrap_or(&[])
            .to_vec();
        anyhow::ensure!(
            !slice.is_empty(),
            "external initializer {} resolved to empty ({location}[{offset}..{:?}])",
            init.get_name(),
            length
        );
        init.set_raw_data(slice);
    }
    Ok(())
}

pub fn prepare_onnx_file(
    path: &Path,
) -> Result<(
    BundleManifest,
    Vec<BundleNode>,
    HashMap<String, Vec<f32>>,
    HashMap<String, Vec<i64>>,
    HashMap<String, Vec<usize>>,
)> {
    scalar_consts_clear();
    if_branches_clear();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut model = ModelProto::new();
    model
        .merge_from_bytes(&bytes)
        .context("parse ONNX protobuf")?;
    // Resolve ONNX EXTERNAL-data initializers (weights stored in a sibling `.data`
    // file at per-tensor offsets — the >2 GB LM/DiT export convention). The pinned
    // `onnx` proto crate predates external data, so `external_data` (field 13) and
    // `data_location` (field 14) land in `unknown_fields`; read them there, slice the
    // referenced file, and inline the bytes as `raw_data` so the normal initializer
    // path sees ordinary tensors. Without this every external weight imports empty →
    // "unsupported dtype". Needed by MOSS-TTS-Nano / ChatterBox / any big split-weight
    // export.
    resolve_external_initializers(&mut model, path)
        .with_context(|| format!("resolve external data for {}", path.display()))?;
    let graph = model.get_graph();

    let mut params = HashMap::new();
    let mut i64_params = HashMap::new();
    for init in graph.get_initializer() {
        let name = init.get_name().to_string();
        // Rank-0 scalar initializers (empty `dims`) must be tracked like folded
        // Constant scalars so Gather can drop the gathered axis. ChatterBox
        // speech_encoder's `/Gather_2(audio, /Constant_1=0)` is the canonical case:
        // without the squeeze, Shape sees `[1, L]` and n_frames becomes
        // `(1-400)/160+1 = -1`, collapsing the STFT framing mask.
        if init.get_dims().is_empty() {
            SCALAR_CONSTS.with(|c| {
                c.borrow_mut().insert(name.clone());
            });
        }
        match init.get_data_type() as i32 {
            6 | 7 | 9 => {
                i64_params.insert(name.clone(), tensor_to_i64(&name, init)?);
            }
            _ => {
                params.insert(name.clone(), tensor_to_f32(&name, init)?);
            }
        }
    }

    // Constant folding: `Constant` op nodes carry their tensor in a `value`
    // (or `value_float`/`value_int`/`value_floats`/`value_ints`) attribute.
    // PyTorch exports emit these for scalars and small tensors, so materialize
    // them into params/i64_params exactly like initializers, then drop the node.
    let mut const_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut folded_constants: HashSet<String> = HashSet::new();
    for n in graph.get_node() {
        fold_constant_node(
            n,
            &mut params,
            &mut i64_params,
            &mut const_shapes,
            &mut folded_constants,
        );
    }
    // Also fold `Constant`s inside `If`-branch subgraphs. `lower_if` resolves the
    // condition at compile time and emits the taken branch's output; when that
    // output is a cached constant table (e.g. the Zipformer relative-position
    // embedding, a `then`-branch `Constant`), its data must be a param. Subgraph
    // tensor names are globally unique, so they share the top-level param maps.
    for n in graph.get_node() {
        if n.get_op_type() != "If" {
            continue;
        }
        for a in n.get_attribute() {
            if matches!(a.get_field_type(), AttributeProto_AttributeType::GRAPH) {
                for sn in a.get_g().get_node() {
                    fold_constant_node(
                        sn,
                        &mut params,
                        &mut i64_params,
                        &mut const_shapes,
                        &mut folded_constants,
                    );
                }
            }
        }
    }

    let inputs: Vec<IoMeta> = graph
        .get_input()
        .iter()
        .filter(|i| !params.contains_key(i.get_name()))
        .map(io_meta_from_value_info)
        .collect();

    let outputs: Vec<IoMeta> = graph
        .get_output()
        .iter()
        .map(io_meta_from_value_info)
        .collect();

    let value_info = build_value_info_map(graph);

    let is_folded = |n: &rlx_onnx_proto::onnx::NodeProto| -> bool {
        n.get_op_type() == "Constant"
            && !n.get_output().is_empty()
            && folded_constants.contains(n.get_output()[0].as_str())
    };
    // Build one `BundleNode` from an ONNX node — reused for the top-level graph AND
    // for `If`-branch subgraph nodes (so `lower_if` can inline a computed branch).
    let build_node = |n: &rlx_onnx_proto::onnx::NodeProto| -> BundleNode {
        let mut attrs = HashMap::new();
        for a in n.get_attribute() {
            if let Some(v) = parse_attribute(a) {
                attrs.insert(a.get_name().to_string(), v);
            }
        }
        // For `If`, record each branch's output tensor names so `lower_if` can map
        // the taken branch's outputs to the `If`'s outputs.
        if n.get_op_type() == "If" {
            for a in n.get_attribute() {
                let key = match a.get_name() {
                    "then_branch" => "_then_outputs",
                    "else_branch" => "_else_outputs",
                    _ => continue,
                };
                if matches!(a.get_field_type(), AttributeProto_AttributeType::GRAPH) {
                    let outs: Vec<String> = a
                        .get_g()
                        .get_output()
                        .iter()
                        .map(|o| o.get_name().to_string())
                        .collect();
                    attrs.insert(key.to_string(), json!(outs));
                }
            }
        }
        let name = if n.get_name().is_empty() {
            n.get_op_type().to_string()
        } else {
            n.get_name().to_string()
        };
        let output_meta: Vec<serde_json::Value> = n
            .get_output()
            .iter()
            .map(|out| {
                value_info
                    .get(out)
                    .cloned()
                    .unwrap_or_else(|| json!({"shape": ["?"], "dtype": "f32"}))
            })
            .collect();
        let output_meta = if output_meta.is_empty() {
            vec![json!({"shape": ["?"], "dtype": "f32"})]
        } else {
            output_meta
        };
        BundleNode {
            name,
            op: n.get_op_type().to_string(),
            inputs: n.get_input().to_vec(),
            outputs: n.get_output().to_vec(),
            attrs,
            output_meta,
        }
    };

    let mut nodes: Vec<BundleNode> = graph
        .get_node()
        .iter()
        .filter(|n| !is_folded(n))
        .map(&build_node)
        .collect();

    // Capture each `If`'s branch subgraph nodes (minus already-folded Constants) so
    // `lower_if` can INLINE the taken branch when its output is computed, not a
    // folded constant (MOSS codec's `If(shape[1]==1, squeeze(wave), wave)`).
    for n in graph.get_node() {
        if n.get_op_type() != "If" {
            continue;
        }
        let mut then_nodes = Vec::new();
        let mut else_nodes = Vec::new();
        for a in n.get_attribute() {
            let target = match a.get_name() {
                "then_branch" => &mut then_nodes,
                "else_branch" => &mut else_nodes,
                _ => continue,
            };
            if matches!(a.get_field_type(), AttributeProto_AttributeType::GRAPH) {
                for sn in a.get_g().get_node() {
                    if !is_folded(sn) {
                        target.push(build_node(sn));
                    }
                }
            }
        }
        let if_name = if n.get_name().is_empty() {
            "If".to_string()
        } else {
            n.get_name().to_string()
        };
        IF_BRANCHES.with(|c| c.borrow_mut().insert(if_name, (then_nodes, else_nodes)));
    }

    let manifest = BundleManifest {
        source_onnx: path.display().to_string(),
        inputs,
        outputs,
        node_count: nodes.len(),
        initializer_count: params.len(),
        op_histogram: nodes.iter().fold(HashMap::new(), |mut m, n| {
            *m.entry(n.op.clone()).or_insert(0) += 1;
            m
        }),
    };

    let mut init_shapes = HashMap::new();
    for init in graph.get_initializer() {
        init_shapes.insert(
            init.get_name().to_string(),
            init.get_dims().iter().map(|&d| d.max(1) as usize).collect(),
        );
    }
    init_shapes.extend(const_shapes);

    let opts = ImportOptions::default();
    propagate_shapes(&mut nodes, &manifest, &init_shapes, &opts);

    Ok((manifest, nodes, params, i64_params, init_shapes))
}

/// Load ONNX from disk and lower to HIR + f32 params.
pub fn build_hir_from_onnx_file(
    path: &Path,
    opts: ImportOptions,
) -> Result<(
    rlx_ir::hir::HirModule,
    HashMap<String, Vec<f32>>,
    ImportReport,
    BundleManifest,
)> {
    let (manifest, nodes, params, i64_params, init_shapes) = prepare_onnx_file(path)?;
    let (hir, params, _typed, report) = build_hir_from_parts(
        &manifest,
        nodes,
        params,
        crate::tensor_data::TypedParams::new(),
        i64_params,
        &init_shapes,
        opts,
    )?;
    Ok((hir, params, report, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn onnx_test_model() -> Option<std::path::PathBuf> {
        std::env::var("RLX_ONNX_TEST_MODEL")
            .ok()
            .map(std::path::PathBuf::from)
            .filter(|p| p.exists())
    }

    #[test]
    fn onnx_matmul_integer_rewritten() {
        let Some(path) = onnx_test_model() else {
            return;
        };
        let (manifest, nodes, params, _i64, init_shapes) =
            prepare_onnx_file(&path).expect("prepare");
        let opts = crate::ImportOptions::quant_bundle();
        let rewritten = crate::rewrite::rewrite_graph(
            nodes,
            &params,
            &init_shapes,
            &manifest,
            &opts,
            &std::collections::HashSet::new(),
        );
        let remaining = rewritten
            .nodes
            .iter()
            .filter(|n| n.op == "MatMulInteger")
            .count();
        assert_eq!(
            remaining, 0,
            "expected all MatMulInteger rewritten, {remaining} remain"
        );
    }

    #[test]
    fn onnx_prepare_infers_shapes() {
        let Some(path) = onnx_test_model() else {
            return;
        };
        let (manifest, nodes, ..) = prepare_onnx_file(&path).expect("prepare");
        assert_eq!(manifest.node_count, nodes.len());
        let with_shapes = nodes
            .iter()
            .filter(|n| {
                n.output_meta.first().is_some_and(|m| {
                    m.get("shape")
                        .and_then(|s| s.as_array())
                        .map(|a| {
                            !a.is_empty()
                                && a.iter()
                                    .any(|d| d.as_u64().is_some() || d.as_str().is_some())
                        })
                        .unwrap_or(false)
                })
            })
            .count();
        assert!(
            with_shapes > nodes.len() / 4,
            "expected shape propagation on raw ONNX, got {with_shapes}/{}",
            nodes.len()
        );
    }

    #[test]
    fn onnx_prepare_loads_i64_constants() {
        let Some(path) = onnx_test_model() else {
            return;
        };
        let (manifest, nodes, params, i64_params, init_shapes) =
            prepare_onnx_file(&path).expect("prepare onnx");
        assert_eq!(manifest.node_count, nodes.len());
        assert!(!params.is_empty());
        assert!(!i64_params.is_empty(), "expected i64 constant initializers");
        assert!(init_shapes.len() >= params.len() + i64_params.len());
    }

    #[test]
    fn scalar_initializer_registered_for_gather_squeeze() {
        // ChatterBox speech_encoder (and similar Whisper-style frontends) use a
        // rank-0 i64 initializer as a Gather index. Those must land in
        // `take_scalar_consts` so lowering drops the gathered axis — otherwise
        // STFT framing collapses (`binary_infer at /Sub_2`).
        let candidates = [
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../../rlx-models/weights/tts/chatterbox/onnx/speech_encoder.onnx"),
            PathBuf::from(
                "/Users/Shared/rlx-models/weights/tts/chatterbox/onnx/speech_encoder.onnx",
            ),
        ];
        let Some(path) = candidates.into_iter().find(|p| p.is_file()) else {
            eprintln!("skip: chatterbox speech_encoder.onnx not present");
            return;
        };
        let mut named = std::collections::HashMap::new();
        named.insert("batch_size".into(), 1usize);
        named.insert("num_samples".into(), 48_000);
        let opts = ImportOptions {
            sequence_length: 100,
            named_lengths: named,
            max_waveform_samples: 48_000,
            strict: false,
            ..Default::default()
        };
        let (hir, _params, report, manifest) =
            build_hir_from_onnx_file(&path, opts).expect("import speech_encoder");
        assert_eq!(report.stubbed, 0, "stubbed={}", report.stubbed);
        assert!(!hir.nodes().is_empty());
        assert_eq!(manifest.outputs.len(), 4);
    }

    #[test]
    fn micro_bench_onnx_initializers_have_raw_data() {
        let path = std::path::Path::new("/tmp/bench_micro.onnx");
        if !path.exists() {
            eprintln!("skip: {}", path.display());
            return;
        }
        let (.., params, _i64, init_shapes) = prepare_onnx_file(path).expect("prepare");
        assert!(params.contains_key("w"));
        assert!(params.contains_key("b"));
        assert!(init_shapes.contains_key("w"));
    }
}
