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

//! `rlx-ir` [`Graph`] → [`crate::model::Model`] lowering.
//!
//! Recognizes a single-output chain of FPGA-supported ops:
//! [`Op::QConv2d`], [`Op::Activation`] (ReLU), [`Op::Pool`] (max),
//! [`Op::QMatMul`], [`Op::TopK`] (k=1 / argmax), plus optional
//! [`Op::Reshape`] / [`Op::Squeeze`] passthroughs.
//!
//! Weights / biases / per-channel requant tables come from:
//! 1. [`Op::Constant`] inputs baked by [`crate::ir::to_graph`], or
//! 2. Overrides in [`FromGraphOptions`].

use std::collections::HashMap;

use rlx_ir::op::{Activation, ReduceOp};
use rlx_ir::{Dim, Graph, NodeId, Op, Shape};

use crate::export_config::GraphIoBind;
use crate::model::{ExtraOutput, Layer, Model};
use crate::quant::quantize_multiplier;

/// Side-table / override bindings for Graph → Model lowering.
#[derive(Debug, Clone)]
pub struct FromGraphOptions {
    /// Default packed weight bit-width when not inferable (`2`, `4`, or `8`).
    pub weight_bits: u8,
    pub weight_encoding: crate::model::WeightEncoding,
    /// Param / constant name → packed i8 weight bytes.
    pub weights: HashMap<String, Vec<i8>>,
    /// Param / constant name → i32 bias.
    pub biases: HashMap<String, Vec<i32>>,
    /// Layer stem (e.g. `"conv1"`) → per-channel f32 multipliers.
    pub mults: HashMap<String, Vec<f32>>,
    /// Layer stem → per-channel Q0.31 `(M0, shift)` (takes precedence over `mults`).
    pub requant: HashMap<String, Vec<(i32, i32)>>,
    /// Which Graph input / outputs feed the soft datapath ports.
    pub bind: GraphIoBind,
}

impl Default for FromGraphOptions {
    fn default() -> Self {
        Self {
            weight_bits: 8,
            weight_encoding: crate::model::WeightEncoding::SignedInt,
            weights: HashMap::new(),
            biases: HashMap::new(),
            mults: HashMap::new(),
            requant: HashMap::new(),
            bind: GraphIoBind::default(),
        }
    }
}

impl FromGraphOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build overrides that reproduce `model`'s weights / biases / requant.
    pub fn from_model(model: &Model) -> Self {
        let mut opts = Self::new();
        for layer in &model.layers {
            match layer {
                Layer::Conv2d {
                    name,
                    weight_bits,
                    weights,
                    bias,
                    requant,
                    ..
                }
                | Layer::Dense {
                    name,
                    weight_bits,
                    weights,
                    bias,
                    requant,
                    ..
                } => {
                    opts.weight_bits = *weight_bits;
                    opts.weights.insert(format!("{name}_w"), weights.clone());
                    if let Some(b) = bias {
                        opts.biases.insert(format!("{name}_b"), b.clone());
                    }
                    opts.requant.insert((*name).to_string(), requant.clone());
                }
                _ => {}
            }
        }
        opts
    }
}

impl Model {
    /// Recognize an FPGA-supported subgraph in `g`.
    ///
    /// Equivalent to [`Self::from_graph_opts`] with default options.
    pub fn from_graph(g: &Graph) -> Result<Self, String> {
        Self::from_graph_opts(g, &FromGraphOptions::default())
    }

    /// Lower `g` with optional weight / requant overrides.
    pub fn from_graph_opts(g: &Graph, opts: &FromGraphOptions) -> Result<Self, String> {
        // Reject distributed collectives early (same posture as cerebras/qnn).
        if let Some(name) = g.nodes().iter().find_map(|n| match &n.op {
            Op::Custom { name, .. } if name.starts_with("collective.") => Some(name.clone()),
            _ => None,
        }) {
            return Err(format!(
                "rlx-fpga: '{name}' is a distributed host/transport collective \
                 — it cannot be expressed in a Verilog datapath. Export the \
                 single-device inference subgraph only."
            ));
        }

        if g.outputs.is_empty() {
            return Err("rlx-fpga: graph has no outputs".into());
        }

        let input_id = resolve_input_id(g, &opts.bind)?;
        let primary_out = resolve_primary_output(g, &opts.bind)?;

        // Reject unbound activation Inputs (multi-input graphs need an explicit bind).
        // Sideband-bound scalar Inputs are excluded — they are soft ports, not datapath.
        let sideband: std::collections::HashSet<&str> = opts
            .bind
            .sideband_inputs
            .iter()
            .map(|s| s.as_str())
            .collect();
        let activation_inputs: Vec<_> = g
            .nodes()
            .iter()
            .filter(|n| match &n.op {
                Op::Input { name } => !sideband.contains(name.as_str()),
                _ => false,
            })
            .collect();
        if activation_inputs.len() > 1 && opts.bind.input.is_none() {
            let names: Vec<_> = activation_inputs
                .iter()
                .filter_map(|n| match &n.op {
                    Op::Input { name } => Some(name.as_str()),
                    _ => None,
                })
                .collect();
            return Err(format!(
                "rlx-fpga: graph has multiple activation Op::Input nodes {names:?}; \
                 set IoConfig.bind.input for the image/activation tensor, and list \
                 scalars in bind.sideband_inputs (or IoConfig.sidebands)"
            ));
        }

        let input_shape = g.shape(input_id);
        let input_len = input_shape.num_elements().unwrap_or(0);
        if input_len == 0 {
            return Err("rlx-fpga: input shape has zero elements".into());
        }

        let chain = collect_compute_chain(g, input_id, primary_out)?;
        let mut layers = Vec::new();
        let mut name_slots: Vec<String> = Vec::new();

        for &nid in &chain {
            let node = g.node(nid);
            let layer_name = intern_name(suggest_layer_name(g, nid, layers.len()));
            name_slots.push(layer_name.to_string());

            let layer = match &node.op {
                Op::QConv2d {
                    kernel_size,
                    stride,
                    padding,
                    x_zp,
                    w_zp,
                    out_zp,
                    mult,
                    ..
                } => {
                    if node.inputs.len() < 3 {
                        return Err(format!(
                            "QConv2d {layer_name} expects ≥3 inputs (x, w, bias); got {}",
                            node.inputs.len()
                        ));
                    }
                    let x_shape = spatial_nchw(g, node.inputs[0], "QConv2d activation")?;
                    let (kh, kw) = pair2(kernel_size, "kernel_size")?;
                    let (stride_h, stride_w) = pair2(stride, "stride")?;
                    let (pad_h, pad_w) = pair2(padding, "padding")?;
                    let w_data =
                        load_i8_tensor(g, node.inputs[1], &format!("{layer_name}_w"), opts)?;
                    let bias =
                        load_i32_tensor_opt(g, node.inputs[2], &format!("{layer_name}_b"), opts)?;
                    let w_shape = g.shape(node.inputs[1]);
                    let c_out = static_dim(w_shape, 0, "QConv2d weight C_out")?;
                    let c_in = static_dim(w_shape, 1, "QConv2d weight C_in")?;
                    let weight_bits =
                        infer_weight_bits(w_data.len(), c_out * c_in * kh * kw, opts.weight_bits);
                    let requant = load_requant(g, layer_name, c_out, *mult, opts)?;
                    let weight_encoding = load_weight_encoding(g, layer_name, opts);
                    Layer::Conv2d {
                        name: layer_name,
                        h_in: x_shape.h,
                        w_in: x_shape.w,
                        c_in,
                        c_out,
                        kh,
                        kw,
                        pad_h,
                        pad_w,
                        stride_h,
                        stride_w,
                        x_zp: *x_zp,
                        w_zp: *w_zp,
                        out_zp: *out_zp,
                        weight_bits,
                        weight_encoding,
                        requant,
                        weights: w_data,
                        bias,
                    }
                }
                Op::Activation(Activation::Relu) => {
                    let len = g
                        .shape(node.inputs[0])
                        .num_elements()
                        .unwrap_or(0)
                        .max(g.shape(nid).num_elements().unwrap_or(0));
                    Layer::Relu {
                        name: layer_name,
                        len,
                        zero_point: 0,
                    }
                }
                Op::Pool {
                    kind: ReduceOp::Max,
                    kernel_size,
                    stride,
                    ..
                } => {
                    let x = spatial_nchw(g, node.inputs[0], "MaxPool activation")?;
                    let (kh, kw) = pair2(kernel_size, "pool kernel")?;
                    let (stride_h, stride_w) = pair2(stride, "pool stride")?;
                    Layer::MaxPool2d {
                        name: layer_name,
                        h_in: x.h,
                        w_in: x.w,
                        c: x.c,
                        kh,
                        kw,
                        stride_h,
                        stride_w,
                    }
                }
                Op::QMatMul {
                    x_zp,
                    w_zp,
                    out_zp,
                    mult,
                } => {
                    if node.inputs.len() < 2 {
                        return Err(format!(
                            "QMatMul {layer_name} expects ≥2 inputs (x, w[, bias]); got {}",
                            node.inputs.len()
                        ));
                    }
                    let w_data =
                        load_i8_tensor(g, node.inputs[1], &format!("{layer_name}_w"), opts)?;
                    let bias = if node.inputs.len() >= 3 {
                        load_i32_tensor_opt(g, node.inputs[2], &format!("{layer_name}_b"), opts)?
                    } else {
                        None
                    };
                    // Treat all-zero bias Constant as absent.
                    let bias = match bias {
                        Some(b) if b.iter().all(|&x| x == 0) => None,
                        other => other,
                    };
                    let w_shape = g.shape(node.inputs[1]);
                    let in_features = static_dim(w_shape, 0, "QMatMul in_features")?;
                    let out_features = static_dim(w_shape, 1, "QMatMul out_features")?;
                    let weight_bits = infer_weight_bits(
                        w_data.len(),
                        in_features * out_features,
                        opts.weight_bits,
                    );
                    let requant = load_requant(g, layer_name, out_features, *mult, opts)?;
                    let weight_encoding = load_weight_encoding(g, layer_name, opts);
                    Layer::Dense {
                        name: layer_name,
                        in_features,
                        out_features,
                        x_zp: *x_zp,
                        w_zp: *w_zp,
                        out_zp: *out_zp,
                        weight_bits,
                        weight_encoding,
                        requant,
                        weights: w_data,
                        bias,
                    }
                }
                Op::TopK { k } if *k == 1 => {
                    let len = g
                        .shape(node.inputs[0])
                        .num_elements()
                        .ok_or_else(|| "TopK input has dynamic numel".to_string())?;
                    Layer::Argmax {
                        name: layer_name,
                        len,
                    }
                }
                Op::Reshape { .. } => {
                    // Geometry passthrough — not a hardware layer.
                    continue;
                }
                other => {
                    // Clear guidance when the chain still looks like f32 training IR.
                    if matches!(
                        other,
                        Op::MatMul | Op::Conv { .. } | Op::ScaledMatMul { .. }
                    ) || (g.shape(nid).dtype() == rlx_ir::DType::F32
                        && matches!(other, Op::Activation(_) | Op::Pool { .. }))
                    {
                        return Err("rlx-fpga: graph still has f32 / unquantized compute ops. \
                             Quantize first (ExportQuantMode::Int8 | Int4 | Fp4 via \
                             prepare_model / ExportSession), or lower to QConv2d / QMatMul."
                            .into());
                    }
                    return Err(format!(
                        "rlx-fpga: unsupported op in export chain: {other:?}. \
                         Supported: QConv2d, Relu, MaxPool, QMatMul, TopK(k=1), Reshape."
                    ));
                }
            };
            layers.push(layer);
        }

        if layers.is_empty() {
            return Err("rlx-fpga: no supported compute layers found in graph".into());
        }

        let extra_outputs = resolve_extra_outputs(g, &opts.bind, &layers, &name_slots, &chain)?;

        Ok(Model {
            name: g.name.clone(),
            input_len,
            layers,
            extra_outputs,
        })
    }
}

fn resolve_input_id(g: &Graph, bind: &GraphIoBind) -> Result<NodeId, String> {
    if let Some(name) = &bind.input {
        return g
            .input_id(name)
            .ok_or_else(|| format!("rlx-fpga: bind.input {name:?} does not match any Op::Input"));
    }
    g.nodes()
        .iter()
        .find(|n| matches!(n.op, Op::Input { .. }))
        .map(|n| n.id)
        .ok_or_else(|| "rlx-fpga: graph has no Op::Input".to_string())
}

fn resolve_primary_output(g: &Graph, bind: &GraphIoBind) -> Result<NodeId, String> {
    if let Some(name) = bind.outputs.first() {
        return find_named_node(g, name).ok_or_else(|| {
            format!("rlx-fpga: bind.outputs[0]={name:?} not found as a graph node/output")
        });
    }
    g.outputs
        .first()
        .copied()
        .ok_or_else(|| "rlx-fpga: graph has no outputs".to_string())
}

fn find_named_node(g: &Graph, name: &str) -> Option<NodeId> {
    if let Some(id) = g.node_id_by_name(name) {
        return Some(id);
    }
    // Match against graph output nodes by stored Node::name or op-derived label.
    for &oid in &g.outputs {
        let n = g.node(oid);
        if n.name.as_deref() == Some(name) {
            return Some(oid);
        }
        if let Op::Input { name: nm } | Op::Param { name: nm } = &n.op {
            if nm == name {
                return Some(oid);
            }
        }
    }
    g.nodes().iter().find_map(|n| {
        if n.name.as_deref() == Some(name) {
            Some(n.id)
        } else {
            None
        }
    })
}

fn resolve_extra_outputs(
    g: &Graph,
    bind: &GraphIoBind,
    layers: &[Layer],
    name_slots: &[String],
    chain: &[NodeId],
) -> Result<Vec<ExtraOutput>, String> {
    if bind.outputs.len() <= 1 {
        return Ok(Vec::new());
    }
    let mut extras = Vec::new();
    for name in bind.outputs.iter().skip(1) {
        // Prefer layer-name match on the compute chain.
        if let Some(idx) = name_slots.iter().position(|n| n == name) {
            extras.push(ExtraOutput {
                name: name.clone(),
                len: layers[idx].out_len(),
                after_layer: idx,
            });
            continue;
        }
        let nid = find_named_node(g, name).ok_or_else(|| {
            format!(
                "rlx-fpga: extra bind.output {name:?} is not on the compute chain \
                 (layer names: {name_slots:?})"
            )
        })?;
        let Some(chain_idx) = chain.iter().position(|&id| id == nid) else {
            return Err(format!(
                "rlx-fpga: extra bind.output {name:?} is not on the single compute chain \
                 from the bound input to the primary output"
            ));
        };
        extras.push(ExtraOutput {
            name: name.clone(),
            len: layers[chain_idx].out_len(),
            after_layer: chain_idx,
        });
    }
    Ok(extras)
}

// ── helpers ──────────────────────────────────────────────────────────

struct Nchw {
    #[allow(dead_code)]
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

fn spatial_nchw(g: &Graph, id: NodeId, what: &str) -> Result<Nchw, String> {
    let s = g.shape(id);
    match s.rank() {
        4 => Ok(Nchw {
            n: static_dim(s, 0, &format!("{what} N"))?,
            c: static_dim(s, 1, &format!("{what} C"))?,
            h: static_dim(s, 2, &format!("{what} H"))?,
            w: static_dim(s, 3, &format!("{what} W"))?,
        }),
        1 => {
            // Flat activation (Dense path / legacy). Infer a 1×1×1×L stand-in.
            let len = s.num_elements().unwrap_or(0);
            Ok(Nchw {
                n: 1,
                c: 1,
                h: 1,
                w: len,
            })
        }
        r => Err(format!(
            "rlx-fpga: {what} expected rank-4 NCHW or rank-1 flat; got rank {r}"
        )),
    }
}

fn pair2(v: &[usize], what: &str) -> Result<(usize, usize), String> {
    match v {
        [a, b] => Ok((*a, *b)),
        [a] => Ok((*a, *a)),
        _ => Err(format!(
            "rlx-fpga: {what} must have 1 or 2 entries; got {v:?}"
        )),
    }
}

fn static_dim(s: &Shape, i: usize, what: &str) -> Result<usize, String> {
    if i >= s.rank() {
        return Err(format!(
            "rlx-fpga: {what}: rank {} missing dim {i}",
            s.rank()
        ));
    }
    match s.dim(i) {
        Dim::Static(n) => Ok(n),
        Dim::Dynamic(sym) => Err(format!(
            "rlx-fpga: {what} is dynamic (symbol {sym}); FPGA export needs static shapes"
        )),
    }
}

fn infer_weight_bits(packed_len: usize, logical_n: usize, default: u8) -> u8 {
    if logical_n == 0 {
        return default;
    }
    if packed_len == logical_n {
        8
    } else if packed_len == logical_n.div_ceil(2) {
        4
    } else if packed_len == logical_n.div_ceil(4) {
        2
    } else {
        default
    }
}

fn load_i8_tensor(
    g: &Graph,
    id: NodeId,
    key: &str,
    opts: &FromGraphOptions,
) -> Result<Vec<i8>, String> {
    if let Some(v) = opts.weights.get(key) {
        return Ok(v.clone());
    }
    match &g.node(id).op {
        Op::Constant { data } => Ok(data.iter().map(|&b| b as i8).collect()),
        Op::Param { name } => opts.weights.get(name).cloned().ok_or_else(|| {
            format!(
                "rlx-fpga: weight param '{name}' has no Constant data and no \
                     FromGraphOptions.weights[{key:?}] override"
            )
        }),
        other => Err(format!(
            "rlx-fpga: weight input for {key} is {other:?}; expected Constant or Param"
        )),
    }
}

fn load_i32_tensor_opt(
    g: &Graph,
    id: NodeId,
    key: &str,
    opts: &FromGraphOptions,
) -> Result<Option<Vec<i32>>, String> {
    if let Some(v) = opts.biases.get(key) {
        return Ok(Some(v.clone()));
    }
    match &g.node(id).op {
        Op::Constant { data } => {
            if data.is_empty() {
                return Ok(None);
            }
            if data.len() % 4 != 0 {
                return Err(format!(
                    "rlx-fpga: bias Constant for {key} has {} bytes (not multiple of 4)",
                    data.len()
                ));
            }
            let v: Vec<i32> = data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(Some(v))
        }
        Op::Param { name } => Ok(opts.biases.get(name).cloned()),
        _ => Ok(None),
    }
}

fn load_requant(
    g: &Graph,
    layer_name: &str,
    channels: usize,
    scalar_mult: f32,
    opts: &FromGraphOptions,
) -> Result<Vec<(i32, i32)>, String> {
    if let Some(r) = opts.requant.get(layer_name) {
        return Ok(r.clone());
    }
    if let Some(m) = opts.mults.get(layer_name) {
        return Ok(m.iter().map(|&x| quantize_multiplier(x)).collect());
    }
    // Prefer split side tables `{name}_requant_m0` + `{name}_requant_shift`.
    let m0_key = format!("{layer_name}_requant_m0");
    let sh_key = format!("{layer_name}_requant_shift");
    if let (Some(m0_id), Some(sh_id)) = (
        find_named_constant(g, &m0_key),
        find_named_constant(g, &sh_key),
    ) {
        let m0s = load_i32_const(g, m0_id, &m0_key)?;
        let shs = load_i32_const(g, sh_id, &sh_key)?;
        if m0s.len() != shs.len() {
            return Err(format!(
                "rlx-fpga: {m0_key} len {} != {sh_key} len {}",
                m0s.len(),
                shs.len()
            ));
        }
        return Ok(m0s.into_iter().zip(shs).collect());
    }
    // Interleaved `{name}_requant` Constant.
    let key = format!("{layer_name}_requant");
    if let Some(nid) = find_named_constant(g, &key) {
        if let Op::Constant { data } = &g.node(nid).op {
            if data.len() % 8 != 0 {
                return Err(format!(
                    "rlx-fpga: {key} Constant length {} not multiple of 8",
                    data.len()
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 8);
            for chunk in data.chunks_exact(8) {
                let m0 = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let sh = i32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                out.push((m0, sh));
            }
            return Ok(out);
        }
    }
    let pair = quantize_multiplier(scalar_mult);
    Ok(vec![pair; channels.max(1)])
}

fn load_i32_const(g: &Graph, id: NodeId, key: &str) -> Result<Vec<i32>, String> {
    match &g.node(id).op {
        Op::Constant { data } => {
            if data.len() % 4 != 0 {
                return Err(format!("{key}: byte len not multiple of 4"));
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        _ => Err(format!("{key}: not a Constant")),
    }
}

fn load_weight_encoding(
    g: &Graph,
    layer_name: &str,
    opts: &FromGraphOptions,
) -> crate::model::WeightEncoding {
    let key = format!("{layer_name}_wenc");
    if let Some(nid) = find_named_constant(g, &key) {
        if let Ok(v) = load_i32_const(g, nid, &key) {
            if v.first() == Some(&1) {
                return crate::model::WeightEncoding::Fp4E2M1;
            }
        }
    }
    opts.weight_encoding
}

fn find_named_constant(g: &Graph, name: &str) -> Option<NodeId> {
    g.nodes().iter().find_map(|n| {
        if n.name.as_deref() == Some(name) && matches!(n.op, Op::Constant { .. }) {
            Some(n.id)
        } else {
            None
        }
    })
}

fn suggest_layer_name(g: &Graph, nid: NodeId, idx: usize) -> String {
    let node = g.node(nid);
    if let Some(n) = &node.name {
        return n.clone();
    }
    // Prefer weight constant / param stem: "conv1_w" → "conv1".
    if matches!(node.op, Op::QConv2d { .. } | Op::QMatMul { .. }) && node.inputs.len() >= 2 {
        let w = g.node(node.inputs[1]);
        if let Some(wn) = w.name.as_deref().or(match &w.op {
            Op::Param { name } => Some(name.as_str()),
            _ => None,
        }) {
            if let Some(stem) = wn.strip_suffix("_w") {
                return stem.to_string();
            }
        }
    }
    match &node.op {
        Op::QConv2d { .. } => format!("conv{idx}"),
        Op::Activation { .. } => format!("relu{idx}"),
        Op::Pool { .. } => format!("pool{idx}"),
        Op::QMatMul { .. } => format!("fc{idx}"),
        Op::TopK { .. } => format!("argmax{idx}"),
        _ => format!("layer{idx}"),
    }
}

/// Intern a layer name for `'static` storage in [`Layer`].
fn intern_name(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Walk from `input` to `output`, collecting compute nodes in order.
/// Skips fan-out weight Constants that aren't on the activation spine.
fn collect_compute_chain(g: &Graph, input: NodeId, output: NodeId) -> Result<Vec<NodeId>, String> {
    // Reverse walk from output → input along the activation (first) input,
    // then reverse. Handles the sequential CNN / MLP case.
    let mut rev = Vec::new();
    let mut cur = output;
    let mut guard = 0usize;
    while cur != input {
        guard += 1;
        if guard > g.len() + 2 {
            return Err(
                "rlx-fpga: failed to walk activation chain (cycle or disconnected output)".into(),
            );
        }
        let node = g.node(cur);
        match &node.op {
            Op::Input { .. } => break,
            Op::Param { .. } | Op::Constant { .. } => {
                return Err(format!(
                    "rlx-fpga: activation chain reached leaf {:?} before input",
                    node.op
                ));
            }
            Op::Reshape { .. } | Op::Activation { .. } | Op::Pool { .. } | Op::TopK { .. } => {
                rev.push(cur);
                cur = node
                    .inputs
                    .first()
                    .copied()
                    .ok_or_else(|| format!("rlx-fpga: node {cur} has no inputs"))?;
            }
            Op::QConv2d { .. } | Op::QMatMul { .. } => {
                rev.push(cur);
                cur = node
                    .inputs
                    .first()
                    .copied()
                    .ok_or_else(|| format!("rlx-fpga: node {cur} has no activation input"))?;
            }
            other => {
                return Err(format!(
                    "rlx-fpga: unsupported op on activation chain: {other:?}"
                ));
            }
        }
    }
    rev.reverse();
    // Drop pure reshape nodes from the returned chain? Keep them so the
    // match arm can `continue` — simpler to filter here.
    Ok(rev
        .into_iter()
        .filter(|&id| !matches!(g.node(id).op, Op::Reshape { .. }))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::to_graph;
    use crate::model::tinyconv_mnist_from_cortexm;

    #[test]
    fn round_trip_tinyconv_via_to_graph() {
        let m = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&m);
        let opts = FromGraphOptions::from_model(&m);
        let m2 = Model::from_graph_opts(&ir.graph, &opts).expect("from_graph");
        assert_eq!(m2.input_len, m.input_len);
        assert_eq!(m2.layers.len(), m.layers.len());
        for (a, b) in m.layers.iter().zip(m2.layers.iter()) {
            assert_eq!(a.name(), b.name());
            assert_eq!(a.out_len(), b.out_len());
        }
    }

    #[test]
    fn round_trip_reads_baked_constants() {
        let m = tinyconv_mnist_from_cortexm();
        let ir = to_graph(&m);
        // No overrides — Constants baked by to_graph must suffice.
        let m2 = Model::from_graph(&ir.graph).expect("from_graph without overrides");
        assert_eq!(m2.layers.len(), m.layers.len());
        match (&m.layers[0], &m2.layers[0]) {
            (
                Layer::Conv2d {
                    weights: w0,
                    requant: r0,
                    ..
                },
                Layer::Conv2d {
                    weights: w1,
                    requant: r1,
                    ..
                },
            ) => {
                assert_eq!(w0, w1);
                assert_eq!(r0, r1);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn rejects_collectives() {
        let mut g = Graph::new("bad");
        let x = g.input("x", Shape::new(&[4], rlx_ir::DType::F32));
        let y = g.add_node(
            Op::Custom {
                name: "collective.all_reduce".into(),
                num_inputs: 1,
                attrs: vec![],
            },
            vec![x],
            Shape::new(&[4], rlx_ir::DType::F32),
        );
        g.set_outputs(vec![y]);
        let err = Model::from_graph(&g).unwrap_err();
        assert!(err.contains("collective"), "{err}");
    }
}
