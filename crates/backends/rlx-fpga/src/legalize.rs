// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! fp32 (or mixed) Graph → FPGA-ready INT8/INT4/FP4 quantized [`Model`].
//!
//! FPGA fabric runs integer MAC + Q0.31 requant. This pass:
//! * Accepts already-legal `QConv2d` / `QMatMul` graphs (no-op → `from_graph`).
//! * PTQ-quantizes f32 `Conv` / `MatMul` chains when weight tensors are
//!   available as [`Op::Constant`] or via [`LegalizeOptions`].
//! * [`ExportQuantMode::Fp4`] encodes weights on the F4E2M1 grid, packs
//!   4-bit codes, and sets [`WeightEncoding::Fp4E2M1`] so codegen emits
//!   an FP4→fixed LUT unpack (MAC still integer).

use std::collections::HashMap;

use rlx_ir::op::{Activation, BinaryOp, ReduceOp};
use rlx_ir::{DType, Graph, NodeId, Op, ScaledFormat};

use crate::export_config::GraphIoBind;
use crate::from_graph::FromGraphOptions;
use crate::model::{Layer, Model, WeightEncoding};
use crate::pack::pack;
use crate::quant::quantize_multiplier;

/// Weight / activation quantization mode for FPGA export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExportQuantMode {
    /// Symmetric signed INT8 weights (`weight_bits = 8`).
    #[default]
    Int8,
    /// Symmetric signed INT4 weights, nibble-packed (`weight_bits = 4`).
    Int4,
    /// OCP FP4 E2M1 weight codes, nibble-packed (`WeightEncoding::Fp4E2M1`).
    Fp4,
}

impl ExportQuantMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "int8" | "i8" | "q8" => Ok(Self::Int8),
            "int4" | "i4" | "q4" => Ok(Self::Int4),
            "fp4" | "f4" | "f4e2m1" => Ok(Self::Fp4),
            other => Err(format!(
                "unknown ExportQuantMode {other:?}; expected int8 | int4 | fp4"
            )),
        }
    }

    pub fn weight_bits(self) -> u8 {
        match self {
            Self::Int8 => 8,
            Self::Int4 | Self::Fp4 => 4,
        }
    }

    pub fn encoding(self) -> WeightEncoding {
        match self {
            Self::Int8 | Self::Int4 => WeightEncoding::SignedInt,
            Self::Fp4 => WeightEncoding::Fp4E2M1,
        }
    }
}

/// Inputs for PTQ when the Graph carries `Param` leaves without Constant data.
#[derive(Debug, Clone, Default)]
pub struct LegalizeOptions {
    /// Param / constant name → row-major f32 weights.
    pub weights_f32: HashMap<String, Vec<f32>>,
    /// Param / constant name → f32 bias.
    pub biases_f32: HashMap<String, Vec<f32>>,
    /// Activation tensor scales (name → scale). Missing taps default to
    /// symmetric calibration from a unit max-abs of 1.0 (`scale = 1/127`).
    pub act_scales: HashMap<String, f32>,
    /// Append an Argmax / TopK(k=1) when the graph ends on logits.
    pub append_argmax: bool,
}

impl LegalizeOptions {
    pub fn new() -> Self {
        Self {
            append_argmax: true,
            ..Self::default()
        }
    }
}

/// True when the graph is already a supported INT8 Q* FPGA chain.
pub fn is_fpga_quantized(g: &Graph) -> bool {
    g.nodes()
        .iter()
        .any(|n| matches!(n.op, Op::QConv2d { .. } | Op::QMatMul { .. }))
        && !g
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::MatMul | Op::Conv { .. } | Op::ScaledMatMul { .. }))
}

/// True when the graph looks like an f32 CNN/MLP we can PTQ.
pub fn is_fp32_export_candidate(g: &Graph) -> bool {
    g.nodes().iter().any(|n| {
        matches!(n.op, Op::MatMul | Op::Conv { .. }) && g.shape(n.id).dtype() == DType::F32
    })
}

/// Prepare a [`Model`] for export: pass through Q* graphs, or PTQ f32 graphs.
pub fn prepare_model(
    g: &Graph,
    mode: ExportQuantMode,
    legalize: &LegalizeOptions,
    from_graph: &FromGraphOptions,
) -> Result<Model, String> {
    reject_hopeless(g)?;
    if is_fpga_quantized(g) {
        let mut opts = from_graph.clone();
        if opts.weight_bits == 0 || opts.weight_bits == 8 {
            // Prefer mode when caller set quant mode explicitly via config.
            opts.weight_bits = mode.weight_bits();
        }
        return Model::from_graph_opts(g, &opts);
    }
    if is_fp32_export_candidate(g) {
        return legalize_fp32_to_model(g, mode, legalize, &from_graph.bind);
    }
    Err(unsupported_graph_message(g))
}

fn reject_hopeless(g: &Graph) -> Result<(), String> {
    if let Some(name) = g.nodes().iter().find_map(|n| match &n.op {
        Op::Custom { name, .. } if name.starts_with("collective.") => Some(name.clone()),
        _ => None,
    }) {
        return Err(format!(
            "rlx-fpga: '{name}' cannot be exported — strip collectives first"
        ));
    }
    if g.nodes()
        .iter()
        .any(|n| matches!(n.op, Op::ScaledMatMul { .. }))
    {
        return Err(
            "rlx-fpga: Op::ScaledMatMul is a GPU low-precision GEMM path. \
             For FPGA, export an f32 MatMul/Conv graph and set \
             ExportQuantMode::Fp4 (PTQ onto the F4E2M1 grid → integer MAC), \
             or quantize to QMatMul/QConv2d (INT8/INT4) first."
                .into(),
        );
    }
    Ok(())
}

fn unsupported_graph_message(g: &Graph) -> String {
    let kinds: Vec<String> = g
        .nodes()
        .iter()
        .filter(|n| {
            !matches!(
                n.op,
                Op::Input { .. } | Op::Param { .. } | Op::Constant { .. }
            )
        })
        .take(12)
        .map(|n| format!("{:?}", n.op.kind()))
        .collect();
    format!(
        "rlx-fpga: graph is not export-ready. Expected either:\n\
           (1) INT8 ops: QConv2d / Relu / MaxPool / QMatMul / TopK(k=1), or\n\
           (2) f32 Conv/MatMul chain + LegalizeOptions weight tensors \
             (ExportQuantMode::Int8 | Int4 | Fp4).\n\
         Saw compute ops: [{}]. \
         Tip: train/quantize on CPU, then export; or call \
         legalize_fp32_to_model with weight bindings.",
        kinds.join(", ")
    )
}

/// PTQ an f32 Conv/MatMul inference graph into an FPGA [`Model`].
pub fn legalize_fp32_to_model(
    g: &Graph,
    mode: ExportQuantMode,
    opts: &LegalizeOptions,
    bind: &GraphIoBind,
) -> Result<Model, String> {
    let input_id = if let Some(name) = &bind.input {
        g.input_id(name).ok_or_else(|| {
            format!("rlx-fpga legalize: bind.input {name:?} does not match any Op::Input")
        })?
    } else {
        g.nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Input { .. }))
            .map(|n| n.id)
            .ok_or_else(|| "rlx-fpga legalize: no Op::Input".to_string())?
    };
    let input = g.node(input_id);
    let input_shape = g.shape(input.id);
    let (in_n, in_c, in_h, in_w) = match input_shape.rank() {
        4 => (
            input_shape.dim(0).unwrap_static(),
            input_shape.dim(1).unwrap_static(),
            input_shape.dim(2).unwrap_static(),
            input_shape.dim(3).unwrap_static(),
        ),
        2 => (
            input_shape.dim(0).unwrap_static(),
            1,
            1,
            input_shape.dim(1).unwrap_static(),
        ),
        1 => (1, 1, 1, input_shape.dim(0).unwrap_static()),
        r => {
            return Err(format!(
                "rlx-fpga legalize: input rank {r} not supported (want NCHW or flat)"
            ));
        }
    };
    let _ = in_n;
    let input_len = in_c * in_h * in_w;

    let out_id = if let Some(name) = bind.outputs.first() {
        // Prefer explicit node name / input-param name.
        g.nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(name.as_str()))
            .map(|n| n.id)
            .or_else(|| g.node_id_by_name(name))
            .or_else(|| g.outputs.first().copied())
            .ok_or_else(|| format!("rlx-fpga legalize: bind.outputs[0]={name:?} not found"))?
    } else {
        g.outputs.first().copied().ok_or("no outputs")?
    };
    let chain = walk_activation_chain(g, input.id, out_id)?;

    let x_scale = opts
        .act_scales
        .get("input")
        .or_else(|| opts.act_scales.get("model_input"))
        .copied()
        .unwrap_or(1.0 / 127.0);

    let mut layers = Vec::new();
    let mut h = in_h;
    let mut w = in_w;
    let mut c = in_c;
    let mut prev_scale = x_scale;
    let mut layer_i = 0usize;

    for &nid in &chain {
        let node = g.node(nid);
        match &node.op {
            Op::Conv {
                kernel_size,
                stride,
                padding,
                ..
            } => {
                let (kh, kw) = pair2(kernel_size)?;
                let (sh, sw) = pair2(stride)?;
                let (ph, pw) = pair2(padding)?;
                if node.inputs.len() < 2 {
                    return Err("Conv needs weight input".into());
                }
                let w_name = tensor_name(g, node.inputs[1], layer_i, "conv");
                let w_f32 = load_f32(g, node.inputs[1], &w_name, &opts.weights_f32)?;
                let w_shape = g.shape(node.inputs[1]);
                let c_out = w_shape.dim(0).unwrap_static();
                let c_in = w_shape.dim(1).unwrap_static();
                if c_in != c {
                    return Err(format!(
                        "Conv {w_name}: weight C_in={c_in} != activation C={c}"
                    ));
                }
                let (packed, w_scales, encoding) =
                    quantize_oihw(&w_f32, c_out, c_in, kh, kw, mode)?;
                let bias_f = node
                    .inputs
                    .get(2)
                    .map(|&b| {
                        let bn = tensor_name(g, b, layer_i, "bias");
                        load_f32(g, b, &bn, &opts.biases_f32)
                    })
                    .transpose()?;
                // Optional separate BiasAdd later — fold if present as next Add.
                let out_scale = opts.act_scales.get(&w_name).copied().unwrap_or(1.0 / 127.0);
                let bias_i32 = bias_f.map(|b| quantize_bias(&b, prev_scale, &w_scales));
                let requant: Vec<(i32, i32)> = w_scales
                    .iter()
                    .map(|&ws| quantize_multiplier((prev_scale * ws) / out_scale))
                    .collect();
                let name = intern(w_name.trim_end_matches("_w"));
                layers.push(Layer::Conv2d {
                    name,
                    h_in: h,
                    w_in: w,
                    c_in: c,
                    c_out,
                    kh,
                    kw,
                    pad_h: ph,
                    pad_w: pw,
                    stride_h: sh,
                    stride_w: sw,
                    x_zp: 0,
                    w_zp: 0,
                    out_zp: 0,
                    weight_bits: mode.weight_bits(),
                    weight_encoding: encoding,
                    requant,
                    weights: packed,
                    bias: bias_i32,
                });
                h = (h + 2 * ph - kh) / sh + 1;
                w = (w + 2 * pw - kw) / sw + 1;
                c = c_out;
                prev_scale = out_scale;
                layer_i += 1;
            }
            Op::Activation(Activation::Relu) => {
                let name = intern(&format!("relu{layer_i}"));
                layers.push(Layer::Relu {
                    name,
                    len: h * w * c,
                    zero_point: 0,
                });
                layer_i += 1;
            }
            Op::Pool {
                kind: ReduceOp::Max,
                kernel_size,
                stride,
                ..
            } => {
                let (kh, kw) = pair2(kernel_size)?;
                let (sh, sw) = pair2(stride)?;
                let name = intern(&format!("pool{layer_i}"));
                layers.push(Layer::MaxPool2d {
                    name,
                    h_in: h,
                    w_in: w,
                    c,
                    kh,
                    kw,
                    stride_h: sh,
                    stride_w: sw,
                });
                h = (h - kh) / sh + 1;
                w = (w - kw) / sw + 1;
                layer_i += 1;
            }
            Op::MatMul => {
                if node.inputs.len() < 2 {
                    return Err("MatMul needs weight".into());
                }
                let w_name = tensor_name(g, node.inputs[1], layer_i, "fc");
                let w_f32 = load_f32(g, node.inputs[1], &w_name, &opts.weights_f32)?;
                let w_shape = g.shape(node.inputs[1]);
                let in_f = w_shape.dim(0).unwrap_static();
                let out_f = w_shape.dim(1).unwrap_static();
                let (packed, w_scales, encoding) = quantize_iou(&w_f32, in_f, out_f, mode)?;
                let out_scale = opts.act_scales.get(&w_name).copied().unwrap_or(1.0 / 127.0);
                let bias_i32 = None; // may be filled by following Add
                let requant: Vec<(i32, i32)> = w_scales
                    .iter()
                    .map(|&ws| quantize_multiplier((prev_scale * ws) / out_scale))
                    .collect();
                let name = intern(w_name.trim_end_matches("_w"));
                layers.push(Layer::Dense {
                    name,
                    in_features: in_f,
                    out_features: out_f,
                    x_zp: 0,
                    w_zp: 0,
                    out_zp: 0,
                    weight_bits: mode.weight_bits(),
                    weight_encoding: encoding,
                    requant,
                    weights: packed,
                    bias: bias_i32,
                });
                h = 1;
                w = 1;
                c = out_f;
                prev_scale = out_scale;
                layer_i += 1;
            }
            Op::Binary(BinaryOp::Add) => {
                // Fold bias into the previous Dense/Conv when possible.
                if let Some(Layer::Dense {
                    bias,
                    out_features,
                    name,
                    ..
                }) = layers.last_mut()
                {
                    if bias.is_none() && node.inputs.len() >= 2 {
                        let b_id = if is_weightish(g, node.inputs[1]) {
                            node.inputs[1]
                        } else {
                            node.inputs[0]
                        };
                        let bn = tensor_name(g, b_id, layer_i, &format!("{name}_b"));
                        if let Ok(bf) = load_f32(g, b_id, &bn, &opts.biases_f32) {
                            let w_scales = vec![prev_scale; *out_features];
                            *bias = Some(quantize_bias(&bf, prev_scale, &w_scales));
                        }
                    }
                }
            }
            Op::Reshape { .. } | Op::TopK { .. } | Op::Reduce { .. } | Op::Softmax { .. } => {}
            other => {
                return Err(format!(
                    "rlx-fpga legalize: unsupported op in f32 chain: {other:?}"
                ));
            }
        }
    }

    if layers.is_empty() {
        return Err("rlx-fpga legalize: no Conv/MatMul layers found".into());
    }

    let ends_with_argmax = matches!(layers.last(), Some(Layer::Argmax { .. }));
    if opts.append_argmax && !ends_with_argmax {
        if let Some(Layer::Dense { out_features, .. }) = layers.last() {
            let len = *out_features;
            layers.push(Layer::Argmax {
                name: intern("argmax"),
                len,
            });
        }
    }

    Ok(Model {
        name: g.name.clone(),
        input_len,
        layers,
        extra_outputs: vec![],
    })
}

fn is_weightish(g: &Graph, id: NodeId) -> bool {
    matches!(g.node(id).op, Op::Param { .. } | Op::Constant { .. })
}

fn walk_activation_chain(g: &Graph, input: NodeId, output: NodeId) -> Result<Vec<NodeId>, String> {
    let mut rev = Vec::new();
    let mut cur = output;
    for _ in 0..g.len() + 4 {
        if cur == input {
            break;
        }
        let node = g.node(cur);
        match &node.op {
            Op::Input { .. } => break,
            Op::Param { .. } | Op::Constant { .. } => {
                return Err("activation chain hit a leaf before input".into());
            }
            Op::Conv { .. }
            | Op::MatMul
            | Op::Activation(_)
            | Op::Pool { .. }
            | Op::Reshape { .. }
            | Op::Binary(_)
            | Op::TopK { .. }
            | Op::Reduce { .. }
            | Op::Softmax { .. } => {
                rev.push(cur);
                cur = match &node.op {
                    Op::Binary(BinaryOp::Add) if node.inputs.len() == 2 => {
                        if is_weightish(g, node.inputs[1]) {
                            node.inputs[0]
                        } else if is_weightish(g, node.inputs[0]) {
                            node.inputs[1]
                        } else {
                            node.inputs[0]
                        }
                    }
                    _ => node.inputs.first().copied().ok_or("no inputs")?,
                };
            }
            other => {
                return Err(format!("unsupported on f32 chain: {other:?}"));
            }
        }
    }
    rev.reverse();
    Ok(rev)
}

fn pair2(v: &[usize]) -> Result<(usize, usize), String> {
    match v {
        [a, b] => Ok((*a, *b)),
        [a] => Ok((*a, *a)),
        _ => Err(format!("expected 1–2 spatial dims, got {v:?}")),
    }
}

fn tensor_name(g: &Graph, id: NodeId, idx: usize, kind: &str) -> String {
    let n = g.node(id);
    if let Some(name) = &n.name {
        return name.clone();
    }
    match &n.op {
        Op::Param { name } => name.clone(),
        _ => format!("{kind}{idx}_w"),
    }
}

fn load_f32(
    g: &Graph,
    id: NodeId,
    key: &str,
    table: &HashMap<String, Vec<f32>>,
) -> Result<Vec<f32>, String> {
    if let Some(v) = table.get(key) {
        return Ok(v.clone());
    }
    // Also try without _w suffix / with common aliases.
    if let Some(stem) = key.strip_suffix("_w") {
        if let Some(v) = table.get(stem) {
            return Ok(v.clone());
        }
    }
    match &g.node(id).op {
        Op::Constant { data } => {
            if data.len() % 4 != 0 {
                return Err(format!("{key}: Constant byte len not multiple of 4"));
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        Op::Param { name } => table.get(name).cloned().ok_or_else(|| {
            format!(
                "rlx-fpga legalize: missing f32 weights for '{name}' \
                 (set LegalizeOptions.weights_f32[{key:?}] or bake Op::Constant)"
            )
        }),
        other => Err(format!("{key}: expected Constant/Param, got {other:?}")),
    }
}

fn quantize_bias(b: &[f32], in_scale: f32, w_scale: &[f32]) -> Vec<i32> {
    b.iter()
        .zip(w_scale.iter().cycle())
        .map(|(&x, &ws)| {
            let acc = (in_scale * ws).max(1e-12);
            (x / acc).round().clamp(i32::MIN as f32, i32::MAX as f32) as i32
        })
        .collect()
}

fn max_pos_code(bits: u8) -> i32 {
    match bits {
        8 => 127,
        4 => 7,
        2 => 1,
        _ => 127,
    }
}

fn quantize_oihw(
    w: &[f32],
    c_out: usize,
    c_in: usize,
    kh: usize,
    kw: usize,
    mode: ExportQuantMode,
) -> Result<(Vec<i8>, Vec<f32>, WeightEncoding), String> {
    let row = c_in * kh * kw;
    if w.len() != c_out * row {
        return Err(format!(
            "weight len {} != c_out*c_in*kh*kw {}",
            w.len(),
            c_out * row
        ));
    }
    // Keep OIHW (IR layout); FPGA reference uses cortexm NHWC for TinyConv
    // blobs — legalized custom graphs keep OIHW and expect matching kernels.
    // For export Model, pack in IR order; TinyConv path still uses cortexm.
    match mode {
        ExportQuantMode::Int8 | ExportQuantMode::Int4 => {
            let bits = mode.weight_bits();
            let qmax = max_pos_code(bits);
            let mut scales = Vec::with_capacity(c_out);
            let mut logical = vec![0i8; w.len()];
            for oc in 0..c_out {
                let row_s = &w[oc * row..(oc + 1) * row];
                let amax = row_s.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
                let s = amax / qmax as f32;
                scales.push(s);
                for (k, &v) in row_s.iter().enumerate() {
                    logical[oc * row + k] = ((v / s).round() as i32).clamp(-qmax, qmax) as i8;
                }
            }
            Ok((pack(&logical, bits), scales, WeightEncoding::SignedInt))
        }
        ExportQuantMode::Fp4 => quantize_fp4_rows(w, c_out, row),
    }
}

fn quantize_iou(
    w: &[f32],
    in_f: usize,
    out_f: usize,
    mode: ExportQuantMode,
) -> Result<(Vec<i8>, Vec<f32>, WeightEncoding), String> {
    if w.len() != in_f * out_f {
        return Err(format!(
            "FC weight len {} != in*out {}",
            w.len(),
            in_f * out_f
        ));
    }
    // Quantize per output column (OI layout for FPGA Dense: [O, I] preferred).
    // Incoming IR MatMul is often [I, O]; transpose to [O, I].
    let mut oi = vec![0f32; w.len()];
    for i in 0..in_f {
        for o in 0..out_f {
            oi[o * in_f + i] = w[i * out_f + o];
        }
    }
    match mode {
        ExportQuantMode::Int8 | ExportQuantMode::Int4 => {
            let bits = mode.weight_bits();
            let qmax = max_pos_code(bits);
            let mut scales = Vec::with_capacity(out_f);
            let mut logical = vec![0i8; oi.len()];
            for o in 0..out_f {
                let row = &oi[o * in_f..(o + 1) * in_f];
                let amax = row.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
                let s = amax / qmax as f32;
                scales.push(s);
                for (k, &v) in row.iter().enumerate() {
                    logical[o * in_f + k] = ((v / s).round() as i32).clamp(-qmax, qmax) as i8;
                }
            }
            Ok((pack(&logical, bits), scales, WeightEncoding::SignedInt))
        }
        ExportQuantMode::Fp4 => quantize_fp4_rows(&oi, out_f, in_f),
    }
}

/// Per-row FP4 E2M1 encode. Codes packed as INT4 nibbles.
/// `w_scale[oc]` maps decoded float → original weight units.
fn quantize_fp4_rows(
    w: &[f32],
    rows: usize,
    row_len: usize,
) -> Result<(Vec<i8>, Vec<f32>, WeightEncoding), String> {
    let fmt = ScaledFormat::F4E2M1;
    let max_f = fmt.max_finite().max(1e-12);
    let mut scales = Vec::with_capacity(rows);
    let mut codes = vec![0i8; w.len()];
    for r in 0..rows {
        let row = &w[r * row_len..(r + 1) * row_len];
        let amax = row.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
        let s = amax / max_f;
        scales.push(s);
        for (k, &v) in row.iter().enumerate() {
            let code = fmt.encode(v / s) as i8;
            codes[r * row_len + k] = code;
        }
    }
    Ok((pack(&codes, 4), scales, WeightEncoding::Fp4E2M1))
}

fn intern(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::Shape;

    #[test]
    fn quant_mode_parse() {
        assert_eq!(
            ExportQuantMode::parse("int8").unwrap(),
            ExportQuantMode::Int8
        );
        assert_eq!(ExportQuantMode::parse("fp4").unwrap(), ExportQuantMode::Fp4);
    }

    #[test]
    fn fp4_pack_roundtrip_len() {
        let w: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let (packed, scales, enc) = quantize_fp4_rows(&w, 2, 8).unwrap();
        assert_eq!(enc, WeightEncoding::Fp4E2M1);
        assert_eq!(packed.len(), 8); // 16 codes / 2
        assert_eq!(scales.len(), 2);
    }

    #[test]
    fn scaled_matmul_rejected() {
        let mut g = Graph::new("sm");
        let x = g.input("x", Shape::new(&[4, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 4], DType::U8));
        let y = g.add_node(
            Op::ScaledMatMul {
                lhs_format: ScaledFormat::F4E2M1,
                rhs_format: ScaledFormat::F4E2M1,
                scale_layout: rlx_ir::ScaleLayout::PerTensor,
                has_bias: false,
            },
            vec![x, w],
            Shape::new(&[4, 4], DType::F32),
        );
        g.set_outputs(vec![y]);
        let err = prepare_model(
            &g,
            ExportQuantMode::Fp4,
            &LegalizeOptions::default(),
            &FromGraphOptions::default(),
        )
        .unwrap_err();
        assert!(err.contains("ScaledMatMul"), "{err}");
    }
}
