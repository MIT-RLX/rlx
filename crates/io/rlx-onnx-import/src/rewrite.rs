// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX graph rewrites before HIR lowering (quant fusion, dead quant removal).

use std::collections::{HashMap, HashSet};

use crate::bundle::{BundleManifest, BundleNode};
use crate::lower::ImportOptions;
use crate::shape_propagate::propagate_shapes;
use crate::tensor_data::f32_tensor;
use rlx_ir::{DType, Shape};

pub struct RewriteResult {
    pub nodes: Vec<BundleNode>,
    pub extra_params: HashMap<String, Vec<f32>>,
    pub extra_shapes: HashMap<String, Vec<usize>>,
}

/// Apply all import-time rewrites to the node list (in order).
pub fn rewrite_graph(
    nodes: Vec<BundleNode>,
    params: &HashMap<String, Vec<f32>>,
    init_shapes: &HashMap<String, Vec<usize>>,
    manifest: &BundleManifest,
    opts: &ImportOptions,
    quant_weight_keys: &HashSet<String>,
) -> RewriteResult {
    let mut nodes = nodes;
    if let Some(patch) = opts.pre_shape_propagate {
        patch(&mut nodes, opts);
    }
    propagate_shapes(&mut nodes, manifest, init_shapes, opts);
    if let Some(patch) = opts.post_shape_propagate {
        patch(&mut nodes, opts);
    }
    let mut extra_params = HashMap::new();
    let mut extra_shapes = HashMap::new();
    // ONNX-standard integer dequant → f32 MatMul/Conv (both profiles).
    rewrite_conv_integer(
        &mut nodes,
        params,
        init_shapes,
        &mut extra_params,
        &mut extra_shapes,
    );
    rewrite_matmul_integer(
        &mut nodes,
        params,
        init_shapes,
        opts,
        &mut extra_params,
        &mut extra_shapes,
    );
    // Kokoro / ISTFTNet exports `atan2(y,x)` as `atan(y/x)` plus quadrant
    // correction via `Where(Greater(y,0), atan+π, atan−π)`. When `y` is exact
    // 0 (common for STFT DC / Nyquist imag bins under the native DFT lowering),
    // `Greater` picks the `atan−π` branch (−π) while numpy/`f32::atan2` and
    // ORT's slightly-nonzero imag pick `+π`. That π/−π flip is fed as a raw
    // feature into `noise_convs` and rings at the ISTFT hop rate (~4800 Hz).
    // Promote the compare to `GreaterOrEqual` so `y==0` matches atan2.
    rewrite_atan2_greater_to_geq(&mut nodes, params);
    if opts.quantize_bundle_rewrites {
        rewrite_quant_matmul_to_qmatmul(&mut nodes);
        rewrite_dynamic_quant(&mut nodes);
        rewrite_f32_quant_matmul_bypass_output_scales(&mut nodes, quant_weight_keys);
        rewrite_f32_quant_conv_bypass_output_scales(&mut nodes);
        prune_quant_matmul_epilogue_nodes(&mut nodes);
        prune_dead_dynamic_quant(&mut nodes);
    }
    RewriteResult {
        nodes,
        extra_params,
        extra_shapes,
    }
}

/// Rewrite `Greater(y, 0)` → `GreaterOrEqual(y, 0)` when it is the quadrant
/// selector of an `atan(y/x)`-style atan2 expansion (see `rewrite_graph`).
fn rewrite_atan2_greater_to_geq(nodes: &mut [BundleNode], params: &HashMap<String, Vec<f32>>) {
    let by_out: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |o| (o.as_str(), i)))
        .collect();
    let consumers: HashMap<&str, Vec<usize>> = {
        let mut m: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for inp in &n.inputs {
                if !inp.is_empty() {
                    m.entry(inp.as_str()).or_default().push(i);
                }
            }
        }
        m
    };

    let is_scalar_zero = |name: &str| -> bool {
        if let Some(v) = params.get(name) {
            return v.len() == 1 && v[0] == 0.0;
        }
        by_out.get(name).and_then(|&i| {
            let n = &nodes[i];
            if n.op != "Constant" {
                return None;
            }
            // Constant value may live in attrs (`value` tensor) or already be
            // folded into `params` under the output name; also accept the common
            // float-zero attr encoding used by shape-propagated graphs.
            n.attrs.get("value").and_then(|v| {
                v.as_f64().map(|f| f == 0.0).or_else(|| {
                    v.as_array()
                        .map(|a| a.len() == 1 && a[0].as_f64() == Some(0.0))
                })
            })
        }) == Some(true)
    };

    let mut promote: Vec<usize> = Vec::new();
    for (gi, gnode) in nodes.iter().enumerate() {
        if gnode.op != "Greater" || gnode.inputs.len() < 2 || gnode.outputs.is_empty() {
            continue;
        }
        if !is_scalar_zero(&gnode.inputs[1]) {
            continue;
        }
        let gout = gnode.outputs[0].as_str();
        let Some(cons) = consumers.get(gout) else {
            continue;
        };
        // Must feed exactly one Where as its condition.
        if cons.len() != 1 {
            continue;
        }
        let w = &nodes[cons[0]];
        if w.op != "Where" || w.inputs.len() < 3 || w.inputs[0] != gout {
            continue;
        }
        // True/false branches should be Add/Sub of the same Atan (atan±π).
        let (t_name, f_name) = (&w.inputs[1], &w.inputs[2]);
        let (Some(&ti), Some(&fi)) = (by_out.get(t_name.as_str()), by_out.get(f_name.as_str()))
        else {
            continue;
        };
        let (tn, fn_) = (&nodes[ti], &nodes[fi]);
        let (add, sub) = match (tn.op.as_str(), fn_.op.as_str()) {
            ("Add", "Sub") => (tn, fn_),
            ("Sub", "Add") => (fn_, tn),
            _ => continue,
        };
        if add.inputs.is_empty() || sub.inputs.is_empty() {
            continue;
        }
        if add.inputs[0] != sub.inputs[0] {
            continue;
        }
        let Some(&ai) = by_out.get(add.inputs[0].as_str()) else {
            continue;
        };
        if nodes[ai].op != "Atan" {
            continue;
        }
        // Atan's input should be Div(y, x) with the same y as Greater's left.
        let atan_in = nodes[ai].inputs.first().map(String::as_str).unwrap_or("");
        let Some(&di) = by_out.get(atan_in) else {
            continue;
        };
        if nodes[di].op != "Div" || nodes[di].inputs.len() < 2 {
            continue;
        }
        if nodes[di].inputs[0] != gnode.inputs[0] {
            continue;
        }
        promote.push(gi);
    }

    for gi in promote {
        nodes[gi].op = "GreaterOrEqual".into();
    }
}

fn rewrite_dynamic_quant(nodes: &mut [BundleNode]) {
    let qmatmul_act_q: HashSet<String> = nodes
        .iter()
        .filter(|n| n.op == "QMatMul" && !n.inputs.is_empty())
        .map(|n| n.inputs[0].clone())
        .collect();
    let mut alias: HashMap<String, String> = HashMap::new();
    for node in nodes.iter() {
        if node.op != "DynamicQuantizeLinear" || node.inputs.is_empty() || node.outputs.is_empty() {
            continue;
        }
        // Keep real DQL outputs wired into `QMatMul` (uint8 + scale + zp).
        if qmatmul_act_q.contains(&node.outputs[0]) {
            continue;
        }
        // Attention Q/K/V need distinct DQL exports (each feeds its own `QMatMul`).
        if node.name.contains("/attention/") {
            continue;
        }
        alias.insert(node.outputs[0].clone(), node.inputs[0].clone());
    }
    if alias.is_empty() {
        return;
    }
    for node in nodes.iter_mut() {
        for inp in node.inputs.iter_mut() {
            if let Some(src) = alias.get(inp.as_str()) {
                *inp = src.clone();
            }
        }
    }
}

fn dequant_weight(
    params: &HashMap<String, Vec<f32>>,
    init_shapes: &HashMap<String, Vec<usize>>,
    w_q: &str,
) -> Option<(String, Vec<f32>, Vec<usize>)> {
    let w = f32_tensor(params, w_q).or_else(|| {
        w_q.strip_suffix("_quantized")
            .and_then(|base| f32_tensor(params, base))
    })?;
    let scale_name = w_q
        .strip_suffix("_quantized")
        .map(|p| format!("{p}_scale"))?;
    let zp_name = w_q
        .strip_suffix("_quantized")
        .map(|p| format!("{p}_zero_point"))?;
    let mut out = w;
    // The `_quantized` params may hold either the RAW integer codes (0..255 / -128..127) —
    // in which case we must dequantize `(w - zp)·scale` — or values a prior loader ALREADY
    // dequantized to f32 (e.g. the kitten bundle's `load_f32_params`). Re-dequantizing the
    // latter turns every weight into ≈`-zp·scale` (a constant), blowing up the conv. Detect
    // raw codes by their integer-valued-ness and only dequantize those.
    let already_dequant = out.iter().any(|&x| (x - x.round()).abs() > 1e-4);
    if !already_dequant {
        if let Some(scale) = f32_tensor(params, &scale_name) {
            let s = scale.first().copied().unwrap_or(1.0);
            let z = f32_tensor(params, &zp_name)
                .and_then(|z| z.first().copied())
                .unwrap_or(0.0) as i32;
            for x in &mut out {
                *x = (*x - z as f32) * s;
            }
        }
    }
    let shape = init_shapes
        .get(w_q)
        .or_else(|| {
            w_q.strip_suffix("_quantized")
                .and_then(|base| init_shapes.get(base))
        })
        .cloned()
        .unwrap_or_else(|| vec![out.len()]);
    let w_name = format!("{w_q}_f32_import");
    Some((w_name, out, shape))
}

fn rewrite_conv_integer(
    nodes: &mut [BundleNode],
    params: &HashMap<String, Vec<f32>>,
    init_shapes: &HashMap<String, Vec<usize>>,
    extra_params: &mut HashMap<String, Vec<f32>>,
    extra_shapes: &mut HashMap<String, Vec<usize>>,
) {
    // With `RLX_CONVINT_FLOAT_ACT`, route ConvInteger through the pre-quant FLOAT activation
    // (mirrors `rewrite_matmul_integer`): `Conv(act_f32, w_f32)` is the full correct dequant,
    // `conv((act_q-act_zp)·act_scale, (w_q-w_zp)·w_scale)`, so the dropped input zero-point is
    // no longer missing. The default (flag unset) keeps the legacy `Conv(act_q, w_f32)` path.
    // The output-scale epilogue is pruned for float-act convs (act_scale is already baked in) —
    // see `rewrite_f32_quant_conv_bypass_output_scales`. Falls back to `act_q` for any conv whose
    // activation has no DynamicQuantizeLinear producer (e.g. a static/graph-input activation).
    let float_act = std::env::var("RLX_CONVINT_FLOAT_ACT").is_ok();
    struct ConvPlan {
        idx: usize,
        w_name: String,
        act_f32: Option<String>,
    }
    let mut plans: Vec<ConvPlan> = Vec::new();
    {
        let producers: HashMap<&str, &BundleNode> = nodes
            .iter()
            .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
            .collect();
        for (i, node) in nodes.iter().enumerate() {
            if node.op != "ConvInteger" || node.inputs.len() < 2 {
                continue;
            }
            let w_q = node.inputs[1].clone();
            let Some((w_name, data, shape)) = dequant_weight(params, init_shapes, &w_q) else {
                continue;
            };
            extra_params.insert(w_name.clone(), data);
            extra_shapes.insert(w_name.clone(), shape);
            let act_f32 = if float_act {
                trace_pre_quant(node.inputs[0].as_str(), &producers, params).map(str::to_string)
            } else {
                None
            };
            plans.push(ConvPlan {
                idx: i,
                w_name,
                act_f32,
            });
        }
    }
    for p in plans {
        let node = &mut nodes[p.idx];
        node.op = "Conv".to_string();
        if let Some(act) = p.act_f32 {
            node.inputs[0] = act;
        }
        node.inputs[1] = p.w_name;
        node.inputs.truncate(2);
        node.output_meta.iter_mut().for_each(|m| {
            if let Some(obj) = m.as_object_mut() {
                if let Some(dt) = obj.get_mut("dtype") {
                    *dt = serde_json::json!("f32");
                }
            }
        });
    }
}

fn trace_pre_quant<'a>(
    name: &'a str,
    producers: &HashMap<&'a str, &'a BundleNode>,
    params: &HashMap<String, Vec<f32>>,
) -> Option<&'a str> {
    if params.contains_key(name) {
        return None;
    }
    let node = producers.get(name)?;
    match node.op.as_str() {
        "DynamicQuantizeLinear" | "Cast" if !node.inputs.is_empty() => {
            trace_pre_quant(&node.inputs[0], producers, params)
        }
        _ => Some(name),
    }
}

fn output_shape_usize(
    nodes: &[BundleNode],
    init_shapes: &HashMap<String, Vec<usize>>,
    name: &str,
    seq_len: usize,
) -> Option<Vec<usize>> {
    if let Some(s) = init_shapes.get(name) {
        return Some(s.clone());
    }
    for n in nodes {
        for (i, out) in n.outputs.iter().enumerate() {
            if out == name {
                let meta = n.output_meta.get(i)?;
                let arr = meta.get("shape")?.as_array()?;
                let dims: Vec<usize> = arr
                    .iter()
                    .filter_map(|d| {
                        d.as_u64().map(|x| x as usize).or_else(|| {
                            d.as_str().and_then(|s| {
                                if s == "sequence_length" || s.starts_with("unk__") || s == "?" {
                                    Some(seq_len)
                                } else {
                                    None
                                }
                            })
                        })
                    })
                    .collect();
                if !dims.is_empty() {
                    return Some(dims);
                }
            }
        }
    }
    None
}

fn matmul_out_dims(
    act_dims: Option<&[usize]>,
    w_dims: &[usize],
    seq_len: usize,
) -> Option<Vec<usize>> {
    if let Some(act_dims) = act_dims {
        let sa = Shape::new(act_dims, DType::F32);
        let sb = Shape::new(w_dims, DType::F32);
        if let Ok(out) = rlx_ir::shape::matmul_shape(&sa, &sb) {
            return Some(out.dims().iter().map(|d| d.unwrap_static()).collect());
        }
        if act_dims.len() >= 2 && w_dims.len() >= 2 {
            let mut out = act_dims.to_vec();
            *out.last_mut()? = w_dims[w_dims.len() - 1];
            return Some(out);
        }
    }
    if w_dims.len() >= 2 {
        let n = w_dims[w_dims.len() - 1];
        if w_dims[0] >= 64 {
            return Some(vec![1, seq_len, n]);
        }
        return Some(vec![1, n]);
    }
    None
}

fn rewrite_matmul_integer(
    nodes: &mut [BundleNode],
    params: &HashMap<String, Vec<f32>>,
    init_shapes: &HashMap<String, Vec<usize>>,
    opts: &ImportOptions,
    extra_params: &mut HashMap<String, Vec<f32>>,
    extra_shapes: &mut HashMap<String, Vec<usize>>,
) {
    let producers: HashMap<&str, &BundleNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();
    let merged: HashMap<String, Vec<f32>> = params
        .iter()
        .chain(extra_params.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    struct Patch {
        idx: usize,
        act_f32: String,
        w_name: String,
        out_dims: Vec<usize>,
    }
    let mut patches: Vec<Patch> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if node.op != "MatMulInteger" || node.inputs.len() < 2 {
            continue;
        }
        let act_q = &node.inputs[0];
        let w_q = &node.inputs[1];
        if !merged.contains_key(w_q) {
            continue;
        }
        let act_f32 = match trace_pre_quant(act_q.as_str(), &producers, params) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let Some((w_name, data, shape)) = dequant_weight(&merged, init_shapes, w_q) else {
            continue;
        };
        let act_dims = output_shape_usize(nodes, init_shapes, &act_f32, opts.sequence_length);
        let Some(out_dims) = matmul_out_dims(act_dims.as_deref(), &shape, opts.sequence_length)
        else {
            continue;
        };
        extra_params.insert(w_name.clone(), data);
        extra_shapes.insert(w_name.clone(), shape);
        patches.push(Patch {
            idx: i,
            act_f32,
            w_name,
            out_dims,
        });
    }
    for p in patches {
        nodes[p.idx].op = "MatMul".to_string();
        nodes[p.idx].inputs = vec![p.act_f32, p.w_name];
        nodes[p.idx].output_meta = vec![serde_json::json!({
            "shape": p.out_dims,
            "dtype": "f32",
        })];
    }
}

/// Replace non-embedding quant matmul epilogues with `QMatMul` (ORT `MatMulInteger` semantics).
fn rewrite_quant_matmul_to_qmatmul(nodes: &mut [BundleNode]) {
    let producers: HashMap<&str, &BundleNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();
    struct QMatMulPatch {
        idx: usize,
        act_q: String,
        act_scale: String,
        act_zp: String,
        scaled_out: String,
        w_q: String,
        w_scale: String,
        w_zp: String,
    }
    let mut patches: Vec<QMatMulPatch> = Vec::new();
    for (idx, node) in nodes.iter().enumerate() {
        if node.op != "MatMul" || node.inputs.len() < 2 {
            continue;
        }
        if !node.inputs[1].contains("quant_f32_weight") {
            continue;
        }
        // Decoder waveform `Gemm` keeps f32 epilogue; LSTM `Gemm` fc layers use `QMatMul`.
        if node.name.contains("Gemm") {
            let lstm_fc = node.name.contains("lstms") || node.name.contains("/lstm/");
            if !lstm_fc {
                continue;
            }
        }
        let Some(mm_out) = node.outputs.first() else {
            continue;
        };
        let Some(epilogue) = trace_f32_quant_scale_epilogue(&producers, mm_out) else {
            continue;
        };
        if epilogue.bypass_to_matmul {
            continue;
        }
        let Some(scales_mul) = epilogue
            .scales_mul_out
            .as_deref()
            .and_then(|out| producers.get(out).copied())
        else {
            continue;
        };
        let Some(w_scale) = scales_mul.inputs.get(1).cloned() else {
            continue;
        };
        let w_base = w_scale
            .strip_suffix("_scale")
            .unwrap_or(w_scale.as_str())
            .to_string();
        let act_f32 = node.inputs[0].clone();
        let Some(dql) = nodes.iter().find(|n| {
            n.op == "DynamicQuantizeLinear"
                && n.inputs.first() == Some(&act_f32)
                && n.outputs.len() >= 3
        }) else {
            continue;
        };
        patches.push(QMatMulPatch {
            idx,
            act_q: dql.outputs[0].clone(),
            act_scale: dql.outputs[1].clone(),
            act_zp: dql.outputs[2].clone(),
            scaled_out: epilogue.scaled_out,
            w_q: format!("{w_base}_quantized"),
            w_scale,
            w_zp: format!("{w_base}_zero_point"),
        });
    }
    for p in patches {
        let node = &mut nodes[p.idx];
        node.op = "QMatMul".to_string();
        node.inputs = vec![p.act_q, p.act_scale, p.act_zp, p.w_q, p.w_scale, p.w_zp];
        node.outputs = vec![p.scaled_out];
    }
}

fn prune_dead_dynamic_quant(nodes: &mut Vec<BundleNode>) {
    let mut consumers: HashMap<String, usize> = HashMap::new();
    for node in nodes.iter() {
        for inp in &node.inputs {
            *consumers.entry(inp.clone()).or_default() += 1;
        }
    }
    nodes.retain(|node| {
        if node.op != "DynamicQuantizeLinear" {
            return true;
        }
        node.outputs
            .iter()
            .any(|o| consumers.get(o).copied().unwrap_or(0) > 0)
    });
}

fn prune_quant_matmul_epilogue_nodes(nodes: &mut Vec<BundleNode>) {
    let mut consumers: HashMap<String, usize> = HashMap::new();
    for node in nodes.iter() {
        for inp in &node.inputs {
            *consumers.entry(inp.clone()).or_default() += 1;
        }
    }
    nodes.retain(|node| {
        if node.name.contains("MatMul_quant_output_scale_mul")
            || (node.name.contains("MatMul") && node.name.contains("_output_quantized_cast"))
            || (node.name.contains("Conv") && node.name.contains("_output_quantized_cast"))
        {
            return false;
        }
        if (node.name.contains("MatMul_quant_scales_mul")
            || node.name.contains("Conv_quant_scales_mul")
            // Float-activation ConvInteger prunes its output-scale Mul (act_scale is baked
            // into the float activation); its `scaled_out` is aliased away → unconsumed. The
            // legacy `Conv(act_q,…)` path keeps it (still feeds the bias add).
            || node.name.contains("Conv_quant_output_scale_mul"))
            && node.op == "Mul"
        {
            return node
                .outputs
                .iter()
                .any(|o| consumers.get(o).copied().unwrap_or(0) > 0);
        }
        true
    });
}

/// Bundle export lowers `MatMulInteger` to f32 `MatMul` with dequantized weights but leaves
/// the post-int cast + output-scale `Mul` chain. `rewrite_matmul_integer` already folded
/// weight scales into `*_quant_f32_weight`, so the epilogue must apply **activation scale only**:
/// `cast(f32_matmul) * act_scale`. Embedding-fed Q/K/V matmuls are an exception: their f32
/// matmul output already matches ORT without the epilogue.
/// `ConvInteger` → f32 `Conv` (dequantized weights) still leaves an i32 cast + scale `Mul`
/// epilogue. Rewire it to `f32_conv * act_scale` (weight scale is already in the import weight).
fn rewrite_f32_quant_conv_bypass_output_scales(nodes: &mut [BundleNode]) {
    let producers: HashMap<&str, &BundleNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();
    struct ConvScalePatch {
        scaled_out: String,
        conv_out: String,
        act_scale: String,
    }
    let mut patches: Vec<ConvScalePatch> = Vec::new();
    // Float-activation convs (`Conv(act_f32, w_f32)`) already carry BOTH scales, so the
    // `Conv_quant_output_scale_mul` epilogue must be pruned (not re-applied): alias its output
    // to the conv output and let later DCE drop the dead Mul (mirrors the MatMulInteger bypass).
    let mut aliases: HashMap<String, String> = HashMap::new();
    {
        for node in nodes.iter() {
            if node.op != "Conv" || node.inputs.len() < 2 {
                continue;
            }
            if !node.inputs[1].ends_with("_f32_import") {
                continue;
            }
            let Some(conv_out) = node.outputs.first().cloned() else {
                continue;
            };
            let Some(epilogue) = trace_f32_quant_scale_epilogue(&producers, &conv_out) else {
                continue;
            };
            // A quantized (`*_quantized`) input marks the legacy `act_q` path (apply act_scale);
            // anything else is the pre-quant float activation (prune the scale epilogue).
            if !node.inputs[0].ends_with("_quantized") {
                if epilogue.scaled_out != conv_out {
                    aliases.insert(epilogue.scaled_out, conv_out.clone());
                }
                continue;
            }
            let Some(scales_mul) = epilogue
                .scales_mul_out
                .as_deref()
                .and_then(|name| producers.get(name).copied())
            else {
                continue;
            };
            let Some(act_scale) = scales_mul.inputs.first().cloned() else {
                continue;
            };
            patches.push(ConvScalePatch {
                scaled_out: epilogue.scaled_out,
                conv_out,
                act_scale,
            });
        }
    }
    for patch in patches {
        for mul in nodes.iter_mut() {
            if mul.op != "Mul" || !mul.name.contains("Conv_quant_output_scale_mul") {
                continue;
            }
            if mul.outputs.first().map(String::as_str) != Some(patch.scaled_out.as_str()) {
                continue;
            }
            if mul.inputs.len() < 2 {
                continue;
            }
            mul.inputs[0] = patch.conv_out.clone();
            mul.inputs[1] = patch.act_scale.clone();
        }
    }
    if !aliases.is_empty() {
        for node in nodes.iter_mut() {
            for inp in node.inputs.iter_mut() {
                if let Some(src) = aliases.get(inp.as_str()) {
                    *inp = src.clone();
                }
            }
        }
    }
}

fn rewrite_f32_quant_matmul_bypass_output_scales(
    nodes: &mut [BundleNode],
    _quant_weight_keys: &HashSet<String>,
) {
    let producers: HashMap<&str, &BundleNode> = nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |o| (o.as_str(), n)))
        .collect();
    let mut aliases: HashMap<String, String> = HashMap::new();
    for node in nodes.iter() {
        if node.op != "MatMul" || node.inputs.len() < 2 {
            continue;
        }
        let w = node.inputs[1].as_str();
        if !w.contains("quant_f32_weight") {
            continue;
        }
        let Some(mm_out) = node.outputs.first() else {
            continue;
        };
        let Some(epilogue) = trace_f32_quant_scale_epilogue(&producers, mm_out) else {
            continue;
        };
        if epilogue.scaled_out.as_str() != mm_out.as_str() {
            aliases.insert(epilogue.scaled_out, mm_out.clone());
        }
    }
    if aliases.is_empty() {
        return;
    }
    for node in nodes.iter_mut() {
        for inp in node.inputs.iter_mut() {
            if let Some(src) = aliases.get(inp.as_str()) {
                *inp = src.clone();
            }
        }
    }
}

struct F32QuantScaleEpilogue {
    scaled_out: String,
    scales_mul_out: Option<String>,
    bypass_to_matmul: bool,
}

fn trace_f32_quant_scale_epilogue(
    producers: &HashMap<&str, &BundleNode>,
    mm_out: &str,
) -> Option<F32QuantScaleEpilogue> {
    let cast = producers.values().find(|n| {
        n.op == "Cast" && n.inputs.len() == 1 && n.inputs[0] == mm_out && n.outputs.len() == 1
    })?;
    let cast_out = cast.outputs[0].as_str();
    let out_mul = producers.values().find(|n| {
        n.op == "Mul"
            && n.name.contains("quant_output_scale_mul")
            && n.inputs.len() >= 2
            && n.inputs[0] == cast_out
            && n.outputs.len() == 1
    })?;
    let scaled_out = out_mul.outputs[0].clone();
    let scales_in = out_mul.inputs[1].as_str();
    let scales_mul = producers
        .get(scales_in)
        .copied()
        .filter(|n| n.op == "Mul" && n.name.contains("quant_scales_mul") && n.inputs.len() >= 2);
    let (scales_mul_out, act_scale) =
        scales_mul.map(|n| (n.outputs.first().cloned(), n.inputs.first().cloned()))?;
    let bypass_to_matmul = act_scale
        .as_deref()
        .is_some_and(|s| s.ends_with("Add_output_0_scale"));
    Some(F32QuantScaleEpilogue {
        scaled_out,
        scales_mul_out,
        bypass_to_matmul,
    })
}

#[cfg(test)]
mod rewrite_tests {
    use super::*;
    use crate::bundle::{load_bundle, onnx_bundle_dir};
    use crate::lower::ImportOptions;

    #[test]
    fn gemm_quant_matmuls_stay_f32_after_rewrite() {
        let dir = onnx_bundle_dir();
        if !dir.join("manifest.json").exists() {
            return;
        }
        let bundle = load_bundle(&dir).expect("bundle");
        let opts = ImportOptions::quant_bundle();
        let params = HashMap::new();
        let init_shapes = HashMap::new();
        let out = rewrite_graph(
            bundle.nodes.clone(),
            &params,
            &init_shapes,
            &bundle.manifest,
            &opts,
            &HashSet::new(),
        );
        let gemm_mm: Vec<_> = out
            .nodes
            .iter()
            .filter(|n| n.name.contains("Gemm") && n.op == "MatMul")
            .map(|n| n.name.as_str())
            .collect();
        assert!(
            gemm_mm.is_empty(),
            "Gemm quant matmuls should not be rewritten to QMatMul: {gemm_mm:?}"
        );
    }

    #[test]
    fn f0_proj_conv_epilogue_bypassed_after_rewrite() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../rlx-models/crates/kitten_tts_mini_rlx/weights/rlx_bundle");
        if !dir.join("manifest.json").exists() {
            return;
        }
        let bundle = load_bundle(&dir).expect("bundle");
        let opts = ImportOptions::quant_bundle();
        let mut params = HashMap::new();
        let mut init_shapes = HashMap::new();
        let st = bundle.weights().expect("weights");
        for name in st.names() {
            let key = name.to_string();
            let view = st.tensor(&key).expect("tensor");
            let f32s: Vec<f32> = view
                .data()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            params.insert(key.clone(), f32s);
            init_shapes.insert(key, view.shape().to_vec());
        }
        let out = rewrite_graph(
            bundle.nodes.clone(),
            &params,
            &init_shapes,
            &bundle.manifest,
            &opts,
            &HashSet::new(),
        );
        let conv = out
            .nodes
            .iter()
            .find(|n| n.name == "/F0_proj/Conv_quant")
            .expect("F0_proj conv");
        assert_eq!(conv.op, "Conv");
        assert!(
            conv.inputs[1].ends_with("_f32_import"),
            "expected dequantized conv weight, got {}",
            conv.inputs[1]
        );
        let bias_add = out
            .nodes
            .iter()
            .find(|n| n.name == "/F0_proj/Conv_output_0_Cast_to_float16_input_0_bias_add")
            .expect("bias add");
        let scale_mul = out
            .nodes
            .iter()
            .find(|n| n.name == "/F0_proj/Conv_quant_output_scale_mul")
            .expect("scale mul");
        assert_eq!(
            scale_mul.inputs.first().map(String::as_str),
            Some("/F0_proj/Conv_output_0_Cast_to_float16_input_0_output_quantized"),
            "scale mul should read f32 conv output"
        );
        assert_eq!(
            scale_mul.inputs.get(1).map(String::as_str),
            Some("/F0.2/Div_output_0_scale"),
            "scale mul should apply activation scale only"
        );
        assert_eq!(
            bias_add.inputs.first().map(String::as_str),
            Some("/F0_proj/Conv_output_0_Cast_to_float16_input_0quant_scaled_output"),
            "bias add should consume scaled conv output"
        );
    }

    #[test]
    fn atan2_quadrant_greater_promoted_to_geq() {
        // Synthetic atan2 expansion matching Kokoro's ISTFTNet export:
        //   Div(y,x) → Atan → Add/Sub ±π → Where(Greater(y,0), …)
        let mut nodes = vec![
            BundleNode {
                name: "c0".into(),
                op: "Constant".into(),
                inputs: vec![],
                outputs: vec!["zero".into()],
                attrs: HashMap::from([("value".into(), serde_json::json!(0.0))]),
                output_meta: vec![],
            },
            BundleNode {
                name: "div".into(),
                op: "Div".into(),
                inputs: vec!["y".into(), "x".into()],
                outputs: vec!["yx".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
            BundleNode {
                name: "atan".into(),
                op: "Atan".into(),
                inputs: vec!["yx".into()],
                outputs: vec!["a".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
            BundleNode {
                name: "add".into(),
                op: "Add".into(),
                inputs: vec!["a".into(), "pi".into()],
                outputs: vec!["ap".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
            BundleNode {
                name: "sub".into(),
                op: "Sub".into(),
                inputs: vec!["a".into(), "pi".into()],
                outputs: vec!["am".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
            BundleNode {
                name: "gt".into(),
                op: "Greater".into(),
                inputs: vec!["y".into(), "zero".into()],
                outputs: vec!["cond".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
            BundleNode {
                name: "where".into(),
                op: "Where".into(),
                inputs: vec!["cond".into(), "ap".into(), "am".into()],
                outputs: vec!["ph".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
        ];
        let params = HashMap::new();
        rewrite_atan2_greater_to_geq(&mut nodes, &params);
        let gt = nodes.iter().find(|n| n.name == "gt").expect("gt");
        assert_eq!(
            gt.op, "GreaterOrEqual",
            "atan2 quadrant Greater(y,0) should become GreaterOrEqual"
        );
    }

    #[test]
    fn unrelated_greater_not_promoted() {
        let mut nodes = vec![
            BundleNode {
                name: "gt".into(),
                op: "Greater".into(),
                inputs: vec!["y".into(), "zero".into()],
                outputs: vec!["cond".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
            BundleNode {
                name: "where".into(),
                op: "Where".into(),
                inputs: vec!["cond".into(), "a".into(), "b".into()],
                outputs: vec!["out".into()],
                attrs: HashMap::new(),
                output_meta: vec![],
            },
        ];
        let params = HashMap::from([("zero".into(), vec![0.0f32])]);
        rewrite_atan2_greater_to_geq(&mut nodes, &params);
        assert_eq!(nodes[0].op, "Greater", "non-atan2 Greater must stay");
    }

    #[test]
    fn kokoro_decoder_atan2_greater_promoted_when_weights_present() {
        // Weights live in the sibling rlx-models repo; try a repo-relative
        // path first, then a common absolute location, and skip if absent.
        let rel = "weights/tts/kokoro-82m/onnx/rlx-split/decoder_raw.onnx";
        let candidates = [
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../../rlx-models")
                .join(rel),
            std::path::PathBuf::from("/Users/Shared/rlx-models").join(rel),
        ];
        let Some(onnx) = candidates.into_iter().find(|p| p.is_file()) else {
            eprintln!("skip: kokoro-82m decoder_raw.onnx not present");
            return;
        };
        let (manifest, nodes, params, _i64, init_shapes) =
            crate::onnx_file::prepare_onnx_file(&onnx).expect("load decoder_raw");
        let opts = ImportOptions::default();
        let out = rewrite_graph(
            nodes,
            &params,
            &init_shapes,
            &manifest,
            &opts,
            &HashSet::new(),
        );
        let gt = out
            .nodes
            .iter()
            .find(|n| n.name == "/decoder/decoder/generator/Greater")
            .expect("Kokoro atan2 Greater node");
        assert_eq!(
            gt.op, "GreaterOrEqual",
            "Kokoro generator Greater(imag,0) should become GreaterOrEqual"
        );
    }
}
