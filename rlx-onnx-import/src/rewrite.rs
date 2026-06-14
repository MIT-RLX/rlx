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
    if opts.quantize_bundle_rewrites {
        rewrite_quant_matmul_to_qmatmul(&mut nodes);
        rewrite_dynamic_quant(&mut nodes);
        rewrite_f32_quant_matmul_bypass_output_scales(&mut nodes, quant_weight_keys);
        prune_quant_matmul_epilogue_nodes(&mut nodes);
        prune_dead_dynamic_quant(&mut nodes);
    }
    RewriteResult {
        nodes,
        extra_params,
        extra_shapes,
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
    if let Some(scale) = f32_tensor(params, &scale_name) {
        let s = scale.first().copied().unwrap_or(1.0);
        let z = f32_tensor(params, &zp_name)
            .and_then(|z| z.first().copied())
            .unwrap_or(0.0) as i32;
        for x in &mut out {
            *x = (*x - z as f32) * s;
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
    for node in nodes.iter_mut() {
        if node.op != "ConvInteger" || node.inputs.len() < 2 {
            continue;
        }
        let w_q = node.inputs[1].clone();
        let Some((w_name, data, shape)) = dequant_weight(params, init_shapes, &w_q) else {
            continue;
        };
        extra_params.insert(w_name.clone(), data);
        extra_shapes.insert(w_name.clone(), shape);
        node.op = "Conv".to_string();
        node.inputs[1] = w_name;
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
        {
            return false;
        }
        if node.name.contains("MatMul_quant_scales_mul") && node.op == "Mul" {
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
}
