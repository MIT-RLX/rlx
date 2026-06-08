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

//! Fill missing `output_meta.shape` in ONNX bundles before HIR lowering.
//!
//! ONNX shape inference leaves many exported nodes with `shape: []`; treating
//! that as a scalar breaks the decoder (ConvTranspose, Gemm, …). We forward-
//! infer static/symbolic shapes from inputs and weight shapes.

use std::collections::HashMap;

use crate::bundle::{BundleManifest, BundleNode, topo_sort_nodes};
use crate::lower::ImportOptions;

type Dims = Vec<serde_json::Value>;
type TensorMeta = (Dims, String);

fn meta_is_empty(meta: &serde_json::Value) -> bool {
    meta.get("shape")
        .and_then(|s| s.as_array())
        .map(|a| a.is_empty())
        .unwrap_or(true)
}

fn normalize_dtype(s: &str) -> String {
    match s {
        "f32" | "type_1" | "type_10" | "type_11" => "f32".to_string(),
        "i64" | "type_7" => "i64".to_string(),
        "i32" | "type_6" => "i32".to_string(),
        "bool" | "type_9" => "bool".to_string(),
        other => other.to_string(),
    }
}

fn meta_from_dims(dims: Dims, dtype: &str) -> serde_json::Value {
    serde_json::json!({ "shape": dims, "dtype": normalize_dtype(dtype) })
}

fn dims_from_usize(d: &[usize]) -> Dims {
    d.iter().map(|&x| serde_json::json!(x)).collect()
}

/// Infer a static size for an ONNX `unk__*` placeholder.
fn default_unk_dim(dims: &Dims, i: usize, opts: &ImportOptions) -> usize {
    if i == 0 {
        return 1;
    }
    let rank = dims.len();
    let channel_axis = dims
        .iter()
        .position(|d| d.as_u64().is_some_and(|n| n >= 64));
    if rank == 3 {
        return match channel_axis {
            Some(1) => match i {
                1 => dims[1].as_u64().unwrap_or(512) as usize,
                2 => opts.sequence_length,
                _ => 1,
            },
            Some(2) => match i {
                1 => opts.sequence_length,
                2 => dims[2].as_u64().unwrap_or(512) as usize,
                _ => 1,
            },
            _ => match i {
                1 => opts.sequence_length,
                2 => 1,
                _ => 1,
            },
        };
    }
    if i + 1 == rank && rank >= 2 {
        return opts.sequence_length;
    }
    1
}

fn resolve_dims(
    dims: &Dims,
    opts: &ImportOptions,
    sym_env: &mut HashMap<String, usize>,
    hint: Option<&Dims>,
) -> Dims {
    let ncl_static = dims.len() == 3 && dims[1].as_u64().is_some_and(|c| c >= 64);
    let blc_static = dims.len() == 3 && dims[2].as_u64().is_some_and(|c| c >= 64);
    dims.iter()
        .enumerate()
        .map(|(i, d)| {
            // `duration_proj` MatMul: `[batch_unk, seq_unk, 50]`.
            if dims.len() == 3 && dims[2].as_u64() == Some(50) {
                return match i {
                    0 => serde_json::json!(1),
                    1 => {
                        if opts.dynamic_sequence {
                            serde_json::json!("sequence_length")
                        } else {
                            serde_json::json!(opts.sequence_length)
                        }
                    }
                    2 => serde_json::json!(50),
                    _ => serde_json::json!(1),
                };
            }
            if ncl_static {
                return match i {
                    0 => serde_json::json!(1),
                    1 => d.clone(),
                    2 if d.as_u64() == Some(1) => serde_json::json!(1),
                    2 => serde_json::json!(opts.sequence_length),
                    _ => serde_json::json!(1),
                };
            }
            if blc_static {
                return match i {
                    0 => serde_json::json!(1),
                    1 => {
                        if d.as_u64().is_some() {
                            d.clone()
                        } else {
                            serde_json::json!(opts.sequence_length)
                        }
                    }
                    2 => d.clone(),
                    _ => serde_json::json!(1),
                };
            }
            if let Some(n) = d.as_u64() {
                return serde_json::json!(n);
            }
            let Some(s) = d.as_str() else {
                return serde_json::json!(1);
            };
            if s == "sequence_length" {
                return if opts.dynamic_sequence {
                    serde_json::json!("sequence_length")
                } else {
                    serde_json::json!(opts.sequence_length)
                };
            }
            if s.starts_with("unk__") {
                if let Some(&v) = sym_env.get(s) {
                    return if opts.dynamic_sequence {
                        serde_json::json!("sequence_length")
                    } else {
                        serde_json::json!(v)
                    };
                }
                if let Some(h) = hint.and_then(|h| h.get(i)) {
                    if let Some(n) = h.as_u64() {
                        sym_env.insert(s.to_string(), n as usize);
                        return if opts.dynamic_sequence {
                            serde_json::json!("sequence_length")
                        } else {
                            serde_json::json!(n)
                        };
                    }
                    if let Some(s2) = h.as_str() {
                        if s2 == "sequence_length" {
                            sym_env.insert(s.to_string(), opts.sequence_length);
                            return if opts.dynamic_sequence {
                                serde_json::json!("sequence_length")
                            } else {
                                serde_json::json!(opts.sequence_length)
                            };
                        }
                        if let Some(&n) = sym_env.get(s2) {
                            sym_env.insert(s.to_string(), n);
                            return serde_json::json!(n);
                        }
                    }
                }
                let v = default_unk_dim(dims, i, opts);
                sym_env.insert(s.to_string(), v);
                return serde_json::json!(v);
            }
            if let Ok(v) = crate::lower::resolve_dim(d, opts) {
                return serde_json::json!(v);
            }
            serde_json::json!(1)
        })
        .collect()
}

fn resolve_static(
    dims: &Dims,
    opts: &ImportOptions,
    sym_env: &HashMap<String, usize>,
) -> Option<Vec<usize>> {
    let resolved = resolve_dims(dims, opts, &mut sym_env.clone(), None);
    resolved
        .iter()
        .map(|d| d.as_u64().map(|x| x as usize))
        .collect()
}

fn get(env: &HashMap<String, TensorMeta>, name: &str) -> Option<TensorMeta> {
    env.get(name).cloned()
}

fn broadcast_dims(a: &Dims, b: &Dims) -> Option<Dims> {
    if a.is_empty() {
        return Some(b.to_vec());
    }
    if b.is_empty() {
        return Some(a.to_vec());
    }
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let pad = long.len() - short.len();
    let mut out = vec![serde_json::json!(1); pad];
    for (i, d) in short.iter().enumerate() {
        let li = &long[pad + i];
        match (d, li) {
            (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
                let xv = x.as_u64()? as usize;
                let yv = y.as_u64()? as usize;
                out.push(serde_json::json!(xv.max(yv)));
            }
            _ if d == li => out.push(d.clone()),
            (serde_json::Value::Number(x), _) if x.as_u64()? == 1 => out.push(li.clone()),
            (_, serde_json::Value::Number(y)) if y.as_u64()? == 1 => out.push(d.clone()),
            _ => out.push(d.clone()),
        }
    }
    Some(out)
}

fn matmul_dims(
    a: &Dims,
    b: &Dims,
    opts: &ImportOptions,
    sym_env: &HashMap<String, usize>,
) -> Option<Dims> {
    let sa = resolve_static(a, opts, sym_env)?;
    let sb = resolve_static(b, opts, sym_env)?;
    let out = rlx_ir::shape::matmul_shape(
        &rlx_ir::Shape::new(&sa, rlx_ir::DType::F32),
        &rlx_ir::Shape::new(&sb, rlx_ir::DType::F32),
    )
    .ok()?;
    Some(
        out.dims()
            .iter()
            .map(|d| serde_json::json!(d.unwrap_static()))
            .collect(),
    )
}

fn conv_output_dims(
    x: &Dims,
    w: &Dims,
    node: &BundleNode,
    transpose: bool,
    opts: &ImportOptions,
    sym_env: &HashMap<String, usize>,
) -> Option<Dims> {
    let xs = resolve_static(x, opts, sym_env)?;
    let ws = resolve_static(w, opts, sym_env)?;
    if xs.len() < 2 || ws.len() < 2 {
        return None;
    }
    let groups = node
        .attrs
        .get("group")
        .and_then(|v| v.as_i64())
        .unwrap_or(1) as usize;
    let stride = node
        .attrs
        .get("strides")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let pad = node
        .attrs
        .get("pads")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let n = *xs.first()?;
    let _ci = if xs.len() >= 2 { xs[1] } else { 1 };
    let li = xs.last().copied()?;
    let co = if transpose {
        ws.get(1).copied()? * groups
    } else {
        ws[0]
    };
    let lo = if transpose {
        rlx_ir::shape::conv_transpose2d_spatial_output(
            li,
            ws.get(2).copied().unwrap_or(1),
            stride,
            pad,
            1,
            0,
        )
    } else {
        li
    };
    Some(vec![
        serde_json::json!(n),
        serde_json::json!(co),
        serde_json::json!(lo),
    ])
}

fn infer_output(
    node: &BundleNode,
    env: &HashMap<String, TensorMeta>,
    init_shapes: &HashMap<String, Vec<usize>>,
    opts: &ImportOptions,
    sym_env: &HashMap<String, usize>,
) -> Option<TensorMeta> {
    let dtype = node
        .output_meta
        .first()
        .and_then(|m| m.get("dtype"))
        .and_then(|d| d.as_str())
        .map(normalize_dtype)
        .unwrap_or_else(|| "f32".to_string());

    let in0 = node.inputs.first().filter(|s| !s.is_empty())?;
    let inp0 = get(env, in0)?;

    match node.op.as_str() {
        "Add" | "Mul" | "Sub" | "Div" | "Pow" | "Where" => {
            let a = get(env, &node.inputs[0])?;
            let b = get(env, node.inputs.get(1).filter(|s| !s.is_empty())?)?;
            let dims = broadcast_dims(&a.0, &b.0)?;
            Some((dims, dtype))
        }
        "MatMul" | "Gemm" | "MatMulInteger" => {
            let a = get(env, &node.inputs[0])?;
            let w_in = node.inputs.get(1).filter(|s| !s.is_empty())?;
            let b = get(env, w_in).or_else(|| {
                init_shapes.get(w_in.as_str()).map(|shape| {
                    (
                        shape
                            .iter()
                            .map(|&d| serde_json::json!(d))
                            .collect::<Dims>(),
                        "f32".to_string(),
                    )
                })
            })?;
            let dims = matmul_dims(&a.0, &b.0, opts, sym_env)?;
            Some((dims, dtype))
        }
        "Conv" => {
            let w = get(env, node.inputs.get(1).filter(|s| !s.is_empty())?)?;
            let dims = conv_output_dims(&inp0.0, &w.0, node, false, opts, sym_env)?;
            Some((dims, dtype))
        }
        "ConvTranspose" => {
            let w = get(env, node.inputs.get(1).filter(|s| !s.is_empty())?)?;
            let dims = conv_output_dims(&inp0.0, &w.0, node, true, opts, sym_env)?;
            Some((dims, dtype))
        }
        "Gather" => {
            let table = get(env, &node.inputs[0])?;
            let idx = get(env, node.inputs.get(1).filter(|s| !s.is_empty())?)?;
            let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0);
            let rank = table.0.len() as i64;
            let ax = axis.rem_euclid(rank.max(1)) as usize;
            let mut out = idx.0.clone();
            if table.0.len() > ax + 1 {
                out.extend(table.0.iter().skip(ax + 1).cloned());
            }
            Some((out, dtype))
        }
        "Concat" => {
            let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0);
            let mut acc: Option<Dims> = None;
            for inp in &node.inputs {
                if inp.is_empty() {
                    continue;
                }
                let Some(t) = get(env, inp) else {
                    continue;
                };
                acc = Some(match acc {
                    None => t.0.clone(),
                    Some(mut d) => {
                        let rank = d.len().max(t.0.len());
                        while d.len() < rank {
                            d.insert(0, serde_json::json!(1));
                        }
                        let mut td = t.0.clone();
                        while td.len() < rank {
                            td.insert(0, serde_json::json!(1));
                        }
                        let ax = axis.rem_euclid(rank as i64) as usize;
                        if ax < d.len() {
                            if let (Some(a), Some(b)) = (d[ax].as_u64(), td[ax].as_u64()) {
                                d[ax] = serde_json::json!(a + b);
                            }
                        }
                        d
                    }
                });
            }
            acc.map(|d| (d, dtype))
        }
        "Transpose" => {
            let perm: Vec<usize> = node
                .attrs
                .get("perm")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|d| d.as_u64().map(|x| x as usize))
                        .collect()
                })
                .unwrap_or_else(|| (0..inp0.0.len()).collect());
            let mut out = vec![serde_json::json!(1); inp0.0.len()];
            for (i, &p) in perm.iter().enumerate() {
                if p < inp0.0.len() {
                    out[i] = inp0.0[p].clone();
                }
            }
            Some((out, dtype))
        }
        "SequenceEmpty" => Some((vec![serde_json::json!(0)], "i64".to_string())),
        "Loop" => Some((
            vec![serde_json::json!(opts.sequence_length)],
            "i64".to_string(),
        )),
        "ConcatFromSequence" => Some((
            vec![serde_json::json!(opts.sequence_length)],
            "i64".to_string(),
        )),
        "Expand" => {
            let shape_src = node
                .inputs
                .get(1)
                .filter(|s| !s.is_empty())
                .and_then(|n| get(env, n))?;
            let in_dims = resolve_static(&inp0.0, opts, sym_env)?;
            let tg_dims = resolve_static(&shape_src.0, opts, sym_env)?;
            let out = crate::layout::expand_output_dims(&in_dims, &tg_dims)
                .map(|d| d.iter().map(|&x| serde_json::json!(x)).collect())
                .unwrap_or_else(|| shape_src.0.clone());
            Some((out, dtype))
        }
        "Unsqueeze" => {
            let mut dims = inp0.0.clone();
            let axes: Vec<i64> = node
                .inputs
                .get(1)
                .and_then(|n| init_shapes.get(n))
                .map(|_| vec![1i64])
                .or_else(|| {
                    node.attrs
                        .get("axes")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect())
                })
                .unwrap_or_else(|| vec![0]);
            for ax in axes {
                let pos = ax.rem_euclid(dims.len() as i64 + 1) as usize;
                dims.insert(pos.min(dims.len()), serde_json::json!(1));
            }
            Some((dims, dtype))
        }
        "Squeeze" => {
            let mut dims = inp0.0.clone();
            dims.retain(|d| d.as_u64() != Some(1));
            Some((dims, dtype))
        }
        "Slice" => Some(inp0),
        "DynamicQuantizeLinear" => Some(inp0),
        "DynamicQuantizeLSTM" => {
            let hidden = node
                .attrs
                .get("hidden_size")
                .and_then(|v| v.as_i64())
                .unwrap_or(256) as usize;
            let bidir = node
                .attrs
                .get("direction")
                .and_then(|v| v.as_str())
                .is_some_and(|s| s == "bidirectional");
            let dirs = if bidir { 2 } else { 1 };
            let in_dims = resolve_static(&inp0.0, opts, sym_env)?;
            if in_dims.len() == 3 {
                let seq = *in_dims.first()?;
                let batch = in_dims.get(1).copied().unwrap_or(1);
                return Some((
                    vec![
                        serde_json::json!(seq),
                        serde_json::json!(dirs),
                        serde_json::json!(batch),
                        serde_json::json!(hidden),
                    ],
                    dtype,
                ));
            }
            None
        }
        "Reshape" => {
            let in_dims = resolve_static(&inp0.0, opts, sym_env)?;
            if in_dims.len() == 4
                && in_dims.get(1) == Some(&2)
                && in_dims.get(3).is_some_and(|&h| h >= 64)
            {
                return Some((
                    vec![
                        serde_json::json!(in_dims[0]),
                        serde_json::json!(in_dims.get(2).copied().unwrap_or(1)),
                        serde_json::json!(in_dims[3] * 2),
                    ],
                    dtype,
                ));
            }
            if in_dims.len() == 4
                && in_dims.get(2) == Some(&2)
                && in_dims.get(3).is_some_and(|&h| h >= 64)
            {
                return Some((
                    vec![
                        serde_json::json!(in_dims[0]),
                        serde_json::json!(in_dims[1]),
                        serde_json::json!(in_dims[3] * 2),
                    ],
                    dtype,
                ));
            }
            None
        }
        "Cast"
        | "Relu"
        | "Tanh"
        | "Sigmoid"
        | "Sqrt"
        | "Sin"
        | "Cos"
        | "Exp"
        | "Neg"
        | "Abs"
        | "Atan"
        | "Floor"
        | "Round"
        | "Erf"
        | "LeakyRelu"
        | "Clip"
        | "Softmax"
        | "LayerNormalization"
        | "InstanceNormalization"
        | "ReduceMean"
        | "ReduceSum"
        | "ReduceMax"
        | "ReduceMin"
        | "ReduceProd"
        | "CumSum"
        | "Pad" => Some(inp0),
        "Resize" => Some(inp0),
        _ => None,
    }
}

pub fn propagate_shapes(
    nodes: &mut [BundleNode],
    manifest: &BundleManifest,
    init_shapes: &HashMap<String, Vec<usize>>,
    opts: &ImportOptions,
) {
    let mut env: HashMap<String, TensorMeta> = HashMap::new();
    let mut sym_env: HashMap<String, usize> = HashMap::new();

    for (name, shape) in init_shapes {
        env.insert(name.clone(), (dims_from_usize(shape), "f32".to_string()));
    }
    for io in &manifest.inputs {
        let dims: Dims = io
            .meta
            .shape
            .iter()
            .map(|d| match d {
                serde_json::Value::Number(n) => serde_json::json!(n),
                serde_json::Value::String(s) => serde_json::json!(s),
                _ => serde_json::json!(1),
            })
            .collect();
        env.insert(io.name.clone(), (dims, normalize_dtype(&io.meta.dtype)));
    }
    for io in &manifest.outputs {
        let dims: Dims = io
            .meta
            .shape
            .iter()
            .map(|d| match d {
                serde_json::Value::Number(n) => serde_json::json!(n),
                serde_json::Value::String(s) => serde_json::json!(s),
                _ => serde_json::json!(1),
            })
            .collect();
        env.insert(io.name.clone(), (dims, normalize_dtype(&io.meta.dtype)));
    }

    let sorted = topo_sort_nodes(nodes.to_vec());
    let name_to_idx: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.name.clone(), i))
        .collect();

    for snode in sorted {
        let idx = match name_to_idx.get(&snode.name) {
            Some(&i) => i,
            None => continue,
        };
        let node = &mut nodes[idx];
        let hint = node
            .inputs
            .iter()
            .find(|s| !s.is_empty())
            .and_then(|n| get(&env, n))
            .map(|t| t.0.clone());
        for (i, out) in node.outputs.iter().enumerate() {
            if i >= node.output_meta.len() {
                continue;
            }
            let dt = node.output_meta[i]
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("f32");
            let dims = if meta_is_empty(&node.output_meta[i]) {
                infer_output(node, &env, init_shapes, opts, &sym_env)
            } else {
                node.output_meta[i]
                    .get("shape")
                    .and_then(|s| s.as_array())
                    .map(|a| (a.to_vec(), normalize_dtype(dt)))
            };
            let Some((dims, dtype)) = dims else {
                continue;
            };
            let dims = resolve_dims(&dims, opts, &mut sym_env, hint.as_ref());
            node.output_meta[i] = meta_from_dims(dims.clone(), &dtype);
            env.insert(out.clone(), (dims, dtype));
        }
    }
}
