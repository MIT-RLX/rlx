// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

thread_local! {
    /// Folded i64 CONSTANT values (axes/shape tensors) visible to shape inference.
    /// Opset-13+ ops (Unsqueeze/Squeeze/Reduce/…) pass axes as an INPUT tensor, not
    /// an attribute, so the meta pass needs the value to place the new/removed dim
    /// correctly. Populated by the caller (`set_shape_i64_consts`) from the same
    /// `i64_params` the lowerer uses. Empty for callers that don't set it (they keep
    /// the old attribute-only behavior).
    static SHAPE_I64_CONSTS: std::cell::RefCell<HashMap<String, Vec<i64>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Seed the folded i64 constants used by shape inference for input-form axes.
/// Call immediately before [`propagate_shapes`]; cleared automatically at the start
/// of each `propagate_shapes` is NOT done — callers own the lifetime, so re-set per
/// graph. Passing an empty map restores attribute-only behavior.
pub fn set_shape_i64_consts(consts: HashMap<String, Vec<i64>>) {
    SHAPE_I64_CONSTS.with(|c| *c.borrow_mut() = consts);
}

fn shape_i64_const(name: &str) -> Option<Vec<i64>> {
    SHAPE_I64_CONSTS.with(|c| c.borrow().get(name).cloned())
}

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
            // Preserve already-concrete dims verbatim. The channel-first (ncl) and
            // blc heuristics below otherwise clobber concrete values — forcing a
            // tensor's axis-0 to 1 (which mangles conv weight out-channels read as
            // `[Cout, Cin, K]`) and axis-2 to `sequence_length` (which collapses a
            // vocoder's upsampled length back to the input length). Only *symbolic*
            // dims need the heuristics to decide batch-vs-length.
            if d.as_u64().is_some() {
                return d.clone();
            }
            // Explicit per-name length override (`named_lengths`) wins over every
            // heuristic below — lets one graph carry two+ distinct dynamic lengths
            // (e.g. CFM cross-attention `text_length` vs `latent_length`) instead of
            // collapsing every length axis to `sequence_length`. Mirrors the same
            // lookup in `lower::resolve_dim_ir` so INPUT dims (resolved there) and
            // propagated intermediate dims (resolved here) agree. No-op when empty.
            if let Some(name) = d.as_str() {
                if let Some(&v) = opts.named_lengths.get(name) {
                    return serde_json::json!(v);
                }
            }
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
            // Recognize common named dynamic dims that models export instead of
            // the generic `sequence_length`/`unk__` (e.g. `batch_size`,
            // `text_length`, `latent_length`, `source_audio_len`, `max_duration`).
            // Batch-like names → 1; length/time-like names → the compile sequence
            // length. Only reached for names NOT already handled above, so this
            // cannot change the working `sequence_length`/`unk__` graphs.
            {
                let low = s.to_ascii_lowercase();
                let is_batch = low == "n" || low == "b" || low.contains("batch");
                let is_len = low.contains("length")
                    || low.contains("_len")
                    || low.ends_with("len")
                    || low.contains("seq")
                    || low.contains("duration")
                    || low.contains("frame")
                    || low.contains("audio");
                if is_batch {
                    return serde_json::json!(1);
                }
                if is_len {
                    return if opts.dynamic_sequence {
                        serde_json::json!("sequence_length")
                    } else {
                        serde_json::json!(opts.sequence_length)
                    };
                }
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
    // Numpy broadcasting: the shorter operand is right-aligned and its missing
    // LEADING dims are implicitly 1, so the result's leading `pad` dims come
    // entirely from the longer operand — NOT filled with 1. Filling with 1 here
    // collapsed a `bias[C] + act[1, L, C]` add to `[1, 1, C]`, wiping the length
    // axis (any downstream op that trusts the propagated meta, e.g. a Conv, then
    // inherited the collapse — MiraTTS/BiCodec detokenizer 249→1).
    let mut out: Dims = long[..pad].to_vec();
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
    } else if stride > 1 {
        // STRIDED forward conv length: (li + 2·pad − dilation·(k−1) − 1)/stride + 1.
        // Needed for an STFT-as-conv audio front-end (kernel 1024, stride 256,
        // F5-TTS): the length must shrink (13024 → 47), not pass through. Gated on
        // stride > 1 to preserve the long-standing `lo = li` behavior for STRIDE-1
        // convs — the ConvNeXt / VITS pattern is `explicit Pad → VALID conv` whose
        // net length is unchanged, but the Pad's added width is not always tracked
        // in the meta; there `lo = li` is the right net answer and the real formula
        // would (wrongly) shrink it and break a downstream residual add.
        let k = ws.last().copied().unwrap_or(1);
        let dil = node
            .attrs
            .get("dilations")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let eff = dil * k.saturating_sub(1);
        (li + 2 * pad).saturating_sub(eff).saturating_sub(1) / stride + 1
    } else {
        li
    };
    Some(vec![
        serde_json::json!(n),
        serde_json::json!(co),
        serde_json::json!(lo),
    ])
}

/// Compute an Einsum output shape from the equation and input dim lists.
/// Returns `None` for ellipsis equations or label/rank mismatches.
fn einsum_output_dims(equation: &str, shapes: &[Dims]) -> Option<Dims> {
    let eq: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    if eq.contains("...") {
        return None;
    }
    let (lhs, rhs) = match eq.split_once("->") {
        Some((l, r)) => (l.to_string(), Some(r.to_string())),
        None => (eq.clone(), None),
    };
    let terms: Vec<&str> = lhs.split(',').collect();
    if terms.len() != shapes.len() {
        return None;
    }
    let mut size: HashMap<char, serde_json::Value> = HashMap::new();
    let mut counts: HashMap<char, usize> = HashMap::new();
    for (term, dims) in terms.iter().zip(shapes) {
        let chars: Vec<char> = term.chars().collect();
        if chars.len() != dims.len() {
            return None;
        }
        for (c, d) in chars.iter().zip(dims) {
            size.entry(*c).or_insert_with(|| d.clone());
            *counts.entry(*c).or_insert(0) += 1;
        }
    }
    let out_chars: Vec<char> = match rhs {
        Some(r) => r.chars().collect(),
        None => {
            let mut once: Vec<char> = counts
                .iter()
                .filter(|&(_, &n)| n == 1)
                .map(|(&c, _)| c)
                .collect();
            once.sort_unstable();
            once
        }
    };
    out_chars.iter().map(|c| size.get(c).cloned()).collect()
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
        "GatherND" => {
            let data = get(env, &node.inputs[0])?;
            let idx = get(env, node.inputs.get(1).filter(|s| !s.is_empty())?)?;
            let batch = node
                .attrs
                .get("batch_dims")
                .and_then(|v| v.as_i64())
                .unwrap_or(0)
                .max(0) as usize;
            let k = idx.0.last().and_then(|d| d.as_u64())? as usize;
            // out = indices.shape[:-1] + data.shape[batch+k:]
            let mut out: Dims = idx.0[..idx.0.len().saturating_sub(1)].to_vec();
            out.extend(data.0.iter().skip(batch + k).cloned());
            Some((out, dtype))
        }
        "OneHot" => {
            let idx = get(env, &node.inputs[0])?;
            let axis = node
                .attrs
                .get("axis")
                .and_then(|v| v.as_i64())
                .unwrap_or(-1);
            let rank = idx.0.len() as i64;
            let pos = if axis < 0 { rank + 1 + axis } else { axis }.clamp(0, rank) as usize;
            // depth comes from input 1 (a scalar initializer); unknown → symbolic.
            let depth = node
                .inputs
                .get(1)
                .and_then(|n| init_shapes.get(n.as_str()))
                .and_then(|s| s.first().copied())
                .map(|d| serde_json::json!(d))
                .unwrap_or_else(|| serde_json::json!("onehot_depth"));
            let mut out = idx.0.clone();
            out.insert(pos.min(out.len()), depth);
            Some((out, "f32".to_string()))
        }
        "NonZero" => {
            let rank = inp0.0.len();
            Some((
                vec![serde_json::json!(rank), serde_json::json!("nonzero_count")],
                "i64".to_string(),
            ))
        }
        "Einsum" => {
            let equation = node.attrs.get("equation").and_then(|v| v.as_str())?;
            let shapes: Vec<Dims> = node
                .inputs
                .iter()
                .filter(|s| !s.is_empty())
                .map(|n| get(env, n).map(|t| t.0.clone()))
                .collect::<Option<_>>()?;
            let out = einsum_output_dims(equation, &shapes)?;
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
            // A Transpose's output rank equals its perm length; size `out` by the
            // perm (not the input rank) so a perm longer than the currently-inferred
            // input rank can't index out of bounds (guard both indices).
            let mut out = vec![serde_json::json!(1); perm.len().max(inp0.0.len())];
            for (i, &p) in perm.iter().enumerate() {
                if i < out.len() && p < inp0.0.len() {
                    out[i] = inp0.0[p].clone();
                }
            }
            Some((out, dtype))
        }
        "SequenceEmpty" => Some((vec![serde_json::json!(0)], "i64".to_string())),
        "Loop" => Some((
            vec![serde_json::json!(
                crate::control_flow::alignment_buffer_upper_bound(
                    opts.sequence_length,
                    opts.max_waveform_samples,
                    opts.max_frames_per_token,
                )
            )],
            "i64".to_string(),
        )),
        "ConcatFromSequence" => Some((
            vec![serde_json::json!(
                crate::control_flow::alignment_buffer_upper_bound(
                    opts.sequence_length,
                    opts.max_waveform_samples,
                    opts.max_frames_per_token,
                )
            )],
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
            // Axes: prefer the folded CONSTANT value of the axes-input tensor
            // (opset-13+ form) so a non-1 axis places the new dim correctly. The old
            // code hardcoded `[1]` whenever the axes input had a known shape, which
            // silently mis-placed e.g. `Unsqueeze(mask, [-1])` → `[1,1,seq]` instead
            // of `[1,seq,1]`, transposing attention masks (MOSS-TTS GPT-2). Fall back
            // to the `axes` attribute, then `[0]`.
            let axes: Vec<i64> = node
                .inputs
                .get(1)
                .filter(|n| !n.is_empty())
                .and_then(|n| shape_i64_const(n))
                .or_else(|| {
                    node.attrs
                        .get("axes")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect())
                })
                // axes input present but value unknown → single unsqueeze at 1 (legacy)
                .or_else(|| {
                    node.inputs
                        .get(1)
                        .filter(|n| !n.is_empty() && init_shapes.contains_key(*n))
                        .map(|_| vec![1i64])
                })
                .unwrap_or_else(|| vec![0]);
            // Insert low→high on the OUTPUT rank so multi-axis unsqueeze is stable.
            let out_rank = dims.len() as i64 + axes.len() as i64;
            let mut positions: Vec<usize> = axes
                .iter()
                .map(|&a| a.rem_euclid(out_rank) as usize)
                .collect();
            positions.sort_unstable();
            for pos in positions {
                dims.insert(pos.min(dims.len()), serde_json::json!(1));
            }
            Some((dims, dtype))
        }
        "Squeeze" => {
            let mut dims = inp0.0.clone();
            // Explicit axes (attr or folded input) squeeze only those positions;
            // otherwise drop all size-1 dims (ONNX default).
            let axes: Option<Vec<i64>> = node
                .inputs
                .get(1)
                .filter(|n| !n.is_empty())
                .and_then(|n| shape_i64_const(n))
                .or_else(|| {
                    node.attrs
                        .get("axes")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect())
                });
            match axes {
                Some(ax) if !ax.is_empty() => {
                    let rank = dims.len() as i64;
                    let mut drop: Vec<usize> = ax
                        .iter()
                        .map(|&a| a.rem_euclid(rank.max(1)) as usize)
                        .collect();
                    drop.sort_unstable_by(|a, b| b.cmp(a));
                    for pos in drop {
                        if pos < dims.len() && dims[pos].as_u64() == Some(1) {
                            dims.remove(pos);
                        }
                    }
                }
                _ => dims.retain(|d| d.as_u64() != Some(1)),
            }
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
        | "CumProd"
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
