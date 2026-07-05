// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! The aten→rlx op registry: `torch-ir.json` (Core ATen) → [`Lowered`].
//!
//! One handler per aten op family reads its args (shapes are already concrete,
//! carried in `node.out`) and appends [`Instr`]s in the rlx [`Call`] vocabulary.
//! Unsupported ops are collected up-front and reported together so a user sees
//! the whole coverage gap at once.

use crate::call::*;
use crate::ir::*;
use anyhow::{Context, Result, anyhow, bail};
use rlx_ir::hir::{GridMode, GridPad};
use rlx_ir::op::{Activation, BinaryOp, MaskKind, ReduceOp};

/// `aten.native_layer_norm.default` → (`native_layer_norm`, `default`).
fn op_key(op: &str) -> (String, String) {
    if let Some(rest) = op.strip_prefix("aten.") {
        let mut it = rest.splitn(2, '.');
        let base = it.next().unwrap_or(rest).to_string();
        let over = it.next().unwrap_or("").to_string();
        (base, over)
    } else {
        (op.to_string(), String::new())
    }
}

/// Ops the registry can lower. Keep in sync with the match in [`lower`].
const SUPPORTED: &[&str] = &[
    // shape
    "view",
    "_unsafe_view",
    "reshape",
    "flatten",
    "unsqueeze",
    "squeeze",
    "permute",
    "transpose",
    "t",
    "expand",
    "slice",
    "cat",
    "clone",
    "contiguous",
    "detach",
    "alias",
    "lift_fresh_copy",
    "_to_copy",
    "to",
    "_getitem",
    "select",
    "split_with_sizes",
    "split",
    "chunk",
    "full",
    "full_like",
    "constant_pad_nd",
    // constant producers (uninitialized `empty` treated as zeros)
    "zeros",
    "zeros_like",
    "new_zeros",
    "ones",
    "ones_like",
    "new_ones",
    "empty",
    "empty_like",
    "new_empty",
    // linalg
    "mm",
    "bmm",
    "matmul",
    "addmm",
    "linear",
    "baddbmm",
    // elementwise
    "add",
    "sub",
    "mul",
    "div",
    "rsub",
    "pow",
    "clamp",
    "clamp_min",
    "clamp_max",
    "reciprocal",
    // activations
    "gelu",
    "relu",
    "silu",
    "sigmoid",
    "tanh",
    "exp",
    "sqrt",
    "rsqrt",
    "neg",
    "abs",
    "sin",
    "cos",
    "leaky_relu",
    "hardtanh",
    "hardsigmoid",
    "hardswish",
    // comparison / select / broadcast / ramp
    "le",
    "ge",
    "lt",
    "gt",
    "eq",
    "ne",
    "where",
    "arange",
    "gather",
    "masked_fill",
    // norms
    "layer_norm",
    "native_layer_norm",
    "rms_norm",
    "_native_batch_norm_legit_no_training",
    "batch_norm",
    "native_batch_norm",
    "_native_batch_norm_legit",
    "group_norm",
    "native_group_norm",
    // nn
    "embedding",
    "_softmax",
    "softmax",
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_attention_math",
    "convolution",
    "index",
    "upsample_nearest2d",
    "_upsample_nearest_exact2d",
    "upsample_bilinear2d",
    "upsample_bicubic2d",
    "_upsample_bilinear2d_aa",
    "_upsample_bicubic2d_aa",
    "pixel_shuffle",
    "pixel_unshuffle",
    "grid_sampler",
    "grid_sampler_2d",
    // pooling
    "max_pool2d",
    "max_pool2d_with_indices",
    "avg_pool2d",
    "adaptive_avg_pool2d",
    "_adaptive_avg_pool2d",
    // reduce / routing
    "mean",
    "sum",
    "topk",
];

/// Export-only no-op ops (assertions, symbolic guards) — dropped entirely.
fn is_noop(base: &str) -> bool {
    base.starts_with("_assert") || base == "sym_constrain_range_for_size"
}

/// Normalize a possibly-negative axis against `rank`.
fn norm_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (axis + rank as i64).max(0) as usize
    } else {
        axis as usize
    }
}

pub fn lower(ir: &TorchIr) -> Result<Lowered> {
    // ── coverage check ───────────────────────────────────────────────────────
    let mut unsupported: Vec<String> = Vec::new();
    for n in &ir.nodes {
        let (base, _) = op_key(&n.op);
        if is_noop(&base) {
            continue;
        }
        if !SUPPORTED.contains(&base.as_str()) && !unsupported.contains(&n.op) {
            unsupported.push(n.op.clone());
        }
    }
    if !unsupported.is_empty() {
        unsupported.sort();
        bail!(
            "aten→rlx registry does not yet support {} op(s):\n  {}\n\
             Extend crates/io/rlx-torch-import/src/lower.rs (add a handler + a \
             SUPPORTED entry).",
            unsupported.len(),
            unsupported.join("\n  ")
        );
    }

    let mut lo = Lowered {
        name: ir.model_name.clone(),
        ..Default::default()
    };

    for i in &ir.inputs {
        let dyn_dims: Vec<Option<u32>> = match &i.dynamic {
            Some(marks) => marks
                .iter()
                .map(|&m| (m >= 0).then_some(m as u32))
                .collect(),
            None => vec![None; i.shape.len()],
        };
        lo.inputs.push(InputDef {
            name: i.id.clone(),
            shape: dims_usize(&i.shape),
            dyn_dims,
            dtype: dtype_from_str(&i.dtype)?,
        });
    }
    for w in &ir.weights {
        lo.params.push(ParamDef {
            value_id: w.id.clone(),
            key: w.key.clone(),
            shape: dims_usize(&w.shape),
            dtype: dtype_from_str(&w.dtype)?,
        });
    }

    let mut zero_ctr = 0usize;
    // node id → per-index result names, for tuple-producing ops (split, ...).
    let mut multi: std::collections::HashMap<String, Vec<Value>> = std::collections::HashMap::new();
    // value id → shape, so handlers can recover an operand's *input* rank
    // (topological order guarantees predecessors are populated first).
    let mut value_shape: std::collections::HashMap<String, Vec<i64>> =
        std::collections::HashMap::new();
    let mut value_dtype: std::collections::HashMap<String, rlx_ir::DType> =
        std::collections::HashMap::new();
    for i in &ir.inputs {
        value_shape.insert(i.id.clone(), i.shape.clone());
        value_dtype.insert(i.id.clone(), dtype_from_str(&i.dtype)?);
    }
    for w in &ir.weights {
        value_shape.insert(w.id.clone(), w.shape.clone());
        value_dtype.insert(w.id.clone(), dtype_from_str(&w.dtype)?);
    }
    for n in &ir.nodes {
        lower_node(
            n,
            &mut lo,
            &mut zero_ctr,
            &mut multi,
            &value_shape,
            &value_dtype,
        )
        .with_context(|| format!("lowering node {} ({})", n.id, n.op))?;
        if let Some((s, dt)) = primary_out(&n.out) {
            value_shape.insert(n.id.clone(), s);
            if let Ok(d) = dtype_from_str(&dt) {
                // Intermediates live on the f32 host arena — a byte-sized int/bool
                // node would be mis-read. Track non-float outputs as F32 (their
                // small integer values are exact) so broadcasts/casts stay f32.
                let d = if is_float_dtype(d) {
                    d
                } else {
                    rlx_ir::DType::F32
                };
                value_dtype.insert(n.id.clone(), d);
            }
        }
        set_note(&mut lo, &n.id, node_note(n, &value_shape));
    }

    // Original aten op histogram (for the generated provenance header).
    let mut hist: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for n in &ir.nodes {
        *hist.entry(n.op.clone()).or_insert(0) += 1;
    }
    let mut hist: Vec<(String, usize)> = hist.into_iter().collect();
    hist.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    lo.source_histogram = hist;

    // outputs
    for o in &ir.outputs {
        if let Some(r) = &o.reference {
            lo.outputs.push(r.clone());
        } else {
            bail!("constant graph outputs are not supported");
        }
    }
    Ok(lo)
}

/// Resolve arg `i` as a node/value reference.
fn arg_ref(n: &NodeDef, i: usize) -> Result<Value> {
    n.args
        .get(i)
        .and_then(|a| a.as_ref_name())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("arg {i} of {} is not a tensor ref", n.op))
}

fn out_shape(n: &NodeDef) -> Result<Vec<i64>> {
    primary_out(&n.out)
        .map(|(s, _)| s)
        .ok_or_else(|| anyhow!("node {} has no tensor output shape", n.id))
}

fn out_dtype(n: &NodeDef) -> Result<rlx_ir::DType> {
    let (_, dt) =
        primary_out(&n.out).ok_or_else(|| anyhow!("node {} has no output dtype", n.id))?;
    dtype_from_str(&dt)
}

fn push(lo: &mut Lowered, result: &str, call: Call) {
    lo.instrs.push(Instr {
        result: result.to_string(),
        call,
        note: None,
    });
}

/// Attach a provenance comment to the instruction that defines `result`.
fn set_note(lo: &mut Lowered, result: &str, note: String) {
    if let Some(ins) = lo.instrs.iter_mut().rev().find(|i| i.result == result) {
        ins.note = Some(note);
    }
}

/// Format an aten node's provenance: `op  (arg[shape], ...) -> [shape] dtype`.
fn node_note(n: &NodeDef, value_shape: &std::collections::HashMap<String, Vec<i64>>) -> String {
    let ins: Vec<String> = n
        .args
        .iter()
        .filter_map(|a| a.as_ref_name())
        .map(|nm| match value_shape.get(nm) {
            Some(s) => format!("{nm}{s:?}"),
            None => nm.to_string(),
        })
        .collect();
    let out = match primary_out(&n.out) {
        Some((s, dt)) => format!("{s:?} {dt}"),
        None => "()".to_string(),
    };
    format!("{}  ({}) -> {}", n.op, ins.join(", "), out)
}

/// Materialize a broadcastable rank-1 scalar constant, return its value name.
fn scalar_const(lo: &mut Lowered, id: &str, value: f64, dtype: rlx_ir::DType) -> Value {
    let nm = format!("{id}__scalar");
    push(
        lo,
        &nm,
        Call::Full {
            value,
            shape: vec![1],
            dtype,
        },
    );
    nm
}

/// Broadcast a (non-scalar) tensor operand up to `target` if its shape differs.
/// rlx elementwise kernels don't do full outer broadcasting (e.g. `[1,S]` op
/// `[S,1]` → `[S,S]`), so pre-expand both sides to the result shape.
fn bcast_ref(
    lo: &mut Lowered,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
    value_dtype: &std::collections::HashMap<String, rlx_ir::DType>,
    name: &str,
    target: &[usize],
    tag: &str,
) -> Value {
    match value_shape.get(name) {
        Some(s) if dims_usize(s) != *target && s.iter().product::<i64>() > 1 => {
            let dt = value_dtype.get(name).copied().unwrap_or(rlx_ir::DType::F32);
            let nm = format!("{name}__bc{tag}");
            let ti: Vec<i64> = target.iter().map(|&d| d as i64).collect();
            push(
                lo,
                &nm,
                Call::Node(crate::nodeop::NodeOp::Expand {
                    x: name.to_string(),
                    target: ti,
                    out: target.to_vec(),
                    out_dtype: dt,
                }),
            );
            nm
        }
        _ => name.to_string(),
    }
}

fn lower_node(
    n: &NodeDef,
    lo: &mut Lowered,
    zero_ctr: &mut usize,
    multi: &mut std::collections::HashMap<String, Vec<Value>>,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
    value_dtype: &std::collections::HashMap<String, rlx_ir::DType>,
) -> Result<()> {
    let (base, over) = op_key(&n.op);
    let id = n.id.clone();
    if is_noop(&base) {
        return Ok(());
    }
    match base.as_str() {
        // ── pure shape / aliases ────────────────────────────────────────────
        "view" | "_unsafe_view" | "reshape" | "flatten" | "unsqueeze" | "squeeze" => {
            let x = arg_ref(n, 0)?;
            push(
                lo,
                &id,
                Call::Reshape {
                    x,
                    shape: out_shape(n)?,
                },
            );
        }
        "expand" => {
            let x = arg_ref(n, 0)?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("expand has no out"))?;
            push(
                lo,
                &id,
                Call::Node(crate::nodeop::NodeOp::Expand {
                    x,
                    target: out.clone(),
                    out: dims_usize(&out),
                    out_dtype: dtype_from_str(&dt)?,
                }),
            );
        }
        "permute" => {
            let x = arg_ref(n, 0)?;
            let perm = n.args[1]
                .as_int_list()
                .ok_or_else(|| anyhow!("permute needs an int-list perm"))?;
            let rank = perm.len();
            let perm = perm.iter().map(|&p| norm_axis(p, rank)).collect();
            push(lo, &id, Call::Transpose { x, perm });
        }
        "transpose" => {
            let x = arg_ref(n, 0)?;
            let rank = out_shape(n)?.len();
            let d0 = norm_axis(n.args[1].as_int().unwrap_or(0), rank);
            let d1 = norm_axis(n.args[2].as_int().unwrap_or(1), rank);
            let mut perm: Vec<usize> = (0..rank).collect();
            perm.swap(d0, d1);
            push(lo, &id, Call::Transpose { x, perm });
        }
        "t" => {
            let x = arg_ref(n, 0)?;
            let rank = out_shape(n)?.len();
            let perm = if rank == 2 {
                vec![1, 0]
            } else {
                (0..rank).collect()
            };
            push(lo, &id, Call::Transpose { x, perm });
        }
        "slice" => {
            // slice.Tensor(x, dim, start, end, step)
            let x = arg_ref(n, 0)?;
            let rank = out_shape(n)?.len();
            let dim = norm_axis(n.args.get(1).and_then(|a| a.as_int()).unwrap_or(0), rank);
            let start = n.args.get(2).and_then(|a| a.as_int()).unwrap_or(0).max(0) as usize;
            let step = n.args.get(4).and_then(|a| a.as_int()).unwrap_or(1);
            if step != 1 {
                bail!("slice with step {step} != 1 is not supported");
            }
            let len = out_shape(n)?[dim] as usize;
            push(
                lo,
                &id,
                Call::Narrow {
                    x,
                    axis: dim,
                    start,
                    len,
                },
            );
        }
        "cat" => {
            let xs_all = n.args[0]
                .as_ref_list()
                .ok_or_else(|| anyhow!("cat needs a list of tensor refs"))?;
            // Drop empty (numel-0) inputs — e.g. empty KV-cache placeholders on
            // the first decode step (`cat([past_empty, key])`).
            let xs: Vec<Value> = xs_all
                .into_iter()
                .filter(|nm| {
                    value_shape
                        .get(nm)
                        .map(|s| s.iter().product::<i64>() != 0)
                        .unwrap_or(true)
                })
                .collect();
            if xs.is_empty() {
                bail!("cat with only empty inputs");
            }
            if xs.len() == 1 {
                push(lo, &id, Call::Alias(xs.into_iter().next().unwrap()));
            } else {
                let rank = out_shape(n)?.len();
                let axis = norm_axis(n.args.get(1).and_then(|a| a.as_int()).unwrap_or(0), rank);
                push(lo, &id, Call::Concat { xs, axis });
            }
        }
        "clone" | "contiguous" | "detach" | "alias" | "lift_fresh_copy" => {
            push(lo, &id, Call::Alias(arg_ref(n, 0)?));
        }
        "_to_copy" | "to" => {
            let x = arg_ref(n, 0)?;
            let want = out_dtype(n)?;
            // Alias only when the *input* dtype already matches; otherwise cast.
            if value_dtype.get(&x).copied() == Some(want) {
                push(lo, &id, Call::Alias(x));
            } else {
                push(lo, &id, Call::Cast { x, to: want });
            }
        }
        "_getitem" => {
            // getitem(parent, index) — index into a tuple-producing op's outputs.
            let parent = arg_ref(n, 0)?;
            let index = n.args.get(1).and_then(|a| a.as_int()).unwrap_or(0) as usize;
            if let Some(outs) = multi.get(&parent) {
                let v = outs
                    .get(index)
                    .ok_or_else(|| anyhow!("getitem index {index} out of range for {parent}"))?;
                push(lo, &id, Call::Alias(v.clone()));
            } else if index == 0 {
                push(lo, &id, Call::Alias(parent));
            } else {
                bail!("getitem index {index} (secondary output of {parent}) not supported");
            }
        }
        "select" => {
            // select.int(x, dim, index) — pick one index along dim, drop the dim.
            let x = arg_ref(n, 0)?;
            let out = out_shape(n)?;
            let rank_in = out.len() + 1;
            let dim = norm_axis(n.args.get(1).and_then(|a| a.as_int()).unwrap_or(0), rank_in);
            let index = n.args.get(2).and_then(|a| a.as_int()).unwrap_or(0).max(0) as usize;
            let tmp = format!("{id}__sel");
            push(
                lo,
                &tmp,
                Call::Narrow {
                    x,
                    axis: dim,
                    start: index,
                    len: 1,
                },
            );
            push(lo, &id, Call::Reshape { x: tmp, shape: out });
        }
        "split_with_sizes" => {
            // split_with_sizes(x, sizes, dim) → tuple of narrows.
            let x = arg_ref(n, 0)?;
            let outs = all_outs(&n.out);
            let rank = outs
                .first()
                .and_then(|o| o.as_ref())
                .map(|(s, _)| s.len())
                .unwrap_or(1);
            let dim = norm_axis(n.args.get(2).and_then(|a| a.as_int()).unwrap_or(0), rank);
            let sizes = n.args[1]
                .as_int_list()
                .ok_or_else(|| anyhow!("split_with_sizes needs an int-list of sizes"))?;
            let mut start = 0usize;
            let mut names = Vec::with_capacity(sizes.len());
            for (i, &sz) in sizes.iter().enumerate() {
                let nm = format!("{id}__s{i}");
                push(
                    lo,
                    &nm,
                    Call::Narrow {
                        x: x.clone(),
                        axis: dim,
                        start,
                        len: sz as usize,
                    },
                );
                start += sz as usize;
                names.push(nm);
            }
            multi.insert(id.clone(), names);
        }
        "split" | "chunk" => {
            // split.Tensor(x, split_size, dim) / chunk(x, chunks, dim) → tuple of
            // narrows. Piece sizes come straight from the concrete per-output
            // shapes, so equal `split`, remainder `split`, and `chunk` are one path.
            let x = arg_ref(n, 0)?;
            let outs = all_outs(&n.out);
            let rank = outs
                .first()
                .and_then(|o| o.as_ref())
                .map(|(s, _)| s.len())
                .unwrap_or(1);
            let dim = norm_axis(n.args.get(2).and_then(|a| a.as_int()).unwrap_or(0), rank);
            let mut start = 0usize;
            let mut names = Vec::with_capacity(outs.len());
            for (i, o) in outs.iter().enumerate() {
                let (oshape, _) = o
                    .as_ref()
                    .ok_or_else(|| anyhow!("split/chunk output {i} is not a tensor"))?;
                let len = *oshape
                    .get(dim)
                    .ok_or_else(|| anyhow!("split/chunk dim {dim} out of range"))?
                    as usize;
                let nm = format!("{id}__s{i}");
                push(
                    lo,
                    &nm,
                    Call::Narrow {
                        x: x.clone(),
                        axis: dim,
                        start,
                        len,
                    },
                );
                start += len;
                names.push(nm);
            }
            multi.insert(id.clone(), names);
        }
        "zeros" | "zeros_like" | "new_zeros" | "empty" | "empty_like" | "new_empty" => {
            lower_const_fill(n, lo, 0.0)?
        }
        "ones" | "ones_like" | "new_ones" => lower_const_fill(n, lo, 1.0)?,
        "full" => {
            let (shape, dt) =
                primary_out(&n.out).ok_or_else(|| anyhow!("full has no out shape"))?;
            let value = n.args.get(1).and_then(|a| a.as_float()).unwrap_or(0.0);
            push(
                lo,
                &id,
                Call::Full {
                    value,
                    shape: dims_usize(&shape),
                    dtype: dtype_from_str(&dt)?,
                },
            );
        }
        "full_like" => {
            let (shape, dt) =
                primary_out(&n.out).ok_or_else(|| anyhow!("full_like has no out shape"))?;
            let value = n.args.get(1).and_then(|a| a.as_float()).unwrap_or(0.0);
            push(
                lo,
                &id,
                Call::Full {
                    value,
                    shape: dims_usize(&shape),
                    dtype: dtype_from_str(&dt)?,
                },
            );
        }

        // ── linalg ──────────────────────────────────────────────────────────
        "mm" | "bmm" | "matmul" => {
            push(lo, &id, Call::Mm(arg_ref(n, 0)?, arg_ref(n, 1)?));
        }
        "addmm" => {
            // addmm(bias, m1, m2) = bias + m1 @ m2
            let bias = arg_ref(n, 0)?;
            let m1 = arg_ref(n, 1)?;
            let m2 = arg_ref(n, 2)?;
            let tmp = format!("{id}__mm");
            push(lo, &tmp, Call::Mm(m1, m2));
            push(lo, &id, Call::Binary(BinaryOp::Add, tmp, bias));
        }
        "linear" => {
            // linear(x, weight[out,in], bias?) = x @ weightᵀ + bias
            let x = arg_ref(n, 0)?;
            let w = arg_ref(n, 1)?;
            let wt = format!("{id}__wt");
            let rank_w = 2usize;
            let perm = if rank_w == 2 { vec![1, 0] } else { vec![] };
            push(lo, &wt, Call::Transpose { x: w, perm });
            let mm = format!("{id}__mm");
            push(lo, &mm, Call::Mm(x, wt));
            match n.args.get(2) {
                Some(b) if !b.is_none() => {
                    let bias = b.as_ref_name().unwrap().to_string();
                    push(lo, &id, Call::Binary(BinaryOp::Add, mm, bias));
                }
                _ => push(lo, &id, Call::Alias(mm)),
            }
        }

        // ── elementwise ─────────────────────────────────────────────────────
        "add" | "sub" | "mul" | "div" => {
            let op = match base.as_str() {
                "add" => BinaryOp::Add,
                "sub" => BinaryOp::Sub,
                "mul" => BinaryOp::Mul,
                _ => BinaryOp::Div,
            };
            if let Some(alpha) = n.kwargs.get("alpha").and_then(|a| a.as_float()) {
                if (alpha - 1.0).abs() > 1e-9 {
                    bail!("{}.{} with alpha={alpha} != 1 not supported", base, over);
                }
            }
            let a = arg_ref(n, 0)?;
            match n.args.get(1) {
                Some(Arg::Ref { reference }) => {
                    push(lo, &id, Call::Binary(op, a, reference.clone()));
                }
                Some(other) => {
                    let v = other
                        .as_float()
                        .ok_or_else(|| anyhow!("{}.{}: non-scalar rhs", base, over))?;
                    let c = scalar_const(lo, &id, v, out_dtype(n)?);
                    push(lo, &id, Call::Binary(op, a, c));
                }
                None => bail!("{}.{}: missing rhs", base, over),
            }
        }
        "reciprocal" => {
            // reciprocal(x) = 1 / x
            let x = arg_ref(n, 0)?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("reciprocal has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let one = scalar_const(lo, &id, 1.0, dt);
            push(
                lo,
                &id,
                Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                    op: BinaryOp::Div,
                    a: one,
                    b: x,
                    out: dims_usize(&out),
                    out_dtype: dt,
                }),
            );
        }
        "rsub" => {
            // rsub(x, scalar) = scalar - x
            let x = arg_ref(n, 0)?;
            let v = n
                .args
                .get(1)
                .and_then(|a| a.as_float())
                .ok_or_else(|| anyhow!("rsub needs a scalar"))?;
            let c = scalar_const(lo, &id, v, out_dtype(n)?);
            push(lo, &id, Call::Binary(BinaryOp::Sub, c, x));
        }
        "pow" => {
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("pow has no out"))?;
            let dt = dtype_from_str(&dt)?;
            // pow.Scalar(base_scalar, exp_tensor) has a scalar base; the
            // Tensor_Scalar / Tensor_Tensor overloads have a tensor base.
            let (a, b) = if over == "Scalar" {
                let base = n
                    .args
                    .first()
                    .and_then(|a| a.as_float())
                    .ok_or_else(|| anyhow!("pow.Scalar base"))?;
                (scalar_const(lo, &id, base, dt), arg_ref(n, 1)?)
            } else {
                let x = arg_ref(n, 0)?;
                let rhs = match n.args.get(1) {
                    Some(Arg::Ref { reference }) => reference.clone(),
                    Some(other) => {
                        let v = other
                            .as_float()
                            .ok_or_else(|| anyhow!("pow: non-scalar exp"))?;
                        scalar_const(lo, &id, v, dt)
                    }
                    None => bail!("pow: missing exponent"),
                };
                (x, rhs)
            };
            push(
                lo,
                &id,
                Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                    op: BinaryOp::Pow,
                    a,
                    b,
                    out: dims_usize(&out),
                    out_dtype: dt,
                }),
            );
        }
        "clamp" | "clamp_min" | "clamp_max" => {
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("clamp has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let out = dims_usize(&out);
            let mut cur = arg_ref(n, 0)?;
            let lo_v = if base == "clamp_max" {
                None
            } else {
                n.args.get(1).and_then(|a| a.as_float())
            };
            let hi_v = if base == "clamp_min" {
                None
            } else if base == "clamp_max" {
                n.args.get(1).and_then(|a| a.as_float())
            } else {
                n.args.get(2).and_then(|a| a.as_float())
            };
            if let Some(lv) = lo_v {
                let c = scalar_const(lo, &format!("{id}__lo"), lv, dt);
                let nm = format!("{id}__cmin");
                push(
                    lo,
                    &nm,
                    Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                        op: BinaryOp::Max,
                        a: cur,
                        b: c,
                        out: out.clone(),
                        out_dtype: dt,
                    }),
                );
                cur = nm;
            }
            if let Some(hv) = hi_v {
                let c = scalar_const(lo, &format!("{id}__hi"), hv, dt);
                push(
                    lo,
                    &id,
                    Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                        op: BinaryOp::Min,
                        a: cur,
                        b: c,
                        out,
                        out_dtype: dt,
                    }),
                );
            } else {
                push(lo, &id, Call::Alias(cur));
            }
        }

        // ── activations ─────────────────────────────────────────────────────
        "gelu" => {
            let x = arg_ref(n, 0)?;
            let approx = n
                .kwargs
                .get("approximate")
                .and_then(|a| match a {
                    Arg::Str { v } => Some(v.as_str()),
                    _ => None,
                })
                .unwrap_or("none");
            let act = if approx == "tanh" {
                Activation::GeluApprox
            } else {
                Activation::Gelu
            };
            push(lo, &id, Call::Act(act, x));
        }
        "relu" => push(lo, &id, Call::Act(Activation::Relu, arg_ref(n, 0)?)),
        "silu" => push(lo, &id, Call::Act(Activation::Silu, arg_ref(n, 0)?)),
        "sigmoid" => push(lo, &id, Call::Act(Activation::Sigmoid, arg_ref(n, 0)?)),
        "tanh" => push(lo, &id, Call::Act(Activation::Tanh, arg_ref(n, 0)?)),
        "exp" => push(lo, &id, Call::Act(Activation::Exp, arg_ref(n, 0)?)),
        "sqrt" => push(lo, &id, Call::Act(Activation::Sqrt, arg_ref(n, 0)?)),
        "rsqrt" => push(lo, &id, Call::Act(Activation::Rsqrt, arg_ref(n, 0)?)),
        "neg" => push(lo, &id, Call::Act(Activation::Neg, arg_ref(n, 0)?)),
        "abs" => push(lo, &id, Call::Act(Activation::Abs, arg_ref(n, 0)?)),
        "sin" => push(lo, &id, Call::Act(Activation::Sin, arg_ref(n, 0)?)),
        "cos" => push(lo, &id, Call::Act(Activation::Cos, arg_ref(n, 0)?)),
        "leaky_relu" => {
            // leaky_relu(x, slope=0.01) = max(x, slope·x)  (exact for 0 ≤ slope ≤ 1).
            let x = arg_ref(n, 0)?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("leaky_relu has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let out = dims_usize(&out);
            let slope = n.args.get(1).and_then(|a| a.as_float()).unwrap_or(0.01);
            let s = scalar_const(lo, &format!("{id}_lslope"), slope, dt);
            let sx = format!("{id}__lrx");
            push(lo, &sx, Call::Binary(BinaryOp::Mul, x.clone(), s));
            push(
                lo,
                &id,
                Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                    op: BinaryOp::Max,
                    a: x,
                    b: sx,
                    out,
                    out_dtype: dt,
                }),
            );
        }
        "hardtanh" => {
            // hardtanh(x, min=-1, max=1) = min(max(x, min), max).
            let x = arg_ref(n, 0)?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("hardtanh has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let out = dims_usize(&out);
            let lo_v = n.args.get(1).and_then(|a| a.as_float()).unwrap_or(-1.0);
            let hi_v = n.args.get(2).and_then(|a| a.as_float()).unwrap_or(1.0);
            let lc = scalar_const(lo, &format!("{id}_htlo"), lo_v, dt);
            let t = format!("{id}__ht");
            push(
                lo,
                &t,
                Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                    op: BinaryOp::Max,
                    a: x,
                    b: lc,
                    out: out.clone(),
                    out_dtype: dt,
                }),
            );
            let hc = scalar_const(lo, &format!("{id}_hthi"), hi_v, dt);
            push(
                lo,
                &id,
                Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                    op: BinaryOp::Min,
                    a: t,
                    b: hc,
                    out,
                    out_dtype: dt,
                }),
            );
        }
        "hardsigmoid" => {
            let x = arg_ref(n, 0)?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("hardsigmoid has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let out = dims_usize(&out);
            let hs = push_hardsigmoid(lo, &id, x, &out, dt);
            push(lo, &id, Call::Alias(hs));
        }
        "hardswish" => {
            // hardswish(x) = x · hardsigmoid(x).
            let x = arg_ref(n, 0)?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("hardswish has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let out = dims_usize(&out);
            let hs = push_hardsigmoid(lo, &id, x.clone(), &out, dt);
            push(lo, &id, Call::Binary(BinaryOp::Mul, x, hs));
        }
        "masked_fill" => {
            // masked_fill(x, bool_mask, value). `Op::Where` can't read a bool cond
            // on the f32 arena, so use arithmetic (as the sdpa mask path does):
            //   out = x + mask·(value − x)   with mask cast to f32 {0,1}.
            let x = arg_ref(n, 0)?;
            let mask = arg_ref(n, 1)?;
            let value = n
                .args
                .get(2)
                .and_then(|a| a.as_float())
                .ok_or_else(|| anyhow!("masked_fill: only scalar fill value supported"))?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("masked_fill has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let out = dims_usize(&out);
            let maskf0 = format!("{id}__mff");
            push(
                lo,
                &maskf0,
                Call::Cast {
                    x: mask.clone(),
                    to: rlx_ir::DType::F32,
                },
            );
            // Broadcast the mask up to the output shape if it is smaller.
            let mask_shape = value_shape
                .get(&mask)
                .cloned()
                .unwrap_or_else(|| out.iter().map(|&d| d as i64).collect());
            let maskf = if dims_usize(&mask_shape) == out {
                maskf0
            } else {
                let me = format!("{id}__mfe");
                let target: Vec<i64> = out.iter().map(|&d| d as i64).collect();
                push(
                    lo,
                    &me,
                    Call::Node(crate::nodeop::NodeOp::Expand {
                        x: maskf0,
                        target,
                        out: out.clone(),
                        out_dtype: rlx_ir::DType::F32,
                    }),
                );
                me
            };
            let valc = format!("{id}__mfv");
            push(
                lo,
                &valc,
                Call::Full {
                    value,
                    shape: out.clone(),
                    dtype: dt,
                },
            );
            let diff = format!("{id}__mfd");
            push(lo, &diff, Call::Binary(BinaryOp::Sub, valc, x.clone())); // value − x
            let md = format!("{id}__mfm");
            push(lo, &md, Call::Binary(BinaryOp::Mul, maskf, diff));
            push(lo, &id, Call::Binary(BinaryOp::Add, x, md)); // x + mask·(value−x)
        }

        // ── norms ───────────────────────────────────────────────────────────
        "layer_norm" | "native_layer_norm" => {
            // layer_norm(x, normalized_shape, weight, bias, eps, ...). weight/bias
            // are None when elementwise_affine=False (e.g. adaLN in FLUX) —
            // synthesize gamma=ones / beta=zeros of the normalized shape.
            let x = arg_ref(n, 0)?;
            let nsh = dims_usize(
                &n.args
                    .get(1)
                    .and_then(|a| a.as_int_list())
                    .unwrap_or_default(),
            );
            let gamma = match n.args.get(2) {
                Some(Arg::Ref { reference }) => reference.clone(),
                _ => {
                    let g = format!("{id}__lng");
                    push(
                        lo,
                        &g,
                        Call::Full {
                            value: 1.0,
                            shape: nsh.clone(),
                            dtype: rlx_ir::DType::F32,
                        },
                    );
                    g
                }
            };
            let beta = match n.args.get(3) {
                Some(Arg::Ref { reference }) => reference.clone(),
                _ => {
                    let b = format!("{id}__lnb");
                    push(
                        lo,
                        &b,
                        Call::Full {
                            value: 0.0,
                            shape: nsh,
                            dtype: rlx_ir::DType::F32,
                        },
                    );
                    b
                }
            };
            let eps = n.args.get(4).and_then(|a| a.as_float()).unwrap_or(1e-5) as f32;
            push(
                lo,
                &id,
                Call::Ln {
                    x,
                    gamma,
                    beta,
                    eps,
                },
            );
        }
        "rms_norm" => {
            // rms_norm(x, normalized_shape, weight, eps) — no beta in torch.
            let x = arg_ref(n, 0)?;
            let gamma = arg_ref(n, 2)?;
            let eps = n.args.get(3).and_then(|a| a.as_float()).unwrap_or(1e-6) as f32;
            let (gshape, gdt) = lo
                .params
                .iter()
                .find(|p| p.value_id == gamma)
                .map(|p| (p.shape.clone(), p.dtype))
                .ok_or_else(|| anyhow!("rms_norm gamma {gamma} is not a param"))?;
            let beta_key = format!("_rlx_zero_beta_{}", *zero_ctr);
            *zero_ctr += 1;
            lo.zero_params.push(ParamDef {
                value_id: beta_key.clone(),
                key: beta_key.clone(),
                shape: gshape,
                dtype: gdt,
            });
            push(
                lo,
                &id,
                Call::RmsNorm {
                    x,
                    gamma,
                    beta: beta_key,
                    eps,
                },
            );
        }

        // ── nn ──────────────────────────────────────────────────────────────
        "embedding" => {
            // embedding(weight, indices, ...)
            let table = arg_ref(n, 0)?;
            let indices = arg_ref(n, 1)?;
            push(
                lo,
                &id,
                Call::Gather {
                    table,
                    indices,
                    axis: 0,
                },
            );
        }
        "_softmax" | "softmax" => {
            let x = arg_ref(n, 0)?;
            let rank = out_shape(n)?.len();
            let axis = norm_axis(n.args.get(1).and_then(|a| a.as_int()).unwrap_or(-1), rank) as i32;
            push(lo, &id, Call::Softmax { x, axis });
        }
        "scaled_dot_product_attention" => lower_sdpa(n, lo, value_shape, value_dtype)?,
        "_scaled_dot_product_flash_attention"
        | "_scaled_dot_product_flash_attention_for_cpu"
        | "_scaled_dot_product_efficient_attention"
        | "_scaled_dot_product_attention_math" => {
            lower_sdpa_variant(n, lo, value_shape, value_dtype, &base)?
        }
        "convolution" => lower_convolution(n, lo)?,
        "baddbmm" => lower_baddbmm(n, lo)?,
        "upsample_nearest2d" | "_upsample_nearest_exact2d" => {
            lower_upsample_nearest2d(n, lo, value_shape)?
        }
        "upsample_bilinear2d" => lower_upsample_2d(n, lo, false, false)?,
        "upsample_bicubic2d" => lower_upsample_2d(n, lo, true, false)?,
        "_upsample_bilinear2d_aa" => lower_upsample_2d(n, lo, false, true)?,
        "_upsample_bicubic2d_aa" => lower_upsample_2d(n, lo, true, true)?,
        "pixel_shuffle" => lower_pixel_shuffle(n, lo, value_shape, true)?,
        "pixel_unshuffle" => lower_pixel_shuffle(n, lo, value_shape, false)?,
        "grid_sampler" | "grid_sampler_2d" => lower_grid_sampler(n, lo)?,
        "constant_pad_nd" => lower_constant_pad_nd(n, lo, value_shape)?,
        "index" => {
            // index.Tensor(x, [None*, idx, None*]) — single integer-tensor index
            // on one axis → gather.
            let x = arg_ref(n, 0)?;
            let list = match n.args.get(1) {
                Some(Arg::List { list }) => list,
                _ => bail!("index.Tensor needs a list of index tensors"),
            };
            let non_none: Vec<(usize, &Arg)> = list
                .iter()
                .enumerate()
                .filter(|(_, a)| !a.is_none())
                .collect();
            if non_none.len() != 1 {
                bail!(
                    "only single-axis integer indexing is supported (got {} index tensors)",
                    non_none.len()
                );
            }
            let (axis, idx_arg) = non_none[0];
            let idx = idx_arg
                .as_ref_name()
                .ok_or_else(|| anyhow!("index tensor must be a node ref"))?
                .to_string();
            push(
                lo,
                &id,
                Call::Gather {
                    table: x,
                    indices: idx,
                    axis,
                },
            );
        }
        "max_pool2d" | "max_pool2d_with_indices" => lower_pool(n, lo, ReduceOp::Max)?,
        "avg_pool2d" => lower_pool(n, lo, ReduceOp::Mean)?,
        "adaptive_avg_pool2d" | "_adaptive_avg_pool2d" => {
            // Only global average pooling (output [*, *, 1, 1]) is supported;
            // that is exactly a mean over the spatial axes with keepdim.
            let x = arg_ref(n, 0)?;
            let out = out_shape(n)?;
            let rank = out.len();
            if rank != 4 || out[2] != 1 || out[3] != 1 {
                bail!("adaptive_avg_pool2d only supports global pooling to [N,C,1,1]");
            }
            push(
                lo,
                &id,
                Call::Reduce {
                    op: ReduceOp::Mean,
                    x,
                    axes: vec![2, 3],
                    keep_dim: true,
                },
            );
        }
        "_native_batch_norm_legit_no_training"
        | "batch_norm"
        | "native_batch_norm"
        | "_native_batch_norm_legit" => lower_batch_norm(n, lo)?,
        "native_group_norm" => lower_group_norm(n, lo, true)?,
        "group_norm" => lower_group_norm(n, lo, false)?,
        "topk" => {
            // topk(x, k, dim=-1, largest, sorted) -> (values, indices).
            // rlx Op::TopK is argtopk (indices only), so: get indices, then
            // reconstruct values by gathering x at flat (row*E + idx) offsets.
            let x = arg_ref(n, 0)?;
            let k = n.args.get(1).and_then(|a| a.as_int()).unwrap_or(1) as usize;
            let outs = all_outs(&n.out);
            let rank = outs
                .first()
                .and_then(|o| o.as_ref())
                .map(|(s, _)| s.len())
                .unwrap_or(1);
            let dim = norm_axis(n.args.get(2).and_then(|a| a.as_int()).unwrap_or(-1), rank);
            if dim != rank.saturating_sub(1) {
                bail!("topk is only supported on the last axis (got dim {dim} of {rank})");
            }
            if n.args.get(3).and_then(|a| a.as_bool()) == Some(false) {
                bail!("topk largest=false is not supported");
            }
            let (vs, vd) = outs
                .first()
                .and_then(|o| o.clone())
                .ok_or_else(|| anyhow!("topk has no values output"))?;
            let (is, idt) = outs
                .get(1)
                .and_then(|o| o.clone())
                .ok_or_else(|| anyhow!("topk has no indices output"))?;
            let xshape = value_shape
                .get(&x)
                .ok_or_else(|| anyhow!("topk input {x} has no known shape"))?;
            let xr = xshape.len();
            let e = xshape[xr - 1] as usize;
            let outer: usize = xshape[..xr - 1]
                .iter()
                .map(|&d| d.max(1) as usize)
                .product();

            // indices (correct, i64)
            let inm = format!("{id}__idx");
            push(
                lo,
                &inm,
                Call::Node(crate::nodeop::NodeOp::TopK {
                    x: x.clone(),
                    k,
                    out: dims_usize(&is),
                    out_dtype: dtype_from_str(&idt)?,
                }),
            );
            // values via gather: flat_x[row*E + idx]
            let off = format!("{id}__off");
            push(
                lo,
                &off,
                Call::Iota {
                    rows: outer,
                    step: e as i64,
                    dtype: rlx_ir::DType::I64,
                },
            );
            let idx2 = format!("{id}__idx2");
            push(
                lo,
                &idx2,
                Call::Reshape {
                    x: inm.clone(),
                    shape: vec![outer as i64, k as i64],
                },
            );
            let glob = format!("{id}__glob");
            push(
                lo,
                &glob,
                Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                    op: BinaryOp::Add,
                    a: idx2,
                    b: off,
                    out: vec![outer, k],
                    out_dtype: rlx_ir::DType::I64,
                }),
            );
            let flatx = format!("{id}__flatx");
            push(
                lo,
                &flatx,
                Call::Reshape {
                    x,
                    shape: vec![(outer * e) as i64],
                },
            );
            let vf = format!("{id}__vf");
            push(
                lo,
                &vf,
                Call::Gather {
                    table: flatx,
                    indices: glob,
                    axis: 0,
                },
            );
            let vnm = format!("{id}__vals");
            push(
                lo,
                &vnm,
                Call::Reshape {
                    x: vf,
                    shape: vs.clone(),
                },
            );
            let _ = vd;

            multi.insert(id.clone(), vec![vnm, inm]);
        }

        // ── reduce ──────────────────────────────────────────────────────────
        "mean" | "sum" => {
            let x = arg_ref(n, 0)?;
            // Normalize (possibly negative) axes against the *input* rank.
            let in_rank = value_shape
                .get(&x)
                .map(|s| s.len())
                .or_else(|| primary_out(&n.out).map(|(s, _)| s.len()))
                .unwrap_or(0);
            let axes: Vec<usize> = match n.args.get(1).and_then(|a| a.as_int_list()) {
                Some(list) if !list.is_empty() => {
                    list.into_iter().map(|a| norm_axis(a, in_rank)).collect()
                }
                // An empty (or absent) dim list reduces over ALL dims — the aten
                // `sum.dim_IntList` / `mean.dim` convention, and how bare
                // `Tensor.sum()` / `.mean()` decompose. Reducing *nothing* here
                // (returning the input) breaks the downstream shapes.
                _ => (0..in_rank).collect(),
            };
            let keep = n.args.get(2).and_then(|a| a.as_bool()).unwrap_or(false);
            let op = if base == "mean" {
                ReduceOp::Mean
            } else {
                ReduceOp::Sum
            };
            push(
                lo,
                &id,
                Call::Reduce {
                    op,
                    x,
                    axes,
                    keep_dim: keep,
                },
            );
        }

        "le" | "ge" | "lt" | "gt" | "eq" | "ne" => {
            let cmp = match base.as_str() {
                "le" => rlx_ir::op::CmpOp::Le,
                "ge" => rlx_ir::op::CmpOp::Ge,
                "lt" => rlx_ir::op::CmpOp::Lt,
                "gt" => rlx_ir::op::CmpOp::Gt,
                "eq" => rlx_ir::op::CmpOp::Eq,
                _ => rlx_ir::op::CmpOp::Ne,
            };
            let (out, _dt) = primary_out(&n.out).ok_or_else(|| anyhow!("compare has no out"))?;
            let out_dims = dims_usize(&out);
            let a0 = arg_ref(n, 0)?;
            let a = bcast_ref(lo, value_shape, value_dtype, &a0, &out_dims, "a");
            let b = match n.args.get(1) {
                Some(Arg::Ref { reference }) => {
                    bcast_ref(lo, value_shape, value_dtype, reference, &out_dims, "b")
                }
                Some(other) => {
                    let v = other
                        .as_float()
                        .ok_or_else(|| anyhow!("compare: non-scalar rhs"))?;
                    scalar_const(lo, &id, v, rlx_ir::DType::F32)
                }
                None => bail!("compare: missing rhs"),
            };
            // Compare yields bool, which the f32 host arena under-sizes (a byte
            // per element → fewer f32 slots) — reading it as a plain f32 tensor
            // (e.g. into a matmul) goes out of bounds. Follow it with a Cast to
            // F32 {0,1}: the Cast node is properly f32-sized and is what the graph
            // consumes (same pattern as grid_sample / masked_fill).
            let cmp_tmp = format!("{id}__cmp");
            push(
                lo,
                &cmp_tmp,
                Call::Node(crate::nodeop::NodeOp::Compare {
                    op: cmp,
                    a,
                    b,
                    out: out_dims,
                    out_dtype: rlx_ir::DType::Bool,
                }),
            );
            push(
                lo,
                &id,
                Call::Cast {
                    x: cmp_tmp,
                    to: rlx_ir::DType::F32,
                },
            );
        }
        "where" => {
            // where(cond, a, b) — a/b may be scalars.
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("where has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let cond = arg_ref(n, 0)?;
            let operand = |lo: &mut Lowered, slot: usize, tag: &str| -> Result<Value> {
                match n.args.get(slot) {
                    Some(Arg::Ref { reference }) => Ok(reference.clone()),
                    Some(other) => {
                        let v = other
                            .as_float()
                            .ok_or_else(|| anyhow!("where: bad operand"))?;
                        Ok(scalar_const(lo, &format!("{id}_{tag}"), v, dt))
                    }
                    None => bail!("where: missing operand"),
                }
            };
            let a = operand(lo, 1, "a")?;
            let b = operand(lo, 2, "b")?;
            push(
                lo,
                &id,
                Call::Node(crate::nodeop::NodeOp::Where {
                    cond,
                    a,
                    b,
                    out: dims_usize(&out),
                    out_dtype: dt,
                }),
            );
        }
        "arange" => {
            // arange.start_step(start, end, step) or arange(end).
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("arange has no out"))?;
            let dt = dtype_from_str(&dt)?;
            // Integer/bool arange is materialized as F32: the host arena is f32,
            // small ramp values are exact, and gather accepts f32 indices. Keeping
            // a byte-sized int/bool intermediate would be mis-read as f32.
            let dt = if is_float_dtype(dt) {
                dt
            } else {
                rlx_ir::DType::F32
            };
            let len = out.first().copied().unwrap_or(0).max(0) as usize;
            let (start, step) = if over.starts_with("start") {
                (
                    n.args.first().and_then(|a| a.as_float()).unwrap_or(0.0),
                    n.args.get(2).and_then(|a| a.as_float()).unwrap_or(1.0),
                )
            } else {
                (0.0, 1.0)
            };
            push(
                lo,
                &id,
                Call::Arange {
                    start,
                    step,
                    len,
                    dtype: dt,
                },
            );
        }
        "gather" => {
            // aten.gather(x, dim, index) — per-row gather. Supported on the
            // last axis via the flatten + row-offset trick.
            let x = arg_ref(n, 0)?;
            let index = arg_ref(n, 2)?;
            let (out, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("gather has no out"))?;
            let dt = dtype_from_str(&dt)?;
            let xshape = value_shape
                .get(&x)
                .ok_or_else(|| anyhow!("gather input {x} has no known shape"))?;
            let xr = xshape.len();
            let dim = norm_axis(n.args.get(1).and_then(|a| a.as_int()).unwrap_or(-1), xr);
            if dim != xr - 1 {
                bail!("aten.gather only supported on the last axis (got {dim} of {xr})");
            }
            let d = xshape[xr - 1] as usize;
            let outer: usize = xshape[..xr - 1]
                .iter()
                .map(|&v| v.max(1) as usize)
                .product();
            let cols = *out.last().unwrap_or(&1) as usize;
            let off = format!("{id}__off");
            push(
                lo,
                &off,
                Call::Iota {
                    rows: outer,
                    step: d as i64,
                    dtype: rlx_ir::DType::I64,
                },
            );
            let idx2 = format!("{id}__idx2");
            push(
                lo,
                &idx2,
                Call::Reshape {
                    x: index,
                    shape: vec![outer as i64, cols as i64],
                },
            );
            let glob = format!("{id}__glob");
            push(
                lo,
                &glob,
                Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                    op: BinaryOp::Add,
                    a: idx2,
                    b: off,
                    out: vec![outer, cols],
                    out_dtype: rlx_ir::DType::I64,
                }),
            );
            let flatx = format!("{id}__flatx");
            push(
                lo,
                &flatx,
                Call::Reshape {
                    x,
                    shape: vec![(outer * d) as i64],
                },
            );
            let vf = format!("{id}__vf");
            push(
                lo,
                &vf,
                Call::Gather {
                    table: flatx,
                    indices: glob,
                    axis: 0,
                },
            );
            push(
                lo,
                &id,
                Call::Reshape {
                    x: vf,
                    shape: out.clone(),
                },
            );
            let _ = dt;
        }

        other => bail!("internal: no handler for supported op {other}"),
    }
    Ok(())
}

fn lower_sdpa(
    n: &NodeDef,
    lo: &mut Lowered,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
    value_dtype: &std::collections::HashMap<String, rlx_ir::DType>,
) -> Result<()> {
    // Public scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
    //   is_causal=False, scale=None).
    build_sdpa(
        n,
        lo,
        value_shape,
        value_dtype,
        sdpa_is_causal(n, 5),
        sdpa_mask(n, 3)?,
    )
}

/// Backend-specific SDPA overloads (`decomposition=core`) — same q/k/v at args
/// 0-2, but the mask/bias and `is_causal` move, and each returns a tuple whose
/// element 0 is the attention output (reached via getitem[0], so bind it under
/// the node id like `native_layer_norm`).
fn lower_sdpa_variant(
    n: &NodeDef,
    lo: &mut Lowered,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
    value_dtype: &std::collections::HashMap<String, rlx_ir::DType>,
    base: &str,
) -> Result<()> {
    let (is_causal, mask) = match base {
        // (q,k,v, dropout_p, is_causal, return_debug_mask, *, scale) — no mask.
        "_scaled_dot_product_flash_attention" | "_scaled_dot_product_flash_attention_for_cpu" => {
            (sdpa_is_causal(n, 4), None)
        }
        // (q,k,v, attn_bias, compute_log_sumexp, dropout_p, is_causal, *, scale).
        "_scaled_dot_product_efficient_attention" => (sdpa_is_causal(n, 6), sdpa_mask(n, 3)?),
        // math / fallback: same arg layout as the public op.
        _ => (sdpa_is_causal(n, 5), sdpa_mask(n, 3)?),
    };
    build_sdpa(n, lo, value_shape, value_dtype, is_causal, mask)
}

fn sdpa_is_causal(n: &NodeDef, pos: usize) -> bool {
    n.args
        .get(pos)
        .and_then(|a| a.as_bool())
        .or_else(|| n.kwargs.get("is_causal").and_then(|a| a.as_bool()))
        .unwrap_or(false)
}

fn sdpa_mask(n: &NodeDef, pos: usize) -> Result<Option<Value>> {
    if n.args.get(pos).map(|a| !a.is_none()).unwrap_or(false) {
        Ok(Some(arg_ref(n, pos)?))
    } else {
        Ok(None)
    }
}

fn build_sdpa(
    n: &NodeDef,
    lo: &mut Lowered,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
    value_dtype: &std::collections::HashMap<String, rlx_ir::DType>,
    is_causal: bool,
    mask: Option<Value>,
) -> Result<()> {
    // q/k/v are [B, H, S, D].
    let q = arg_ref(n, 0)?;
    let k = arg_ref(n, 1)?;
    let v = arg_ref(n, 2)?;
    let (qshape, qdt) = primary_out(&n.out).ok_or_else(|| anyhow!("sdpa has no out shape"))?;
    if qshape.len() != 4 {
        bail!("sdpa expects rank-4 [B,H,S,D] q/k/v, got {:?}", qshape);
    }
    let num_heads = qshape[1] as usize;
    let head_dim = qshape[3] as usize;
    let out = dims_usize(&qshape);
    let out_dtype = dtype_from_str(&qdt)?;
    if !is_causal {
        if let Some(mask) = mask {
            let mdt = value_dtype
                .get(&mask)
                .copied()
                .unwrap_or(rlx_ir::DType::F32);
            let mshape = value_shape
                .get(&mask)
                .cloned()
                .unwrap_or_else(|| vec![qshape[0], 1, qshape[2], qshape[2]]);
            // torch accepts a boolean mask (True = keep). Convert it to the additive
            // float bias rlx's Attention(MaskKind::Bias) expects: 0 where True,
            // large-negative where False. Pure arithmetic — `(mask_f32 - 1) * 1e30`
            // — avoids the bool-arena / where / -inf(NaN) corner cases.
            let bias0 = if mdt == rlx_ir::DType::Bool {
                let md = dims_usize(&mshape);
                let mf = format!("{}__maskf", n.id);
                push(
                    lo,
                    &mf,
                    Call::Cast {
                        x: mask,
                        to: rlx_ir::DType::F32,
                    },
                );
                let one = scalar_const(lo, &format!("{}_one", n.id), 1.0, rlx_ir::DType::F32);
                let sub = format!("{}__msub", n.id);
                push(
                    lo,
                    &sub,
                    Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                        op: BinaryOp::Sub,
                        a: mf,
                        b: one,
                        out: md.clone(),
                        out_dtype: rlx_ir::DType::F32,
                    }),
                );
                let big = scalar_const(lo, &format!("{}_big", n.id), 1e30, rlx_ir::DType::F32);
                let add = format!("{}__addmask", n.id);
                push(
                    lo,
                    &add,
                    Call::Node(crate::nodeop::NodeOp::BinaryShaped {
                        op: BinaryOp::Mul,
                        a: sub,
                        b: big,
                        out: md,
                        out_dtype: rlx_ir::DType::F32,
                    }),
                );
                add
            } else {
                mask
            };
            // Attention(MaskKind::Bias) wants the bias at [B, num_heads, Sq, Sk];
            // broadcast a [B, 1, Sq, Sk] (or [1, 1, Sq, Sk]) mask up to all heads.
            let sk = *mshape.last().unwrap_or(&qshape[2]);
            let target = vec![qshape[0], num_heads as i64, qshape[2], sk];
            let bias = if mshape == target {
                bias0
            } else {
                let bx = format!("{}__biasx", n.id);
                push(
                    lo,
                    &bx,
                    Call::Node(crate::nodeop::NodeOp::Expand {
                        x: bias0,
                        target: target.clone(),
                        out: dims_usize(&target),
                        out_dtype: rlx_ir::DType::F32,
                    }),
                );
                bx
            };
            push(
                lo,
                &n.id,
                Call::AttentionBias {
                    q,
                    k,
                    v,
                    bias,
                    num_heads,
                    head_dim,
                    out,
                    out_dtype,
                },
            );
            return Ok(());
        }
    }
    let mask_kind = if is_causal {
        MaskKind::Causal
    } else {
        MaskKind::None
    };
    push(
        lo,
        &n.id,
        Call::Attention {
            q,
            k,
            v,
            num_heads,
            head_dim,
            mask: mask_kind,
            out,
            out_dtype,
        },
    );
    Ok(())
}

fn pair(list: &[i64], default: i64) -> [usize; 2] {
    [
        *list.first().unwrap_or(&default) as usize,
        *list.get(1).unwrap_or(list.first().unwrap_or(&default)) as usize,
    ]
}

fn lower_convolution(n: &NodeDef, lo: &mut Lowered) -> Result<()> {
    // convolution(x, weight, bias, stride, padding, dilation, transposed,
    //   output_padding, groups)
    let x = arg_ref(n, 0)?;
    let weight = arg_ref(n, 1)?;
    let bias = n
        .args
        .get(2)
        .and_then(|a| a.as_ref_name())
        .map(String::from);
    let stride = pair(&n.args[3].as_int_list().unwrap_or(vec![1, 1]), 1);
    let padding = pair(&n.args[4].as_int_list().unwrap_or(vec![0, 0]), 0);
    let dilation = pair(
        &n.args
            .get(5)
            .and_then(|a| a.as_int_list())
            .unwrap_or(vec![1, 1]),
        1,
    );
    let transposed = n.args.get(6).and_then(|a| a.as_bool()).unwrap_or(false);
    let output_padding = pair(
        &n.args
            .get(7)
            .and_then(|a| a.as_int_list())
            .unwrap_or(vec![0, 0]),
        0,
    );
    let groups = n.args.get(8).and_then(|a| a.as_int()).unwrap_or(1) as usize;

    let (oshape, odt) = primary_out(&n.out).ok_or_else(|| anyhow!("conv has no out shape"))?;
    if oshape.len() != 4 {
        bail!("only 2-D convolution (rank-4) supported, got {:?}", oshape);
    }
    let wshape = lo
        .params
        .iter()
        .find(|p| p.value_id == weight)
        .map(|p| p.shape.clone())
        .ok_or_else(|| anyhow!("conv weight {weight} is not a param"))?;
    if wshape.len() != 4 {
        bail!("conv weight must be rank-4, got {:?}", wshape);
    }
    let kernel = [wshape[2], wshape[3]];
    let out = dims_usize(&oshape);
    let out_dtype = dtype_from_str(&odt)?;
    let conv_res = if bias.is_some() {
        format!("{}__conv", n.id)
    } else {
        n.id.clone()
    };
    let call = if transposed {
        Call::ConvTranspose2d {
            x,
            weight,
            kernel,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            out: out.clone(),
            out_dtype,
        }
    } else {
        Call::Conv2d {
            x,
            weight,
            kernel,
            stride,
            padding,
            groups,
            out: out.clone(),
            out_dtype,
        }
    };
    push(lo, &conv_res, call);
    if let Some(b) = bias {
        // reshape bias [C] -> [1, C, 1, 1] then broadcast-add
        let c = out[1];
        let br = format!("{}__bias", n.id);
        push(
            lo,
            &br,
            Call::Reshape {
                x: b,
                shape: vec![1, c as i64, 1, 1],
            },
        );
        push(lo, &n.id, Call::Binary(BinaryOp::Add, conv_res, br));
    }
    Ok(())
}

/// GroupNorm on NCHW → `Op::GroupNorm` (input, gamma[C], beta[C]).
///
/// `native` selects the ATen arg layout:
/// - `native_group_norm(input, weight, bias, N, C, HxW, group, eps)` → 3 outputs
///   (out, mean, rstd); only `out` (getitem[0]) is consumed.
/// - `group_norm(input, num_groups, weight, bias, eps, cudnn_enabled)` → 1 output.
///
/// SD/SDXL/VAE always use affine (weight+bias present) rank-4 GroupNorm, which is
/// exactly what `Op::GroupNorm` implements (per-group over `(C/G)×H×W`).
fn lower_group_norm(n: &NodeDef, lo: &mut Lowered, native: bool) -> Result<()> {
    let x = arg_ref(n, 0)?;
    let (gamma, beta, num_groups, eps) = if native {
        let gamma = arg_ref(n, 1)?;
        let beta = arg_ref(n, 2)?;
        let num_groups = n.args.get(6).and_then(|a| a.as_int()).unwrap_or(1) as usize;
        let eps = n.args.get(7).and_then(|a| a.as_float()).unwrap_or(1e-5) as f32;
        (gamma, beta, num_groups, eps)
    } else {
        let num_groups = n.args.get(1).and_then(|a| a.as_int()).unwrap_or(1) as usize;
        let gamma = arg_ref(n, 2)?;
        let beta = arg_ref(n, 3)?;
        let eps = n.args.get(4).and_then(|a| a.as_float()).unwrap_or(1e-5) as f32;
        (gamma, beta, num_groups, eps)
    };
    let (oshape, odt) =
        primary_out(&n.out).ok_or_else(|| anyhow!("group_norm has no out shape"))?;
    if oshape.len() != 4 {
        bail!("group_norm expects rank-4 NCHW, got {:?}", oshape);
    }
    push(
        lo,
        &n.id,
        Call::Node(crate::nodeop::NodeOp::GroupNorm {
            x,
            gamma,
            beta,
            num_groups,
            eps,
            out: dims_usize(&oshape),
            out_dtype: dtype_from_str(&odt)?,
        }),
    );
    Ok(())
}

/// `upsample_nearest2d` (`.default` and `.vec`) → chained `Op::ResizeNearest2x`.
///
/// RLX's `ResizeNearest2x` doubles the spatial dims; a power-of-two scale is a
/// chain of those. The scale is recovered from the concrete in/out shapes (both
/// NCHW), so it covers SD/SDXL/VAE decoders (always nearest ×2). Non-uniform or
/// non-power-of-two scales bail with a clear message.
fn lower_upsample_nearest2d(
    n: &NodeDef,
    lo: &mut Lowered,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<()> {
    let x = arg_ref(n, 0)?;
    let (oshape, odt) =
        primary_out(&n.out).ok_or_else(|| anyhow!("upsample_nearest2d has no out shape"))?;
    let out_dtype = dtype_from_str(&odt)?;
    if oshape.len() != 4 {
        bail!(
            "upsample_nearest2d expects rank-4 NCHW output, got {:?}",
            oshape
        );
    }
    let in_shape = value_shape
        .get(&x)
        .cloned()
        .ok_or_else(|| anyhow!("upsample_nearest2d: unknown input shape for {x}"))?;
    if in_shape.len() != 4 {
        bail!(
            "upsample_nearest2d input must be rank-4 NCHW, got {:?}",
            in_shape
        );
    }
    let (n_, c_, in_h, in_w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (out_h, out_w) = (oshape[2], oshape[3]);
    if in_h <= 0 || in_w <= 0 || out_h % in_h != 0 || out_w % in_w != 0 {
        bail!("upsample_nearest2d: non-integer scale {in_shape:?} -> {oshape:?}");
    }
    let sh = (out_h / in_h) as usize;
    let sw = (out_w / in_w) as usize;
    if sh != sw {
        bail!("upsample_nearest2d: non-uniform scale {sh}x{sw} not supported");
    }
    if sh == 1 {
        push(lo, &n.id, Call::Alias(x));
        return Ok(());
    }
    if !sh.is_power_of_two() {
        bail!(
            "upsample_nearest2d: only power-of-two nearest scale supported (got {sh}); \
             RLX has Op::ResizeNearest2x (2×)"
        );
    }
    let steps = sh.trailing_zeros() as usize;
    let mut cur = x;
    let (mut cur_h, mut cur_w) = (in_h, in_w);
    for s in 0..steps {
        cur_h *= 2;
        cur_w *= 2;
        let res_name = if s == steps - 1 {
            n.id.clone()
        } else {
            format!("{}__up{}", n.id, s)
        };
        push(
            lo,
            &res_name,
            Call::Node(crate::nodeop::NodeOp::ResizeNearest2x {
                x: cur.clone(),
                out: vec![n_ as usize, c_ as usize, cur_h as usize, cur_w as usize],
                out_dtype,
            }),
        );
        cur = res_name;
    }
    Ok(())
}

/// `upsample_{bilinear,bicubic}2d` (`.default` + `.vec`) → `Call::Resize`
/// (separable interpolation matmuls in the HIR builder). Output size comes from
/// the node's concrete meta shape; `align_corners` is arg 2 in both overloads.
fn lower_upsample_2d(n: &NodeDef, lo: &mut Lowered, cubic: bool, antialias: bool) -> Result<()> {
    let x = arg_ref(n, 0)?;
    let (oshape, odt) =
        primary_out(&n.out).ok_or_else(|| anyhow!("upsample_2d has no out shape"))?;
    if oshape.len() != 4 {
        bail!(
            "upsample_{{bilinear,bicubic}}2d expects rank-4 NCHW, got {:?}",
            oshape
        );
    }
    let align_corners = n.args.get(2).and_then(|a| a.as_bool()).unwrap_or(false);
    push(
        lo,
        &n.id,
        Call::Resize {
            x,
            out_h: oshape[2] as usize,
            out_w: oshape[3] as usize,
            align_corners,
            cubic,
            antialias,
            out: dims_usize(&oshape),
            out_dtype: dtype_from_str(&odt)?,
        },
    );
    Ok(())
}

/// `grid_sampler[_2d](input, grid, interp_mode, padding_mode, align_corners)` →
/// `Call::GridSample` (decomposed in the HIR builder). Enum codes follow ATen:
/// interp 0=bilinear/1=nearest/2=bicubic; pad 0=zeros/1=border/2=reflection.
fn lower_grid_sampler(n: &NodeDef, lo: &mut Lowered) -> Result<()> {
    let input = arg_ref(n, 0)?;
    let grid = arg_ref(n, 1)?;
    let mode = match n.args.get(2).and_then(|a| a.as_int()).unwrap_or(0) {
        0 => GridMode::Bilinear,
        1 => GridMode::Nearest,
        2 => GridMode::Bicubic,
        m => bail!("grid_sampler: unknown interpolation_mode {m}"),
    };
    let pad = match n.args.get(3).and_then(|a| a.as_int()).unwrap_or(0) {
        0 => GridPad::Zeros,
        1 => GridPad::Border,
        2 => GridPad::Reflection,
        p => bail!("grid_sampler: unknown padding_mode {p}"),
    };
    let align_corners = n.args.get(4).and_then(|a| a.as_bool()).unwrap_or(false);
    let (oshape, odt) =
        primary_out(&n.out).ok_or_else(|| anyhow!("grid_sampler has no out shape"))?;
    if oshape.len() != 4 {
        bail!(
            "only grid_sampler_2d (rank-4) supported, got output {:?}",
            oshape
        );
    }
    push(
        lo,
        &n.id,
        Call::GridSample {
            input,
            grid,
            mode,
            pad,
            align_corners,
            out: dims_usize(&oshape),
            out_dtype: dtype_from_str(&odt)?,
        },
    );
    Ok(())
}

/// `pixel_shuffle(x, r)` / `pixel_unshuffle(x, r)` → reshape + permute + reshape
/// (no dedicated op needed — it is pure data movement).
///
/// shuffle:   `[N, C·r², H, W]` → `[N, C, H·r, W·r]`
/// unshuffle: `[N, C, H·r, W·r]` → `[N, C·r², H, W]`
fn lower_pixel_shuffle(
    n: &NodeDef,
    lo: &mut Lowered,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
    shuffle: bool,
) -> Result<()> {
    let x = arg_ref(n, 0)?;
    let r = n.args.get(1).and_then(|a| a.as_int()).unwrap_or(1);
    if r < 1 {
        bail!("pixel_(un)shuffle: factor must be ≥ 1, got {r}");
    }
    let in_shape = value_shape
        .get(&x)
        .cloned()
        .ok_or_else(|| anyhow!("pixel_(un)shuffle: unknown input shape for {x}"))?;
    if in_shape.len() != 4 {
        bail!("pixel_(un)shuffle expects rank-4 NCHW, got {:?}", in_shape);
    }
    if r == 1 {
        push(lo, &n.id, Call::Alias(x));
        return Ok(());
    }
    let (bn, cin, hin, win) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    // Split-shape (6-D), permutation, and final (4-D) shape differ per direction.
    let (split6, perm, out4): (Vec<i64>, Vec<usize>, Vec<i64>) = if shuffle {
        let c = cin / (r * r);
        (
            vec![bn, c, r, r, hin, win],
            vec![0, 1, 4, 2, 5, 3], // → [N, C, H, r, W, r]
            vec![bn, c, hin * r, win * r],
        )
    } else {
        let (h, w) = (hin / r, win / r);
        (
            vec![bn, cin, h, r, w, r],
            vec![0, 1, 3, 5, 2, 4], // → [N, C, r, r, H, W]
            vec![bn, cin * r * r, h, w],
        )
    };
    let s1 = format!("{}__ps_r", n.id);
    push(lo, &s1, Call::Reshape { x, shape: split6 });
    let s2 = format!("{}__ps_p", n.id);
    push(lo, &s2, Call::Transpose { x: s1, perm });
    push(lo, &n.id, Call::Reshape { x: s2, shape: out4 });
    Ok(())
}

/// `constant_pad_nd(input, pad, value)` → `concat` with constant-filled blocks.
///
/// `pad` is a flat `[lo, hi, lo, hi, …]` list applied to the **last** dims (last
/// pair first). RLX has no dedicated pad op, but constant-mode padding on axis
/// `a` is exactly `concat([fill_lo, x, fill_hi], a)` with the fills = `Op::Constant`
/// of `value`. This covers e.g. Sana's linear-attention ones-padding
/// (`F.pad(v, (0,1), value=1.0)`). Only non-negative (padding, not cropping)
/// constant pads are supported.
fn lower_constant_pad_nd(
    n: &NodeDef,
    lo: &mut Lowered,
    value_shape: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<()> {
    let x = arg_ref(n, 0)?;
    let pad = n
        .args
        .get(1)
        .and_then(|a| a.as_int_list())
        .ok_or_else(|| anyhow!("constant_pad_nd needs an int-list `pad`"))?;
    let value = n.args.get(2).and_then(|a| a.as_float()).unwrap_or(0.0);
    let (_, odt) =
        primary_out(&n.out).ok_or_else(|| anyhow!("constant_pad_nd has no out shape"))?;
    let dt = dtype_from_str(&odt)?;
    let mut cur_shape = value_shape
        .get(&x)
        .cloned()
        .ok_or_else(|| anyhow!("constant_pad_nd: unknown input shape for {x}"))?;
    let rank = cur_shape.len();
    if pad.len() % 2 != 0 || pad.len() / 2 > rank {
        bail!("constant_pad_nd: bad pad {pad:?} for rank {rank}");
    }
    if pad.iter().any(|&p| p < 0) {
        bail!("constant_pad_nd: negative (cropping) pad not supported: {pad:?}");
    }
    let mut cur = x;
    for pair in 0..pad.len() / 2 {
        let (lo_amt, hi_amt) = (pad[2 * pair], pad[2 * pair + 1]);
        if lo_amt == 0 && hi_amt == 0 {
            continue;
        }
        let axis = rank - 1 - pair;
        let mut pieces: Vec<Value> = Vec::new();
        let block = |lo: &mut Lowered, amt: i64, tag: &str| -> Value {
            let mut bshape = cur_shape.clone();
            bshape[axis] = amt;
            let nm = format!("{}__pad{}{}", n.id, pair, tag);
            push(
                lo,
                &nm,
                Call::Full {
                    value,
                    shape: dims_usize(&bshape),
                    dtype: dt,
                },
            );
            nm
        };
        if lo_amt > 0 {
            let b = block(lo, lo_amt, "lo");
            pieces.push(b);
        }
        pieces.push(cur.clone());
        if hi_amt > 0 {
            let b = block(lo, hi_amt, "hi");
            pieces.push(b);
        }
        cur_shape[axis] += lo_amt + hi_amt;
        let nm = format!("{}__cat{}", n.id, pair);
        push(lo, &nm, Call::Concat { xs: pieces, axis });
        cur = nm;
    }
    // Bind the final (or unpadded) tensor to the node id.
    push(lo, &n.id, Call::Alias(cur));
    Ok(())
}

/// `zeros`/`ones`/`empty` (and their `_like` / `new_` variants) → `Op::Constant`.
/// Uninitialized `empty` is materialized as zeros (safe: it is only ever read
/// after being overwritten or multiplied by `beta = 0`, e.g. baddbmm bias).
fn lower_const_fill(n: &NodeDef, lo: &mut Lowered, value: f64) -> Result<()> {
    let (shape, dt) = primary_out(&n.out).ok_or_else(|| anyhow!("{} has no out shape", n.op))?;
    push(
        lo,
        &n.id,
        Call::Full {
            value,
            shape: dims_usize(&shape),
            dtype: dtype_from_str(&dt)?,
        },
    );
    Ok(())
}

/// `baddbmm(input, batch1, batch2, *, beta=1, alpha=1)`
///   = `beta * input + alpha * (batch1 @ batch2)`.
///
/// The non-SDPA attention score path (diffusers `Attention.get_attention_scores`)
/// uses `beta = 0` — that branch drops `input` entirely, so the (uninitialized)
/// bias never affects the result.
fn lower_baddbmm(n: &NodeDef, lo: &mut Lowered) -> Result<()> {
    let input = arg_ref(n, 0)?;
    let b1 = arg_ref(n, 1)?;
    let b2 = arg_ref(n, 2)?;
    // beta/alpha are keyword-only in the schema but may also arrive positionally.
    let scalar = |k: &str, pos: usize| -> f64 {
        n.kwargs
            .get(k)
            .and_then(|a| a.as_float())
            .or_else(|| n.args.get(pos).and_then(|a| a.as_float()))
            .unwrap_or(1.0)
    };
    let beta = scalar("beta", 3);
    let alpha = scalar("alpha", 4);
    let (_, odt) = primary_out(&n.out).ok_or_else(|| anyhow!("baddbmm has no out shape"))?;
    let dt = dtype_from_str(&odt)?;
    let id = &n.id;

    let mm = format!("{id}__mm");
    push(lo, &mm, Call::Mm(b1, b2));
    // term = alpha * (b1 @ b2)
    let term = if (alpha - 1.0).abs() < 1e-12 {
        mm
    } else {
        let a = scalar_const(lo, id, alpha, dt);
        let t = format!("{id}__amm");
        push(lo, &t, Call::Binary(BinaryOp::Mul, mm, a));
        t
    };
    // out = term + beta * input   (beta == 0 ⇒ drop input; beta == 1 ⇒ plain add)
    if beta.abs() < 1e-12 {
        push(lo, id, Call::Alias(term));
    } else if (beta - 1.0).abs() < 1e-12 {
        push(lo, id, Call::Binary(BinaryOp::Add, term, input));
    } else {
        let bc = scalar_const(lo, id, beta, dt);
        let bi = format!("{id}__binp");
        push(lo, &bi, Call::Binary(BinaryOp::Mul, input, bc));
        push(lo, id, Call::Binary(BinaryOp::Add, term, bi));
    }
    Ok(())
}

/// Push `hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)` and return its result value.
/// Shared by `hardsigmoid` and `hardswish`.
fn push_hardsigmoid(
    lo: &mut Lowered,
    id: &str,
    x: Value,
    out: &[usize],
    dt: rlx_ir::DType,
) -> Value {
    use crate::nodeop::NodeOp;
    let sixth = scalar_const(lo, &format!("{id}_hs6"), 1.0 / 6.0, dt);
    let xm = format!("{id}__hs_m");
    push(lo, &xm, Call::Binary(BinaryOp::Mul, x, sixth));
    let half = scalar_const(lo, &format!("{id}_hshalf"), 0.5, dt);
    let xp = format!("{id}__hs_p");
    push(lo, &xp, Call::Binary(BinaryOp::Add, xm, half));
    let zero = scalar_const(lo, &format!("{id}_hs0"), 0.0, dt);
    let mx = format!("{id}__hs_max");
    push(
        lo,
        &mx,
        Call::Node(NodeOp::BinaryShaped {
            op: BinaryOp::Max,
            a: xp,
            b: zero,
            out: out.to_vec(),
            out_dtype: dt,
        }),
    );
    let one = scalar_const(lo, &format!("{id}_hs1"), 1.0, dt);
    let mn = format!("{id}__hs_min");
    push(
        lo,
        &mn,
        Call::Node(NodeOp::BinaryShaped {
            op: BinaryOp::Min,
            a: mx,
            b: one,
            out: out.to_vec(),
            out_dtype: dt,
        }),
    );
    mn
}

fn lower_pool(n: &NodeDef, lo: &mut Lowered, kind: ReduceOp) -> Result<()> {
    // {max,avg}_pool2d(x, kernel, stride=[], padding=0, ...)
    let x = arg_ref(n, 0)?;
    let (oshape, odt) = primary_out(&n.out).ok_or_else(|| anyhow!("pool has no out shape"))?;
    let kernel = n.args[1]
        .as_int_list()
        .ok_or_else(|| anyhow!("pool needs a kernel_size list"))?;
    let kernel = pair(&kernel, 1);
    let stride = match n.args.get(2).and_then(|a| a.as_int_list()) {
        Some(s) if !s.is_empty() => pair(&s, kernel[0] as i64),
        _ => kernel,
    };
    let padding = pair(
        &n.args
            .get(3)
            .and_then(|a| a.as_int_list())
            .unwrap_or(vec![0, 0]),
        0,
    );
    push(
        lo,
        &n.id,
        Call::Node(crate::nodeop::NodeOp::Pool {
            kind,
            x,
            kernel: kernel.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            out: dims_usize(&oshape),
            out_dtype: dtype_from_str(&odt)?,
        }),
    );
    Ok(())
}

fn lower_batch_norm(n: &NodeDef, lo: &mut Lowered) -> Result<()> {
    // (x, weight, bias, running_mean, running_var, [training], momentum, eps).
    // Decomposed into layout-agnostic elementwise ops (channel axis = 1),
    // because rlx `Op::BatchNormInference` assumes channels-LAST while torch is
    // NCHW: out = (x - mean) * rsqrt(var + eps) * gamma + beta, stats broadcast
    // as [1, C, 1, ...].
    let x = arg_ref(n, 0)?;
    let gamma = arg_ref(n, 1)?;
    let beta = arg_ref(n, 2)?;
    let mean = arg_ref(n, 3)?;
    let var = arg_ref(n, 4)?;
    let eps = n
        .args
        .iter()
        .rev()
        .find_map(|a| match a {
            Arg::Float { v } => Some(*v as f32),
            _ => None,
        })
        .unwrap_or(1e-5);
    let (oshape, odt) =
        primary_out(&n.out).ok_or_else(|| anyhow!("batch_norm has no out shape"))?;
    let dt = dtype_from_str(&odt)?;
    let rank = oshape.len();
    let channels = *oshape
        .get(1)
        .ok_or_else(|| anyhow!("batch_norm expects rank>=2"))?;
    let id = &n.id;

    // inv = rsqrt(var + eps)   (per channel, [C])
    let epsc = scalar_const(lo, id, eps as f64, dt);
    let vpe = format!("{id}__vpe");
    push(lo, &vpe, Call::Binary(BinaryOp::Add, var, epsc));
    let inv = format!("{id}__inv");
    push(lo, &inv, Call::Act(rlx_ir::op::Activation::Rsqrt, vpe));
    // scale = gamma * inv ; shift = beta - mean * scale   ([C])
    let scale = format!("{id}__scale");
    push(lo, &scale, Call::Binary(BinaryOp::Mul, gamma, inv));
    let ms = format!("{id}__ms");
    push(lo, &ms, Call::Binary(BinaryOp::Mul, mean, scale.clone()));
    let shift = format!("{id}__shift");
    push(lo, &shift, Call::Binary(BinaryOp::Sub, beta, ms));
    // reshape scale/shift to broadcast on the channel axis: [1, C, 1, ...]
    let bshape: Vec<i64> = (0..rank)
        .map(|i| if i == 1 { channels } else { 1 })
        .collect();
    let scr = format!("{id}__scr");
    push(
        lo,
        &scr,
        Call::Reshape {
            x: scale,
            shape: bshape.clone(),
        },
    );
    let shr = format!("{id}__shr");
    push(
        lo,
        &shr,
        Call::Reshape {
            x: shift,
            shape: bshape,
        },
    );
    // out = x * scale + shift
    let xs = format!("{id}__xs");
    push(lo, &xs, Call::Binary(BinaryOp::Mul, x, scr));
    push(lo, id, Call::Binary(BinaryOp::Add, xs, shr));
    Ok(())
}
