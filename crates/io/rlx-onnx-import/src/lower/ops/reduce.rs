// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `reduce` — extracted from the `ops` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, bail};
use rlx_ir::dynamic::sym;
use rlx_ir::hir::{HirMut, HirNodeId, HirOp};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, ReduceOp};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Dim, HirGraphExt, HirModule, Op, Shape};

use crate::bundle::RlxBundle;
use crate::bundle::{BundleManifest, BundleNode, topo_sort_nodes};
use crate::control_flow::{self, DURATION_CARRY};
use crate::rewrite::rewrite_graph;
use crate::tensor_data::i64_tensor;
use crate::tensor_data::{TypedParams, quant_matmul_weight_key};

use crate::lower::options::{ImportOptions, ImportReport};

use super::*;

pub(super) fn lower_reduce(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let keep = node
        .attrs
        .get("keepdims")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        != 0;
    let rank = m.shape(x).rank();
    let axes = reduce_axes(node, ctx, rank);
    // Composite reductions with no native op — decompose to a pre-map + sum
    // (+ post-map): ReduceL2 = sqrt(Σx²), ReduceSumSquare = Σx²,
    // ReduceL1 = Σ|x|, ReduceLogSum = log(Σx), ReduceLogSumExp = log(Σeˣ).
    // (F5-TTS GRN blocks use ReduceL2.)
    if matches!(
        op,
        "ReduceL2" | "ReduceL1" | "ReduceSumSquare" | "ReduceLogSum" | "ReduceLogSumExp"
    ) {
        let xs = m.shape(x).clone();
        let pre = match op {
            "ReduceL2" | "ReduceSumSquare" => m.add_node(Op::Binary(BinaryOp::Mul), vec![x, x], xs),
            "ReduceL1" => m.add_node(Op::Activation(Activation::Abs), vec![x], xs),
            "ReduceLogSumExp" => m.add_node(Op::Activation(Activation::Exp), vec![x], xs),
            _ => x, // ReduceLogSum sums the input directly
        };
        let summed = m.sum(pre, axes, keep);
        let ss = m.shape(summed).clone();
        let id = match op {
            "ReduceL2" => m.add_node(Op::Activation(Activation::Sqrt), vec![summed], ss),
            "ReduceLogSum" | "ReduceLogSumExp" => {
                m.add_node(Op::Activation(Activation::Log), vec![summed], ss)
            }
            _ => summed, // ReduceSumSquare, ReduceL1
        };
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let rop = match op {
        "ReduceSum" => ReduceOp::Sum,
        "ReduceMax" => ReduceOp::Max,
        "ReduceMin" => ReduceOp::Min,
        "ReduceProd" => ReduceOp::Prod,
        _ => ReduceOp::Mean,
    };
    // The CPU `Reduce` kernel is f32-only; an integer input (e.g. MOSS's
    // `ReduceSum` over an i64 {0,1} sampling mask to COUNT the selected position)
    // would be read as f32 garbage. Reduce in f32 and cast the result back to the
    // integer dtype (counts are small, well within f32's exact range).
    let in_dt = m.shape(x).dtype();
    let int_reduce = matches!(in_dt, DType::I64 | DType::I32);
    let xr = if int_reduce {
        let s = m.shape(x).clone().with_dtype(DType::F32);
        m.add_node(Op::Cast { to: DType::F32 }, vec![x], s)
    } else {
        x
    };
    let id = match rop {
        ReduceOp::Mean => m.mean(xr, axes, keep),
        ReduceOp::Sum => m.sum(xr, axes, keep),
        _ => {
            // The Reduce runs on `xr` — which is f32 when the input was integer
            // (see the int_reduce cast above) — so its *declared* dtype must
            // follow `xr`, not the ONNX node's output type (the pre-cast integer
            // dtype). The int_reduce cast-back below restores that dtype.
            // Declaring the Reduce i64 while feeding it f32 is invalid IR that
            // only surfaces once the node is inlined out of a subgraph (e.g. a
            // While cond) and reaches the shape verifier.
            let xr_dt = m.shape(xr).dtype();
            let out_s = output_shape(ctx, node, m, xr).with_dtype(xr_dt);
            m.add_node(
                Op::Reduce {
                    op: rop,
                    axes,
                    keep_dim: keep,
                },
                vec![xr],
                out_s,
            )
        }
    };
    let id = if int_reduce {
        let s = m.shape(id).clone().with_dtype(in_dt);
        m.add_node(Op::Cast { to: in_dt }, vec![id], s)
    } else {
        id
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `CumProd` — cumulative product along `axis` (input 1, baked into attrs
/// when constant). Mirrors `CumSum`'s `exclusive`/`reverse` attributes, packed
/// as `[axis_i32, exclusive_u8, reverse_u8]`.
pub(super) fn lower_cumprod(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank().max(1);
    let axis = node
        .inputs
        .get(1)
        .and_then(|n| i64_tensor(&ctx.i64_params, &ctx.params, n))
        .and_then(|v| v.first().copied())
        .map(|a| normalize_axis(a, rank))
        .unwrap_or(0) as i32;
    let exclusive = node
        .attrs
        .get("exclusive")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let reverse = node
        .attrs
        .get("reverse")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let out_s = output_shape(ctx, node, m, x);
    let mut attrs = axis.to_le_bytes().to_vec();
    attrs.push(u8::from(exclusive));
    attrs.push(u8::from(reverse));
    let id = m.add_node(
        Op::Custom {
            name: "onnx.CumProd".to_string(),
            num_inputs: 1,
            attrs,
        },
        vec![x],
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// `out[..,j,..] = data[..,idx[..,j,..],..]` — gather along `axis` matching the
/// batch coordinates (numpy `take_along_axis` / ONNX GatherElements). NOT a plain
/// `Gather`, which inserts the WHOLE index shape at `axis` and adds a rank (TopK
/// values `Gather(x[1,1024], idx[1,25], 1)` → `[1,1,25]` instead of `[1,25]`).
/// `idx` shares `data`'s rank; all dims equal except `axis`. Returns `idx`-shaped.
fn take_along_axis(
    m: &mut HirMut<'_>,
    ctx: &LowerCtx<'_>,
    data: HirNodeId,
    idx: HirNodeId,
    axis: usize,
) -> HirNodeId {
    let ds = m.shape(data).clone();
    let is = m.shape(idx).clone();
    let rank = ds.rank();
    let data_dims: Vec<usize> = ds.dims().iter().map(|d| dim_usize(*d, ctx.opts)).collect();
    let out_dims: Vec<usize> = is.dims().iter().map(|d| dim_usize(*d, ctx.opts)).collect();
    let mut dstride = vec![1i64; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        dstride[k] = dstride[k + 1] * data_dims[k + 1] as i64;
    }
    let mut ostride = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        ostride[k] = ostride[k + 1] * out_dims[k + 1];
    }
    let total: usize = out_dims.iter().product();
    let mut base = vec![0i64; total];
    for (lin, b) in base.iter_mut().enumerate() {
        let mut off = 0i64;
        for k in 0..rank {
            if k == axis {
                continue;
            }
            let coord = (lin / ostride[k]) % out_dims[k].max(1);
            off += coord as i64 * dstride[k];
        }
        *b = off;
    }
    let i64c = |v: &[i64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let idx = if m.shape(idx).dtype() != DType::I64 {
        let s = m.shape(idx).clone().with_dtype(DType::I64);
        m.add_node(Op::Cast { to: DType::I64 }, vec![idx], s)
    } else {
        idx
    };
    let s_axis_c = m.add_node(
        Op::Constant {
            data: i64c(&[dstride[axis]]),
        },
        vec![],
        Shape::new(&[1], DType::I64),
    );
    let idx_scaled = m.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![idx, s_axis_c],
        Shape::from_dims(is.dims(), DType::I64),
    );
    let base_id = m.add_node(
        Op::Constant { data: i64c(&base) },
        vec![],
        Shape::new(&out_dims, DType::I64),
    );
    let flat_idx = m.add_node(
        Op::Binary(BinaryOp::Add),
        vec![idx_scaled, base_id],
        Shape::new(&out_dims, DType::I64),
    );
    let flat_idx_1d = m.reshape_(flat_idx, vec![total as i64]);
    let data_total: usize = data_dims.iter().product();
    let flat_data = m.reshape_(data, vec![data_total as i64]);
    let gathered = m.add_node(
        Op::Gather { axis: 0 },
        vec![flat_data, flat_idx_1d],
        Shape::new(&[total], ds.dtype()),
    );
    m.reshape_(gathered, out_dims.iter().map(|&d| d as i64).collect())
}

pub(super) fn lower_topk(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank().max(1);
    let axis = normalize_axis(
        node.attrs
            .get("axis")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1),
        rank,
    );
    // `k` is authoritative as ONNX input[1] (a 1-elem i64 tensor). Prefer it: the
    // output-meta path resolves to 1 when the declared indices shape carries a
    // symbolic batch that `resolve_shape` rejects (MOSS `TopK(logits, 25)` came
    // back k=1 → the top-25 sampling collapsed to a single candidate).
    let k = node
        .inputs
        .get(1)
        .filter(|s| !s.is_empty())
        .and_then(|n| {
            i64_tensor(&ctx.i64_params, &ctx.params, n)
                .or_else(|| eval_static_shape_vector(ctx, m, n, 0))
        })
        .and_then(|v| v.first().copied())
        .map(|k| k.max(1) as usize)
        .or_else(|| {
            node.output_meta
                .get(1)
                .or(node.output_meta.first())
                .and_then(|meta| resolve_shape(meta, ctx.opts).ok())
                .map(|s| {
                    if s.rank() == 0 {
                        1
                    } else {
                        dim_usize(s.dim(axis.min(s.rank().saturating_sub(1))), ctx.opts)
                    }
                })
        })
        .unwrap_or(1)
        .max(1);
    // TopK reduces the selected axis to `k`. `output_shape` can resolve to the
    // INPUT extent on that axis when the declared meta carries a symbolic batch
    // (e.g. MOSS's `TopK(logits[.,1024], 25)` came back `[.,1024]` instead of
    // `[.,25]`), and that stale 1024 then poisons the sampled-token embedding
    // Gather downstream → a spurious dim in every later unrolled attention. Force
    // the reduced axis to `k`.
    // Indices shape = input with the reduced axis forced to `k` (`output_shape`
    // can resolve to the input extent when the meta batch is symbolic → stale
    // 1024). Indices are i64 (ONNX TopK output[1]); the Op::TopK kernel writes
    // per-row argmax indices.
    let idx_shape = {
        let base = output_shape(ctx, node, m, x);
        let mut dims: Vec<Dim> = (0..base.rank()).map(|d| base.dim(d)).collect();
        if axis < dims.len() {
            dims[axis] = Dim::Static(k);
        }
        Shape::from_dims(&dims, DType::I64)
    };
    let indices = m.add_node(Op::TopK { k }, vec![x], idx_shape);
    if node.outputs.len() >= 2 && !node.outputs[1].is_empty() {
        ctx.env.insert(node.outputs[1].clone(), indices);
    }
    if !node.outputs.is_empty() && !node.outputs[0].is_empty() {
        // values[..,j] = x[..,indices[..,j]] — take-along-axis, NOT plain Gather.
        let values = take_along_axis(m, ctx, x, indices, axis);
        ctx.env.insert(node.outputs[0].clone(), values);
    }
    Ok(true)
}

/// Lower ONNX `ArgMax` / `ArgMin` → native `Op::ArgMax` / `Op::ArgMin`. The
/// index output is I64 (ONNX convention); the reduced axis is removed
/// (`keepdims=0`) or collapsed to 1 (`keepdims=1`).
pub(super) fn lower_arg_reduce(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    is_max: bool,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank().max(1);
    let axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        rank,
    );
    let keep_dim = node
        .attrs
        .get("keepdims")
        .and_then(|v| v.as_i64())
        .unwrap_or(1)
        != 0;
    let in_s = m.shape(x).clone();
    let mut dims: Vec<Dim> = (0..in_s.rank()).map(|d| in_s.dim(d)).collect();
    if keep_dim {
        if axis < dims.len() {
            dims[axis] = Dim::Static(1);
        }
    } else if axis < dims.len() {
        dims.remove(axis);
    }
    let out_s = Shape::from_dims(&dims, DType::I64);
    let op = if is_max {
        Op::ArgMax { axis, keep_dim }
    } else {
        Op::ArgMin { axis, keep_dim }
    };
    let id = m.add_node(op, vec![x], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_cumsum(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank().max(1);
    let axis = node
        .inputs
        .get(1)
        .and_then(|n| i64_tensor(&ctx.i64_params, &ctx.params, n))
        .and_then(|v| v.first().copied())
        .map(|a| normalize_axis(a, rank))
        .unwrap_or(0);
    let exclusive = node
        .attrs
        .get("exclusive")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    // CumSum is shape-PRESERVING — its input is authoritative when static; the
    // ONNX `output_meta` can be a defaulted symbolic length (the ChatterBox S3Gen
    // sine-source phase accumulator `CumSum([1,9,11520])` came back `[3,128,1]`).
    let in_s = m.shape(x).clone();
    let s = if in_s.is_static() {
        in_s
    } else {
        resolve_shape(&node.output_meta[0], ctx.opts).unwrap_or_else(|_| m.shape(x).clone())
    };
    let last = rank.saturating_sub(1);
    let id = if rank > 0 && axis != last {
        // The cumsum kernel scans the *last* axis, so we swap `axis`↔`last`,
        // scan, then swap back. Skipping the swap-back leaves the result in the
        // transposed layout while still wearing the original output shape — the
        // data comes out permuted (e.g. `[1,9,148]` mislabelled as `[1,148,9]`).
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(axis, last);
        let t = m.transpose_(x, perm.clone());
        let ts = m.shape(t).clone();
        let cum = m.add_node(
            Op::Cumsum {
                axis: last as i32,
                exclusive,
            },
            vec![t],
            ts,
        );
        // `perm` is a single transposition, hence its own inverse.
        m.transpose_(cum, perm)
    } else {
        m.add_node(
            Op::Cumsum {
                axis: axis as i32,
                exclusive,
            },
            vec![x],
            s,
        )
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
