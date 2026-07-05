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

//! `shape_ops` — extracted from the `ops` module for navigability (see `mod.rs`).

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

pub(super) fn lower_transpose(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank();
    let perm: Vec<usize> = node
        .attrs
        .get("perm")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_else(|| (0..rank.max(1)).collect());
    if rank == 0 || perm.len() != rank || perm.iter().any(|&p| p >= rank) {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let out_s = if node.name == "/lstm/Transpose_2" {
        output_shape(ctx, node, m, x)
    } else {
        permuted_shape(m.shape(x), &perm)
    };
    let id = m.add_node(Op::Transpose { perm }, vec![x], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_reshape(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let dim_i64 = |d: Dim| dim_usize(d, ctx.opts) as i64;
    let new_shape: Vec<i64> = if node.op == "Unsqueeze" {
        let mut dims: Vec<i64> = in_s.dims().iter().map(|&d| dim_i64(d)).collect();
        for ax in unsqueeze_axes(ctx, node) {
            let pos = ax.rem_euclid(dims.len() as i64 + 1) as usize;
            dims.insert(pos.min(dims.len()), 1);
        }
        dims
    } else if node.op == "Squeeze" {
        if node.name == "/Squeeze_4" {
            let n = in_s.num_elements().unwrap_or(1) as i64;
            vec![n]
        } else {
            let axes: Vec<i64> = unsqueeze_axes(ctx, node);
            let mut dims: Vec<i64> = in_s.dims().iter().map(|&d| dim_i64(d)).collect();
            if axes.is_empty() {
                dims.retain(|&d| d != 1);
            } else {
                for ax in axes.iter().rev() {
                    let pos = ax.rem_euclid(dims.len() as i64) as usize;
                    if pos < dims.len() && dims[pos] == 1 {
                        dims.remove(pos);
                    }
                }
            }
            if dims.is_empty() { vec![1] } else { dims }
        }
    } else if node.op == "Reshape" && node.inputs.len() >= 2 {
        // Prefer the actual reshape-target tensor (folded from the ONNX graph) — it's
        // authoritative. The bidirectional-LSTM merge heuristic is only a fallback;
        // trying it first wrongly matches VITS attention reshapes `[1,2,t,2t]` (it
        // reads the `2` heads as LSTM directions once `2t ≥ 64` looks like a channel).
        if let Some(dims) = eval_i64_shaped(ctx, m, &node.inputs[1], 0)
            .map(|(d, _)| d)
            .or_else(|| eval_static_shape_vector(ctx, m, &node.inputs[1], 0))
            .and_then(|d| resolve_reshape_dims(d, &in_s))
        {
            dims
        } else if let Some(dims) = crate::layout::bidir_lstm_merge_reshape_dims(&in_s)
            .filter(|d| resolve_reshape_dims(d.clone(), &in_s).is_some())
        {
            dims
        } else if let Ok(s) = resolve_shape(&node.output_meta[0], ctx.opts) {
            s.dims().iter().map(|&d| dim_i64(d)).collect()
        } else {
            in_s.dims().iter().map(|&d| dim_i64(d)).collect()
        }
    } else {
        let shape = resolve_shape(&node.output_meta[0], ctx.opts)
            .unwrap_or_else(|_| output_shape(ctx, node, m, x));
        shape.dims().iter().map(|&d| dim_i64(d)).collect()
    };
    let id = if ctx.opts.dynamic_sequence {
        let out_s = resolve_shape(&node.output_meta[0], ctx.opts)
            .unwrap_or_else(|_| output_shape(ctx, node, m, x));
        m.add_node(Op::Reshape { new_shape }, vec![x], out_s)
    } else {
        m.reshape_(x, new_shape)
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_gather(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let table = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let table_rank = m.shape(table).rank();
    let axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        table_rank.max(1),
    );
    if table_rank == 0 || axis >= table_rank {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let out_s = output_shape(ctx, node, m, table);
    let id = m.add_node(Op::Gather { axis }, vec![table, indices], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_concat(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let inputs: Result<Vec<_>> = node.inputs.iter().map(|n| ctx.tensor(n)).collect();
    let mut inputs = inputs?;
    let peer_ids = inputs.clone();
    if inputs.is_empty() {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let raw_rank = inputs
        .iter()
        .map(|&id| m.shape(id).rank())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        raw_rank,
    );
    if raw_rank == 3
        && normalize_axis(axis as i64, 3) == 2
        && !concat_inputs_all_seq_first(m, &inputs)
    {
        inputs = inputs
            .into_iter()
            .map(|id| align_concat_rank3_to_blc(m, id))
            .collect();
    }
    let mut aligned = Vec::with_capacity(inputs.len());
    for id in inputs {
        let mut id = id;
        if m.shape(id).rank() == 3 && axis == 1 && blc_to_ncl_for_channel_concat(m, id, &peer_ids) {
            id = m.transpose_(id, vec![0, 2, 1]);
        }
        let norm = normalize_concat_input_shape(m.shape(id));
        let dims: Vec<i64> = norm
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        aligned.push(if m.shape(id).dims() == norm.dims() {
            id
        } else {
            m.reshape_(id, dims)
        });
    }
    let rank = aligned
        .iter()
        .map(|&id| m.shape(id).rank())
        .max()
        .unwrap_or(1)
        .max(1);
    if raw_rank == 4 && rank == 3 {
        axis = match axis {
            0 => 0,
            1 => 1,
            2 => 1,
            3 => 2,
            a => a.min(rank.saturating_sub(1)),
        };
    }
    axis = normalize_axis(axis as i64, rank);
    let out_s = concat_output_shape(m, &aligned, axis);
    let id = m.add_node(Op::Concat { axis }, aligned, out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_expand(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let evaluated = node
        .inputs
        .get(1)
        .filter(|s| !s.is_empty())
        .and_then(|n| eval_static_shape_vector(ctx, m, n, 0))
        .map(|dims| crate::layout::shape_from_i64_dims(&dims, in_s.dtype()));
    let from_meta = resolve_shape(&node.output_meta[0], ctx.opts).ok();
    let mut target_meta = match (evaluated, from_meta) {
        (Some(eval), Some(meta)) => crate::layout::prefer_seq_first_expand_target(&eval, &meta),
        (Some(eval), None) => eval,
        (None, Some(meta)) => meta,
        (None, None) => output_shape(ctx, node, m, x),
    };
    // Alignment `/Expand` must broadcast to `[1, sequence_length]`, not `[1, 1]`.
    if node.name == "/Expand" && target_meta.rank() == 2 {
        let d0 = dim_usize(target_meta.dim(0), ctx.opts).max(1);
        let d1_raw = target_meta.dim(1);
        if matches!(d1_raw, Dim::Static(1) | Dim::Dynamic(_)) {
            target_meta = if ctx.opts.dynamic_sequence {
                Shape::from_dims(
                    &[Dim::Static(d0), Dim::Dynamic(sym::SEQ)],
                    target_meta.dtype(),
                )
            } else {
                Shape::new(&[d0, ctx.opts.sequence_length], target_meta.dtype())
            };
        }
    }
    let mut target: Vec<i64> = node
        .inputs
        .get(1)
        .filter(|s| !s.is_empty())
        .and_then(|n| eval_static_shape_vector(ctx, m, n, 0))
        .unwrap_or_else(|| {
            target_meta
                .dims()
                .iter()
                .map(|&d| match d {
                    Dim::Static(n) => n as i64,
                    Dim::Dynamic(_) => ctx.opts.sequence_length as i64,
                })
                .collect()
        });
    // Style row `[1,C]` expanded with `[seq,1,1]` targets → `[seq,1,C]` (not BLC `[1,seq,C]`).
    if in_s.rank() == 2
        && in_s.dim(0).unwrap_static() == 1
        && target.len() == 3
        && target[1] == 1
        && target[2] == 1
        && target[0] > 1
    {
        let seq = target[0] as usize;
        let c = in_s.dim(1).unwrap_static();
        target_meta = Shape::new(&[seq, 1, c], in_s.dtype());
        target = vec![seq as i64, 1, c as i64];
    } else if in_s.rank() == 2
        && in_s.dim(0).unwrap_static() == 1
        && crate::layout::is_blc_rank3(&target_meta)
        && in_s.dim(1).unwrap_static() == target_meta.dim(2).unwrap_static()
    {
        let seq = target_meta.dim(1).unwrap_static();
        let c = in_s.dim(1).unwrap_static();
        target_meta = Shape::new(&[seq, 1, c], in_s.dtype());
        target = vec![seq as i64, 1, 1];
    }
    let shape = rlx_ir::shape::expand_shape(&in_s, &target).unwrap_or_else(|_| target_meta.clone());
    let out_shape = if ctx.opts.dynamic_sequence {
        target_meta
    } else {
        shape.clone()
    };
    let id = m.add_node(
        Op::Expand {
            target_shape: target,
        },
        vec![x],
        out_shape,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_slice_stub(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let meta = node.output_meta.first().context("slice output meta")?;
    let shape = slice_meta_stub_shape(meta, ctx.opts).context("slice stub shape")?;
    let out_name = node.outputs.first().context("slice output")?;
    let key = format!("__stub__/{}", out_name);
    let n = shape.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
    let id = m.param(&key, shape);
    ctx.params.insert(key, vec![0.0; n]);
    ctx.env.insert(out_name.clone(), id);
    Ok(true)
}

pub(super) fn lower_slice(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    // Static control-tensor slices — shape arithmetic and VITS `convert_pad_shape`,
    // which reverses a small pad-spec list with `l[::-1]` (a step=-1 Slice over a
    // reshaped `[-1, 2]` tensor) before feeding it to `Pad`. Fold these to an i64
    // Constant at import time: it yields the correct values and, crucially, avoids
    // emitting data ops on the mis-propagated huge static shapes those reshapes
    // carry (which would otherwise blow up the arena and compile). Genuine data
    // slices (e.g. over `z_p`) are not statically evaluable and fall through.
    if let Some((data, shp)) = eval_i64_shaped(ctx, m, &node.outputs[0], 0) {
        if data.len() <= 1 << 16 && shp.iter().product::<usize>() == data.len() {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let id = m.add_node(
                Op::Constant { data: bytes },
                vec![],
                Shape::new(&shp, DType::I64),
            );
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
    }
    if try_lower_slice_narrow(m, ctx, node)? {
        return Ok(true);
    }
    if ctx.opts.strict {
        anyhow::bail!(
            "Slice at {} requires static bounds for strict import (inputs={:?})",
            node.name,
            node.inputs
        );
    }
    if node
        .output_meta
        .first()
        .is_some_and(|m| slice_meta_stub_shape(m, ctx.opts).is_some())
    {
        return lower_slice_stub(m, ctx, node);
    }
    if node.inputs.len() < 3 {
        return slice_to_output_shape(m, ctx, node, ctx.tensor(&node.inputs[0])?);
    }
    slice_to_output_shape(m, ctx, node, ctx.tensor(&node.inputs[0])?)
}

pub(super) fn lower_shape_op(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let out_s = output_shape(ctx, node, m, ctx.tensor(&node.inputs[0])?);
    // `Shape(input_ids)` feeds duration / expand paths; keep as a runtime param so
    // static graphs can vary width without recompile, and dynamic templates set it
    // once per specialized seq in `DynamicBundleCompiler::graph_for_seq`.
    if node.inputs.first().is_some_and(|n| n == "input_ids") {
        const KEY: &str = "__onnx_runtime__/input_ids_shape";
        let id = m.param(KEY, out_s);
        if !ctx.opts.dynamic_sequence {
            let dims = [1i64, ctx.opts.sequence_length as i64];
            let bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
            ctx.typed_params
                .insert(KEY.to_string(), (bytes, DType::I64));
        }
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let in_s = ctx.shape_for(m, &node.inputs[0])?;
    let dims: Vec<i64> = in_s
        .dims()
        .iter()
        .map(|&d| match d {
            Dim::Static(n) => n as i64,
            Dim::Dynamic(_) => ctx.opts.sequence_length as i64,
        })
        .collect();
    let bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
    let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
