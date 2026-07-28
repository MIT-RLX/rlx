// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    let id = m.add_node(Op::Transpose { perm: perm.clone() }, vec![x], out_s);
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
        // `/Squeeze_4` legacy flatten: some exports use it on a `[1,…,N,…,1]`
        // signal (≤1 non-unit axis), where a full Squeeze IS `[N] = numel`. Guard
        // on that shape — a bare NAME match wrongly flattened the ChatterBox
        // speech_encoder's `/Squeeze_4` over `[1,198,512]` (TWO non-unit dims) to
        // `[101376]` instead of the correct `[198,512]`, breaking the mel STFT.
        let one_nonunit = in_s.dims().iter().filter(|&&d| dim_i64(d) > 1).count() <= 1;
        if node.name == "/Squeeze_4" && one_nonunit {
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
            if dims.is_empty() {
                // Empty after retain: ONNX Squeeze of a length-1 vector `[v]` (shape
                // `[1]`) would drop the sole size-1 axis. Keep `[1]` so scalar shape
                // values (e.g. Soprano SymSize after `Shape(…, start=2,end=3)` →
                // `[T]`) survive into Sub/Mul/Add for Vocos upsample arange math.
                vec![1]
            } else {
                dims
            }
        }
    } else if node.op == "Flatten" {
        // ONNX Flatten(axis): (∏ d[0..axis), ∏ d[axis..]). Stale `output_meta` for
        // Soprano's `/model/Flatten` is head_dim `[128]` while attention_mask is
        // `[1, seq]` → must derive from the live input.
        let rank = in_s.rank() as i64;
        let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(1);
        let axis = if axis < 0 { axis + rank } else { axis }.clamp(0, rank);
        let dims_u: Vec<i64> = in_s.dims().iter().map(|&d| dim_i64(d).max(0)).collect();
        if axis == 0 {
            let n = dims_u.iter().product::<i64>().max(1);
            vec![1, n]
        } else {
            let outer = dims_u[..axis as usize].iter().product::<i64>().max(1);
            let inner = dims_u[axis as usize..].iter().product::<i64>().max(1);
            vec![outer, inner]
        }
    } else if node.op == "Reshape" && node.inputs.len() >= 2 {
        // Prefer the actual reshape-target tensor (folded from the ONNX graph) — it's
        // authoritative. The bidirectional-LSTM merge heuristic is only a fallback;
        // trying it first wrongly matches VITS attention reshapes `[1,2,t,2t]` (it
        // reads the `2` heads as LSTM directions once `2t ≥ 64` looks like a channel).
        // Try each folding source and keep the FIRST whose product matches the
        // input's element count. `resolve_reshape_dims` must gate EACH source
        // independently — if `eval_i64_shaped` returns a stale/defaulted target
        // (e.g. the S3Gen sine-source `Reshape(Expand[1,1,11520], Shape(Add_1))`
        // where the i64 path reported the meta-defaulted `[1,1,128]`), a single
        // trailing `.and_then` would reject it and NEVER try the (correct)
        // `eval_static_shape_vector`, silently falling back to the stale meta.
        if let Some(dims) = eval_i64_shaped(ctx, m, &node.inputs[1], 0)
            .map(|(d, _)| d)
            .and_then(|d| resolve_reshape_dims(d, &in_s))
            .or_else(|| {
                eval_static_shape_vector(ctx, m, &node.inputs[1], 0)
                    .and_then(|d| resolve_reshape_dims(d, &in_s))
            })
        {
            dims
        } else if let Some(dims) = crate::layout::bidir_lstm_merge_reshape_dims(&in_s)
            .filter(|d| resolve_reshape_dims(d.clone(), &in_s).is_some())
        {
            dims
        } else if let Ok(s) = resolve_shape(&node.output_meta[0], ctx.opts) {
            let mut dims: Vec<i64> = s.dims().iter().map(|&d| dim_i64(d)).collect();
            // Vocos upsample `view` meta is often scribbled to head_dim (128) while
            // the true length is `4*s53 - 3`. Prefer the caller's named pin.
            if let Some(&frames) = ctx.opts.named_lengths.get("4*s53 - 3") {
                let frames = frames as i64;
                if frames > 1
                    && (node.name.contains("view")
                        || node.name.ends_with("_view")
                        || node.name.contains("Reshape_11"))
                    && (dims == [128] || dims == [1, 128] || dims == [128, 1])
                {
                    dims = if dims.len() == 2 && dims[0] == 1 {
                        vec![1, frames]
                    } else if dims.len() == 2 && dims[1] == 1 {
                        vec![frames, 1]
                    } else {
                        vec![frames]
                    };
                }
            }
            // Vocos ISTFT: flatten windowed frames `[1, n_fft, F] → [1, n_fft*F]`.
            // Meta / stale Concat often yields `[1, F]` (= frames) while input has
            // `8192*s53 - 6144` elements (`2048*(4*T-3)`).
            if let Some(&flat) = ctx.opts.named_lengths.get("8192*s53 - 6144") {
                let flat = flat as i64;
                let in_n = in_s.num_elements().unwrap_or(0) as i64;
                let meta_n: i64 = dims.iter().copied().filter(|&d| d > 0).product();
                if flat > 1 && in_n == flat && meta_n != flat {
                    if node.name.contains("_unsafe_view_1") {
                        dims = vec![flat];
                    } else if node.name.contains("_unsafe_view") {
                        dims = vec![1, flat];
                    }
                }
            }
            // Reject meta when product cannot reshape the input (avoids silent
            // `[1,45]` for a 92160-element tensor).
            if let Some(in_n) = in_s.num_elements() {
                let meta_n: i64 = dims.iter().copied().filter(|&d| d > 0).product();
                if meta_n > 0 && meta_n as usize != in_n {
                    if let Some(&flat) = ctx.opts.named_lengths.get("8192*s53 - 6144") {
                        if in_n == flat {
                            dims = if node.name.contains("_unsafe_view_1") {
                                vec![flat as i64]
                            } else {
                                vec![1, flat as i64]
                            };
                        } else {
                            dims = vec![in_n as i64];
                        }
                    } else {
                        dims = vec![in_n as i64];
                    }
                }
            }
            dims
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

/// ONNX Gather output shape: `table[:axis] ++ indices.shape ++ table[axis+1:]`.
/// Preserves each `Dim` (static or dynamic) verbatim from the operands.
fn gather_output_shape(table: &Shape, idx: &Shape, axis: usize) -> Option<Shape> {
    if axis >= table.rank() {
        return None;
    }
    let mut dims: Vec<Dim> = Vec::with_capacity(table.rank() - 1 + idx.rank());
    for i in 0..axis {
        dims.push(table.dim(i));
    }
    for i in 0..idx.rank() {
        dims.push(idx.dim(i));
    }
    for i in (axis + 1)..table.rank() {
        dims.push(table.dim(i));
    }
    Some(Shape::from_dims(&dims, table.dtype()))
}

pub(super) fn lower_gather(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    // Fold a pure shape-arithmetic gather (e.g. `Gather(Shape(x), idx)`) to a
    // small i64 Constant. A real Gather here would read the `Shape` output's
    // tensor SHAPE — which `lower_shape_op` materializes as the *input* shape
    // (`[1,100]` not `[2]`) — and produce a vector instead of the intended scalar
    // dim, poisoning downstream arithmetic (luxtts `prompt_features_len/text_len`).
    //
    // Do NOT fold large i64 gathers (Vocos ISTFT `Gather(add_307[1,n_fft,F], 0)` →
    // `[n_fft,F]` index grid). `eval_i64_shaped` on the output can spuriously
    // report a scalar `0` from attribute/index pollution and zero the window path.
    if let Some((data, shp)) = eval_i64_shaped(ctx, m, &node.outputs[0], 0) {
        let table_is_small_shape_vec = eval_i64_shaped(ctx, m, &node.inputs[0], 0)
            .map(|(d, s)| d.len() <= 64 && s.len() <= 2)
            .unwrap_or(false);
        if table_is_small_shape_vec
            && !data.is_empty()
            && data.len() <= 1 << 16
            && shp.iter().product::<usize>() == data.len()
        {
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
    let table = ctx.tensor(&node.inputs[0])?;
    let mut indices = ctx.tensor(&node.inputs[1])?;
    // Backends' Gather kernels only branch i64-vs-f32 on the index tensor: a
    // non-i64 INTEGER index (I32/U32, valid ONNX) is read through the f32 slice,
    // so `2797` (bytes 0x00000AED) becomes an f32 subnormal ≈ 3.9e-42 → `as usize`
    // = 0 → every lookup collapses to row 0. Normalize integer indices to I64 so
    // the well-supported path runs. (MiraTTS/BiCodec speaker-codebook Gather fed
    // I32 `context_tokens`; the I64 speech Gather was unaffected.)
    {
        let idx_dt = m.shape(indices).dtype();
        if idx_dt != DType::I64 && idx_dt != DType::F32 && idx_dt != DType::F16 {
            let cast_s = m.shape(indices).clone().with_dtype(DType::I64);
            indices = m.add_node(Op::Cast { to: DType::I64 }, vec![indices], cast_s);
        }
    }
    let table_s = m.shape(table).clone();
    let table_rank = table_s.rank();
    let axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        table_rank.max(1),
    );
    if table_rank == 0 || axis >= table_rank {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    // ONNX Gather output = table[:axis] ++ indices.shape ++ table[axis+1:]
    // (rank = table_rank - 1 + idx_rank). Derive the shape from the *live lowered
    // operands* at the compile length — this is ground truth. The ONNX-declared
    // `output_meta` is NOT reliable for dynamic-sequence exports: shape inference
    // bakes the symbolic seq dim to a concrete (often max) value that can disagree
    // with the actual index. Kokoro's BERT position embedding is the canonical
    // case: `position_embeddings.Gather(weight[512,128], idx[1,seq])` carries a
    // stale `output_meta=[1,512,128]` (seq→512 max) even though the real index is
    // `[1,15]`, so trusting meta yielded `[1,512,128]` and misaligned the embedding
    // `Add` (durations/prosody garbage). `gather_output_shape` from operands gives
    // the correct `[1,15,128]`. Fall back to `output_meta` only when the operand
    // rule can't apply (axis out of range) — the case that once needed meta
    // (dynamic `Expand` indices, meta *missing*) is already covered by operands.
    let idx_s = m.shape(indices).clone();
    let out_s = gather_output_shape(&table_s, &idx_s, axis)
        .unwrap_or_else(|| output_shape(ctx, node, m, table));
    let mut id = m.add_node(Op::Gather { axis }, vec![table, indices], out_s.clone());
    // ONNX `Gather` with a rank-0 SCALAR index removes the gathered axis. rlx pads
    // scalars to `[1]` (no rank-0 tensors), so the gather leaves a spurious size-1
    // dim at `axis` — reshape it away to match ONNX rank. Without this, F5-TTS's
    // `Gather(time_embed, time_step_scalar, axis=1)` produced `[1,1,256]` instead of
    // `[1,256]`, and the extra dim propagated through the whole AdaLN Gemm/MLP,
    // shifting a downstream axis-1 Slice off the real (6144) axis.
    if ctx.scalars.contains(&node.inputs[1]) && out_s.rank() > 1 && axis < out_s.rank() {
        let dims: Vec<i64> = out_s
            .dims()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, d)| d.unwrap_static() as i64)
            .collect();
        id = m.reshape_(id, dims);
    }
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `Tile(x, repeats)` — repeat `x` `repeats[i]` times along axis `i`.
/// Decomposed to per-axis `Concat` of copies (works on every backend; no native
/// Tile op). `repeats` is a static int tensor (initializer or const Concat).
pub(super) fn lower_tile(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank();
    let repeats: Vec<i64> = node
        .inputs
        .get(1)
        .filter(|s| !s.is_empty())
        .and_then(|n| {
            i64_tensor(&ctx.i64_params, &ctx.params, n)
                .or_else(|| eval_static_shape_vector(ctx, m, n, 0))
        })
        .unwrap_or_else(|| vec![1; rank]);
    let mut cur = x;
    for a in 0..rank {
        let r = repeats.get(a).copied().unwrap_or(1).max(1) as usize;
        if r <= 1 {
            continue;
        }
        cur = m.concat_(vec![cur; r], a);
    }
    ctx.env.insert(node.outputs[0].clone(), cur);
    Ok(true)
}

/// ONNX `GatherElements(data, indices, axis)` — first-class `Op::GatherElements`.
/// Extra size-1 index dims (MOSS) are squeezed then reshaped back after gather.
pub(super) fn lower_gather_elements(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let idx = ctx.tensor(&node.inputs[1])?;
    let ds = m.shape(data).clone();
    let is = m.shape(idx).clone();
    let rank = ds.rank();
    let axis = normalize_axis(
        node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0),
        rank.max(1),
    ) as i32;
    let orig_out_dims: Vec<i64> = is.dims().iter().map(|d| d.unwrap_static() as i64).collect();
    let mut idx = idx;
    let mut is = is;
    while is.rank() > rank {
        let pos = is.dims().iter().position(|d| d.unwrap_static() == 1);
        let Some(pos) = pos else { break };
        let new: Vec<i64> = is
            .dims()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != pos)
            .map(|(_, d)| d.unwrap_static() as i64)
            .collect();
        idx = m.reshape_(idx, new);
        is = m.shape(idx).clone();
    }
    if rank == 0 || is.rank() != rank {
        ctx.unsupported("GatherElements(rank-mismatch)");
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let gathered = m.add_node(Op::GatherElements { axis }, vec![data, idx], is.clone());
    let out = if orig_out_dims.len() != is.rank() {
        m.reshape_(gathered, orig_out_dims)
    } else {
        gathered
    };
    ctx.env.insert(node.outputs[0].clone(), out);
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
    let mut target_meta = match (evaluated.clone(), from_meta.clone()) {
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
    // Vocos / ISTFT ScatterElements builds zeros via Expand(0, shape=Concat(…)).
    // Soprano's decoder: shape folding yields either a bloated
    // `[2048, 512*2048, 2048*T]` or a collapsed `[2049,1,1]`, and shape_propagate
    // value_info on the Expand nodes is stale (`[128]` / `[1,128]`). When the
    // caller pinned `2048*s53` in `named_lengths`, force that buffer length.
    if let Some(&buf) = ctx.opts.named_lengths.get("2048*s53") {
        let is_zeros = node.name.ends_with("zeros")
            || node.name.ends_with("zeros_1")
            || node.name.contains("/zeros")
            || node.name == "node_zeros"
            || node.name == "node_zeros_1";
        if is_zeros && buf > 0 {
            let eval_elems: usize = target.iter().map(|&d| d.max(0) as usize).product();
            let need_fix = eval_elems > 64 * 1024 * 1024
                || eval_elems < buf
                || target.len() > 2
                || (target.len() == 3 && target[0] as usize != buf && target[0] as usize != 1);
            if need_fix {
                if node.name.contains("zeros_1") {
                    target = vec![buf as i64];
                    target_meta = Shape::new(&[buf], in_s.dtype());
                } else {
                    target = vec![1, buf as i64];
                    target_meta = Shape::new(&[1, buf], in_s.dtype());
                }
            }
        }
    }
    // Vocos ISTFT window: Expand `unsqueeze_7` `[1, n_fft, 1]` over
    // `Concat([1],[1], frames)` → `[1, n_fft, frames]`. Stale Concat/meta often
    // yields `[3,2048,1]` (or similar) when `frames = 4*s53-3` fails to fold.
    if let Some(&frames) = ctx.opts.named_lengths.get("4*s53 - 3") {
        if frames > 1
            && (node.name.contains("expand_1") || node.name.ends_with("/expand_1"))
            && in_s.rank() == 3
        {
            let d0 = in_s.dim(0).unwrap_static();
            let d1 = in_s.dim(1).unwrap_static();
            let d2 = in_s.dim(2).unwrap_static();
            if d0 == 1 && d1 >= 256 && d2 == 1 {
                let want_elems = d1 * frames;
                let have: usize = target.iter().map(|&d| d.max(0) as usize).product();
                if have != want_elems {
                    target = vec![1, d1 as i64, frames as i64];
                    target_meta = Shape::new(&[1, d1, frames], in_s.dtype());
                }
            }
        }
    }
    // Fall back: prefer meta over absurd evaluations when no named buffer.
    const MAX_EXPAND_ELEMS: usize = 64 * 1024 * 1024;
    let eval_elems: usize = target.iter().map(|&d| d.max(0) as usize).product();
    if eval_elems > MAX_EXPAND_ELEMS {
        if let Some(meta) = from_meta.clone() {
            let meta_dims: Vec<i64> = meta
                .dims()
                .iter()
                .map(|&d| match d {
                    Dim::Static(n) => n as i64,
                    Dim::Dynamic(_) => ctx.opts.sequence_length as i64,
                })
                .collect();
            let meta_elems: usize = meta_dims.iter().map(|&d| d.max(0) as usize).product();
            if meta_elems > 0 && meta_elems <= MAX_EXPAND_ELEMS && meta_elems >= 2048 {
                target = meta_dims;
                target_meta = meta;
            }
        }
    }
    let shape = rlx_ir::shape::expand_shape(&in_s, &target).unwrap_or_else(|_| target_meta.clone());
    let out_shape = if ctx.opts.dynamic_sequence {
        target_meta
    } else {
        shape.clone()
    };
    // `Op::Expand`'s `target_shape` is the OUTPUT (post-broadcast) shape — some
    // backends (wgpu) read it as the literal output extent, while others (CPU)
    // use `node.shape`. The raw ONNX shape-input can under-specify an axis as 1
    // where the INPUT dim is larger (ONNX broadcast keeps the max): e.g. a
    // prepended learned token `Expand([1,64,1], shape=[1,1,1]) → [1,64,1]`.
    // Passing the raw `[1,1,1]` makes wgpu compute out_dims `[1,1,1]` → every
    // output reads `input[0]` → the whole tensor collapses to one value
    // (supertonic's sentence_token went uniform on wgpu → wrong duration +
    // near-silent audio; CPU was unaffected). Reconcile each axis up to the
    // static broadcast output so `target_shape` matches `node.shape`.
    if shape.rank() == target.len() {
        for (i, d) in shape.dims().iter().enumerate() {
            if let Dim::Static(n) = d {
                if target[i] < *n as i64 {
                    target[i] = *n as i64;
                }
            }
        }
    }
    // Identity expand (target == input shape, all static) is now a no-op — pass
    // the input through instead of emitting a redundant `Op::Expand`.
    let in_dims = in_s.dims();
    if in_dims.len() == target.len()
        && in_dims
            .iter()
            .zip(&target)
            .all(|(d, &t)| matches!(d, Dim::Static(n) if *n as i64 == t))
    {
        ctx.env.insert(node.outputs[0].clone(), x);
        return Ok(true);
    }
    // GQA head-repeat: Expand `[1,1,1,S,H] → [1,1,n_heads,S,H]` (Soprano/Qwen3).
    // Some Expand backends mishandle this pattern; Concat of copies along the
    // heads axis is exact and keeps Layer-0 K/V parity through attention.
    if in_s.rank() == 5
        && target.len() == 5
        && matches!(in_s.dim(0), Dim::Static(1))
        && matches!(in_s.dim(1), Dim::Static(1))
        && matches!(in_s.dim(2), Dim::Static(1))
        && target[0] == 1
        && target[1] == 1
        && target[2] > 1
        && matches!(in_s.dim(3), Dim::Static(s) if s as i64 == target[3])
        && matches!(in_s.dim(4), Dim::Static(h) if h as i64 == target[4])
    {
        let heads = target[2] as usize;
        let id = m.concat_(vec![x; heads], 2);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
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
    let rank = in_s.rank() as i64;
    // ONNX `Shape` (opset 15+) optional `start`/`end` select a half-open slice of
    // axes — e.g. Soprano Vocos `Shape(hidden, start=2, end=3)` → `[s53]` only.
    // Ignoring them kept the full `[1,512,T]` vector, broke `arange` length, and
    // left ISTFT frames at head_dim (128) with silent audio.
    let start = node
        .attrs
        .get("start")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let end = node
        .attrs
        .get("end")
        .and_then(|v| v.as_i64())
        .unwrap_or(rank);
    let start = start.rem_euclid(rank.max(1));
    let end = if end < 0 {
        end.rem_euclid(rank.max(1))
    } else {
        end.min(rank)
    };
    let start_u = start as usize;
    let end_u = end.max(start) as usize;
    let dims: Vec<i64> = in_s
        .dims()
        .iter()
        .enumerate()
        .filter(|(i, _)| *i >= start_u && *i < end_u)
        .map(|(_, &d)| match d {
            Dim::Static(n) => n as i64,
            Dim::Dynamic(_) => ctx.opts.sequence_length as i64,
        })
        .collect();
    let bytes: Vec<u8> = dims.iter().flat_map(|d| d.to_le_bytes()).collect();
    // `Shape(x)` is a 1-D i64 vector of length `end-start` — NOT the input shape.
    let vec_s = Shape::new(&[dims.len().max(1)], DType::I64);
    let id = m.add_node(Op::Constant { data: bytes }, vec![], vec_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
