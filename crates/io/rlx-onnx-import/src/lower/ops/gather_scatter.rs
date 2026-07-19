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

//! `gather_scatter` — extracted from the `ops` module for navigability (see `mod.rs`).

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

pub(super) fn lower_scatter_nd(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let updates = ctx.tensor(&node.inputs[2])?;
    // ONNX `reduction`: none | add | mul | max | min (opset 16+).
    let reduction = match node
        .attrs
        .get("reduction")
        .and_then(|v| v.as_str())
        .unwrap_or("none")
    {
        "add" => rlx_ir::ScatterNdReduction::Add,
        "mul" => rlx_ir::ScatterNdReduction::Mul,
        "max" => rlx_ir::ScatterNdReduction::Max,
        "min" => rlx_ir::ScatterNdReduction::Min,
        _ => rlx_ir::ScatterNdReduction::None,
    };
    let s = m.shape(data).clone();
    // The CPU Scatter/Gather-ND/Elements kernels copy elements as f32 (`sl(*data)`
    // + `*_f32`). Non-f32 data is silently misread at the wrong byte width — e.g.
    // ChatterBox's conditional_decoder scatters a BOOL mask, whose 1-byte elements
    // read as 4-byte f32 corrupt the result. Route non-f32 data+updates through
    // f32 (Cast is exact for bool / |int| < 2^24 — the only realistic non-f32
    // operands for these index ops) and cast the scattered result back.
    let dt = s.dtype();
    if dt != DType::F32 {
        let sf = s.clone().with_dtype(DType::F32);
        let uf = m.shape(updates).clone().with_dtype(DType::F32);
        let data_f = m.add_node(Op::Cast { to: DType::F32 }, vec![data], sf.clone());
        let upd_f = m.add_node(Op::Cast { to: DType::F32 }, vec![updates], uf);
        let scat = m.add_node(Op::ScatterNd { reduction }, vec![data_f, indices, upd_f], sf);
        let back = m.add_node(Op::Cast { to: dt }, vec![scat], s);
        ctx.env.insert(node.outputs[0].clone(), back);
        return Ok(true);
    }
    let id = m.add_node(Op::ScatterNd { reduction }, vec![data, indices, updates], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_scatter_elements(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let updates = ctx.tensor(&node.inputs[2])?;
    let axis = node.attrs.get("axis").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let reduction = match node
        .attrs
        .get("reduction")
        .and_then(|v| v.as_str())
        .unwrap_or("none")
    {
        "add" => rlx_ir::ScatterNdReduction::Add,
        "mul" => rlx_ir::ScatterNdReduction::Mul,
        "max" => rlx_ir::ScatterNdReduction::Max,
        "min" => rlx_ir::ScatterNdReduction::Min,
        _ => rlx_ir::ScatterNdReduction::None,
    };
    let s = m.shape(data).clone();
    // Same f32-only-kernel guard as ScatterND (see lower_scatter_nd).
    let dt = s.dtype();
    if dt != DType::F32 {
        let sf = s.clone().with_dtype(DType::F32);
        let uf = m.shape(updates).clone().with_dtype(DType::F32);
        let data_f = m.add_node(Op::Cast { to: DType::F32 }, vec![data], sf.clone());
        let upd_f = m.add_node(Op::Cast { to: DType::F32 }, vec![updates], uf);
        let scat = m.add_node(
            Op::ScatterElements { axis, reduction },
            vec![data_f, indices, upd_f],
            sf,
        );
        let back = m.add_node(Op::Cast { to: dt }, vec![scat], s);
        ctx.env.insert(node.outputs[0].clone(), back);
        return Ok(true);
    }
    let id = m.add_node(
        Op::ScatterElements { axis, reduction },
        vec![data, indices, updates],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `GatherND` (opset 11+).
pub(super) fn lower_gather_nd(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let batch_dims = node
        .attrs
        .get("batch_dims")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let batch = batch_dims.max(0) as usize;
    let data_s = m.shape(data).clone();
    let idx_s = m.shape(indices).clone();
    let out_s = if idx_s.rank() >= 1 {
        let index_depth = idx_s.dim(idx_s.rank() - 1).unwrap_static();
        let mut dims: Vec<usize> = idx_s.dims()[..idx_s.rank() - 1]
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let skip = batch + index_depth;
        for d in data_s.dims().iter().skip(skip) {
            dims.push(d.unwrap_static());
        }
        if dims.is_empty() {
            output_shape(ctx, node, m, data)
        } else {
            Shape::new(&dims, data_s.dtype())
        }
    } else {
        output_shape(ctx, node, m, data)
    };
    // Same f32-only-kernel guard as ScatterND (see lower_scatter_nd): GatherND's
    // CPU kernel copies data elements as f32, so route non-f32 data through f32.
    let dt = out_s.dtype();
    if dt != DType::F32 {
        let df = m.shape(data).clone().with_dtype(DType::F32);
        let of = out_s.clone().with_dtype(DType::F32);
        let data_f = m.add_node(Op::Cast { to: DType::F32 }, vec![data], df);
        let g = m.add_node(Op::GatherNd { batch_dims }, vec![data_f, indices], of);
        let back = m.add_node(Op::Cast { to: dt }, vec![g], out_s);
        ctx.env.insert(node.outputs[0].clone(), back);
        return Ok(true);
    }
    let id = m.add_node(Op::GatherNd { batch_dims }, vec![data, indices], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `OneHot` (opset 9+). Inputs `[indices, depth, values]`; `values` is
/// `[off_value, on_value]`. The `axis` attribute (default -1) selects where the
/// new depth axis is inserted and is forwarded in the op attrs (i32 LE).
pub(super) fn lower_one_hot(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let indices = ctx.tensor(&node.inputs[0])?;
    let depth = ctx.tensor(&node.inputs[1])?;
    let values = ctx.tensor(&node.inputs[2])?;
    let axis = node
        .attrs
        .get("axis")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    // Prefer the model-provided output shape; fall back to inserting `depth`
    // (when statically known) into the indices shape at `axis`.
    let out_s = resolve_shape(&node.output_meta[0], ctx.opts)
        .ok()
        .filter(|s| s.num_elements().unwrap_or(0) > 0)
        .unwrap_or_else(|| {
            let idx_s = m.shape(indices);
            let rank = idx_s.rank();
            let depth_val = i64_tensor(&ctx.i64_params, &ctx.params, &node.inputs[1])
                .and_then(|v| v.first().copied())
                .unwrap_or(0)
                .max(0) as usize;
            let pos = if axis < 0 {
                (rank as i32 + 1 + axis).max(0) as usize
            } else {
                (axis as usize).min(rank)
            };
            let mut dims: Vec<usize> = idx_s.dims().iter().map(|d| d.unwrap_static()).collect();
            dims.insert(pos.min(dims.len()), depth_val);
            Shape::new(&dims, m.shape(values).dtype())
        });
    let attrs = axis.to_le_bytes().to_vec();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.OneHot".to_string(),
            num_inputs: 3,
            attrs,
        },
        vec![indices, depth, values],
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `NonZero` (opset 9+). Output is `[rank, nnz]` of I64 indices — `nnz` is
/// data-dependent, so the static buffer is sized at the model-provided shape
/// when available, else the `[rank, numel]` upper bound. The kernel zero-pads
/// any unused tail.
pub(super) fn lower_non_zero(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x);
    let rank = in_s.rank().max(1);
    let numel = in_s.num_elements().unwrap_or(0);
    // Fold when the input is a COMPILE-TIME-CONSTANT 1-D mask — `NonZero` then has
    // no CPU kernel and a data-dependent length, but its result is static (e.g.
    // the ISTFTNet ISTFT window-sum normalization: `NonZero(window_sum > 0)`, and
    // `window_sum` is a fixed initializer). Emit the nonzero indices as a
    // [1, count] i64 Constant (ONNX NonZero of a rank-1 input = one index row).
    if in_s.rank() == 1 {
        if let Some(vals) = eval_const_f64_vec(ctx, m, &node.inputs[0], 0) {
            let idx: Vec<i64> = vals
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v != 0.0)
                .map(|(i, _)| i as i64)
                .collect();
            let bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
            let id = m.add_node(
                Op::Constant { data: bytes },
                vec![],
                Shape::new(&[1, idx.len()], DType::I64),
            );
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
    }
    // A `NonZero` over an ALL-nonzero mask has exactly `numel` results (one index
    // row per rank). The ChatterBox speech_encoder builds a valid-frame mask as
    // `NonZero(ConstantOfShape(frames, fill=1))` — every element is nonzero, so
    // the (declared-symbolic, meta-DEFAULTED) output length is really the static
    // frame count = the input's element count. Prefer `[rank, numel]` there.
    let input_all_nonzero = bundle_node_for_output_ctx(ctx, &node.inputs[0])
        .filter(|n| n.op == "ConstantOfShape")
        .map(|n| {
            n.attrs
                .get("value")
                .and_then(|v| {
                    v.get("tensor")
                        .and_then(|t| t.get("scalar"))
                        .and_then(|s| s.as_f64())
                        .or_else(|| {
                            v.as_array()
                                .and_then(|a| a.first())
                                .and_then(|x| x.as_f64())
                        })
                        .or_else(|| v.as_f64())
                        .or_else(|| v.as_i64().map(|x| x as f64))
                })
                .unwrap_or(0.0)
                != 0.0
        })
        .unwrap_or(false);
    // For an ALL-nonzero mask the result is data-INDEPENDENT: it enumerates every
    // position in row-major order, i.e. the `[rank, numel]` I64 coordinate matrix
    // (`out[axis*numel + j] = (j / stride[axis]) % dim[axis]`). Bake it as an I64
    // `Constant` instead of a host-delegated `Op::Custom("onnx.NonZero")` — the
    // custom kernel has no GPU implementation and, on f32-arena backends
    // (Metal/wgpu/CoreML), its I64 output loses its dtype so a downstream
    // Gather/Scatter `expect_i64` fails ("indices: expected I64, got F32"). A
    // Constant carries its dtype on every backend and needs no kernel at all.
    let dims_static: Option<Vec<usize>> = m
        .shape(x)
        .dims()
        .iter()
        .map(|d| match d {
            rlx_ir::Dim::Static(n) => Some(*n),
            _ => None,
        })
        .collect();
    if input_all_nonzero && numel > 0 {
        if let Some(dims) = dims_static.filter(|d| d.len() == rank) {
            let mut stride = vec![1usize; rank];
            for a in (0..rank.saturating_sub(1)).rev() {
                stride[a] = stride[a + 1] * dims[a + 1].max(1);
            }
            let mut idx = vec![0i64; rank * numel];
            for a in 0..rank {
                let (st, da) = (stride[a].max(1), dims[a].max(1));
                for j in 0..numel {
                    idx[a * numel + j] = ((j / st) % da) as i64;
                }
            }
            let bytes: Vec<u8> = idx.iter().flat_map(|v| v.to_le_bytes()).collect();
            let id = m.add_node(
                Op::Constant { data: bytes },
                vec![],
                Shape::new(&[rank, numel], DType::I64),
            );
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
    }
    // Genuinely data-dependent NonZero — host-delegate to the CPU kernel.
    let out_s = resolve_shape(&node.output_meta[0], ctx.opts)
        .ok()
        .filter(|s| s.num_elements().unwrap_or(0) > 0)
        .unwrap_or_else(|| Shape::new(&[rank, numel], DType::I64));
    let id = m.add_node(
        Op::Custom {
            name: "onnx.NonZero".to_string(),
            num_inputs: 1,
            attrs: vec![],
        },
        vec![x],
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
