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

pub(super) fn lower_scatter_nd(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let updates = ctx.tensor(&node.inputs[2])?;
    let s = m.shape(data).clone();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ScatterND".to_string(),
            num_inputs: 3,
            attrs: vec![],
        },
        vec![data, indices, updates],
        s,
    );
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
    let s = m.shape(data).clone();
    let attrs = axis.to_le_bytes().to_vec();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ScatterElements".to_string(),
            num_inputs: 3,
            attrs,
        },
        vec![data, indices, updates],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}


/// ONNX `GatherND` (opset 11+). Gathers slices from `data` indexed by the
/// trailing axis of `indices`; lowered to a reference CPU kernel. The
/// `batch_dims` attribute is forwarded in the op attrs (i32 LE).
pub(super) fn lower_gather_nd(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let data = ctx.tensor(&node.inputs[0])?;
    let indices = ctx.tensor(&node.inputs[1])?;
    let batch_dims = node
        .attrs
        .get("batch_dims")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let out_s = output_shape(ctx, node, m, data);
    let attrs = batch_dims.to_le_bytes().to_vec();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.GatherND".to_string(),
            num_inputs: 2,
            attrs,
        },
        vec![data, indices],
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}


/// ONNX `OneHot` (opset 9+). Inputs `[indices, depth, values]`; `values` is
/// `[off_value, on_value]`. The `axis` attribute (default -1) selects where the
/// new depth axis is inserted and is forwarded in the op attrs (i32 LE).
pub(super) fn lower_one_hot(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
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
pub(super) fn lower_non_zero(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x);
    let rank = in_s.rank().max(1);
    let numel = in_s.num_elements().unwrap_or(0);
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

