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
    let rop = match op {
        "ReduceSum" => ReduceOp::Sum,
        "ReduceMax" => ReduceOp::Max,
        "ReduceMin" => ReduceOp::Min,
        "ReduceProd" => ReduceOp::Prod,
        _ => ReduceOp::Mean,
    };
    let id = match rop {
        ReduceOp::Mean => m.mean(x, axes, keep),
        ReduceOp::Sum => m.sum(x, axes, keep),
        _ => m.add_node(
            Op::Reduce {
                op: rop,
                axes,
                keep_dim: keep,
            },
            vec![x],
            output_shape(ctx, node, m, x),
        ),
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}


/// ONNX `CumProd` — cumulative product along `axis` (input 1, baked into attrs
/// when constant). Mirrors `CumSum`'s `exclusive`/`reverse` attributes, packed
/// as `[axis_i32, exclusive_u8, reverse_u8]`.
pub(super) fn lower_cumprod(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
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


pub(super) fn lower_topk(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let rank = m.shape(x).rank().max(1);
    let axis = normalize_axis(
        node.attrs
            .get("axis")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1),
        rank,
    );
    let k = node
        .output_meta
        .get(1)
        .or(node.output_meta.first())
        .and_then(|m| resolve_shape(m, ctx.opts).ok())
        .map(|s| {
            if s.rank() == 0 {
                1
            } else {
                dim_usize(s.dim(axis.min(s.rank().saturating_sub(1))), ctx.opts)
            }
        })
        .unwrap_or(1)
        .max(1);
    let idx_shape = output_shape(ctx, node, m, x);
    let indices = m.add_node(Op::TopK { k }, vec![x], idx_shape);
    if node.outputs.len() >= 2 {
        ctx.env.insert(node.outputs[1].clone(), indices);
        let values = m.gather_(x, indices, axis);
        ctx.env.insert(node.outputs[0].clone(), values);
    } else if !node.outputs.is_empty() {
        ctx.env.insert(node.outputs[0].clone(), indices);
    }
    Ok(true)
}


pub(super) fn lower_cumsum(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
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
    let s = resolve_shape(&node.output_meta[0], ctx.opts).unwrap_or_else(|_| m.shape(x).clone());
    let last = rank.saturating_sub(1);
    let (src, ax) = if rank > 0 && axis != last {
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.swap(axis, last);
        let t = m.transpose_(x, perm);
        (t, last as i32)
    } else {
        (x, axis as i32)
    };
    let id = m.add_node(
        Op::Cumsum {
            axis: ax,
            exclusive,
        },
        vec![src],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

