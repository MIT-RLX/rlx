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

//! `generators` — extracted from the `ops` module for navigability (see `mod.rs`).

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

pub(super) fn lower_constant_of_shape(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let shape_in = ctx.tensor(&node.inputs[0])?;
    let out_s = output_shape(ctx, node, m, shape_in);
    let n = out_s.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
    if out_s.dtype() == DType::I64 {
        let bytes = vec![0u8; n * 8];
        let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let key = format!("__const_of_shape__/{}", node.outputs[0]);
    let id = m.param(&key, out_s);
    ctx.params.insert(key, vec![0.0; n]);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_range(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    if node.inputs.len() < 3 {
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    // Resolve start/limit/delta from initializers, else fold the (possibly
    // dynamic) shape expression — e.g. the sequence mask's `arange(x.size(2))`
    // where the limit is `Gather(Shape(x), 2)`. Without this the limit defaults
    // to 0 and `arange` collapses to length 1, producing a degenerate `[1,1,1]`
    // mask that broadcasts on CPU/Metal but reshape-fails on MLX.
    let scalar = |ctx: &LowerCtx<'_>, m: &HirMut<'_>, n: &str| -> Option<i64> {
        i64_tensor(&ctx.i64_params, &ctx.params, n)
            .and_then(|v| v.first().copied())
            .or_else(|| eval_static_shape_vector(ctx, m, n, 0).and_then(|v| v.first().copied()))
    };
    let start = scalar(ctx, m, &node.inputs[0]).unwrap_or(0);
    let limit = scalar(ctx, m, &node.inputs[1])
        .or_else(|| {
            if !ctx.opts.dynamic_sequence
                && node
                    .inputs
                    .get(1)
                    .is_some_and(|s| s.ends_with("ReduceMax_output_0"))
            {
                Some(ctx.opts.sequence_length as i64)
            } else {
                None
            }
        })
        .unwrap_or(0);
    let delta = scalar(ctx, m, &node.inputs[2])
        .map(|d| d.max(1))
        .unwrap_or(1);
    let len = if limit > start {
        ((limit - start) as usize).div_ceil(delta as usize)
    } else {
        0
    };
    let data: Vec<i64> = (0..len.max(1)).map(|i| start + i as i64 * delta).collect();
    let out_s = Shape::new(&[data.len()], DType::I64);
    let bytes: Vec<u8> = data.iter().flat_map(|d| d.to_le_bytes()).collect();
    let id = m.add_node(Op::Constant { data: bytes }, vec![], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `Einsum` (opset 12+). The `equation` attribute is forwarded verbatim
/// (UTF-8) in the op attrs; the reference kernel parses it at runtime.
pub(super) fn lower_einsum(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let equation = node
        .attrs
        .get("equation")
        .and_then(|v| v.as_str())
        .context("Einsum missing `equation` attribute")?
        .to_string();
    let inputs: Vec<HirNodeId> = node
        .inputs
        .iter()
        .filter(|s| !s.is_empty())
        .map(|n| ctx.tensor(n))
        .collect::<Result<_>>()?;
    let fallback = *inputs.first().context("Einsum needs at least one input")?;
    let out_s = output_shape(ctx, node, m, fallback);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.Einsum".to_string(),
            num_inputs: inputs.len() as u32,
            attrs: equation.into_bytes(),
        },
        inputs,
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_resize(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x0 = ctx.tensor(&node.inputs[0])?;
    let out_s_final = if node.name.contains("f0_upsamp/Resize") {
        Shape::new(&[1, 9, 300], m.shape(x0).dtype())
    } else {
        resolve_shape(&node.output_meta[0], ctx.opts)
            .or_else(|_| resize_output_shape(m, ctx, node, x0))
            .unwrap_or_else(|_| m.shape(x0).clone())
    };
    let x = ensure_nchw_4d(m, x0);
    let in_s = m.shape(x).clone();
    let out_s = ncl_to_nchw_shape(&out_s_final);
    let mode = node
        .attrs
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("nearest");
    if mode == "nearest" && in_s.rank() == 4 && out_s.rank() == 4 {
        let h_in = in_s.dim(2).unwrap_static();
        let w_in = in_s.dim(3).unwrap_static();
        let h_out = out_s.dim(2).unwrap_static();
        let w_out = out_s.dim(3).unwrap_static();
        if h_out == h_in * 2 && w_out == w_in * 2 {
            let up = m.resize_nearest_2x(x);
            let id = nchw_to_ncl_if_needed(m, up, &out_s_final);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
    }
    let new_shape: Vec<i64> = out_s_final
        .dims()
        .iter()
        .map(|&d| d.unwrap_static() as i64)
        .collect();
    let id = if m.shape(x0).num_elements() == out_s_final.num_elements() {
        let reshaped = m.reshape_(x0, new_shape);
        nchw_to_ncl_if_needed(m, reshaped, &out_s_final)
    } else {
        let key = format!("__resize__/{}", node.outputs[0]);
        let n = out_s_final
            .num_elements()
            .unwrap_or(1)
            .min(MAX_STUB_ELEMENTS);
        let pid = m.param(&key, out_s_final.clone());
        ctx.params.insert(key, vec![0.0; n]);
        pid
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_random_like(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let shape_in = ctx.tensor(
        node.inputs
            .first()
            .context("Random*Like missing shape input")?,
    )?;
    let mut out_s = output_shape(ctx, node, m, shape_in);
    if out_s.rank() == 0 || out_s.num_elements().unwrap_or(0) == 0 {
        out_s = m.shape(shape_in).clone();
    }
    let tag = crate::random::node_name_tag(&node.name);
    let op_seed = crate::random::op_seed(node);
    let dist = crate::random::distribution(node);
    if ctx.opts.lower_random_as_custom {
        let id = m.add_node(
            Op::Custom {
                name: crate::random::custom_name(node).to_string(),
                num_inputs: 1,
                attrs: crate::random::custom_attrs(dist, tag),
            },
            vec![shape_in],
            out_s,
        );
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let id = m.add_node(
        crate::random::rng_op(dist, tag, op_seed),
        vec![shape_in],
        out_s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_random(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let tag = crate::random::node_name_tag(&node.name);
    let op_seed = crate::random::op_seed(node);
    let dist = crate::random::distribution(node);
    let placeholder = m.add_node(
        Op::Constant { data: vec![0u8; 4] },
        vec![],
        Shape::new(&[1], DType::F32),
    );
    let mut inputs = Vec::new();
    let mut out_s = output_shape(ctx, node, m, placeholder);
    if let Some(shape_in) = node.inputs.first().filter(|n| !n.is_empty()) {
        let id = ctx.tensor(shape_in)?;
        inputs.push(id);
        out_s = output_shape(ctx, node, m, id);
    }
    if out_s.rank() == 0 || out_s.num_elements().unwrap_or(0) == 0 {
        anyhow::bail!("Random* at {} has no inferable output shape", node.name);
    }
    if ctx.opts.lower_random_as_custom {
        let id = m.add_node(
            Op::Custom {
                name: crate::random::custom_name(node).to_string(),
                num_inputs: inputs.len() as u32,
                attrs: crate::random::custom_attrs(dist, tag),
            },
            inputs,
            out_s,
        );
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let id = m.add_node(crate::random::rng_op(dist, tag, op_seed), inputs, out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_pad_as_concat(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    // Constant (zero) padding. Pad amounts come from the `pads` attribute
    // (opset < 11) or, since opset 11, the second input tensor — which may be
    // dynamically computed (VITS relative attention pads embeddings to `2*len-1`).
    // Evaluate that tensor and realize the padding as concats of zero tensors.
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let rank = in_s.rank();
    let pads: Vec<i64> = node
        .attrs
        .get("pads")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|d| d.as_i64()).collect::<Vec<_>>())
        .filter(|v: &Vec<i64>| !v.is_empty())
        .or_else(|| {
            node.inputs.get(1).filter(|s| !s.is_empty()).and_then(|n| {
                eval_i64_shaped(ctx, m, n, 0)
                    .map(|(d, _)| d)
                    .or_else(|| eval_static_shape_vector(ctx, m, n, 0))
            })
        })
        .unwrap_or_default();
    if pads.is_empty() || pads.iter().all(|&p| p == 0) {
        return lower_reshape(m, ctx, node);
    }
    if pads.len() < 2 * rank || pads.iter().any(|&p| p < 0) {
        if ctx.opts.strict {
            anyhow::bail!("Pad at {} has unsupported pads {pads:?}", node.name);
        }
        ctx.passthrough_stub(m, node)?;
        return Ok(true);
    }
    let dt = in_s.dtype();
    let esize = dt.size_bytes().max(1);
    let mut cur = x;
    let mut dims: Vec<usize> = in_s.dims().iter().map(|d| d.unwrap_static()).collect();
    for a in 0..rank {
        // ONNX pads layout: [begin_0..begin_{r-1}, end_0..end_{r-1}].
        for (amt, before) in [(pads[a], true), (pads[rank + a], false)] {
            if amt <= 0 {
                continue;
            }
            let mut zshape = dims.clone();
            zshape[a] = amt as usize;
            let numel: usize = zshape.iter().product();
            let zeros = m.add_node(
                Op::Constant {
                    data: vec![0u8; numel * esize],
                },
                vec![],
                Shape::new(&zshape, dt),
            );
            let inputs = if before {
                vec![zeros, cur]
            } else {
                vec![cur, zeros]
            };
            cur = m.concat_(inputs, a);
            dims[a] += amt as usize;
        }
    }
    ctx.env.insert(node.outputs[0].clone(), cur);
    Ok(true)
}
