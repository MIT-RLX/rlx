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

pub(super) fn lower_cast(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let to = node.attrs.get("to").and_then(|v| v.as_i64()).unwrap_or(1);
    let dtype = match to {
        1 => DType::F32,
        7 => DType::I64,
        6 => DType::I32,
        9 => DType::Bool,
        _ => DType::F32,
    };
    let in_s = m.shape(x).clone();
    let in_dims = in_s.dims().to_vec();
    let mut out_s = output_shape(ctx, node, m, x).with_dtype(dtype);
    if node.outputs.iter().any(|o| o == "waveform") {
        let cap = ctx.opts.max_waveform_samples;
        let n = in_s.num_elements().unwrap_or(1).min(cap);
        out_s = Shape::new(&[n], dtype);
    } else if node.outputs.iter().any(|o| o == "duration") {
        let n = in_s.num_elements().unwrap_or(1);
        out_s = Shape::new(&[n.max(1)], dtype);
    } else if !node
        .outputs
        .iter()
        .any(|o| o == "waveform" || o == "duration")
        && out_s.num_elements() != in_s.num_elements()
    {
        out_s = in_s.clone().with_dtype(dtype);
    }
    let needs_reshape = out_s.dims() != in_dims.as_slice()
        && !node
            .outputs
            .iter()
            .any(|o| o.contains("Transpose_2_output_0"));
    let cast_s = in_s.with_dtype(dtype);
    let cast_id = m.add_node(Op::Cast { to: dtype }, vec![x], cast_s);
    let id = if needs_reshape {
        let new_shape: Vec<i64> = out_s
            .dims()
            .iter()
            .map(|d| d.unwrap_static() as i64)
            .collect();
        m.reshape_(cast_id, new_shape)
    } else {
        cast_id
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}


pub(super) fn lower_dynamic_quant(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let feeds_qmatmul = node.outputs.first().is_some_and(|q| {
        ctx.nodes
            .iter()
            .any(|n| n.op == "QMatMul" && n.inputs.first().is_some_and(|i| i == q))
    });
    if !feeds_qmatmul {
        if !node.outputs.is_empty() {
            ctx.env.insert(node.outputs[0].clone(), x);
        }
        if node.outputs.len() > 1 {
            let scale_id =
                ctx.f32_scalar_param(m, &format!("__dql_scale__/{}", node.outputs[1]), 1.0);
            ctx.env.insert(node.outputs[1].clone(), scale_id);
        }
        if node.outputs.len() > 2 {
            let zp_id = ctx.f32_scalar_param(m, &format!("__dql_zp__/{}", node.outputs[2]), 0.0);
            ctx.env.insert(node.outputs[2].clone(), zp_id);
        }
        return Ok(true);
    }
    for (i, out_name) in node.outputs.iter().enumerate() {
        let meta = node.output_meta.get(i).or_else(|| node.output_meta.first());
        let mut shape = meta
            .map(|m| resolve_shape(m, ctx.opts))
            .transpose()?
            .unwrap_or_else(|| m.shape(x).clone());
        shape = match i {
            0 => shape.with_dtype(DType::U8),
            1 => Shape::new(&[], DType::F32),
            2 => Shape::new(&[], DType::U8),
            _ => shape,
        };
        let id = m.add_node(
            Op::Custom {
                name: "onnx.DynamicQuantizeLinearExport".to_string(),
                num_inputs: 1,
                attrs: vec![i as u8],
            },
            vec![x],
            shape,
        );
        ctx.env.insert(out_name.clone(), id);
    }
    Ok(true)
}


pub(super) fn lower_dynamic_quantize_lstm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let mut inputs = Vec::new();
    for name in &node.inputs {
        if name.is_empty() {
            continue;
        }
        inputs.push(ctx.tensor(name)?);
    }
    let hidden_size = node
        .attrs
        .get("hidden_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(256) as usize;
    let bidirectional = node
        .attrs
        .get("direction")
        .and_then(|v| v.as_str())
        .map(|s| s == "bidirectional")
        .unwrap_or(true);
    let attrs = lstm_attrs_bytes(hidden_size, bidirectional);
    let mut x = inputs[0];
    let xs = m.shape(x).clone();
    if is_ncl_rank3(&xs) {
        x = m.transpose_(x, vec![2, 0, 1]);
        inputs[0] = x;
    }
    let out_shape = lstm_y_shape(m.shape(x), hidden_size, bidirectional);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.DynamicQuantizeLSTM".to_string(),
            num_inputs: inputs.len() as u32,
            attrs,
        },
        inputs,
        out_shape,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    for (i, out_name) in node.outputs.iter().enumerate().skip(1) {
        let meta = node.output_meta.get(i).or_else(|| node.output_meta.first());
        if let Some(meta) = meta {
            if let Ok(shape) = resolve_shape(meta, ctx.opts) {
                let key = format!("__lstm_extra__/{}", out_name);
                let n = shape.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
                let pid = m.param(&key, shape);
                ctx.params.insert(key, vec![0.0; n]);
                ctx.env.insert(out_name.clone(), pid);
            }
        }
    }
    Ok(true)
}

