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

//! `binary` — extracted from the `ops` module for navigability (see `mod.rs`).

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

pub(super) fn lower_binary(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let bop = match op {
        "Mul" => BinaryOp::Mul,
        "Sub" => BinaryOp::Sub,
        "Div" => BinaryOp::Div,
        "Max" => BinaryOp::Max,
        "Min" => BinaryOp::Min,
        _ => BinaryOp::Add,
    };
    let a_aligned = align_binary_operand(m, a, b);
    let b_aligned = align_binary_operand(m, b, a);
    let fix_name = node.name.as_str();
    let a_in =
        if fix_name.contains("l_sin_gen") || fix_name.contains("/decoder/generator/m_source/") {
            apply_import_shape_fix(m, ctx, fix_name, a_aligned)
        } else {
            a_aligned
        };
    let b_in =
        if fix_name.contains("l_sin_gen") || fix_name.contains("/decoder/generator/m_source/") {
            apply_import_shape_fix(m, ctx, fix_name, b_aligned)
        } else {
            b_aligned
        };
    let id = binary_infer(m, bop, a_in, b_in, &node.name);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_pow(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let s = output_shape(ctx, node, m, a);
    let id = m.add_node(Op::Binary(BinaryOp::Pow), vec![a, b], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_clip(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let min_v = node
        .attrs
        .get("min")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .or_else(|| {
            (node.inputs.len() > 1)
                .then(|| node.inputs[1].as_str())
                .and_then(|n| ctx.params.get(n).and_then(|v| v.first().copied()))
        })
        .unwrap_or(f32::NEG_INFINITY);
    let max_v = node
        .attrs
        .get("max")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .or_else(|| {
            (node.inputs.len() > 2)
                .then(|| node.inputs[2].as_str())
                .and_then(|n| ctx.params.get(n).and_then(|v| v.first().copied()))
        })
        .unwrap_or(f32::INFINITY);
    let s = m.shape(x).clone();
    let min_id = ctx.f32_scalar_param(m, &format!("__clip_min__/{}", node.name), min_v);
    let max_id = ctx.f32_scalar_param(m, &format!("__clip_max__/{}", node.name), max_v);
    let clipped_hi = m.add_node(Op::Binary(BinaryOp::Min), vec![x, max_id], s.clone());
    let id = m.add_node(Op::Binary(BinaryOp::Max), vec![clipped_hi, min_id], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_where(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let cond = ctx.tensor(&node.inputs[0])?;
    let on_t = ctx.tensor(&node.inputs[1])?;
    let on_f = ctx.tensor(&node.inputs[2])?;
    let s_t = m.shape(on_t).clone();
    let s_f = m.shape(on_f).clone();
    let s = rlx_ir::shape::binary_shape(&s_t, &s_f)
        .and_then(|ab| rlx_ir::shape::binary_shape(m.shape(cond), &ab))
        .map(|s| s.with_dtype(s_t.dtype()))
        .unwrap_or_else(|_| output_shape(ctx, node, m, on_t));
    let cond_s = s.clone().with_dtype(m.shape(cond).dtype());
    let cond_bc = expand_operand_to_shape(m, cond, &cond_s);
    let on_t_bc = expand_operand_to_shape(m, on_t, &s);
    let on_f_bc = expand_operand_to_shape(m, on_f, &s);
    let id = m.add_node(Op::Where, vec![cond_bc, on_t_bc, on_f_bc], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_compare(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    if op == "Not" && node.inputs.len() == 1 {
        let x = ctx.tensor(&node.inputs[0])?;
        let x_s = m.shape(x).clone();
        let s = x_s.clone().with_dtype(DType::Bool);
        if x_s.dtype() == DType::Bool {
            let false_id = m.add_node(
                Op::Constant { data: vec![0u8] },
                vec![],
                Shape::new(&[1], DType::Bool),
            );
            let false_bc = expand_operand_to_shape(m, false_id, &x_s);
            let id = m.add_node(Op::Compare(CmpOp::Eq), vec![x, false_bc], s);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
        let zero = ctx.f32_scalar_param(m, &format!("__not_zero__/{}", node.name), 0.0);
        let id = m.eq(x, zero);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if op == "And" && node.inputs.len() == 2 {
        let a = ctx.tensor(&node.inputs[0])?;
        let b = ctx.tensor(&node.inputs[1])?;
        let z = ctx.f32_scalar_param(m, &format!("__and_z__/{}", node.name), 0.0);
        let prod = m.mul(a, b);
        let s = output_shape(ctx, node, m, prod).with_dtype(DType::Bool);
        let id = m.add_node(Op::Compare(CmpOp::Ne), vec![prod, z], s);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if op == "Or" && node.inputs.len() == 2 {
        // a ∨ b  ≡  (a + b) ≠ 0  for boolean {0,1} operands.
        let a = ctx.tensor(&node.inputs[0])?;
        let b = ctx.tensor(&node.inputs[1])?;
        let z = ctx.f32_scalar_param(m, &format!("__or_z__/{}", node.name), 0.0);
        let sum = binary_infer_add(m, a, b, &node.name);
        let s = output_shape(ctx, node, m, sum).with_dtype(DType::Bool);
        let id = m.add_node(Op::Compare(CmpOp::Ne), vec![sum, z], s);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if node.inputs.len() < 2 {
        ctx.unsupported(op);
        return Ok(false);
    }
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let cmp = match op {
        "Less" => CmpOp::Lt,
        "Greater" => CmpOp::Gt,
        "LessOrEqual" => CmpOp::Le,
        "GreaterOrEqual" => CmpOp::Ge,
        _ => CmpOp::Eq,
    };
    let sa = m.shape(a).clone();
    let sb = m.shape(b).clone();
    let s = rlx_ir::shape::binary_shape(&sa, &sb)
        .map(|sh| sh.with_dtype(DType::Bool))
        .unwrap_or_else(|_| output_shape(ctx, node, m, a).with_dtype(DType::Bool));
    let a_in = expand_operand_to_shape(m, a, &s);
    let b_in = expand_operand_to_shape(m, b, &s);
    let id = m.add_node(Op::Compare(cmp), vec![a_in, b_in], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_mod(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let b = ctx.tensor(&node.inputs[1])?;
    let fmod = node.attrs.get("fmod").and_then(|v| v.as_i64()).unwrap_or(0);
    let s = output_shape(ctx, node, m, a);
    let attrs = fmod.to_le_bytes().to_vec();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.Mod".to_string(),
            num_inputs: 2,
            attrs,
        },
        vec![a, b],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_is_nan(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let s = output_shape(ctx, node, m, x).with_dtype(DType::Bool);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.IsNaN".to_string(),
            num_inputs: 1,
            attrs: vec![],
        },
        vec![x],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
