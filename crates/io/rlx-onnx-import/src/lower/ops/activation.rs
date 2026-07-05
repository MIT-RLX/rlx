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

//! `activation` — extracted from the `ops` module for navigability (see `mod.rs`).

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

pub(super) fn lower_act_copy(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let s = m.shape(x).clone();
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ActCopy".to_string(),
            num_inputs: 1,
            attrs: vec![],
        },
        vec![x],
        s,
    );
    if let Some(out) = node.outputs.first() {
        ctx.env.insert(out.clone(), id);
    }
    Ok(true)
}

pub(super) fn lower_activation(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    act: Activation,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let in_s = m.shape(x).clone();
    let mut s = output_shape(ctx, node, m, x);
    let meta_empty = node
        .output_meta
        .first()
        .and_then(|m| m.get("shape"))
        .and_then(|s| s.as_array())
        .map(|a| a.is_empty())
        .unwrap_or(true);
    if meta_empty && s.num_elements() != in_s.num_elements() {
        s = in_s.clone();
    }
    if node.name == "/Round" {
        s = in_s;
    }
    let id = m.activation(act, x, s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_activation_map(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    op: &str,
) -> Result<bool> {
    let act = match op {
        "Tanh" => Activation::Tanh,
        "Sigmoid" => Activation::Sigmoid,
        "Sqrt" => Activation::Sqrt,
        "Sin" => Activation::Sin,
        "Cos" => Activation::Cos,
        "Exp" => Activation::Exp,
        "Neg" => Activation::Neg,
        "Abs" => Activation::Abs,
        "Atan" => Activation::Atan,
        "Floor" | "Round" => Activation::Round,
        "Erf" => Activation::GeluApprox,
        _ => Activation::Relu,
    };
    lower_activation(m, ctx, node, act)
}

pub(super) fn lower_leaky_relu(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let alpha = node
        .attrs
        .get("alpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.01) as f32;
    let x = ctx.tensor(&node.inputs[0])?;
    let s = m.shape(x).clone();
    let key = format!("__leaky_alpha__/{}", node.name);
    let alpha_id = ctx.f32_scalar_param(m, &key, alpha);
    let out_s = output_shape(ctx, node, m, x);
    let pos = m.add_node(Op::Activation(Activation::Relu), vec![x], out_s.clone());
    let neg = m.neg(x);
    let nneg = m.add_node(Op::Activation(Activation::Relu), vec![neg], out_s.clone());
    let scaled = m.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![nneg, alpha_id],
        out_s.clone(),
    );
    // LeakyReLU(x) = relu(x) + alpha*min(x,0) = relu(x) - alpha*relu(-x).
    // `scaled` = alpha*relu(-x) is the magnitude of the negative branch, so it
    // must be SUBTRACTED — adding it flips the sign of every negative activation.
    let id = m.add_node(Op::Binary(BinaryOp::Sub), vec![pos, scaled], out_s);
    let _ = s;
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_identity(
    _m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    for out in &node.outputs {
        ctx.env.insert(out.clone(), x);
    }
    Ok(true)
}

pub(super) fn lower_dropout(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    lower_identity(m, ctx, node)
}
