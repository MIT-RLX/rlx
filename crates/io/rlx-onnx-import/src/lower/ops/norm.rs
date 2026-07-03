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

//! `norm` — extracted from the `ops` module for navigability (see `mod.rs`).

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

pub(super) fn lower_softmax(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let axis = node
        .attrs
        .get("axis")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let id = m.sm(x, axis);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}


pub(super) fn lower_layer_norm(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let mut x = ctx.tensor(&node.inputs[0])?;
    let meta_s = output_shape(ctx, node, m, x);
    if m.shape(x).rank() == 3 && is_blc_rank3(&meta_s) {
        let (x_t, _) = ncl_channel_axis1_to_blc(m, x, &meta_s);
        x = x_t;
    }
    let s = m.shape(x).clone();
    let mut gamma = ctx.tensor(&node.inputs[1])?;
    let mut beta = ctx.tensor(&node.inputs[2])?;
    let axis = node
        .attrs
        .get("axis")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1) as i32;
    let rank = m.shape(x).rank();
    if rank >= 2 && m.shape(gamma).rank() == 1 {
        let c = m.shape(gamma).dim(0).unwrap_static();
        let mut broadcast: Vec<i64> = vec![1; rank];
        let ax = if axis < 0 {
            (rank as i32 + axis) as usize
        } else {
            axis as usize
        };
        if ax < rank {
            broadcast[ax] = c as i64;
        }
        gamma = m.reshape_(gamma, broadcast.clone());
        beta = m.reshape_(beta, broadcast);
    }
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let id = m.add_node(Op::LayerNorm { axis, eps }, vec![x, gamma, beta], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}


pub(super) fn lower_instance_norm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let mut x = ctx.tensor(&node.inputs[0])?;
    let gamma = ctx.tensor(&node.inputs[1])?;
    let beta = ctx.tensor(&node.inputs[2])?;
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let meta_s = output_shape(ctx, node, m, x);
    let gamma_c = m.shape(gamma).dim(0).unwrap_static();
    if m.shape(x).rank() == 4 && m.shape(x).dim(1).unwrap_static() == gamma_c {
        let n = m.shape(x).dim(0).unwrap_static();
        let c = m.shape(x).dim(1).unwrap_static();
        let l = m.shape(x).dim(2).unwrap_static();
        x = m.reshape_(x, vec![n as i64, c as i64, l as i64]);
    }
    if m.shape(x).rank() == 3 {
        let xs = m.shape(x).clone();
        let d1 = xs.dim(1).unwrap_static();
        let d2 = xs.dim(2).unwrap_static();
        if d2 == gamma_c && d1 != gamma_c && is_typical_channel(d2) {
            x = m.transpose_(x, vec![0, 2, 1]);
        }
    }
    if m.shape(x).rank() == 3 && is_blc_rank3(&meta_s) {
        let (x_t, _) = ncl_channel_axis1_to_blc(m, x, &meta_s);
        x = x_t;
    }
    let out_s = m.shape(x).clone();
    let in_s = out_s.clone();
    let rank = in_s.rank();
    if rank < 2 {
        return lower_layer_norm(m, ctx, node);
    }
    let mut gamma_u = gamma;
    let mut beta_u = beta;
    if m.shape(gamma).rank() == 1 && rank >= 2 {
        let mut c = m.shape(gamma).dim(0).unwrap_static();
        let ch_axis = if rank == 4 && m.shape(x).dim(1).unwrap_static() >= 64 {
            1usize
        } else if rank == 4 && m.shape(x).dim(3).unwrap_static() <= c {
            3usize
        } else if meta_layout_ncl(&meta_s) {
            1usize
        } else {
            channel_axis_for_param(m, gamma, x)
        };
        let c_x = m.shape(x).dim(ch_axis).unwrap_static();
        if c_x < c {
            c = c_x;
            gamma_u = m.narrow_(gamma_u, 0, 0, c);
            beta_u = m.narrow_(beta_u, 0, 0, c);
        }
        let mut broadcast: Vec<i64> = vec![1; rank];
        broadcast[ch_axis] = c as i64;
        gamma_u = m.reshape_(gamma_u, broadcast.clone());
        beta_u = m.reshape_(beta_u, broadcast);
    }
    let spatial: Vec<usize> = (2..rank).collect();
    let mean = m.mean(x, spatial.clone(), true);
    let centered = m.sub(x, mean);
    let sq = m.mul(centered, centered);
    let var = m.mean(sq, spatial, true);
    let eps_id = ctx.f32_scalar_param(m, &format!("__in_eps__/{}", node.name), eps);
    let var_eps = m.add(var, eps_id);
    let std = m.sqrt(var_eps);
    let norm = m.div(centered, std);
    let scaled = m.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![norm, gamma_u],
        out_s.clone(),
    );
    let id = m.add_node(Op::Binary(BinaryOp::Add), vec![scaled, beta_u], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}


pub(super) fn lower_batch_norm(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, node: &BundleNode) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let gamma = ctx.tensor(&node.inputs[1])?;
    let beta = ctx.tensor(&node.inputs[2])?;
    let mean = ctx.tensor(&node.inputs[3])?;
    let var = ctx.tensor(&node.inputs[4])?;
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let s = output_shape(ctx, node, m, x);
    let id = m.add_node(
        Op::BatchNormInference { eps },
        vec![x, gamma, beta, mean, var],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

