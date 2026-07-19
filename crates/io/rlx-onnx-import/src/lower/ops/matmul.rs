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

//! `matmul` — extracted from the `ops` module for navigability (see `mod.rs`).

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

pub(super) fn lower_qmatmul(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let act_q = ctx.tensor(&node.inputs[0])?;
    let act_scale = ctx.tensor(&node.inputs[1])?;
    let act_zp = ctx.tensor(&node.inputs[2])?;
    let w = ctx.ensure_typed_param(m, &node.inputs[3])?;
    let w_scale = ctx.ensure_f32_param(m, &node.inputs[4])?;
    let w_zp = ctx.ensure_typed_param(m, &node.inputs[5])?;
    let sa = m.shape(act_q).clone();
    let sb = m.shape(w).clone();
    let s = infer_matmul_output_shape(&sa, &sb, ctx.opts.sequence_length).with_dtype(DType::F32);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.QMatMul".to_string(),
            num_inputs: 6,
            attrs: vec![],
        },
        vec![act_q, act_scale, act_zp, w, w_scale, w_zp],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_matmul(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let a = ctx.tensor(&node.inputs[0])?;
    let w_name = node.inputs[1].as_str();
    let sa = m.shape(a).clone();
    if ctx.opts.use_quantized_kernels {
        if let Some(q_key) = quant_matmul_weight_key(w_name, &ctx.quant_weight_keys) {
            let w = ctx.ensure_typed_param(m, &q_key)?;
            let sb = m.shape(w).clone();
            let s = rlx_ir::shape::matmul_shape(&sa, &sb)
                .unwrap_or_else(|_| output_shape(ctx, node, m, a));
            let base = q_key.strip_suffix("_quantized").unwrap_or(q_key.as_str());
            let scale_name = format!("{base}_scale");
            let zp_name = format!("{base}_zero_point");
            let n_out = s.dim(s.rank().saturating_sub(1)).unwrap_static().max(1);
            let k_inner = sa.dim(sa.rank().saturating_sub(1)).unwrap_static().max(1);
            let scale_key = format!("{scale_name}__dequant_broadcast_{n_out}");
            let zp_key = format!("{zp_name}__dequant_broadcast_{n_out}");
            if !ctx.params.contains_key(&scale_key) {
                let s0 = ctx
                    .params
                    .get(&scale_name)
                    .and_then(|v| v.first().copied())
                    .unwrap_or(1.0);
                let z0 = ctx
                    .params
                    .get(&zp_name)
                    .and_then(|v| v.first().copied())
                    .unwrap_or(0.0);
                ctx.params.insert(scale_key.clone(), vec![s0; n_out]);
                ctx.params.insert(zp_key.clone(), vec![z0; n_out]);
            }
            let scale = ctx.ensure_f32_param(m, &scale_key)?;
            let zp = ctx.ensure_f32_param(m, &zp_key)?;
            let scheme = QuantScheme::Int8BlockAsym {
                block_size: k_inner.max(1) as u32,
            };
            let id = m.add_node(Op::DequantMatMul { scheme }, vec![a, w, scale, zp], s);
            ctx.env.insert(node.outputs[0].clone(), id);
            return Ok(true);
        }
    }
    let b = ctx.tensor(w_name)?;
    let sb = m.shape(b).clone();
    let s = infer_matmul_output_shape(&sa, &sb, ctx.opts.sequence_length);
    // Broadcast batch dims explicitly so backends without batched-matmul broadcasting
    // (e.g. wgpu) see equal leading dims. The VITS relative-position matmul is
    // `q[1,h,t,d] × rel[1,1,d,2t-1]` — batch `[1,h]` vs `[1,1]`. CPU/Metal broadcast
    // implicitly; the explicit Expand is a no-op for them and required for wgpu.
    let (a, b) = broadcast_matmul_batch(m, a, b, &s);
    let id = m.add_node(Op::MatMul, vec![a, b], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// Transpose the last two axes of `x` (ONNX Gemm `transA`/`transB`).
fn transpose_last2(m: &mut HirMut<'_>, x: HirNodeId) -> HirNodeId {
    let s = m.shape(x).clone();
    let r = s.rank();
    if r < 2 {
        return x;
    }
    let mut perm: Vec<usize> = (0..r).collect();
    perm.swap(r - 2, r - 1);
    let dims: Vec<usize> = perm.iter().map(|&i| s.dim(i).unwrap_static()).collect();
    let out = Shape::new(&dims, s.dtype());
    m.add_node(Op::Transpose { perm }, vec![x], out)
}

/// `x * factor` (scalar), for Gemm `alpha`/`beta`.
fn scale_by(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    x: HirNodeId,
    factor: f32,
    key: String,
) -> HirNodeId {
    let sc = m.param(&key, Shape::new(&[1], DType::F32));
    ctx.params.insert(key, vec![factor]);
    let s = m.shape(x).clone();
    m.add_node(Op::Binary(BinaryOp::Mul), vec![x, sc], s)
}

pub(super) fn lower_gemm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let mut a = ctx.tensor(&node.inputs[0])?;
    let mut b = ctx.tensor(&node.inputs[1])?;
    // ONNX Gemm: Y = alpha·(op(A)·op(B)) + beta·C. A PyTorch `Linear` exports as
    // `transB=1` with weight `[out, in]`, so op(B) transposes it to `[in, out]`;
    // without this the matmul keeps the `in` feature dim (wrong output shape).
    if node
        .attrs
        .get("transA")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0
    {
        a = transpose_last2(m, a);
    }
    if node
        .attrs
        .get("transB")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0
    {
        b = transpose_last2(m, b);
    }
    let sa = m.shape(a).clone();
    let sb = m.shape(b).clone();
    let s = infer_matmul_output_shape(&sa, &sb, ctx.opts.sequence_length);
    let mut id = m.add_node(Op::MatMul, vec![a, b], s.clone());
    let alpha = node
        .attrs
        .get("alpha")
        .and_then(|v| v.as_f64())
        .map(|x| x as f32)
        .unwrap_or(1.0);
    if alpha != 1.0 {
        id = scale_by(m, ctx, id, alpha, format!("__gemm_alpha__/{}", node.name));
    }
    if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        let mut c = ctx.tensor(&node.inputs[2])?;
        let beta = node
            .attrs
            .get("beta")
            .and_then(|v| v.as_f64())
            .map(|x| x as f32)
            .unwrap_or(1.0);
        if beta != 1.0 {
            c = scale_by(m, ctx, c, beta, format!("__gemm_beta__/{}", node.name));
        }
        id = binary_infer_add(m, id, c, &node.name);
    }
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
