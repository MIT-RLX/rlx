// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

pub(super) fn lower_softmax(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
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

pub(super) fn lower_layer_norm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
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
    let gamma_c = m.shape(gamma).dim(0).unwrap_static();
    // ONNX InstanceNormalization is NCL/NCHW (channel on axis 1). Skip BLC
    // layout heuristics — group-style reshapes like `[1,32,3200]` look like BLC
    // (`L > C`) and must not be transposed.
    if m.shape(x).rank() == 4 && m.shape(x).dim(1).unwrap_static() == gamma_c {
        let n = m.shape(x).dim(0).unwrap_static();
        let c = m.shape(x).dim(1).unwrap_static();
        let l = m.shape(x).dim(2).unwrap_static() * m.shape(x).dim(3).unwrap_static();
        x = m.reshape_(x, vec![n as i64, c as i64, l as i64]);
    }
    if m.shape(x).rank() == 3 {
        let d1 = m.shape(x).dim(1).unwrap_static();
        let d2 = m.shape(x).dim(2).unwrap_static();
        if d2 == gamma_c && d1 != gamma_c {
            x = m.transpose_(x, vec![0, 2, 1]);
        }
    }
    let out_s = m.shape(x).clone();
    let rank = out_s.rank();
    if rank < 2 {
        return lower_layer_norm(m, ctx, node);
    }
    let ch_axis = if rank >= 2 && m.shape(x).dim(1).unwrap_static() == gamma_c {
        1usize
    } else {
        channel_axis_for_param(m, gamma, x)
    };
    let mut c = gamma_c;
    let c_x = m.shape(x).dim(ch_axis).unwrap_static();
    let mut gamma_u = gamma;
    let mut beta_u = beta;
    if c_x < c {
        c = c_x;
        gamma_u = m.narrow_(gamma_u, 0, 0, c);
        beta_u = m.narrow_(beta_u, 0, 0, c);
    }
    let mut broadcast: Vec<i64> = vec![1; rank];
    broadcast[ch_axis] = c as i64;
    gamma_u = m.reshape_(gamma_u, broadcast.clone());
    beta_u = m.reshape_(beta_u, broadcast);
    // KittenTTS: the mel/time axis is compiled to a padded slot (~28x the runtime
    // frames). A plain InstanceNorm reduces mean/variance over the whole padded axis,
    // so the zero padding dilutes the statistics → the F0/N AdaIN blocks over-normalize
    // and the vocoder collapses to near-silence. Route rank-3 `[N,C,T]` InstanceNorms to
    // a host-delegate kernel that reduces over the *active* mel frames only.
    if rank == 3 && ch_axis == 1 && std::env::var("RLX_KITTEN_INORM_ACTIVE").is_ok() {
        // Byte 4 flags a VOCODER-generator AdaIN. The generator runs at several upsampled rates
        // (ups.0/ups.1/resblocks/noise_res), and for short utterances some of those axis lengths
        // fall BELOW the prosody cap — so a size threshold in the kernel misclassifies them and
        // reduces over the padded extent (zero padding pulls mean→0 leaving a positive DC and
        // shrinks std, inflating the trunk ~1.5×). The node name is the only reliable signal:
        // `/decoder/generator/*` runs at the vocoder rate, everything else at the prosody rate.
        let mut attrs = eps.to_le_bytes().to_vec();
        attrs.push(u8::from(node.name.contains("/generator/")));
        let id = m.add_node(
            Op::Custom {
                name: "onnx.KittenInstanceNormActive".to_string(),
                num_inputs: 3,
                attrs,
            },
            vec![x, gamma_u, beta_u],
            out_s,
        );
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    let spatial: Vec<usize> = (0..rank).filter(|&a| a != 0 && a != ch_axis).collect();
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

/// A zero bias tensor of length `n` (RMSNorm scale-only path). Cached per key.
fn zero_beta(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, key: &str, n: usize) -> HirNodeId {
    if let Some(&id) = ctx.env.get(key) {
        return id;
    }
    let id = m.param(key, Shape::new(&[n], DType::F32));
    ctx.params.insert(key.to_string(), vec![0.0; n]);
    ctx.env.insert(key.to_string(), id);
    id
}

/// `com.microsoft::SimplifiedLayerNormalization` (and the plain ONNX
/// `SimplifiedLayerNormalization`): RMSNorm with a scale and no bias —
/// `y = x / sqrt(mean(x²) + eps) · gamma`.
pub(super) fn lower_simplified_layer_norm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let gamma = ctx.tensor(&node.inputs[1])?;
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    let n = match m.shape(gamma).dim(0) {
        rlx_ir::Dim::Static(c) => c,
        _ => m.shape(x).dim(m.shape(x).rank() - 1).unwrap_static(),
    };
    let beta = zero_beta(m, ctx, &format!("__rmsnorm_beta__/{}", node.name), n);
    let id = m.rms_norm(x, gamma, beta, eps);
    ctx.env.insert(node.outputs[0].clone(), id);
    // Optional stats outputs (mean / inv_std_dev) — zero-stub if consumed.
    for out in node.outputs.iter().skip(1).filter(|o| !o.is_empty()) {
        let key = format!("__rmsnorm_stat__/{out}");
        let sid = m.param(&key, Shape::new(&[1], DType::F32));
        ctx.params.insert(key.clone(), vec![0.0]);
        ctx.env.insert(out.clone(), sid);
    }
    Ok(true)
}

/// `com.microsoft::SkipSimplifiedLayerNormalization`: fused residual-add +
/// RMSNorm. `sum = input + skip (+ bias)`, `output = RMSNorm(sum) · gamma`.
/// Output 3 (`input_skip_bias_sum`) is the residual stream feeding the next
/// block — it MUST be bound, or the residual chain breaks.
pub(super) fn lower_skip_simplified_layer_norm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let skip = ctx.tensor(&node.inputs[1])?;
    let gamma = ctx.tensor(&node.inputs[2])?;
    let eps = node
        .attrs
        .get("epsilon")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5) as f32;
    // sum = input + skip (+ optional bias input[3], broadcast over the last dim).
    let mut sum = m.add(x, skip);
    if let Some(bias_name) = node.inputs.get(3).filter(|n| !n.is_empty()) {
        if let Ok(bias) = ctx.tensor(bias_name) {
            sum = m.add(sum, bias);
        }
    }
    let n = match m.shape(gamma).dim(0) {
        rlx_ir::Dim::Static(c) => c,
        _ => m.shape(x).dim(m.shape(x).rank() - 1).unwrap_static(),
    };
    let beta = zero_beta(m, ctx, &format!("__skiprms_beta__/{}", node.name), n);
    let normed = m.rms_norm(sum, gamma, beta, eps);
    ctx.env.insert(node.outputs[0].clone(), normed);
    // Output 3: residual sum (input + skip + bias) — the next block's skip input.
    if let Some(sum_out) = node.outputs.get(3).filter(|o| !o.is_empty()) {
        ctx.env.insert(sum_out.clone(), sum);
    }
    // Outputs 1/2 (mean / inv_std_dev) — zero-stub if consumed.
    for out in [node.outputs.get(1), node.outputs.get(2)]
        .into_iter()
        .flatten()
        .filter(|o| !o.is_empty())
    {
        let key = format!("__skiprms_stat__/{out}");
        let sid = m.param(&key, Shape::new(&[1], DType::F32));
        ctx.params.insert(key.clone(), vec![0.0]);
        ctx.env.insert(out.clone(), sid);
    }
    Ok(true)
}

pub(super) fn lower_batch_norm(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
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
    // ONNX `BatchNormalization` puts channels on axis 1 (`[N, C, …]`); the
    // per-channel γ/β/mean/var have length C. The backend `Op::BatchNormInference`
    // kernel takes channels from the LAST axis (channels-last). When the input is
    // channel-first (`[N,C,L]` NCL — e.g. supertonic's vocoder `final_norm`, whose
    // ConvNeXt residual is NCL), feeding it directly makes the kernel read the
    // length as the channel count → normalizes the wrong axis and the output
    // explodes (amax ~5e2 vs ~0.8). Move channels to the last axis, run BN, move
    // back. No-op when the input is already channel-last.
    let x_s = m.shape(x).clone();
    let rank = x_s.rank();
    let c = m.shape(gamma).dim(0).unwrap_static();
    let channels_last = rank >= 2 && x_s.dim(rank - 1).unwrap_static() == c;
    // ONNX `BatchNormalization` normalizes per-channel with channels on axis 1
    // (`[N, C, …]`). The backend `Op::BatchNormInference` kernel takes channels
    // from the LAST axis, so a channel-first input (`[N,C,L]` — supertonic's
    // vocoder `final_norm`) is normalized over the wrong axis and explodes.
    // Decompose to elementwise ops with the per-channel params reshaped to
    // broadcast along axis 1 (`[1,C,1,…]`): `y = (x-mean)/√(var+eps)·γ + β`.
    // Layout-agnostic and backend-agnostic (mirrors the `Erf`/`Softplus`
    // decompositions); only used when the fast kernel's channels-last assumption
    // does not already match the data.
    let id = if channels_last || rank < 3 {
        m.add_node(
            Op::BatchNormInference { eps },
            vec![x, gamma, beta, mean, var],
            s,
        )
    } else {
        // Reshape each `[C]` param to `[1, C, 1, …]` so it broadcasts on axis 1.
        let mut pshape = vec![1i64; rank];
        pshape[1] = c as i64;
        let g_r = m.reshape_(gamma, pshape.clone());
        let b_r = m.reshape_(beta, pshape.clone());
        let m_r = m.reshape_(mean, pshape.clone());
        let v_r = m.reshape_(var, pshape);
        let eps_id = ctx.f32_scalar_param(m, &format!("__bn_eps__/{}", node.name), eps);
        let pshape_s = {
            let mut d = vec![1usize; rank];
            d[1] = c;
            Shape::new(&d, x_s.dtype())
        };
        let ve = m.add_node(
            Op::Binary(BinaryOp::Add),
            vec![v_r, eps_id],
            pshape_s.clone(),
        );
        let std = m.add_node(Op::Activation(Activation::Sqrt), vec![ve], pshape_s.clone());
        let cen = m.add_node(Op::Binary(BinaryOp::Sub), vec![x, m_r], s.clone());
        let nrm = m.add_node(Op::Binary(BinaryOp::Div), vec![cen, std], s.clone());
        let scl = m.add_node(Op::Binary(BinaryOp::Mul), vec![nrm, g_r], s.clone());
        m.add_node(Op::Binary(BinaryOp::Add), vec![scl, b_r], s)
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
