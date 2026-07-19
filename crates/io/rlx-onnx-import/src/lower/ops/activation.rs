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
    // A unary elementwise op preserves shape EXACTLY. When the input shape is
    // fully static it is authoritative — the ONNX-declared `output_meta` may be
    // stale (e.g. a graph-split extraction inherits the monolithic graph's
    // symbolic dims, which resolved to different concrete values), which silently
    // reshaped Kokoro's AdaIN `Sqrt` from [1,514,1] variance to a bogus
    // [1,512,74]. Only consult `output_meta` when the input is under-inferred.
    let mut s = if in_s.is_static() {
        in_s.clone()
    } else {
        output_shape(ctx, node, m, x)
    };
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
        s = in_s.clone();
    }
    // `Abs`/`Neg` on a SIGNED-integer tensor: rlx's `Activation` kernels are
    // f32-only and misread the integer buffer — T5's relative-position
    // `abs(key - query)` on i64 returned i64::MAX for every negative value, so
    // every "key before query" pair collapsed to the same position-bias bucket
    // (constant bias → wrong bidirectional attention, encoder cosine ~0.75).
    // Relative positions are tiny, so route through f32 (exact for |x| < 2^24);
    // every backend already supports Cast + f32 Abs/Neg.
    let dt = in_s.dtype();
    if matches!(act, Activation::Abs | Activation::Neg)
        && matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    {
        let sf = s.clone().with_dtype(DType::F32);
        let xf = m.add_node(Op::Cast { to: DType::F32 }, vec![x], sf.clone());
        let af = m.add_node(Op::Activation(act), vec![xf], sf);
        let back = m.add_node(Op::Cast { to: dt }, vec![af], s.clone());
        ctx.env.insert(node.outputs[0].clone(), back);
        return Ok(true);
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
        "Log" => Activation::Log,
        "Neg" => Activation::Neg,
        "Abs" => Activation::Abs,
        "Atan" => Activation::Atan,
        "Floor" | "Round" => Activation::Round,
        "Erf" => Activation::GeluApprox,
        _ => Activation::Relu,
    };
    lower_activation(m, ctx, node, act)
}

/// ONNX `Sign(x)` → −1 / 0 / +1. Zipformer relative-position embedding (LuxTTS /
/// ZipVoice `encoder_pos`) uses this in the `If` else branch when `2T−1 > max_len`
/// (1999); without it the branch stubs and long CFM attention shapes collapse.
pub(super) fn lower_sign(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let out_s = unary_out_shape(ctx, node, m, x);
    let dt = out_s.dtype();
    let name = &node.name;
    // Compare / arithmetic kernels are f32-clean; integers take the same cast
    // round-trip as Abs/Neg (relative-position indices are tiny).
    let (xf, sf) = if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
        let sf = out_s.clone().with_dtype(DType::F32);
        (
            m.add_node(Op::Cast { to: DType::F32 }, vec![x], sf.clone()),
            sf,
        )
    } else {
        (x, out_s.clone())
    };
    let zero = ctx.f32_scalar_param(m, &format!("__sign_zero__/{name}"), 0.0);
    let zero_bc = expand_operand_to_shape(m, zero, &sf);
    let gt_b = m.add_node(
        Op::Compare(CmpOp::Gt),
        vec![xf, zero_bc],
        sf.clone().with_dtype(DType::Bool),
    );
    let lt_b = m.add_node(
        Op::Compare(CmpOp::Lt),
        vec![xf, zero_bc],
        sf.clone().with_dtype(DType::Bool),
    );
    let gt = m.add_node(Op::Cast { to: DType::F32 }, vec![gt_b], sf.clone());
    let lt = m.add_node(Op::Cast { to: DType::F32 }, vec![lt_b], sf.clone());
    // sign = 1·[x>0] − 1·[x<0]
    let signed = m.add_node(Op::Binary(BinaryOp::Sub), vec![gt, lt], sf);
    let id = if matches!(dt, DType::I8 | DType::I16 | DType::I32 | DType::I64) {
        m.add_node(Op::Cast { to: dt }, vec![signed], out_s)
    } else {
        signed
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `Erf(x)` — the true error function, NOT a GELU. Models that export GELU
/// in *decomposed* form (`x·0.5·(1 + erf(x/√2))` as discrete Div/Erf/Add/Mul/Mul
/// nodes — supertonic/ConvNeXt, many DiT decoders) rely on a real `erf` here;
/// aliasing `Erf → GeluApprox` computes a GELU-of-a-GELU and corrupts the whole
/// activation. Composed from existing ops (Abs/Mul/Add/Div/Exp/Neg/Sub + a Where
/// for the sign) via Abramowitz-Stegun 7.1.26 (max abs error ≈ 1.5e-7), so every
/// backend gets it for free with no new activation variant.
pub(super) fn lower_erf(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let s = unary_out_shape(ctx, node, m, x);
    let name = &node.name;
    let sc = |m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, tag: &str, v: f32| {
        ctx.f32_scalar_param(m, &format!("__erf_{tag}__/{name}"), v)
    };
    let one = sc(m, ctx, "one", 1.0);
    let p = sc(m, ctx, "p", 0.327_591_1);
    // t = 1 / (1 + p·|x|)
    let ax = m.add_node(Op::Activation(Activation::Abs), vec![x], s.clone());
    let pax = m.add_node(Op::Binary(BinaryOp::Mul), vec![ax, p], s.clone());
    let denom = m.add_node(Op::Binary(BinaryOp::Add), vec![pax, one], s.clone());
    let t = m.add_node(Op::Binary(BinaryOp::Div), vec![one, denom], s.clone());
    // poly = t·(a1 + t·(a2 + t·(a3 + t·(a4 + t·a5))))  (Horner)
    let mut acc = sc(m, ctx, "a5", 1.061_405_4);
    for (i, coef) in [-1.453_152_1_f32, 1.421_413_8, -0.284_496_72, 0.254_829_6]
        .into_iter()
        .enumerate()
    {
        let tm = m.add_node(Op::Binary(BinaryOp::Mul), vec![t, acc], s.clone());
        let c = sc(m, ctx, &format!("a{i}"), coef);
        acc = m.add_node(Op::Binary(BinaryOp::Add), vec![tm, c], s.clone());
    }
    let poly = m.add_node(Op::Binary(BinaryOp::Mul), vec![t, acc], s.clone());
    // y = 1 - poly·exp(-x²)   (≈ erf(|x|))
    let x2 = m.add_node(Op::Binary(BinaryOp::Mul), vec![x, x], s.clone());
    let negx2 = m.add_node(Op::Activation(Activation::Neg), vec![x2], s.clone());
    let ex = m.add_node(Op::Activation(Activation::Exp), vec![negx2], s.clone());
    let pe = m.add_node(Op::Binary(BinaryOp::Mul), vec![poly, ex], s.clone());
    let y = m.add_node(Op::Binary(BinaryOp::Sub), vec![one, pe], s.clone());
    // erf is odd: erf(x) = sign(x)·y (y ≈ erf(|x|) ≥ 0). Branchless sign
    // `x/√(x²+τ)` (≈ ±1; 0 at x=0, where y=0 too) — avoids a Compare/Where on a
    // broadcast scalar, which was unreliable and silently collapsed to one branch.
    let tiny = sc(m, ctx, "tiny", 1e-20);
    let x2t = m.add_node(Op::Binary(BinaryOp::Add), vec![x2, tiny], s.clone());
    let inv = m.add_node(Op::Activation(Activation::Rsqrt), vec![x2t], s.clone());
    let sign = m.add_node(Op::Binary(BinaryOp::Mul), vec![x, inv], s.clone());
    let erf = m.add_node(Op::Binary(BinaryOp::Mul), vec![sign, y], s);
    ctx.env.insert(node.outputs[0].clone(), erf);
    Ok(true)
}

/// ONNX `Softplus(x) = ln(1 + e^x)`. Composed from `Exp`/`Log` (both native
/// activations) + a scalar `+1`. Used by Mish (`x·tanh(softplus(x))`) in the F5
/// conv positional embedding, exported as discrete Softplus/Tanh/Mul nodes.
pub(super) fn lower_softplus(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let out_s = unary_out_shape(ctx, node, m, x);
    let e = m.add_node(Op::Activation(Activation::Exp), vec![x], out_s.clone());
    let one = ctx.f32_scalar_param(m, &format!("__softplus_one__/{}", node.name), 1.0);
    let e1 = m.add_node(Op::Binary(BinaryOp::Add), vec![e, one], out_s.clone());
    let id = m.add_node(Op::Activation(Activation::Log), vec![e1], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `Elu(x) = x` for `x ≥ 0` else `alpha·(exp(x) − 1)`. Branchless, composed
/// from existing ops (Relu/Neg/Exp + scalar Sub/Mul/Add) so every backend gets it
/// for free with no new activation variant:
///   `Elu(x) = relu(x) + alpha·(exp(min(x,0)) − 1)`, where `min(x,0) = −relu(−x)`.
/// Used by the ChatterBox S3Gen F0-predictor condnet (was unsupported → stubbed to
/// zeros, collapsing the neural-source-filter excitation).
pub(super) fn lower_elu(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let alpha = node
        .attrs
        .get("alpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let s = unary_out_shape(ctx, node, m, x);
    let name = &node.name;
    let pos = m.add_node(Op::Activation(Activation::Relu), vec![x], s.clone());
    // min(x,0) = −relu(−x)
    let negx = m.add_node(Op::Activation(Activation::Neg), vec![x], s.clone());
    let relu_negx = m.add_node(Op::Activation(Activation::Relu), vec![negx], s.clone());
    let min_x0 = m.add_node(Op::Activation(Activation::Neg), vec![relu_negx], s.clone());
    let ex = m.add_node(Op::Activation(Activation::Exp), vec![min_x0], s.clone());
    let one = ctx.f32_scalar_param(m, &format!("__elu_one__/{name}"), 1.0);
    let exm1 = m.add_node(Op::Binary(BinaryOp::Sub), vec![ex, one], s.clone());
    let a = ctx.f32_scalar_param(m, &format!("__elu_alpha__/{name}"), alpha);
    let scaled = m.add_node(Op::Binary(BinaryOp::Mul), vec![exm1, a], s.clone());
    let id = m.add_node(Op::Binary(BinaryOp::Add), vec![pos, scaled], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `Ceil(x)` = `round(x) + (round(x) < x)`, and `Floor(x)` = `round(x) -
/// (round(x) > x)`. Exact for any round-half mode (rlx has no native ceil/floor;
/// `Round` alone is wrong for these — a length `ceil` off by 1 changes the whole
/// output shape). `up=true` → ceil, `up=false` → floor.
pub(super) fn lower_round_dir(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
    up: bool,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let out_s = unary_out_shape(ctx, node, m, x);
    let r = m.add_node(Op::Activation(Activation::Round), vec![x], out_s.clone());
    let cmp = if up { CmpOp::Lt } else { CmpOp::Gt };
    // `Compare` yields a *bool* (1-byte) tensor. Feeding it straight into a
    // float `Add`/`Sub` reads those bytes as f32 — a `0x01` becomes a denormal
    // ~1e-45, so the ±1 correction silently vanishes and `Floor`/`Ceil` collapse
    // to bare `Round` whenever rounding crosses an integer (e.g. `Floor(0.6)→1`).
    // Cast the mask to f32 first so the correction is a real `1.0`.
    let mask_b = m.add_node(
        Op::Compare(cmp),
        vec![r, x],
        out_s.clone().with_dtype(DType::Bool),
    );
    let mask = m.add_node(Op::Cast { to: DType::F32 }, vec![mask_b], out_s.clone());
    let op = if up { BinaryOp::Add } else { BinaryOp::Sub };
    let id = m.add_node(Op::Binary(op), vec![r, mask], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `Reciprocal(x) = 1/x` — no native op; lower as `1 / x` (scalar numerator
/// broadcasts). Used e.g. for the CFM timestep `1/total_step`.
pub(super) fn lower_reciprocal(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let out_s = unary_out_shape(ctx, node, m, x);
    let one = ctx.f32_scalar_param(m, &format!("__recip_one__/{}", node.name), 1.0);
    let id = m.add_node(Op::Binary(BinaryOp::Div), vec![one, x], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// ONNX `PRelu(x, slope) = x>0 ? x : slope*x` = `relu(x) - slope*relu(-x)`, with
/// a per-channel `slope` tensor (unlike LeakyReLU's scalar). `binary_infer`
/// broadcasts `slope` (`[C]` or `[1,C,1]`) across the activation.
pub(super) fn lower_prelu(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(&node.inputs[0])?;
    let slope = ctx.tensor(&node.inputs[1])?;
    let out_s = unary_out_shape(ctx, node, m, x);
    let pos = m.add_node(Op::Activation(Activation::Relu), vec![x], out_s.clone());
    let neg = m.neg(x);
    let nneg = m.add_node(Op::Activation(Activation::Relu), vec![neg], out_s.clone());
    let scaled = super::binary_infer(m, BinaryOp::Mul, nneg, slope, &node.name);
    let id = m.add_node(Op::Binary(BinaryOp::Sub), vec![pos, scaled], out_s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
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
    // LeakyReLU is strictly elementwise — its output shape IS the input shape.
    // Using the ONNX `output_meta` (via `output_shape`) is wrong when that meta
    // carries a symbolic length that shape-propagation baked to a default (the
    // ChatterBox S3Gen encoder's pre-lookahead conv: input length 32, but the
    // declared LeakyRelu meta resolved to 128 → the whole encoder length cascade
    // diverged). Preserve the actual input shape.
    let in_s = m.shape(x).clone();
    let key = format!("__leaky_alpha__/{}", node.name);
    let alpha_id = ctx.f32_scalar_param(m, &key, alpha);
    // Input shape is authoritative when static (matches `lower_activation`); only
    // fall back to the ONNX `output_meta` when the input is under-inferred.
    let out_s = if in_s.is_static() {
        in_s
    } else {
        output_shape(ctx, node, m, x)
    };
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
