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
    // Element-wise: the output shape is the broadcast of the ACTUAL lowered input
    // shapes — never the statically-propagated `output_meta`. In kittentts's
    // generator the STFT harmonic source's framewise reshapes fool the shape
    // propagation into `[168,11,57]`, but the real magnitude tensor is the conv
    // input's `[1,11,10081]`; trusting meta corrupts every downstream op (→ the
    // `Add_3` broadcast panic). Fall back to meta only if broadcast is undefined.
    let sa = m.shape(a).clone();
    let sb = m.shape(b).clone();
    let s = rlx_ir::shape::binary_shape(&sa, &sb).unwrap_or_else(|_| output_shape(ctx, node, m, a));
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
    // Clip is `Max(Min(x, hi), lo)` with f32 scalar bounds. For an INTEGER input
    // (MOSS clamps the i64 sampled-position count to `[0, k-1]`) `Min(i64, f32)`
    // is a mixed-dtype binary that reads the i64 as f32 garbage — clip in f32 and
    // cast back (counts/indices are small, exact in f32).
    let in_dt = m.shape(x).dtype();
    let int_clip = matches!(in_dt, DType::I64 | DType::I32);
    let x = if int_clip {
        let fs = m.shape(x).clone().with_dtype(DType::F32);
        m.add_node(Op::Cast { to: DType::F32 }, vec![x], fs)
    } else {
        x
    };
    let s = m.shape(x).clone();
    let min_id = ctx.f32_scalar_param(m, &format!("__clip_min__/{}", node.name), min_v);
    let max_id = ctx.f32_scalar_param(m, &format!("__clip_max__/{}", node.name), max_v);
    let clipped_hi = m.add_node(Op::Binary(BinaryOp::Min), vec![x, max_id], s.clone());
    let id = m.add_node(Op::Binary(BinaryOp::Max), vec![clipped_hi, min_id], s);
    let id = if int_clip {
        let is = m.shape(id).clone().with_dtype(in_dt);
        m.add_node(Op::Cast { to: in_dt }, vec![id], is)
    } else {
        id
    };
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

pub(super) fn lower_where(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let cond0 = ctx.tensor(&node.inputs[0])?;
    let on_t = ctx.tensor(&node.inputs[1])?;
    let on_f = ctx.tensor(&node.inputs[2])?;
    let s_t = m.shape(on_t).clone();
    let s_f = m.shape(on_f).clone();
    // ONNX `Where` output = multidirectional broadcast of (cond, x, y). Broadcast
    // the value operands first; a mask (`cond`) that fails to broadcast must NOT
    // shrink the result below the value shape — falling back to `on_t` alone
    // collapsed a `[4,1,15,15]` masked-attention Where to the scalar `Constant`
    // on-true value `[1]` (Zipformer padding mask mis-sized vs attention weights).
    let val_bc = rlx_ir::shape::binary_shape(&s_t, &s_f);
    let s = match val_bc {
        Ok(ab) => rlx_ir::shape::binary_shape(m.shape(cond0), &ab)
            .unwrap_or(ab)
            .with_dtype(s_t.dtype()),
        Err(_) => output_shape(ctx, node, m, on_t),
    };
    // Cast Bool/I32/I64 masks to F32 *before* Expand/Where. Mid-dim Expand of a
    // 1-byte Bool (and ElementwiseRegion, which always loads f32) mis-strides the
    // condition into an alternating T/F pattern — MioTTS prenet windowed attention
    // `Where(Expand(trilu), 0, -inf)` became `[0,-inf,0,-inf,…]` (MOSS causal-mask
    // And/Or hit the same class of bug; see `expand_bool_pair_to_f32`).
    let cond = bool_operand_to_f32(m, cond0);
    let cond_s = s.clone().with_dtype(DType::F32);
    let cond_bc = expand_operand_to_shape(m, cond, &cond_s);
    let on_t_bc = expand_operand_to_shape(m, on_t, &s);
    let on_f_bc = expand_operand_to_shape(m, on_f, &s);
    let id = m.add_node(Op::Where, vec![cond_bc, on_t_bc, on_f_bc], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}

/// Cast a Bool operand to F32 {0.0,1.0} so the elementwise arithmetic that
/// implements And/Or/Xor (`a·b`, `a+b`) runs on a real f32 buffer instead of
/// misreading the 1-byte Bool as 4-byte f32. No-op for already-numeric operands.
/// Cast a logical operand to f32. Logical ops (And/Or/Xor) always have {0,1}
/// operands, but the importer stores ONNX **bool** constants as I64 (bool type 9
/// is folded into `i64_params`, see `onnx_file`), so an operand can arrive as
/// Bool *or* I64/I32. All are cast to f32 so the downstream arithmetic reads the
/// right element width — leaving a 1-byte Bool (or an I64 whose data is only
/// 1 byte/elem) to be misread by an f32/Expand kernel is what produced MOSS's
/// broken causal mask.
fn bool_operand_to_f32(m: &mut HirMut<'_>, id: HirNodeId) -> HirNodeId {
    let dt = m.shape(id).dtype();
    if matches!(dt, DType::Bool | DType::I64 | DType::I32) {
        let s = m.shape(id).clone().with_dtype(DType::F32);
        m.add_node(Op::Cast { to: DType::F32 }, vec![id], s)
    } else {
        id
    }
}

/// Expand two boolean operands to their common broadcast shape, THEN cast to
/// f32. Logical ops (And/Or/Xor) are lowered to arithmetic on f32 (`a·b`,
/// `a+b`); if we leave the operands at their original (broadcastable) shapes
/// and rely on the arithmetic kernel to broadcast, we hit a mid-dim broadcast
/// bug (e.g. `[1,1,17,17] · [1,1,1,17]` produced an *alternating* result on the
/// CPU BinaryFull path → MOSS local-decoder's causal mask `And(causal, all-true)`
/// came out `[1,0,1,0,…]` instead of triangular). Expanding first (via the
/// reliable `Op::Expand`, exactly as the general `Compare` path does) makes the
/// arithmetic strictly element-wise, sidestepping the kernel broadcast entirely.
fn expand_bool_pair_to_f32(
    m: &mut HirMut<'_>,
    a0: HirNodeId,
    b0: HirNodeId,
) -> (HirNodeId, HirNodeId, Shape) {
    let sa = m.shape(a0).clone();
    let sb = m.shape(b0).clone();
    let bshape = rlx_ir::shape::binary_shape(&sa, &sb).unwrap_or_else(|_| sa.clone());
    // Cast Bool → f32 FIRST (contiguous, exact), THEN Expand the f32. Expanding a
    // 1-byte Bool tensor across a middle dim mis-strides on the broadcast path.
    let a_f = bool_operand_to_f32(m, a0);
    let b_f = bool_operand_to_f32(m, b0);
    let f32_shape = bshape.clone().with_dtype(DType::F32);
    let a = expand_operand_to_shape(m, a_f, &f32_shape);
    let b = expand_operand_to_shape(m, b_f, &f32_shape);
    (a, b, bshape)
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
        // The CPU `Compare` thunk has no broadcast — expand the scalar to `x`'s
        // shape so it isn't read out-of-bounds (see `expand_bool_pair_to_f32`).
        let zero = expand_operand_to_shape(m, zero, &x_s.clone().with_dtype(DType::F32));
        let id = m.eq(x, zero);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if op == "And" && node.inputs.len() == 2 {
        // a ∧ b ≡ (a·b) ≠ 0. Cast Bool → F32 FIRST: the arithmetic kernel reads a
        // 1-byte Bool operand as 4-byte f32 (denormal garbage), so `Bool·Bool` on
        // raw bools collapses to ~0 → all-False. (MOSS local-decoder's causal
        // attention mask `And(causal, all-true)` zeroed out → attention attends to
        // nothing → nan softmax.)
        // a ∧ b ≡ (a·b) as Bool for boolean {0,1} operands. The product is
        // already {0,1}; cast straight back to Bool (NO scalar `Compare` — that
        // thunk has no broadcast and reads a scalar rhs out of bounds).
        let (a, b, bshape) = expand_bool_pair_to_f32(
            m,
            ctx.tensor(&node.inputs[0])?,
            ctx.tensor(&node.inputs[1])?,
        );
        let prod = m.mul(a, b);
        let s = bshape.with_dtype(DType::Bool);
        let id = m.add_node(Op::Cast { to: DType::Bool }, vec![prod], s);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if op == "Or" && node.inputs.len() == 2 {
        // a ∨ b ≡ (a + b) as Bool for boolean {0,1} operands ({0,1,2} ≠ 0 ⇔ or).
        let (a, b, bshape) = expand_bool_pair_to_f32(
            m,
            ctx.tensor(&node.inputs[0])?,
            ctx.tensor(&node.inputs[1])?,
        );
        let sum = binary_infer_add(m, a, b, &node.name);
        let s = bshape.with_dtype(DType::Bool);
        let id = m.add_node(Op::Cast { to: DType::Bool }, vec![sum], s);
        ctx.env.insert(node.outputs[0].clone(), id);
        return Ok(true);
    }
    if op == "Xor" && node.inputs.len() == 2 {
        // a ⊕ b ≡ (a ≠ b) for boolean operands. Cast both to f32 {0,1} (handles
        // Bool *and* I64-stored bool) and compare — equal-shape, no scalar.
        let (a, b, bshape) = expand_bool_pair_to_f32(
            m,
            ctx.tensor(&node.inputs[0])?,
            ctx.tensor(&node.inputs[1])?,
        );
        let s = bshape.with_dtype(DType::Bool);
        let id = m.add_node(Op::Compare(CmpOp::Ne), vec![a, b], s);
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
    // Broadcast *dims* only — keep each operand's native dtype. Expanding I64
    // ranges into a Bool-typed `[1,1,S,S]` (old behaviour) left the Expand
    // output declared as 1-byte Bool while the data was still 8-byte I64;
    // CPU Compare then took the `elem==1` path and the causal mask became
    // all-false → Softmax NaN. Metal/MLX survived via widen+CastHost.
    let dims =
        rlx_ir::shape::binary_shape(&sa, &sb).unwrap_or_else(|_| output_shape(ctx, node, m, a));
    let a_in = expand_operand_to_shape(m, a, &dims.clone().with_dtype(sa.dtype()));
    let b_in = expand_operand_to_shape(m, b, &dims.clone().with_dtype(sb.dtype()));
    let s = dims.with_dtype(DType::Bool);
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
    // IEEE: `x != x` is true iff `x` is NaN. Prefer a native `Compare` over
    // `Op::Custom("onnx.IsNaN")` so Metal/MLX/wgpu run device kernels instead of
    // the CPU host-delegate (Metal also widens Bool Custom outputs to F32,
    // which broke the reference kernel's `expect_bool_mut`).
    let id = m.add_node(Op::Compare(CmpOp::Ne), vec![x, x], s);
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
