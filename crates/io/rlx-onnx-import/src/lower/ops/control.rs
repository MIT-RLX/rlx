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

//! `control` — extracted from the `ops` module for navigability (see `mod.rs`).

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

/// Lower ONNX `If` (subgraph lowering not implemented; stub when import is non-strict).
pub(super) fn lower_if(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    // Resolve the taken branch at compile time. The parser recorded both
    // branches' output tensor names (`_then_outputs`/`_else_outputs`) and folded
    // their `Constant`s into params; here we fold the condition and map the taken
    // branch's outputs onto this `If`'s outputs. This lowers the common
    // "cached-table" `If` (e.g. the Zipformer relative-position embedding, whose
    // `then` branch is a `Constant` window table selected when `2T-1 ≤ max_len`)
    // without inlining the (large) recompute branch. Falls back to the zero stub
    // when the branch's outputs are not statically resolvable.
    let cond = eval_if_condition(&*ctx, &*m, &node.inputs[0]);
    let take_then = cond.unwrap_or(true); // default to the cached ("then") path
    let key = if take_then {
        "_then_outputs"
    } else {
        "_else_outputs"
    };
    let branch_outs: Vec<String> = node
        .attrs
        .get(key)
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    // Fast path: the taken branch's outputs are folded constants / already-lowered
    // params (the cached-table `If`, e.g. Zipformer rel-pos embedding).
    if branch_outs.len() == node.outputs.len()
        && branch_outs.iter().all(|b| resolvable_param(ctx, b))
    {
        for (out, bn) in node.outputs.iter().zip(&branch_outs) {
            let id = materialize_param(m, ctx, bn)
                .ok_or_else(|| anyhow!("If branch output {bn} unresolvable"))?;
            ctx.env.insert(out.clone(), id);
        }
        return Ok(true);
    }
    // Inline path: the taken branch COMPUTES its output (`Squeeze`/`Identity`/… of
    // an outer-scope tensor already in env). Lower the branch's nodes in order, then
    // alias this `If`'s outputs onto them. (MOSS codec's final audio-reshape `If`.)
    let branch_nodes = ctx
        .if_branches
        .get(&node.name)
        .map(|(t, e)| if take_then { t.clone() } else { e.clone() });
    if let Some(branch_nodes) = branch_nodes {
        if !branch_nodes.is_empty() && branch_outs.len() == node.outputs.len() {
            // Folded branch `Constant`s live in `ctx.params` / `i64_params` and were
            // stripped from `branch_nodes`. `lower_node` only materializes f32 params
            // when they appear in `inits` — an empty set left Mul/etc. hanging on
            // missing env entries (LuxTTS Zipformer `encoder_pos` else branch).
            let inits: HashSet<String> = ctx
                .params
                .keys()
                .chain(ctx.i64_params.keys())
                .chain(ctx.typed_params.keys())
                .cloned()
                .collect();
            let mut ok = true;
            for bn in &branch_nodes {
                if lower_node(m, ctx, bn, &inits).is_err() {
                    ok = false;
                    break;
                }
            }
            if ok && branch_outs.iter().all(|b| ctx.env.contains_key(b)) {
                for (out, bo) in node.outputs.iter().zip(&branch_outs) {
                    let id = ctx.env[bo];
                    ctx.env.insert(out.clone(), id);
                }
                return Ok(true);
            }
        }
    }
    lower_if_stub(m, ctx, node)
}

/// Whether an `If`-branch output name resolves to a compile-time tensor (an
/// already-lowered value or a folded param) — i.e. the branch can be emitted
/// without inlining its subgraph nodes.
fn resolvable_param(ctx: &LowerCtx<'_>, name: &str) -> bool {
    ctx.env.contains_key(name) || ctx.params.contains_key(name) || ctx.i64_params.contains_key(name)
}

/// Materialize a folded param (or already-lowered tensor) into the env, mirroring
/// the input-materialization `lower_node` performs for ordinary node inputs.
fn materialize_param(m: &mut HirMut<'_>, ctx: &mut LowerCtx<'_>, name: &str) -> Option<HirNodeId> {
    if let Some(&id) = ctx.env.get(name) {
        return Some(id);
    }
    if let Some(v) = ctx.i64_params.get(name) {
        let dims = ctx
            .init_shapes
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![v.len()]);
        let bytes: Vec<u8> = v.iter().flat_map(|d| d.to_le_bytes()).collect();
        let id = m.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(&dims, DType::I64),
        );
        ctx.env.insert(name.to_string(), id);
        return Some(id);
    }
    if ctx.params.contains_key(name) {
        let n = ctx.params[name].len();
        let dims = ctx
            .init_shapes
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![n]);
        let id = m.param(name, Shape::new(&dims, DType::F32));
        ctx.env.insert(name.to_string(), id);
        return Some(id);
    }
    None
}

/// Fold an `If` condition to a bool at compile time: walk `Cast`/`Identity` back
/// to the comparison node, then evaluate its two integer (shape-derived) operands
/// and compare. Returns None if the condition can't be resolved statically.
fn eval_if_condition(ctx: &LowerCtx<'_>, m: &HirMut<'_>, cond_name: &str) -> Option<bool> {
    let mut name = cond_name.to_string();
    let mut cmp = None;
    for _ in 0..8 {
        let node = bundle_node_for_output_ctx(ctx, &name)?;
        match node.op.as_str() {
            "Cast" | "Identity" if !node.inputs.is_empty() => name = node.inputs[0].clone(),
            "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual" | "Equal" => {
                cmp = Some(node);
                break;
            }
            _ => return None,
        }
    }
    let cmp = cmp?;
    let a = *eval_i64_shaped(ctx, m, &cmp.inputs[0], 0)?.0.first()?;
    let b = *eval_i64_shaped(ctx, m, &cmp.inputs[1], 0)?.0.first()?;
    Some(match cmp.op.as_str() {
        "Greater" => a > b,
        "GreaterOrEqual" => a >= b,
        "Less" => a < b,
        "LessOrEqual" => a <= b,
        "Equal" => a == b,
        _ => return None,
    })
}

pub(super) fn lower_if_stub(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    if ctx.opts.strict {
        anyhow::bail!(
            "If at {} is not lowered to subgraph HIR yet (strict import)",
            node.name
        );
    }
    ctx.record_stub(node, "If");
    for out_name in &node.outputs {
        let sh = Shape::new(&[1, 1, ctx.opts.sequence_length], DType::F32);
        let key = format!("__stub__/{}", out_name);
        let n = sh.num_elements().unwrap_or(1).min(MAX_STUB_ELEMENTS);
        let id = m.param(&key, sh);
        ctx.params.insert(key, vec![0.0; n]);
        ctx.env.insert(out_name.clone(), id);
    }
    Ok(true)
}

pub(super) fn lower_sequence_empty(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let out = node
        .outputs
        .first()
        .context("SequenceEmpty missing output")?;
    let shape = Shape::new(&[0], DType::I64);
    let id = m.add_node(Op::Constant { data: vec![] }, vec![], shape);
    ctx.env.insert(out.clone(), id);
    Ok(true)
}

pub(super) fn lower_control_flow(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    match node.op.as_str() {
        "If" => lower_if(m, ctx, node),
        "Loop" => lower_loop(m, ctx, node),
        "Scan" => lower_scan(m, ctx, node),
        "SplitToSequence" => lower_split_to_sequence(m, ctx, node),
        "ConcatFromSequence" => lower_concat_from_sequence(m, ctx, node),
        "SequenceEmpty" => lower_sequence_empty(m, ctx, node),
        other => anyhow::bail!("unexpected control-flow op {other}"),
    }
}

pub(super) fn lower_scan(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    if ctx.opts.strict {
        anyhow::bail!("Scan at {} not implemented", node.name);
    }
    ctx.passthrough_stub(m, node)?;
    Ok(true)
}

pub(super) fn lower_split_to_sequence(
    _m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let x = ctx.tensor(node.inputs.first().context("SplitToSequence input")?)?;
    for out in &node.outputs {
        ctx.env.insert(out.clone(), x);
    }
    Ok(true)
}

pub(super) fn lower_loop(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    // Loop output is consumed by ConcatFromSequence; fusion reads upstream tensors directly.
    let out = node.outputs.first().context("Loop missing output")?;
    let n = control_flow::alignment_frame_upper_bound(
        ctx.opts.sequence_length,
        ctx.opts.max_frames_per_token,
    );
    let shape = Shape::new(&[n], DType::I64);
    let id = m.add_node(Op::Constant { data: vec![] }, vec![], shape);
    ctx.env.insert(out.clone(), id);
    Ok(true)
}

pub(super) fn lower_concat_from_sequence(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let align = control_flow::resolve_duration_align_inputs(ctx.nodes)
        .context("ConcatFromSequence: duration alignment inputs")?;
    let duration_mask = ctx.tensor(&align.duration_mask)?;
    let range_ids = ctx.tensor(&align.range_ids)?;
    let split_lens = ctx.tensor(&align.split_lens)?;
    let trip = ctx.tensor(&align.trip_count)?;
    let n = control_flow::alignment_frame_upper_bound(
        ctx.opts.sequence_length,
        ctx.opts.max_frames_per_token,
    );
    let s = Shape::new(&[n], DType::I64);
    let id = m.add_node(
        Op::Custom {
            name: "onnx.ConcatFromSequence".to_string(),
            num_inputs: 4,
            attrs: vec![],
        },
        vec![duration_mask, range_ids, split_lens, trip],
        s,
    );
    ctx.env.insert(node.outputs[0].clone(), id);
    Ok(true)
}
