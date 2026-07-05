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
    let _cond = ctx.tensor(&node.inputs[0])?;
    lower_if_stub(m, ctx, node)
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
