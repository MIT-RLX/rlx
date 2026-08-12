// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::Histogram` to primitives that are native on every backend
//! (`Compare` + mul + `Reduce::Sum` + `Concat`) — the semantic oracle for every
//! backend that does not claim `OpKind::Histogram` (i.e. all except CPU). The
//! decomposition is O(n · bins) but runs on the backend's own kernels, so a
//! histogram over device tensors never round-trips to the host. CPU keeps the
//! O(n) native `Thunk::Histogram`. Runs in the legalize loop like `LowerSlice`.
//!
//! Semantics (must match the CPU kernel): half-open buckets closed at the top —
//! bin `b` counts `lo_b <= x < hi_b` where `lo_b = min + b·width`,
//! `width = (max-min)/bins`. Out-of-range elements are dropped and `x == max`
//! lands in the last bin (the top edge is nudged to the next representable f32
//! so `x <= max` is included exactly, and nothing above `max` leaks in).

use crate::rewriter::{MatchRewrite, RewriteCtx};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{BinaryOp, CmpOp, ReduceOp};
use rlx_ir::*;

/// Immediate next f32 toward +∞ (a tight `nextafter(x, +inf)`), so a strict
/// `< next_up(max)` test is exactly `<= max`.
fn next_up(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let next = if x >= 0.0 { bits + 1 } else { bits - 1 };
    f32::from_bits(next)
}

/// Decompose one `Op::Histogram` (input `x` already remapped) to primitives.
pub fn lower_histogram(g: &mut Graph, x: NodeId, bins: usize, min: f32, max: f32) -> NodeId {
    let dtype = g.shape(x).dtype();
    let rank = g.shape(x).rank();
    let all_axes: Vec<usize> = (0..rank).collect();
    let count_shape = Shape::new(&[1], DType::F32);
    // All-axes reduce with `keep_dim` → one entry per input axis, all 1.
    let keepdim_shape = Shape::new(&vec![1usize; rank], DType::F32);
    let x_f32 = g.shape(x).clone().with_dtype(DType::F32);
    let x_bool = g.shape(x).clone().with_dtype(DType::Bool);
    let width = (max - min) / bins as f32;

    let mut counts: Vec<NodeId> = Vec::with_capacity(bins);
    for b in 0..bins {
        let lo = min + b as f32 * width;
        let hi = if b + 1 == bins {
            next_up(max) // include x == max, exclude anything above
        } else {
            min + (b + 1) as f32 * width
        };
        let lo_c = g.full(&[1], lo, dtype);
        let hi_c = g.full(&[1], hi, dtype);
        // mask = (x >= lo) && (x < hi). `Compare` yields bool; cast each to f32
        // so the logical AND is a multiply and the count is a plain sum.
        let ge = g.add_node(Op::Compare(CmpOp::Ge), vec![x, lo_c], x_bool.clone());
        let lt = g.add_node(Op::Compare(CmpOp::Lt), vec![x, hi_c], x_bool.clone());
        let ge_f = g.add_node(Op::Cast { to: DType::F32 }, vec![ge], x_f32.clone());
        let lt_f = g.add_node(Op::Cast { to: DType::F32 }, vec![lt], x_f32.clone());
        let inb = g.add_node(Op::Binary(BinaryOp::Mul), vec![ge_f, lt_f], x_f32.clone());
        // Sum over every axis → a single count. `keep_dim` so the result stays
        // rank-`rank` ([1,1,…,1]) rather than collapsing to rank-0: shape
        // inference rewrites a `keep_dim: false` all-axes reduce to `dims: []`,
        // and concatenating rank-0 operands along axis 0 is invalid — wgpu
        // indexes `in_shape[axis]` out of bounds and MLX rejects it outright
        // ("Axis 0 is out of bounds for array with 0 dimensions"). Reshape to
        // `[1]` so `concat_` sees honest rank-1 inputs on every backend.
        let cnt = g.reduce(
            inb,
            ReduceOp::Sum,
            all_axes.clone(),
            true,
            keepdim_shape.clone(),
        );
        let cnt = g.reshape(cnt, vec![1], count_shape.clone());
        counts.push(cnt);
    }
    g.concat_(counts, 0)
}

/// Rewrite every `Op::Histogram` node into primitives.
pub struct LowerHistogram;

impl MatchRewrite for LowerHistogram {
    fn name(&self) -> &str {
        "lower_histogram"
    }

    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Histogram]
    }

    fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
        let Op::Histogram { bins, min, max } = &node.op else {
            return None;
        };
        let x = ctx.input(0);
        Some(lower_histogram(ctx.out, x, *bins, *min, *max))
    }
}
