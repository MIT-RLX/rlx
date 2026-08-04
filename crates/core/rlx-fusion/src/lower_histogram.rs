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

use crate::pass::Pass;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{BinaryOp, CmpOp, ReduceOp};
use rlx_ir::*;
use std::collections::HashMap;

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
        // Sum over every axis → a single count, kept as shape [1] for concat.
        let cnt = g.reduce(
            inb,
            ReduceOp::Sum,
            all_axes.clone(),
            false,
            count_shape.clone(),
        );
        counts.push(cnt);
    }
    g.concat_(counts, 0)
}

/// Rewrite every `Op::Histogram` node into primitives.
pub struct LowerHistogram;

impl Pass for LowerHistogram {
    fn name(&self) -> &str {
        "lower_histogram"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::Histogram { .. }))
        {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_id = if let Op::Histogram { bins, min, max } = &node.op {
                let x = id_map[&node.inputs[0]];
                lower_histogram(&mut new_graph, x, *bins, *min, *max)
            } else {
                let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}
