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

//! Lower `Op::FakeQuantize` to primitives. Semantic oracle for backends
//! without a native fake-quant kernel.
//!
//! Forward is exactly the CPU kernel's `clamp(round(x/s), -qmax, qmax) · s`
//! expressed as one primitive chain:
//!
//! ```text
//! s   = max( reduce_max(|x|) / qmax , 1e-12 )     (per `axis` channel)
//! out = clamp( round(x / s), -qmax, qmax ) · s
//! ```
//!
//! The single canonical `Activation::Round` (half-to-even) is what makes this
//! agree bit-for-bit across every backend — the earlier per-backend prototype
//! diverged precisely because each backend rounded ties its own way. Only the
//! stateless **`PerBatch`** scale mode decomposes here; `EMA` / `Fixed` carry a
//! mutable/looked-up state tensor and stay on their native / host-staged path.

use crate::pass::Pass;
use rlx_ir::op::{Activation, BinaryOp, ReduceOp, ScaleMode};
use rlx_ir::*;
use std::collections::HashMap;

fn qmax_for(bits: u8) -> f32 {
    match bits {
        8 => 127.0,
        4 => 7.0,
        2 => 1.0,
        n => panic!("FakeQuantize: unsupported bits {n}"),
    }
}

fn static_numel(s: &Shape) -> usize {
    s.dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n,
            _ => panic!("FakeQuantize lowering requires static dims"),
        })
        .product()
}

fn static_dims_i64(s: &Shape) -> Vec<i64> {
    s.dims()
        .iter()
        .map(|d| match d {
            Dim::Static(n) => *n as i64,
            _ => panic!("FakeQuantize lowering requires static dims"),
        })
        .collect()
}

/// Decompose one `Op::FakeQuantize` (input `x` already remapped). Returns
/// `None` for the stateful `EMA` / `Fixed` modes, which keep their native path.
pub fn lower_fake_quantize(
    g: &mut Graph,
    x: NodeId,
    bits: u8,
    axis: Option<usize>,
    scale_mode: ScaleMode,
) -> Option<NodeId> {
    if !matches!(scale_mode, ScaleMode::PerBatch) {
        return None;
    }
    let xshape = g.shape(x).clone();
    let rank = xshape.rank();
    let qmax = qmax_for(bits);

    // Per-channel max-abs → scale, keeping the channel axis (`axis`) alive.
    let abs = g.activation(Activation::Abs, x, xshape.clone());
    let reduce_axes: Vec<usize> = (0..rank).filter(|a| Some(*a) != axis).collect();
    let rshape = shape::reduce_shape(&xshape, &reduce_axes, true).expect("reduce shape");
    let mx = g.reduce(abs, ReduceOp::Max, reduce_axes, true, rshape.clone());

    let rnum = static_numel(&rshape);
    // A constant tensor filled with a single value = that value's LE bytes
    // repeated `rnum` times.
    let inv_bytes: Vec<u8> = (1.0f32 / qmax).to_le_bytes().repeat(rnum);
    let inv = g.add_node(Op::Constant { data: inv_bytes }, vec![], rshape.clone());
    let eps_bytes: Vec<u8> = 1e-12f32.to_le_bytes().repeat(rnum);
    let eps = g.add_node(Op::Constant { data: eps_bytes }, vec![], rshape.clone());

    let sc = g.binary(BinaryOp::Mul, mx, inv, rshape.clone());
    let sc = g.binary(BinaryOp::Max, sc, eps, rshape.clone());

    // Broadcast scale up to `x`'s full shape (keep_dim kept the rank aligned).
    let sc_full = g.add_node(
        Op::Expand {
            target_shape: static_dims_i64(&xshape),
        },
        vec![sc],
        xshape.clone(),
    );

    let xdiv = g.binary(BinaryOp::Div, x, sc_full, xshape.clone());
    let r = g.activation(Activation::Round, xdiv, xshape.clone());
    let rc = g.add_node(
        Op::Clamp {
            min: -qmax,
            max: qmax,
        },
        vec![r],
        xshape.clone(),
    );
    Some(g.binary(BinaryOp::Mul, rc, sc_full, xshape))
}

/// Rewrite `Op::FakeQuantize` (PerBatch) nodes into primitives.
pub struct LowerFakeQuantize;

impl Pass for LowerFakeQuantize {
    fn name(&self) -> &str {
        "lower_fake_quantize"
    }

    fn run(&self, graph: Graph) -> Graph {
        if !graph.nodes().iter().any(
            |n| matches!(&n.op, Op::FakeQuantize { scale_mode, .. } if matches!(scale_mode, ScaleMode::PerBatch)),
        ) {
            return graph;
        }

        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for node in graph.nodes() {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
            let new_id = if let Op::FakeQuantize {
                bits,
                axis,
                ste: _,
                scale_mode,
            } = &node.op
            {
                match lower_fake_quantize(&mut new_graph, inputs[0], *bits, *axis, *scale_mode) {
                    Some(id) => id,
                    None => new_graph.add_node(node.op.clone(), inputs, node.shape.clone()),
                }
            } else {
                new_graph.add_node(node.op.clone(), inputs, node.shape.clone())
            };
            id_map.insert(node.id, new_id);
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|i| id_map[i]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}
