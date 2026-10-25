// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! I32 / U32 on the CPU backend: stored as the integers they are.
//!
//! Some kernels read an integer tensor as i32 (casts, scans, the fixed-point
//! FFT, custom ops); the element-wise, reduce and index kernels compute in
//! f32. Feeding one from the other read the bits the wrong way: a cast
//! after an integer multiply returned `1065353216` for `1`, and token ids
//! fed as integers all selected row 0. [`legalize_int_storage`] keeps every
//! I32 / U32 tensor in integer form and wraps each f32-computing op that
//! touches one: its integer inputs are cast to f32, and an integer result is
//! computed in f32 and cast back. I64 kernels are integer already, but the
//! reduce kernels have none: a reduce over I64 goes through f64.

use rlx_ir::stateful::rebuild;
use rlx_ir::{DType, Graph, Op, Shape};

fn is_int(dt: DType) -> bool {
    matches!(dt, DType::I32 | DType::U32)
}

/// Ops whose kernels move or read I32 / U32 as integers: the rest compute
/// in f32.
fn integer_safe(op: &Op) -> bool {
    matches!(
        op,
        Op::Input { .. }
            | Op::Param { .. }
            | Op::Constant { .. }
            | Op::Cast { .. }
            | Op::Reshape { .. }
            | Op::Narrow { .. }
            | Op::Transpose { .. }
            | Op::Concat { .. }
            | Op::Expand { .. }
            | Op::FftQ { .. }
            | Op::Custom { .. }
            | Op::DynamicUpdateSlice { .. }
            // Integer kernels: i8 operands, and an I32 bias read as raw i32
            // (cast to f32 here, its bits became a huge bias and every output
            // saturated to -128 / 127).
            | Op::QMatMul { .. }
            | Op::QConv2d { .. }
    )
}

/// See the module docs. A graph with no I32 / U32 tensor is returned as is.
pub fn legalize_int_storage(graph: Graph) -> Graph {
    let touches_int = graph.nodes().iter().any(|n| {
        !integer_safe(&n.op)
            && (is_int(n.shape.dtype())
                || n.inputs
                    .iter()
                    .any(|i| is_int(graph.node(*i).shape.dtype())))
    });
    // Gather reads its table as data and its indices as f32 (or i64).
    let gather_int_index = graph.nodes().iter().any(|n| {
        matches!(n.op, Op::Gather { .. })
            && n.inputs
                .get(1)
                .is_some_and(|i| is_int(graph.node(*i).shape.dtype()))
    });
    let reduce_i64 = graph.nodes().iter().any(|n| {
        matches!(n.op, Op::Reduce { .. })
            && n.inputs
                .first()
                .is_some_and(|i| graph.node(*i).shape.dtype() == DType::I64)
    });
    if !touches_int && !gather_int_index && !reduce_i64 {
        return graph;
    }
    rebuild(&graph, |g, _attrs, node, inputs| {
        if integer_safe(&node.op) {
            return Ok(None);
        }
        let cast = |g: &mut Graph, id, to: DType| {
            let dims = g.node(id).shape.dims().to_vec();
            g.add_node(Op::Cast { to }, vec![id], Shape::from_dims(&dims, to))
        };
        // Gather: only the indices go to f32; the table is data.
        if let Op::Gather { .. } = node.op {
            let Some(&idx) = inputs.get(1) else {
                return Ok(None);
            };
            if !is_int(g.node(idx).shape.dtype()) {
                return Ok(None);
            }
            let idx = cast(g, idx, DType::F32);
            let mut ins = inputs.to_vec();
            ins[1] = idx;
            return Ok(Some(g.add_node(node.op.clone(), ins, node.shape.clone())));
        }
        // The reduce kernels have no I64 arm: through f64, exact to 2^53.
        if let Op::Reduce { .. } = node.op
            && let Some(&x) = inputs.first()
            && g.node(x).shape.dtype() == DType::I64
        {
            let x = cast(g, x, DType::F64);
            let dims = node.shape.dims().to_vec();
            let r = g.add_node(
                node.op.clone(),
                vec![x],
                Shape::from_dims(&dims, DType::F64),
            );
            return Ok(Some(if node.shape.dtype() == DType::F64 {
                r
            } else {
                cast(g, r, node.shape.dtype())
            }));
        }
        let out_int = is_int(node.shape.dtype());
        let in_int = inputs.iter().any(|i| is_int(g.node(*i).shape.dtype()));
        if !out_int && !in_int {
            return Ok(None);
        }
        let ins: Vec<_> = inputs
            .iter()
            .map(|&i| {
                if is_int(g.node(i).shape.dtype()) {
                    cast(g, i, DType::F32)
                } else {
                    i
                }
            })
            .collect();
        if !out_int {
            return Ok(Some(g.add_node(node.op.clone(), ins, node.shape.clone())));
        }
        let dims = node.shape.dims().to_vec();
        let f = g.add_node(node.op.clone(), ins, Shape::from_dims(&dims, DType::F32));
        Ok(Some(cast(g, f, node.shape.dtype())))
    })
    .expect("rlx-cpu: integer storage legalization keeps the graph's order")
}
