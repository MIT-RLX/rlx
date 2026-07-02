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

//
// Primitive compositions for training `*Backward` ops (higher-order AD).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

/// `CumsumBackward` via batched matmul with a static lower-triangular ones matrix.
pub fn compose_cumsum_backward(
    g: &mut Graph,
    dy: NodeId,
    out_shape: &Shape,
    axis: i32,
    exclusive: bool,
) -> NodeId {
    let rank = out_shape.rank();
    assert!(rank >= 1, "compose_cumsum_backward: rank >= 1");
    let ax = axis_pos(axis, rank);
    let l = out_shape.dim(ax).unwrap_static();
    assert!(l <= 256, "compose_cumsum_backward: L={l} too large");

    let perm = perm_move_axis_last(rank, ax);
    let dy_work = if ax != rank - 1 {
        let dy_shape = g.node(dy).shape.clone();
        let permuted = apply_perm_shape(&dy_shape, &perm);
        g.add_node(Op::Transpose { perm: perm.clone() }, vec![dy], permuted)
    } else {
        dy
    };
    let work_shape = g.node(dy_work).shape.clone();
    let batch = work_shape.num_elements().expect("static cumsum bwd") / l;
    let dt = work_shape.dtype();
    let flat_dy = g.reshape_(dy_work, vec![batch as i64, l as i64]);
    let mat = cumsum_backward_matrix(l, exclusive);
    let m_node = f32_tensor_const(mat, Shape::new(&[l, l], dt), g);
    let dx_flat = g.matmul(flat_dy, m_node, Shape::new(&[batch, l], dt));
    let dims: Vec<i64> = work_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    let dx = g.reshape_(dx_flat, dims);
    if ax != rank - 1 {
        let inv = invert_perm(&perm);
        g.add_node(Op::Transpose { perm: inv }, vec![dx], out_shape.clone())
    } else {
        dx
    }
}


/// `GatherBackward` via `ScatterAdd` (axis 0) or flattened scatter (other axes).
pub fn compose_gather_backward(
    g: &mut Graph,
    dy: NodeId,
    indices: NodeId,
    table_shape: &Shape,
    axis: i32,
) -> NodeId {
    let rank = table_shape.rank();
    let ax = axis_pos(axis, rank);
    if ax == 0 {
        return g.add_node(Op::ScatterAdd, vec![dy, indices], table_shape.clone());
    }

    let _dy_shape = g.node(dy).shape.clone();
    let idx_shape = g.node(indices).shape.clone();
    let outer = dim_product(table_shape, 0, ax);
    let axis_dim = table_shape.dim(ax).unwrap_static();
    let trailing = dim_product(table_shape, ax + 1, rank);
    let num_idx = idx_shape.num_elements().expect("static gather idx");
    let dt = table_shape.dtype();

    let updates = g.reshape_(dy, vec![(outer * num_idx) as i64, trailing as i64]);

    let idx_rep = if outer == 1 {
        g.reshape_(indices, vec![num_idx as i64])
    } else {
        let mut parts: Vec<NodeId> = Vec::with_capacity(outer);
        for _ in 0..outer {
            parts.push(g.reshape_(indices, vec![num_idx as i64]));
        }
        g.concat_(parts, 0)
    };

    let mut outer_ids: Vec<f32> = Vec::with_capacity(outer * num_idx);
    for o in 0..outer {
        for _ in 0..num_idx {
            outer_ids.push(o as f32);
        }
    }
    let outer_node = f32_tensor_const(outer_ids, Shape::new(&[outer * num_idx], dt), g);
    let axis_s = scalar_const(axis_dim as f64, &Shape::scalar(dt), g);
    let axis_b = broadcast_scalar(g, axis_s, &Shape::new(&[outer * num_idx], dt));
    let base = g.mul(outer_node, axis_b);
    let flat_idx = g.add(base, idx_rep);
    let scattered = g.add_node(
        Op::ScatterAdd,
        vec![updates, flat_idx],
        Shape::new(&[outer * axis_dim, trailing], dt),
    );

    let dims: Vec<i64> = table_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static() as i64)
        .collect();
    g.reshape_(scattered, dims)
}

