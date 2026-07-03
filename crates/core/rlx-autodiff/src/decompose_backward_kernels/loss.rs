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

//! `loss` — extracted from the `decompose_backward_kernels` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

/// `SoftmaxCrossEntropyBackward` via softmax + compare-built one-hot.
pub fn compose_softmax_cross_entropy_backward(
    g: &mut Graph,
    logits: NodeId,
    labels: NodeId,
    d_loss: NodeId,
    out_shape: &Shape,
) -> NodeId {
    let [n, c] = static_dim2(out_shape).expect("static [N,C] logits");
    let dt = out_shape.dtype();
    let sm = g.softmax(logits, -1, out_shape.clone());
    let labels_flat = if g.node(labels).shape.rank() == 1 {
        labels
    } else {
        g.reshape_(labels, vec![n as i64])
    };
    let mut cols: Vec<NodeId> = Vec::with_capacity(c);
    let labels_shape = g.node(labels_flat).shape.clone();
    let one = scalar_const(1.0, &Shape::scalar(dt), g);
    let zero = scalar_const(0.0, &Shape::scalar(dt), g);
    let one_b = broadcast_scalar(g, one, &labels_shape);
    let zero_b = broadcast_scalar(g, zero, &labels_shape);
    for ci in 0..c {
        let class = scalar_const(ci as f64, &Shape::scalar(dt), g);
        let class_b = broadcast_scalar(g, class, &labels_shape);
        let eq = compare_eq(g, labels_flat, class_b);
        let col = g.add_node(Op::Where, vec![eq, one_b, zero_b], labels_shape.clone());
        cols.push(col);
    }
    // `cols[ci]` is `[N]`; concatenating along axis 0 yields a CLASS-major
    // `[C*N]` buffer (all of class 0's N entries, then class 1's, …). That is a
    // `[C, N]` matrix — reshaping it directly to `[N, C]` row-major scrambles
    // the one-hot (the old bug: SCE-backward produced a transposed one-hot, so
    // decompose-path backends — Metal/wgpu — never learned). Reshape to `[C, N]`
    // then transpose to `[N, C]`.
    let one_hot_flat = g.concat_(cols, 0);
    let one_hot_cn = g.reshape_(one_hot_flat, vec![c as i64, n as i64]);
    let one_hot = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![one_hot_cn],
        Shape::new(&[n, c], dt),
    );
    let diff = g.sub(sm, one_hot);
    let dl_b = broadcast_scalar(g, d_loss, out_shape);
    g.mul(diff, dl_b)
}

