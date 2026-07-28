// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//
// Primitive compositions for training `*Backward` ops (higher-order AD).

//! `quant` — extracted from the `decompose_backward_kernels` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use rlx_ir::infer::GraphExt;
use rlx_ir::op::{AttentionBwdWrt, CmpOp, MaskKind, SteKind};
use rlx_ir::shape;
use rlx_ir::shape::Dim;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::*;

/// `FakeQuantizeBackward` STE paths (matches CPU `training_bwd` kernels).
pub fn compose_fake_quantize_backward(
    g: &mut Graph,
    x: NodeId,
    dy: NodeId,
    out_shape: &Shape,
    bits: u8,
    axis: Option<usize>,
    ste: SteKind,
) -> NodeId {
    let len = out_shape.num_elements().expect("static fake_quant");
    let dt = out_shape.dtype();
    let q_max = q_max_for_bits(bits);

    let (chan_dim, _inner) = match axis {
        None => (1usize, len),
        Some(ax) => {
            let rank = out_shape.rank();
            assert!(ax < rank, "fake_quant axis in range");
            let cd = out_shape.dim(ax).unwrap_static();
            let inner = dim_product(out_shape, ax + 1, rank);
            (cd, inner)
        }
    };

    if matches!(ste, SteKind::Identity) {
        return dy;
    }

    let abs_x = g.add_node(
        Op::Activation(rlx_ir::op::Activation::Abs),
        vec![x],
        out_shape.clone(),
    );

    let flat_abs = g.reshape_(abs_x, vec![len as i64]);
    let max_abs = if chan_dim == 1 {
        g.reduce(
            flat_abs,
            rlx_ir::op::ReduceOp::Max,
            vec![0],
            false,
            Shape::new(&[1], dt),
        )
    } else {
        assert!(axis.is_some(), "per-channel fake_quant needs axis");
        let ax = axis.unwrap();
        g.reduce(
            abs_x,
            rlx_ir::op::ReduceOp::Max,
            vec![ax],
            true,
            Shape::from_dims(
                &out_shape
                    .dims()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, d)| if i == ax { None } else { Some(*d) })
                    .collect::<Vec<_>>(),
                dt,
            ),
        )
    };

    let q_s = scalar_const(q_max as f64, &Shape::scalar(dt), g);
    let max_shape = g.node(max_abs).shape.clone();
    let q_b = broadcast_scalar(g, q_s, &max_shape);
    let scale = g.div(max_abs, q_b);
    let scale_shape = g.node(scale).shape.clone();
    let zero_node = f32_tensor_const(vec![0.0], out_shape.clone(), g);

    match ste {
        SteKind::Identity => dy,
        SteKind::ClippedIdentity => {
            let q_b2 = broadcast_scalar(g, q_s, &scale_shape);
            let bound = g.mul(q_b2, scale);
            let bound_b = broadcast_scalar(g, bound, out_shape);
            let cmp = compare_ge(g, bound_b, abs_x);
            g.add_node(Op::Where, vec![cmp, dy, zero_node], out_shape.clone())
        }
        SteKind::Tanh => {
            let scale_b = broadcast_scalar(g, scale, out_shape);
            let z = g.div(x, scale_b);
            let t = g.add_node(
                Op::Activation(rlx_ir::op::Activation::Tanh),
                vec![z],
                out_shape.clone(),
            );
            let t2 = g.mul(t, t);
            let one_s = scalar_const(1.0, &Shape::scalar(dt), g);
            let one = broadcast_scalar(g, one_s, out_shape);
            let att = g.sub(one, t2);
            g.mul(dy, att)
        }
        SteKind::HardTanh => {
            let q_b2 = broadcast_scalar(g, q_s, &scale_shape);
            let bound = g.mul(q_b2, scale);
            let bound_b = broadcast_scalar(g, bound, out_shape);
            let ratio = g.div(abs_x, bound_b);
            let one_s = scalar_const(1.0, &Shape::scalar(dt), g);
            let one = broadcast_scalar(g, one_s, out_shape);
            let att_raw = g.sub(one, ratio);
            let zero = f32_tensor_const(vec![0.0], out_shape.clone(), g);
            let pos = compare_ge(g, att_raw, zero);
            let att = g.add_node(Op::Where, vec![pos, att_raw, zero], out_shape.clone());
            g.mul(dy, att)
        }
    }
}
