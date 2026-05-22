// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! HIR → MIR lowering for depthwise 1D convolution blocks.

use crate::infer::GraphExt;
use crate::{Graph, NodeId, Op, Shape};

/// Causal depthwise Conv1d on `[batch, seq, channels]` (BSC) tensors.
///
/// Inputs:
///   - `input`   `[B, S, C]`
///   - `weight`  `[C, 1, 1, K]` — packed grouped Conv2d kernel (NCHW trick)
///   - `left_pad` `[B, K-1, C]` — zero causal padding prepended along seq
///
/// Output: `[B, S, C]`
///
/// Lowers via left-pad concat → BSC→BCW→NCHW → grouped `Op::Conv` → BSC.
pub fn lower_depthwise_conv1d_causal(
    g: &mut Graph,
    input: NodeId,
    weight: NodeId,
    left_pad: NodeId,
    kernel_size: usize,
    out_shape: Shape,
) -> NodeId {
    let in_shape = g.node(input).shape.clone();
    let batch = in_shape.dim(0).unwrap_static();
    let seq = in_shape.dim(1).unwrap_static();
    let channels = in_shape.dim(2).unwrap_static();
    let dtype = in_shape.dtype();
    let k = kernel_size;
    let padded_len = (k - 1) + seq;

    let padded = g.concat(
        vec![left_pad, input],
        1,
        Shape::new(&[batch, padded_len, channels], dtype),
    );
    let bcw = g.transpose_(padded, vec![0, 2, 1]);
    let nchw = g.reshape_(
        bcw,
        vec![batch as i64, channels as i64, 1, padded_len as i64],
    );
    let conv = g.add_node(
        Op::Conv {
            kernel_size: vec![1, k],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: channels,
        },
        vec![nchw, weight],
        Shape::new(&[batch, channels, 1, seq], dtype),
    );
    let bcs = g.reshape_(conv, vec![batch as i64, channels as i64, seq as i64]);
    let out = g.transpose_(bcs, vec![0, 2, 1]);
    debug_assert_eq!(g.node(out).shape, out_shape);
    out
}
