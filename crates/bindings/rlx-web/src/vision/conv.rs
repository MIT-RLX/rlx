// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! Shared NCHW conv/pool helpers for vision graphs.

use rlx_ir::op::*;
use rlx_ir::{DType, Graph, NodeId, Shape};

pub fn conv2d(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    batch: usize,
    c_out: usize,
    h_out: usize,
    w_out: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> NodeId {
    g.add_node(
        Op::Conv {
            kernel_size: kernel.to_vec(),
            stride: stride.to_vec(),
            padding: padding.to_vec(),
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[batch, c_out, h_out, w_out], DType::F32),
    )
}

pub fn maxpool2x2(
    g: &mut Graph,
    x: NodeId,
    batch: usize,
    c: usize,
    h_out: usize,
    w_out: usize,
) -> NodeId {
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[batch, c, h_out, w_out], DType::F32),
    )
}

pub fn avgpool(
    g: &mut Graph,
    x: NodeId,
    batch: usize,
    c: usize,
    kernel: [usize; 2],
    h_out: usize,
    w_out: usize,
) -> NodeId {
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Mean,
            kernel_size: kernel.to_vec(),
            stride: kernel.to_vec(),
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[batch, c, h_out, w_out], DType::F32),
    )
}

/// Per-channel bias broadcast for `[B, C, H, W]` feature maps.
pub fn bias_add_4d(
    g: &mut Graph,
    x: NodeId,
    bias: NodeId,
    batch: usize,
    c: usize,
    h: usize,
    w: usize,
) -> NodeId {
    let f = DType::F32;
    let bias_4d = g.add_node(
        Op::Reshape {
            new_shape: vec![1, c as i64, 1, 1],
        },
        vec![bias],
        Shape::new(&[1, c, 1, 1], f),
    );
    g.binary(BinaryOp::Add, x, bias_4d, Shape::new(&[batch, c, h, w], f))
}
