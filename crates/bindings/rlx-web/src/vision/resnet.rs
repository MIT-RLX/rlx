// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! CIFAR-sized ResNet-style classifier (two residual blocks, no batch norm).

use rlx_ir::op::*;
use rlx_ir::{DType, Graph, NodeId, Shape};

use super::conv::{avgpool, bias_add_4d, conv2d};

pub const INPUT: &[usize] = &[3, 32, 32];
pub const NUM_CLASSES: usize = 10;

#[allow(dead_code)]
pub fn param_ids() -> [&'static str; 14] {
    [
        "stem_w", "stem_b", "b1a_w", "b1a_b", "b1b_w", "b1b_b", "b2a_w", "b2a_b", "b2b_w", "b2b_b",
        "skip_w", "skip_b", "fc_w", "fc_b",
    ]
}

fn relu_block(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    b: NodeId,
    batch: usize,
    c: usize,
    h: usize,
    w_sp: usize,
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> NodeId {
    let f = DType::F32;
    let h_out = (h + 2 * padding[0] - kernel[0]) / stride[0] + 1;
    let w_out = (w_sp + 2 * padding[1] - kernel[1]) / stride[1] + 1;
    let y = conv2d(g, x, w, batch, c, h_out, w_out, kernel, stride, padding);
    let y = bias_add_4d(g, y, b, batch, c, h_out, w_out);
    g.activation(
        Activation::Relu,
        y,
        Shape::new(&[batch, c, h_out, w_out], f),
    )
}

fn body(g: &mut Graph, batch: usize) -> (NodeId, NodeId, [NodeId; 14]) {
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[batch, 3, 32, 32], f));
    let labels = g.input("labels", Shape::new(&[batch], f));

    let stem_w = g.param("stem_w", Shape::new(&[32, 3, 3, 3], f));
    let stem_b = g.param("stem_b", Shape::new(&[32], f));
    let b1a_w = g.param("b1a_w", Shape::new(&[32, 32, 3, 3], f));
    let b1a_b = g.param("b1a_b", Shape::new(&[32], f));
    let b1b_w = g.param("b1b_w", Shape::new(&[32, 32, 3, 3], f));
    let b1b_b = g.param("b1b_b", Shape::new(&[32], f));
    let b2a_w = g.param("b2a_w", Shape::new(&[64, 32, 3, 3], f));
    let b2a_b = g.param("b2a_b", Shape::new(&[64], f));
    let b2b_w = g.param("b2b_w", Shape::new(&[64, 64, 3, 3], f));
    let b2b_b = g.param("b2b_b", Shape::new(&[64], f));
    let skip_w = g.param("skip_w", Shape::new(&[64, 32, 1, 1], f));
    let skip_b = g.param("skip_b", Shape::new(&[64], f));
    let fc_w = g.param("fc_w", Shape::new(&[64, 10], f));
    let fc_b = g.param("fc_b", Shape::new(&[10], f));
    let params = [
        stem_w, stem_b, b1a_w, b1a_b, b1b_w, b1b_b, b2a_w, b2a_b, b2b_w, b2b_b, skip_w, skip_b,
        fc_w, fc_b,
    ];

    let stem = relu_block(
        g,
        x,
        stem_w,
        stem_b,
        batch,
        32,
        32,
        32,
        [3, 3],
        [1, 1],
        [1, 1],
    );

    let b1a = relu_block(
        g,
        stem,
        b1a_w,
        b1a_b,
        batch,
        32,
        32,
        32,
        [3, 3],
        [1, 1],
        [1, 1],
    );
    let b1b = conv2d(g, b1a, b1b_w, batch, 32, 32, 32, [3, 3], [1, 1], [1, 1]);
    let b1b = bias_add_4d(g, b1b, b1b_b, batch, 32, 32, 32);
    let res1 = g.binary(
        BinaryOp::Add,
        b1b,
        stem,
        Shape::new(&[batch, 32, 32, 32], f),
    );
    let res1 = g.activation(Activation::Relu, res1, Shape::new(&[batch, 32, 32, 32], f));

    let b2a = relu_block(
        g,
        res1,
        b2a_w,
        b2a_b,
        batch,
        64,
        32,
        32,
        [3, 3],
        [2, 2],
        [1, 1],
    );
    let b2b = conv2d(g, b2a, b2b_w, batch, 64, 16, 16, [3, 3], [1, 1], [1, 1]);
    let b2b = bias_add_4d(g, b2b, b2b_b, batch, 64, 16, 16);
    let skip = conv2d(g, res1, skip_w, batch, 64, 16, 16, [1, 1], [2, 2], [0, 0]);
    let skip = bias_add_4d(g, skip, skip_b, batch, 64, 16, 16);
    let res2 = g.binary(
        BinaryOp::Add,
        b2b,
        skip,
        Shape::new(&[batch, 64, 16, 16], f),
    );
    let res2 = g.activation(Activation::Relu, res2, Shape::new(&[batch, 64, 16, 16], f));

    let pooled = avgpool(g, res2, batch, 64, [16, 16], 1, 1);
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, 64],
        },
        vec![pooled],
        Shape::new(&[batch, 64], f),
    );
    let mm = g.matmul(flat, fc_w, Shape::new(&[batch, 10], f));
    let logits = g.binary(BinaryOp::Add, mm, fc_b, Shape::new(&[batch, 10], f));

    (logits, labels, params)
}

pub fn build_forward(batch: usize) -> Graph {
    let mut g = Graph::new("resnet_fwd");
    let (logits, _, _) = body(&mut g, batch);
    g.set_outputs(vec![logits]);
    g
}

pub fn build_loss(batch: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("resnet_loss");
    let (logits, labels, params) = body(&mut g, batch);
    let f = DType::F32;
    let loss_per = g.softmax_cross_entropy_with_logits(logits, labels);
    let loss = g.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0],
            keep_dim: false,
        },
        vec![loss_per],
        Shape::from_dims(&[], f),
    );
    g.set_outputs(vec![loss]);
    (g, params.to_vec())
}
