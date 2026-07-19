// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! Small three-block CNN for CIFAR-10 (32×32 RGB).

use rlx_ir::op::*;
use rlx_ir::{DType, Graph, NodeId, Shape};

use super::conv::{bias_add_4d, conv2d, maxpool2x2};

pub const INPUT: &[usize] = &[3, 32, 32];
pub const NUM_CLASSES: usize = 10;

#[allow(dead_code)]
pub fn param_ids() -> [&'static str; 8] {
    [
        "conv1_w", "conv1_b", "conv2_w", "conv2_b", "conv3_w", "conv3_b", "fc_w", "fc_b",
    ]
}

fn body(g: &mut Graph, batch: usize) -> (NodeId, NodeId, [NodeId; 8]) {
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[batch, 3, 32, 32], f));
    let labels = g.input("labels", Shape::new(&[batch], f));

    let conv1_w = g.param("conv1_w", Shape::new(&[32, 3, 3, 3], f));
    let conv1_b = g.param("conv1_b", Shape::new(&[32], f));
    let conv2_w = g.param("conv2_w", Shape::new(&[64, 32, 3, 3], f));
    let conv2_b = g.param("conv2_b", Shape::new(&[64], f));
    let conv3_w = g.param("conv3_w", Shape::new(&[128, 64, 3, 3], f));
    let conv3_b = g.param("conv3_b", Shape::new(&[128], f));
    let fc_w = g.param("fc_w", Shape::new(&[2048, 10], f));
    let fc_b = g.param("fc_b", Shape::new(&[10], f));
    let params = [
        conv1_w, conv1_b, conv2_w, conv2_b, conv3_w, conv3_b, fc_w, fc_b,
    ];

    let c1 = conv2d(g, x, conv1_w, batch, 32, 32, 32, [3, 3], [1, 1], [1, 1]);
    let c1 = bias_add_4d(g, c1, conv1_b, batch, 32, 32, 32);
    let c1 = g.activation(Activation::Relu, c1, Shape::new(&[batch, 32, 32, 32], f));
    let p1 = maxpool2x2(g, c1, batch, 32, 16, 16);

    let c2 = conv2d(g, p1, conv2_w, batch, 64, 16, 16, [3, 3], [1, 1], [1, 1]);
    let c2 = bias_add_4d(g, c2, conv2_b, batch, 64, 16, 16);
    let c2 = g.activation(Activation::Relu, c2, Shape::new(&[batch, 64, 16, 16], f));
    let p2 = maxpool2x2(g, c2, batch, 64, 8, 8);

    let c3 = conv2d(g, p2, conv3_w, batch, 128, 8, 8, [3, 3], [1, 1], [1, 1]);
    let c3 = bias_add_4d(g, c3, conv3_b, batch, 128, 8, 8);
    let c3 = g.activation(Activation::Relu, c3, Shape::new(&[batch, 128, 8, 8], f));
    let p3 = maxpool2x2(g, c3, batch, 128, 4, 4);

    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, 2048],
        },
        vec![p3],
        Shape::new(&[batch, 2048], f),
    );
    let mm = g.matmul(flat, fc_w, Shape::new(&[batch, 10], f));
    let logits = g.binary(BinaryOp::Add, mm, fc_b, Shape::new(&[batch, 10], f));

    (logits, labels, params)
}

pub fn build_forward(batch: usize) -> Graph {
    let mut g = Graph::new("cifar_cnn_fwd");
    let (logits, _, _) = body(&mut g, batch);
    g.set_outputs(vec![logits]);
    g
}

pub fn build_loss(batch: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("cifar_cnn_loss");
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
