// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! TinyConv MNIST — matches the rlx-cortexm / CoreML MNIST runner.

use rlx_ir::op::*;
use rlx_ir::{DType, Graph, NodeId, Shape};

use super::conv::{bias_add_4d, conv2d, maxpool2x2};

pub const INPUT: &[usize] = &[1, 28, 28];
pub const NUM_CLASSES: usize = 10;

#[allow(dead_code)]
pub fn param_ids() -> [&'static str; 6] {
    ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"]
}

fn body(g: &mut Graph, batch: usize) -> (NodeId, NodeId, [NodeId; 6]) {
    let f = DType::F32;
    let x = g.input("x", Shape::new(&[batch, 1, 28, 28], f));
    let labels = g.input("labels", Shape::new(&[batch], f));

    let conv1_w = g.param("conv1_w", Shape::new(&[8, 1, 3, 3], f));
    let conv1_b = g.param("conv1_b", Shape::new(&[8], f));
    let conv2_w = g.param("conv2_w", Shape::new(&[16, 8, 3, 3], f));
    let conv2_b = g.param("conv2_b", Shape::new(&[16], f));
    let fc_w = g.param("fc_w", Shape::new(&[400, 10], f));
    let fc_b = g.param("fc_b", Shape::new(&[10], f));
    let params = [conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b];

    let c1 = conv2d(g, x, conv1_w, batch, 8, 26, 26, [3, 3], [1, 1], [0, 0]);
    let c1 = bias_add_4d(g, c1, conv1_b, batch, 8, 26, 26);
    let c1 = g.activation(Activation::Relu, c1, Shape::new(&[batch, 8, 26, 26], f));
    let p1 = maxpool2x2(g, c1, batch, 8, 13, 13);

    let c2 = conv2d(g, p1, conv2_w, batch, 16, 11, 11, [3, 3], [1, 1], [0, 0]);
    let c2 = bias_add_4d(g, c2, conv2_b, batch, 16, 11, 11);
    let c2 = g.activation(Activation::Relu, c2, Shape::new(&[batch, 16, 11, 11], f));
    let p2 = maxpool2x2(g, c2, batch, 16, 5, 5);

    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, 400],
        },
        vec![p2],
        Shape::new(&[batch, 400], f),
    );
    let mm = g.matmul(flat, fc_w, Shape::new(&[batch, 10], f));
    let logits = g.binary(BinaryOp::Add, mm, fc_b, Shape::new(&[batch, 10], f));

    (logits, labels, params)
}

pub fn build_forward(batch: usize) -> Graph {
    let mut g = Graph::new("mnist_cnn_fwd");
    let (logits, _, _) = body(&mut g, batch);
    g.set_outputs(vec![logits]);
    g
}

pub fn build_loss(batch: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("mnist_cnn_loss");
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
