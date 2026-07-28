// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flattened MNIST MLP (784 → 128 → 10) — conv-free vision baseline.

use rlx_ir::op::*;
use rlx_ir::{DType, Graph, NodeId, Shape};

pub const INPUT: &[usize] = &[784];
pub const HIDDEN: usize = 128;
pub const NUM_CLASSES: usize = 10;

#[allow(dead_code)]
pub fn param_ids() -> [&'static str; 4] {
    ["w1", "b1", "w2", "b2"]
}

fn body(g: &mut Graph, batch: usize) -> (NodeId, NodeId, [NodeId; 4]) {
    let f = DType::F32;
    let h = HIDDEN;
    let x = g.input("x", Shape::new(&[batch, 784], f));
    let labels = g.input("labels", Shape::new(&[batch], f));

    let w1 = g.param("w1", Shape::new(&[784, h], f));
    let b1 = g.param("b1", Shape::new(&[h], f));
    let w2 = g.param("w2", Shape::new(&[h, 10], f));
    let b2 = g.param("b2", Shape::new(&[10], f));
    let params = [w1, b1, w2, b2];

    let z = g.matmul(x, w1, Shape::new(&[batch, h], f));
    let z = g.binary(BinaryOp::Add, z, b1, Shape::new(&[batch, h], f));
    let a = g.activation(Activation::Relu, z, Shape::new(&[batch, h], f));
    let mm = g.matmul(a, w2, Shape::new(&[batch, 10], f));
    let logits = g.binary(BinaryOp::Add, mm, b2, Shape::new(&[batch, 10], f));

    (logits, labels, params)
}

pub fn build_forward(batch: usize) -> Graph {
    let mut g = Graph::new("mnist_mlp_fwd");
    let (logits, _, _) = body(&mut g, batch);
    g.set_outputs(vec![logits]);
    g
}

pub fn build_loss(batch: usize) -> (Graph, Vec<NodeId>) {
    let mut g = Graph::new("mnist_mlp_loss");
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
