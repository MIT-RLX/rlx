// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Shared TinyConv-MNIST forward graph (matches rlx-cortexm-trainer layout).

use rlx_ir::op::*;
use rlx_ir::*;

pub struct TinyConvForward {
    pub graph: Graph,
    pub params: Vec<NodeId>,
    pub input: NodeId,
    pub labels: NodeId,
    #[allow(dead_code)]
    pub loss: NodeId,
}

pub fn build_tinyconv_forward(batch: usize) -> TinyConvForward {
    let f = DType::F32;
    let mut g = Graph::new("tinyconv_fwd");

    let x = g.input("x", Shape::new(&[batch, 1, 28, 28], f));
    let labels = g.input("labels", Shape::new(&[batch], f));

    let conv1_w = g.param("conv1_w", Shape::new(&[8, 1, 3, 3], f));
    let conv1_b = g.param("conv1_b", Shape::new(&[8], f));
    let conv2_w = g.param("conv2_w", Shape::new(&[16, 8, 3, 3], f));
    let conv2_b = g.param("conv2_b", Shape::new(&[16], f));
    let fc_w = g.param("fc_w", Shape::new(&[400, 10], f));
    let fc_b = g.param("fc_b", Shape::new(&[10], f));

    let c1 = conv2d(&mut g, x, conv1_w, batch, 8, 28, 28, 26, 26);
    let c1 = bias_add_4d(&mut g, c1, conv1_b, batch, 8, 26, 26);
    let c1 = g.activation(Activation::Relu, c1, Shape::new(&[batch, 8, 26, 26], f));
    let p1 = maxpool(&mut g, c1, batch, 8, 13, 13);

    let c2 = conv2d(&mut g, p1, conv2_w, batch, 16, 13, 13, 11, 11);
    let c2 = bias_add_4d(&mut g, c2, conv2_b, batch, 16, 11, 11);
    let c2 = g.activation(Activation::Relu, c2, Shape::new(&[batch, 16, 11, 11], f));
    let p2 = maxpool(&mut g, c2, batch, 16, 5, 5);

    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, 400],
        },
        vec![p2],
        Shape::new(&[batch, 400], f),
    );
    let mm = g.matmul(flat, fc_w, Shape::new(&[batch, 10], f));
    let logits = g.binary(BinaryOp::Add, mm, fc_b, Shape::new(&[batch, 10], f));

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

    let params = vec![conv1_w, conv1_b, conv2_w, conv2_b, fc_w, fc_b];
    TinyConvForward {
        graph: g,
        params,
        input: x,
        labels,
        loss,
    }
}

fn conv2d(
    g: &mut Graph,
    x: NodeId,
    w: NodeId,
    b: usize,
    c_out: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> NodeId {
    let _ = (h_in, w_in);
    g.add_node(
        Op::Conv {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![x, w],
        Shape::new(&[b, c_out, h_out, w_out], DType::F32),
    )
}

fn maxpool(g: &mut Graph, x: NodeId, b: usize, c: usize, h_out: usize, w_out: usize) -> NodeId {
    g.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![x],
        Shape::new(&[b, c, h_out, w_out], DType::F32),
    )
}

fn bias_add_4d(
    g: &mut Graph,
    x: NodeId,
    bias: NodeId,
    b: usize,
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
    g.binary(BinaryOp::Add, x, bias_4d, Shape::new(&[b, c, h, w], f))
}
