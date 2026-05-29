// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Sparse UMAP cross-entropy loss graph (attraction + negative sampling).

use rlx_autodiff::grad_with_loss;
use rlx_compile::legalize_broadcast;
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use super::mlp::{ModelSpec, ParamSlot};

#[derive(Debug, Clone)]
pub struct UmapTrainGraph {
    pub forward: Graph,
    pub backward: Graph,
    pub d_output: NodeId,
    pub loss: NodeId,
    pub params: Vec<ParamSlot>,
    pub x: NodeId,
    pub edge_h: NodeId,
    pub edge_t: NodeId,
    pub n_edges: usize,
    pub n_pos: usize,
}

fn scalar_const(g: &mut Graph, v: f32) -> NodeId {
    g.add_node(
        Op::Constant {
            data: v.to_le_bytes().to_vec(),
        },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

fn unary_log(g: &mut Graph, x: NodeId) -> NodeId {
    let s = rlx_ir::shape::unary_shape(g.shape(x));
    g.activation(Activation::Log, x, s)
}

fn clamp_min(g: &mut Graph, x: NodeId, eps: f32, eps_node: NodeId) -> NodeId {
    let s = rlx_ir::shape::binary_shape(g.shape(x), g.shape(eps_node)).unwrap();
    g.binary(BinaryOp::Max, x, eps_node, s)
}

/// Build forward (scalar loss) + backward graphs for a fixed edge batch layout.
pub fn build_train_graph(spec: &ModelSpec, n_pos: usize, n_neg: usize) -> UmapTrainGraph {
    let n_edges = n_pos + n_neg;
    let f = DType::F32;
    let d_out = spec.output_dim;

    let mut g = Graph::new("umap_train");
    let x = g.input("x", Shape::new(&[spec.n, spec.input_dim], f));
    let edge_h = g.input("edge_h", Shape::new(&[n_edges], f));
    let edge_t = g.input("edge_t", Shape::new(&[n_edges], f));
    let kernel_a_in = g.input("kernel_a", Shape::new(&[1], f));
    let kernel_b_in = g.input("kernel_b", Shape::new(&[1], f));
    let repulsion_in = g.input("repulsion", Shape::new(&[1], f));

    let one = scalar_const(&mut g, 1.0);
    let eps = scalar_const(&mut g, 1e-8);
    let eps6 = scalar_const(&mut g, 1e-6);

    let mut params = Vec::new();
    let mut h = x;
    let mut in_d = spec.input_dim;
    for (li, &hd) in spec.hidden.iter().enumerate() {
        let w_name = format!("umap_w{li}");
        let b_name = format!("umap_b{li}");
        let w = g.param(&w_name, Shape::new(&[in_d, hd], f));
        let b = g.param(&b_name, Shape::new(&[hd], f));
        params.push(ParamSlot {
            name: w_name,
            param: w,
            grad: None,
        });
        params.push(ParamSlot {
            name: b_name,
            param: b,
            grad: None,
        });
        let mm = g.mm(h, w);
        let lin = g.add(mm, b);
        h = g.relu(lin);
        in_d = hd;
    }
    let w = g.param("umap_w_out", Shape::new(&[in_d, d_out], f));
    let b = g.param("umap_b_out", Shape::new(&[d_out], f));
    params.push(ParamSlot {
        name: "umap_w_out".into(),
        param: w,
        grad: None,
    });
    params.push(ParamSlot {
        name: "umap_b_out".into(),
        param: b,
        grad: None,
    });
    let mm_out = g.mm(h, w);
    let embed = g.add(mm_out, b);

    let head = g.gather_(embed, edge_h, 0);
    let tail = g.gather_(embed, edge_t, 0);
    let diff = g.sub(head, tail);
    let sq = g.mul(diff, diff);
    let dist_sq = g.sum(sq, vec![1], false);

    let dist_pos = g.narrow_(dist_sq, 0, 0, n_pos);
    let dist_neg = g.narrow_(dist_sq, 0, n_pos, n_neg);

    let dist_pos_c = clamp_min(&mut g, dist_pos, 1e-8, eps);
    let dist_neg_c = clamp_min(&mut g, dist_neg, 1e-8, eps);

    let dist_pow_pos = {
        let s = rlx_ir::shape::binary_shape(g.shape(dist_pos_c), g.shape(kernel_b_in)).unwrap();
        g.binary(BinaryOp::Pow, dist_pos_c, kernel_b_in, s)
    };
    let a_d_pos = g.mul(dist_pow_pos, kernel_a_in);
    let denom_pos = g.add(a_d_pos, one);
    let q_pos = g.div(one, denom_pos);
    let q_pos_clamped = clamp_min(&mut g, q_pos, 1e-6, eps6);
    let log_q_pos = unary_log(&mut g, q_pos_clamped);
    let attract = g.neg(log_q_pos);
    let attraction = g.mean(attract, vec![0], false);

    let dist_pow_neg = {
        let s = rlx_ir::shape::binary_shape(g.shape(dist_neg_c), g.shape(kernel_b_in)).unwrap();
        g.binary(BinaryOp::Pow, dist_neg_c, kernel_b_in, s)
    };
    let a_d_neg = g.mul(dist_pow_neg, kernel_a_in);
    let denom_neg = g.add(a_d_neg, one);
    let one_minus_q = g.div(a_d_neg, denom_neg);
    let omq_clamped = clamp_min(&mut g, one_minus_q, 1e-6, eps6);
    let log_omq = unary_log(&mut g, omq_clamped);
    let repulse = g.neg(log_omq);
    let repulsion = g.mean(repulse, vec![0], false);

    let scaled_rep = g.mul(repulsion_in, repulsion);
    let loss = g.add(attraction, scaled_rep);
    g.set_outputs(vec![loss]);
    let loss_node = loss;

    let (g, remap) = legalize_broadcast::run_with_remap(g);
    let params: Vec<ParamSlot> = params
        .into_iter()
        .map(|mut p| {
            p.param = remap[&p.param];
            p
        })
        .collect();
    let loss = remap[&loss_node];
    let wrt: Vec<NodeId> = params.iter().map(|p| p.param).collect();
    let bwd = grad_with_loss(&g, &wrt);
    let d_output = bwd
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output");
    let grad_ids: Vec<NodeId> = bwd.outputs[1..=params.len()].to_vec();
    let params: Vec<ParamSlot> = params
        .into_iter()
        .zip(grad_ids)
        .map(|(mut p, grad)| {
            p.grad = Some(grad);
            p
        })
        .collect();

    UmapTrainGraph {
        forward: g,
        backward: bwd,
        d_output,
        loss,
        params,
        x: remap[&x],
        edge_h: remap[&edge_h],
        edge_t: remap[&edge_t],
        n_edges,
        n_pos,
    }
}
