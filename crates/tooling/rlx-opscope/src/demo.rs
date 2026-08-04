// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small illustrative graph builders (`mlp` / `transformer` / `moe`) shared by
//! the structural/shape demos. Shapes are representative; the structural and
//! shape miners read op kinds + dims, they don't execute these.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

pub const S: usize = 16; // seq
pub const D: usize = 64; // model dim

pub fn sh(dims: &[usize]) -> Shape {
    Shape::new(dims, DType::F32)
}

pub fn residual_block(g: &mut Graph, x: NodeId, i: usize) -> NodeId {
    let (s, dd) = (sh(&[S, D]), sh(&[D, D]));
    let w = g.param(format!("W{i}"), dd);
    let b = g.param(format!("b{i}"), sh(&[D]));
    let h = g.matmul(x, w, s.clone());
    let h = g.add_node(Op::Binary(BinaryOp::Add), vec![h, b], s.clone());
    let h = g.activation(Activation::Relu, h, s.clone());
    g.add_node(Op::Binary(BinaryOp::Add), vec![x, h], s)
}

pub fn attention(g: &mut Graph, x: NodeId, i: usize) -> NodeId {
    let (s, dd) = (sh(&[S, D]), sh(&[D, D]));
    let wq = g.param(format!("wq{i}"), dd.clone());
    let wk = g.param(format!("wk{i}"), dd.clone());
    let wv = g.param(format!("wv{i}"), dd.clone());
    let wo = g.param(format!("wo{i}"), dd);
    let q = g.matmul(x, wq, s.clone());
    let k = g.matmul(x, wk, s.clone());
    let v = g.matmul(x, wv, s.clone());
    let scores = g.matmul(q, k, sh(&[S, S]));
    let p = g.softmax(scores, -1, sh(&[S, S]));
    let a = g.matmul(p, v, s.clone());
    let o = g.matmul(a, wo, s.clone());
    g.add_node(Op::Binary(BinaryOp::Add), vec![x, o], s) // residual
}

pub fn ffn(g: &mut Graph, x: NodeId, i: usize) -> NodeId {
    let (s, dd) = (sh(&[S, D]), sh(&[D, D]));
    let w1 = g.param(format!("f1{i}"), dd.clone());
    let w2 = g.param(format!("f2{i}"), dd);
    let h = g.matmul(x, w1, s.clone());
    let h = g.activation(Activation::Silu, h, s.clone());
    let y = g.matmul(h, w2, s.clone());
    g.add_node(Op::Binary(BinaryOp::Add), vec![x, y], s)
}

pub fn moe(g: &mut Graph, x: NodeId, i: usize) -> NodeId {
    let (s, e, k) = (sh(&[S, D]), 8usize, 2usize);
    let wg = g.param(format!("wg{i}"), sh(&[D, e]));
    let gate = g.matmul(x, wg, sh(&[S, e]));
    let idx = g.add_node(Op::TopK { k }, vec![gate], sh(&[S, k]));
    let experts = g.param(format!("ex{i}"), sh(&[e, D, D]));
    let y = g.add_node(Op::GroupedMatMul, vec![x, experts, idx], s.clone());
    g.add_node(Op::Binary(BinaryOp::Add), vec![x, y], s)
}

/// Build an `layers`-deep demo graph. `kind` ∈ {mlp, transformer, moe}.
pub fn build(kind: &str, layers: usize) -> Graph {
    let mut g = Graph::new(kind);
    let mut x = g.input("x", sh(&[S, D]));
    for i in 0..layers {
        x = match kind {
            "transformer" => {
                let a = attention(&mut g, x, i);
                ffn(&mut g, a, i)
            }
            "moe" => {
                let a = attention(&mut g, x, i);
                moe(&mut g, a, i)
            }
            _ => residual_block(&mut g, x, i),
        };
    }
    g.set_outputs(vec![x]);
    g
}
