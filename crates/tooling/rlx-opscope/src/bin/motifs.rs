// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opscope-motifs` — mine recurring op-subsequences (fusion-kernel candidates)
//! from a graph. Demonstrated on a synthetic N-layer MLP whose repeated
//! `matmul → bias-add → relu` block should surface as the top motif.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};
use rlx_opscope::motifs::linear_op_motifs;

/// `layers` repeated blocks of `x = relu(x @ W + b)`, all `[B,D] · [D,D]`.
fn mlp_stack(layers: usize, b: usize, d: usize) -> Graph {
    let mut g = Graph::new("mlp_stack");
    let mut x = g.input("x", Shape::new(&[b, d], DType::F32));
    let hw = Shape::new(&[b, d], DType::F32);
    for i in 0..layers {
        let w = g.param(format!("W{i}"), Shape::new(&[d, d], DType::F32));
        let bias = g.param(format!("b{i}"), Shape::new(&[d], DType::F32));
        let h = g.matmul(x, w, hw.clone());
        let h = g.add_node(Op::Binary(BinaryOp::Add), vec![h, bias], hw.clone());
        x = g.activation(Activation::Relu, h, hw.clone());
    }
    g.set_outputs(vec![x]);
    g
}

fn main() {
    let layers = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(6usize);
    let g = mlp_stack(layers, 32, 64);
    println!(
        "graph: {} nodes, {} layers of (matmul → bias-add → relu)\n",
        g.nodes().len(),
        layers
    );

    let motifs = linear_op_motifs(&g, 2, 6, 2);
    println!(
        "{:>5} {:>4}   recurring op-subsequence (fusion candidate)",
        "score", "×"
    );
    println!("{}", "-".repeat(72));
    for m in motifs.iter().take(12) {
        println!("{:>5} {:>4}   {}", m.score(), m.count, m.seq.join(" → "));
    }
}
