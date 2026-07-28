// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared two-layer MLP graph builder.
//!
//! The same forward/loss graphs feed every backend (CPU, WebGPU, WebGL):
//! only the executor differs. `build_loss` produces a scalar loss as the
//! first output so it can be handed straight to `rlx_autodiff::grad_with_loss`.

use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};

#[derive(Copy, Clone)]
pub struct MlpDims {
    pub in_dim: usize,
    pub hidden: usize,
    pub out_dim: usize,
}

/// Parameter node ids, in order `[w1, b1, w2, b2]`.
pub type Params = [NodeId; 4];

/// `relu(x·W1 + b1)·W2 + b2`. Returns the output node and the param ids.
fn body(g: &mut Graph, d: MlpDims) -> (NodeId, Params) {
    let x = g.input("x", Shape::new(&[1, d.in_dim], DType::F32));
    let w1 = g.param("w1", Shape::new(&[d.in_dim, d.hidden], DType::F32));
    // Biases as a row [1, n] so the add is exact-shape with batch size 1.
    let b1 = g.param("b1", Shape::new(&[1, d.hidden], DType::F32));
    let w2 = g.param("w2", Shape::new(&[d.hidden, d.out_dim], DType::F32));
    let b2 = g.param("b2", Shape::new(&[1, d.out_dim], DType::F32));

    let h = g.matmul(x, w1, Shape::new(&[1, d.hidden], DType::F32));
    let h = g.add(h, b1);
    let h = g.relu(h);
    let y = g.matmul(h, w2, Shape::new(&[1, d.out_dim], DType::F32));
    let y = g.add(y, b2);
    (y, [w1, b1, w2, b2])
}

/// Forward graph: single output `y` (inference).
pub fn build_forward(d: MlpDims) -> (Graph, Params) {
    let mut g = Graph::new("rlx_web_mlp");
    let (y, params) = body(&mut g, d);
    g.set_outputs(vec![y]);
    (g, params)
}

/// Loss graph: single scalar output `Σ (y − target)²` (the value the
/// backward pass differentiates). Adds a `target` input.
pub fn build_loss(d: MlpDims) -> (Graph, Params) {
    let mut g = Graph::new("rlx_web_mlp_loss");
    let (y, params) = body(&mut g, d);
    let target = g.input("target", Shape::new(&[1, d.out_dim], DType::F32));
    let diff = g.sub(y, target);
    let sq = g.mul(diff, diff);
    let loss = g.sum(sq, vec![0, 1], false);
    g.set_outputs(vec![loss]);
    (g, params)
}
