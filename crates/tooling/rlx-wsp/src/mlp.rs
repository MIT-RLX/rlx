// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The programmable network: a plain MLP built as an rlx graph.
//!
//! Layer `l` is `h_l = act(h_{l-1} @ l{l}.w + l{l}.b)` with `l{l}.w` laid out
//! `[fan_in, fan_out]` (row = input unit, column = output unit) and `l{l}.b`
//! `[fan_out]`; the last layer has no activation and produces action logits.
//! Every hidden activation is an extra graph output, so a probe returns the
//! internal units as well as the logits. Nodes are named for `trace`:
//! `l{k}.z` (pre-activation), `l{k}.h` (activation), `logits`.

use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Shape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Act {
    Tanh,
    Relu,
}

impl Act {
    fn op(self) -> Activation {
        match self {
            Act::Tanh => Activation::Tanh,
            Act::Relu => Activation::Relu,
        }
    }
}

/// Layer widths `[obs_dim, hidden.., n_actions]` plus the hidden activation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlpSpec {
    pub widths: Vec<usize>,
    pub act: Act,
}

impl MlpSpec {
    pub fn n_layers(&self) -> usize {
        self.widths.len() - 1
    }

    /// `(name, rows, cols)` for every param, weights before biases per layer.
    pub fn params(&self) -> Vec<(String, usize, usize)> {
        let mut out = Vec::with_capacity(2 * self.n_layers());
        for l in 0..self.n_layers() {
            let (fi, fo) = (self.widths[l], self.widths[l + 1]);
            out.push((format!("l{l}.w"), fi, fo));
            out.push((format!("l{l}.b"), 1, fo));
        }
        out
    }

    pub fn n_weights(&self) -> usize {
        self.params().iter().map(|(_, r, c)| r * c).sum()
    }

    /// Graph over a fixed `batch`; outputs are `[logits, h0, h1, ..]`.
    pub fn build_graph(&self, batch: usize) -> Graph {
        let sh = |d: &[usize]| Shape::new(d, DType::F32);
        let mut g = Graph::new("wsp_mlp");
        let mut x = g.input("obs", sh(&[batch, self.widths[0]]));
        let mut hidden = Vec::new();
        for l in 0..self.n_layers() {
            let (fi, fo) = (self.widths[l], self.widths[l + 1]);
            let w = g.param(format!("l{l}.w"), sh(&[fi, fo]));
            let b = g.param(format!("l{l}.b"), sh(&[fo]));
            let z = g.matmul(x, w, sh(&[batch, fo]));
            x = g.binary(BinaryOp::Add, z, b, sh(&[batch, fo]));
            if l + 1 < self.n_layers() {
                g.node_mut(x).name = Some(format!("l{l}.z"));
                x = g.activation(self.act.op(), x, sh(&[batch, fo]));
                g.node_mut(x).name = Some(format!("l{l}.h"));
                hidden.push(x);
            } else {
                g.node_mut(x).name = Some("logits".into());
            }
        }
        let mut outs = vec![x];
        outs.extend(hidden);
        g.set_outputs(outs);
        g
    }
}
