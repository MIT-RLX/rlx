// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! The adapters as rlx graph ops, so they run on any backend.
//!
//! The host-side functions in [`crate::adapters`] are the definition and the
//! parity reference. This module expresses the same update as a graph, which is
//! what you want when the adapter sits inside a model being executed on an
//! accelerator: materialising `ΔW` on the host and copying it back per step
//! would dominate the cost of a method whose whole point is being cheap.
//!
//! Only the adapters that are pure tensor algebra are here. OFT needs a matrix
//! inverse per block and stays host-side ([`crate::oft`]) — its blocks are tiny
//! and the inverse is not worth a graph round-trip.

use rlx_ir::infer::GraphExt;
use rlx_ir::{DType, Graph, NodeId, Shape};

use crate::LoraConfig;

/// `ΔW = (α/r)·B A` as a graph fragment, given existing `A`/`B` nodes.
///
/// Returns the `[out, in]` delta node so a caller can add it to a weight that
/// is already in the graph, rather than round-tripping through the host.
pub fn lora_delta_node(
    g: &mut Graph,
    a: NodeId,
    b: NodeId,
    in_features: usize,
    out_features: usize,
    cfg: LoraConfig,
) -> NodeId {
    let _ = (in_features, out_features);
    let prod = g.mm(b, a);
    // `full`, not `constant`: a scalar constant feeding a matmul consumer has
    // been folded into a biased matmul upstream, which silently zeroes all but
    // the first element.
    let s = g.full(&[1, 1], cfg.scaling() as f32, DType::F32);
    g.mul(prod, s)
}

/// A standalone graph computing `ΔW` from `A` and `B` inputs.
pub fn lora_delta_graph(in_features: usize, out_features: usize, cfg: LoraConfig) -> Graph {
    let mut g = Graph::new("lora_delta");
    let a = g.input("a", Shape::new(&[cfg.r, in_features], DType::F32));
    let b = g.input("b", Shape::new(&[out_features, cfg.r], DType::F32));
    let d = lora_delta_node(&mut g, a, b, in_features, out_features, cfg);
    g.set_outputs(vec![d]);
    g
}

/// IA3 as a graph: rescale a `[rows, out]` activation by a learned `[1, out]`.
pub fn ia3_graph(rows: usize, out_features: usize) -> Graph {
    let mut g = Graph::new("ia3");
    let y = g.input("y", Shape::new(&[rows, out_features], DType::F32));
    let l = g.input("l", Shape::new(&[1, out_features], DType::F32));
    let out = g.mul(y, l);
    g.set_outputs(vec![out]);
    g
}

/// AdaLoRA: `ΔW = (α/r)·P diag(Λ) Q`, with `Λ` as a `[1, r]` row so the scaling
/// is a broadcast multiply rather than a materialised diagonal matrix.
pub fn adalora_delta_graph(in_features: usize, out_features: usize, cfg: LoraConfig) -> Graph {
    let mut g = Graph::new("adalora_delta");
    let p = g.input("p", Shape::new(&[out_features, cfg.r], DType::F32));
    let lam = g.input("lambda", Shape::new(&[1, cfg.r], DType::F32));
    let q = g.input("q", Shape::new(&[cfg.r, in_features], DType::F32));
    let pl = g.mul(p, lam);
    let prod = g.mm(pl, q);
    let s = g.full(&[1, 1], cfg.scaling() as f32, DType::F32);
    let d = g.mul(prod, s);
    g.set_outputs(vec![d]);
    g
}

/// DoRA: `W' = m ⊙ (W + ΔW) / ‖W + ΔW‖_row`.
///
/// The norm is per OUTPUT ROW, which is `sum` over axis 1 with `keep_dim` —
/// reducing the other axis produces a model that trains and is not DoRA.
pub fn dora_graph(in_features: usize, out_features: usize) -> Graph {
    let mut g = Graph::new("dora");
    let w = g.input("w", Shape::new(&[out_features, in_features], DType::F32));
    let d = g.input(
        "delta",
        Shape::new(&[out_features, in_features], DType::F32),
    );
    let m = g.input("m", Shape::new(&[out_features, 1], DType::F32));

    let v = g.add(w, d);
    let sq = g.mul(v, v);
    let ss = g.sum(sq, vec![1], true);
    let norm = g.sqrt(ss);
    let dir = g.div(v, norm);
    let out = g.mul(dir, m);
    g.set_outputs(vec![out]);
    g
}
