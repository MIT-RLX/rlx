// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Megakernel composition, measured — a standalone study, not a feature.**
//!
//! CAKE §5.1 rewrites Alpha-MoE into a single fused MoE megakernel: "It fuses
//! routed gather, two projections, activation, requantization, and
//! route-weighted output accumulation into one device program... the reference
//! launches five GPU activities, whereas Alpha-MoE uses an output reset and one
//! megakernel." The API-level speedups (6.2x at N=256) come substantially from
//! removed scheduling gaps, not from better arithmetic.
//!
//! This crate asks the narrower question rlx can actually answer today: **how
//! many separate operations does an MoE block still take after rlx's fusion
//! pipeline has run, and does fusing change the answer it computes?**
//!
//! # Why dispatch count and not wall time
//!
//! Wall time on a contended machine is not a measurement — this tree produced
//! a 9x spread across repeated runs of an identical configuration, and a
//! confident table built on it that had to be retracted. Scheduled-op count is
//! exact, device-free, reproducible anywhere, and is the quantity CAKE itself
//! reports for this result. It is a *proxy* for launch overhead, and this crate
//! does not claim it is a speedup.
//!
//! # What it does not do
//!
//! It does not implement a megakernel. rlx has no schedule IR in which "one
//! device program" is expressible — roles, barriers and pipeline stages are not
//! declared, so there is nothing to fuse them into. This measures the gap
//! rather than closing it, which is the honest thing a standalone study can do.

use rlx_ir::{DType, Graph, GraphExt, Op, Shape};
use rlx_opt::rlx_compile::fusion_pipeline::{FusionTarget, fuse};

const F: DType = DType::F32;

/// Ops that never become a device dispatch, so counting them would overstate
/// what a backend actually launches.
fn is_dispatch(op: &Op) -> bool {
    !matches!(
        op,
        Op::Input { .. } | Op::Param { .. } | Op::Constant { .. } | Op::Reshape { .. }
    )
}

/// Number of ops in `graph` that would become a device dispatch.
pub fn dispatch_count(graph: &Graph) -> usize {
    graph.nodes().iter().filter(|n| is_dispatch(&n.op)).count()
}

/// Ops whose cost is arithmetic rather than data movement.
///
/// Raw dispatch count is a **bad** monotonicity metric on its own, and this
/// crate learned that the hard way: rlx fuses the two projections that share an
/// input into one `Concat` + `MatMul` + two `Narrow`s, taking a single-expert
/// block from 3 matmuls to 2 while raising the node count 6 -> 8. That is the
/// right trade and a raw count scores it as a regression.
pub fn heavy_count(graph: &Graph) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|n| {
            matches!(
                n.op,
                Op::MatMul
                    | Op::Attention { .. }
                    | Op::GroupedMatMul
                    | Op::DequantMatMul { .. }
                    | Op::Conv { .. }
            )
        })
        .count()
}

/// A dense MoE block, written out as the primitives a backend would launch:
/// per-expert gate/up projections, a sigmoid activation, the elementwise
/// product, the down projection, and a route-weighted accumulation.
///
/// Dense rather than sparse-routed on purpose: a top-k gather needs data
/// dependent indices, and the point here is to count the arithmetic dispatches
/// an expert costs, not to model routing.
pub fn moe_block(experts: usize, tokens: usize, d: usize, ff: usize) -> Graph {
    let mut g = Graph::new("moe");
    let hs = Shape::new(&[tokens, d], F);
    let fs = Shape::new(&[tokens, ff], F);
    let x = g.input("x", hs.clone());

    let mut acc: Option<rlx_ir::NodeId> = None;
    for e in 0..experts {
        let gw = g.param(format!("e{e}.gate").as_str(), Shape::new(&[d, ff], F));
        let uw = g.param(format!("e{e}.up").as_str(), Shape::new(&[d, ff], F));
        let dw = g.param(format!("e{e}.down").as_str(), Shape::new(&[ff, d], F));
        let w = g.param(format!("e{e}.weight").as_str(), Shape::new(&[tokens, d], F));

        let gate = g.matmul(x, gw, fs.clone());
        let up = g.matmul(x, uw, fs.clone());
        let act = g.add_node(
            Op::Activation(rlx_ir::op::Activation::Sigmoid),
            vec![gate],
            fs.clone(),
        );
        let prod = g.mul(act, up);
        let down = g.matmul(prod, dw, hs.clone());
        let weighted = g.mul(down, w);
        acc = Some(match acc {
            None => weighted,
            Some(a) => g.add(a, weighted),
        });
    }
    g.set_outputs(vec![acc.expect("at least one expert")]);
    g
}

/// What fusion achieved on one graph.
#[derive(Debug, Clone, Copy)]
pub struct Composition {
    pub before: usize,
    pub after: usize,
    /// Arithmetic-heavy ops before fusion.
    pub heavy_before: usize,
    /// Arithmetic-heavy ops after fusion. This is the count that must not grow.
    pub heavy_after: usize,
    pub target: FusionTarget,
}

impl Composition {
    /// Dispatches removed as a fraction of the original.
    pub fn reduction(&self) -> f64 {
        if self.before == 0 {
            return 0.0;
        }
        1.0 - self.after as f64 / self.before as f64
    }

    /// CAKE's framing: a megakernel is `after == 1`. Nothing in rlx reaches
    /// this today; the method exists so a report can say so precisely rather
    /// than implying progress toward it.
    pub fn is_megakernel(&self) -> bool {
        self.after <= 1
    }
}

/// Run rlx's fusion pipeline for `target` and report the dispatch delta.
pub fn compose(graph: &Graph, target: FusionTarget) -> Composition {
    let before = dispatch_count(graph);
    let heavy_before = heavy_count(graph);
    let fused = fuse(graph.clone(), target);
    Composition {
        before,
        after: dispatch_count(&fused),
        heavy_before,
        heavy_after: heavy_count(&fused),
        target,
    }
}

/// Fuse and return the graph, for callers that need to run it.
pub fn compose_graph(graph: &Graph, target: FusionTarget) -> Graph {
    fuse(graph.clone(), target)
}
