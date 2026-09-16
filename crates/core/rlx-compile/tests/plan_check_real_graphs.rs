// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The plan gate against graphs shaped like real models, in every planner
//! configuration the backends actually use.
//!
//! A gate that only ever sees a five-node toy proves nothing about the graphs
//! it will run on. False positives here are the failure that matters: a gate
//! that flags legitimate slot reuse gets disabled, and then it protects
//! nothing.

use rlx_compile::memory::{
    MemoryPlanOptions, plan_memory, plan_memory_aligned, plan_memory_f32_uniform,
    plan_memory_native, plan_memory_with_options,
};
use rlx_compile::plan_check::check_plan;
use rlx_ir::{DType, Graph, GraphExt, Shape};

const F: DType = DType::F32;

/// A decode-shaped transformer stack: attention projections, a gated MLP with
/// a real branch, and residual adds. Branching plus short-lived intermediates
/// is what produces interesting arena reuse.
fn transformer(layers: usize, d: usize, ff: usize, seq: usize) -> Graph {
    let mut g = Graph::new("xf");
    let hs = Shape::new(&[seq, d], F);
    let mut h = g.input("x", hs.clone());
    for l in 0..layers {
        let gamma = g.param(format!("l{l}.g").as_str(), Shape::new(&[d], F));
        let beta = g.param(format!("l{l}.b").as_str(), Shape::new(&[d], F));
        let normed = g.add_node(
            rlx_ir::Op::RmsNorm {
                axis: -1,
                eps: 1e-6,
            },
            vec![h, gamma, beta],
            hs.clone(),
        );
        let mut attn = normed;
        for name in ["q", "k", "v", "o"] {
            let w = g.param(format!("l{l}.{name}").as_str(), Shape::new(&[d, d], F));
            attn = g.matmul(attn, w, hs.clone());
        }
        h = g.add(h, attn);

        let gw = g.param(format!("l{l}.gate").as_str(), Shape::new(&[d, ff], F));
        let uw = g.param(format!("l{l}.up").as_str(), Shape::new(&[d, ff], F));
        let fs = Shape::new(&[seq, ff], F);
        let gate = g.matmul(h, gw, fs.clone());
        let up = g.matmul(h, uw, fs.clone());
        let act = g.mul(gate, up);
        let dw = g.param(format!("l{l}.down").as_str(), Shape::new(&[ff, d], F));
        let mlp = g.matmul(act, dw, hs.clone());
        h = g.add(h, mlp);
    }
    g.set_outputs(vec![h]);
    g
}

fn in_order_opts() -> MemoryPlanOptions {
    let mut o = MemoryPlanOptions::inference();
    o.pin_output_ancestors = false;
    o.dequant_host_fallback = false;
    o
}

#[test]
fn every_planner_configuration_produces_a_safe_plan() {
    let mut reuse_seen = false;
    for (layers, d, ff, seq) in [(1, 64, 128, 1), (4, 128, 256, 1), (2, 64, 128, 32)] {
        let g = transformer(layers, d, ff, seq);
        let plans: Vec<(&str, _)> = vec![
            ("default", plan_memory(&g)),
            ("aligned256", plan_memory_aligned(&g, 256)),
            ("native", plan_memory_native(&g, 64)),
            ("f32_uniform", plan_memory_f32_uniform(&g, 64)),
            (
                "in_order",
                plan_memory_with_options(&g, 64, in_order_opts()),
            ),
        ];
        for (label, plan) in plans {
            if plan.bytes_saved() > 0 {
                reuse_seen = true;
            }
            let report = check_plan(&g, &plan);
            assert!(
                report.is_clean(),
                "{label} plan for {layers}L d={d} ff={ff} seq={seq} flagged:\n{}",
                report.render()
            );
            assert!(
                report.checked_nodes > 0,
                "{label}: gate inspected nothing, so 'clean' is vacuous"
            );
        }
    }
    assert!(
        reuse_seen,
        "no configuration exercised slot reuse — the overlap logic never ran"
    );
}

#[test]
fn the_gate_scales_to_a_realistic_layer_count() {
    // The overlap scan is pairwise; this pins that a model-sized graph is
    // still checkable rather than quadratically hopeless.
    let g = transformer(28, 512, 1024, 1);
    let plan = plan_memory_with_options(&g, 64, in_order_opts());
    let report = check_plan(&g, &plan);
    assert!(report.is_clean(), "{}", report.render());
    assert!(
        report.checked_nodes > 100,
        "expected a large plan, got {}",
        report.checked_nodes
    );
}
