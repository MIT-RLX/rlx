// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A graph that indexes with a data-dependent index tensor must plan
//! conservatively.
//!
//! Hosted steps defer their arena write to the next flush, so the planner's
//! "an operand dies at its last consumer" rule stops describing when a buffer
//! is actually free. `plan_memory_f32_uniform` answers that by pinning output
//! ancestors — but only for graphs it recognises as host-indexing, and the
//! predicate used to list only the exotic gathers.
//!
//! Plain `Gather` is the common one: every embedding table and every codebook
//! lookup is one. A graph whose only host indexing was a plain `Gather` planned
//! as if it had none, and rlx-peakflow's encoder came back on wgpu holding
//! activations that were not its own — while every op matched in isolation and
//! `check_plan` called the plan clean. `RLX_PIN_OUTPUT_ANCESTORS=1` was the
//! difference, which is exactly what this predicate controls.

use rlx_compile::memory::plan_memory_f32_uniform;
use rlx_ir::{DType, Graph, GraphExt, Shape};

const F: DType = DType::F32;

/// Enough elementwise chain to give the planner real reuse pressure, so the
/// pinned and unpinned plans differ in arena size.
fn chain(g: &mut Graph, x: rlx_ir::NodeId, shape: &Shape, depth: usize) -> rlx_ir::NodeId {
    let mut h = x;
    for _ in 0..depth {
        let a = g.mul(h, h);
        let b = g.add(a, h);
        h = g.mul(b, a);
        let _ = shape;
    }
    h
}

fn with_gather() -> Graph {
    let mut g = Graph::new("gather");
    let shape = Shape::new(&[64, 64], F);
    let table = g.param("table", Shape::new(&[128, 64], F));
    let idx = g.input("idx", Shape::new(&[64], F));
    let rows = g.gather_(table, idx, 0);
    let h = chain(&mut g, rows, &shape, 6);
    g.set_outputs(vec![h]);
    g
}

fn without_gather() -> Graph {
    let mut g = Graph::new("no-gather");
    let shape = Shape::new(&[64, 64], F);
    let x = g.input("x", shape.clone());
    let h = chain(&mut g, x, &shape, 6);
    g.set_outputs(vec![h]);
    g
}

#[test]
fn a_plain_gather_makes_the_planner_pin() {
    // The observable consequence of pinning is that ancestors keep their slots
    // to the end, so the arena is larger than the same chain without one.
    let with = plan_memory_f32_uniform(&with_gather(), 16);
    let without = plan_memory_f32_uniform(&without_gather(), 16);
    assert!(
        with.bytes_saved() < without.bytes_saved(),
        "a gather graph reused as aggressively as one without: {} vs {} bytes saved",
        with.bytes_saved(),
        without.bytes_saved()
    );
}

#[test]
fn the_chain_without_a_gather_still_reuses() {
    // The pin must not become unconditional: a graph with no data-dependent
    // indexing is exactly the case slot reuse exists for, and pinning
    // everything is what blew a HiFi-GAN arena past wgpu's 4 GiB binding limit.
    let plan = plan_memory_f32_uniform(&without_gather(), 16);
    assert!(
        plan.bytes_saved() > 0,
        "the planner stopped reusing slots for a plain elementwise chain"
    );
}

#[test]
fn the_plan_is_still_well_formed_either_way() {
    for g in [with_gather(), without_gather()] {
        let plan = plan_memory_f32_uniform(&g, 16);
        let report = rlx_compile::plan_check::check_plan(&g, &plan);
        assert!(report.is_clean(), "{}", report.render());
    }
}
