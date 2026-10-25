// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The CPU planner no longer pins a graph's whole output-ancestor DAG just
//! because the graph contains a `Gather`.
//!
//! That pin exists for backends whose hosted gather writes its result at a
//! later flush. The CPU executor runs every thunk inline, in schedule order, so
//! ordinary liveness is exact there — and the pin was costing every activation
//! its slot: a Qwen3-VL vision tower (one position-table `Gather`) planned
//! ~24 GB for a 4928-patch image instead of ~5.
//!
//! What has to hold is that reuse never changes an answer: a gather feeding a
//! deep chain whose intermediates all die early (so the planner packs them
//! into each other's slots) must come back bit-identical with the pin forced
//! on (`RLX_PIN_OUTPUT_ANCESTORS=1`) and off.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::{Device, Session};

const F: DType = DType::F32;

/// Table lookup, then a residual chain with a second gather in the middle,
/// so gathered rows are read again long after most intermediates have died.
fn graph() -> Graph {
    let mut g = Graph::new("gather_reuse");
    let table = g.param("table", Shape::new(&[32, 16], F));
    let idx = g.input("idx", Shape::new(&[24], F));
    let rows = g.gather_(table, idx, 0);
    let mut h = rows;
    for i in 0..12 {
        let a = g.mul(h, h);
        let b = g.add(a, rows);
        let c = g.mul(b, h);
        h = g.sub(c, a);
        if i == 6 {
            let again = g.gather_(table, idx, 0);
            h = g.add(h, again);
        }
    }
    g.set_outputs(vec![h]);
    g
}

fn run() -> Vec<f32> {
    let mut c = Session::new(Device::Cpu).compile(graph());
    let table: Vec<f32> = (0..32 * 16)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 400.0)
        .collect();
    let idx: Vec<f32> = (0..24).map(|i| ((i * 7) % 32) as f32).collect();
    c.set_param("table", &table);
    c.run(&[("idx", &idx)])[0].clone()
}

#[test]
fn reuse_matches_the_pinned_plan_bit_for_bit() {
    rlx_ir::env::remove("RLX_PIN_OUTPUT_ANCESTORS");
    let reused = run();
    rlx_ir::env::set("RLX_PIN_OUTPUT_ANCESTORS", "1");
    let pinned = run();
    rlx_ir::env::remove("RLX_PIN_OUTPUT_ANCESTORS");
    assert_eq!(reused.len(), 24 * 16);
    assert!(reused.iter().all(|v| v.is_finite()));
    assert_eq!(
        reused.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        pinned.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "slot reuse changed the result of a gather-driven graph on CPU"
    );
}
