// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Static weight packs must get exclusively-owned arena slots — checked
//! without a GPU.**
//!
//! The backend materialises a `Concat` over `Param`s (the fused QKV / gate+up
//! weights) once and skips it on later `run()`s. That is only sound if nothing
//! else is assigned its slot, so `plan_f32_uniform_with` pins packs for the
//! whole graph. When it does not, the runtime's slot-exclusivity check
//! correctly refuses to skip and the optimisation goes **silently dead** —
//! which is exactly what happened on CUDA (0 of 140 pack steps qualified on a
//! 28-layer graph, while a 1-layer graph looked fine).
//!
//! `plan_f32_uniform_with` is a pure function of the `Graph`, so this checks the
//! property that actually matters on any machine — no ROCm device, no HIP, no
//! rig. That is deliberate. ROCm device tests skip when no GPU is present and
//! still report `ok`, so on a rig whose card is unavailable (a wedged or
//! RAS-fenced GPU enumerates zero HSA agents) a planner regression would sail
//! through unnoticed. `RLX_REQUIRE_DEVICE=1` turns those skips into failures;
//! this test needs neither.
//!
//! The device-level behaviour still needs `static_weight_pack_rebind.rs` on real
//! hardware; this covers the half that does not need any.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use std::collections::HashMap;

const F: DType = DType::F32;

/// `layers` stacked blocks, each `matmul(x, concat([a, b]))` — the shape the
/// matmul-fusion passes produce, and the one where per-layer packs compete for
/// slots with the previous layer's dead activations.
fn deep_packed_graph(layers: usize, h: usize, ff: usize) -> Graph {
    let mut g = Graph::new("deep_packs");
    let x = g.input("x", Shape::new(&[1, h], F));
    let mut cur = x;
    for l in 0..layers {
        let a = g.param(format!("a{l}"), Shape::new(&[h, ff], F));
        let b = g.param(format!("b{l}"), Shape::new(&[h, ff], F));
        let pack = g.concat_(vec![a, b], 1);
        let m = g.matmul(cur, pack, Shape::new(&[1, 2 * ff], F));
        let n = g.narrow_(m, 1, 0, h);
        cur = g.add(cur, n);
    }
    g.set_outputs(vec![cur]);
    g
}

#[test]
fn every_static_weight_pack_owns_its_slot_exclusively() {
    const H: usize = 256;
    const FF: usize = 512;

    for layers in [1usize, 2, 8, 28] {
        let g = deep_packed_graph(layers, H, FF);
        let plan = rlx_rocm::arena::plan_f32_uniform_with(&g, 16, true);

        // How many materialising nodes land on each offset.
        let mut owners: HashMap<usize, usize> = HashMap::new();
        for node in g.nodes() {
            if rlx_opt::memory::is_pure_view(&g, node) {
                continue;
            }
            if let Some(slot) = plan.assignments.get(&node.id) {
                *owners.entry(slot.offset).or_insert(0) += 1;
            }
        }

        let mut memo: HashMap<rlx_ir::NodeId, bool> = HashMap::new();
        let mut packs = 0usize;
        let mut shared = Vec::new();
        for node in g.nodes() {
            if matches!(node.op, rlx_ir::Op::Param { .. } | rlx_ir::Op::Input { .. }) {
                continue;
            }
            if !rlx_opt::memory::is_static_weight_tensor(&g, node.id, &mut memo) {
                continue;
            }
            packs += 1;
            let Some(slot) = plan.assignments.get(&node.id) else {
                continue;
            };
            if owners.get(&slot.offset).copied() != Some(1) {
                shared.push((node.id, slot.offset, owners[&slot.offset]));
            }
        }

        assert_eq!(
            packs, layers,
            "{layers} layers should build {layers} weight packs, found {packs} — \
             the graph under test no longer exercises what it claims to"
        );
        assert!(
            shared.is_empty(),
            "{layers} layers: {} of {packs} packs share their arena slot with \
             another materialising node {shared:?}. Skipping them would read \
             bytes that node rewrites on the next run, so the runtime will \
             refuse and the optimisation is dead.",
            shared.len()
        );
    }
}
