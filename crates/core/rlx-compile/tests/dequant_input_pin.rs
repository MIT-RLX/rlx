// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `dequant_host_fallback` pins a dequant matmul's activation input, because a
//! deferred host flush reads it after later GPU ops ran. It must pin *that*
//! tensor, not everything upstream of it: pinning every ancestor left a packed
//! Qwen3-1.7B prefill with no reused slot at all (2.8 GB at 297 tokens).

use rlx_compile::memory::{BufferSlot, MemoryPlan, MemoryPlanOptions, plan_memory_with_options};
use rlx_ir::quant::QuantScheme;
use rlx_ir::{DType, Graph, GraphExt, NodeId, Op, Shape};

fn overlaps(a: &BufferSlot, b: &BufferSlot) -> bool {
    a.offset < b.offset + b.size && b.offset < a.offset + a.size
}

/// Nodes scheduled after `after` whose slot shares bytes with `id`'s.
fn later_sharers(plan: &MemoryPlan, id: NodeId, after: NodeId) -> Vec<NodeId> {
    let start = plan.schedule.iter().position(|&n| n == after).unwrap();
    let slot = &plan.assignments[&id];
    plan.schedule[start + 1..]
        .iter()
        .filter(|n| plan.assignments.get(n).is_some_and(|s| overlaps(s, slot)))
        .copied()
        .collect()
}

#[test]
fn dequant_input_is_pinned_and_its_ancestors_are_not() {
    let f = Shape::new(&[64, 64], DType::F32);
    let mut g = Graph::new("dequant_pin");
    let x = g.input("x", f.clone());
    let a = g.mul(x, x);
    let b = g.mul(a, a);
    let w = g.param("w", Shape::new(&[64 * 64 / 32 * 34], DType::U8));
    let y = g.add_node(
        Op::DequantMatMul {
            scheme: QuantScheme::GgufQ8_0,
        },
        vec![b, w],
        f.clone(),
    );
    let c = g.mul(y, y);
    let d = g.mul(c, c);
    let e = g.add(d, c);
    let h = g.mul(e, e);
    g.set_outputs(vec![h]);

    let mut opts = MemoryPlanOptions::inference();
    opts.pin_output_ancestors = false;
    opts.dequant_host_fallback = true;
    let plan = plan_memory_with_options(&g, 64, opts);

    assert!(
        later_sharers(&plan, b, y).is_empty(),
        "the dequant input's slot was reused before the graph ended"
    );
    assert!(
        !later_sharers(&plan, a, b).is_empty(),
        "an ancestor of the dequant input was pinned too (no later node reuses it)"
    );
}
