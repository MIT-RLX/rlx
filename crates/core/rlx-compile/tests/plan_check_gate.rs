// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The program-safety / schedule-semantics gate must do two things, and the
//! second is the harder one.
//!
//! 1. **Fire on the defects it was built for.** Each hazard is constructed by
//!    corrupting a real plan, because a checker that only passes is
//!    indistinguishable from a checker that inspects nothing — a failure mode
//!    this tree has shipped more than once.
//! 2. **Stay silent on plans the planner actually produces.** Slot reuse is the
//!    planner's entire purpose. A gate that calls legitimate reuse a hazard is
//!    worse than no gate: it would have to be disabled, and then it protects
//!    nothing.

use rlx_compile::memory::{
    BufferSlot, MemoryPlan, MemoryPlanOptions, plan_memory, plan_memory_with_options,
};
use rlx_compile::plan_check::{PlanFindingKind, check_plan};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};

const F: DType = DType::F32;

/// A small graph with real reuse pressure: a chain of elementwise ops whose
/// intermediates die immediately, so the planner shares slots aggressively.
fn chain_graph() -> Graph {
    let mut g = Graph::new("chain");
    let shape = Shape::new(&[64, 64], F);
    let x = g.input("x", shape.clone());
    let w = g.param("w", shape.clone());
    let a = g.matmul(x, w, shape.clone());
    let b = g.mul(a, a);
    let c = g.add(b, a);
    let d = g.mul(c, b);
    let e = g.add(d, c);
    g.set_outputs(vec![e]);
    g
}

/// The in-order-executor configuration (CPU, wgpu): `pin_output_ancestors`
/// off, which is the *only* setting under which the planner actually reuses
/// slots. The default pins the whole ancestor DAG and, as its own doc says,
/// "destroys slot reuse on deep feed-forward graphs" — so a gate tested only
/// against the default would never exercise its overlap logic.
fn reusing_plan(g: &Graph) -> MemoryPlan {
    let mut opts = MemoryPlanOptions::inference();
    opts.pin_output_ancestors = false;
    opts.dequant_host_fallback = false;
    plan_memory_with_options(g, 64, opts)
}

fn kinds(g: &Graph, p: &MemoryPlan) -> Vec<PlanFindingKind> {
    check_plan(g, p).findings.iter().map(|f| f.kind).collect()
}

#[test]
fn real_plans_are_clean() {
    // The load-bearing test. If the planner's own output trips this gate, the
    // gate is wrong, not the planner. Both configurations are checked: the
    // pinned default and the reuse-heavy in-order one.
    let g = chain_graph();
    for plan in [plan_memory(&g), reusing_plan(&g)] {
        let report = check_plan(&g, &plan);
        assert!(
            report.is_clean(),
            "planner output flagged as unsafe:\n{}",
            report.render()
        );
        assert!(
            report.checked_nodes > 0,
            "gate inspected nothing — a clean report here would be vacuous"
        );
    }
}

#[test]
fn the_planner_actually_reuses_slots_here() {
    // Guards the test above from becoming trivial. Without this, a graph that
    // stopped producing slot reuse would let `real_plans_are_clean` pass while
    // never running the overlap comparison at all — which is what happened on
    // the first attempt, where the default's ancestor pinning left
    // bytes_saved() == 0.
    let g = chain_graph();
    assert!(
        reusing_plan(&g).bytes_saved() > 0,
        "graph must exercise slot reuse or the clean-plan test proves nothing"
    );
}

#[test]
fn detects_overlapping_live_buffers() {
    // The wgpu HostTensorCache defect in miniature: two buffers that are both
    // live given the same bytes.
    let g = chain_graph();
    let mut plan = reusing_plan(&g);
    let ids: Vec<NodeId> = plan.schedule.clone();
    // Point the last two scheduled nodes at the same offset. Both are live at
    // the end (one feeds the other), so this must be reported.
    let victim = ids[ids.len() - 1];
    let other = ids[ids.len() - 2];
    let slot = plan.assignments[&other].clone();
    plan.assignments.insert(
        victim,
        BufferSlot {
            offset: slot.offset,
            size: slot.size,
        },
    );
    let found = kinds(&g, &plan);
    assert!(
        found.iter().any(|k| matches!(
            k,
            PlanFindingKind::LiveOverlap | PlanFindingKind::OutputClobbered
        )),
        "aliased live buffers not reported; got {found:?}"
    );
}

#[test]
fn detects_a_slot_past_the_end_of_the_arena() {
    let g = chain_graph();
    let mut plan = reusing_plan(&g);
    let id = plan.schedule[0];
    let slot = plan.assignments[&id].clone();
    plan.assignments.insert(
        id,
        BufferSlot {
            offset: plan.arena_size,
            size: slot.size.max(1),
        },
    );
    assert!(
        kinds(&g, &plan).contains(&PlanFindingKind::OutOfArena),
        "a slot starting at arena_size must be reported"
    );
}

#[test]
fn detects_a_consumer_scheduled_before_its_producer() {
    let g = chain_graph();
    let mut plan = reusing_plan(&g);
    // Reverse the schedule: every consumer now precedes its producer.
    plan.schedule.reverse();
    let found = kinds(&g, &plan);
    assert!(
        found.contains(&PlanFindingKind::OrderViolation),
        "a reversed schedule must be an order violation; got {found:?}"
    );
}

#[test]
fn detects_a_node_scheduled_twice() {
    let g = chain_graph();
    let mut plan = reusing_plan(&g);
    let dup = plan.schedule[0];
    plan.schedule.push(dup);
    assert!(
        kinds(&g, &plan).contains(&PlanFindingKind::DuplicateSchedule),
        "a node appearing twice in the schedule must be reported"
    );
}

#[test]
fn detects_an_output_that_is_never_produced() {
    let g = chain_graph();
    let mut plan = reusing_plan(&g);
    let out = g.outputs[0];
    plan.schedule.retain(|id| *id != out);
    assert!(
        kinds(&g, &plan).contains(&PlanFindingKind::OutputNotScheduled),
        "dropping the output from the schedule must be reported"
    );
}

#[test]
fn findings_split_into_the_two_gate_categories() {
    // The categories are the point: they name different repair targets, and
    // collapsing them would lose that. Corrupt one of each and check both
    // buckets are non-empty.
    let g = chain_graph();
    let mut plan = reusing_plan(&g);
    let id = plan.schedule[0];
    let slot = plan.assignments[&id].clone();
    plan.assignments.insert(
        id,
        BufferSlot {
            offset: plan.arena_size,
            size: slot.size.max(1),
        },
    );
    plan.schedule.push(id);

    let report = check_plan(&g, &plan);
    assert!(
        report.program_safety().count() > 0,
        "expected a program-safety finding:\n{}",
        report.render()
    );
    assert!(
        report.schedule_semantics().count() > 0,
        "expected a schedule-semantics finding:\n{}",
        report.render()
    );
}

#[test]
fn a_clean_report_over_nothing_says_so() {
    // Distinguishing "checked and safe" from "inspected nothing" is the
    // difference between a gate and a decoration.
    let g = Graph::new("empty");
    let plan = MemoryPlan {
        arena_size: 0,
        assignments: Default::default(),
        schedule: Vec::new(),
    };
    let report = check_plan(&g, &plan);
    assert!(report.is_clean());
    assert_eq!(report.checked_nodes, 0);
    assert!(
        report.render().contains("not checked"),
        "an empty plan must not read as verified: {}",
        report.render()
    );
}
