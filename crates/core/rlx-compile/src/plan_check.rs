// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Program-safety and schedule-semantics gates** over a [`MemoryPlan`].
//!
//! `rlx_ir::verify` checks that a graph is well-formed and `rlx_ir::repr_check`
//! checks that producers and consumers agree on representation. Neither looks
//! at the *schedule*: the execution order and the arena offsets every tensor
//! lands on. That is where a whole class of defects lives, and this tree has
//! shipped several of them:
//!
//! * A wgpu deferred host-to-device flush ran in `HashSet` order, so a dead
//!   Cholesky factor was written over a live `LogDet` scalar whose slot the
//!   planner had legitimately reused. ~47% flake rate, invisible to any
//!   graph-level check.
//! * A Metal `Simd4x4` dispatch wrote 32 rows for a 16-row `C`, overrunning
//!   into whatever tensor the arena placed next (all-zero Q outputs on Gemma
//!   prefill).
//!
//! Both are *schedule* faults: the graph is correct, the representations
//! agree, and the arithmetic is right. What is wrong is where a buffer sits
//! relative to another buffer that is still live.
//!
//! # The two categories
//!
//! Following CAKE's Table 1, which separates the pre-compile gates by what
//! they protect:
//!
//! * **Program safety** — memory-use and ordering hazards. Overlapping live
//!   buffers, reading a slot that has been handed to a later writer, an output
//!   clobbered after it is produced, a slot outside the arena.
//! * **Schedule semantics** — structural invariants of the declared schedule
//!   itself. Topological order, every node scheduled exactly once, every
//!   output produced.
//!
//! # Why liveness is recomputed here
//!
//! [`crate::memory`] already computes live ranges — and this module
//! deliberately does not use them. Checking a plan with the same liveness
//! analysis that produced it is a tautology: any bug in that analysis is
//! present identically on both sides and cancels out. Liveness is therefore
//! re-derived here from `(graph, plan.schedule)` alone, which is the same
//! reason `fd_backward_gate` checks each backend against its own forward
//! rather than against another backend.

use std::collections::{HashMap, HashSet};

use rlx_ir::{Graph, NodeId};

use crate::memory::{BufferSlot, MemoryPlan, collect_view_aliases};

/// What class of contract a finding violates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanFindingKind {
    /// Two buffers that are live at the same time occupy overlapping bytes.
    LiveOverlap,
    /// A node reads a buffer whose bytes were given to a writer that already ran.
    ReadAfterOverwrite,
    /// A graph output's bytes are written by a node scheduled after it.
    OutputClobbered,
    /// A slot extends past the end of the arena.
    OutOfArena,
    /// A node runs before a node it depends on.
    OrderViolation,
    /// A node appears in the schedule more than once.
    DuplicateSchedule,
    /// A graph output is never produced by the schedule.
    OutputNotScheduled,
}

impl PlanFindingKind {
    /// `true` for memory-use / ordering hazards, `false` for structural
    /// invariants of the schedule. The split is the disposition boundary: both
    /// block, but they name different repair targets.
    pub const fn is_program_safety(self) -> bool {
        matches!(
            self,
            Self::LiveOverlap | Self::ReadAfterOverwrite | Self::OutputClobbered | Self::OutOfArena
        )
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LiveOverlap => "live-overlap",
            Self::ReadAfterOverwrite => "read-after-overwrite",
            Self::OutputClobbered => "output-clobbered",
            Self::OutOfArena => "out-of-arena",
            Self::OrderViolation => "order-violation",
            Self::DuplicateSchedule => "duplicate-schedule",
            Self::OutputNotScheduled => "output-not-scheduled",
        }
    }
}

/// One violation, naming the affected nodes so the message is a repair target
/// rather than a bare "invalid plan".
#[derive(Debug, Clone)]
pub struct PlanFinding {
    pub kind: PlanFindingKind,
    /// The node the finding is anchored at.
    pub node: NodeId,
    /// The other node involved, when the finding is about a pair.
    pub other: Option<NodeId>,
    pub detail: String,
}

/// Result of [`check_plan`], including what it was able to inspect.
///
/// `checked_nodes` exists so a clean report can be distinguished from a report
/// that inspected nothing — the same reason `repr_check` carries
/// `checked_kinds`. A plan with no assignments produces no findings, and that
/// is not the same as a plan being safe.
#[derive(Debug, Clone)]
pub struct PlanReport {
    pub findings: Vec<PlanFinding>,
    /// Nodes that had an arena assignment and were therefore inspected.
    pub checked_nodes: usize,
    /// Nodes skipped because they alias another node's storage (views).
    pub aliased_nodes: usize,
}

impl PlanReport {
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn program_safety(&self) -> impl Iterator<Item = &PlanFinding> {
        self.findings.iter().filter(|f| f.kind.is_program_safety())
    }

    pub fn schedule_semantics(&self) -> impl Iterator<Item = &PlanFinding> {
        self.findings.iter().filter(|f| !f.kind.is_program_safety())
    }

    /// Human-readable summary. States coverage even when clean, because
    /// "no findings" over zero inspected nodes means unchecked, not verified.
    pub fn render(&self) -> String {
        let mut out = String::new();
        if self.findings.is_empty() {
            out.push_str(&format!(
                "plan check: clean over {} assigned node(s) ({} aliased)\n",
                self.checked_nodes, self.aliased_nodes
            ));
            if self.checked_nodes == 0 {
                out.push_str(
                    "  NOTE: no node had an arena assignment — this plan was not \
                     checked, not proven safe.\n",
                );
            }
            return out;
        }
        for f in &self.findings {
            let other = f.other.map_or(String::new(), |o| format!(" vs %{}", o.0));
            out.push_str(&format!(
                "  [{}] %{}{other}: {}\n",
                f.kind.as_str(),
                f.node.0,
                f.detail
            ));
        }
        out
    }
}

/// Half-open byte interval `[start, end)`.
fn overlaps(a: &BufferSlot, b: &BufferSlot) -> bool {
    a.offset < b.offset + b.size && b.offset < a.offset + a.size
}

/// Check a memory plan for program-safety and schedule-semantics violations.
///
/// Pure: no device, no allocation of the arena, no execution.
pub fn check_plan(graph: &Graph, plan: &MemoryPlan) -> PlanReport {
    let mut findings = Vec::new();

    // ── Schedule semantics ──────────────────────────────────────────────
    // Position of each node in the schedule; duplicates reported once.
    let mut step_of: HashMap<NodeId, usize> = HashMap::new();
    for (step, id) in plan.schedule.iter().enumerate() {
        if let Some(prev) = step_of.insert(*id, step) {
            findings.push(PlanFinding {
                kind: PlanFindingKind::DuplicateSchedule,
                node: *id,
                other: None,
                detail: format!("scheduled at step {prev} and again at step {step}"),
            });
        }
    }

    // Every input must be produced before its consumer runs. A node absent
    // from the schedule is not an order violation on its own — inputs,
    // params and constants legitimately never execute.
    for (step, id) in plan.schedule.iter().enumerate() {
        for input in &graph.node(*id).inputs {
            if let Some(&in_step) = step_of.get(input)
                && in_step >= step
            {
                findings.push(PlanFinding {
                    kind: PlanFindingKind::OrderViolation,
                    node: *id,
                    other: Some(*input),
                    detail: format!(
                        "runs at step {step} but reads %{} which runs at step {in_step}",
                        input.0
                    ),
                });
            }
        }
    }

    for out in &graph.outputs {
        let produced = step_of.contains_key(out);
        let is_leaf = graph.node(*out).inputs.is_empty();
        if !produced && !is_leaf {
            findings.push(PlanFinding {
                kind: PlanFindingKind::OutputNotScheduled,
                node: *out,
                other: None,
                detail: "declared as a graph output but never scheduled".to_string(),
            });
        }
    }

    // ── Program safety ──────────────────────────────────────────────────
    // Views share their parent's bytes by design, so they are not independent
    // storage and must not be compared against it. Fold each view onto its
    // root and check roots only.
    let aliases = collect_view_aliases(graph);
    let storage_of = |id: NodeId| -> NodeId { aliases.get(&id).map_or(id, |(root, _)| *root) };

    for (id, slot) in &plan.assignments {
        if slot.offset + slot.size > plan.arena_size {
            findings.push(PlanFinding {
                kind: PlanFindingKind::OutOfArena,
                node: *id,
                other: None,
                detail: format!(
                    "slot [{}, {}) extends past arena_size {}",
                    slot.offset,
                    slot.offset + slot.size,
                    plan.arena_size
                ),
            });
        }
    }

    // Liveness, re-derived from the schedule alone (see module docs).
    //
    // birth = the step that writes the storage; death = the last step that
    // reads it. Graph outputs live to the end: the caller reads them after
    // the final step, so their bytes are not free at any point in the run.
    let last_step = plan.schedule.len();
    let mut live: HashMap<NodeId, (usize, usize)> = HashMap::new();
    for (step, id) in plan.schedule.iter().enumerate() {
        let store = storage_of(*id);
        live.entry(store).or_insert((step, step)).1 = step;
        for input in &graph.node(*id).inputs {
            let in_store = storage_of(*input);
            let e = live.entry(in_store).or_insert((0, step));
            e.1 = e.1.max(step);
        }
    }
    let outputs: HashSet<NodeId> = graph.outputs.iter().map(|o| storage_of(*o)).collect();
    for out in &outputs {
        if let Some(e) = live.get_mut(out) {
            e.1 = last_step;
        }
    }

    // Pairwise overlap among concurrently-live storage. Sorted by offset so
    // the scan can stop early instead of being quadratic in the common case.
    let mut slots: Vec<(NodeId, &BufferSlot, (usize, usize))> = plan
        .assignments
        .iter()
        .filter(|(id, _)| storage_of(**id) == **id) // roots only
        .filter_map(|(id, slot)| live.get(id).map(|r| (*id, slot, *r)))
        .collect();
    slots.sort_by_key(|(_, s, _)| s.offset);

    let mut aliased_nodes = 0usize;
    for id in plan.assignments.keys() {
        if storage_of(*id) != *id {
            aliased_nodes += 1;
        }
    }

    for i in 0..slots.len() {
        let (id_a, slot_a, (birth_a, death_a)) = slots[i];
        for &(id_b, slot_b, (birth_b, death_b)) in slots.iter().skip(i + 1) {
            // Sorted by offset: once b starts past a's end, so does everything
            // after it.
            if slot_b.offset >= slot_a.offset + slot_a.size {
                break;
            }
            if !overlaps(slot_a, slot_b) {
                continue;
            }
            // Live ranges are inclusive of their endpoints. Sharing a slot is
            // the planner's whole purpose; it is only a fault when the ranges
            // actually intersect.
            let disjoint = death_a < birth_b || death_b < birth_a;
            if disjoint {
                continue;
            }
            let (kind, detail) = if outputs.contains(&id_a) || outputs.contains(&id_b) {
                (
                    PlanFindingKind::OutputClobbered,
                    format!(
                        "graph-output bytes [{}, {}) overlap [{}, {}) while both are live \
                         (steps {birth_a}..={death_a} and {birth_b}..={death_b})",
                        slot_a.offset,
                        slot_a.offset + slot_a.size,
                        slot_b.offset,
                        slot_b.offset + slot_b.size
                    ),
                )
            } else {
                (
                    PlanFindingKind::LiveOverlap,
                    format!(
                        "bytes [{}, {}) overlap [{}, {}) while both are live \
                         (steps {birth_a}..={death_a} and {birth_b}..={death_b})",
                        slot_a.offset,
                        slot_a.offset + slot_a.size,
                        slot_b.offset,
                        slot_b.offset + slot_b.size
                    ),
                )
            };
            findings.push(PlanFinding {
                kind,
                node: id_a,
                other: Some(id_b),
                detail,
            });
        }
    }

    // A consumer reading a slot after the planner handed those bytes to a
    // writer that already ran. Distinct from LiveOverlap: this one names the
    // read that would return the wrong bytes.
    for (step, id) in plan.schedule.iter().enumerate() {
        for input in &graph.node(*id).inputs {
            let in_store = storage_of(*input);
            let Some(in_slot) = plan.assignments.get(&in_store) else {
                continue;
            };
            for (w_step, writer) in plan.schedule.iter().enumerate() {
                if w_step >= step {
                    break;
                }
                let w_store = storage_of(*writer);
                if w_store == in_store {
                    continue;
                }
                let Some(w_slot) = plan.assignments.get(&w_store) else {
                    continue;
                };
                if !overlaps(in_slot, w_slot) {
                    continue;
                }
                // Only a fault if the input was born before the writer ran;
                // otherwise the bytes were legitimately recycled first.
                if live
                    .get(&in_store)
                    .is_some_and(|(birth, _)| *birth < w_step)
                {
                    findings.push(PlanFinding {
                        kind: PlanFindingKind::ReadAfterOverwrite,
                        node: *id,
                        other: Some(*writer),
                        detail: format!(
                            "reads %{} at step {step}, but %{} wrote overlapping bytes at \
                             step {w_step}",
                            input.0, writer.0
                        ),
                    });
                }
            }
        }
    }

    let checked_nodes = plan.assignments.len();
    PlanReport {
        findings,
        checked_nodes,
        aliased_nodes,
    }
}
