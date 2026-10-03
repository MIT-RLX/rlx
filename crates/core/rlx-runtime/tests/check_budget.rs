// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The checker's **budget** axis: what a graph costs on one architecture.
//!
//! The other axes answer "is this correct". This one answers "what will it
//! spend", which is the question you cannot ask a parity test — every number
//! here comes from a graph that computes exactly the right answer. Three of
//! them are architecture-dependent in a way that is easy to get wrong silently:
//!
//! * an f32-uniform arena stores a BF16 activation in 4 B/elem, not 2, so the
//!   same graph has two different memory footprints depending on the backend;
//! * parameters are uploaded once and stay resident, so counting them as
//!   per-run host traffic would swamp the number that actually matters;
//! * a kind that lowers through portable common-IR is correct and off the
//!   native fast path, which no output comparison can see.

use rlx_ir::{DType, Graph, Shape, op::BinaryOp};
use rlx_opt::rlx_compile::fusion_pipeline::FusionTarget;
use rlx_runtime::check::{CheckOptions, Severity, check_graph};

/// `(x + b) * b` in `dtype`, with `b` a parameter. Small, well-formed, and
/// carries one activation per op so the arena numbers are easy to reason about.
fn graph_in(dtype: DType) -> Graph {
    let s = Shape::new(&[64, 64], dtype);
    let mut g = Graph::new("budget");
    let x = g.input("x", s.clone());
    let b = g.param("b", s.clone());
    let add = g.binary(BinaryOp::Add, x, b, s.clone());
    let mul = g.binary(BinaryOp::Mul, add, b, s.clone());
    g.set_outputs(vec![mul]);
    g
}

fn opts_for(t: FusionTarget) -> CheckOptions {
    CheckOptions {
        backends: vec![t],
        ..Default::default()
    }
}

fn budget(t: FusionTarget, dtype: DType) -> rlx_runtime::check::Budget {
    let g = graph_in(dtype);
    let report = check_graph(&g, &opts_for(t));
    report
        .backends
        .first()
        .and_then(|b| b.budget.clone())
        .expect("budget for a well-formed graph")
}

/// A backend that stores activations at their native width has, by definition,
/// nothing to report here — the baseline is itself.
#[test]
fn native_width_backends_report_no_overhead() {
    for dtype in [DType::F32, DType::BF16] {
        let b = budget(FusionTarget::Cuda, dtype);
        assert_eq!(b.width_policy, "native");
        assert_eq!(b.width_overhead_bytes, 0, "{dtype:?}");
    }
}

/// CPU and Metal are native for everything EXCEPT low-precision floats, which
/// they widen to 4 B/elem: neither has a bf16 kernel (CPU computes in f32 and
/// reads every activation slot as `&[f32]`; Metal's `HalfFlag` has no BF16
/// lane), so a packed bf16 activation would be read at the wrong width. That
/// widening is a real, reportable memory cost — and it is zero on an f32 graph,
/// where the policy changes nothing.
#[test]
fn half_widening_backends_report_their_overhead() {
    for (t, policy) in [
        (FusionTarget::Cpu, "native-half-widened"),
        (FusionTarget::Metal, "native-bf16-widened"),
    ] {
        let f32_graph = budget(t, DType::F32);
        assert_eq!(f32_graph.width_policy, policy, "{t:?}");
        assert_eq!(
            f32_graph.width_overhead_bytes, 0,
            "{t:?}: an f32 graph stores the same either way"
        );
        let bf16 = budget(t, DType::BF16);
        assert!(
            bf16.width_overhead_bytes > 0,
            "{t:?}: bf16 activations are widened, so the cost must be reported"
        );
    }
}

/// wgpu's f32-uniform arena is free for an f32 graph and costs 2 B/elem on
/// every BF16 activation. That is the number a hybrid arena would recover, and
/// the reason it is worth reporting rather than assuming.
#[test]
fn f32_uniform_arena_costs_low_precision_activations() {
    let f32_graph = budget(FusionTarget::Wgpu, DType::F32);
    assert_eq!(f32_graph.width_policy, "f32-uniform");
    assert_eq!(
        f32_graph.width_overhead_bytes, 0,
        "an f32 graph stores the same either way"
    );

    let bf16 = budget(FusionTarget::Wgpu, DType::BF16);
    assert!(
        bf16.width_overhead_bytes > 0,
        "a BF16 graph must cost more on an f32-uniform arena, got {bf16:?}"
    );
    // Same graph, strictly more memory, purely because of where it runs.
    // Compared against CUDA, which is the remaining fully-native backend —
    // CPU and Metal widen bf16 activations too, for correctness.
    let native = budget(FusionTarget::Cuda, DType::BF16);
    assert!(
        bf16.arena_bytes > native.arena_bytes,
        "wgpu {} should exceed cuda {}",
        bf16.arena_bytes,
        native.arena_bytes
    );
}

/// Per-run host traffic is inputs in and outputs back. A parameter is uploaded
/// once and stays resident; folding it in here would report a 7 GB "per-run"
/// cost for a model that moves a few KB per token.
#[test]
fn host_io_excludes_resident_parameters() {
    let b = budget(FusionTarget::Cpu, DType::F32);
    let one = 64 * 64 * 4;
    assert_eq!(
        b.host_io_bytes,
        one * 2,
        "expected one input + one output, not the parameter"
    );
}

/// Nothing this axis finds is an error. A budget note says the program is
/// correct and something is being spent to get there — a finding that could
/// fail a build would mean rejecting correct programs for being expensive.
#[test]
fn budget_findings_are_notes_only() {
    let g = graph_in(DType::BF16);
    let report = check_graph(&g, &opts_for(FusionTarget::Wgpu));
    for d in &report.diagnostics {
        if d.code.starts_with("budget-") {
            assert_eq!(d.severity, Severity::Note, "{d:?}");
        }
    }
}

/// Turning the axis off removes the notes but keeps the structured numbers —
/// the summary is what a tool reads, the notes are what a human reads.
#[test]
fn disabling_the_axis_silences_notes_not_numbers() {
    let g = graph_in(DType::BF16);
    let opts = CheckOptions {
        budget: false,
        ..opts_for(FusionTarget::Wgpu)
    };
    let report = check_graph(&g, &opts);
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|d| d.code.starts_with("budget-")),
        "budget notes should be suppressed"
    );
    assert!(
        report.backends[0].budget.is_some(),
        "the summary should still carry the numbers"
    );
}
