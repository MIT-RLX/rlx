// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The **schedule** axis of `check_graph` — CAKE's *hint* disposition.
//!
//! `check` already gates (shape/repr errors) and reports (fusion/numeric
//! warnings). This axis is the third kind of signal: nothing here is wrong. A
//! decode matmul routed to the default `64x64` tile computes exactly the right
//! answer; it just does 64× the row arithmetic to get there and throws the rest
//! away. That is not a defect to block on and not a warning about correctness —
//! it is a pointer at where the machine is being wasted.
//!
//! Three properties matter more than the wording, and all are tested below:
//!
//! * **It must stay quiet on healthy graphs.** A note that fires on every
//!   matmul is a note nobody reads, which costs more than not having it. Two
//!   earlier versions failed this: per-node reporting gave 196 lines for a
//!   28-layer decode graph, and per-shape reporting gave 3 lines that were one
//!   fact about rlx's default tile restated per shape.
//! * **It must report gain, not waste.** These are not the same signal. A shape
//!   using 1.6% of its tile's arithmetic has 1.56x available; one using 93.8%
//!   has 1.16x. Reporting "98.4% discarded" implies a 60x opportunity that does
//!   not exist.
//! * **It must never be an error.** Severity is the whole distinction between
//!   this axis and the repr gate; if a schedule note could fail a build,
//!   `check` would start rejecting correct programs for being slow.

use rlx_ir::{DType, Graph, GraphExt, Shape};
use rlx_runtime::check::{CheckOptions, Severity, check_graph};

const F: DType = DType::F32;

fn matmul_graph(m: usize, k: usize, n: usize) -> Graph {
    let mut g = Graph::new("mm");
    let x = g.input("x", Shape::new(&[m, k], F));
    let w = g.param("w", Shape::new(&[k, n], F));
    let y = g.matmul(x, w, Shape::new(&[m, n], F));
    g.set_outputs(vec![y]);
    g
}

/// GPU-only options, so the axis is exercised without dragging in every
/// backend's fusion pipeline.
fn gpu_opts() -> CheckOptions {
    CheckOptions {
        backends: vec![rlx_compile::fusion_pipeline::FusionTarget::Cuda],
        dispatch: false,
        fusion: false,
        numeric: false,
        repr: false,
        schedule: true,
        // Off: this file isolates the schedule (hint) axis. The plan gates are
        // errors and would drown the notes under test.
        plan: false,
    }
}

fn schedule_notes(g: &Graph) -> Vec<String> {
    check_graph(g, &gpu_opts())
        .diagnostics
        .into_iter()
        .filter(|d| d.code == "schedule-tile")
        .map(|d| d.message)
        .collect()
}

#[test]
fn decode_matmul_reports_the_gain_a_narrower_tile_would_give() {
    let notes = schedule_notes(&matmul_graph(1, 4096, 4096));
    assert_eq!(notes.len(), 1, "expected exactly one note, got {notes:?}");
    assert!(notes[0].contains("1x4096x4096"), "{}", notes[0]);
    // The number quoted must be the available speedup, not the discarded
    // fraction. 1.56x is what a 32x32x32 tile models here; 98.4% is what the
    // old wording said and would have been a 60x claim.
    assert!(notes[0].contains("1.56x"), "{}", notes[0]);
    assert!(
        notes[0].contains("32x32x32"),
        "the winning tile must be named: {}",
        notes[0]
    );
}

#[test]
fn shapes_with_nothing_to_gain_stay_quiet() {
    // No candidate tile models materially better than the default here (~1.16x,
    // under the 1.25x bar), so there is nothing to act on and the axis must say
    // nothing. Note these are NOT the shapes with the best MAC utilization —
    // that was the wrong axis to pick quiet shapes on.
    for (m, k, n) in [(48, 1024, 1024), (64, 1024, 1024), (64, 64, 64)] {
        let notes = schedule_notes(&matmul_graph(m, k, n));
        assert!(
            notes.is_empty(),
            "{m}x{k}x{n} should be quiet, got {notes:?}"
        );
    }
}

#[test]
fn a_whole_decode_graph_collapses_to_one_note() {
    // A 28-layer decode step runs 196 matmuls across 3 distinct shapes, every
    // one of them m=1. Per-node reporting produced 196 identical lines and
    // per-shape reporting produced 3 — all of them the same finding, since the
    // modelled gain is ~1.56x for every decode shape.
    let (d, ff, layers) = (2048usize, 5632usize, 28usize);
    let mut g = Graph::new("decode");
    let mut h = g.input("x", Shape::new(&[1, d], F));
    for l in 0..layers {
        // Attention projections: four [d,d] matmuls off the residual.
        for name in ["q", "k", "v", "o"] {
            let w = g.param(format!("l{l}.{name}").as_str(), Shape::new(&[d, d], F));
            h = g.matmul(h, w, Shape::new(&[1, d], F));
        }
        // MLP: gate and up BRANCH from the same input — chaining them would
        // make `up` a [ff,ff] matmul, a shape no real model runs.
        let gw = g.param(format!("l{l}.gate").as_str(), Shape::new(&[d, ff], F));
        let uw = g.param(format!("l{l}.up").as_str(), Shape::new(&[d, ff], F));
        let gate = g.matmul(h, gw, Shape::new(&[1, ff], F));
        let up = g.matmul(h, uw, Shape::new(&[1, ff], F));
        let act = g.mul(gate, up);
        let dw = g.param(format!("l{l}.down").as_str(), Shape::new(&[ff, d], F));
        h = g.matmul(act, dw, Shape::new(&[1, d], F));
    }
    g.set_outputs(vec![h]);

    let notes = schedule_notes(&g);
    assert_eq!(
        notes.len(),
        1,
        "the whole graph is worth exactly one note, got:\n{notes:#?}"
    );
    // The scale must survive the collapse, or one line reads like one matmul
    // when it stands for the entire model.
    assert!(
        notes[0].contains("196 node(s)"),
        "node count must be carried: {}",
        notes[0]
    );
    assert!(
        notes[0].contains("3 matmul shape(s)"),
        "shape count must be carried: {}",
        notes[0]
    );
}

#[test]
fn the_threshold_is_where_it_says_it_is() {
    // Pins MIN_MODELLED_GAIN = 1.25 by straddling it:
    //   64x1024x1024  -> 1.16x modelled -> quiet
    //   128x1024x1024 -> 1.27x modelled -> note
    // Without this pair the constant can be moved without any test noticing.
    assert!(
        schedule_notes(&matmul_graph(64, 1024, 1024)).is_empty(),
        "1.16x is below the 1.25x bar and must stay quiet"
    );
    assert_eq!(
        schedule_notes(&matmul_graph(128, 1024, 1024)).len(),
        1,
        "1.27x is above the 1.25x bar and must be reported"
    );
}

#[test]
fn schedule_findings_are_never_errors() {
    // The load-bearing property: this axis must not be able to fail a build.
    // A correct-but-slow program is not a broken program.
    let report = check_graph(&matmul_graph(1, 4096, 4096), &gpu_opts());
    let sched: Vec<_> = report
        .diagnostics
        .iter()
        .filter(|d| d.code == "schedule-tile")
        .collect();
    assert!(
        !sched.is_empty(),
        "the axis must have produced something to test"
    );
    for d in &sched {
        assert_eq!(
            d.severity,
            Severity::Note,
            "schedule finding escalated: {}",
            d.message
        );
        assert!(
            d.hint.is_some(),
            "a hint disposition without a remedy is just noise"
        );
    }
    assert_eq!(report.errors(), 0, "a slow schedule must not be an error");
    assert!(!report.has_errors());
}

#[test]
fn the_axis_can_be_turned_off() {
    let off = CheckOptions {
        schedule: false,
        ..gpu_opts()
    };
    let report = check_graph(&matmul_graph(1, 4096, 4096), &off);
    assert!(report.diagnostics.iter().all(|d| d.code != "schedule-tile"));
}

#[test]
fn cpu_only_checks_do_not_get_tile_notes() {
    // CPU does not dispatch through the GPU tile table, so a tile note there
    // would be describing a schedule that never runs.
    let cpu = CheckOptions {
        backends: vec![rlx_compile::fusion_pipeline::FusionTarget::Cpu],
        ..gpu_opts()
    };
    let report = check_graph(&matmul_graph(1, 4096, 4096), &cpu);
    assert!(report.diagnostics.iter().all(|d| d.code != "schedule-tile"));
}

#[test]
fn dynamic_extents_are_skipped_not_guessed() {
    // The tile that runs depends on a runtime value, so there is no schedule to
    // assess. Reporting one would be a confident claim about a shape that may
    // never occur.
    let mut g = Graph::new("dyn");
    let x = g.input(
        "x",
        Shape::from_dims(&[rlx_ir::Dim::Dynamic(0), rlx_ir::Dim::Static(4096)], F),
    );
    let w = g.param("w", Shape::new(&[4096, 4096], F));
    let y = g.matmul(
        x,
        w,
        Shape::from_dims(&[rlx_ir::Dim::Dynamic(0), rlx_ir::Dim::Static(4096)], F),
    );
    g.set_outputs(vec![y]);
    assert!(
        schedule_notes(&g).is_empty(),
        "a dynamic m must not be scheduled statically"
    );
}
