// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pass tests written as text-in / text-out fixtures.
//!
//! The point of [`rlx_ir::text`]: a pass test is a pair of strings rather than
//! thirty lines of graph construction plus hand-rolled structural assertions.
//! Cheap tests get written, and what a pass *does* is legible in the diff when
//! one changes.
//!
//! Each case below states the input IR and the expected output IR verbatim. A
//! mismatch prints both, so the failure tells you what the pass produced
//! instead of which assertion tripped.

use rlx_fusion::lower_fma::LowerFma;
use rlx_fusion::pass::Pass;
use rlx_ir::{IgnoreConfig, text};

/// Run `pass` on the graph in `input` and require the result to match
/// `expected`, comparing IR rather than text so formatting is not under test.
fn check(pass: &dyn Pass, input: &str, expected: &str) {
    let graph = text::parse(input).unwrap_or_else(|e| panic!("bad input fixture: {e}"));
    let want = text::parse(expected).unwrap_or_else(|e| panic!("bad expected fixture: {e}"));

    let got = pass.run(graph);

    // Node names and the graph name are incidental to what a lowering does.
    assert!(
        got.structurally_eq(&want, IgnoreConfig::SEMANTIC),
        "`{}` produced unexpected IR\n\n--- got ---\n{}\n--- want ---\n{}",
        pass.name(),
        text::print(&got),
        text::print(&want),
    );
}

#[test]
fn lower_fma_expands_to_mul_add() {
    check(
        &LowerFma,
        r#"
        graph @fma {
          %0 = {"Input":{"name":"a"}} : [8] f32
          %1 = {"Input":{"name":"b"}} : [8] f32
          %2 = {"Input":{"name":"c"}} : [8] f32
          %3 = Fma(%0, %1, %2) : [8] f32
          return %3
        }
        "#,
        r#"
        graph @fma {
          %0 = {"Input":{"name":"a"}} : [8] f32
          %1 = {"Input":{"name":"b"}} : [8] f32
          %2 = {"Input":{"name":"c"}} : [8] f32
          %3 = {"Binary":"Mul"}(%0, %1) : [8] f32
          %4 = {"Binary":"Add"}(%3, %2) : [8] f32
          return %4
        }
        "#,
    );
}

#[test]
fn lower_fma_leaves_an_fma_free_graph_alone() {
    let unchanged = r#"
        graph @plain {
          %0 = {"Input":{"name":"a"}} : [4] f32
          %1 = {"Input":{"name":"b"}} : [4] f32
          %2 = {"Binary":"Add"}(%0, %1) : [4] f32
          return %2
        }
        "#;
    check(&LowerFma, unchanged, unchanged);
}

#[test]
fn lower_fma_handles_several_fmas_and_shared_operands() {
    // Two Fmas, the second consuming the first, sharing operand %1.
    check(
        &LowerFma,
        r#"
        graph @chain {
          %0 = {"Input":{"name":"a"}} : [2] f32
          %1 = {"Input":{"name":"b"}} : [2] f32
          %2 = Fma(%0, %1, %1) : [2] f32
          %3 = Fma(%2, %1, %0) : [2] f32
          return %3
        }
        "#,
        r#"
        graph @chain {
          %0 = {"Input":{"name":"a"}} : [2] f32
          %1 = {"Input":{"name":"b"}} : [2] f32
          %2 = {"Binary":"Mul"}(%0, %1) : [2] f32
          %3 = {"Binary":"Add"}(%2, %1) : [2] f32
          %4 = {"Binary":"Mul"}(%3, %1) : [2] f32
          %5 = {"Binary":"Add"}(%4, %0) : [2] f32
          return %5
        }
        "#,
    );
}

/// The fixture format is only useful if it survives a round-trip; a pass that
/// emitted un-reprintable IR would make every fixture a lie.
#[test]
fn pass_output_reprints_and_reparses() {
    let graph = text::parse(
        r#"
        graph @rt {
          %0 = {"Input":{"name":"a"}} : [8] f32
          %1 = {"Input":{"name":"b"}} : [8] f32
          %2 = {"Input":{"name":"c"}} : [8] f32
          %3 = Fma(%0, %1, %2) : [8] f32
          return %3
        }
        "#,
    )
    .unwrap();

    let lowered = LowerFma.run(graph);
    let reparsed = text::parse(&text::print(&lowered)).unwrap();
    assert_eq!(lowered.fingerprint(), reparsed.fingerprint());
}
