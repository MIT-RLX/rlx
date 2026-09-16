// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The corpus gate. A compiler change that breaks a family fails here, with
//! the family named.

#[test]
fn corpus_is_clean() {
    let report = rlx_corpus::run();
    assert!(
        report.is_clean(),
        "corpus regressions:\n{}",
        report.render()
    );
    assert!(report.cases_run > 0, "corpus is empty");
    assert!(
        report.plans_checked >= report.cases_run,
        "each case must be planned"
    );
    eprintln!("{}", report.render());
}

#[test]
fn opkind_total_is_current() {
    // The coverage denominator is a hardcoded count of `OpKind` variants. If
    // ops are added and this is not updated, coverage silently overstates
    // itself. Re-derive from the source rather than trusting the constant.
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../core/rlx-ir/src/op.rs"
    ))
    .expect("op.rs readable");
    let body = src
        .split_once("pub enum OpKind {")
        .expect("OpKind enum present")
        .1
        .split_once("\n}")
        .expect("enum terminates")
        .0;
    let actual = body
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty()
                && !t.starts_with("//")
                && !t.starts_with("#[")
                && t.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        })
        .count();
    assert_eq!(
        actual,
        rlx_corpus::OPKIND_TOTAL,
        "OpKind has {actual} variants but OPKIND_TOTAL says {} — coverage is being \
         reported against a stale denominator",
        rlx_corpus::OPKIND_TOTAL
    );
}

#[test]
fn coverage_is_reported_and_not_overstated() {
    let (covered, total) = rlx_corpus::coverage();
    assert!(covered > 0, "corpus touches no ops");
    assert!(covered <= total, "covered {covered} exceeds total {total}");
    assert!(
        rlx_corpus::run().render().contains("UNCOVERED"),
        "the report must state that uncovered ops are uncovered, not passing"
    );
}
