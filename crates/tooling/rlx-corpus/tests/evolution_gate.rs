// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The ledger's own invariants. Not "no ungated defects" — that would be a
//! lie-generator. What is gated is that the ledger stays honest and its named
//! mechanisms stay real.

use rlx_corpus::evolution::{Disposition, LEDGER, report, ungated};

#[test]
fn every_entry_names_a_symptom_and_a_disposition() {
    for e in LEDGER {
        assert!(!e.defect.is_empty(), "entry with no slug");
        assert!(
            e.symptom.len() > 20,
            "{}: a one-word symptom is not evidence",
            e.defect
        );
        if e.disposition.is_gated() {
            assert!(
                e.disposition.mechanism().len() > 3,
                "{}: claims to be gated but names no mechanism",
                e.defect
            );
        }
    }
}

#[test]
fn slugs_are_unique() {
    let mut seen = std::collections::BTreeSet::new();
    for e in LEDGER {
        assert!(seen.insert(e.defect), "duplicate ledger slug: {}", e.defect);
    }
}

#[test]
fn the_ledger_records_the_backlog_rather_than_hiding_it() {
    let r = report();
    assert!(r.total > 0, "empty ledger");
    // The honest expectation: some defects are NOT gated, and the ledger must
    // surface them. A ledger reporting 100% coverage would mean entries were
    // being omitted, not that the tree was perfect.
    assert!(
        !ungated().is_empty(),
        "a ledger with no ungated entries is a ledger that stopped recording"
    );
    let text = rlx_corpus::evolution::render();
    assert!(text.contains("BACKLOG"), "the backlog must be printed");
    for e in ungated() {
        assert!(
            text.contains(e.defect),
            "ungated defect {} not surfaced",
            e.defect
        );
    }
}

#[test]
fn gated_entries_point_at_paths_or_symbols_that_look_real() {
    // Weak by design: this cannot verify a test exists without running it.
    // What it does catch is a mechanism field filled with prose instead of a
    // pointer, which is how a ledger degrades into a changelog.
    for e in LEDGER {
        if let Disposition::StaticGate(m) | Disposition::RegressionTest(m) = e.disposition {
            assert!(
                m.contains("::") || m.contains('/') || m.contains(".rs") || m.contains(".py"),
                "{}: mechanism {m:?} is prose, not a pointer",
                e.defect
            );
        }
    }
}
