// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gaps in the static gates, found by the corpus and pinned so they stay
//! visible.
//!
//! These are *characterization* tests: they assert the gap as it currently
//! exists, so closing it makes them fail and someone has to delete the entry
//! deliberately. That is the intended lifecycle, and `expand-from-non-unit-dim`
//! has now been through it — the gap it pinned is closed, so the entry below
//! asserts the **rule** instead of the hole, and the ledger records the
//! mechanism.
//!
//! There are no open pinned gaps in this file right now. That is not the same
//! as "the verifier is complete": see `no_pinned_gap_is_still_ungated` for the
//! part that is actually checked, and `rlx_corpus::evolution::ungated()` for
//! the defects that have no mechanism at all.

use rlx_ir::{DType, Graph, Op, Shape};

const F: DType = DType::F32;

fn expand(from: &[usize], to: &[i64]) -> Graph {
    let mut g = Graph::new("expand");
    let x = g.input("x", Shape::new(from, F));
    let dims: Vec<usize> = to.iter().map(|&d| d as usize).collect();
    let y = g.add_node(
        Op::Expand {
            target_shape: to.to_vec(),
        },
        vec![x],
        Shape::new(&dims, F),
    );
    g.set_outputs(vec![y]);
    g
}

/// **Graduated.** `Op::Expand` from a non-unit dimension is not a legal
/// broadcast, and now something static rejects it.
///
/// Found by writing a corpus case that expanded `[8,128] -> [8,256]`. All
/// three device-free gates accepted it — structural verify, shape verify and
/// the memory-plan checks — and the graph then reached the CPU backend, which
/// panicked with `index out of bounds: the len is 1024 but the index is 1024`
/// deep inside `exec_dispatch`.
///
/// A backend panic is exactly the "opaque runtime crash" CAKE §3.2 says should
/// become a verifier rule. It is now one, in `rlx_ir::verify` — the rule itself
/// was never missing, only dropped: `expand_shape` already computed it and
/// `infer_shape` discarded it at a `.ok()`.
#[test]
fn expand_from_a_non_unit_dim_is_rejected() {
    let g = expand(&[8, 128], &[8, 256]);

    let structural = rlx_ir::verify::verify(&g);
    assert!(
        !structural.is_empty(),
        "the rule regressed — this is the corpus case that panicked in rlx-cpu \
         with an out-of-bounds read after clearing every static gate"
    );

    // Shape verify catches it too, by the second half of the same fix: the
    // node's declared shape IS what a legal expand to [8,256] would produce,
    // so the old "declared vs inferred" comparison could never have found it.
    // What finds it is `expand_shape`'s rejection now reaching the verifier
    // instead of being dropped at a `.ok()`.
    let shapes = rlx_ir::verify::verify_shapes(&g);
    assert_eq!(shapes.len(), 1, "{shapes:?}");
    assert!(shapes[0].message.contains("cannot broadcast"), "{shapes:?}");

    // repr_check is genuinely unaffected — it reasons about representation,
    // not operand legality — so it still passes. Recorded so a future change
    // that makes it fail is noticed rather than assumed.
    assert!(rlx_ir::repr_check::check_graph(&g).findings.is_empty());
}

/// The legal form is accepted, so the test above is about the illegal case and
/// not about `Expand` being unchecked in general.
#[test]
fn expand_from_a_unit_dim_is_accepted() {
    let g = expand(&[1, 128], &[8, 128]);
    assert!(rlx_ir::verify::verify(&g).is_empty());
    assert!(rlx_ir::verify::verify_shapes(&g).is_empty());
}

/// A pinned gap and the ledger must agree.
///
/// The characterization tests above are the corpus's half of the bookkeeping;
/// `rlx_corpus::evolution::LEDGER` is the project's half. Graduating a gap
/// means editing both, and editing only one is the easy mistake — a test that
/// now asserts a rule while the ledger still reports the defect as `UNGATED`
/// makes the backlog count wrong in the direction that matters (it overstates
/// the risk, so the real backlog gets less attention).
#[test]
fn no_pinned_gap_is_still_ungated() {
    let still_ungated = rlx_corpus::evolution::ungated()
        .into_iter()
        .find(|e| e.defect == "expand-from-non-unit-dim");
    assert!(
        still_ungated.is_none(),
        "`expand_from_a_non_unit_dim_is_rejected` asserts the rule exists, but the \
         evolution ledger still lists expand-from-non-unit-dim as Ungated. One of \
         the two is wrong."
    );
}
