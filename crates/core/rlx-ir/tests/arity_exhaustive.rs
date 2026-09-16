// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "test-support")]
// Without `test-support` there is no `sample_ops` to iterate, and an
// integration test compiles regardless of features — an ungated one breaks the
// default build for everyone.

//! `Op::arity` checked over **every** `OpKind`, not the subset some test built.
//!
//! `rlx-corpus` reaches 13 of 187 kinds, so a wrong arity on any of the other
//! 174 was invisible: too strict and `verify` rejects valid graphs, too loose
//! and a malformed one sails through. `sample_ops::all` makes the whole op set
//! iterable, and the match behind it fails to compile when an op is added.

use rlx_ir::op::Arity;
use rlx_ir::sample_ops;
use std::collections::BTreeSet;

/// The sample table must cover the op set exactly — no gaps, no duplicates.
#[test]
fn every_opkind_has_exactly_one_sample() {
    let all = sample_ops::all();
    assert_eq!(
        all.len(),
        sample_ops::ALL_KINDS.len(),
        "sample table and kind list disagree"
    );

    let names: BTreeSet<String> = all.iter().map(|(k, _)| format!("{k:?}")).collect();
    assert_eq!(names.len(), all.len(), "ALL_KINDS contains a duplicate");

    // Re-derived from the source, the way `rlx-corpus` pins its coverage
    // denominator: a hand-maintained count is the thing that drifts.
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/op.rs"))
        .expect("op.rs readable");
    let body = src
        .split_once("pub enum OpKind {")
        .expect("OpKind enum present")
        .1
        .split_once("\n}")
        .expect("enum terminates")
        .0;
    let from_source = body
        .lines()
        .map(str::trim)
        .filter(|t| {
            !t.is_empty()
                && !t.starts_with("//")
                && !t.starts_with("#[")
                && t.chars().next().is_some_and(char::is_uppercase)
        })
        .count();
    assert_eq!(
        all.len(),
        from_source,
        "sample table covers {} kinds but op.rs declares {from_source}",
        all.len()
    );
}

/// A sample must map back to the kind it was requested for. A copy-paste in
/// the table would otherwise leave one kind untested and another doubled.
#[test]
fn every_sample_reports_its_own_kind() {
    for (kind, op) in sample_ops::all() {
        assert_eq!(
            op.kind(),
            kind,
            "sample for {kind:?} reports {:?}",
            op.kind()
        );
    }
}

/// An arity nothing can satisfy silently rejects every graph using that op.
#[test]
fn every_arity_is_satisfiable() {
    for (kind, op) in sample_ops::all() {
        let a = op.arity();
        if let Arity::Range { min, max } = a {
            assert!(min <= max, "{kind:?}: unsatisfiable {a:?}");
        }
        assert!(
            a.accepts(a.min_operands()),
            "{kind:?}: {a:?} rejects its own minimum"
        );
        if let Some(max) = a.max_operands() {
            assert!(a.accepts(max), "{kind:?}: {a:?} rejects its own maximum");
            assert!(
                !a.accepts(max + 1),
                "{kind:?}: {a:?} accepts more than its maximum"
            );
        }
    }
}

/// `arity` must not panic for any kind.
///
/// The lifted ops (`Concat`, the `Rng`s, the leaves) are `unreachable!` inside
/// the fixed-arity table, so an edit that stops `arity` intercepting one turns
/// a silent wrong count into a panic — which only shows up if something calls
/// `arity` on that kind. Nothing did, before this.
#[test]
fn arity_never_panics() {
    for (kind, op) in sample_ops::all() {
        let a = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op.arity()));
        assert!(a.is_ok(), "arity panicked for {kind:?}");
    }
}

/// Cross-check against an **independently maintained** table.
///
/// `capability::is_leaf` classifies every `OpKind`; `arity` is a separate
/// hand-written table. A node with no operands is exactly a leaf, so the two
/// must agree — and because neither is derived from the other, a disagreement
/// is a real defect in one of them rather than a tautology. This is the only
/// check here that could fail without `arity` itself being self-inconsistent.
#[test]
fn leaf_classification_agrees_with_zero_arity() {
    let mut bad = Vec::new();
    for (kind, op) in sample_ops::all() {
        let nullary = op.arity() == Arity::Exact(0);
        if op.is_leaf() != nullary {
            bad.push(format!(
                "{kind:?}: is_leaf={} but arity={}",
                op.is_leaf(),
                op.arity()
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "capability table and arity table disagree on {} op(s):\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}

/// `num_inputs` is documented as the arity's lower bound, across the whole op
/// set rather than the few ops a unit test names.
#[test]
fn num_inputs_is_the_arity_minimum_for_every_kind() {
    for (kind, op) in sample_ops::all() {
        assert_eq!(
            op.num_inputs(),
            op.arity().min_operands(),
            "{kind:?} disagrees"
        );
    }
}

// ── field-dependent arities ────────────────────────────────────────────────
//
// `sample_ops` pins one setting per op, so a field that changes the operand
// count is only covered at that one point. These lock the documented operand
// lists at every setting, so a table edit that quietly changes a count fails
// here instead of showing up as a rejected graph in some backend.

use rlx_ir::op::{MaskKind, Op, ScaleMode};
use rlx_ir::quant::{ScaleLayout, ScaledFormat};

fn n(op: Op) -> usize {
    op.arity().min_operands()
}

#[test]
fn optional_bias_adds_exactly_one_operand() {
    let smm = |has_bias| {
        n(Op::ScaledMatMul {
            lhs_format: ScaledFormat::F8E4M3,
            rhs_format: ScaledFormat::F8E4M3,
            scale_layout: ScaleLayout::PerTensor,
            has_bias,
        })
    };
    let sgmm = |has_bias| {
        n(Op::ScaledGroupedMatMul {
            lhs_format: ScaledFormat::F8E4M3,
            rhs_format: ScaledFormat::F8E4M3,
            scale_layout: ScaleLayout::PerTensor,
            has_bias,
        })
    };
    assert_eq!(smm(true), smm(false) + 1, "ScaledMatMul");
    assert_eq!(sgmm(true), sgmm(false) + 1, "ScaledGroupedMatMul");
}

/// A mask is a fourth operand only when it is a real tensor. `MaskKind::None`
/// and `Causal` are shape-free, so demanding one would reject every causal
/// attention graph.
#[test]
fn attention_mask_operand_tracks_mask_kind() {
    let att = |mask_kind| {
        n(Op::Attention {
            num_heads: 1,
            head_dim: 1,
            v_head_dim: None,
            mask_kind,
            score_scale: None,
            attn_logit_softcap: None,
        })
    };
    assert_eq!(att(MaskKind::Custom), att(MaskKind::None) + 1);
    assert_eq!(att(MaskKind::Bias), att(MaskKind::None) + 1);
    assert_eq!(att(MaskKind::Causal), att(MaskKind::None));
}

#[test]
fn fake_quantize_scale_mode_tracks_operands() {
    let fq = |scale_mode| {
        n(Op::FakeQuantize {
            bits: 8,
            axis: None,
            ste: rlx_ir::op::SteKind::Identity,
            scale_mode,
        })
    };
    // An observed/fixed scale arrives as an operand; a per-batch one does not.
    assert_eq!(fq(ScaleMode::Fixed), fq(ScaleMode::PerBatch) + 1);
}

/// GGUF K-quants carry their scales inside the packed bytes, so they take two
/// operands where the legacy Int8 schemes take four. Getting this backwards
/// silently reads a scale tensor that is not there.
#[test]
fn dequant_matmul_operands_follow_the_scheme() {
    let int8 = n(Op::DequantMatMul {
        scheme: rlx_ir::quant::QuantScheme::Int8Block { block_size: 32 },
    });
    assert_eq!(int8, 4, "Int8 scheme should take x, w, scale, zp");
}

/// Variadic-by-field ops must actually honour the field, or a graph with the
/// right operand count gets rejected.
#[test]
fn count_carrying_ops_honour_their_field() {
    for k in [0u32, 1, 5] {
        assert_eq!(
            n(Op::Custom {
                name: "c".into(),
                num_inputs: k,
                attrs: Vec::new(),
            }),
            k as usize,
            "Custom must report its declared operand count"
        );
    }
    for num_xs in [0u32, 1, 3] {
        assert_eq!(
            n(Op::ScanBackward {
                body_vjp: Box::new(rlx_ir::Graph::new("b")),
                length: 1,
                save_trajectory: false,
                num_xs,
                num_checkpoints: 0,
                forward_body: None,
            }),
            3 + num_xs as usize,
            "ScanBackward = init, trajectory, upstream, then each xs"
        );
    }
}

// ── the variants table guards itself ───────────────────────────────────────
//
// `sample_ops::variants` is hand-written: it flips the one field that moves an
// op's operand count. Two ways that goes wrong, with opposite symptoms.

/// Flipping the *wrong* field produces a second variant with the same arity —
/// so `max_operands` still understates the op, and the backend-indexing gate
/// reports a legitimate read as out of range. The extra variant makes the
/// mistake look handled, which is worse than not listing the op at all.
#[test]
fn every_multi_variant_kind_actually_changes_arity() {
    let mut useless = Vec::new();
    for &kind in sample_ops::ALL_KINDS {
        let vs = sample_ops::variants(kind);
        if vs.len() < 2 {
            continue;
        }
        let distinct: std::collections::BTreeSet<usize> =
            vs.iter().map(|op| op.arity().min_operands()).collect();
        if distinct.len() < 2 {
            useless.push(format!(
                "{kind:?} ({} variants, all arity {:?})",
                vs.len(),
                distinct
            ));
        }
    }
    assert!(
        useless.is_empty(),
        "variants() flips a field that does not move the operand count for:\n  {}",
        useless.join("\n  ")
    );
}

/// `COUNT_FROM_FIELD` exempts an op from the upper-bound check entirely, so a
/// kind listed there by mistake is a silent blind spot rather than noise.
/// Every entry must genuinely be unbounded, and nothing else may be.
#[test]
fn count_from_field_is_exactly_the_unbounded_set() {
    for &kind in sample_ops::COUNT_FROM_FIELD {
        assert_eq!(
            sample_ops::max_operands(kind),
            None,
            "{kind:?} is exempted as count-carrying but reports a bound"
        );
    }
    // An op is exempt from the upper-bound check only for one of two stated
    // reasons. Anything else reaching `None` is exempt by accident, and an op
    // exempted by accident is never checked at all.
    for &kind in sample_ops::ALL_KINDS {
        if sample_ops::max_operands(kind).is_some() {
            continue;
        }
        let by_field = sample_ops::COUNT_FROM_FIELD.contains(&kind);
        let variadic = matches!(sample_ops::sample(kind).arity(), Arity::AtLeast(_));
        assert!(
            by_field || variadic,
            "{kind:?} has no upper bound but is neither count-carrying nor \
             variadic — it would be silently exempt from every bound check"
        );
    }
}

/// A variant must still report the kind it was built from — flipping a field
/// must not turn one op into another.
#[test]
fn variants_never_change_the_kind() {
    for &kind in sample_ops::ALL_KINDS {
        for op in sample_ops::variants(kind) {
            assert_eq!(
                op.kind(),
                kind,
                "a variant of {kind:?} became {:?}",
                op.kind()
            );
        }
    }
}
