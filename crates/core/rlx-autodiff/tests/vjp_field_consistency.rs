// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **P7 gate — analysis-consistency between the IR data model and the VJPs.**
//!
//! CAKE's Appendix B.1 lists eight IR design principles. P7 is:
//!
//! > **Analysis-consistent.** Accompany changes to the IR data model with
//! > corresponding analysis updates.
//!
//! That is the RoPE bug stated as a rule. `Op::Rope` gained a `style` field;
//! `vjp_rope` destructured it as `Op::Rope { head_dim, n_rot, .. }`; the `..`
//! dropped the pairing convention on the floor. Every `RopeStyle::GptJ` rotation
//! — the GGUF convention — then received a NeoX adjoint. The forward was right,
//! so nothing downstream complained; the gradient was simply wrong, on every
//! backend, for full and partial rotary alike.
//!
//! A convention cannot prevent that. `..` is *silent* by design: it compiles
//! before and after the field is added, and there is no diagnostic. So this test
//! makes the elision explicit instead — every VJP that discards an op field must
//! appear below with a written reason. Adding a field to an op whose VJP elides
//! is then still silent at the type level, but adding a *new elision* fails here
//! until someone says why it is safe.
//!
//! ## What this does and does not prove
//!
//! It does not prove the waived VJPs are correct. It proves the set of VJPs that
//! throw information away is **known and justified** rather than accidental — and
//! it makes growing that set a deliberate act. Numeric correctness is the
//! finite-difference gate's job (`rlx-runtime/tests/fd_backward_gate.rs`), which
//! is where the RoPE defect was actually caught; the two are complements, not
//! substitutes.
//!
//! Same shape as `rlx-cuda/tests/launch_arity.rs`: a static scan over the tree's
//! own source, so it cannot drift out of sync with the code it describes.

use std::collections::BTreeMap;

/// A VJP allowed to discard op fields, and why.
///
/// Adding an entry is cheap; the point is that it is *written down*. If you are
/// here because the test failed, the question to answer is: **could the field I
/// am eliding change this gradient?** If yes, destructure it. If no, say so.
struct Waiver {
    /// `vjp_*` function name.
    func: &'static str,
    /// Why discarding the field(s) cannot change the gradient.
    reason: &'static str,
}

/// The audited set. Every entry carries a derivation.
///
/// Two of these (`vjp_scaled_quantize`, `vjp_scaled_quant_scale`) shipped briefly
/// marked `UNVERIFIED` — listed rather than quietly omitted, because an
/// unexamined elision is exactly the state the RoPE bug lived in. Both are now
/// derived *and* pinned by
/// `tests/scaled_quant_vjp_field_invariance.rs`, which tests the invariance the
/// waiver claims instead of trusting the prose. If a reason ever has to be
/// weakened again, mark it `UNVERIFIED` — the ratchet below will notice.
const WAIVERS: &[Waiver] = &[
    Waiver {
        func: "vjp_reshape",
        reason: "Op::Reshape's only field is the OUTPUT shape. The adjoint reshapes \
                 back to the INPUT's shape, which it reads from the graph, so the \
                 field is genuinely unused.",
    },
    Waiver {
        func: "vjp_cast",
        reason: "Op::Cast's only field is the target dtype. The adjoint casts back to \
                 the INPUT's dtype, read from the graph, so the field is unused.",
    },
    Waiver {
        func: "vjp_expand",
        reason: "Op::Expand's only field is `target_shape` (the output). The adjoint \
                 reduce-sums the cotangent back to the input's shape, read from the \
                 graph, so the field is unused.",
    },
    Waiver {
        func: "vjp_fake_quantize",
        reason: "Destructures bits/axis/ste — everything the straight-through \
                 estimator depends on. Only `scale_mode` is elided, and it selects how \
                 the FORWARD picks its scale, not how the STE routes the cotangent.",
    },
    Waiver {
        func: "vjp_gaussian_splat_render",
        reason: "Destructures every rasterizer parameter it forwards to the backward \
                 op; the trailing elision covers fields the backward op does not take.",
    },
    Waiver {
        func: "vjp_gaussian_splat_render_backward",
        reason: "Second-order rasterizer term is not implemented; the arm exists to \
                 give a clear unsupported path rather than a wrong gradient, so no \
                 field can affect its result.",
    },
    Waiver {
        func: "vjp_custom_fn",
        reason: "Op::CustomFn carries a caller-supplied VJP body. Its remaining fields \
                 are opaque to autodiff BY CONSTRUCTION — the caller owns the \
                 gradient.",
    },
    Waiver {
        func: "vjp_custom_fn_2",
        reason: "The no-VJP-body arm of Op::CustomFn: there is no gradient to get \
                 wrong, only an error to report.",
    },
    Waiver {
        func: "vjp_custom",
        reason: "Op::Custom is an extension seam keyed by name; its attrs are an \
                 opaque byte blob autodiff cannot interpret.",
    },
    Waiver {
        func: "vjp_synth_reconstruct",
        reason: "Elision is inside a nested SynthKind::Codebook pattern; `entry_dim` \
                 is the only member the reconstruction adjoint uses.",
    },
    Waiver {
        func: "vjp_scaled_quantize",
        reason: "Returns `vec![(0, upstream)]` — a straight-through estimator. The \
                 result is the cotangent itself, so it cannot depend on `format` or \
                 `scale_layout`, which only choose the forward's byte encoding. \
                 Verified empirically by tests/scaled_quant_vjp_field_invariance.rs \
                 across FP8/FP6, OCP/FNUZ and per-tensor/block/NVFP4 layouts.",
    },
    Waiver {
        func: "vjp_scaled_quant_scale",
        reason: "Returns `vec![]` — the scale is a detached amax statistic, so there \
                 is no gradient for a field to influence. Verified by \
                 scaled_quant_vjp_field_invariance.rs, which asserts strict autodiff \
                 REFUSES to build a backward pass through it.",
    },
];

/// Source of the VJP rules.
fn autodiff_src() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/autodiff.rs");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Every `vjp_*` function that destructures `&node.op` while eliding fields,
/// mapped to the variant(s) it elides.
fn elisions(src: &str) -> BTreeMap<String, Vec<String>> {
    let lines: Vec<&str> = src.lines().collect();
    // Function boundaries: a top-level `fn vjp_...(`.
    let mut bounds: Vec<(String, usize)> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if let Some(rest) = l.strip_prefix("fn vjp_")
            && let Some(paren) = rest.find('(')
        {
            bounds.push((format!("vjp_{}", &rest[..paren]), i));
        }
    }
    bounds.push(("<end>".to_string(), lines.len()));

    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for w in bounds.windows(2) {
        let (name, start) = (&w[0].0, w[0].1);
        let end = w[1].1;
        let body = lines[start..end].join("\n");
        // `let Op::<Variant> { <fields> } = &node.op`
        let mut rest = body.as_str();
        while let Some(p) = rest.find("let Op::") {
            let after = &rest[p + "let Op::".len()..];
            let Some(brace) = after.find('{') else { break };
            let variant: String = after[..brace]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            // Match to the `= &node.op` that closes this destructure.
            let Some(eq) = after.find("= &node.op") else {
                break;
            };
            if eq > brace {
                let fields = &after[brace..eq];
                if fields.contains("..") {
                    out.entry(name.clone()).or_default().push(variant);
                }
            }
            rest = &after[brace + 1..];
        }
    }
    out
}

/// The scanner must actually find things — a regex that silently matches nothing
/// would make this test vacuously green, which is the failure mode a static lint
/// is most prone to.
#[test]
fn scanner_finds_the_vjp_functions_it_claims_to_scan() {
    let src = autodiff_src();
    let n_vjp = src.lines().filter(|l| l.starts_with("fn vjp_")).count();
    assert!(
        n_vjp > 50,
        "expected many vjp_* functions, found {n_vjp} — the scanner or the file moved"
    );
    let found = elisions(&src);
    assert!(
        !found.is_empty(),
        "the scanner found zero elisions; if that is real, delete the waiver list, \
         but far more likely the pattern stopped matching"
    );
}

/// **The gate.** Every VJP that discards an op field must be waived with a
/// reason, and every waiver must correspond to a real elision.
#[test]
fn every_vjp_field_elision_is_explicitly_waived() {
    let src = autodiff_src();
    let found = elisions(&src);

    let waived: BTreeMap<&str, &str> = WAIVERS.iter().map(|w| (w.func, w.reason)).collect();

    // 1. No unwaived elision.
    let mut unwaived: Vec<String> = Vec::new();
    for (func, variants) in &found {
        if !waived.contains_key(func.as_str()) {
            unwaived.push(format!(
                "{func} (elides fields of Op::{})",
                variants.join(", Op::")
            ));
        }
    }
    assert!(
        unwaived.is_empty(),
        "these VJP rules discard op fields with `..` and are not waived:\n  {}\n\n\
         This is the RoPE `style` defect's exact shape: an op gains a field, the VJP's \
         `..` swallows it, and the gradient goes wrong on every backend while the \
         forward stays right. Either destructure the field, or add a Waiver in \
         tests/vjp_field_consistency.rs saying why it cannot affect the gradient.",
        unwaived.join("\n  ")
    );

    // 2. No stale waiver. A waiver for a VJP that no longer elides is a claim
    //    nobody is checking, and it would mask a future re-introduction.
    let stale: Vec<&str> = waived
        .keys()
        .filter(|f| !found.contains_key(**f))
        .copied()
        .collect();
    assert!(
        stale.is_empty(),
        "these waivers no longer correspond to an elision — delete them: {stale:?}"
    );
}

/// Waiver reasons must say something. An empty or one-word reason defeats the
/// purpose, which is to force the author to answer "could this field change the
/// gradient?".
#[test]
fn waiver_reasons_are_substantive() {
    for w in WAIVERS {
        assert!(
            w.reason.len() > 40,
            "waiver for {} has no real reason: {:?}",
            w.func,
            w.reason
        );
    }
}

/// Surface the unverified waivers as a count rather than letting them blend in.
/// This is Appendix C's discipline — report the coverage limitation instead of
/// treating it as a pass. The assertion is a ratchet: it fails if the number
/// GROWS, so an unexamined elision cannot quietly accumulate.
#[test]
fn unverified_waivers_are_reported_and_do_not_grow() {
    let unverified: Vec<&str> = WAIVERS
        .iter()
        .filter(|w| w.reason.contains("UNVERIFIED"))
        .map(|w| w.func)
        .collect();
    eprintln!(
        "P7 gate: {} of {} waivers carry an argument; {} are UNVERIFIED: {unverified:?}",
        WAIVERS.len() - unverified.len(),
        WAIVERS.len(),
        unverified.len()
    );
    assert!(
        unverified.is_empty(),
        "unverified elisions reappeared: {unverified:?}. Every waiver must carry a \
         derivation — and ideally a test that pins it, the way \
         scaled_quant_vjp_field_invariance.rs pins the ScaledQuantize pair. Deriving \
         the argument is the work; the waiver is only the record of it.",
    );
}
