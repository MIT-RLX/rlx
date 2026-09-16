// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Closes the two `UNVERIFIED` waivers in `vjp_field_consistency.rs` by
//! **testing** the claim they make instead of asserting it.
//!
//! `vjp_scaled_quantize` destructures `Op::ScaledQuantize { .. }`, discarding
//! `format` and `scale_layout`; `vjp_scaled_quant_scale` does the same for
//! `Op::ScaledQuantScale`. The P7 gate demands a reason. Reading the rules gives
//! one:
//!
//! * `vjp_scaled_quantize` returns `vec![(0, upstream)]` — a **straight-through
//!   estimator**. The cotangent reaches input 0 unchanged.
//! * `vjp_scaled_quant_scale` returns `vec![]` — the scale is a detached
//!   statistic (amax), so there is no gradient at all.
//!
//! Neither result mentions a field, so no field can change it. That is an
//! argument from reading the source, which is exactly the kind of claim that
//! rots: someone implements a real quantizer derivative, the argument silently
//! stops holding, and the waiver still says it does. So this file pins the
//! property empirically — **the gradient must be invariant under changing
//! `format` and `scale_layout`** — and fails the moment that stops being true.
//!
//! ## Why not a finite-difference case instead
//!
//! The obvious move would be to add `ScaledQuantize` to `fd_backward_gate`. That
//! would be wrong, and it is worth stating why rather than leaving the op looking
//! merely un-covered.
//!
//! A straight-through estimator is a deliberate *substitute* for the derivative,
//! not an approximation of it. The true derivative of a quantizer is zero almost
//! everywhere, with impulses at the step boundaries; the STE reports 1 instead so
//! that gradients flow through quantization-aware training at all. Central
//! differences measure the true derivative, so an FD check would compare ~0
//! against 1 and fail *by design* — it would be testing that rlx does not
//! implement quantization-aware training.
//!
//! Field-invariance is the property that is actually claimed, so that is the
//! property tested. Picking the instrument that matches the claim matters more
//! than reaching for the instrument that is already built.

use rlx_autodiff::{GradWithLossOptions, Wrt, grad_with_loss_wrt};
use rlx_ir::quant::{ScaleLayout, ScaledFormat};
use rlx_ir::{DType, Graph, Op, Shape};

const F: DType = DType::F32;
const N: usize = 12;

fn scaled_quantize_graph(format: ScaledFormat, scale_layout: ScaleLayout) -> Graph {
    let mut g = Graph::new("scaled_quantize");
    let x = g.input("x", Shape::new(&[N], F));
    // `ScaledQuantize` consumes the tensor and its precomputed scale, and emits
    // PACKED BYTES — the output dtype is U8, not F32. (Declaring F32 here is
    // rejected by the verifier: "declared [12] f32, inferred [12] u8". Worth
    // stating, because it is the reason the straight-through estimator has to
    // exist at all: there is no differentiable path through a byte packing.)
    let scale = g.param("scale", Shape::new(&[1], F));
    let y = g.add_node(
        Op::ScaledQuantize {
            format,
            scale_layout,
        },
        vec![x, scale],
        Shape::new(&[N], DType::U8),
    );
    g.set_outputs(vec![y]);
    g
}

fn grad_of(g: Graph, x0: &[f32], cot: &[f32], scale: &[f32]) -> Vec<f32> {
    let bwd = grad_with_loss_wrt(
        &g,
        &[Wrt::Leaf("x".into())],
        GradWithLossOptions::STRICT.with_aux(false),
    );
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(bwd);
    compiled.set_param("scale", scale);
    let outs = compiled.run(&[("x", x0), ("d_output", cot)]);
    outs.last().expect("backward produced no outputs").clone()
}

/// Every `(format, scale_layout)` pairing must produce the SAME gradient.
///
/// This is the waiver's claim, stated as a test. If a real quantizer derivative
/// ever lands, this fails and forces `vjp_field_consistency.rs` to be revisited —
/// which is the whole point of P7.
#[test]
fn scaled_quantize_gradient_is_invariant_under_format_and_layout() {
    let x0: Vec<f32> = (0..N).map(|i| 0.4 * ((i % 7) as f32 - 3.0)).collect();
    let cot: Vec<f32> = (0..N).map(|i| 0.5 + 0.25 * ((i % 5) as f32)).collect();
    let scale = [1.0f32];

    // Deliberately spans FP8 vs FP6, OCP vs AMD FNUZ, and per-tensor vs two
    // different block layouts — if any field reached the gradient, one of these
    // pairs would diverge.
    let combos = [
        (ScaledFormat::F8E4M3, ScaleLayout::PerTensor),
        (ScaledFormat::F8E5M2, ScaleLayout::PerTensor),
        (ScaledFormat::F8E4M3Fnuz, ScaleLayout::PerTensor),
        (ScaledFormat::F8E4M3, ScaleLayout::BlockMxE8M0 { block: 32 }),
        (ScaledFormat::F6E2M3, ScaleLayout::BlockMxE8M0 { block: 32 }),
        (ScaledFormat::F8E4M3, ScaleLayout::Nvfp4 { group: 16 }),
    ];

    let mut baseline: Option<(String, Vec<f32>)> = None;
    for (format, layout) in combos {
        let label = format!("{format:?}/{layout:?}");
        let g = grad_of(scaled_quantize_graph(format, layout), &x0, &cot, &scale);
        match &baseline {
            None => baseline = Some((label, g)),
            Some((base_label, base)) => {
                assert_eq!(
                    &g, base,
                    "ScaledQuantize gradient changed between {base_label} and {label} — \
                     a field DOES reach the gradient, so the `..` elision in \
                     vjp_scaled_quantize is no longer safe. Destructure it and update \
                     the waiver in tests/vjp_field_consistency.rs."
                );
            }
        }
    }

    // …and the STE really is the identity on the cotangent, which is the reason
    // no field can matter. Asserting the *value* (not just invariance) means a
    // rule that became field-independent-but-wrong is still caught.
    let (_, grad) = baseline.expect("at least one combo ran");
    assert_eq!(
        grad, cot,
        "the straight-through estimator must pass the cotangent through unchanged"
    );
}

/// `ScaledQuantScale` contributes NO gradient to its input, so there is nothing a
/// field could influence.
///
/// The property is asserted through STRICT mode rather than by reading a zero
/// vector: with `ScaledQuantScale` as the only consumer of `x`, strict autodiff
/// refuses to build a backward pass at all — "no gradient flowed to %0". That
/// refusal *is* the property, and it is a sharper signal than zeros would be
/// (zeros are also what a silently-dropped gradient looks like).
///
/// The scale is a detached statistic (amax); making it differentiable would be a
/// deliberate change, and it would break this test — which is the point.
#[test]
fn scaled_quant_scale_contributes_no_gradient() {
    for (format, layout) in [
        (ScaledFormat::F8E4M3, ScaleLayout::PerTensor),
        (ScaledFormat::F8E5M2, ScaleLayout::BlockMxE8M0 { block: 32 }),
    ] {
        let mut g = Graph::new("scaled_quant_scale");
        let x = g.input("x", Shape::new(&[N], F));
        let scale = g.add_node(
            Op::ScaledQuantScale {
                format,
                scale_layout: layout,
            },
            vec![x],
            Shape::new(&[1], F),
        );
        g.set_outputs(vec![scale]);

        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let built = std::panic::catch_unwind(|| {
            grad_with_loss_wrt(
                &g,
                &[Wrt::Leaf("x".into())],
                GradWithLossOptions::STRICT.with_aux(false),
            )
        });
        std::panic::set_hook(prev);

        let err = built.err().unwrap_or_else(|| {
            panic!(
                "{format:?}/{layout:?}: strict autodiff BUILT a backward pass through \
                 ScaledQuantScale — it now contributes a gradient, so the `..` elision \
                 in vjp_scaled_quant_scale must be re-derived"
            )
        });
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("no gradient flowed"),
            "{format:?}/{layout:?}: expected the no-gradient refusal, got: {msg}"
        );
    }
}
