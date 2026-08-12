// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Pass::trigger_kinds` is a correctness claim, so it gets checked.
//!
//! Declaring a trigger kind asserts the pass is a no-op on any graph that does
//! not contain it. The runner acts on that: it skips re-verification, keeps
//! cached analyses alive, and reports `Unchanged`. A pass that quietly *did*
//! need to fire would be skipped, and the symptom would surface far away — a
//! legalization failure on some backend, or a wrong result from a lowering
//! that never ran.
//!
//! Rather than trust the declarations, this exercises them: run each pass over
//! a graph built only from ops outside its trigger set, and require the IR to
//! come back untouched.

use rlx_fusion::control_flow::{LowerControlFlow, LowerScan};
use rlx_fusion::fusion::{
    FuseAdaLayerNorm, FuseAttentionBlock, FuseConvAffineAct, FuseConvBiasAct, FuseMatMulBiasAct,
    FuseMatMulResidual, FuseResidualLN, FuseResidualRmsNorm, FuseRmsNormReshape,
    FuseSharedInputMatMul, FuseSwiGLU, FuseSwiGLUDualMatmul, FuseTransformerLayer,
    UnfuseElementwiseRegions,
};
use rlx_fusion::lower_axial_rope2d::LowerAxialRope2d;
use rlx_fusion::lower_backward_ops::LowerBackwardOps;
use rlx_fusion::lower_cumulative::LowerCumulative;
use rlx_fusion::lower_dot_general::LowerDotGeneral;
use rlx_fusion::lower_fake_quantize::LowerFakeQuantize;
use rlx_fusion::lower_fma::LowerFma;
use rlx_fusion::lower_histogram::LowerHistogram;
use rlx_fusion::lower_loss_ops::LowerSoftmaxCrossEntropy;
use rlx_fusion::lower_pad::LowerPad;
use rlx_fusion::lower_reduce_axes::LowerNonLastAxisReduce;
use rlx_fusion::lower_scaled_grouped_matmul::LowerScaledGroupedMatMul;
use rlx_fusion::lower_slice::LowerSlice;
use rlx_fusion::lower_spectral::LowerSpectral;
use rlx_fusion::lower_spline_activation::LowerSplineActivation;
use rlx_fusion::lower_spline_backward::LowerSplineActivationBackward;
use rlx_fusion::lower_structural::LowerStructural;
use rlx_fusion::lower_synth_matmul::LowerSynthMatMul;
use rlx_fusion::lower_synth_matmul_backward::LowerSynthMatMulBackward;
use rlx_fusion::lower_synth_reconstruct::LowerSynthReconstruct;
use rlx_fusion::lower_vae_ops::{LowerBatchNormInference, LowerGroupNorm, LowerResizeNearest2x};
use rlx_fusion::pass::{IRStatus, Pass};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, IgnoreConfig, Op, Shape};

/// Every pass that declares triggers, so a new declaration is covered by
/// adding one line here.
fn passes_with_triggers() -> Vec<&'static dyn Pass> {
    vec![
        &LowerAxialRope2d,
        &LowerBackwardOps,
        &LowerCumulative,
        &LowerDotGeneral,
        &LowerFakeQuantize,
        &LowerFma,
        &LowerHistogram,
        &LowerSoftmaxCrossEntropy,
        &LowerPad,
        &LowerNonLastAxisReduce,
        &LowerScaledGroupedMatMul,
        &LowerSlice,
        &LowerSpectral,
        &LowerSplineActivation,
        &LowerSplineActivationBackward,
        &LowerStructural,
        &LowerSynthMatMul,
        &LowerSynthMatMulBackward,
        &LowerSynthReconstruct,
        &LowerBatchNormInference,
        &LowerGroupNorm,
        &LowerResizeNearest2x,
        &LowerScan,
        &LowerControlFlow,
        &UnfuseElementwiseRegions::FOR_CPU,
        // Fusion passes: their anchors are derived from each matcher's own
        // scan, so the same claim applies and gets the same check.
        &FuseAdaLayerNorm,
        &FuseAttentionBlock,
        &FuseConvBiasAct,
        &FuseConvAffineAct,
        &FuseMatMulBiasAct,
        &FuseMatMulResidual,
        &FuseResidualLN,
        &FuseResidualRmsNorm,
        &FuseRmsNormReshape,
        &FuseSharedInputMatMul,
        &FuseSwiGLU,
        &FuseSwiGLUDualMatmul,
        &FuseTransformerLayer,
    ]
}

/// Elementwise arithmetic only — deliberately free of every kind any pass in
/// the list below declares as a trigger, so each check is non-vacuous.
fn ordinary_graph() -> Graph {
    let mut g = Graph::new("ordinary");
    let s = Shape::new(&[4, 16], DType::F32);
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let sum = g.binary(BinaryOp::Add, x, y, s.clone());
    let act = g.add_node(Op::Activation(Activation::Gelu), vec![sum], s.clone());
    let prod = g.binary(BinaryOp::Mul, act, x, s.clone());
    let res = g.binary(BinaryOp::Sub, prod, y, s);
    g.set_outputs(vec![res]);
    g
}

#[test]
fn no_pass_declares_a_trigger_it_does_not_need() {
    let graph = ordinary_graph();

    for pass in passes_with_triggers() {
        let triggers = pass.trigger_kinds();
        assert!(
            !triggers.is_empty(),
            "`{}` is listed here but declares no triggers",
            pass.name()
        );

        // Precondition: the fixture really does avoid this pass's triggers.
        assert!(
            !graph
                .nodes()
                .iter()
                .any(|n| triggers.contains(&n.op.kind())),
            "fixture contains a trigger kind of `{}` — the check would be vacuous",
            pass.name()
        );

        // The claim: with no trigger present, the pass changes nothing.
        let out = pass.run(graph.clone());
        assert!(
            graph.structurally_eq(&out, IgnoreConfig::SEMANTIC),
            "`{}` declares triggers {:?} but modified a graph containing none of them",
            pass.name(),
            triggers
        );

        // ...and the runner is therefore entitled to skip it.
        assert_eq!(
            pass.run_with_status(graph.clone()).ir_changed,
            IRStatus::Unchanged,
            "`{}` must report Unchanged when its trigger is absent",
            pass.name()
        );
    }
}

#[test]
fn a_declared_trigger_makes_can_fire_true() {
    // The other direction: presence of a trigger must not be filtered out.
    let mut g = Graph::new("padded");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let p = g.add_node(
        Op::Pad {
            pads: vec![[1, 1]],
            mode: rlx_ir::PadMode::Constant(0.0),
        },
        vec![x],
        Shape::new(&[6], DType::F32),
    );
    g.set_outputs(vec![p]);

    assert!(LowerPad.can_fire(&g));
    assert!(!LowerSlice.can_fire(&g));
}

/// A declared trigger must actually *short-circuit*, not merely coincide with
/// "the matcher found nothing".
///
/// This distinction bit once already: after the fusion pass bodies were
/// extracted into `fuse_with`, `run_with_status` called it directly and the
/// declarations were inert — every check still passed, because the passes
/// genuinely did not match. Observing the short-circuit needs a side channel,
/// so this uses the analysis cache: a pass that returns on its trigger check
/// never asks for `UseCounts`, while one that runs its matcher does.
#[test]
fn a_missing_trigger_short_circuits_before_any_analysis() {
    use rlx_fusion::analysis::{AnalysisManager, OpKindIndex, UseCounts};

    let graph = ordinary_graph();
    let mut analyses = AnalysisManager::default();

    // Prime the index so the trigger check itself is a cache hit, and confirm
    // the fixture really lacks every anchor.
    assert!(
        !analyses
            .get::<OpKindIndex>(&graph)
            .contains(rlx_ir::OpKind::MatMul)
    );

    for pass in passes_with_triggers() {
        let result = pass.run_with_analyses(graph.clone(), &mut analyses);
        assert_eq!(
            result.ir_changed,
            rlx_fusion::pass::IRStatus::Unchanged,
            "`{}` should have short-circuited",
            pass.name()
        );
    }

    assert!(
        !analyses.is_cached::<UseCounts>(),
        "no pass fired, so none should have needed the use relation — a \
         declared trigger is not being consulted"
    );
}
