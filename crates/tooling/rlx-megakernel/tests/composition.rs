// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! How far rlx composes an MoE block, and whether composing changes the answer.

use rlx_ir::Op;
use rlx_megakernel::{compose, compose_graph, dispatch_count, moe_block};
use rlx_opt::rlx_compile::fusion_pipeline::FusionTarget;
use rlx_runtime::{Device, Session};

const TARGETS: &[FusionTarget] = &[
    FusionTarget::Cpu,
    FusionTarget::Metal,
    FusionTarget::Cuda,
    FusionTarget::Wgpu,
];

#[test]
fn report_moe_composition_across_targets() {
    let g = moe_block(8, 32, 64, 128);
    eprintln!(
        "MoE block: 8 experts, 32 tokens, d=64, ff=128 — {} dispatch(es) unfused",
        dispatch_count(&g)
    );
    for &t in TARGETS {
        let c = compose(&g, t);
        eprintln!(
            "  {:?}: {} -> {} dispatch(es) ({:.0}% removed), arithmetic ops {} -> {}{}",
            t,
            c.before,
            c.after,
            100.0 * c.reduction(),
            c.heavy_before,
            c.heavy_after,
            if c.is_megakernel() {
                "  [MEGAKERNEL]"
            } else {
                ""
            }
        );
    }
}

#[test]
fn fusion_never_increases_the_arithmetic_op_count() {
    // The invariant that actually holds, and the one worth gating.
    //
    // Raw dispatch count is NOT monotone and asserting it was wrong: rlx fuses
    // the two projections sharing an input into one Concat + MatMul + two
    // Narrows, which takes a single-expert block from 3 matmuls to 2 while
    // raising the node count from 6 to 8. That is a good trade scored as a
    // regression by the naive metric.
    for experts in [1usize, 4, 8] {
        let g = moe_block(experts, 32, 64, 128);
        for &t in TARGETS {
            let c = compose(&g, t);
            assert!(
                c.heavy_after <= c.heavy_before,
                "{t:?} took {experts} expert(s) from {} to {} arithmetic op(s)",
                c.heavy_before,
                c.heavy_after
            );
        }
    }
}

#[test]
fn the_shared_input_projection_fusion_fires() {
    // Pins the transformation the metric above exists to accommodate, so a
    // silent loss of it shows up as a failure rather than as a "no change".
    let g = moe_block(1, 32, 64, 128);
    let c = compose(&g, FusionTarget::Cpu);
    assert_eq!(
        c.heavy_before, 3,
        "single expert should start with 3 matmuls"
    );
    assert_eq!(
        c.heavy_after, 2,
        "gate/up share an input and must fuse into one matmul; got {} \
         (dispatches {} -> {})",
        c.heavy_after, c.before, c.after
    );
}

#[test]
fn fusion_preserves_the_computed_value() {
    // Composition that changes the answer is not composition. Checked against
    // an INDEPENDENT f64 oracle rather than against the unfused rlx graph, so
    // a defect shared by both paths cannot hide.
    let g = moe_block(4, 8, 32, 64);
    let fused = compose_graph(&g, FusionTarget::Cpu);

    let fill = |n: usize, seed: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mag = 0.1 + (((i * 37 + seed * 101) % 90) as f32) * 0.01;
                if (i + seed).is_multiple_of(2) {
                    mag
                } else {
                    -mag
                }
            })
            .collect()
    };
    let elems = |graph: &rlx_ir::Graph, name: &str| -> usize {
        graph
            .nodes()
            .iter()
            .find(
                |n| matches!(&n.op, Op::Input { name: nm } | Op::Param { name: nm } if nm == name),
            )
            .map(|n| {
                n.shape
                    .dims()
                    .iter()
                    .map(|d| match d {
                        rlx_ir::Dim::Static(k) => *k,
                        rlx_ir::Dim::Dynamic(_) => 1,
                    })
                    .product()
            })
            .unwrap_or(0)
    };

    let mut inputs: Vec<(&str, Vec<f32>)> = Vec::new();
    let mut params: Vec<(&str, Vec<f32>)> = Vec::new();
    for (i, n) in g.nodes().iter().enumerate() {
        match &n.op {
            Op::Input { name } => inputs.push((
                Box::leak(name.clone().into_boxed_str()),
                fill(elems(&g, name), i),
            )),
            Op::Param { name } => params.push((
                Box::leak(name.clone().into_boxed_str()),
                fill(elems(&g, name), i + 7),
            )),
            _ => {}
        }
    }

    let run = |graph: rlx_ir::Graph| -> Vec<f32> {
        let mut exe = Session::new(Device::Cpu).compile(graph);
        for (n, v) in &params {
            exe.set_param(n, v);
        }
        let feed: Vec<(&str, &[f32])> = inputs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        exe.run(&feed)[0].clone()
    };

    let unfused_out = run(g.clone());
    let fused_out = run(fused);

    let tol = rlx_corpus::oracle::tolerance_for(&g);
    for (label, out) in [("unfused", &unfused_out), ("fused", &fused_out)] {
        let r = rlx_corpus::oracle::validate(&g, &inputs, &params, out);
        assert!(
            r.authority.is_independent(),
            "{label} was not checked against an independent authority: {}",
            r.detail
        );
        match r.max_rel_err {
            Some(e) => assert!(
                e <= tol,
                "{label} disagrees with the oracle: {e:.2e} > {tol:.1e}"
            ),
            None => panic!("{label}: oracle withheld a verdict — {}", r.detail),
        }
    }
}

/// The shared-input projection fusion has a cliff at three experts.
///
/// Measured on CPU: 1 expert 3 -> 2 matmuls, 2 experts 6 -> 3 (all four
/// projections merged into one), then **nothing at 3, 4 or 8** — 24 matmuls
/// stay 24. Since every expert's gate/up pair shares `x` exactly as the
/// one-expert case does, the transformation is applicable and is not being
/// applied.
///
/// Pinned as an observation, not asserted as a bug: the cause is not
/// established here, and a cap on how many consumers the weight-concat pass
/// will merge would be a legitimate design choice. What this test guarantees
/// is that the cliff cannot move without someone noticing.
#[test]
fn the_projection_fusion_has_a_cliff_at_three_experts() {
    let fired: Vec<(usize, usize, usize)> = [1usize, 2, 3, 4, 8]
        .into_iter()
        .map(|e| {
            let c = compose(&moe_block(e, 32, 64, 128), FusionTarget::Cpu);
            (e, c.heavy_before, c.heavy_after)
        })
        .collect();
    assert_eq!(
        fired,
        vec![(1, 3, 2), (2, 6, 3), (3, 9, 9), (4, 12, 12), (8, 24, 24)],
        "the fusion cliff moved; re-derive it rather than editing this expectation"
    );
}

#[test]
fn rlx_does_not_reach_a_megakernel_and_the_study_says_so() {
    // Pins the honest conclusion. If rlx ever does compose an MoE block into a
    // single device program, this fails and the claim in the crate docs has to
    // be rewritten — which is the correct outcome, not a nuisance.
    let g = moe_block(8, 32, 64, 128);
    for &t in TARGETS {
        let c = compose(&g, t);
        assert!(
            !c.is_megakernel(),
            "{t:?} reached {} dispatch(es) — rlx now composes a megakernel; update the study",
            c.after
        );
    }
}
