// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A region that exceeds the kernel limits must be **declined**, not half-built.
//!
//! `MarkElementwiseRegions` replaces every non-tail member of a region with a
//! `NodeId(u32::MAX)` sentinel, on the promise that the tail will emit the real
//! region node and rewire them. The limit check used to live at the tail, and
//! its bail-out ran *after* those sentinels were written — so an over-limit
//! region left consumers pointing at a node that does not exist:
//!
//! ```text
//! IR verifier failed after pass `mark_elementwise_regions`:
//!   at %20: input %4294967295 references non-existent node (graph has 21 nodes)
//! ```
//!
//! Only `debug_assert_valid!` caught that, so a release build handed the
//! dangling operand to a backend. The check now runs in the analysis phase,
//! where regions are already dropped for being too small or forking — before
//! anything is rewritten.

use rlx_fusion::fusion::MarkElementwiseRegions;
use rlx_fusion::limits::{FusionLimits, with_fusion_limits};
use rlx_ir::{DType, Graph, Shape, op::BinaryOp};

/// A chain over `n_inputs` distinct inputs — the `max_elementwise_inputs` axis.
fn wide(n_inputs: usize) -> Graph {
    let s = Shape::new(&[64], DType::F32);
    let mut g = Graph::new("wide");
    let ids: Vec<_> = (0..n_inputs)
        .map(|i| g.input(format!("i{i}"), s.clone()))
        .collect();
    let mut cur = ids[0];
    for (k, id) in ids.iter().enumerate().skip(1) {
        let op = if k % 2 == 0 {
            BinaryOp::Add
        } else {
            BinaryOp::Mul
        };
        cur = g.binary(op, cur, *id, s.clone());
    }
    g.set_outputs(vec![cur]);
    g
}

/// A chain of `steps` ops — the `max_elementwise_steps` axis.
fn deep(steps: usize) -> Graph {
    let s = Shape::new(&[64], DType::F32);
    let mut g = Graph::new("deep");
    let x = g.input("x", s.clone());
    let y = g.input("y", s.clone());
    let mut cur = x;
    for i in 0..steps {
        let op = if i % 2 == 0 {
            BinaryOp::Add
        } else {
            BinaryOp::Sub
        };
        cur = g.binary(op, cur, y, s.clone());
    }
    g.set_outputs(vec![cur]);
    g
}

fn mark(g: Graph) -> Graph {
    with_fusion_limits(FusionLimits::GPU_NATIVE, || {
        rlx_fusion::pass::run_passes(g, &[&MarkElementwiseRegions], false)
    })
}

fn regions(g: &Graph) -> usize {
    g.nodes()
        .iter()
        .filter(|n| format!("{:?}", n.op.kind()).contains("ElementwiseRegion"))
        .count()
}

/// Every emitted region must be within what a kernel can express.
///
/// Asserted instead of "the over-limit chain produces zero regions", because
/// the pass is free to split a long chain into several admissible regions —
/// and that is a better outcome than declining to fuse it at all. What must
/// never happen is an emitted region that exceeds the caps, or a graph that
/// does not verify.
fn assert_every_region_within(g: &Graph, limits: FusionLimits) {
    for n in g.nodes() {
        if let rlx_ir::Op::ElementwiseRegion {
            chain, num_inputs, ..
        } = &n.op
        {
            assert!(
                chain.len() as u32 <= limits.max_elementwise_steps,
                "region at {:?} has {} steps > cap {}",
                n.id,
                chain.len(),
                limits.max_elementwise_steps
            );
            assert!(
                *num_inputs <= limits.max_elementwise_inputs,
                "region at {:?} has {num_inputs} inputs > cap {}",
                n.id,
                limits.max_elementwise_inputs
            );
        }
    }
}

#[test]
fn an_over_wide_region_is_declined_and_the_graph_still_verifies() {
    // 16 inputs is the cap, so this one is admissible and should fuse.
    let ok = mark(wide(16));
    assert!(
        rlx_ir::verify::verify(&ok).is_empty(),
        "16 inputs must verify"
    );
    assert_eq!(regions(&ok), 1, "16 distinct inputs is within the cap");

    // 20 exceeds it. The old code emitted a dangling operand here.
    let over = mark(wide(20));
    let errs = rlx_ir::verify::verify(&over);
    assert!(
        errs.is_empty(),
        "an over-limit region must be declined, not half-built: {errs:?}"
    );
    assert_every_region_within(&over, FusionLimits::GPU_NATIVE);
}

#[test]
fn an_over_deep_region_is_declined_and_the_graph_still_verifies() {
    let ok = mark(deep(32));
    assert!(rlx_ir::verify::verify(&ok).is_empty());
    assert_eq!(regions(&ok), 1, "32 steps is exactly the cap");

    // A 48-step chain may legitimately become several admissible regions
    // rather than none — what matters is that none of them exceeds the cap and
    // the graph still verifies.
    let over = mark(deep(48));
    let errs = rlx_ir::verify::verify(&over);
    assert!(errs.is_empty(), "over-deep chain must stay valid: {errs:?}");
    assert_every_region_within(&over, FusionLimits::GPU_NATIVE);
}
