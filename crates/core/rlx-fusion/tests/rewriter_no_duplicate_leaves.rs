// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fusion pass must never emit two leaves sharing a name.
//!
//! `Rewriter::ensure_mapped` hoists a fused node's operands to the fusion site,
//! copying them ahead of their position. The pass's main loop then walks the
//! rest of the old graph and copies every node it did not fuse — reaching those
//! same operands a second time. Unless `copy_node` refuses to copy an
//! already-mapped node, that second call emits a duplicate and repoints
//! `id_map` at it, so every later consumer reads the duplicate.
//!
//! For a `Param` that is silent corruption rather than wasted work: binding is
//! by name and reaches one node, so the duplicate keeps the arena's zeros. A
//! forward graph hides it — the fused-away matmul was the param's only reader,
//! so the duplicate is dead code — but a backward graph reads each weight twice
//! (mirrored forward, then `dX = dY · Wᵀ`), and there the duplicate is live.
//!
//! `verify_unique_leaf_names` is the check; this pins the pass that broke it.

use rlx_fusion::FuseSharedInputMatMul;
use rlx_fusion::pass::Pass;
use rlx_ir::op::BinaryOp;
use rlx_ir::{DType, Graph, Shape, verify_unique_leaf_names};

/// Two matmuls sharing an input (so `FuseSharedInputMatMul` groups them), with
/// the second weight read *again* somewhere else — the shape a backward graph
/// has, and the shape that exposes the duplication.
fn shared_input_graph() -> Graph {
    let f = DType::F32;
    let mut g = Graph::new("shared_input");
    let x = g.input("x", Shape::new(&[2, 3], f));
    let w1 = g.param("w1", Shape::new(&[3, 4], f));
    let a = g.matmul(x, w1, Shape::new(&[2, 4], f));
    // Declared after the first matmul, so the fused node's operand hoist has to
    // pull it backwards past `a`.
    let w2 = g.param("w2", Shape::new(&[3, 4], f));
    let b = g.matmul(x, w2, Shape::new(&[2, 4], f));
    // The second, independent read of w2. Without it the duplicate is dead and
    // the bug is invisible.
    let w2_sq = g.binary(BinaryOp::Mul, w2, w2, Shape::new(&[3, 4], f));
    g.set_outputs(vec![a, b, w2_sq]);
    g
}

#[test]
fn fuse_shared_input_matmul_does_not_duplicate_params() {
    let graph = shared_input_graph();
    assert!(
        verify_unique_leaf_names(&graph).is_empty(),
        "the input graph should be well-formed to begin with"
    );

    let fused = FuseSharedInputMatMul.run(graph);
    let errors = verify_unique_leaf_names(&fused);
    assert!(
        errors.is_empty(),
        "fusion emitted duplicate leaves: {}",
        errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    );
}

/// The pass must still fire — otherwise the test above passes vacuously.
#[test]
fn fuse_shared_input_matmul_still_fires() {
    let fused = FuseSharedInputMatMul.run(shared_input_graph());
    let concats = fused
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, rlx_ir::Op::Concat { .. }))
        .count();
    assert_eq!(
        concats, 1,
        "expected the two weights to be concatenated into one matmul"
    );
}
