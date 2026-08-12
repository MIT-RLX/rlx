// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared graph cleanup before the fusion / backend pipeline.

use crate::CompileOptions;
use rlx_ir::Graph;
use rlx_opt::pass::Pass as _;

/// Param specialization, algebraic simplify, DCE, and constant folding.
pub fn precompile_cleanup(graph: Graph, options: &CompileOptions) -> Graph {
    // Decompose registered custom ops that opt into a `lower` rule into
    // primitives BEFORE fusion / legalize / kernel dispatch. Every backend that
    // whitelists `OpKind::Custom` (all of them) would otherwise pass legalize and
    // then hard-fail at kernel-dispatch time for a custom op it has no kernel
    // for; lowering here lets such an op run on any backend with no kernel. A
    // no-op unless the graph carries a custom op, and idempotent with the same
    // pass inside `rewrite_for_backend`.
    let mut graph = rlx_opt::lower_custom_ops(graph);
    if options
        .param_bindings
        .as_ref()
        .is_some_and(|b| !b.is_empty())
    {
        let bindings = options.param_bindings.as_ref().unwrap();
        graph = rlx_opt::specialize_params(&graph, bindings);
    }
    post_specialize_cleanup(graph, options)
}

/// DCE / fold after fusion — skips param specialization (already applied pre-fusion).
pub fn post_fusion_cleanup(graph: Graph, options: &CompileOptions) -> Graph {
    post_specialize_cleanup(graph, options)
}

fn post_specialize_cleanup(graph: Graph, options: &CompileOptions) -> Graph {
    let mut graph = rlx_opt::AlgebraicSimplify.run(graph);
    // Sparse conditional constant propagation. Runs after `AlgebraicSimplify`
    // (which manufactures constants via `mul(x, 0)` and friends) and before
    // `DeadCodeElimination` (which then collects the branch a resolved `Where`
    // no longer selects). It reaches what neither the folder nor the algebraic
    // rules can: a `Where` whose predicate is constant but whose arms are not.
    // `RLX_DISABLE_SCCP=1` opts out for A/B, mirroring `RLX_DISABLE_CSE`.
    if options.constant_folding && rlx_ir::env::var("RLX_DISABLE_SCCP").as_deref() != Some("1") {
        graph = rlx_opt::rlx_compile::sccp::SCCPPass.run(graph);
    }
    // Value-number away structurally-identical nodes (bit-exact). Backward graphs
    // are the big beneficiary — reverse-mode AD re-emits the same subexpression
    // per use, e.g. multi-stage weight synthesis recomputes `upstreamᵀ·x` (a
    // Transpose + GEMM) once per stage. `RLX_DISABLE_CSE=1` opts out for A/B.
    if rlx_ir::env::var("RLX_DISABLE_CSE").as_deref() != Some("1") {
        graph = rlx_opt::CommonSubexpressionElimination.run(graph);
    }
    if options.dce {
        graph = rlx_opt::DeadCodeElimination.run(graph);
    }
    if options.constant_folding {
        graph = rlx_opt::ConstantFolding.run(graph);
    }
    graph
}
