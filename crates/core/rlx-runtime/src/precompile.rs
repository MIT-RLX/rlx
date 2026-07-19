// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
    if options.dce {
        graph = rlx_opt::DeadCodeElimination.run(graph);
    }
    if options.constant_folding {
        graph = rlx_opt::ConstantFolding.run(graph);
    }
    graph
}
