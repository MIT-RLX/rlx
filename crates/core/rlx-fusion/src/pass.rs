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

//! Pass infrastructure — trait + pipeline runner.

use rlx_ir::Graph;

/// A graph-to-graph transformation pass.
pub trait Pass {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Transform the graph. Returns a new graph (or the same if no changes).
    fn run(&self, graph: Graph) -> Graph;
}

/// Run a sequence of passes, printing the graph after each if `verbose`.
///
/// When `RLX_FUSION_REPORT=1`, prints a [`fusion_report::FusionReport`]
/// comparing the input graph to the fused result.
///
/// In debug builds, the verifier (#50 in PLAN.md, lifted from MAX) runs
/// after every pass via [`rlx_ir::debug_assert_valid!`] — so any optimizer
/// bug that introduces a malformed graph is caught at the boundary where it
/// was introduced. In release builds the check is not compiled in.
pub fn run_passes(mut graph: Graph, passes: &[&dyn Pass], verbose: bool) -> Graph {
    let before = rlx_ir::env::flag("RLX_FUSION_REPORT").then(|| graph.clone());
    for pass in passes {
        if verbose {
            eprintln!("--- before {} ---\n{graph}", pass.name());
        }
        graph = pass.run(graph);
        rlx_ir::stamp_pass_origins(&mut graph, pass.name());
        rlx_ir::debug_assert_valid!(&graph, format!("after pass `{}`", pass.name()));
    }
    if verbose {
        eprintln!("--- final ---\n{graph}");
    }
    if let Some(before) = before {
        let report = crate::fusion_report::FusionReport::analyze(&before, &graph);
        eprintln!("{report}");
    }
    graph
}
