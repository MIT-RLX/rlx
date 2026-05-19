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
/// In debug builds (`cfg!(debug_assertions)`), the verifier (#50 in
/// PLAN.md, lifted from MAX) runs after every pass — so any optimizer
/// bug that introduces a malformed graph is caught at the boundary
/// where it was introduced, not later in the codegen path. In release
/// builds the verifier is compiled out (free).
pub fn run_passes(mut graph: Graph, passes: &[&dyn Pass], verbose: bool) -> Graph {
    for pass in passes {
        if verbose {
            eprintln!("--- before {} ---\n{graph}", pass.name());
        }
        graph = pass.run(graph);
        // Verify after each pass (debug only). A panic with the pass
        // name in the message points directly at the broken pass.
        if cfg!(debug_assertions) {
            let errors = rlx_ir::verify::verify(&graph);
            if !errors.is_empty() {
                let msg = errors
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("\n  ");
                panic!("verifier failed after pass `{}`:\n  {msg}", pass.name());
            }
        }
    }
    if verbose {
        eprintln!("--- final ---\n{graph}");
    }
    graph
}
