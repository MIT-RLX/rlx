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

use std::sync::{Arc, OnceLock, RwLock};

use rlx_ir::Graph;

/// A graph-to-graph transformation pass.
pub trait Pass {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Transform the graph. Returns a new graph (or the same if no changes).
    fn run(&self, graph: Graph) -> Graph;
}

/// Registry of downstream-registered IR passes, run **after** the built-in
/// fusion pipeline (so core fusion invariants hold) but **before** backend
/// legalization (so a pass's output — e.g. a custom fused op — is still lowered
/// / legalized). Empty by default: zero cost until a downstream crate registers
/// one. This is the extension seam for custom fusion / rewrite rules without
/// editing the core pass list.
///
/// A registered pass should fast-path return the graph unchanged when its
/// trigger pattern is absent — it runs on *every* compiled graph in the process.
static IR_PASS_REGISTRY: OnceLock<RwLock<Vec<Arc<dyn Pass + Send + Sync>>>> = OnceLock::new();

fn ir_pass_registry() -> &'static RwLock<Vec<Arc<dyn Pass + Send + Sync>>> {
    IR_PASS_REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

/// Register a downstream IR pass to run after the built-in fusion passes.
pub fn register_ir_pass(pass: Arc<dyn Pass + Send + Sync>) {
    ir_pass_registry().write().unwrap().push(pass);
}

/// Snapshot of registered downstream passes, in registration order.
pub fn registered_ir_passes() -> Vec<Arc<dyn Pass + Send + Sync>> {
    ir_pass_registry().read().unwrap().clone()
}

/// Run every registered downstream pass over `graph`, in registration order.
/// A no-op (returns the graph untouched) when none are registered.
pub fn run_registered_ir_passes(mut graph: Graph) -> Graph {
    for pass in registered_ir_passes() {
        graph = pass.run(graph);
        rlx_ir::stamp_pass_origins(&mut graph, pass.name());
        rlx_ir::debug_assert_valid!(&graph, format!("after registered pass `{}`", pass.name()));
    }
    graph
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

#[cfg(test)]
mod registry_tests {
    use super::*;
    use rlx_ir::{DType, Op, Shape};

    /// Sentinel-gated pass: only transforms a graph containing an input named
    /// `__ir_pass_sentinel`; a strict pass-through otherwise, so registering it
    /// globally cannot contaminate other tests' graphs. When triggered it
    /// negates the single output.
    struct SentinelNegate;
    impl Pass for SentinelNegate {
        fn name(&self) -> &str {
            "sentinel_negate"
        }
        fn run(&self, mut graph: Graph) -> Graph {
            let has_sentinel = graph
                .nodes()
                .iter()
                .any(|n| matches!(&n.op, Op::Input { name } if name == "__ir_pass_sentinel"));
            if !has_sentinel {
                return graph; // fast-path: untouched, safe for unrelated graphs
            }
            let out = graph.outputs[0];
            let shape = graph.node(out).shape.clone();
            let neg = graph.add_node(
                Op::Activation(rlx_ir::op::Activation::Neg),
                vec![out],
                shape,
            );
            graph.set_outputs(vec![neg]);
            graph
        }
    }

    #[test]
    fn registered_pass_runs_only_on_its_trigger() {
        register_ir_pass(Arc::new(SentinelNegate));

        // Graph WITHOUT the sentinel: the registered pass must leave it alone.
        let mut plain = Graph::new("plain");
        let a = plain.input("a", Shape::new(&[2], DType::F32));
        plain.set_outputs(vec![a]);
        let plain_len = plain.len();
        let plain_out = run_registered_ir_passes(plain);
        assert_eq!(
            plain_out.len(),
            plain_len,
            "unrelated graph must be untouched"
        );

        // Graph WITH the sentinel: the pass appends a Neg and repoints the output.
        let mut g = Graph::new("triggered");
        let s = g.input("__ir_pass_sentinel", Shape::new(&[2], DType::F32));
        g.set_outputs(vec![s]);
        let out = run_registered_ir_passes(g);
        assert!(
            matches!(
                out.node(out.outputs[0]).op,
                Op::Activation(rlx_ir::op::Activation::Neg)
            ),
            "registered pass should have negated the output"
        );
    }
}
