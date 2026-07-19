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

//! `cargo rlx check` — device-free static analysis of an rlx graph.
//!
//! The analysis itself lives in [`rlx_runtime::check`] (so it's reachable from
//! the `#[rlx_model(check)]` self-check hook without a new dependency). This
//! crate is the CLI front door plus the built-in [`demo`] graphs; it re-exports
//! the checker so `rlx_check::check_graph` keeps working.
//!
//! ```no_run
//! let g = rlx_check::demo::build("swiglu").unwrap();
//! let report = rlx_check::check_graph(&g, &rlx_check::CheckOptions::default());
//! print!("{}", report.render());
//! assert!(!report.has_errors());
//! ```

pub mod demo;
pub mod scaffold;

pub use rlx_runtime::check::{
    BackendSummary, CheckOptions, CheckReport, Diagnostic, Legality, Severity, all_backends,
    backend_device, backend_name, check_graph, default_backends, model_self_check, parse_backend,
};

use rlx_ir::Graph;

/// Parse a graph (MIR-level [`Graph`]) from its JSON serialization.
pub fn parse_graph_json(s: &str) -> Result<Graph, serde_json::Error> {
    serde_json::from_str(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(name: &str) -> CheckReport {
        let g = demo::build(name).unwrap_or_else(|| panic!("no demo {name}"));
        check_graph(&g, &CheckOptions::default())
    }

    #[test]
    fn clean_mlp_has_no_errors() {
        let r = check("mlp");
        assert_eq!(r.errors(), 0, "unexpected errors: {:#?}", r.diagnostics);
        assert!(!r.backends.is_empty());
        // CPU is always compiled in — its real op claim must accept a plain MLP.
        let cpu = r
            .backends
            .iter()
            .find(|b| b.backend == "cpu")
            .expect("cpu summary");
        let leg = cpu.legality.as_ref().expect("cpu legality available");
        assert!(leg.compile_ready);
        assert_eq!(leg.unsupported_kinds, 0);
    }

    #[test]
    fn bad_shape_is_an_error() {
        let r = check("badshape");
        assert!(r.has_errors());
        assert!(r.diagnostics.iter().any(|d| d.code == "shape"));
        // Backend analysis is skipped on a graph that fails verification.
        assert!(r.backends.is_empty());
    }

    #[test]
    fn nan_div_is_a_numeric_warning() {
        let r = check("nan");
        assert_eq!(r.errors(), 0);
        assert!(
            r.diagnostics
                .iter()
                .any(|d| d.code == "numeric" && d.message.to_lowercase().contains("inf"))
        );
    }

    #[test]
    fn swiglu_gate_first_is_a_missed_fusion() {
        let r = check("swiglu");
        assert_eq!(r.errors(), 0);
        let miss = r
            .diagnostics
            .iter()
            .filter(|d| d.code == "missed-fusion")
            .count();
        assert!(miss >= 1, "expected a missed fusion");
        // Deduped across backends → not one-per-target.
        assert!(miss <= 3, "missed fusions should be deduped, got {miss}");
    }

    #[test]
    fn json_round_trips_through_graph() {
        let g = demo::build("mlp").unwrap();
        let json = serde_json::to_string(&g).expect("serialize graph");
        let back = parse_graph_json(&json).expect("deserialize graph");
        let a = check_graph(&g, &CheckOptions::default());
        let b = check_graph(&back, &CheckOptions::default());
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.errors(), b.errors());
    }

    #[test]
    fn self_check_hook_is_reachable() {
        // The same entry point `#[rlx_model(check)]` calls.
        let g = demo::build("mlp").unwrap();
        model_self_check("mlp", &g);
    }
}
