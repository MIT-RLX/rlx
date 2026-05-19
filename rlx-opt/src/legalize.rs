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

//! PLAN L4: backend op-set legalization.
//!
//! Backends declare the `OpKind`s they can lower via
//! `Backend::supported_ops()`. This module's `legalize_for_backend`
//! checks a graph against that set and returns the list of nodes
//! whose op is unsupported. Caller (typically `Backend::compile`)
//! decides whether to error out, decompose to atomic ops, or warn.
//!
//! The point isn't perf — it's bug prevention. Without this check,
//! an unsupported op silently dispatches to a fallback path (or
//! `Thunk::Nop` / "unsupported op" panic) at runtime; with it, the
//! compiler refuses to lower a graph it can't run faithfully.

use rlx_ir::{Graph, NodeId, OpKind};

/// Result of [`legalize_for_backend`] — list of `(node, kind)` pairs
/// whose op is outside the backend's claimed set. `Ok(())` when the
/// graph is fully legalized.
pub type LegalizeResult = Result<(), Vec<(NodeId, OpKind)>>;

/// Check `graph` against the backend's `supported` op set.
///
/// **Empty `supported` == "no claim made — accept everything".**
/// This is the default `Backend::supported_ops()` return value;
/// existing backends keep working unchanged.
///
/// When `supported` is non-empty, every node's `op.kind()` must be
/// in the set. Returns the list of offenders (kind + node id) so the
/// caller can produce a useful error message.
pub fn legalize_for_backend(graph: &Graph, supported: &[OpKind]) -> LegalizeResult {
    if supported.is_empty() {
        return Ok(());
    }
    let mut bad = Vec::new();
    for node in graph.nodes() {
        let k = node.op.kind();
        if !supported.contains(&k) {
            bad.push((node.id, k));
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(bad) }
}

/// Helper: format the legalize error as a single human-readable
/// diagnostic. Used by backend `compile` paths to panic with a clear
/// message when legalization fails.
pub fn format_legalize_error(backend_name: &str, errors: &[(NodeId, OpKind)]) -> String {
    use std::fmt::Write as _;
    let mut s = format!(
        "rlx-opt: backend {backend_name:?} doesn't claim support for {} op kind(s):\n",
        errors.len(),
    );
    for (id, kind) in errors {
        let _ = writeln!(s, "  - node {id:?}: {kind:?}");
    }
    s.push_str(
        "  Backend::supported_ops() must include each kind, or rewrite \
         the graph upstream to remove them.",
    );
    // Special-case the most common downstream-user friction:
    // `Op::Custom` rejected because the per-backend kernel registry
    // isn't wired yet. Point them at the right registry module so
    // they know where to plug in.
    if errors.iter().any(|(_, k)| *k == OpKind::Custom) {
        s.push_str(
            "\n  `Op::Custom` is registered by name; the IR-level \
             extension (`rlx_ir::register_op`) routes shape inference \
             and autodiff. Per-backend execution requires registering \
             a kernel in that backend's `op_registry`:\n\
             \x20  - CPU:   `rlx_cpu::op_registry::register_cpu_kernel`\n\
             \x20  - Metal: `rlx_metal::op_registry::register_metal_kernel` \
             (trait surface only — execution dispatch not wired yet)\n\
             \x20  - MLX:   `rlx_mlx::op_registry::register_mlx_kernel` \
             (trait surface only — execution dispatch not wired yet)\n\
             \x20For now, pin custom-op graphs to `Device::Cpu`.",
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::infer::GraphExt;
    use rlx_ir::*;

    fn tiny_graph() -> Graph {
        let f = DType::F32;
        let mut g = Graph::new("legalize");
        let a = g.input("a", Shape::new(&[4], f));
        let b = g.input("b", Shape::new(&[4], f));
        let s = g.add(a, b);
        let r = g.relu(s);
        g.set_outputs(vec![r]);
        g
    }

    #[test]
    fn empty_supported_set_accepts_anything() {
        let g = tiny_graph();
        assert!(legalize_for_backend(&g, &[]).is_ok());
    }

    #[test]
    fn supported_set_with_all_required_kinds_passes() {
        let g = tiny_graph();
        // tiny_graph uses Input + Binary + Activation.
        let supported = &[OpKind::Input, OpKind::Binary, OpKind::Activation];
        assert!(legalize_for_backend(&g, supported).is_ok());
    }

    #[test]
    fn unsupported_op_kind_is_reported() {
        let g = tiny_graph();
        // Drop Activation from the supported set — should flag the relu node.
        let supported = &[OpKind::Input, OpKind::Binary];
        let result = legalize_for_backend(&g, supported);
        let errors = result.expect_err("should fail");
        // Exactly one offender — the Activation node.
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].1, OpKind::Activation);
    }

    #[test]
    fn multiple_offenders_all_reported() {
        let g = tiny_graph();
        // No Binary, no Activation supported.
        let supported = &[OpKind::Input];
        let result = legalize_for_backend(&g, supported);
        let errors = result.expect_err("should fail");
        assert_eq!(errors.len(), 2);
        let kinds: Vec<OpKind> = errors.iter().map(|(_, k)| *k).collect();
        assert!(kinds.contains(&OpKind::Binary));
        assert!(kinds.contains(&OpKind::Activation));
    }

    #[test]
    fn format_error_includes_kind_and_count() {
        let g = tiny_graph();
        let supported = &[OpKind::Input];
        let errors = legalize_for_backend(&g, supported).unwrap_err();
        let msg = format_legalize_error("test_backend", &errors);
        assert!(msg.contains("test_backend"));
        assert!(msg.contains("2 op kind"));
        assert!(msg.contains("Binary"));
        assert!(msg.contains("Activation"));
    }
}
