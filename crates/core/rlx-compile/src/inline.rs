// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline a subgraph into a target graph.
//!
//! Walks `source`'s nodes in declaration order and replays each into
//! `target`, with `Op::Input`s replaced by caller-supplied
//! `NodeId`s. Returns the `NodeId`s in `target` that correspond to
//! `source`'s outputs.
//!
//! ## Use case
//!
//! Compositional graph building. The classic shape: take an existing
//! residual / Jacobian graph (built once via the assembler) and embed
//! it inside an outer graph (e.g. an `Op::Scan` body) where some of
//! its `Op::Input`s should be wired to existing `target` values
//! (e.g. carry, constants, slices) instead of staying free.
//!
//! ## Contract
//!
//! * `input_bindings`: every `Op::Input` in `source` must appear in
//!   the map. Missing names → `Err`.
//! * `param_bindings`: optional. When `Some`, every `Op::Param` in
//!   `source` must appear here. When `None`, params re-emit as
//!   `Op::Param` in `target` with the same name + shape (shared with
//!   a same-named param `target` already has) — caller is responsible
//!   for binding them later via `set_param`.
//! * Constants and other ops re-emit verbatim with their input
//!   `NodeId`s remapped through the source→target ID map.
//!
//! Recursive sub-graph ops (Scan body, CustomFn body, …) are cloned
//! intact — the inliner doesn't recurse into them since they own
//! their own input scopes.
//!
//! This is [`Graph::splice`] with every input bound; see
//! [`rlx_ir::compose`].

use std::collections::HashMap;

use rlx_ir::*;

/// Inline `source` into `target`. Returns the `target` `NodeId`s
/// that correspond to `source.outputs`.
///
/// # Errors
///
/// Returns `Err(message)` if `source` has an `Op::Input` (or, when
/// `param_bindings` is `Some`, `Op::Param`) whose name isn't in the
/// supplied bindings map, or a binding disagrees with its leaf on shape.
pub fn inline_into(
    target: &mut Graph,
    source: &Graph,
    input_bindings: &HashMap<String, NodeId>,
    param_bindings: Option<&HashMap<String, NodeId>>,
) -> Result<Vec<NodeId>, String> {
    let mut splice = Splice::new();
    let mut seen = std::collections::HashSet::new();
    for node in source.nodes() {
        let (kind, name, map, map_name) = match (&node.op, param_bindings) {
            (Op::Input { name }, _) => ("Input", name, input_bindings, "input_bindings"),
            (Op::Param { name }, Some(pm)) => ("Param", name, pm, "param_bindings"),
            _ => continue,
        };
        if !seen.insert(name) {
            continue;
        }
        let bound = map
            .get(name)
            .ok_or_else(|| format!("inline_into: source Op::{kind} '{name}' not in {map_name}"))?;
        splice = splice.bind(name.clone(), *bound);
    }
    target
        .splice(source, &splice)
        .map(|s| s.outputs)
        .map_err(|e| format!("inline_into: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;

    #[test]
    fn inline_replaces_inputs_with_target_nodes() {
        // Source: y = x * 2 (one input "x", one output).
        let s = Shape::new(&[1], DType::F32);
        let mut src = Graph::new("src");
        let x = src.input("x", s.clone());
        let two = src.add_node(
            Op::Constant {
                data: 2.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            s.clone(),
        );
        let y = src.binary(BinaryOp::Mul, x, two, s.clone());
        src.set_outputs(vec![y]);

        // Target: build a graph with a Constant 5, inline source with
        // x bound to the 5 → output should be 10 conceptually (just
        // checking the wiring; we don't run this test, just the
        // structural check).
        let mut tgt = Graph::new("tgt");
        let five = tgt.add_node(
            Op::Constant {
                data: 5.0_f32.to_le_bytes().to_vec(),
            },
            vec![],
            s.clone(),
        );
        let mut bindings: HashMap<String, NodeId> = HashMap::new();
        bindings.insert("x".to_string(), five);
        let outs = inline_into(&mut tgt, &src, &bindings, None).expect("inline");
        assert_eq!(outs.len(), 1);
        // The output node should be a Mul whose first input is the
        // target's `five` constant (not source's `x` Input).
        let out_node = tgt.node(outs[0]);
        assert!(matches!(out_node.op, Op::Binary(BinaryOp::Mul)));
        assert_eq!(out_node.inputs[0], five);
    }

    #[test]
    fn inline_errors_on_missing_input_binding() {
        let s = Shape::new(&[1], DType::F32);
        let mut src = Graph::new("src");
        let x = src.input("x", s.clone());
        src.set_outputs(vec![x]);

        let mut tgt = Graph::new("tgt");
        let bindings: HashMap<String, NodeId> = HashMap::new(); // empty
        let result = inline_into(&mut tgt, &src, &bindings, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("'x'"));
    }
}
