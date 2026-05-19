// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

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
//!   `Op::Param` in `target` with the same name + shape — caller is
//!   responsible for binding them later via `set_param`.
//! * Constants and other ops re-emit verbatim with their input
//!   `NodeId`s remapped through the source→target ID map.
//!
//! Recursive sub-graph ops (Scan body, CustomFn body, …) are cloned
//! intact — the inliner doesn't recurse into them since they own
//! their own input scopes.

use std::collections::HashMap;

use rlx_ir::*;

/// Inline `source` into `target`. Returns the `target` `NodeId`s
/// that correspond to `source.outputs`.
///
/// # Errors
///
/// Returns `Err(name)` if `source` has an `Op::Input` (or, when
/// `param_bindings` is `Some`, `Op::Param`) whose name isn't in the
/// supplied bindings map.
pub fn inline_into(
    target: &mut Graph,
    source: &Graph,
    input_bindings: &HashMap<String, NodeId>,
    param_bindings: Option<&HashMap<String, NodeId>>,
) -> Result<Vec<NodeId>, String> {
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::with_capacity(source.nodes().len());

    for node in source.nodes() {
        let new_id = match &node.op {
            Op::Input { name } => *input_bindings.get(name).ok_or_else(|| {
                format!("inline_into: source Op::Input '{name}' not in input_bindings")
            })?,
            Op::Param { name } => match param_bindings {
                Some(pm) => *pm.get(name).ok_or_else(|| {
                    format!("inline_into: source Op::Param '{name}' not in param_bindings")
                })?,
                None => target.param(name.clone(), node.shape.clone()),
            },
            Op::Constant { data } => target.add_node(
                Op::Constant { data: data.clone() },
                vec![],
                node.shape.clone(),
            ),
            _ => {
                let new_inputs: Vec<NodeId> = node
                    .inputs
                    .iter()
                    .map(|i| {
                        *id_map.get(i).expect(
                            "inline_into: input NodeId not yet mapped — \
                         source graph isn't in topo order?",
                        )
                    })
                    .collect();
                target.add_node(node.op.clone(), new_inputs, node.shape.clone())
            }
        };
        id_map.insert(node.id, new_id);
    }

    Ok(source
        .outputs
        .iter()
        .map(|o| *id_map.get(o).expect("output NodeId missing from map"))
        .collect())
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
