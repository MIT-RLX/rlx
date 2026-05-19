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

//! Convert selected `Op::Param` nodes to `Op::Input` nodes.
//!
//! ## Why
//!
//! `vmap`'s MVP rule keeps `Op::Param` shared across the batch — that
//! matches the ML use case (model weights stay fixed across a
//! mini-batch). For Monte Carlo over device parameters (Vth mismatch,
//! resistor tolerance, mobility variation), the same scalar value
//! that a forward graph treats as a Param needs to vary per draw.
//! This pass swaps the Op variant from `Param` → `Input` for the
//! listed names, leaving everything else identical. After running
//! it, `vmap` will batch those nodes alongside any other inputs in
//! its batched-name list, so each draw can bind its own value.
//!
//! ## Contract
//!
//! Same name, same shape, same NodeId topology — only the `Op` enum
//! variant changes. Anything that referred to the param by NodeId
//! still resolves to the same logical leaf. A graph that's been
//! through this pass is structurally indistinguishable from one
//! built directly with `Graph::input(name, shape)` at the promoted
//! sites.

use std::collections::{HashMap, HashSet};

use rlx_ir::*;

/// Convert every `Op::Param { name }` whose `name` is in the list to
/// `Op::Input { name }` of the same shape. All other ops are copied
/// through unchanged. Returns a fresh `Graph`.
///
/// Names not present as Params in `graph` are silently ignored — same
/// permissive behavior as `vmap`'s name list, so callers don't have
/// to track which Params each device emits.
pub fn promote_params_to_inputs(graph: &Graph, names: &[&str]) -> Graph {
    let name_set: HashSet<&str> = names.iter().copied().collect();
    let mut out = Graph::new(graph.name.clone());
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

    for node in graph.nodes() {
        let new_id = match &node.op {
            Op::Param { name } if name_set.contains(name.as_str()) => {
                out.input(name.clone(), node.shape.clone())
            }
            _ => {
                let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                out.add_node(node.op.clone(), new_inputs, node.shape.clone())
            }
        };
        id_map.insert(node.id, new_id);
    }

    let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|o| id_map[o]).collect();
    out.set_outputs(new_outputs);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;

    #[test]
    fn promote_swaps_listed_param_only() {
        let s = Shape::new(&[1], DType::F32);
        let mut g = Graph::new("t");
        let x = g.input("x", s.clone());
        let w = g.param("w", s.clone());
        let b = g.param("b", s.clone());
        let xw = g.binary(BinaryOp::Mul, x, w, s.clone());
        let y = g.binary(BinaryOp::Add, xw, b, s.clone());
        g.set_outputs(vec![y]);

        // Promote only "w" — "b" stays a Param.
        let g2 = promote_params_to_inputs(&g, &["w"]);

        let mut input_names: Vec<String> = Vec::new();
        let mut param_names: Vec<String> = Vec::new();
        for n in g2.nodes() {
            match &n.op {
                Op::Input { name } => input_names.push(name.clone()),
                Op::Param { name } => param_names.push(name.clone()),
                _ => {}
            }
        }
        input_names.sort();
        param_names.sort();
        assert_eq!(input_names, vec!["w".to_string(), "x".to_string()]);
        assert_eq!(param_names, vec!["b".to_string()]);

        // Output count + topology preserved.
        assert_eq!(g2.outputs.len(), 1);
        assert_eq!(g2.nodes().len(), g.nodes().len());
    }

    #[test]
    fn promote_silently_ignores_unknown_names() {
        let s = Shape::new(&[1], DType::F32);
        let mut g = Graph::new("t");
        let _ = g.input("x", s.clone());
        let p = g.param("p", s.clone());
        g.set_outputs(vec![p]);

        // "missing" doesn't exist — should be a no-op for that name.
        let g2 = promote_params_to_inputs(&g, &["missing", "p"]);
        let promoted = g2
            .nodes()
            .iter()
            .filter(|n| matches!(&n.op, Op::Input { name } if name == "p"))
            .count();
        assert_eq!(promoted, 1);
    }
}
