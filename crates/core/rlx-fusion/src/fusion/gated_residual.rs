// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuse DiT gated residual `x + gate·y` into [`Op::GatedResidual`].

use crate::graph_rewrite::Rewriter;
use crate::pass::Pass;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

/// Fuses `Add(x, Mul(Expand(gate), y))` into [`Op::GatedResidual`].
///
/// Prefers the pre-`Expand` gate tensor when present so the fused op can
/// broadcast `[B,1,D]` over `[B,S,D]` without materializing the expand.
pub struct FuseGatedResidual;

impl Pass for FuseGatedResidual {
    fn name(&self) -> &str {
        "fuse_gated_residual"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }

        let mut matches: HashMap<NodeId, Match> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if let Some(m) = try_match(&graph, node, &is_output) {
                for &id in &m.interior {
                    fused_away.insert(id, ());
                }
                matches.insert(m.out_id, m);
            }
        }

        let mut rw = Rewriter::new(&graph.name);
        for node in graph.nodes() {
            if fused_away.contains_key(&node.id) {
                continue;
            }
            if let Some(m) = matches.get(&node.id) {
                rw.ensure_mapped(&graph, &[m.x, m.y, m.gate]);
                let fused_id =
                    rw.add_fused(Op::GatedResidual, &[m.x, m.y, m.gate], node.shape.clone());
                for &id in &m.interior {
                    rw.replace(id, fused_id);
                }
                rw.replace(node.id, fused_id);
                continue;
            }
            rw.copy_node(node);
        }
        rw.finish(&graph.outputs)
    }
}

struct Match {
    out_id: NodeId,
    x: NodeId,
    y: NodeId,
    gate: NodeId,
    interior: Vec<NodeId>,
}

fn try_match(graph: &Graph, out: &Node, is_output: &HashMap<NodeId, ()>) -> Option<Match> {
    if !matches!(out.op, Op::Binary(BinaryOp::Add)) || out.inputs.len() != 2 {
        return None;
    }
    let (a, b) = (out.inputs[0], out.inputs[1]);
    if let Some(m) = match_add_mul(graph, out.id, a, b, is_output) {
        return Some(m);
    }
    match_add_mul(graph, out.id, b, a, is_output)
}

fn match_add_mul(
    graph: &Graph,
    out_id: NodeId,
    x_id: NodeId,
    mul_id: NodeId,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    let mul = graph.node(mul_id);
    if !matches!(mul.op, Op::Binary(BinaryOp::Mul)) || mul.inputs.len() != 2 {
        return None;
    }
    if is_output.contains_key(&mul_id) || graph.use_count(mul_id) != 1 {
        return None;
    }

    let x_shape = &graph.node(x_id).shape;
    let (p, q) = (mul.inputs[0], mul.inputs[1]);

    // Prefer: Mul(Expand(gate), y) where y matches x shape.
    if let Some(m) = classify_gate_y(graph, out_id, x_id, x_shape, p, q, mul_id, is_output) {
        return Some(m);
    }
    classify_gate_y(graph, out_id, x_id, x_shape, q, p, mul_id, is_output)
}

fn classify_gate_y(
    graph: &Graph,
    out_id: NodeId,
    x_id: NodeId,
    x_shape: &Shape,
    gate_side: NodeId,
    y_side: NodeId,
    mul_id: NodeId,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    // DiT gated residual is `x + gate·y` with distinct streams. Rejecting
    // `y == x` is required so we do not absorb AdaLayerNorm's unfuse
    // middle `n + n·expand(scale)` (which would otherwise look like
    // GatedResidual(n, n, scale) / GatedResidual(n, expand(scale), n)).
    if y_side == x_id {
        return None;
    }
    // y must already be full residual-stream shape (same as x).
    if graph.node(y_side).shape.dims() != x_shape.dims() {
        return None;
    }
    let (gate, gate_expand) = peel_expand(graph, gate_side);

    // Gate must be a true modulation broadcast (strictly smaller than `x`,
    // typically `[B,1,D]`). Same-shaped "gates" are rejected so a full
    // residual-stream tensor is never treated as the gate operand.
    let gate_shape = &graph.node(gate).shape;
    if !strict_modulation_broadcast(gate_shape, x_shape) {
        return None;
    }

    // Expand may still be live as a VJP operand (Mul backward reads the
    // forward expanded gate). Only absorb it when mul is the sole consumer.
    if let Some(e) = gate_expand {
        if is_output.contains_key(&e) || graph.use_count(e) != 1 {
            return None;
        }
    }

    let mut interior = vec![mul_id];
    if let Some(e) = gate_expand {
        interior.push(e);
    }
    Some(Match {
        out_id,
        x: x_id,
        y: y_side,
        gate,
        interior,
    })
}

fn peel_expand(graph: &Graph, id: NodeId) -> (NodeId, Option<NodeId>) {
    let n = graph.node(id);
    if matches!(n.op, Op::Expand { .. }) && n.inputs.len() == 1 {
        (n.inputs[0], Some(id))
    } else {
        (id, None)
    }
}

/// True when `gate` NumPy-broadcasts to `x` with matching last (feature) dim
/// and is strictly smaller than `x` (at least one axis where gate is 1 and
/// `x` is not).
fn strict_modulation_broadcast(gate: &Shape, x: &Shape) -> bool {
    if gate.rank() == 0 || x.rank() == 0 || gate.dims() == x.dims() {
        return false;
    }
    match (gate.dim(gate.rank() - 1), x.dim(x.rank() - 1)) {
        (Dim::Static(gd), Dim::Static(xd)) if gd == xd => {}
        _ => return false,
    }
    shape::broadcast(gate, x)
        .map(|b| b.dims() == x.dims())
        .unwrap_or(false)
}
