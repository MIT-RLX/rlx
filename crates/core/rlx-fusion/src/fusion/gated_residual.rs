// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuse DiT gated residual `x + gate·y` into [`Op::GatedResidual`].

use crate::analysis::{AnalysisManager, LazyUseCounts, OpKindIndex, UseCounts};
use crate::graph_rewrite::Rewriter;
use crate::pass::{Pass, PassResult};
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

/// Fuses `Add(x, Mul(Expand(gate), y))` into [`Op::GatedResidual`].
///
/// Prefers the pre-`Expand` gate tensor when present so the fused op can
/// broadcast `[B,1,D]` over `[B,S,D]` without materializing the expand.
pub struct FuseGatedResidual;

impl FuseGatedResidual {
    /// The pass body, parameterised over where its use-counts come from.
    ///
    /// `shared` carries the pipeline-wide relation when one is cached, so a
    /// run of ~20 fusion passes builds it once rather than once per pass.
    /// `None` falls back to a deferred per-pass build, which is what a direct
    /// `pass.run(graph)` call gets.
    fn fuse_with(&self, graph: Graph, shared: Option<&UseCounts>) -> PassResult {
        let uses = LazyUseCounts::from_shared(shared, &graph);

        let mut is_output: HashMap<NodeId, ()> = HashMap::new();
        for &oid in &graph.outputs {
            is_output.insert(oid, ());
        }

        let mut matches: HashMap<NodeId, Match> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if let Some(m) = try_match(&graph, &uses, node, &is_output) {
                for &id in &m.interior {
                    fused_away.insert(id, ());
                }
                matches.insert(m.out_id, m);
            }
        }

        // Nothing matched: hand back the original graph instead of rebuilding
        // it node-for-node into an identical copy. On a graph that merely
        // *contains* this pass's anchor op without the full pattern, that
        // rebuild was the pass's entire cost.
        if matches.is_empty() {
            return PassResult::unchanged(graph);
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
        rw.finish_reporting(&graph.outputs)
    }
}

impl Pass for FuseGatedResidual {
    fn name(&self) -> &str {
        "fuse_gated_residual"
    }

    fn run(&self, graph: Graph) -> Graph {
        self.fuse_with(graph, None).graph
    }

    fn run_with_status(&self, graph: Graph) -> PassResult {
        if !self.can_fire(&graph) {
            return PassResult::unchanged(graph);
        }
        self.fuse_with(graph, None)
    }

    fn run_with_analyses(&self, graph: Graph, analyses: &mut AnalysisManager) -> PassResult {
        // Answer the trigger check from the shared op-kind index first: a pass
        // whose anchor op is absent must not even pay for the use relation.
        let triggers = self.trigger_kinds();
        if !triggers.is_empty() && !analyses.get::<OpKindIndex>(&graph).contains_any(triggers) {
            return PassResult::unchanged(graph);
        }
        let shared = analyses.get::<UseCounts>(&graph);
        self.fuse_with(graph, Some(shared))
    }
}

struct Match {
    out_id: NodeId,
    x: NodeId,
    y: NodeId,
    gate: NodeId,
    interior: Vec<NodeId>,
}

fn try_match(
    graph: &Graph,
    uses: &LazyUseCounts,
    out: &Node,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    if !matches!(out.op, Op::Binary(BinaryOp::Add)) || out.inputs.len() != 2 {
        return None;
    }
    // **F32 only.** Every backend's `GatedResidual` kernel is f32: the CPU one
    // reads and writes through the f32 slice helpers (`sl` / `sl_mut`), and
    // Metal's takes `device const float*`. Neither checks the node dtype, so
    // fusing an F64 `Add(x, Mul(gate, y))` here produces a kernel that walks
    // 4-byte lanes over 8-byte data and returns garbage — silently, since the
    // shapes all still agree.
    //
    // Declining to fuse leaves the plain `Mul` + `Add`, which do have correct
    // per-dtype kernels (`BinaryFullF64`), so this costs a fusion opportunity
    // that no backend can currently take anyway. Same guard, same reason, as
    // `ConstantFolding`'s `node.shape.dtype() != DType::F32` bail-out.
    //
    // Regression test: `does_not_fuse_non_f32` below.
    if out.shape.dtype() != DType::F32 {
        return None;
    }
    let (a, b) = (out.inputs[0], out.inputs[1]);
    if let Some(m) = match_add_mul(graph, uses, out.id, a, b, is_output) {
        return Some(m);
    }
    match_add_mul(graph, uses, out.id, b, a, is_output)
}

fn match_add_mul(
    graph: &Graph,
    uses: &LazyUseCounts,
    out_id: NodeId,
    x_id: NodeId,
    mul_id: NodeId,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    let mul = graph.node(mul_id);
    if !matches!(mul.op, Op::Binary(BinaryOp::Mul)) || mul.inputs.len() != 2 {
        return None;
    }
    if is_output.contains_key(&mul_id) || uses.use_count(mul_id) != 1 {
        return None;
    }

    let x_shape = &graph.node(x_id).shape;
    let (p, q) = (mul.inputs[0], mul.inputs[1]);

    // Prefer: Mul(Expand(gate), y) where y matches x shape.
    if let Some(m) = classify_gate_y(graph, uses, out_id, x_id, x_shape, p, q, mul_id, is_output) {
        return Some(m);
    }
    classify_gate_y(graph, uses, out_id, x_id, x_shape, q, p, mul_id, is_output)
}

fn classify_gate_y(
    graph: &Graph,
    uses: &LazyUseCounts,
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
        if is_output.contains_key(&e) || uses.use_count(e) != 1 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn build(dtype: DType) -> Graph {
        // `Add(b, Mul(a, gate))` with a `[1]` gate broadcasting over `[2, 1]` —
        // the shape the LEiDA voxel pipeline produces when it scales a per-row
        // reduction by a scalar constant.
        let mut g = Graph::new("gr");
        let a = g.input("a", Shape::new(&[2, 1], dtype));
        let b = g.input("b", Shape::new(&[2, 1], dtype));
        let gate = g.input("gate", Shape::new(&[1], dtype));
        let mul = g.add_node(
            Op::Binary(BinaryOp::Mul),
            vec![a, gate],
            Shape::new(&[2, 1], dtype),
        );
        let add = g.add_node(
            Op::Binary(BinaryOp::Add),
            vec![mul, b],
            Shape::new(&[2, 1], dtype),
        );
        g.set_outputs(vec![add]);
        g
    }

    fn fused_count(g: &Graph) -> usize {
        g.nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::GatedResidual))
            .count()
    }

    #[test]
    fn fuses_f32() {
        let out = FuseGatedResidual.run(build(DType::F32));
        assert_eq!(fused_count(&out), 1, "f32 should still fuse");
    }

    /// Every backend's `GatedResidual` kernel is f32-only and does not check the
    /// node dtype, so fusing a non-F32 graph silently walks 4-byte lanes over
    /// wider data. Leaving `Mul` + `Add` in place routes to the correct
    /// per-dtype kernels instead.
    #[test]
    fn does_not_fuse_non_f32() {
        for dtype in [DType::F64, DType::F16, DType::BF16, DType::I32] {
            let out = FuseGatedResidual.run(build(dtype));
            assert_eq!(
                fused_count(&out),
                0,
                "{dtype:?} must not fuse into GatedResidual"
            );
            // ...and the original ops survive so the graph still computes.
            let muls = out
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Mul)))
                .count();
            let adds = out
                .nodes()
                .iter()
                .filter(|n| matches!(n.op, Op::Binary(BinaryOp::Add)))
                .count();
            assert_eq!((muls, adds), (1, 1), "{dtype:?} should keep Mul + Add");
        }
    }
}
