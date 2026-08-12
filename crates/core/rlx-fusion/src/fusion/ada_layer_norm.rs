// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuse DiT adaLN-Zero `norm(x)·(1+scale)+shift` into [`Op::AdaLayerNorm`].

use crate::analysis::{AnalysisManager, LazyUseCounts, OpKindIndex, UseCounts};
use crate::graph_rewrite::Rewriter;
use crate::pass::{Pass, PassResult};
use rlx_ir::OpKind;
use rlx_ir::op::*;
use rlx_ir::*;
use std::collections::HashMap;

/// Fuses affine-free LayerNorm/RmsNorm + scale/shift modulation into
/// [`Op::AdaLayerNorm`].
///
/// Matches either imported DiT form:
/// ```text
///   n = norm(x)                         # gamma=1, beta=0 constants
///   out = n * (1 + expand(scale)) + expand(shift)
/// ```
/// or the unfuse identity form:
/// ```text
///   out = n + n * expand(scale) + expand(shift)
/// ```
pub struct FuseAdaLayerNorm;

impl FuseAdaLayerNorm {
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

        // Collect match sites keyed by the final Add node id.
        let mut matches: HashMap<NodeId, Match> = HashMap::new();
        let mut fused_away: HashMap<NodeId, ()> = HashMap::new();

        for node in graph.nodes() {
            if let Some(m) = try_match_ada(&graph, &uses, node, &is_output) {
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
                rw.ensure_mapped(&graph, &[m.x, m.scale, m.shift]);
                let fused_id = rw.add_fused(
                    Op::AdaLayerNorm {
                        norm: m.norm_kind,
                        eps: m.eps,
                    },
                    &[m.x, m.scale, m.shift],
                    node.shape.clone(),
                );
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

impl Pass for FuseAdaLayerNorm {
    // Required by construction: `affine_free_norm` only accepts a LayerNorm or RmsNorm at the centre of the pattern.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::LayerNorm, OpKind::RmsNorm]
    }

    fn name(&self) -> &str {
        "fuse_ada_layer_norm"
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
    scale: NodeId,
    shift: NodeId,
    norm_kind: AdaNormKind,
    eps: f32,
    /// Intermediate nodes absorbed into the fused op (not the final Add).
    interior: Vec<NodeId>,
}

fn try_match_ada(
    graph: &Graph,
    uses: &LazyUseCounts,
    out: &Node,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    if !matches!(out.op, Op::Binary(BinaryOp::Add)) || out.inputs.len() != 2 {
        return None;
    }
    // Prefer form: Add(scaled, shift) where scaled = Mul(n, (1+scale))
    // or Add(m, shift) where m = Add(n, Mul(n, scale)).
    let (scaled_or_m, shift_side) = (out.inputs[0], out.inputs[1]);
    if let Some(m) = match_shift_add(graph, uses, out.id, scaled_or_m, shift_side, is_output) {
        return Some(m);
    }
    match_shift_add(graph, uses, out.id, shift_side, scaled_or_m, is_output)
}

fn match_shift_add(
    graph: &Graph,
    uses: &LazyUseCounts,
    out_id: NodeId,
    scaled_id: NodeId,
    shift_id: NodeId,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    if is_output.contains_key(&scaled_id) || uses.use_count(scaled_id) != 1 {
        return None;
    }
    let (shift, shift_expand) = peel_expand(graph, shift_id);

    // Form A: Mul(n, one_plus_scale)
    if let Some(m) = match_mul_one_plus(
        graph,
        uses,
        out_id,
        scaled_id,
        shift,
        shift_expand,
        is_output,
    ) {
        return Some(m);
    }
    // Form B: Add(n, Mul(n, scale))
    match_identity_form(
        graph,
        uses,
        out_id,
        scaled_id,
        shift,
        shift_expand,
        is_output,
    )
}

fn match_mul_one_plus(
    graph: &Graph,
    uses: &LazyUseCounts,
    out_id: NodeId,
    scaled_id: NodeId,
    shift: NodeId,
    shift_expand: Option<NodeId>,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    let scaled = graph.node(scaled_id);
    if !matches!(scaled.op, Op::Binary(BinaryOp::Mul)) || scaled.inputs.len() != 2 {
        return None;
    }
    let (a, b) = (scaled.inputs[0], scaled.inputs[1]);
    if let Some(m) = match_norm_times_mod(
        graph,
        uses,
        out_id,
        a,
        b,
        shift,
        shift_expand,
        scaled_id,
        is_output,
    ) {
        return Some(m);
    }
    match_norm_times_mod(
        graph,
        uses,
        out_id,
        b,
        a,
        shift,
        shift_expand,
        scaled_id,
        is_output,
    )
}

fn match_norm_times_mod(
    graph: &Graph,
    uses: &LazyUseCounts,
    out_id: NodeId,
    norm_id: NodeId,
    mod_id: NodeId,
    shift: NodeId,
    shift_expand: Option<NodeId>,
    scaled_id: NodeId,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    let (x, norm_kind, eps) = affine_free_norm(graph, norm_id)?;
    if is_output.contains_key(&norm_id) || uses.use_count(norm_id) != 1 {
        return None;
    }
    let (scale, scale_expand, one_plus) = peel_one_plus_scale(graph, uses, mod_id, is_output)?;
    if let Some(e) = scale_expand {
        if is_output.contains_key(&e) || uses.use_count(e) != 1 {
            return None;
        }
    }
    if let Some(e) = shift_expand {
        if is_output.contains_key(&e) || uses.use_count(e) != 1 {
            return None;
        }
    }
    let (_, _, norm_node) = peel_norm_affine_inputs(graph, norm_id)?;
    let mut interior = vec![norm_node, scaled_id, mod_id];
    if let Some(reshape) = peel_reshape(graph, graph.node(norm_node).inputs[1]).1 {
        interior.push(reshape);
    }
    if let Some(reshape) = peel_reshape(graph, graph.node(norm_node).inputs[2]).1 {
        interior.push(reshape);
    }
    if let Some(e) = shift_expand {
        interior.push(e);
    }
    if let Some(e) = scale_expand {
        interior.push(e);
    }
    if let Some(op) = one_plus {
        interior.push(op);
    }
    Some(Match {
        out_id,
        x,
        scale,
        shift,
        norm_kind,
        eps,
        interior,
    })
}

fn match_identity_form(
    graph: &Graph,
    uses: &LazyUseCounts,
    out_id: NodeId,
    m_id: NodeId,
    shift: NodeId,
    shift_expand: Option<NodeId>,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    let m = graph.node(m_id);
    if !matches!(m.op, Op::Binary(BinaryOp::Add)) || m.inputs.len() != 2 {
        return None;
    }
    if is_output.contains_key(&m_id) || uses.use_count(m_id) != 1 {
        return None;
    }
    let (a, b) = (m.inputs[0], m.inputs[1]);
    if let Some(hit) = match_n_plus_n_scale(
        graph,
        uses,
        out_id,
        a,
        b,
        shift,
        shift_expand,
        m_id,
        is_output,
    ) {
        return Some(hit);
    }
    match_n_plus_n_scale(
        graph,
        uses,
        out_id,
        b,
        a,
        shift,
        shift_expand,
        m_id,
        is_output,
    )
}

fn match_n_plus_n_scale(
    graph: &Graph,
    uses: &LazyUseCounts,
    out_id: NodeId,
    n_id: NodeId,
    ns_id: NodeId,
    shift: NodeId,
    shift_expand: Option<NodeId>,
    m_id: NodeId,
    is_output: &HashMap<NodeId, ()>,
) -> Option<Match> {
    let (x, norm_kind, eps) = affine_free_norm(graph, n_id)?;
    if is_output.contains_key(&n_id) {
        return None;
    }
    // n is used by both m and Mul — use_count must be 2.
    if uses.use_count(n_id) != 2 {
        return None;
    }
    let ns = graph.node(ns_id);
    if !matches!(ns.op, Op::Binary(BinaryOp::Mul)) || ns.inputs.len() != 2 {
        return None;
    }
    if is_output.contains_key(&ns_id) || uses.use_count(ns_id) != 1 {
        return None;
    }
    let (p, q) = (ns.inputs[0], ns.inputs[1]);
    let scale_side = if p == n_id {
        q
    } else if q == n_id {
        p
    } else {
        return None;
    };
    let (scale, scale_expand) = peel_expand(graph, scale_side);
    if let Some(e) = scale_expand {
        // VJP for Mul still reads the forward expanded scale.
        if is_output.contains_key(&e) || uses.use_count(e) != 1 {
            return None;
        }
    }
    if let Some(e) = shift_expand {
        if is_output.contains_key(&e) || uses.use_count(e) != 1 {
            return None;
        }
    }
    let mut interior = vec![n_id, ns_id, m_id];
    if let Some(e) = shift_expand {
        interior.push(e);
    }
    if let Some(e) = scale_expand {
        interior.push(e);
    }
    Some(Match {
        out_id,
        x,
        scale,
        shift,
        norm_kind,
        eps,
        interior,
    })
}

/// Peel a broadcast wrapper (`Expand` or broadcast `Reshape`).
///
/// F5-TTS ONNX lowers `[B,1,D]` modulation via `Reshape`/`Unsqueeze` rather
/// than `Expand`; FLUX fixtures use `Expand`. Both must peel for fusion.
fn peel_expand(graph: &Graph, id: NodeId) -> (NodeId, Option<NodeId>) {
    let n = graph.node(id);
    if matches!(n.op, Op::Expand { .. }) && n.inputs.len() == 1 {
        return (n.inputs[0], Some(id));
    }
    // Broadcast reshape: same element count, rank increased (e.g. [1,D]→[1,1,D]).
    if matches!(n.op, Op::Reshape { .. }) && n.inputs.len() == 1 {
        let inner = graph.node(n.inputs[0]);
        let ok = matches!(
            (inner.shape.num_elements(), n.shape.num_elements()),
            (Some(a), Some(b)) if a == b && n.shape.rank() >= inner.shape.rank()
        );
        if ok {
            return (n.inputs[0], Some(id));
        }
    }
    (id, None)
}

/// Match `Add(1, scale)` / `Add(scale, 1)` with optional Expand on scale.
/// Returns `(scale, scale_expand, Some(one_plus_add_id))`.
fn peel_one_plus_scale(
    graph: &Graph,
    uses: &LazyUseCounts,
    id: NodeId,
    is_output: &HashMap<NodeId, ()>,
) -> Option<(NodeId, Option<NodeId>, Option<NodeId>)> {
    let n = graph.node(id);
    if matches!(n.op, Op::Binary(BinaryOp::Add)) && n.inputs.len() == 2 {
        if is_output.contains_key(&id) || uses.use_count(id) != 1 {
            return None;
        }
        let (a, b) = (n.inputs[0], n.inputs[1]);
        if is_constant_filled(graph, a, 1.0) {
            let (scale, exp) = peel_expand(graph, b);
            return Some((scale, exp, Some(id)));
        }
        if is_constant_filled(graph, b, 1.0) {
            let (scale, exp) = peel_expand(graph, a);
            return Some((scale, exp, Some(id)));
        }
    }
    // Already `1+scale` materialized as a single tensor — treat as scale'
    // (cannot recover raw scale; skip — would change AdaLayerNorm semantics).
    None
}

fn affine_free_norm(graph: &Graph, id: NodeId) -> Option<(NodeId, AdaNormKind, f32)> {
    let (gamma_id, beta_id, norm_node) = peel_norm_affine_inputs(graph, id)?;
    if !is_constant_filled(graph, gamma_id, 1.0) || !is_constant_filled(graph, beta_id, 0.0) {
        return None;
    }
    let n = graph.node(norm_node);
    match &n.op {
        Op::LayerNorm { eps, .. } if n.inputs.len() == 3 => {
            Some((n.inputs[0], AdaNormKind::LayerNorm, *eps))
        }
        Op::RmsNorm { eps, .. } if n.inputs.len() == 3 => {
            Some((n.inputs[0], AdaNormKind::RmsNorm, *eps))
        }
        _ => None,
    }
}

/// Peel a single `Reshape` on LayerNorm/RmsNorm scale/bias operands.
fn peel_norm_affine_inputs(graph: &Graph, id: NodeId) -> Option<(NodeId, NodeId, NodeId)> {
    let n = graph.node(id);
    match &n.op {
        Op::LayerNorm { .. } | Op::RmsNorm { .. } if n.inputs.len() == 3 => {
            let gamma = peel_reshape(graph, n.inputs[1]).0;
            let beta = peel_reshape(graph, n.inputs[2]).0;
            Some((gamma, beta, id))
        }
        _ => None,
    }
}

/// Peel `Reshape` wrapper; returns `(inner, Some(reshape_id))` or `(id, None)`.
fn peel_reshape(graph: &Graph, id: NodeId) -> (NodeId, Option<NodeId>) {
    let n = graph.node(id);
    if matches!(n.op, Op::Reshape { .. }) && n.inputs.len() == 1 {
        (n.inputs[0], Some(id))
    } else {
        (id, None)
    }
}

fn is_constant_filled(graph: &Graph, id: NodeId, value: f32) -> bool {
    let n = graph.node(id);
    let Op::Constant { data } = &n.op else {
        return false;
    };
    // Param specialization always bakes f32. ONNX may still carry f16
    // Constants for affine-free γ=1 / β=0 / scalar 1 before promote.
    match n.shape.dtype() {
        DType::F32 => {
            let want = value.to_le_bytes();
            !data.is_empty() && data.chunks_exact(4).all(|c| c == want)
        }
        DType::F16 => {
            // IEEE f16: +0.0 = 0x0000, +1.0 = 0x3C00 (only values we need).
            let want: [u8; 2] = if value == 0.0 {
                [0x00, 0x00]
            } else if value == 1.0 {
                [0x00, 0x3c]
            } else {
                return false;
            };
            !data.is_empty() && data.chunks_exact(2).all(|c| c == want)
        }
        _ => false,
    }
}
