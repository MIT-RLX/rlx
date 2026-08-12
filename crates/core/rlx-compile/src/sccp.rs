// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sparse conditional constant propagation.
//!
//! Two passes already fold constants here, and each is local:
//!
//! * [`ConstantFolding`](crate::const_fold::ConstantFolding) evaluates a node
//!   when **every** operand is already a constant.
//! * [`AlgebraicSimplify`](crate::algebraic_simplify::AlgebraicSimplify)
//!   rewrites one node when **one** operand is a constant with a special value
//!   (`mul(x, 0)`, `add(x, 0)`, …).
//!
//! Neither propagates a *lattice*, and that is what they cannot see through:
//! a `Where` whose predicate is constant. `where(true, 5.0, x)` has a runtime
//! operand, so the folder refuses it; it is not a `Binary`, so the algebraic
//! rules do not apply. The node is provably `5.0` and both passes leave it
//! alone — as does any chain built on top of it.
//!
//! This pass assigns every node a lattice value and lets it flow:
//!
//! ```text
//!   Top      — not yet known
//!   Const(v) — an F32 tensor with known contents
//!   Alias(n) — provably the same tensor as node `n`, whatever that is
//!   Bottom   — runtime-varying
//! ```
//!
//! `Alias` is the "conditional" part, and it is dtype-agnostic: resolving a
//! predicate tells you *which operand* the result is without knowing its
//! value, so `where(false, x, y)` becomes `y` even for i32 or bool tensors
//! that the F32-only folder could never touch. Constants then flow onward
//! through the alias, which is how a chain collapses in one pass instead of
//! needing fold → simplify → fold → … to a fixpoint.
//!
//! # What this deliberately does not do
//!
//! * **No algebraic identities.** `mul(x, 1)`, `add(x, 0)` and friends belong
//!   to [`algebraic_simplify`](crate::algebraic_simplify), which already owns
//!   them. Duplicating them here would mean two places to keep IEEE-correct.
//! * **No iteration.** Textbook SCCP runs a worklist to a fixpoint because it
//!   handles loops and φ-nodes. rlx's dataflow graph is an acyclic DAG in
//!   topological order, so one forward sweep *is* the fixpoint — every operand
//!   is resolved before its consumer. Iteration would only be needed to reach
//!   values carried across [`Op::Scan`] / [`Op::While`] body boundaries, which
//!   this pass does not enter.
//! # `Op::If` pruning
//!
//! [`LowerControlFlow`](rlx_fusion::control_flow::LowerControlFlow) lowers an
//! `If` by inlining **both** branches and selecting with `Where` — it has to,
//! because in general the predicate is a runtime value. When the predicate is
//! a known constant that is pure waste: half the inlined work is dead on
//! arrival, and DCE can only remove it after the fact.
//!
//! This is the textbook SCCP property that plain constant folding lacks —
//! unreachable-code elimination. A uniformly-true predicate means only the
//! `then` branch is reachable, so only it gets inlined and the `else` body is
//! never materialised at all.

use std::collections::HashMap;

use rlx_fusion::pass::{IRStatus, Pass, PassResult};
use rlx_ir::{DType, Graph, NodeId, Op, OpKind};

use crate::const_fold::{evaluate, is_pure, static_dims};

pub struct SCCPPass;

/// Lattice value for one node.
#[derive(Debug, Clone)]
enum Lat {
    /// Runtime-varying, or known-but-unrepresentable (non-F32 constants).
    Bottom,
    /// An F32 tensor with known contents, sized to the node's shape.
    Const(Vec<f32>),
    /// Provably the same tensor as another node. Shapes are checked equal
    /// before this is assigned, so an alias is a drop-in replacement.
    Alias(NodeId),
}

/// Read a *literal* constant's bytes as booleans, honouring its declared dtype.
///
/// `None` for a dtype this pass will not reason about, which is treated as
/// "unknown" rather than "false" — guessing here would silently select the
/// wrong branch.
fn decode_predicate(data: &[u8], dtype: DType, want: usize) -> Option<Vec<bool>> {
    let bits: Vec<bool> = match dtype {
        DType::Bool => data.iter().map(|&b| b != 0).collect(),
        DType::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) != 0.0)
            .collect(),
        _ => return None,
    };
    (bits.len() == want).then_some(bits)
}

fn decode_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Resolve a predicate operand to per-element booleans.
///
/// Consults the **lattice** first and the node's own bytes second. That order
/// matters: a predicate is rarely a bare `Op::Constant` in a real graph — it is
/// something *computed* from constants, such as `Expand` of a scalar flag, a
/// `Cast`, or a `Compare` of two constant tensors. Asking only "is this node an
/// `Op::Constant`?" is syntactic constant *detection*, and it makes the lattice
/// decorative: every derived predicate reads as unknown. Asking the lattice is
/// constant *propagation*, which is the whole point of the pass.
///
/// The byte fallback still matters because the lattice only carries F32
/// values, while predicates are usually `Bool`.
fn predicate_bits(
    graph: &Graph,
    lattice: &HashMap<NodeId, Lat>,
    pred: NodeId,
    want: usize,
) -> Option<Vec<bool>> {
    let id = resolve(lattice, pred);
    if let Some(values) = value_of(lattice, id)
        && values.len() == want
    {
        return Some(values.iter().map(|&v| v != 0.0).collect());
    }
    let node = graph.node(id);
    let Op::Constant { data } = &node.op else {
        return None;
    };
    decode_predicate(data, node.shape.dtype(), want)
}

/// Follow `Alias` links to the node that actually produces the value.
fn resolve(lattice: &HashMap<NodeId, Lat>, mut id: NodeId) -> NodeId {
    // Aliases only ever point at strictly earlier nodes (operands), so this
    // terminates; the bound is belt-and-braces against a future rule that
    // forgets that invariant.
    for _ in 0..64 {
        match lattice.get(&id) {
            Some(Lat::Alias(target)) if *target != id => id = *target,
            _ => break,
        }
    }
    id
}

/// Lattice value of `id` after alias resolution.
fn value_of(lattice: &HashMap<NodeId, Lat>, id: NodeId) -> Option<&Vec<f32>> {
    match lattice.get(&resolve(lattice, id)) {
        Some(Lat::Const(v)) => Some(v),
        _ => None,
    }
}

impl SCCPPass {
    /// Forward sweep: assign a lattice value to every node, and note every
    /// `Op::If` whose predicate resolves to a constant (with the branch taken).
    fn analyze(graph: &Graph) -> (HashMap<NodeId, Lat>, HashMap<NodeId, bool>) {
        let mut lattice: HashMap<NodeId, Lat> = HashMap::with_capacity(graph.len());
        let mut taken: HashMap<NodeId, bool> = HashMap::new();

        for node in graph.nodes() {
            let lat = match &node.op {
                // Seeds. Only F32 constants carry a representable value; a
                // constant of another dtype is still a fine *alias* target,
                // it just has no `Vec<f32>` meaning here.
                Op::Constant { data } if node.shape.dtype() == DType::F32 => {
                    Lat::Const(decode_f32(data))
                }

                // The conditional rule — the reason this pass exists.
                Op::Where if node.inputs.len() == 3 => Self::transfer_where(graph, &lattice, node),

                // Unreachable-branch elimination: a constant predicate means
                // one branch can never run.
                Op::If { .. } if !node.inputs.is_empty() => {
                    if let Some(b) = Self::resolve_predicate(graph, &lattice, node.inputs[0]) {
                        taken.insert(node.id, b);
                    }
                    Lat::Bottom
                }

                // Everything else: constant only if every operand is.
                //
                // Deliberately NOT restricted to F32 outputs. `Compare` yields
                // `Bool`, and a comparison of two constants is exactly the kind
                // of derived predicate this pass exists to resolve — gating the
                // *analysis* on the dtype the *rewrite* can materialise would
                // throw that away. Materialisation is gated separately below.
                op if is_pure(op) => Self::transfer_pure(graph, &lattice, node),

                _ => Lat::Bottom,
            };
            lattice.insert(node.id, lat);
        }

        (lattice, taken)
    }

    /// A predicate that is a uniformly true or uniformly false constant.
    /// `None` for a runtime predicate or a mixed constant mask — an `If`
    /// selects a whole branch, so a per-element mask decides nothing.
    fn resolve_predicate(
        graph: &Graph,
        lattice: &HashMap<NodeId, Lat>,
        pred: NodeId,
    ) -> Option<bool> {
        let node = graph.node(resolve(lattice, pred));
        let bits = predicate_bits(graph, lattice, pred, node.shape.num_elements()?)?;
        if bits.is_empty() {
            return None;
        }
        if bits.iter().all(|&b| b) {
            Some(true)
        } else if bits.iter().all(|&b| !b) {
            Some(false)
        } else {
            None
        }
    }

    fn transfer_where(graph: &Graph, lattice: &HashMap<NodeId, Lat>, node: &rlx_ir::Node) -> Lat {
        let (cond_id, t_id, f_id) = (node.inputs[0], node.inputs[1], node.inputs[2]);
        let cond_node = graph.node(resolve(lattice, cond_id));
        let Some(elems) = cond_node.shape.num_elements() else {
            return Lat::Bottom;
        };
        let Some(bits) = predicate_bits(graph, lattice, cond_id, elems) else {
            return Lat::Bottom;
        };
        if bits.is_empty() {
            return Lat::Bottom;
        }

        let all_true = bits.iter().all(|&b| b);
        let all_false = bits.iter().all(|&b| !b);

        // A uniform predicate selects one operand outright, whatever its
        // dtype — but only if it already has the result's shape. `Where`
        // broadcasts, so forwarding a smaller operand would silently change
        // the output shape.
        if all_true || all_false {
            let chosen = if all_true { t_id } else { f_id };
            if graph.node(chosen).shape == node.shape {
                return Lat::Alias(chosen);
            }
        }

        // Mixed predicate: only foldable if both arms are known constants of
        // the result's shape (no broadcasting to reason about).
        if node.shape.dtype() != DType::F32 {
            return Lat::Bottom;
        }
        let (Some(t), Some(f)) = (value_of(lattice, t_id), value_of(lattice, f_id)) else {
            return Lat::Bottom;
        };
        if t.len() != elems || f.len() != elems || node.shape.num_elements() != Some(elems) {
            return Lat::Bottom;
        }
        Lat::Const(
            bits.iter()
                .enumerate()
                .map(|(i, &b)| if b { t[i] } else { f[i] })
                .collect(),
        )
    }

    fn transfer_pure(graph: &Graph, lattice: &HashMap<NodeId, Lat>, node: &rlx_ir::Node) -> Lat {
        if node.inputs.is_empty() {
            return Lat::Bottom;
        }
        let Some(values): Option<Vec<&Vec<f32>>> =
            node.inputs.iter().map(|&i| value_of(lattice, i)).collect()
        else {
            return Lat::Bottom;
        };
        // Dims come from the *resolved* producer: after an alias the operand
        // is a different node, and evaluating against the alias's shape would
        // broadcast from the wrong extent.
        let Some(dims): Option<Vec<Vec<usize>>> = node
            .inputs
            .iter()
            .map(|&i| static_dims(&graph.node(resolve(lattice, i)).shape))
            .collect()
        else {
            return Lat::Bottom;
        };
        match evaluate(node, &values, &dims) {
            Some(result) => Lat::Const(result),
            None => Lat::Bottom,
        }
    }
}

impl Pass for SCCPPass {
    fn name(&self) -> &str {
        "sccp"
    }

    /// Both of this pass's unique powers — resolving a `Where` predicate and
    /// pruning an `If` — need one of these ops present. Without them the only
    /// thing left is plain constant folding, which
    /// [`ConstantFolding`](crate::const_fold::ConstantFolding) already does.
    ///
    /// This matters more than it looks: instrumenting the whole `rlx-runtime`
    /// suite showed SCCP becoming actionable **zero** times, so on today's
    /// in-tree models it is pure overhead. The guard makes that overhead a
    /// single `OpKind` scan (or an `OpKindIndex` hit when run through
    /// `run_passes_tracked`) instead of a full lattice sweep.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::Where, OpKind::If]
    }

    fn run(&self, graph: Graph) -> Graph {
        self.run_with_status(graph).graph
    }

    fn run_with_status(&self, graph: Graph) -> PassResult {
        if !self.can_fire(&graph) {
            return PassResult::unchanged(graph);
        }
        let (lattice, taken_branch) = Self::analyze(&graph);

        // Nothing to do unless some node improved on Bottom in a way the
        // rebuild would act on.
        let actionable = graph.nodes().iter().any(|n| {
            taken_branch.contains_key(&n.id)
                || match lattice.get(&n.id) {
                    Some(Lat::Alias(_)) => true,
                    Some(Lat::Const(_)) => !n.op.is_leaf() && n.shape.dtype() == DType::F32,
                    _ => false,
                }
        });
        if !actionable {
            return PassResult::unchanged(graph);
        }

        let mut out = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::with_capacity(graph.len());

        for node in graph.nodes() {
            // A statically-decided `If`: inline only the reachable branch.
            // The other body is never materialised, so there is nothing for
            // DCE to clean up afterwards.
            if let Some(&take_then) = taken_branch.get(&node.id)
                && let Op::If {
                    then_branch,
                    else_branch,
                } = &node.op
            {
                let captures: Vec<NodeId> = node.inputs[1..]
                    .iter()
                    .map(|i| id_map[&resolve(&lattice, *i)])
                    .collect();
                let body = if take_then { then_branch } else { else_branch };
                let inlined =
                    rlx_fusion::control_flow::inline_subgraph_into(body, &captures, &mut out);
                id_map.insert(node.id, inlined);
                continue;
            }

            let new_id = match lattice.get(&node.id) {
                // The node *is* another node: emit nothing and point every
                // later reference at the survivor. This is what lets a
                // resolved `Where` disappear entirely.
                Some(Lat::Alias(target)) => id_map[&resolve(&lattice, *target)],

                // A computed constant becomes a literal — but only for F32,
                // the one encoding `Op::Constant` carries here. A known
                // non-F32 value (a `Bool` comparison result, say) still
                // propagates through the lattice to whatever consumes it; it
                // just stays a computed node rather than becoming a literal.
                Some(Lat::Const(values))
                    if !node.op.is_leaf() && node.shape.dtype() == DType::F32 =>
                {
                    out.add_node(
                        Op::Constant {
                            data: encode_f32(values),
                        },
                        vec![],
                        node.shape.clone(),
                    )
                }

                _ => {
                    let inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
                    let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
                    if node.name.is_some() || node.origin.is_some() {
                        let n = out.node_mut(id);
                        n.name = node.name.clone();
                        n.origin = node.origin.clone();
                    }
                    id
                }
            };
            id_map.insert(node.id, new_id);
        }

        out.set_outputs(graph.outputs.iter().map(|o| id_map[o]).collect());
        PassResult::from_status(out, IRStatus::Changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, Shape};

    fn f32s(dims: &[usize]) -> Shape {
        Shape::new(dims, DType::F32)
    }

    fn f32_const(g: &mut Graph, values: &[f32]) -> NodeId {
        g.add_node(
            Op::Constant {
                data: encode_f32(values),
            },
            vec![],
            f32s(&[values.len()]),
        )
    }

    fn bool_const(g: &mut Graph, bits: &[bool]) -> NodeId {
        g.add_node(
            Op::Constant {
                data: bits.iter().map(|&b| u8::from(b)).collect(),
            },
            vec![],
            Shape::new(&[bits.len()], DType::Bool),
        )
    }

    fn constant_values(g: &Graph, id: NodeId) -> Option<Vec<f32>> {
        match &g.node(id).op {
            Op::Constant { data } => Some(decode_f32(data)),
            _ => None,
        }
    }

    #[test]
    fn folds_a_where_with_a_runtime_arm() {
        // where(true, [5,5], x) == [5,5], even though `x` is runtime.
        // Neither const_fold (needs all operands constant) nor
        // algebraic_simplify (Binary only) can see this.
        let mut g = Graph::new("where_true");
        let x = g.input("x", f32s(&[2]));
        let pred = bool_const(&mut g, &[true, true]);
        let t = f32_const(&mut g, &[5.0, 5.0]);
        let w = g.add_node(Op::Where, vec![pred, t, x], f32s(&[2]));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert_eq!(constant_values(&out, out.outputs[0]), Some(vec![5.0, 5.0]));
    }

    #[test]
    fn a_false_predicate_forwards_the_runtime_arm() {
        // where(false, c, x) == x. The result is not constant, but it is
        // provably `x` — the alias case, which no value-only lattice reaches.
        let mut g = Graph::new("where_false");
        let x = g.input("x", f32s(&[2]));
        let pred = bool_const(&mut g, &[false, false]);
        let c = f32_const(&mut g, &[9.0, 9.0]);
        let w = g.add_node(Op::Where, vec![pred, c, x], f32s(&[2]));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert!(
            matches!(out.node(out.outputs[0]).op, Op::Input { .. }),
            "the Where must collapse to its runtime arm"
        );
    }

    #[test]
    fn alias_works_for_dtypes_the_f32_folder_cannot_touch() {
        let mut g = Graph::new("i32_where");
        let x = g.input("x", Shape::new(&[3], DType::I32));
        let y = g.input("y", Shape::new(&[3], DType::I32));
        let pred = bool_const(&mut g, &[false, false, false]);
        let w = g.add_node(Op::Where, vec![pred, x, y], Shape::new(&[3], DType::I32));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert_eq!(
            out.node(out.outputs[0]).name.as_deref().or_else(|| {
                match &out.node(out.outputs[0]).op {
                    Op::Input { name } => Some(name.as_str()),
                    _ => None,
                }
            }),
            Some("y"),
            "false predicate selects the third operand regardless of dtype"
        );
    }

    #[test]
    fn constants_flow_through_a_resolved_where() {
        // relu(where(true, [-2, 3], x)) == [0, 3]: the constant has to cross
        // the Where for the activation to fold.
        let mut g = Graph::new("chain");
        let x = g.input("x", f32s(&[2]));
        let pred = bool_const(&mut g, &[true, true]);
        let t = f32_const(&mut g, &[-2.0, 3.0]);
        let w = g.add_node(Op::Where, vec![pred, t, x], f32s(&[2]));
        let r = g.add_node(Op::Activation(Activation::Relu), vec![w], f32s(&[2]));
        g.set_outputs(vec![r]);

        let out = SCCPPass.run(g);
        assert_eq!(constant_values(&out, out.outputs[0]), Some(vec![0.0, 3.0]));
    }

    #[test]
    fn mixed_predicate_selects_elementwise() {
        let mut g = Graph::new("mixed");
        let pred = bool_const(&mut g, &[true, false, true]);
        let t = f32_const(&mut g, &[1.0, 2.0, 3.0]);
        let f = f32_const(&mut g, &[10.0, 20.0, 30.0]);
        let w = g.add_node(Op::Where, vec![pred, t, f], f32s(&[3]));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert_eq!(
            constant_values(&out, out.outputs[0]),
            Some(vec![1.0, 20.0, 3.0])
        );
    }

    #[test]
    fn a_predicate_computed_from_constants_resolves() {
        // The realistic shape: the predicate is not a bare `Op::Constant`, it
        // is *derived* from one. `LowerControlFlow` emits exactly this —
        // `Where(Expand(pred), ...)` — and a syntactic "is this node a
        // Constant?" check reads it as unknown.
        let mut g = Graph::new("derived_pred");
        let x = g.input("x", f32s(&[4]));
        let seed = f32_const(&mut g, &[1.0]);
        let pred = g.add_node(
            Op::Expand {
                target_shape: vec![4],
            },
            vec![seed],
            f32s(&[4]),
        );
        let t = f32_const(&mut g, &[7.0, 7.0, 7.0, 7.0]);
        let w = g.add_node(Op::Where, vec![pred, t, x], f32s(&[4]));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert_eq!(
            constant_values(&out, out.outputs[0]),
            Some(vec![7.0, 7.0, 7.0, 7.0]),
            "a predicate expanded from a constant must still resolve"
        );
    }

    #[test]
    fn a_predicate_from_a_constant_comparison_resolves() {
        // `Compare` of two constants is itself constant — the lattice knows,
        // even though no `Op::Constant` node holds the boolean.
        let mut g = Graph::new("cmp_pred");
        let x = g.input("x", f32s(&[2]));
        let lo = f32_const(&mut g, &[1.0, 1.0]);
        let hi = f32_const(&mut g, &[2.0, 2.0]);
        let pred = g.add_node(
            Op::Compare(rlx_ir::op::CmpOp::Lt),
            vec![lo, hi],
            Shape::new(&[2], DType::Bool),
        );
        let t = f32_const(&mut g, &[5.0, 5.0]);
        let w = g.add_node(Op::Where, vec![pred, t, x], f32s(&[2]));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert_eq!(
            constant_values(&out, out.outputs[0]),
            Some(vec![5.0, 5.0]),
            "1 < 2 is statically true, so the Where selects the constant arm"
        );
    }

    #[test]
    fn an_if_with_a_derived_predicate_prunes() {
        let shape = f32s(&[4]);
        let branch = |act: Activation| {
            let mut b = Graph::new("branch");
            let c = b.input("cap", shape.clone());
            let y = b.add_node(Op::Activation(act), vec![c], shape.clone());
            b.set_outputs(vec![y]);
            Box::new(b)
        };
        let mut g = Graph::new("derived_if");
        let x = g.input("x", shape.clone());
        let seed = f32_const(&mut g, &[0.0]);
        let pred = g.add_node(
            Op::Expand {
                target_shape: vec![4],
            },
            vec![seed],
            shape.clone(),
        );
        let node = g.add_node(
            Op::If {
                then_branch: branch(Activation::Gelu),
                else_branch: branch(Activation::Relu),
            },
            vec![pred, x],
            shape,
        );
        g.set_outputs(vec![node]);

        let out = SCCPPass.run(g);
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Relu))),
            "an all-zero derived predicate takes the else branch"
        );
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Gelu)))
        );
    }

    #[test]
    fn a_runtime_predicate_is_left_alone() {
        let mut g = Graph::new("dyn_pred");
        let p = g.input("p", Shape::new(&[2], DType::Bool));
        let x = g.input("x", f32s(&[2]));
        let c = f32_const(&mut g, &[1.0, 1.0]);
        let w = g.add_node(Op::Where, vec![p, c, x], f32s(&[2]));
        g.set_outputs(vec![w]);

        let before = g.len();
        let result = SCCPPass.run_with_status(g);
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert_eq!(result.graph.len(), before);
    }

    #[test]
    fn broadcasting_arms_are_not_forwarded() {
        // `on_true` is [1] but the result is [4]; forwarding it would change
        // the output shape.
        let mut g = Graph::new("bcast");
        let x = g.input("x", f32s(&[4]));
        let pred = bool_const(&mut g, &[true]);
        let t = f32_const(&mut g, &[7.0]);
        let w = g.add_node(Op::Where, vec![pred, t, x], f32s(&[4]));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert_eq!(
            out.node(out.outputs[0]).shape,
            f32s(&[4]),
            "output shape must survive"
        );
    }

    #[test]
    fn a_plain_constant_chain_is_left_to_the_constant_folder() {
        // No `Where` and no `If`, so this pass has nothing it can do that
        // `ConstantFolding` (which runs later in `precompile_cleanup`) does not
        // already do — and it declines without walking the graph twice.
        let mut g = Graph::new("plain");
        let a = f32_const(&mut g, &[2.0, 3.0]);
        let b = f32_const(&mut g, &[4.0, 5.0]);
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f32s(&[2]));
        g.set_outputs(vec![sum]);

        let result = SCCPPass.run_with_status(g);
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert!(matches!(
            result.graph.node(result.graph.outputs[0]).op,
            Op::Binary(BinaryOp::Add)
        ));
    }

    #[test]
    fn constants_still_fold_when_a_where_gives_the_pass_a_reason_to_run() {
        // Same chain, but reachable through a resolved `Where` — now the
        // lattice does propagate through it.
        let mut g = Graph::new("via_where");
        let a = f32_const(&mut g, &[2.0, 3.0]);
        let b = f32_const(&mut g, &[4.0, 5.0]);
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f32s(&[2]));
        let x = g.input("x", f32s(&[2]));
        let pred = bool_const(&mut g, &[true, true]);
        let w = g.add_node(Op::Where, vec![pred, sum, x], f32s(&[2]));
        g.set_outputs(vec![w]);

        let out = SCCPPass.run(g);
        assert_eq!(constant_values(&out, out.outputs[0]), Some(vec![6.0, 8.0]));
    }

    /// Build `If(pred, then: Gelu(x), else: Relu(x))`.
    fn if_graph(pred: bool) -> Graph {
        let shape = f32s(&[4]);
        let branch = |act: Activation| {
            let mut b = Graph::new("branch");
            let c = b.input("cap", shape.clone());
            let y = b.add_node(Op::Activation(act), vec![c], shape.clone());
            b.set_outputs(vec![y]);
            Box::new(b)
        };

        let mut g = Graph::new("cond");
        let x = g.input("x", shape.clone());
        let p = bool_const(&mut g, &[pred; 4]);
        let node = g.add_node(
            Op::If {
                then_branch: branch(Activation::Gelu),
                else_branch: branch(Activation::Relu),
            },
            vec![p, x],
            shape,
        );
        g.set_outputs(vec![node]);
        g
    }

    #[test]
    fn a_constant_if_inlines_only_the_reachable_branch() {
        let out = SCCPPass.run(if_graph(true));

        assert!(
            !out.nodes().iter().any(|n| matches!(n.op, Op::If { .. })),
            "the If must be resolved away"
        );
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Gelu))),
            "the taken branch must be inlined"
        );
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Relu))),
            "the unreachable branch must never be materialised — this is what \
             LowerControlFlow cannot do, since it inlines both and selects"
        );
    }

    #[test]
    fn a_false_predicate_takes_the_other_branch() {
        let out = SCCPPass.run(if_graph(false));
        assert!(
            out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Relu)))
        );
        assert!(
            !out.nodes()
                .iter()
                .any(|n| matches!(n.op, Op::Activation(Activation::Gelu)))
        );
    }

    #[test]
    fn a_runtime_if_is_left_for_lower_control_flow() {
        let shape = f32s(&[4]);
        let branch = |act: Activation| {
            let mut b = Graph::new("branch");
            let c = b.input("cap", shape.clone());
            let y = b.add_node(Op::Activation(act), vec![c], shape.clone());
            b.set_outputs(vec![y]);
            Box::new(b)
        };
        let mut g = Graph::new("dyn_if");
        let x = g.input("x", shape.clone());
        let p = g.input("p", Shape::new(&[4], DType::Bool));
        let node = g.add_node(
            Op::If {
                then_branch: branch(Activation::Gelu),
                else_branch: branch(Activation::Relu),
            },
            vec![p, x],
            shape,
        );
        g.set_outputs(vec![node]);

        let result = SCCPPass.run_with_status(g);
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert!(
            result
                .graph
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::If { .. }))
        );
    }

    #[test]
    fn a_mixed_predicate_decides_nothing() {
        // An `If` selects a whole branch, so a per-element mask cannot resolve
        // it even though it is a constant.
        let shape = f32s(&[4]);
        let branch = |act: Activation| {
            let mut b = Graph::new("branch");
            let c = b.input("cap", shape.clone());
            let y = b.add_node(Op::Activation(act), vec![c], shape.clone());
            b.set_outputs(vec![y]);
            Box::new(b)
        };
        let mut g = Graph::new("mixed_if");
        let x = g.input("x", shape.clone());
        let p = bool_const(&mut g, &[true, false, true, false]);
        let node = g.add_node(
            Op::If {
                then_branch: branch(Activation::Gelu),
                else_branch: branch(Activation::Relu),
            },
            vec![p, x],
            shape,
        );
        g.set_outputs(vec![node]);

        assert_eq!(
            SCCPPass.run_with_status(g).ir_changed,
            IRStatus::Unchanged,
            "a mixed mask must not decide a branch"
        );
    }

    #[test]
    fn a_graph_with_nothing_to_prove_is_untouched() {
        let mut g = Graph::new("runtime");
        let x = g.input("x", f32s(&[2]));
        let y = g.input("y", f32s(&[2]));
        let s = g.add_node(Op::Binary(BinaryOp::Add), vec![x, y], f32s(&[2]));
        g.set_outputs(vec![s]);

        let result = SCCPPass.run_with_status(g);
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
    }
}
