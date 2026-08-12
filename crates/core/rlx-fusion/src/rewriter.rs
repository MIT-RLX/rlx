// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pattern-driven rewriting — the boilerplate every `Lower*` pass shares,
//! written once.
//!
//! Roughly thirty passes in this crate have the identical skeleton:
//!
//! ```text
//! 1. scan for the trigger op; return the graph untouched if absent
//! 2. allocate a fresh Graph and a HashMap<NodeId, NodeId> remap
//! 3. for each node in topological order:
//!      - if it is the trigger, emit the replacement against remapped inputs
//!      - otherwise copy it, remapping its inputs
//! 4. remap the graph outputs
//! ```
//!
//! Only step 3's first branch differs between them. Everything else is copied
//! prose — and it drifts: some passes preserve a copied node's debug `name`
//! and `origin`, others silently drop them; some return early on the trigger
//! check, others rebuild unconditionally.
//!
//! [`MatchRewrite`] asks for the interesting part only. A blanket impl turns
//! any implementation into a [`Pass`](crate::pass::Pass), so a rewrite is
//! usable everywhere a pass is, and gets accurate
//! [`IRStatus`](crate::pass::IRStatus) reporting for free — the driver knows
//! exactly whether a pattern fired, so it never pays for the fingerprint
//! comparison the default [`Pass::run_with_status`](crate::pass::Pass::run_with_status)
//! would.
//!
//! # Convergence
//!
//! A rewrite whose output contains another rewritable node is re-swept until
//! nothing matches, up to [`MatchRewrite::max_rounds`]. This is the
//! rebuild-model equivalent of pliron's worklist re-enqueue: a lowering that
//! emits an op the same pass also lowers converges inside one pass instead of
//! relying on the caller's outer fixpoint loop.
//!
//! # Example
//!
//! ```
//! use rlx_fusion::pass::Pass;
//! use rlx_fusion::rewriter::{MatchRewrite, RewriteCtx};
//! use rlx_ir::op::BinaryOp;
//! use rlx_ir::{Graph, Node, NodeId, Op, OpKind};
//!
//! /// `Fma(a, b, c)` → `Add(Mul(a, b), c)`.
//! struct LowerFma;
//!
//! impl MatchRewrite for LowerFma {
//!     fn name(&self) -> &str { "lower_fma" }
//!     fn trigger_kinds(&self) -> &[OpKind] { &[OpKind::Fma] }
//!
//!     fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
//!         let [a, b, c] = [ctx.inputs[0], ctx.inputs[1], ctx.inputs[2]];
//!         let prod = ctx.emit(Op::Binary(BinaryOp::Mul), vec![a, b], node.shape.clone());
//!         Some(ctx.emit(Op::Binary(BinaryOp::Add), vec![prod, c], node.shape.clone()))
//!     }
//! }
//!
//! # use rlx_ir::{DType, Shape};
//! # let mut g = Graph::new("g");
//! # let s = Shape::new(&[8], DType::F32);
//! # let (a, b, c) = (g.input("a", s.clone()), g.input("b", s.clone()), g.input("c", s.clone()));
//! # let fma = g.add_node(Op::Fma, vec![a, b, c], s.clone());
//! # g.set_outputs(vec![fma]);
//! let out = LowerFma.run(g);
//! assert!(!out.nodes().iter().any(|n| n.op.kind() == OpKind::Fma));
//! ```

use std::collections::HashMap;

use rlx_ir::{Graph, Node, NodeId, Op, OpKind, Shape};

use crate::pass::{IRStatus, Pass, PassResult};

/// Where a matched node's replacement is emitted.
pub struct RewriteCtx<'a> {
    /// The graph under construction. Nodes emitted here are already in
    /// topological order.
    pub out: &'a mut Graph,
    /// The matched node's operands, **already remapped** into [`out`](Self::out).
    ///
    /// Always use these rather than `node.inputs`, which still refer to the
    /// input graph's numbering.
    ///
    /// Note that these index the *output* graph, whose contents can already
    /// differ from the input: an earlier node in the same sweep may have been
    /// rewritten, so `out.node(inputs[0])` need not be the op that
    /// [`MatchRewrite::matches`] saw at that position. Decide what to rewrite
    /// from the `graph`/`node` the matcher is given; use `inputs` only to wire
    /// operands up.
    pub inputs: &'a [NodeId],
}

impl RewriteCtx<'_> {
    /// Append a node to the output graph and return its id.
    pub fn emit(&mut self, op: Op, inputs: Vec<NodeId>, shape: Shape) -> NodeId {
        self.out.add_node(op, inputs, shape)
    }

    /// Append a node carrying a debug name.
    pub fn emit_named(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        shape: Shape,
        name: impl Into<String>,
    ) -> NodeId {
        let id = self.out.add_node(op, inputs, shape);
        self.out.node_mut(id).name = Some(name.into());
        id
    }

    /// The `i`th remapped operand.
    pub fn input(&self, i: usize) -> NodeId {
        self.inputs[i]
    }
}

/// A local graph rewrite: match a node, emit its replacement.
pub trait MatchRewrite {
    /// Human-readable name, used as the pass name.
    fn name(&self) -> &str;

    /// Op kinds this rewrite can fire on.
    ///
    /// The driver scans for these once and returns the graph untouched if none
    /// are present — the fast path every `Lower*` pass hand-writes today.
    /// An empty slice means "no cheap pre-filter"; [`matches`](Self::matches)
    /// is then consulted for every node.
    fn trigger_kinds(&self) -> &[OpKind] {
        &[]
    }

    /// Should this node be rewritten?
    ///
    /// Defaults to "its kind is one of [`trigger_kinds`](Self::trigger_kinds)".
    /// Override when the decision needs the node's attributes or its operands'
    /// shapes — a lowering that only handles f32, say, or only a particular
    /// axis configuration.
    fn matches(&self, _graph: &Graph, node: &Node) -> bool {
        self.trigger_kinds().contains(&node.op.kind())
    }

    /// Emit the replacement for a matched node and return the node that
    /// supersedes it.
    ///
    /// Return `None` to decline after all — the node is then copied verbatim,
    /// exactly as if [`matches`](Self::matches) had returned `false`. Declining
    /// is not a failure; it is how a rewrite handles the configurations it
    /// supports and leaves the rest for a later pass.
    fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId>;

    /// How many times to re-sweep when a rewrite's own output is rewritable.
    ///
    /// The default of 4 is generous for a lowering that expands one level. A
    /// rewrite that can re-match its own output *indefinitely* is a bug this
    /// bound contains rather than fixes — see
    /// [`apply_match_rewrite`] for what happens when the budget runs out.
    fn max_rounds(&self) -> usize {
        4
    }
}

/// Run one sweep. Returns the rebuilt graph and whether any pattern fired.
fn sweep<M: MatchRewrite + ?Sized>(rewrite: &M, graph: &Graph) -> (Graph, bool) {
    let mut out = Graph::new(&graph.name);
    let mut id_map: HashMap<NodeId, NodeId> = HashMap::with_capacity(graph.len());
    let mut fired = false;

    for node in graph.nodes() {
        let inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .filter_map(|i| id_map.get(i).copied())
            .collect();

        // A node whose operands did not all resolve means the input graph was
        // malformed. Copy it through untouched and let the verifier report it
        // at the right place rather than panicking here.
        let resolved = inputs.len() == node.inputs.len();

        let new_id = if resolved && rewrite.matches(graph, node) {
            let mut ctx = RewriteCtx {
                out: &mut out,
                inputs: &inputs,
            };
            match rewrite.rewrite(node, &mut ctx) {
                Some(id) => {
                    fired = true;
                    id
                }
                None => copy_node(&mut out, node, inputs),
            }
        } else {
            copy_node(&mut out, node, inputs)
        };

        id_map.insert(node.id, new_id);
    }

    let outputs = graph
        .outputs
        .iter()
        .filter_map(|o| id_map.get(o).copied())
        .collect();
    out.set_outputs(outputs);
    (out, fired)
}

/// Copy a node verbatim, preserving its debug name and provenance.
///
/// Several hand-written passes drop both; losing `origin` makes a later
/// `RLX_FUSION_REPORT` or NaN localization attribute the node to whichever
/// pass happened to run next.
fn copy_node(out: &mut Graph, node: &Node, inputs: Vec<NodeId>) -> NodeId {
    let id = out.add_node(node.op.clone(), inputs, node.shape.clone());
    if node.name.is_some() || node.origin.is_some() {
        let n = out.node_mut(id);
        n.name = node.name.clone();
        n.origin = node.origin.clone();
    }
    id
}

/// Drive `rewrite` over `graph` to a fixpoint.
///
/// Sweeps until no pattern fires or [`MatchRewrite::max_rounds`] is reached.
/// Exhausting the budget is not an error — the graph is returned as-is and the
/// caller's legalization will report anything still unlowered — but it does
/// mean the rewrite is not converging, so it is logged under
/// `RLX_FUSION_REPORT`.
pub fn apply_match_rewrite<M: MatchRewrite + ?Sized>(rewrite: &M, graph: Graph) -> PassResult {
    // Cheap pre-filter: nothing to do if the trigger kind is absent.
    let triggers = rewrite.trigger_kinds();
    if !triggers.is_empty()
        && !graph
            .nodes()
            .iter()
            .any(|n| triggers.contains(&n.op.kind()))
    {
        return PassResult::unchanged(graph);
    }

    let mut current = graph;
    let mut any_fired = false;

    for round in 0..rewrite.max_rounds().max(1) {
        let (next, fired) = sweep(rewrite, &current);
        current = next;
        if !fired {
            break;
        }
        any_fired = true;

        if round + 1 == rewrite.max_rounds().max(1) && rlx_ir::env::flag("RLX_FUSION_REPORT") {
            eprintln!(
                "rewrite `{}` still matching after {} rounds — not converging",
                rewrite.name(),
                rewrite.max_rounds().max(1)
            );
        }
    }

    PassResult::from_status(current, IRStatus::from(any_fired))
}

impl<M: MatchRewrite> Pass for M {
    fn name(&self) -> &str {
        MatchRewrite::name(self)
    }

    fn trigger_kinds(&self) -> &[OpKind] {
        MatchRewrite::trigger_kinds(self)
    }

    fn run(&self, graph: Graph) -> Graph {
        apply_match_rewrite(self, graph).graph
    }

    fn run_with_status(&self, graph: Graph) -> PassResult {
        // The driver knows whether a pattern fired, so skip the two structural
        // fingerprints the default implementation would compute.
        apply_match_rewrite(self, graph)
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

    /// `Fma(a, b, c)` → `Add(Mul(a, b), c)`.
    struct LowerFma;
    impl MatchRewrite for LowerFma {
        fn name(&self) -> &str {
            "lower_fma"
        }
        fn trigger_kinds(&self) -> &[OpKind] {
            &[OpKind::Fma]
        }
        fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
            let (a, b, c) = (ctx.input(0), ctx.input(1), ctx.input(2));
            let prod = ctx.emit(Op::Binary(BinaryOp::Mul), vec![a, b], node.shape.clone());
            Some(ctx.emit(Op::Binary(BinaryOp::Add), vec![prod, c], node.shape.clone()))
        }
    }

    fn fma_graph() -> Graph {
        let mut g = Graph::new("fma");
        let s = f32s(&[8]);
        let a = g.input("a", s.clone());
        let b = g.input("b", s.clone());
        let c = g.input("c", s.clone());
        let fma = g.add_node(Op::Fma, vec![a, b, c], s);
        g.set_outputs(vec![fma]);
        g
    }

    #[test]
    fn rewrites_and_rewires_outputs() {
        let out = LowerFma.run(fma_graph());
        assert!(!out.nodes().iter().any(|n| n.op.kind() == OpKind::Fma));

        let add = out.node(out.outputs[0]);
        assert!(matches!(add.op, Op::Binary(BinaryOp::Add)));
        assert!(matches!(
            out.node(add.inputs[0]).op,
            Op::Binary(BinaryOp::Mul)
        ));
    }

    #[test]
    fn reports_status_without_fingerprinting() {
        let fired = LowerFma.run_with_status(fma_graph());
        assert_eq!(fired.ir_changed, IRStatus::Changed);

        let mut plain = Graph::new("plain");
        let x = plain.input("x", f32s(&[4]));
        plain.set_outputs(vec![x]);
        let quiet = LowerFma.run_with_status(plain);
        assert_eq!(quiet.ir_changed, IRStatus::Unchanged);
    }

    #[test]
    fn trigger_absent_is_an_early_return() {
        let mut g = Graph::new("no_fma");
        let s = f32s(&[4]);
        let a = g.input("a", s.clone());
        let b = g.input("b", s.clone());
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], s);
        g.set_outputs(vec![sum]);
        let before = g.len();

        let result = LowerFma.run_with_status(g);
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert_eq!(result.graph.len(), before);
    }

    #[test]
    fn declining_copies_the_node_verbatim() {
        /// Matches every Fma but never rewrites — the "wrong dtype" shape.
        struct AlwaysDeclines;
        impl MatchRewrite for AlwaysDeclines {
            fn name(&self) -> &str {
                "declines"
            }
            fn trigger_kinds(&self) -> &[OpKind] {
                &[OpKind::Fma]
            }
            fn rewrite(&self, _: &Node, _: &mut RewriteCtx) -> Option<NodeId> {
                None
            }
        }

        let result = AlwaysDeclines.run_with_status(fma_graph());
        assert_eq!(result.ir_changed, IRStatus::Unchanged);
        assert_eq!(result.graph.len(), 4);
        assert!(
            result
                .graph
                .nodes()
                .iter()
                .any(|n| n.op.kind() == OpKind::Fma)
        );
    }

    #[test]
    fn copied_nodes_keep_name_and_origin() {
        let mut g = fma_graph();
        g.node_mut(NodeId(0)).name = Some("keep_me".into());
        let out = LowerFma.run(g);
        assert_eq!(out.node(NodeId(0)).name.as_deref(), Some("keep_me"));
    }

    /// Walks `Exp` → `Log` → `Sqrt` one step per sweep, so reaching `Sqrt`
    /// requires the driver to re-sweep. A single-sweep driver would stop at
    /// `Log` and leave the caller's outer loop to notice.
    struct StepActivation;
    impl MatchRewrite for StepActivation {
        fn name(&self) -> &str {
            "step_activation"
        }
        fn trigger_kinds(&self) -> &[OpKind] {
            &[OpKind::Activation]
        }
        fn matches(&self, _graph: &Graph, node: &Node) -> bool {
            matches!(
                node.op,
                Op::Activation(Activation::Exp) | Op::Activation(Activation::Log)
            )
        }
        fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
            let next = match node.op {
                Op::Activation(Activation::Exp) => Activation::Log,
                Op::Activation(Activation::Log) => Activation::Sqrt,
                _ => return None,
            };
            let x = ctx.input(0);
            Some(ctx.emit(Op::Activation(next), vec![x], node.shape.clone()))
        }
    }

    fn exp_graph() -> Graph {
        let mut g = Graph::new("exp");
        let s = f32s(&[4]);
        let x = g.input("x", s.clone());
        let e = g.add_node(Op::Activation(Activation::Exp), vec![x], s);
        g.set_outputs(vec![e]);
        g
    }

    #[test]
    fn resweeps_until_convergence() {
        let out = StepActivation.run(exp_graph());
        assert!(
            matches!(
                out.node(out.outputs[0]).op,
                Op::Activation(Activation::Sqrt)
            ),
            "two rewrites must happen in one pass, not one"
        );
    }

    #[test]
    fn round_budget_is_respected() {
        struct OneRound(StepActivation);
        impl MatchRewrite for OneRound {
            fn name(&self) -> &str {
                "one_round"
            }
            fn trigger_kinds(&self) -> &[OpKind] {
                MatchRewrite::trigger_kinds(&self.0)
            }
            fn matches(&self, g: &Graph, n: &Node) -> bool {
                self.0.matches(g, n)
            }
            fn rewrite(&self, n: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
                self.0.rewrite(n, ctx)
            }
            fn max_rounds(&self) -> usize {
                1
            }
        }

        let out = OneRound(StepActivation).run(exp_graph());
        assert!(
            matches!(out.node(out.outputs[0]).op, Op::Activation(Activation::Log)),
            "a one-round budget stops after the first sweep"
        );
    }

    #[test]
    fn a_rewrite_is_usable_as_a_dyn_pass() {
        let passes: Vec<&dyn Pass> = vec![&LowerFma];
        let out = crate::pass::run_passes(fma_graph(), &passes, false);
        assert!(!out.nodes().iter().any(|n| n.op.kind() == OpKind::Fma));
    }
}
