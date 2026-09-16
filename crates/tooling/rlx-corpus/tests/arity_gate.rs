// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Op::arity` checked against real graphs, not hand-built ones.
//!
//! `verify` rejects a node whose operand count its `Arity` does not accept, so
//! an arity that is too *strict* turns working graphs into verifier failures.
//!
//! **Scope.** The corpus reaches 13 of 187 `OpKind`s, so this is a floor, not
//! a comprehensive check — it pins the op families it does cover and fails
//! loudly if the sweep itself stops working. The broad validation is that
//! `verify` calls `arity` on every node of every graph the workspace test
//! suites build.

use rlx_ir::op::Arity;
use std::collections::BTreeSet;

/// Every operand list in the corpus must satisfy its op's arity — including
/// inside nested bodies, where `Scan` / `If` / `While` hide most of the
/// interesting operand counts.
#[test]
fn every_corpus_node_satisfies_its_arity() {
    let mut checked = 0usize;
    let mut kinds: BTreeSet<String> = BTreeSet::new();
    let mut bad: Vec<String> = Vec::new();

    for case in rlx_corpus::cases() {
        walk(&case.graph, &mut |g: &rlx_ir::Graph| {
            for node in g.nodes() {
                checked += 1;
                kinds.insert(format!("{:?}", node.op.kind()));
                let a = node.op.arity();
                if !a.is_unconstrained() && !a.accepts(node.inputs.len()) {
                    bad.push(format!(
                        "{}/{}: {} has {} operand(s), arity says {a}",
                        case.family,
                        case.name,
                        node.op,
                        node.inputs.len()
                    ));
                }
            }
        });
    }

    assert!(
        bad.is_empty(),
        "{} node(s) rejected by their own arity:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
    // Guard the guard: a sweep that silently stopped walking would pass
    // vacuously. Pinned just under what the corpus builds today, so shrinking
    // the corpus or breaking `walk` fails here rather than going quiet.
    assert!(
        checked >= 150 && kinds.len() >= 12,
        "sweep too small to mean anything: {checked} nodes, {} kinds",
        kinds.len()
    );
}

/// `num_inputs` is documented as the arity's lower bound. If the two ever
/// disagree they are two tables again, which is what `Arity` replaced.
#[test]
fn num_inputs_is_the_arity_minimum_across_the_corpus() {
    for case in rlx_corpus::cases() {
        walk(&case.graph, &mut |g: &rlx_ir::Graph| {
            for node in g.nodes() {
                assert_eq!(
                    node.op.num_inputs(),
                    node.op.arity().min_operands(),
                    "{}/{}: {} disagrees",
                    case.family,
                    case.name,
                    node.op
                );
            }
        });
    }
}

/// An arity nothing can satisfy would reject every graph using that op.
#[test]
fn corpus_arities_are_satisfiable() {
    for case in rlx_corpus::cases() {
        walk(&case.graph, &mut |g: &rlx_ir::Graph| {
            for node in g.nodes() {
                let a = node.op.arity();
                if let Arity::Range { min, max } = a {
                    assert!(min <= max, "{}: unsatisfiable {a:?}", node.op);
                }
                assert!(
                    a.accepts(a.min_operands()),
                    "{} rejects its own minimum ({a:?})",
                    node.op
                );
            }
        });
    }
}

/// Apply `f` to a graph and every nested body reachable from it. Nested bodies
/// are where `Scan`/`If`/`While` operand counts live, and they are exactly the
/// ops whose arity is not `Exact`.
fn walk(g: &rlx_ir::Graph, f: &mut impl FnMut(&rlx_ir::Graph)) {
    f(g);
    for node in g.nodes() {
        for body in node.op.subgraphs() {
            walk(body, f);
        }
    }
}
