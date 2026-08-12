// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Common Subexpression Elimination — merge structurally identical nodes.
//!
//! Two interior nodes with the same op, the same (already-remapped) inputs, and
//! the same shape compute bit-identically, so the second is redundant. Walking
//! the graph in topological order and value-numbering each node collapses every
//! such duplicate to its first occurrence.
//!
//! Why it matters (backward graphs especially): reverse-mode AD emits the same
//! subexpression many times. The prime example is **multi-stage weight synthesis**
//! (`rlx-tiny`): `q = x·W₀ + x·W₁ + …` makes every stage's weight-gradient
//! `grad_Wₛ = upstreamᵀ·x` — *identical* across stages (same `upstream`, same `x`).
//! Without CSE each stage recomputes the transpose **and** the GEMM; CSE keeps one
//! copy, cutting a `Transpose`+`MatMul` per extra stage per projection. Unlike
//! MPS transpose-folding (which loses to per-call overhead at small matmul scale),
//! this removes the work outright on whatever kernel the backend already picks.
//!
//! Only **interior** nodes (non-empty inputs) are value-numbered: two `Op::Input`s
//! or `Op::Param`s can share a shape yet denote different values, so leaves are
//! always kept distinct. Merging is bit-exact — the surviving node has identical
//! op/inputs/shape, hence identical output.

use rlx_fusion::pass::Pass;
use rlx_ir::{Graph, NodeId};
use std::collections::HashMap;

pub struct CommonSubexpressionElimination;

impl Pass for CommonSubexpressionElimination {
    fn name(&self) -> &str {
        "common_subexpression_elimination"
    }

    fn run(&self, graph: Graph) -> Graph {
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        // Value-number table, hash-bucketed: `(op, remapped inputs, shape)` →
        // candidate survivors.
        //
        // The key used to be a `(String, Vec<NodeId>, String)` built with two
        // `format!("{:?}", …)` calls **per node**, which is exact but allocates
        // twice for every node in the graph whether or not it has a duplicate.
        // Hashing instead is allocation-free, but a bare 64-bit key would make
        // a collision silently merge two *different* computations — a
        // wrong-output bug, not a missed optimization. So the hash only selects
        // a bucket, and membership is confirmed by exact comparison against the
        // nodes already in it. Misses (the common case) cost one hash; the
        // exact check runs only for genuine hash matches, i.e. essentially only
        // for real duplicates.
        let mut buckets: HashMap<u64, Vec<NodeId>> = HashMap::new();

        for node in graph.nodes() {
            let new_inputs: Vec<NodeId> = node.inputs.iter().map(|id| id_map[id]).collect();

            // Leaves (Input/Param/Constant/… — no inputs) are never merged: a shared
            // shape does not make two inputs the same value.
            if !new_inputs.is_empty() {
                let key = rlx_ir::node_value_key(&node.op, &new_inputs, &node.shape);
                let existing = buckets.get(&key).and_then(|candidates| {
                    candidates.iter().copied().find(|&cand| {
                        let c = new_graph.node(cand);
                        c.inputs == new_inputs
                            && c.shape == node.shape
                            // Deep op equality: `Op`'s derived `PartialEq` stops
                            // at the shallow `Graph` comparison for `Scan` /
                            // `While` / `CustomFn` bodies, so two ops with
                            // different bodies could compare equal.
                            && rlx_ir::ops_deep_eq(&c.op, &node.op, rlx_ir::IgnoreConfig::SEMANTIC)
                    })
                });
                if let Some(existing) = existing {
                    id_map.insert(node.id, existing);
                    continue;
                }
                let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
                if node.name.is_some() || node.origin.is_some() {
                    let n = new_graph.node_mut(new_id);
                    n.name = node.name.clone();
                    n.origin = node.origin.clone();
                }
                buckets.entry(key).or_default().push(new_id);
                id_map.insert(node.id, new_id);
            } else {
                let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
                if node.name.is_some() || node.origin.is_some() {
                    let n = new_graph.node_mut(new_id);
                    n.name = node.name.clone();
                    n.origin = node.origin.clone();
                }
                id_map.insert(node.id, new_id);
            }
        }

        let new_outputs: Vec<NodeId> = graph.outputs.iter().map(|id| id_map[id]).collect();
        new_graph.set_outputs(new_outputs);
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Op, Shape};

    fn f32s(d: &[usize]) -> Shape {
        Shape::new(d, DType::F32)
    }

    #[test]
    fn merges_identical_transpose_matmul() {
        // Two identical `Transpose(u)` and `MatMul(Transpose(u), x)` — the shape a
        // 2-stage synth backward produces — collapse to one each.
        let mut g = Graph::new("bwd");
        let u = g.input("u", f32s(&[4, 3]));
        let x = g.input("x", f32s(&[4, 5]));
        let ut0 = g.add_node(Op::Transpose { perm: vec![1, 0] }, vec![u], f32s(&[3, 4]));
        let ut1 = g.add_node(Op::Transpose { perm: vec![1, 0] }, vec![u], f32s(&[3, 4]));
        let gw0 = g.add_node(Op::MatMul, vec![ut0, x], f32s(&[3, 5]));
        let gw1 = g.add_node(Op::MatMul, vec![ut1, x], f32s(&[3, 5]));
        // Consume both so neither is dead.
        let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![gw0, gw1], f32s(&[3, 5]));
        g.set_outputs(vec![sum]);
        let before = g.nodes().len();

        let out = CommonSubexpressionElimination.run(g);
        let after = out.nodes().len();
        // One Transpose + one MatMul removed (2 inputs + 1 T + 1 MM + 1 Add = 5).
        assert_eq!(after, before - 2, "CSE should drop the duplicate T+MM");
        assert_eq!(after, 5);
    }

    #[test]
    fn keeps_distinct_inputs() {
        // Two same-shape inputs must NOT be merged.
        let mut g = Graph::new("leaves");
        let a = g.input("a", f32s(&[2, 2]));
        let b = g.input("b", f32s(&[2, 2]));
        let s = g.add_node(Op::Binary(BinaryOp::Add), vec![a, b], f32s(&[2, 2]));
        g.set_outputs(vec![s]);
        let out = CommonSubexpressionElimination.run(g);
        // a, b, add all survive.
        assert_eq!(out.nodes().len(), 3);
    }
}

#[cfg(test)]
mod key_equivalence_tests {
    use super::*;
    use rlx_ir::op::{Activation, BinaryOp};
    use rlx_ir::{DType, IgnoreConfig, Op, Shape};
    use std::collections::HashMap;

    /// The previous algorithm verbatim: an exact `(Debug(op), inputs,
    /// Debug(shape))` key. Kept here as the oracle the hash-bucketed version
    /// must agree with — the rewrite trades allocations for a hash, and this
    /// is what proves it did not also trade away exactness.
    fn cse_with_exact_string_keys(graph: Graph) -> Graph {
        let mut new_graph = Graph::new(&graph.name);
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut seen: HashMap<(String, Vec<NodeId>, String), NodeId> = HashMap::new();

        for node in graph.nodes() {
            let new_inputs: Vec<NodeId> = node.inputs.iter().map(|id| id_map[id]).collect();
            if !new_inputs.is_empty() {
                let key = (
                    format!("{:?}", node.op),
                    new_inputs.clone(),
                    format!("{:?}", node.shape),
                );
                if let Some(&existing) = seen.get(&key) {
                    id_map.insert(node.id, existing);
                    continue;
                }
                let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
                seen.insert(key, new_id);
                id_map.insert(node.id, new_id);
            } else {
                let new_id = new_graph.add_node(node.op.clone(), new_inputs, node.shape.clone());
                id_map.insert(node.id, new_id);
            }
        }
        new_graph.set_outputs(graph.outputs.iter().map(|id| id_map[id]).collect());
        new_graph
    }

    /// Lots of duplicate subexpressions plus deliberate near-misses: same op
    /// different operands, same operands different op, same everything
    /// different dtype.
    fn duplicate_rich_graph() -> Graph {
        let mut g = Graph::new("dups");
        let f32s = Shape::new(&[4, 4], DType::F32);
        let f16s = Shape::new(&[4, 4], DType::F16);
        let x = g.input("x", f32s.clone());
        let y = g.input("y", f32s.clone());

        let mut tail = Vec::new();
        for _ in 0..8 {
            // Exact duplicates — must collapse.
            tail.push(g.binary(BinaryOp::Add, x, y, f32s.clone()));
            tail.push(g.binary(BinaryOp::Mul, x, y, f32s.clone()));
            // Operand order differs — must NOT collapse into the above.
            tail.push(g.binary(BinaryOp::Sub, y, x, f32s.clone()));
            tail.push(g.binary(BinaryOp::Sub, x, y, f32s.clone()));
            // Same operands, different activation payload.
            tail.push(g.add_node(Op::Activation(Activation::Gelu), vec![x], f32s.clone()));
            tail.push(g.add_node(Op::Activation(Activation::Relu), vec![x], f32s.clone()));
            // Same op and operands, different result dtype.
            tail.push(g.add_node(Op::Cast { to: DType::F16 }, vec![x], f16s.clone()));
        }
        let mut acc = tail[0];
        for &t in &tail[1..] {
            acc = g.binary(BinaryOp::Add, acc, t, f32s.clone());
        }
        g.set_outputs(vec![acc]);
        g
    }

    #[test]
    fn hash_bucketed_keys_agree_with_exact_string_keys() {
        let graph = duplicate_rich_graph();
        let fast = CommonSubexpressionElimination.run(graph.clone());
        let oracle = cse_with_exact_string_keys(graph);
        assert!(
            fast.structurally_eq(&oracle, IgnoreConfig::SEMANTIC),
            "hash-bucketed CSE diverged from the exact-key oracle\n--- fast ---\n{fast}\n--- oracle ---\n{oracle}"
        );
    }

    #[test]
    fn near_misses_are_not_merged() {
        let out = CommonSubexpressionElimination.run(duplicate_rich_graph());
        let count =
            |pred: &dyn Fn(&rlx_ir::Node) -> bool| out.nodes().iter().filter(|n| pred(n)).count();

        // One survivor each: the duplicates collapsed...
        assert_eq!(
            count(&|n| matches!(n.op, Op::Activation(Activation::Gelu))),
            1
        );
        assert_eq!(
            count(&|n| matches!(n.op, Op::Activation(Activation::Relu))),
            1
        );
        // ...but Sub(x,y) and Sub(y,x) are different values and both survive.
        assert_eq!(count(&|n| matches!(n.op, Op::Binary(BinaryOp::Sub))), 2);
    }

    #[test]
    fn ops_differing_only_inside_a_nested_body_are_not_merged() {
        // `Op`'s derived `PartialEq` compares `Box<Graph>` shallowly (name,
        // node count, outputs), so two Scans with different bodies can compare
        // equal. Merging them would be a wrong-output bug; the exact check uses
        // deep equality precisely for this.
        let shape = Shape::new(&[4], DType::F32);
        let body = |act: Activation| {
            let mut b = Graph::new("body");
            let c = b.input("carry", shape.clone());
            let y = b.add_node(Op::Activation(act), vec![c], shape.clone());
            b.set_outputs(vec![y]);
            Box::new(b)
        };
        let scan = |b| Op::Scan {
            body: b,
            length: 4,
            save_trajectory: false,
            num_bcast: 0,
            num_xs: 0,
            num_checkpoints: 0,
        };

        let mut g = Graph::new("scans");
        let init = g.input("init", shape.clone());
        let a = g.add_node(scan(body(Activation::Gelu)), vec![init], shape.clone());
        let b = g.add_node(scan(body(Activation::Relu)), vec![init], shape.clone());
        let sum = g.binary(BinaryOp::Add, a, b, shape);
        g.set_outputs(vec![sum]);

        let out = CommonSubexpressionElimination.run(g);
        let scans = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Scan { .. }))
            .count();
        assert_eq!(scans, 2, "Scans with different bodies must stay distinct");
    }

    #[test]
    fn identical_nested_bodies_do_merge() {
        let shape = Shape::new(&[4], DType::F32);
        let body = || {
            let mut b = Graph::new("body");
            let c = b.input("carry", shape.clone());
            let y = b.add_node(Op::Activation(Activation::Gelu), vec![c], shape.clone());
            b.set_outputs(vec![y]);
            Box::new(b)
        };
        let scan = |b| Op::Scan {
            body: b,
            length: 4,
            save_trajectory: false,
            num_bcast: 0,
            num_xs: 0,
            num_checkpoints: 0,
        };

        let mut g = Graph::new("scans");
        let init = g.input("init", shape.clone());
        let a = g.add_node(scan(body()), vec![init], shape.clone());
        let b = g.add_node(scan(body()), vec![init], shape.clone());
        let sum = g.binary(BinaryOp::Add, a, b, shape);
        g.set_outputs(vec![sum]);

        let out = CommonSubexpressionElimination.run(g);
        let scans = out
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Scan { .. }))
            .count();
        assert_eq!(scans, 1, "identical Scans are one value");
    }
}
