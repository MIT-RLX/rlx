// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural fingerprints and deep equality for [`Graph`]s.
//!
//! [`Graph`]'s own `PartialEq` is deliberately shallow — it compares name,
//! node count and outputs only, so that [`Op`] (which embeds `Box<Graph>`
//! bodies for `Scan` / `While` / `CustomFn` / …) can keep a cheap derived
//! `PartialEq`. That is the right trade for `Op`, but it leaves the codebase
//! with **no** way to ask "are these two graphs actually the same IR?".
//!
//! This module supplies that, in two forms:
//!
//! * [`structural_hash`] — a `u64` fingerprint, allocation-free on the common
//!   path. Cheap enough to call after every pass, which is what lets
//!   [`Pass`](../../rlx_fusion/pass/trait.Pass.html) report whether it changed
//!   anything and lets the analysis cache decide what to invalidate.
//! * [`structurally_eq`] — exact deep comparison, for golden tests where a
//!   hash match is not good enough.
//!
//! Both take an [`IgnoreConfig`] so that incidental labelling — the graph
//! name, per-node debug names, pass-provenance stamps — can be excluded. Two
//! graphs that differ only in which pass stamped them are the same IR, and a
//! fingerprint that says otherwise would invalidate every cached analysis on
//! every pass.
//!
//! # Keying strategy
//!
//! Ops are keyed by their [`Debug`] rendering, streamed straight into the
//! hasher without building a `String`. This is the same identity CSE has
//! always used (`format!("{:?}", node.op)` as a value number), generalised:
//! every field of every present and future variant participates automatically,
//! with no `Hash`/`Eq` bound on `Op` and no per-op key function to keep in sync.
//!
//! Nested bodies are handled explicitly rather than falling out of `Debug`:
//! an op that carries subgraphs is hashed as a *body-free surrogate* (bodies
//! blanked via [`Op::subgraphs_mut`]) followed by a recursive hash of each
//! real body under the same [`IgnoreConfig`]. Without that, `Debug` would
//! inline the bodies verbatim and the ignore flags would stop applying one
//! level down — a `Scan` whose body nodes were merely renamed would read as
//! changed.
//!
//! # Stability
//!
//! Fingerprints are stable within a process and across processes for a given
//! build, but they are **not** a persistence format: they ride on `Debug`
//! output and on `DefaultHasher`, either of which may shift between compiler
//! or crate versions. Use them for caching, change detection and test
//! assertions — not for on-disk artifacts. [`crate::serialize`] is the
//! durable format.

use std::collections::hash_map::DefaultHasher;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

use crate::graph::{Graph, Node, NodeId};
use crate::op::Op;
use crate::shape::Shape;

/// Which incidental fields to exclude from a structural comparison.
///
/// Construct via the [`EXACT`](Self::EXACT) / [`SEMANTIC`](Self::SEMANTIC)
/// presets, or set the flags directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IgnoreConfig {
    /// Ignore [`Graph::name`].
    pub graph_name: bool,
    /// Ignore [`Node::name`] (debug labels; they carry no semantics).
    pub node_names: bool,
    /// Ignore [`Node::origin`] (pass-provenance stamps).
    pub origins: bool,
}

impl IgnoreConfig {
    /// Everything participates, including names and provenance. Use when you
    /// want to assert that a graph is byte-for-byte what you built.
    pub const EXACT: Self = Self {
        graph_name: false,
        node_names: false,
        origins: false,
    };

    /// Ignore debug-only labelling: graph name, node names, pass provenance.
    ///
    /// This is what "same computation" means in practice, and it is the
    /// default — [`stamp_pass_origins`](crate::stamp_pass_origins) rewrites
    /// every node's origin after every pass, so an origin-sensitive
    /// fingerprint would report a change even for a pass that did nothing.
    pub const SEMANTIC: Self = Self {
        graph_name: true,
        node_names: true,
        origins: true,
    };
}

impl Default for IgnoreConfig {
    fn default() -> Self {
        Self::SEMANTIC
    }
}

/// Streams `Debug`/`Display` output straight into a [`Hasher`], so hashing an
/// op costs no `String` allocation.
struct HashWriter<'a, H: Hasher>(&'a mut H);

impl<H: Hasher> std::fmt::Write for HashWriter<'_, H> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

/// Field separator. Without it `("ab", "c")` and `("a", "bc")` would hash
/// alike, which is exactly how a renamed node could collide with a rewired one.
const SEP: u8 = 0xff;

/// Hash one [`Op`] (including any nested bodies) into `h`.
pub fn hash_op_into<H: Hasher>(op: &Op, cfg: IgnoreConfig, h: &mut H) {
    let bodies = op.subgraphs();
    if bodies.is_empty() {
        // Fast path — no clone, no allocation, whole payload via Debug.
        let _ = write!(HashWriter(h), "{op:?}");
    } else {
        // Blank the bodies so `Debug` renders only this op's own fields, then
        // recurse into the real bodies where the ignore flags still apply.
        let mut surrogate = op.clone();
        for body in surrogate.subgraphs_mut() {
            *body = Graph::new("");
        }
        let _ = write!(HashWriter(h), "{surrogate:?}");
        h.write_u8(SEP);
        (bodies.len() as u64).hash(h);
        for body in bodies {
            hash_graph_into(body, cfg, h);
            h.write_u8(SEP);
        }
    }
    h.write_u8(SEP);
}

/// Hash one [`Node`] into `h`, honouring `cfg`.
fn hash_node_into<H: Hasher>(node: &Node, cfg: IgnoreConfig, h: &mut H) {
    node.id.0.hash(h);
    hash_op_into(&node.op, cfg, h);
    node.inputs.len().hash(h);
    for input in &node.inputs {
        input.0.hash(h);
    }
    node.shape.hash(h);
    if !cfg.node_names {
        node.name.hash(h);
        h.write_u8(SEP);
    }
    if !cfg.origins {
        let _ = write!(HashWriter(h), "{:?}", node.origin);
        h.write_u8(SEP);
    }
}

/// Hash a whole [`Graph`] into `h`, honouring `cfg`.
pub fn hash_graph_into<H: Hasher>(graph: &Graph, cfg: IgnoreConfig, h: &mut H) {
    if !cfg.graph_name {
        graph.name.hash(h);
        h.write_u8(SEP);
    }
    graph.len().hash(h);
    for node in graph.nodes() {
        hash_node_into(node, cfg, h);
    }
    graph.outputs.len().hash(h);
    for out in &graph.outputs {
        out.0.hash(h);
    }
}

/// Structural fingerprint of `graph` under `cfg`.
///
/// Equal IR always yields equal fingerprints. Unequal IR yields unequal
/// fingerprints with overwhelming probability, but this is a 64-bit hash —
/// use [`structurally_eq`] when a false "unchanged" would be a correctness
/// bug rather than a missed optimization.
pub fn structural_hash(graph: &Graph, cfg: IgnoreConfig) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_graph_into(graph, cfg, &mut hasher);
    hasher.finish()
}

/// [`structural_hash`] with [`IgnoreConfig::SEMANTIC`] — the form used for
/// pass change-detection and analysis-cache validation.
pub fn fingerprint(graph: &Graph) -> u64 {
    structural_hash(graph, IgnoreConfig::SEMANTIC)
}

/// Value-number key for a single node, given its inputs *after* remapping.
///
/// This is the CSE key: two interior nodes with the same op, the same
/// remapped inputs and the same shape compute bit-identically. Returns a
/// `u64` rather than the tuple-of-`String`s CSE used to build, which drops
/// two allocations per node.
pub fn node_value_key(op: &Op, remapped_inputs: &[NodeId], shape: &Shape) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_op_into(op, IgnoreConfig::SEMANTIC, &mut hasher);
    remapped_inputs.len().hash(&mut hasher);
    for input in remapped_inputs {
        input.0.hash(&mut hasher);
    }
    shape.hash(&mut hasher);
    hasher.finish()
}

/// Exact deep equality of two ops, including nested bodies.
///
/// Unlike `Op`'s derived `PartialEq`, this does not stop at the shallow
/// `Graph` comparison for body-carrying variants.
pub fn ops_deep_eq(a: &Op, b: &Op, cfg: IgnoreConfig) -> bool {
    // Cheap reject first. The derived `PartialEq` is shallow for bodies, so a
    // `true` here proves nothing — but a `false` does prove inequality.
    if a.kind() != b.kind() {
        return false;
    }

    let (a_bodies, b_bodies) = (a.subgraphs(), b.subgraphs());
    if a_bodies.len() != b_bodies.len() {
        return false;
    }

    if a_bodies.is_empty() {
        return format!("{a:?}") == format!("{b:?}");
    }

    let blank = |op: &Op| {
        let mut surrogate = op.clone();
        for body in surrogate.subgraphs_mut() {
            *body = Graph::new("");
        }
        format!("{surrogate:?}")
    };
    if blank(a) != blank(b) {
        return false;
    }
    a_bodies
        .iter()
        .zip(b_bodies.iter())
        .all(|(x, y)| structurally_eq(x, y, cfg))
}

/// Exact deep equality of two graphs under `cfg`.
///
/// Cheap discriminators (node count, outputs, per-node arity and shape) are
/// checked before any op payload is rendered, so mismatched graphs fail fast.
pub fn structurally_eq(a: &Graph, b: &Graph, cfg: IgnoreConfig) -> bool {
    if !cfg.graph_name && a.name != b.name {
        return false;
    }
    if a.len() != b.len() || a.outputs != b.outputs {
        return false;
    }

    for (x, y) in a.nodes().iter().zip(b.nodes().iter()) {
        if x.id != y.id || x.inputs != y.inputs || x.shape != y.shape {
            return false;
        }
        if !cfg.node_names && x.name != y.name {
            return false;
        }
        if !cfg.origins && format!("{:?}", x.origin) != format!("{:?}", y.origin) {
            return false;
        }
        if !ops_deep_eq(&x.op, &y.op, cfg) {
            return false;
        }
    }
    true
}

impl Graph {
    /// Structural fingerprint under [`IgnoreConfig::SEMANTIC`].
    /// See [`fingerprint`].
    pub fn fingerprint(&self) -> u64 {
        fingerprint(self)
    }

    /// Exact deep equality under `cfg`. See [`structurally_eq`].
    ///
    /// Prefer this over `==`, which is a deliberately shallow
    /// name/count/outputs check.
    pub fn structurally_eq(&self, other: &Graph, cfg: IgnoreConfig) -> bool {
        structurally_eq(self, other, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{Activation, BinaryOp};
    use crate::{DType, Shape};

    fn two_layer(name: &str) -> Graph {
        let mut g = Graph::new(name);
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 4], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 4], DType::F32));
        let act = g.add_node(
            Op::Activation(Activation::Gelu),
            vec![mm],
            Shape::new(&[2, 4], DType::F32),
        );
        g.set_outputs(vec![act]);
        g
    }

    #[test]
    fn identical_graphs_agree() {
        let a = two_layer("m");
        let b = two_layer("m");
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert!(a.structurally_eq(&b, IgnoreConfig::EXACT));
    }

    #[test]
    fn op_payload_change_is_detected() {
        let a = two_layer("m");
        let mut b = two_layer("m");
        // Same kind, same shape, same wiring — only the activation differs.
        b.node_mut(NodeId(3)).op = Op::Activation(Activation::Relu);
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert!(!a.structurally_eq(&b, IgnoreConfig::SEMANTIC));
    }

    #[test]
    fn rewiring_is_detected() {
        let a = two_layer("m");
        let mut b = two_layer("m");
        b.set_inputs(NodeId(3), vec![NodeId(0)]);
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert!(!a.structurally_eq(&b, IgnoreConfig::SEMANTIC));
    }

    #[test]
    fn semantic_config_ignores_labels_exact_does_not() {
        let a = two_layer("m");
        let mut b = two_layer("other_name");
        b.node_mut(NodeId(3)).name = Some("relabelled".into());

        assert_eq!(
            structural_hash(&a, IgnoreConfig::SEMANTIC),
            structural_hash(&b, IgnoreConfig::SEMANTIC)
        );
        assert!(a.structurally_eq(&b, IgnoreConfig::SEMANTIC));

        assert_ne!(
            structural_hash(&a, IgnoreConfig::EXACT),
            structural_hash(&b, IgnoreConfig::EXACT)
        );
        assert!(!a.structurally_eq(&b, IgnoreConfig::EXACT));
    }

    #[test]
    fn shallow_partial_eq_is_the_gap_this_module_closes() {
        // Same name, same node count, same outputs — but different ops.
        // `==` says equal; the structural comparison correctly says otherwise.
        let a = two_layer("m");
        let mut b = two_layer("m");
        b.node_mut(NodeId(3)).op = Op::Binary(BinaryOp::Add);
        b.set_inputs(NodeId(3), vec![NodeId(2), NodeId(2)]);

        assert_eq!(a, b, "Graph::eq is documented-shallow");
        assert!(!a.structurally_eq(&b, IgnoreConfig::SEMANTIC));
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn nested_bodies_participate() {
        let scan = |act: Activation| {
            let mut body = Graph::new("body");
            let c = body.input("carry", Shape::new(&[4], DType::F32));
            let y = body.add_node(Op::Activation(act), vec![c], Shape::new(&[4], DType::F32));
            body.set_outputs(vec![y]);

            let mut g = Graph::new("outer");
            let init = g.input("init", Shape::new(&[4], DType::F32));
            let s = g.add_node(
                Op::Scan {
                    body: Box::new(body),
                    length: 8,
                    save_trajectory: false,
                    num_bcast: 0,
                    num_xs: 0,
                    num_checkpoints: 0,
                },
                vec![init],
                Shape::new(&[4], DType::F32),
            );
            g.set_outputs(vec![s]);
            g
        };

        let a = scan(Activation::Gelu);
        let b = scan(Activation::Gelu);
        let c = scan(Activation::Relu);

        assert_eq!(a.fingerprint(), b.fingerprint());
        assert!(a.structurally_eq(&b, IgnoreConfig::SEMANTIC));

        // A change *inside* the body must surface at the outer graph.
        assert_ne!(a.fingerprint(), c.fingerprint());
        assert!(!a.structurally_eq(&c, IgnoreConfig::SEMANTIC));
    }

    #[test]
    fn ignore_flags_reach_into_nested_bodies() {
        // The point of the body-free-surrogate encoding: renaming a node
        // inside a `Scan` body must not read as a change under SEMANTIC.
        let build = |body_node_name: &str| {
            let mut body = Graph::new("body");
            let c = body.input("carry", Shape::new(&[4], DType::F32));
            let y = body.add_node(
                Op::Activation(Activation::Gelu),
                vec![c],
                Shape::new(&[4], DType::F32),
            );
            body.node_mut(y).name = Some(body_node_name.to_string());
            body.set_outputs(vec![y]);

            let mut g = Graph::new("outer");
            let init = g.input("init", Shape::new(&[4], DType::F32));
            let s = g.add_node(
                Op::Scan {
                    body: Box::new(body),
                    length: 8,
                    save_trajectory: false,
                    num_bcast: 0,
                    num_xs: 0,
                    num_checkpoints: 0,
                },
                vec![init],
                Shape::new(&[4], DType::F32),
            );
            g.set_outputs(vec![s]);
            g
        };

        let a = build("before");
        let b = build("after");
        assert_eq!(a.fingerprint(), b.fingerprint());
        assert!(a.structurally_eq(&b, IgnoreConfig::SEMANTIC));
        assert!(!a.structurally_eq(&b, IgnoreConfig::EXACT));
    }

    #[test]
    fn value_key_separates_operand_order() {
        let shape = Shape::new(&[4], DType::F32);
        let op = Op::Binary(BinaryOp::Sub);
        let ab = node_value_key(&op, &[NodeId(1), NodeId(2)], &shape);
        let ba = node_value_key(&op, &[NodeId(2), NodeId(1)], &shape);
        assert_ne!(ab, ba, "Sub is not commutative — order must key distinctly");
    }
}
