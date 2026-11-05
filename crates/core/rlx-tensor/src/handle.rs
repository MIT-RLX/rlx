// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared graph ownership for [`crate::Tensor`] handles.
//!
//! Each data-bearing tensor ([`crate::Tensor::from_vec`] & friends) starts
//! life owning its own graph. When two tensors from *different* graphs meet
//! in a binary op, the right-hand graph is **adopted** into the left's via
//! [`GraphHandle::adopt`] — the operand's ancestors are copied with their
//! operands remapped.
//! Embedded [`rlx_ir::Op::Constant`] data travels inside the nodes, so there
//! is no side table to merge. The result keeps NumPy-style value semantics
//! with no ambient/global graph state.
//!
//! Adoption is **memoized by first-home identity**: a given node is copied
//! into a target graph at most once, however many graphs it came through. This (a) collapses repeated merges of the
//! same operand into a single node, and (b) — crucially for autodiff — lets a
//! `wrt` tensor be mapped back to the *exact* node the loss was built from, so
//! gradients actually flow to it instead of a stale duplicate.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use rlx_ir::{Graph, NodeId};

/// Process-wide monotonic id stamped on each handle. Used as the stable key
/// for adoption memoization (an `Rc` address could be recycled after free).
static HANDLE_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct Inner {
    id: u64,
    graph: Graph,
    /// A node's first home `(handle id, node id)` -> node id in this graph.
    adopted: HashMap<(u64, u32), NodeId>,
    /// The inverse, for this graph's adopted nodes.
    roots: HashMap<u32, (u64, u32)>,
}

#[derive(Clone, Debug)]
pub(crate) struct GraphHandle(Rc<RefCell<Inner>>);

impl GraphHandle {
    pub(crate) fn new(graph: Graph) -> Self {
        let id = HANDLE_SEQ.fetch_add(1, Ordering::Relaxed);
        Self(Rc::new(RefCell::new(Inner {
            id,
            graph,
            adopted: HashMap::new(),
            roots: HashMap::new(),
        })))
    }

    pub(crate) fn with_graph<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Graph) -> R,
    {
        f(&mut self.0.borrow_mut().graph)
    }

    /// Same underlying graph (cheap pointer compare)?
    pub(crate) fn same(&self, other: &GraphHandle) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// Copy what `other_id` depends on into this graph (memoized), returning
    /// the id it maps to *here*. No-op fast path when both handles already
    /// share a graph.
    ///
    /// Identity is the node's first home: a copy of a copy is the same value,
    /// and a node coming back is this graph's own, so graphs trading results
    /// grow by what each round adds. Nodes are appended in id order, which is
    /// topological for an rlx graph, so each operand is remapped first.
    pub(crate) fn adopt(&self, other: &GraphHandle, other_id: NodeId) -> NodeId {
        if self.same(other) {
            return other_id;
        }
        let src = other.0.borrow();
        let mut inner = self.0.borrow_mut();
        let mut local: HashMap<NodeId, NodeId> = HashMap::new();
        let mut copy: Vec<NodeId> = Vec::new();
        let mut seen = HashSet::new();
        let mut todo = vec![other_id];
        while let Some(id) = todo.pop() {
            if !seen.insert(id) {
                continue;
            }
            match inner.resolve(src.root(id)) {
                Some(here) => {
                    local.insert(id, here);
                }
                None => {
                    copy.push(id);
                    todo.extend(&src.graph.node(id).inputs);
                }
            }
        }
        copy.sort_unstable_by_key(|id| id.0);
        for id in copy {
            let n = src.graph.node(id);
            let inputs: Vec<NodeId> = n.inputs.iter().map(|i| local[i]).collect();
            let new_id =
                inner
                    .graph
                    .append_node(n.op.clone(), inputs, n.shape.clone(), n.name.clone());
            let root = src.root(id);
            inner.adopted.insert(root, new_id);
            inner.roots.insert(new_id.0, root);
            local.insert(id, new_id);
        }
        local[&other_id]
    }
}

impl Inner {
    /// The first home of this graph's node `id`.
    fn root(&self, id: NodeId) -> (u64, u32) {
        self.roots.get(&id.0).copied().unwrap_or((self.id, id.0))
    }

    /// This graph's node for `root`, if it has one.
    fn resolve(&self, root: (u64, u32)) -> Option<NodeId> {
        if root.0 == self.id {
            Some(NodeId(root.1))
        } else {
            self.adopted.get(&root).copied()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn nodes(t: &Tensor) -> usize {
        t.handle.with_graph(|g| g.nodes().len())
    }

    /// Two graphs trading results: what comes back is the graph's own node,
    /// not a copy of a copy, so each round adds the same few nodes.
    #[test]
    fn adoption_does_not_duplicate_a_returning_node() {
        let (mut x, mut y) = (Tensor::from_f64([1.0], [1]), Tensor::from_f64([0.5], [1]));
        let (mut wx, mut wy) = (1.0f64, 0.5f64);
        let mut sizes = Vec::new();
        for _ in 0..12 {
            let (a, b) = (&x * &y, &y * &x);
            (x, y) = (&a + &b, &b / &a);
            (wx, wy) = (2.0 * wx * wy, 1.0);
            sizes.push(nodes(&x).max(nodes(&y)));
        }
        assert_eq!((x.to_vec_f64(), y.to_vec_f64()), (vec![wx], vec![wy]));
        let step = sizes[11] - sizes[10];
        assert!(
            sizes.windows(2).skip(1).all(|w| w[1] - w[0] == step),
            "{sizes:?}"
        );
    }

    /// Only what the operand depends on crosses over.
    #[test]
    fn adoption_copies_ancestors_only() {
        let a = Tensor::from_f64([1.0], [1]);
        let b = Tensor::from_f64([2.0], [1]);
        let used = &b + &b;
        let mut unused = &b * &b;
        for _ in 0..50 {
            unused = &unused * &b;
        }
        let before = nodes(&a);
        let c = &a + &used;
        assert_eq!(nodes(&c), before + 3, "b, b + b, and the sum");
        assert_eq!(c.to_vec_f64(), vec![5.0]);
        let _ = unused;
    }
}
