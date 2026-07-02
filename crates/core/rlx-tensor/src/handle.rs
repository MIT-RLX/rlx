// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Shared graph ownership for [`crate::Tensor`] handles.
//!
//! Each data-bearing tensor ([`crate::Tensor::from_vec`] & friends) starts
//! life owning its own graph. When two tensors from *different* graphs meet
//! in a binary op, the right-hand graph is **adopted** into the left's via
//! [`GraphHandle::adopt`] — every node is copied with its operands remapped.
//! Embedded [`rlx_ir::Op::Constant`] data travels inside the nodes, so there
//! is no side table to merge. The result keeps NumPy-style value semantics
//! with no ambient/global graph state.
//!
//! Adoption is **memoized by source identity**: a given source node is copied
//! into a target graph at most once. This (a) collapses repeated merges of the
//! same operand into a single node, and (b) — crucially for autodiff — lets a
//! `wrt` tensor be mapped back to the *exact* node the loss was built from, so
//! gradients actually flow to it instead of a stale duplicate.

use std::cell::RefCell;
use std::collections::HashMap;
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
    /// `(source handle id, source node id) -> node id in this graph`.
    adopted: HashMap<(u64, u32), NodeId>,
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

    /// Copy `other`'s nodes into this graph (memoized), returning the id that
    /// `other_id` maps to *here*. No-op fast path when both handles already
    /// share a graph.
    ///
    /// Nodes are visited in insertion order, which is topological for an rlx
    /// graph, so each operand is already remapped before its consumer.
    pub(crate) fn adopt(&self, other: &GraphHandle, other_id: NodeId) -> NodeId {
        if self.same(other) {
            return other_id;
        }
        let src_id = other.0.borrow().id;
        // Fast path: this exact source node was already adopted.
        if let Some(&mapped) = self.0.borrow().adopted.get(&(src_id, other_id.0)) {
            return mapped;
        }
        let snapshot = other.with_graph(|src| {
            src.nodes()
                .iter()
                .map(|n| {
                    (
                        n.id,
                        n.op.clone(),
                        n.inputs.clone(),
                        n.shape.clone(),
                        n.name.clone(),
                    )
                })
                .collect::<Vec<_>>()
        });
        let mut inner = self.0.borrow_mut();
        let mut local: HashMap<NodeId, NodeId> = HashMap::with_capacity(snapshot.len());
        let mut mapped = other_id;
        for (old_id, op, inputs, shape, name) in snapshot {
            // Reuse a node copied by an earlier adopt of this source.
            if let Some(&existing) = inner.adopted.get(&(src_id, old_id.0)) {
                local.insert(old_id, existing);
                if old_id == other_id {
                    mapped = existing;
                }
                continue;
            }
            let new_inputs: Vec<NodeId> = inputs.iter().map(|i| local[i]).collect();
            let new_id = inner.graph.append_node(op, new_inputs, shape, name);
            inner.adopted.insert((src_id, old_id.0), new_id);
            local.insert(old_id, new_id);
            if old_id == other_id {
                mapped = new_id;
            }
        }
        mapped
    }
}
