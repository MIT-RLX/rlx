// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared graph rewriter for fusion passes.

use rlx_ir::{Graph, Node, NodeId, Op, Shape};
use std::collections::HashMap;

use crate::pass::{IRStatus, PassResult};

/// Maps old [`NodeId`]s to new ones during graph rewriting.
pub(crate) struct Rewriter {
    pub new_graph: Graph,
    id_map: HashMap<NodeId, NodeId>,
    /// Set the moment this rewriter is used to do anything other than copy.
    ///
    /// Fusion passes are rebuild-style: they walk the input graph and either
    /// copy a node or emit a fused replacement. Copying alone leaves the IR
    /// identical, so "did this pass actually fire?" is exactly "was a fused
    /// node emitted, or a consumer redirected?" — and both funnel through this
    /// one type. Tracking it here means every pass built on the rewriter
    /// reports [`IRStatus`] precisely without maintaining its own flag.
    rewrote: bool,
}

impl Rewriter {
    pub fn new(name: &str) -> Self {
        Self {
            new_graph: Graph::new(name),
            id_map: HashMap::new(),
            rewrote: false,
        }
    }

    pub fn map(&self, old: NodeId) -> NodeId {
        self.id_map[&old]
    }

    pub fn map_inputs(&self, old_inputs: &[NodeId]) -> Vec<NodeId> {
        old_inputs.iter().map(|id| self.map(*id)).collect()
    }

    pub fn ensure_mapped(&mut self, old: &Graph, ids: &[NodeId]) {
        for &id in ids {
            if self.id_map.contains_key(&id) {
                continue;
            }
            let node = old.node(id);
            if !node.inputs.is_empty() {
                self.ensure_mapped(old, &node.inputs);
            }
            self.copy_node(node);
        }
    }

    /// Copy `node` into the new graph, at most once.
    ///
    /// The already-mapped guard is load-bearing, not an optimization. A pass
    /// that emits a fused node ahead of its position calls
    /// [`ensure_mapped`](Self::ensure_mapped) to hoist that node's operands,
    /// which copies them early. The pass's main loop then walks the whole old
    /// graph and copies every node it did not fuse — reaching those same
    /// operands a second time. Without this guard that second call emits a
    /// *duplicate* node and repoints `id_map` at it, so every consumer after
    /// the fusion site reads the duplicate rather than the hoisted original.
    ///
    /// For most ops that is merely wasted work — the duplicate recomputes the
    /// same value. For `Op::Param` it is silent corruption: parameter binding
    /// is by name and reaches only one node, so the duplicate keeps whatever
    /// the arena was initialized with (zeros) and every value flowing through
    /// it vanishes.
    ///
    /// A forward graph hides this, because the fused-away consumer was the
    /// param's only reader and the duplicate is dead code the compiler drops.
    /// A *backward* graph does not: `grad_with_loss` mirrors the forward
    /// alongside the gradient ops, so a weight is read twice — once by the
    /// mirrored matmul and once by `dX = dY · Wᵀ` — and the duplicate is live.
    /// That is how `FuseSharedInputMatMul` silently zeroed gradient terms.
    pub fn copy_node(&mut self, node: &Node) -> NodeId {
        if let Some(&existing) = self.id_map.get(&node.id) {
            return existing;
        }
        let new_inputs = self.map_inputs(&node.inputs);
        let new_id = self
            .new_graph
            .add_node(node.op.clone(), new_inputs, node.shape.clone());
        let new_node = self.new_graph.node_mut(new_id);
        new_node.name = node.name.clone();
        new_node.origin = node.origin.clone();
        self.id_map.insert(node.id, new_id);
        new_id
    }

    pub fn add_fused(&mut self, op: Op, old_inputs: &[NodeId], shape: Shape) -> NodeId {
        self.rewrote = true;
        let new_inputs: Vec<NodeId> = old_inputs.iter().map(|id| self.map(*id)).collect();
        self.new_graph.add_node(op, new_inputs, shape)
    }

    pub fn replace(&mut self, old_id: NodeId, new_id: NodeId) {
        self.rewrote = true;
        self.id_map.insert(old_id, new_id);
    }

    /// Did this rewriter emit a fused node or redirect a consumer?
    ///
    /// A pass that only ever called [`copy_node`](Self::copy_node) produced a
    /// graph identical to its input.
    pub fn fired(&self) -> bool {
        self.rewrote
    }

    pub fn finish(mut self, old_outputs: &[NodeId]) -> Graph {
        let new_outputs = old_outputs.iter().map(|id| self.map(*id)).collect();
        self.new_graph.set_outputs(new_outputs);
        self.new_graph
    }

    /// [`finish`](Self::finish), reporting whether the pass actually fired.
    pub fn finish_reporting(self, old_outputs: &[NodeId]) -> PassResult {
        let status = IRStatus::from(self.fired());
        PassResult::from_status(self.finish(old_outputs), status)
    }
}
