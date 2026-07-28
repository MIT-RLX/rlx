// RLX - versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared graph rewriter for fusion passes.

use rlx_ir::{Graph, Node, NodeId, Op, Shape};
use std::collections::HashMap;

/// Maps old [`NodeId`]s to new ones during graph rewriting.
pub(crate) struct Rewriter {
    pub new_graph: Graph,
    id_map: HashMap<NodeId, NodeId>,
}

impl Rewriter {
    pub fn new(name: &str) -> Self {
        Self {
            new_graph: Graph::new(name),
            id_map: HashMap::new(),
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

    pub fn copy_node(&mut self, node: &Node) -> NodeId {
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
        let new_inputs: Vec<NodeId> = old_inputs.iter().map(|id| self.map(*id)).collect();
        self.new_graph.add_node(op, new_inputs, shape)
    }

    pub fn replace(&mut self, old_id: NodeId, new_id: NodeId) {
        self.id_map.insert(old_id, new_id);
    }

    pub fn finish(mut self, old_outputs: &[NodeId]) -> Graph {
        let new_outputs = old_outputs.iter().map(|id| self.map(*id)).collect();
        self.new_graph.set_outputs(new_outputs);
        self.new_graph
    }
}
