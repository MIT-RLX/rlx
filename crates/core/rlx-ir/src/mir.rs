// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **MIR** — mid-level IR.
//!
//! The fused, backend-neutral tensor DAG that [`rlx_opt`] runs fusion,
//! precision, and legalization passes on. Today MIR is structurally
//! identical to [`Graph`]; the newtype marks pipeline stage and gives
//! us room to attach MIR-only metadata later (alias sets, layout hints).

use crate::{Graph, Node, NodeId, Op};

/// Mid-level module — optimizer input.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct MirModule {
    inner: Graph,
}

/// MIR node / op aliases (same types as the legacy graph API).
pub type MirNode = Node;
pub type MirNodeId = NodeId;
pub type MirOp = Op;

impl MirModule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            inner: Graph::new(name),
        }
    }

    pub fn from_graph(graph: Graph) -> Self {
        Self { inner: graph }
    }

    pub fn into_graph(self) -> Graph {
        self.inner
    }

    pub fn as_graph(&self) -> &Graph {
        &self.inner
    }

    pub fn as_graph_mut(&mut self) -> &mut Graph {
        &mut self.inner
    }

    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.inner.outputs
    }

    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.inner.set_outputs(outputs);
    }
}

impl From<Graph> for MirModule {
    fn from(graph: Graph) -> Self {
        Self::from_graph(graph)
    }
}

impl From<MirModule> for Graph {
    fn from(mir: MirModule) -> Self {
        mir.into_graph()
    }
}

impl std::fmt::Display for MirModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "mir @{}", self.inner)
    }
}
