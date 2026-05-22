// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Cross-stage node provenance — HIR block → MIR node → fusion pass.

use std::fmt;

use crate::hir::HirNodeId;
use crate::{Graph, NodeId};

/// Where a MIR node came from and how it was produced.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeOrigin {
    /// Source HIR block node, when lowered from [`HirModule`].
    pub hir: Option<HirNodeId>,
    /// Human label (`layer0.ffn`, `swiglu_ffn`, param name, …).
    pub label: Option<String>,
    /// Optimizer pass that last created or fused this node.
    pub pass: Option<String>,
}

impl NodeOrigin {
    pub fn from_hir(hir: HirNodeId, label: Option<String>) -> Self {
        Self {
            hir: Some(hir),
            label,
            pass: None,
        }
    }

    pub fn inherit_from_graph(graph: &Graph, inputs: &[NodeId], pass: &str) -> Self {
        let mut out = Self::default();
        for &id in inputs {
            let node = graph.node(id);
            if let Some(ref o) = node.origin {
                if out.hir.is_none() {
                    out.hir = o.hir;
                }
                if out.label.is_none() {
                    out.label = o.label.clone();
                }
            }
            if out.label.is_none() {
                out.label = node.name.clone();
            }
            if out.hir.is_some() && out.label.is_some() {
                break;
            }
        }
        out.pass = Some(pass.to_string());
        out
    }
}

impl fmt::Display for NodeOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(h) = self.hir {
            parts.push(format!("hir={h}"));
        }
        if let Some(l) = &self.label {
            parts.push(format!("\"{l}\""));
        }
        if let Some(p) = &self.pass {
            parts.push(format!("pass={p}"));
        }
        if parts.is_empty() {
            write!(f, "—")
        } else {
            write!(f, "{}", parts.join(", "))
        }
    }
}

/// Best-effort label for diagnostics (origin label, node name, or id).
pub fn node_label(graph: &Graph, id: NodeId) -> String {
    let node = graph.node(id);
    if let Some(ref o) = node.origin {
        if let Some(ref l) = o.label {
            return l.clone();
        }
        if let Some(h) = o.hir {
            return format!("{h}");
        }
    }
    node.name.clone().unwrap_or_else(|| format!("{id}"))
}

/// Stamp nodes created by a pass (no origin yet) by inheriting from inputs.
pub fn stamp_pass_origins(graph: &mut Graph, pass: &str) {
    let ids: Vec<NodeId> = graph.nodes().iter().map(|n| n.id).collect();
    for id in ids {
        if graph.node(id).origin.is_some() {
            continue;
        }
        let inputs = graph.node(id).inputs.clone();
        let origin = NodeOrigin::inherit_from_graph(graph, &inputs, pass);
        graph.node_mut(id).origin = Some(origin);
    }
}
