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

//! The computation graph — a DAG of typed tensor operations.
//!
//! Graphs are append-only during construction (like SSA). Nodes reference
//! inputs by [`NodeId`], forming a directed acyclic graph. The graph
//! owns all nodes and provides traversal, printing, and validation.

use crate::{Op, Shape};

use crate::provenance::NodeOrigin;

/// Stable identifier for a node in the graph. Indices are never reused.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// A single node in the computation graph.
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    /// The operation this node performs.
    pub op: Op,
    /// Input node IDs (operands). Order matches `Op::num_inputs()`.
    pub inputs: Vec<NodeId>,
    /// Output tensor shape (computed at construction time).
    pub shape: Shape,
    /// Human-readable name for debugging.
    pub name: Option<String>,
    /// Cross-stage provenance (HIR block, fusion pass, …).
    pub origin: Option<NodeOrigin>,
}

/// A computation graph — the core IR data structure.
///
/// # Example
/// ```
/// use rlx_ir::*;
///
/// let mut g = Graph::new("bert_layer");
///
/// // Inputs
/// let x = g.input("hidden", Shape::new(&[4, 15, 384], DType::F32));
/// let w = g.param("qkv_weight", Shape::new(&[384, 1152], DType::F32));
/// let b = g.param("qkv_bias", Shape::new(&[1152], DType::F32));
///
/// // QKV projection: matmul + bias
/// let mm = g.matmul(x, w, Shape::new(&[4, 15, 1152], DType::F32));
/// let qkv = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[4, 15, 1152], DType::F32));
///
/// assert_eq!(g.len(), 5);
/// println!("{g}");
/// ```
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug)]
pub struct Graph {
    pub name: String,
    nodes: Vec<Node>,
    /// Output node IDs (the graph's results).
    pub outputs: Vec<NodeId>,
}

// Subgraph equality is structural: same name, same node count, same outputs.
// Full deep equality would require comparing every node and is rarely useful;
// this gives Op derives `PartialEq` cheap structural comparison.
impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.nodes.len() == other.nodes.len()
            && self.outputs == other.outputs
    }
}

impl Graph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a node by ID.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Iterate all nodes in topological order (insertion order = topo order).
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Get the shape of a node's output.
    pub fn shape(&self, id: NodeId) -> &Shape {
        &self.nodes[id.0 as usize].shape
    }

    /// Set the graph outputs.
    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.outputs = outputs;
    }

    /// Replace the input list of a node in place. Used by post-
    /// construction passes (`quant_propagate`, `dce`, etc.) that
    /// rewire consumers without inserting new nodes.
    /// Caller is responsible for shape consistency — this does no
    /// re-inference.
    pub fn set_inputs(&mut self, id: NodeId, inputs: Vec<NodeId>) {
        self.nodes[id.0 as usize].inputs = inputs;
    }

    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0 as usize]
    }

    pub fn nodes_mut(&mut self) -> &mut [Node] {
        &mut self.nodes
    }

    // ── Node constructors ───────────────────────────────────────

    /// Append a node to the graph. `pub(crate)` so per-op builder
    /// files in `rlx_ir::ops::*` can call it (plan #53).
    /// Append a node for backend graph slicing (e.g. TPU HLO segments).
    pub fn append_node(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        shape: Shape,
        name: Option<String>,
    ) -> NodeId {
        self.push(op, inputs, shape, name)
    }

    pub(crate) fn push(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        shape: Shape,
        name: Option<String>,
    ) -> NodeId {
        self.push_ext(op, inputs, shape, name, None)
    }

    pub(crate) fn push_ext(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        shape: Shape,
        name: Option<String>,
        origin: Option<NodeOrigin>,
    ) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            id,
            op,
            inputs,
            shape,
            name,
            origin,
        });
        id
    }

    // Per-op builders moved to `crate::ops::*` (plan #53).
    // Adding new op families = drop a new file in `ops/`, no edits here.

    // ── Analysis helpers ────────────────────────────────────────

    /// Find all nodes that use a given node's output.
    pub fn users(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.inputs.contains(&id))
            .map(|n| n.id)
            .collect()
    }

    /// Count how many nodes use a given node's output.
    pub fn use_count(&self, id: NodeId) -> usize {
        self.nodes.iter().filter(|n| n.inputs.contains(&id)).count()
    }

    /// Topological order (already guaranteed by construction — just node indices).
    pub fn topo_order(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.nodes.len()).map(|i| NodeId(i as u32))
    }

    /// Reverse topological order (outputs first).
    pub fn reverse_topo(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.nodes.len()).rev().map(|i| NodeId(i as u32))
    }

    // ── HIR / MIR / LIR pipeline (higher-order DX) ─────────────────

    /// Fusion-first model definition at HIR level.
    ///
    /// Returns a [`GraphModule`] at HIR stage; call [`GraphModule::lower`]
    /// or pass to [`rlx_opt::CompilePipeline::compile_module`].
    pub fn define(
        name: impl Into<String>,
        build: impl FnOnce(&mut crate::hir::HirModule) -> crate::hir::HirNodeId,
    ) -> crate::GraphModule {
        crate::GraphModule::define(name, build)
    }

    /// Start an empty HIR-stage [`GraphModule`].
    pub fn hir(name: impl Into<String>) -> crate::GraphModule {
        crate::GraphModule::hir(name)
    }

    /// Wrap this MIR graph in a [`GraphModule`] for pipeline operations.
    pub fn module(self) -> crate::GraphModule {
        crate::GraphModule::from_graph(self)
    }

    /// Lower a HIR module to a MIR graph.
    pub fn from_hir(hir: crate::hir::HirModule) -> Result<Self, crate::hir::LowerError> {
        hir.lower_to_mir().map(|m| m.into_graph())
    }

    /// View as [`MirModule`].
    pub fn to_mir(self) -> crate::MirModule {
        crate::MirModule::from_graph(self)
    }

    /// Extract the MIR graph from optimized LIR.
    pub fn from_lir(lir: crate::LirModule) -> Self {
        lir.into_graph()
    }

    /// Annotated text dump ([`inspect_graph`]).
    pub fn inspect(&self) -> String {
        crate::inspect_graph(self)
    }

    /// True if any node shape uses a [`Dim::Dynamic`] symbol.
    pub fn has_dynamic_dims(&self) -> bool {
        crate::dynamic::has_dynamic_dims(self)
    }

    /// All dynamic symbols referenced in this graph.
    pub fn dynamic_symbols(&self) -> Vec<u32> {
        crate::dynamic::collect_dynamic_symbols(self)
    }

    /// Specialize symbolic dims to concrete sizes.
    pub fn bind(&self, bindings: &crate::DimBinding) -> Self {
        crate::dynamic::bind_graph(self, bindings)
    }

    /// Stage-aware dump when wrapped in [`GraphModule`].
    pub fn inspect_module(module: &crate::GraphModule) -> String {
        module.inspect()
    }
}

/// Pretty-print the graph in a readable IR format.
impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "graph @{} {{", self.name)?;
        for node in &self.nodes {
            write!(f, "  {} = {}", node.id, node.op)?;
            if !node.inputs.is_empty() {
                write!(f, "(")?;
                for (i, inp) in node.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{inp}")?;
                }
                write!(f, ")")?;
            }
            writeln!(f, " : {}", node.shape)?;
        }
        if !self.outputs.is_empty() {
            write!(f, "  return ")?;
            for (i, o) in self.outputs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{o}")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        op::{Activation, BinaryOp},
    };

    #[test]
    fn build_simple_graph() {
        let mut g = Graph::new("test");

        let x = g.input("x", Shape::new(&[4, 15, 384], DType::F32));
        let w = g.param("weight", Shape::new(&[384, 1536], DType::F32));
        let b = g.param("bias", Shape::new(&[1536], DType::F32));

        let mm = g.matmul(x, w, Shape::new(&[4, 15, 1536], DType::F32));
        let add = g.binary(BinaryOp::Add, mm, b, Shape::new(&[4, 15, 1536], DType::F32));
        let out = g.activation(
            Activation::Gelu,
            add,
            Shape::new(&[4, 15, 1536], DType::F32),
        );

        g.set_outputs(vec![out]);

        assert_eq!(g.len(), 6);
        assert_eq!(g.use_count(mm), 1); // matmul used by add
        assert_eq!(g.use_count(x), 1); // x used by matmul

        let printed = format!("{g}");
        assert!(printed.contains("matmul(%0, %1)"));
        assert!(printed.contains("Gelu(%4)"));
        assert!(printed.contains("return %5"));
    }

    /// Build a BERT layer to verify the IR can represent real models.
    #[test]
    fn bert_layer_graph() {
        let mut g = Graph::new("bert_layer");
        let f = DType::F32;
        let h = 384;
        let int = 1536;

        // Input
        let x = g.input("hidden", Shape::new(&[4, 15, h], f));

        // QKV
        let qkv_w = g.param("qkv.weight", Shape::new(&[h, 3 * h], f));
        let qkv_b = g.param("qkv.bias", Shape::new(&[3 * h], f));
        let qkv = g.matmul(x, qkv_w, Shape::new(&[4, 15, 3 * h], f));
        let _qkv = g.binary(BinaryOp::Add, qkv, qkv_b, Shape::new(&[4, 15, 3 * h], f));

        // (would split Q/K/V, attention, out_proj here — simplified)

        // FFN
        let int_w = g.param("ffn.weight", Shape::new(&[h, int], f));
        let int_b = g.param("ffn.bias", Shape::new(&[int], f));
        let ffn = g.matmul(x, int_w, Shape::new(&[4, 15, int], f));
        let ffn = g.binary(BinaryOp::Add, ffn, int_b, Shape::new(&[4, 15, int], f));
        let ffn = g.activation(Activation::Gelu, ffn, Shape::new(&[4, 15, int], f));

        let out_w = g.param("ffn_out.weight", Shape::new(&[int, h], f));
        let ffn_out = g.matmul(ffn, out_w, Shape::new(&[4, 15, h], f));

        g.set_outputs(vec![ffn_out]);

        assert!(g.len() > 10);
        println!("{g}");
    }
}
