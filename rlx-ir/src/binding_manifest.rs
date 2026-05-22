// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Reflected binding layout from a specialized [`LirModule`].
//!
//! Host code uses this to fill weights and inputs without hand-maintaining parallel
//! struct layouts (the shading-system “parameter block” pattern).

use crate::lir::LirModule;
use crate::{DType, NodeId, Shape};

/// One named graph boundary with arena layout after buffer planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoBindingEntry {
    pub name: String,
    pub node: NodeId,
    pub dtype: DType,
    pub shape: Shape,
    pub elem_count: usize,
    pub byte_size: usize,
    pub arena_offset: Option<usize>,
    pub arena_size: Option<usize>,
    pub is_view: bool,
}

/// Full I/O + parameter manifest for a compiled graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingManifest {
    pub graph_name: String,
    pub arena_size: usize,
    pub alignment: usize,
    pub inputs: Vec<IoBindingEntry>,
    pub params: Vec<IoBindingEntry>,
    pub outputs: Vec<IoBindingEntry>,
}

impl BindingManifest {
    pub fn from_lir(lir: &LirModule) -> Self {
        let graph = lir.as_graph();
        let plan = lir.plan();
        let io = &plan.io;

        let mut inputs = Vec::new();
        for (name, id) in &io.inputs {
            if let Some(e) = entry_for_node(graph, plan, name.clone(), *id) {
                inputs.push(e);
            }
        }

        let mut params = Vec::new();
        for (name, id) in &io.params {
            if let Some(e) = entry_for_node(graph, plan, name.clone(), *id) {
                params.push(e);
            }
        }

        let mut outputs = Vec::new();
        for (i, id) in io.outputs.iter().enumerate() {
            let name = format!("output{i}");
            if let Some(e) = entry_for_node(graph, plan, name, *id) {
                outputs.push(e);
            }
        }

        Self {
            graph_name: lir.name().to_string(),
            arena_size: plan.arena_size,
            alignment: plan.alignment,
            inputs,
            params,
            outputs,
        }
    }

    pub fn param_names(&self) -> impl Iterator<Item = &str> {
        self.params.iter().map(|p| p.name.as_str())
    }

    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().map(|p| p.name.as_str())
    }

    pub fn param_byte_size(&self, name: &str) -> Option<usize> {
        self.params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.byte_size)
    }

    pub fn total_param_bytes(&self) -> usize {
        self.params.iter().map(|p| p.byte_size).sum()
    }

    /// Group parameters by dot-prefix (`layer0.attn` → block `layer0`).
    pub fn weight_blocks(&self) -> Vec<WeightBlock> {
        let mut blocks: std::collections::HashMap<String, Vec<IoBindingEntry>> =
            std::collections::HashMap::new();
        for p in &self.params {
            let block = p
                .name
                .split('.')
                .next()
                .unwrap_or(&p.name)
                .to_string();
            blocks.entry(block).or_default().push(p.clone());
        }
        let mut out: Vec<WeightBlock> = blocks
            .into_iter()
            .map(|(prefix, params)| {
                let byte_size = params.iter().map(|e| e.byte_size).sum();
                WeightBlock {
                    prefix,
                    params,
                    byte_size,
                }
            })
            .collect();
        out.sort_by(|a, b| a.prefix.cmp(&b.prefix));
        out
    }
}

/// Nested parameter block (Slang PerFrame / material grouping).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBlock {
    pub prefix: String,
    pub params: Vec<IoBindingEntry>,
    pub byte_size: usize,
}

fn entry_for_node(
    graph: &crate::Graph,
    plan: &crate::lir::LirBufferPlan,
    name: String,
    id: NodeId,
) -> Option<IoBindingEntry> {
    let node = graph.node(id);
    let elem_count = node.shape.num_elements().unwrap_or(0);
    let byte_size = elem_count * node.shape.dtype().size_bytes();
    let (arena_offset, arena_size, is_view) = if let Some(alias) = plan.view_aliases.get(&id) {
        let root_slot = plan.slot(alias.root)?;
        (
            Some(root_slot.offset + alias.byte_offset),
            Some(byte_size),
            true,
        )
    } else if let Some(slot) = plan.slot(id) {
        (Some(slot.offset), Some(slot.size), false)
    } else {
        (None, None, false)
    };
    Some(IoBindingEntry {
        name,
        node: id,
        dtype: node.shape.dtype(),
        shape: node.shape.clone(),
        elem_count,
        byte_size,
        arena_offset,
        arena_size,
        is_view,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lir::{LirBufferPlan, LirBufferSlot, LirIoManifest};
    use crate::Graph;

    #[test]
    fn manifest_lists_params_with_sizes() {
        let mut g = Graph::new("t");
        let x = g.input("x", Shape::new(&[2, 4], DType::F32));
        let w = g.param("w", Shape::new(&[4, 3], DType::F32));
        let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
        g.set_outputs(vec![mm]);

        let mut plan = LirBufferPlan::default();
        plan.io = LirIoManifest::collect(&g);
        plan.assignments.insert(x, LirBufferSlot { offset: 0, size: 32 });
        plan.assignments.insert(w, LirBufferSlot { offset: 32, size: 48 });
        plan.assignments.insert(mm, LirBufferSlot { offset: 80, size: 24 });
        plan.arena_size = 104;

        let lir = LirModule::new(crate::MirModule::from_graph(g), plan);
        let m = BindingManifest::from_lir(&lir);
        assert_eq!(m.params.len(), 1);
        assert_eq!(m.params[0].name, "w");
        assert_eq!(m.params[0].byte_size, 48);
        assert_eq!(m.inputs[0].name, "x");
    }
}
