// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Reflection over HIR/MIR/LIR — layout and structure without executing.
//!
//! Host code can introspect unspecialized templates (Slang front-end / reflection API)
//! and specialized layouts independently of backend codegen.

use crate::binding_manifest::BindingManifest;
use crate::component::ModelComponent;
use crate::hir::{HirModule, HirNodeId, HirOp};
use crate::lir::LirModule;
use crate::mir::MirModule;
use crate::shape::DimBinding;
use crate::Shape;

/// Introspection of an unspecialized [`HirModule`] (loadModule analogue).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HirReflection {
    pub name: String,
    pub node_count: usize,
    pub fusion_policy: String,
    pub inputs: Vec<(String, Shape)>,
    pub params: Vec<(String, Shape)>,
    pub outputs: Vec<Shape>,
    pub block_labels: Vec<String>,
}

impl HirReflection {
    pub fn from_hir(hir: &HirModule) -> Self {
        let mut inputs = Vec::new();
        let mut params = Vec::new();
        let mut block_labels = Vec::new();
        for node in hir.nodes().iter() {
            let label = node.name.clone().unwrap_or_else(|| format!("{:?}", node.op));
            match &node.op {
                HirOp::Input { name } => inputs.push((name.clone(), node.shape.clone())),
                HirOp::Param { name } => params.push((name.clone(), node.shape.clone())),
                HirOp::LlamaDecoderBlock { .. }
                | HirOp::SwiGLU
                | HirOp::Attention { .. }
                | HirOp::GatedDeltaNet { .. }
                | HirOp::Qwen35MtpHead { .. } => block_labels.push(label),
                _ => {}
            }
        }
        let outputs = hir
            .outputs
            .iter()
            .map(|&id| hir.node(id).shape.clone())
            .collect();
        HirReflection {
            name: hir.name.clone(),
            node_count: hir.nodes().len(),
            fusion_policy: format!("{:?}", hir.fusion_policy),
            inputs,
            params,
            outputs,
            block_labels,
        }
    }
}

/// MIR-level summary after HIR lower (specializeType / graph shape probe).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MirReflection {
    pub name: String,
    pub node_count: usize,
    pub op_kinds: Vec<(String, usize)>,
}

impl MirReflection {
    pub fn from_mir(mir: &MirModule) -> Self {
        let g = mir.as_graph();
        let mut counts = std::collections::HashMap::new();
        for node in g.nodes() {
            *counts.entry(format!("{:?}", node.op.kind())).or_default() += 1;
        }
        let mut op_kinds: Vec<_> = counts.into_iter().collect();
        op_kinds.sort_by(|a, b| a.0.cmp(&b.0));
        MirReflection {
            name: g.name.clone(),
            node_count: g.nodes().len(),
            op_kinds,
        }
    }
}

/// Layout reflection from specialized LIR (getTypeLayout / parameter block).
pub fn layout_from_lir(lir: &LirModule) -> BindingManifest {
    BindingManifest::from_lir(lir)
}

/// Layout for a concrete [`ModelComponent`] binding without retaining the graph.
pub fn layout_for_binding(lir: &LirModule, _component: &ModelComponent) -> BindingManifest {
    layout_from_lir(lir)
}

/// Compare template vs specialized manifests (dims / arena may differ).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestDiff {
    pub template_arena: usize,
    pub specialized_arena: usize,
    pub params_only_in_template: Vec<String>,
    pub params_only_in_specialized: Vec<String>,
}

impl ManifestDiff {
    pub fn compare(template: &BindingManifest, specialized: &BindingManifest) -> Self {
        let t: std::collections::HashSet<_> = template.param_names().collect();
        let s: std::collections::HashSet<_> = specialized.param_names().collect();
        Self {
            template_arena: template.arena_size,
            specialized_arena: specialized.arena_size,
            params_only_in_template: t.difference(&s).map(|x| (*x).to_string()).collect(),
            params_only_in_specialized: s.difference(&t).map(|x| (*x).to_string()).collect(),
        }
    }
}

/// Block specialization choice (coarse-grained type argument).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockSpecialization {
    Default,
    FusedTransformerLayer,
    UnfusedPrimitives,
}

/// Record of a specialization decision for tooling (specializeType analogue).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecializeBlockRecord {
    pub node: HirNodeId,
    pub label: String,
    pub choice: BlockSpecialization,
}

/// Probe which HIR blocks would take a given specialization (static; no MIR mutate).
pub fn probe_block_specialization(hir: &HirModule, choice: BlockSpecialization) -> Vec<SpecializeBlockRecord> {
    hir.nodes()
        .iter()
        .filter_map(|node| {
            let fused = matches!(
                node.op,
                HirOp::LlamaDecoderBlock { .. } | HirOp::SwiGLU | HirOp::GatedDeltaNet { .. }
            );
            if !fused {
                return None;
            }
            let effective = match choice {
                BlockSpecialization::Default => BlockSpecialization::Default,
                BlockSpecialization::FusedTransformerLayer => BlockSpecialization::FusedTransformerLayer,
                BlockSpecialization::UnfusedPrimitives => BlockSpecialization::UnfusedPrimitives,
            };
            Some(SpecializeBlockRecord {
                node: node.id,
                label: node.name.clone().unwrap_or_else(|| format!("{:?}", node.op)),
                choice: effective,
            })
        })
        .collect()
}

/// Binding-only layout probe when only [`DimBinding`] is known (no full compile).
pub fn symbolic_layout_hint(binding: &DimBinding) -> String {
    format!("DimBinding({:?})", binding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::HirMut;
    use crate::{DType, HirModule};

    #[test]
    fn hir_reflection_lists_inputs() {
        let mut hir = HirModule::new("t");
        let mut gb = HirMut::new(&mut hir);
        let _x = gb.input("x", Shape::new(&[1, 4], DType::F32));
        let _w = gb.param("w", Shape::new(&[4, 2], DType::F32));
        let r = HirReflection::from_hir(&hir);
        assert_eq!(r.inputs.len(), 1);
        assert_eq!(r.params.len(), 1);
    }
}
