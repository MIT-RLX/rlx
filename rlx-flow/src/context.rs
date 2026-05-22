// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Flow context — internal HIR emission surface (not for model authors).

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::hir::{HirModule, HirNodeId};
use rlx_ir::{DType, GraphModule, Shape};

use crate::profile::CompileProfile;
use crate::value::FlowValue;
use crate::weight::WeightSource;

/// Handles for a [`Op::GatedDeltaNet`] / carry scan.
#[derive(Debug, Clone, Copy)]
pub struct GdnInputSlots {
    pub q: HirNodeId,
    pub k: HirNodeId,
    pub v: HirNodeId,
    pub g: HirNodeId,
    pub beta: HirNodeId,
}

/// Cross-stage shared handles (RoPE tables, zero-beta, tied embed, …).
#[derive(Debug, Default)]
pub struct FlowState {
    pub rope_cos: Option<HirNodeId>,
    pub rope_sin: Option<HirNodeId>,
    pub zero_beta: Option<HirNodeId>,
    pub embed_weight: Option<HirNodeId>,
    pub hidden_shape: Option<Shape>,
    pub decode: Option<DecodeBindings>,
    pub residual_skip: Option<HirNodeId>,
    pub residual_shape: Option<Shape>,
    /// Named tensor streams (`img`, `txt`, …) for multi-stream models.
    pub streams: HashMap<String, FlowValue>,
    /// Graph inputs beyond the primary tensor flow (`encoder`, `temb`, …).
    pub inputs: HashMap<String, (HirNodeId, Shape)>,
    /// Named scalar/tensor node refs (RoPE tables, mod params, carry state, …).
    pub named: HashMap<String, HirNodeId>,
    /// Last-published GDN q/k/v/g/beta handles for [`crate::blocks::GdnScanStage`].
    pub gdn: Option<GdnInputSlots>,
    /// Reuse param nodes when multiple stages in one layer load the same key
    /// (e.g. [`crate::blocks::LlamaKvTapStage`] + fused decoder).
    pub loaded_params: HashMap<String, HirNodeId>,
}

/// KV-cache decode inputs bound by [`crate::blocks::BindDecodeInputsStage`].
#[derive(Debug, Clone)]
pub struct DecodeBindings {
    pub cos: HirNodeId,
    pub sin: HirNodeId,
    pub mask: Option<HirNodeId>,
    pub past_k: Vec<HirNodeId>,
    pub past_v: Vec<HirNodeId>,
}

/// Internal builder context. Blocks emit through this — tier-2 via [`crate::escape::Emit`].
pub struct FlowCtx<'a> {
    pub(crate) module: GraphModule,
    pub(crate) params: &'a mut HashMap<String, Vec<f32>>,
    pub(crate) weights: &'a mut dyn WeightSource,
    pub(crate) profile: &'a CompileProfile,
    pub(crate) state: &'a mut FlowState,
}

impl FlowCtx<'_> {
    pub fn hir(&mut self) -> &mut HirModule {
        self.module
            .as_hir_mut()
            .expect("flow context requires HIR stage")
    }

    pub fn node_shape(&self, id: HirNodeId) -> Result<Shape> {
        Ok(self
            .module
            .as_hir()
            .ok_or_else(|| anyhow::anyhow!("flow context requires HIR stage"))?
            .node(id)
            .shape
            .clone())
    }

    pub fn load_param(
        &mut self,
        key: &str,
        transpose: bool,
    ) -> Result<HirNodeId> {
        let cache_key = param_cache_key(key, transpose);
        if let Some(&id) = self.state.loaded_params.get(&cache_key) {
            return Ok(id);
        }
        let (data, shape) = self.weights.take(key, transpose)?;
        let ir_shape = Shape::new(&shape, DType::F32);
        let id = self.hir().param(key, ir_shape);
        self.params.insert(key.to_string(), data);
        self.state.loaded_params.insert(cache_key, id);
        Ok(id)
    }

    pub fn synth_param(&mut self, name: &str, data: Vec<f32>, shape: Shape) -> HirNodeId {
        let id = self.hir().param(name, shape);
        self.params.insert(name.to_string(), data);
        id
    }

    pub fn synth_zeros(&mut self, name: &str, len: usize) -> HirNodeId {
        self.synth_param(name, vec![0f32; len], Shape::new(&[len], DType::F32))
    }

    pub fn input(&mut self, name: &str, shape: Shape) -> HirNodeId {
        self.hir().input(name, shape)
    }

    pub fn wrap(&self, id: HirNodeId, shape: Shape) -> FlowValue {
        FlowValue::new(id, shape)
    }
}

fn param_cache_key(key: &str, transpose: bool) -> String {
    if transpose {
        format!("{key}\0t")
    } else {
        key.to_string()
    }
}
