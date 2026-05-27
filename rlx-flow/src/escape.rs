// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Tier-2 escape hatch — custom HIR emission when blocks are not enough yet.
//!
//! Prefer adding a reusable block under `blocks/` over long-lived custom closures.
//! Custom stages are for one-off arch experiments and novel subgraphs.

use std::collections::HashMap;

use anyhow::Result;
use rlx_ir::hir::HirModule;
use rlx_ir::{GraphModule, HirNodeId, Shape};

use crate::context::{FlowCtx, FlowState};
use crate::profile::CompileProfile;
use crate::value::FlowValue;
use crate::weight::WeightSource;

/// Mutable emission context for custom stages (tier 2).
pub struct Emit<'a> {
    pub module: &'a mut GraphModule,
    pub params: &'a mut HashMap<String, Vec<f32>>,
    pub weights: &'a mut dyn WeightSource,
    pub state: &'a mut FlowState,
    pub profile: &'a CompileProfile,
}

impl<'a> Emit<'a> {
    pub(crate) fn from_ctx(ctx: &'a mut FlowCtx<'_>) -> Self {
        Self {
            module: &mut ctx.module,
            params: ctx.params,
            weights: ctx.weights,
            state: ctx.state,
            profile: ctx.profile,
        }
    }

    pub fn hir(&mut self) -> &mut HirModule {
        self.module
            .as_hir_mut()
            .expect("flow context requires HIR stage")
    }

    pub fn load_param(&mut self, key: &str, transpose: bool) -> Result<HirNodeId> {
        let cache_key = if transpose {
            format!("{key}\0t")
        } else {
            key.to_string()
        };
        if let Some(&id) = self.state.loaded_params.get(&cache_key) {
            return Ok(id);
        }
        let (data, shape) = self.weights.take(key, transpose)?;
        let ir_shape = Shape::new(&shape, rlx_ir::DType::F32);
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
        self.synth_param(
            name,
            vec![0f32; len],
            Shape::new(&[len], rlx_ir::DType::F32),
        )
    }

    pub fn hir_and_params(&mut self) -> (&mut HirModule, &mut HashMap<String, Vec<f32>>) {
        (
            self.module
                .as_hir_mut()
                .expect("flow context requires HIR stage"),
            self.params,
        )
    }

    pub fn wrap(&self, id: HirNodeId, shape: Shape) -> FlowValue {
        FlowValue::new(id, shape)
    }

    /// Look up a declared graph input (see [`FlowState::inputs`]).
    pub fn flow_input(&self, name: &str) -> Result<FlowValue> {
        let (id, shape) = self
            .state
            .inputs
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("flow input missing `{name}`"))?;
        Ok(FlowValue::new(*id, shape.clone()))
    }

    pub fn set_named(&mut self, key: impl Into<String>, id: HirNodeId) {
        self.state.named.insert(key.into(), id);
    }

    pub fn named(&self, key: &str) -> Result<HirNodeId> {
        self.state
            .named
            .get(key)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("named flow handle missing `{key}`"))
    }
}

pub use crate::context::DecodeBindings;
