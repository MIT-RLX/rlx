// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Fluent per-layer composer — stack small blocks without IR/Graph imports.

use std::sync::Arc;

use crate::blocks::{
    GatherAddStage, LayerNormStage, LinearStage, ResidualAddStage, ResidualSaveStage,
    RmsNormStage, SelfAttnPrefillSpec, SelfAttnPrefillStage, SwiGluStage,
};
use crate::stage::FlowStage;

/// Stack transformer sub-blocks into one named layer stage.
#[derive(Debug, Clone, Default)]
pub struct LayerStack {
    name: Option<String>,
    stages: Vec<FlowStage>,
}

impl LayerStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            stages: Vec::new(),
        }
    }

    pub fn layer_norm(
        mut self,
        gamma_key: impl Into<String>,
        beta_key: impl Into<String>,
        eps: f32,
    ) -> Self {
        self.stages
            .push(FlowStage::LayerNorm(LayerNormStage::new(gamma_key, beta_key, eps)));
        self
    }

    pub fn gather_add(
        mut self,
        input_name: impl Into<String>,
        weight_key: impl Into<String>,
    ) -> Self {
        self.stages
            .push(FlowStage::GatherAdd(GatherAddStage::new(input_name, weight_key, 0)));
        self
    }

    pub fn rms_norm(mut self, weight_key: impl Into<String>, eps: f32) -> Self {
        self.stages
            .push(FlowStage::RmsNorm(RmsNormStage::new(weight_key, eps)));
        self
    }

    pub fn linear(mut self, weight_key: impl Into<String>, transpose: bool) -> Self {
        self.stages
            .push(FlowStage::Linear(LinearStage::new(weight_key, transpose)));
        self
    }

    pub fn residual_save(mut self) -> Self {
        self.stages.push(FlowStage::ResidualSave(ResidualSaveStage));
        self
    }

    pub fn residual_add(mut self) -> Self {
        self.stages.push(FlowStage::ResidualAdd(ResidualAddStage));
        self
    }

    pub fn swiglu(
        mut self,
        gate_key: impl Into<String>,
        up_key: impl Into<String>,
        down_key: impl Into<String>,
    ) -> Self {
        self.stages.push(FlowStage::SwiGlu(SwiGluStage::new(
            gate_key, up_key, down_key,
        )));
        self
    }

    pub fn swiglu_hf_mlp(mut self, prefix: impl Into<String>) -> Self {
        self.stages
            .push(FlowStage::SwiGlu(SwiGluStage::hf_mlp(prefix)));
        self
    }

    pub fn self_attn_prefill(mut self, spec: SelfAttnPrefillSpec) -> Self {
        self.stages
            .push(FlowStage::SelfAttnPrefill(SelfAttnPrefillStage::new(spec)));
        self
    }

    pub fn stage(mut self, stage: FlowStage) -> Self {
        self.stages.push(stage);
        self
    }

    pub fn stages(mut self, stages: impl IntoIterator<Item = FlowStage>) -> Self {
        self.stages.extend(stages);
        self
    }

    pub fn build(self) -> FlowStage {
        let inner = FlowStage::Sequence(self.stages);
        match self.name {
            Some(name) => FlowStage::Named {
                name,
                inner: Arc::new(inner),
            },
            None => inner,
        }
    }
}
