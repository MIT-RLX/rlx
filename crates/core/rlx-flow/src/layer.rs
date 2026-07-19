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

//! Fluent per-layer composer — stack small blocks without IR/Graph imports.

use std::sync::Arc;

use crate::blocks::{
    BiMapStage, GatherAddStage, LayerNormStage, LinearStage, LogEigStage, ReEigStage,
    ResidualAddStage, ResidualSaveStage, RmsNormStage, SelfAttnPrefillSpec, SelfAttnPrefillStage,
    SpdBatchNormStage, SwiGluStage,
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
        self.stages.push(FlowStage::LayerNorm(LayerNormStage::new(
            gamma_key, beta_key, eps,
        )));
        self
    }

    pub fn gather_add(
        mut self,
        input_name: impl Into<String>,
        weight_key: impl Into<String>,
    ) -> Self {
        self.stages.push(FlowStage::GatherAdd(GatherAddStage::new(
            input_name, weight_key, 0,
        )));
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

    /// SPDNet BiMap bilinear layer: `Y = W · X · Wᵀ` with `W [out_dim, n]`
    /// (F64, semi-orthogonal). `n` is inferred from the current SPD input at
    /// build time; `out_dim` sizes the output `[out_dim, out_dim]` and the F64
    /// weight param, so it must be supplied here (the flow declares the param
    /// node's shape eagerly).
    pub fn bimap(mut self, w_key: impl Into<String>, out_dim: usize) -> Self {
        self.stages
            .push(FlowStage::BiMap(BiMapStage::new(w_key, out_dim)));
        self
    }

    /// SPDNet ReEig nonlinearity (`Y = U·max(ε,Σ)·Uᵀ`). No weights.
    pub fn reeig(mut self, eps: f32) -> Self {
        self.stages.push(FlowStage::ReEig(ReEigStage::new(eps)));
        self
    }

    /// SPDNet LogEig layer (`Y = logm(X)`; SPD manifold → tangent space). No
    /// weights.
    pub fn logeig(mut self, eps: f32) -> Self {
        self.stages.push(FlowStage::LogEig(LogEigStage::new(eps)));
        self
    }

    /// SPD batch-norm — **eval mode**. Loads the learnable SPD bias `G [n, n]`
    /// and the frozen running Fréchet mean `[n, n]` (both F64), then applies
    /// the affine transport. Training-time batch-mean + running-mean update is
    /// the trainer's job — see [`crate::blocks::SpdBatchNormStage`].
    pub fn spd_batch_norm(
        mut self,
        g_key: impl Into<String>,
        running_mean_key: impl Into<String>,
        eps: f32,
    ) -> Self {
        self.stages
            .push(FlowStage::SpdBatchNorm(SpdBatchNormStage::new(
                g_key,
                running_mean_key,
                eps,
            )));
        self
    }

    pub fn stage(mut self, stage: FlowStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Stack a downstream-defined [`LayerStage`](crate::LayerStage) block
    /// (extension seam — see
    /// [`ModelFlow::layer_stage`](crate::ModelFlow::layer_stage)).
    pub fn layer_stage(mut self, stage: impl crate::LayerStage + 'static) -> Self {
        self.stages.push(FlowStage::dynamic(stage));
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
