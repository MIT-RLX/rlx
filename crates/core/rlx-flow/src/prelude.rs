// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Common imports for model flow authors.

pub use crate::blocks::{
    BertEncoderLayerSpec, BertEncoderLayerStage, BertQkvStyle, CustomStage, EmbedStage,
    FfnActivation, GatherAddStage, GatherFromInputStage, GeluFfnStage, LayerNormStage, LinearStage,
    LlamaDecodeLayerSpec, LlamaDecoderSpec, LmHeadStage, MlaAttnPrefillSpec, MlaAttnPrefillStage,
    NomicEncoderLayerSpec, NomicEncoderLayerStage, ResidualAddStage, ResidualSaveStage,
    RmsNormStage, RopeTablesStage, SelfAttnPrefillSpec, SwiGluStage, llama_prefill_layer_composed,
    llama_prefill_layer_fused, transformer_encoder_layer,
};
pub use crate::context::{DecodeBindings, FlowState};
pub use crate::{
    BuiltModel, CompileProfile, Emit, FlowStage, FlowValue, LayerStack, ModelFlow, ModelRecipe,
    SideOutputs, WeightSource,
};
// Downstream extension seam: implement `LayerStage` for a custom block and drop
// it into any flow via `ModelFlow::layer` — no core `FlowStage` variant needed.
// `FlowCtx` is the emission surface the block receives.
pub use crate::{FlowCtx, LayerStage, StageArtifacts};
