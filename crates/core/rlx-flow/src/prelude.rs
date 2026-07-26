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
