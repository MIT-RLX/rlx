// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

mod attention_stage;
mod attn_mask;
mod cls_pool;
mod qwen3_decode_layer;
mod qwen3_decoder;
mod qwen3_layer;
mod dinov2_layer;
mod vision_layer;
mod vit_attn;
mod layer_scale;
mod nomic_layer;
mod bind_decode;
mod bert_layer;
mod custom;
mod gdn_scan;
mod embed;
mod gather_from_input;
mod gather_last_token;
mod gelu_ffn;
mod layer_norm;
mod linear;
mod llama_decode_layer;
mod llama_decoder;
mod llama_kv_tap;
mod llama_layer;
mod lm_head;
mod repeat;
mod residual;
mod rms_norm;
mod rope;
mod self_attn;
mod swiglu;

pub use attn_mask::{AttnMaskStage, ATTN_MASK};
pub use cls_pool::ClsTokenPoolStage;
pub use qwen3_decode_layer::{Qwen3DecodeLayerSpec, Qwen3DecodeLayerStage};
pub use qwen3_decoder::{Qwen3DecoderSpec, Qwen3DecoderStage};
pub use qwen3_layer::{
    qwen3_decode_layer_fused, qwen3_prefill_layer_fused, qwen3_prefill_layer_fused_kv,
};
pub use dinov2_layer::dinov2_layer_fused;
pub use vision_layer::{VisionSwiGluFfnStage, nomic_vision_layer_fused};
pub use vit_attn::{VitSelfAttnSpec, VitSelfAttnStage};
pub use layer_scale::LayerScaleStage;
pub use nomic_layer::{NomicEncoderLayerSpec, NomicEncoderLayerStage};
pub use bert_layer::{BertEncoderLayerSpec, BertEncoderLayerStage, BertQkvStyle};
pub use bind_decode::BindDecodeInputsStage;
pub use custom::CustomStage;
pub use gdn_scan::GdnScanStage;
pub use embed::EmbedStage;
pub use gather_from_input::{GatherAddStage, GatherFromInputStage};
pub use gather_last_token::GatherLastTokenStage;
pub use gelu_ffn::GeluFfnStage;
pub use layer_norm::LayerNormStage;
pub use linear::LinearStage;
pub use llama_decode_layer::{LlamaDecodeLayerSpec, LlamaDecodeLayerStage};
pub use llama_decoder::{LlamaDecoderSpec, LlamaDecoderStage};
pub use llama_kv_tap::LlamaKvTapStage;
pub use llama_layer::{llama_prefill_layer_composed, llama_prefill_layer_fused};
pub use lm_head::LmHeadStage;
pub use repeat::RepeatStage;
pub use residual::{ResidualAddStage, ResidualSaveStage};
pub use rms_norm::RmsNormStage;
pub use rope::RopeTablesStage;
pub use self_attn::{SelfAttnPrefillSpec, SelfAttnPrefillStage};
pub use swiglu::SwiGluStage;

use anyhow::Result;

use crate::context::FlowCtx;
use crate::value::FlowValue;
use crate::weight::WeightSource;

/// Internal trait for block emission.
pub(crate) trait BlockStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>>;
}
