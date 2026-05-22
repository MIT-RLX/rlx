// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! LLaMA-style prefill layer — fused (default) or small-block composition.

use std::sync::Arc;

use super::{LlamaDecoderSpec, LlamaDecoderStage, SelfAttnPrefillSpec};
use crate::layer::LayerStack;
use crate::stage::FlowStage;

/// Fused HIR composite — fastest path, same as [`LlamaDecoderStage`].
pub fn llama_prefill_layer_fused(layer_idx: usize, spec: LlamaDecoderSpec) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::LlamaDecoder(LlamaDecoderStage::layer(layer_idx, spec))),
    }
}

/// Composed from small blocks — swap individual stages in recipes without touching IR.
pub fn llama_prefill_layer_composed(layer_idx: usize, spec: LlamaDecoderSpec) -> FlowStage {
    let prefix = format!("model.layers.{layer_idx}");
    LayerStack::named(format!("layer{layer_idx}"))
        .residual_save()
        .rms_norm(format!("{prefix}.input_layernorm.weight"), spec.eps)
        .self_attn_prefill(SelfAttnPrefillSpec::hf_layer(
            &prefix,
            spec.num_heads,
            spec.head_dim,
            spec.num_kv_heads,
        ))
        .linear(format!("{prefix}.self_attn.o_proj.weight"), true)
        .residual_add()
        .residual_save()
        .rms_norm(format!("{prefix}.post_attention_layernorm.weight"), spec.eps)
        .swiglu_hf_mlp(&prefix)
        .residual_add()
        .build()
}
