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

//! LLaMA-style prefill layer — fused (default) or small-block composition.

use std::sync::Arc;

use super::{LlamaDecoderSpec, LlamaDecoderStage, SelfAttnPrefillSpec};
use crate::layer::LayerStack;
use crate::stage::FlowStage;

/// Fused HIR composite — fastest path, same as [`LlamaDecoderStage`].
pub fn llama_prefill_layer_fused(layer_idx: usize, spec: LlamaDecoderSpec) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::LlamaDecoder(LlamaDecoderStage::layer(
            layer_idx, spec,
        ))),
    }
}

/// DEBUG: attention half only (residual_save → rmsnorm → attn → o_proj → residual_add).
/// Used to bisect Metal layer divergence between the attn block and the MLP block.
pub fn llama_prefill_layer_attn_only(layer_idx: usize, spec: LlamaDecoderSpec) -> FlowStage {
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
        .build()
}

/// DEBUG: MLP half only (residual_save → rmsnorm → swiglu → residual_add).
pub fn llama_prefill_layer_mlp_only(layer_idx: usize, spec: LlamaDecoderSpec) -> FlowStage {
    let prefix = format!("model.layers.{layer_idx}");
    LayerStack::named(format!("layer{layer_idx}"))
        .residual_save()
        .rms_norm(
            format!("{prefix}.post_attention_layernorm.weight"),
            spec.eps,
        )
        .swiglu_hf_mlp(format!("{prefix}.mlp"))
        .residual_add()
        .build()
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
        .rms_norm(
            format!("{prefix}.post_attention_layernorm.weight"),
            spec.eps,
        )
        // `SwiGluStage::hf_mlp` appends `.gate_proj`/`.up_proj`/`.down_proj` to the
        // prefix as-is, so the HF `.mlp.` infix must be included here.
        .swiglu_hf_mlp(format!("{prefix}.mlp"))
        .residual_add()
        .build()
}
