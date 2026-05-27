// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! [`AttentionStage`] implementations for decode and prefill blocks.

use anyhow::Result;

use super::{BlockStage, LlamaDecodeLayerStage, Qwen3DecodeLayerStage, SelfAttnPrefillStage};
use crate::context::FlowCtx;
use crate::stage_contract::{LayerStage, StageArtifacts};
use crate::stage_interfaces::{AttentionStage, KvCacheContract};
use crate::value::FlowValue;

fn kv_from_decode(ctx: &FlowCtx<'_>, layer_idx: usize) -> Result<KvCacheContract> {
    let decode = ctx
        .state
        .decode
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("AttentionStage: BindDecodeInputs required"))?;
    let k = ctx.node_shape(decode.past_k[layer_idx])?;
    let v = ctx.node_shape(decode.past_v[layer_idx])?;
    Ok(KvCacheContract { k, v })
}

impl LayerStage for SelfAttnPrefillStage {
    fn name(&self) -> &str {
        "self_attn_prefill"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let out = BlockStage::emit(self, ctx, input.clone())?;
        let value = out.unwrap_or(input);
        Ok((
            value.clone(),
            StageArtifacts::hidden_only(value.shape.clone()),
        ))
    }
}

impl AttentionStage for SelfAttnPrefillStage {
    fn cache_contract(&self, ctx: &FlowCtx<'_>, hidden: &rlx_ir::Shape) -> KvCacheContract {
        let _ = ctx;
        KvCacheContract {
            k: hidden.clone(),
            v: hidden.clone(),
        }
    }
}

impl LayerStage for LlamaDecodeLayerStage {
    fn name(&self) -> &str {
        "llama_decode_layer"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let out = BlockStage::emit(self, ctx, input.clone())?;
        let value = out.unwrap_or(input);
        Ok((
            value.clone(),
            StageArtifacts::hidden_only(value.shape.clone()),
        ))
    }
}

impl AttentionStage for LlamaDecodeLayerStage {
    fn cache_contract(&self, ctx: &FlowCtx<'_>, hidden: &rlx_ir::Shape) -> KvCacheContract {
        let _ = hidden;
        kv_from_decode(ctx, self.layer_idx).unwrap_or_else(|_| KvCacheContract {
            k: self.spec.hidden_shape.clone(),
            v: self.spec.hidden_shape.clone(),
        })
    }
}

impl LayerStage for Qwen3DecodeLayerStage {
    fn name(&self) -> &str {
        "qwen3_decode_layer"
    }

    fn emit_layer(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts)> {
        let out = BlockStage::emit(self, ctx, input.clone())?;
        let value = out.unwrap_or(input);
        Ok((
            value.clone(),
            StageArtifacts::hidden_only(value.shape.clone()),
        ))
    }
}

impl AttentionStage for Qwen3DecodeLayerStage {
    fn cache_contract(&self, ctx: &FlowCtx<'_>, hidden: &rlx_ir::Shape) -> KvCacheContract {
        let _ = hidden;
        kv_from_decode(ctx, self.layer_idx).unwrap_or_else(|_| KvCacheContract {
            k: self.spec.hidden_shape.clone(),
            v: self.spec.hidden_shape.clone(),
        })
    }
}
