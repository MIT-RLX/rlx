// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Flow stages — typed block assembly primitives.

use std::sync::Arc;

use anyhow::Result;

use crate::blocks::{
    AttnMaskStage, BertEncoderLayerStage, BindDecodeInputsStage, BlockStage, ClsTokenPoolStage,
    CustomStage, EmbedStage, GatherAddStage, GatherFromInputStage, GatherLastTokenStage,
    GdnScanStage, GeluFfnStage, LayerNormStage, LayerScaleStage, LinearStage,
    LlamaDecodeLayerStage, LlamaDecoderStage, LlamaKvTapStage, LmHeadStage, NomicEncoderLayerStage,
    Qwen3DecodeLayerStage, Qwen3DecoderStage, RepeatStage, ResidualAddStage, ResidualSaveStage,
    RmsNormStage, RopeTablesStage, SelfAttnPrefillStage, SwiGluStage, VisionSwiGluFfnStage,
    VitSelfAttnStage,
};
use crate::context::FlowCtx;
use crate::stream::{DualStreamStage, LoadStreamStage, StoreStreamStage};
use crate::value::FlowValue;
/// One stage in a model flow. Model authors compose these — not HIR ops.
#[derive(Debug, Clone)]
pub enum FlowStage {
    /// Token embedding lookup.
    Embed(EmbedStage),
    /// Precomputed RoPE sin/cos tables as params.
    RopeTables(RopeTablesStage),
    /// Ensure a rank-1 zero vector exists for RMSNorm beta slots.
    ZeroBeta { name: String, len: usize },
    /// Bind decode inputs (RoPE slice, past K/V, mask) into flow state.
    BindDecodeInputs(BindDecodeInputsStage),
    /// Bind or synthesize vision attention mask (all-ones).
    AttnMask(AttnMaskStage),
    /// KV-cache decode layer (concat past K/V, causal/custom attention).
    LlamaDecodeLayer(LlamaDecodeLayerStage),
    /// LLaMA-style fused prefill decoder layer (GQA + RoPE + SwiGLU).
    LlamaDecoder(LlamaDecoderStage),
    /// Side-output K/V projections for a decoder layer (prefill cache export).
    LlamaKvTap(LlamaKvTapStage),
    /// Repeat an inner stage `count` times with a per-index name prefix.
    Repeat(RepeatStage),
    /// Named nested scope (fusion/debug labeling).
    Named { name: String, inner: Arc<FlowStage> },
    /// Run stages in order; side-effect stages may leave the main tensor unchanged.
    Sequence(Vec<FlowStage>),
    /// Final RMSNorm before LM head.
    RmsNorm(RmsNormStage),
    /// Gather last token along sequence axis (dynamic prefill).
    GatherLastToken(GatherLastTokenStage),
    /// Causal LM head matmul.
    LmHead(LmHeadStage),
    /// Matmul against a loaded weight (`LinearStage`).
    Linear(LinearStage),
    /// Save skip connection for residual add.
    ResidualSave(ResidualSaveStage),
    /// Add saved skip connection.
    ResidualAdd(ResidualAddStage),
    /// SwiGLU feed-forward (gate/up/down).
    SwiGlu(SwiGluStage),
    /// Prefill self-attention (QKV + RoPE + GQA + causal mask).
    SelfAttnPrefill(SelfAttnPrefillStage),
    /// Gated DeltaNet scan (inputs via [`FlowState::gdn`]).
    GdnScan(GdnScanStage),
    /// Store active flow into a named stream.
    StoreStream(StoreStreamStage),
    /// Load active flow from a named stream.
    LoadStream(LoadStreamStage),
    /// Dual-stream transform (img/txt, …).
    DualStream(DualStreamStage),
    /// Tier-2 custom subgraph (see `rlx_flow::escape`).
    Custom(CustomStage),
    /// BERT-style encoder layer (fused QKV + padding-mask attention + GELU FFN).
    BertEncoderLayer(BertEncoderLayerStage),
    /// NomicBERT encoder layer (fused QKV + RoPE + padding-mask + SwiGLU).
    NomicEncoderLayer(NomicEncoderLayerStage),
    /// Qwen3 decoder layer (QK-norm + GQA + RoPE + SwiGLU).
    Qwen3Decoder(Qwen3DecoderStage),
    /// Qwen3 KV-cache decode layer (concat past K/V + QK-norm + GQA).
    Qwen3DecodeLayer(Qwen3DecodeLayerStage),
    /// ViT fused QKV self-attention with padding mask.
    VitSelfAttn(VitSelfAttnStage),
    /// DINOv2 LayerScale (gamma multiply).
    LayerScale(LayerScaleStage),
    /// NomicVision SwiGLU FFN with intermediate LayerNorm.
    VisionSwiGluFfn(VisionSwiGluFfnStage),
    /// CLS token pooling `[B, seq, H]` → `[B, H]`.
    ClsTokenPool(ClsTokenPoolStage),
    /// LayerNorm with gamma + beta.
    LayerNorm(LayerNormStage),
    /// GELU feed-forward (intermediate + output dense).
    GeluFfn(GeluFfnStage),
    /// Gather embedding table rows from a named side input.
    GatherFromInput(GatherFromInputStage),
    /// Add gather-from-side-input embedding to active hidden tensor.
    GatherAdd(GatherAddStage),
}

impl FlowStage {
    pub(crate) fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: Option<FlowValue>,
    ) -> Result<Option<FlowValue>> {
        match self {
            FlowStage::Embed(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("Embed requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::RopeTables(s) => {
                s.emit(ctx)?;
                Ok(input)
            }
            FlowStage::ZeroBeta { name, len } => {
                let id = ctx.synth_zeros(name, *len);
                ctx.state.named.insert(name.clone(), id);
                if ctx.state.zero_beta.is_none() {
                    ctx.state.zero_beta = Some(id);
                }
                Ok(input)
            }
            FlowStage::BindDecodeInputs(s) => {
                s.emit(ctx)?;
                Ok(input)
            }
            FlowStage::AttnMask(s) => {
                s.emit(ctx)?;
                Ok(input)
            }
            FlowStage::LlamaDecodeLayer(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("LlamaDecodeLayer requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::LlamaDecoder(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("LlamaDecoder requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::LlamaKvTap(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("LlamaKvTap requires input"))?;
                s.emit(ctx, input.clone())?;
                Ok(Some(input))
            }
            FlowStage::Repeat(s) => s.emit(ctx, input),
            FlowStage::Named { name, inner } => {
                let input = input.ok_or_else(|| anyhow::anyhow!("Named block requires input"))?;
                let out = inner.emit(ctx, Some(input))?;
                let value = out.expect("named inner stage produced no output");
                ctx.hir().node_mut(value.id).name = Some(name.clone());
                Ok(Some(value))
            }
            FlowStage::Sequence(stages) => {
                let mut value = input;
                for stage in stages {
                    value = stage.emit(ctx, value)?;
                }
                Ok(value)
            }
            FlowStage::RmsNorm(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("RmsNorm requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::GatherLastToken(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("GatherLastToken requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::LmHead(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("LmHead requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::Linear(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("Linear requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::ResidualSave(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("ResidualSave requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::ResidualAdd(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("ResidualAdd requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::SwiGlu(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("SwiGlu requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::SelfAttnPrefill(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("SelfAttnPrefill requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::GdnScan(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("GdnScan requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::StoreStream(s) => s.emit(ctx, input),
            FlowStage::LoadStream(s) => s.emit(ctx, input),
            FlowStage::DualStream(s) => s.emit(ctx, input),
            FlowStage::Custom(s) => s.emit(ctx, input),
            FlowStage::BertEncoderLayer(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("BertEncoderLayer requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::NomicEncoderLayer(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("NomicEncoderLayer requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::Qwen3Decoder(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("Qwen3Decoder requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::Qwen3DecodeLayer(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("Qwen3DecodeLayer requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::VitSelfAttn(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("VitSelfAttn requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::LayerScale(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("LayerScale requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::VisionSwiGluFfn(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("VisionSwiGluFfn requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::ClsTokenPool(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("ClsTokenPool requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::LayerNorm(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("LayerNorm requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::GeluFfn(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("GeluFfn requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::GatherFromInput(s) => {
                let input =
                    input.ok_or_else(|| anyhow::anyhow!("GatherFromInput requires input"))?;
                s.emit(ctx, input)
            }
            FlowStage::GatherAdd(s) => {
                let input = input.ok_or_else(|| anyhow::anyhow!("GatherAdd requires input"))?;
                s.emit(ctx, input)
            }
        }
    }
}
