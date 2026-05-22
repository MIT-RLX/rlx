// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::op::MaskKind;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;
use crate::weight::WeightSource;

#[derive(Debug, Clone)]
pub struct LlamaDecoderStage {
    pub layer_prefix: String,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub eps: f32,
    pub mask: MaskKind,
    pub hidden_shape: rlx_ir::Shape,
}

impl LlamaDecoderStage {
    pub fn layer(layer_idx: usize, spec: LlamaDecoderSpec) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            num_heads: spec.num_heads,
            head_dim: spec.head_dim,
            num_kv_heads: spec.num_kv_heads,
            eps: spec.eps,
            mask: spec.mask,
            hidden_shape: spec.hidden_shape,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlamaDecoderSpec {
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub eps: f32,
    pub mask: MaskKind,
    pub hidden_shape: rlx_ir::Shape,
}

impl BlockStage for LlamaDecoderStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let lp = &self.layer_prefix;
        let zero_beta = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("LlamaDecoder requires ZeroBeta stage"))?;
        let cos = ctx
            .state
            .rope_cos
            .ok_or_else(|| anyhow::anyhow!("LlamaDecoder requires RopeTables stage"))?;
        let sin = ctx
            .state
            .rope_sin
            .ok_or_else(|| anyhow::anyhow!("LlamaDecoder requires RopeTables stage"))?;

        let in_ln_g = ctx.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
        let q_w = ctx.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?;
        let k_w = ctx.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
        let v_w = ctx.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;
        let o_w = ctx.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
        let post_ln_g = ctx.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
        let gate_w = ctx.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
        let up_w = ctx.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
        let down_w = ctx.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;

        let id = ctx.hir().llama_decoder_block(
            input.id,
            in_ln_g,
            zero_beta,
            q_w,
            k_w,
            v_w,
            o_w,
            post_ln_g,
            zero_beta,
            gate_w,
            up_w,
            down_w,
            cos,
            sin,
            None,
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
            self.eps,
            self.mask,
            self.hidden_shape.clone(),
        );

        Ok(Some(ctx.wrap(id, self.hidden_shape.clone())))
    }
}
