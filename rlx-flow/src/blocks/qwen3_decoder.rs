// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;

use std::sync::{Arc, Mutex};

use super::BlockStage;
use super::self_attn::repeat_kv;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct Qwen3DecoderSpec {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub hidden_shape: rlx_ir::Shape,
    pub batch: usize,
    pub seq: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen3DecoderStage {
    pub layer_prefix: String,
    pub spec: Qwen3DecoderSpec,
    pub kv_sink: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
}

impl Qwen3DecoderStage {
    pub fn layer(layer_idx: usize, spec: Qwen3DecoderSpec) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            spec,
            kv_sink: None,
        }
    }

    pub fn layer_with_kv(
        layer_idx: usize,
        spec: Qwen3DecoderSpec,
        kv_sink: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            spec,
            kv_sink: Some(kv_sink),
        }
    }
}

impl BlockStage for Qwen3DecoderStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let lp = &self.layer_prefix;
        let spec = &self.spec;
        let nh = spec.num_heads;
        let nkv = spec.num_kv_heads;
        let dh = spec.head_dim;
        let group = nh / nkv;

        let zero_beta_h = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("Qwen3Decoder requires ZeroBeta"))?;
        let zero_beta_dh = ctx
            .state
            .named
            .get("zero_beta.head")
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Qwen3Decoder requires zero_beta.head"))?;
        let cos = ctx
            .state
            .rope_cos
            .ok_or_else(|| anyhow::anyhow!("Qwen3Decoder requires RopeTables"))?;
        let sin = ctx
            .state
            .rope_sin
            .ok_or_else(|| anyhow::anyhow!("Qwen3Decoder requires RopeTables"))?;

        let in_ln_g = ctx.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
        let q_w = ctx.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?;
        let k_w = ctx.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
        let v_w = ctx.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;
        let q_norm_g = ctx.load_param(&format!("{lp}.self_attn.q_norm.weight"), false)?;
        let k_norm_g = ctx.load_param(&format!("{lp}.self_attn.k_norm.weight"), false)?;
        let o_w = ctx.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
        let post_ln_g = ctx.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
        let gate_w = ctx.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
        let up_w = ctx.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
        let down_w = ctx.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;

        let mut gb = HirMut::new(ctx.hir());
        let skip = input.id;

        let normed_in = gb.rms_norm(skip, in_ln_g, zero_beta_h, spec.eps);
        let q = gb.mm(normed_in, q_w);
        let k = gb.mm(normed_in, k_w);
        let v = gb.mm(normed_in, v_w);

        let q_normed = per_head_rms(
            &mut gb,
            q,
            q_norm_g,
            zero_beta_dh,
            spec.batch,
            spec.seq,
            nh,
            dh,
            spec.eps,
        );
        let k_normed = per_head_rms(
            &mut gb,
            k,
            k_norm_g,
            zero_beta_dh,
            spec.batch,
            spec.seq,
            nkv,
            dh,
            spec.eps,
        );

        let q_rope = gb.rope(q_normed, cos, sin, dh);
        let k_rope = gb.rope(k_normed, cos, sin, dh);
        if let Some(ref sink) = self.kv_sink {
            sink.lock().expect("qwen3 kv sink").push(k_rope);
            sink.lock().expect("qwen3 kv sink").push(v);
        }
        let k_rep = repeat_kv(&mut gb, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut gb, v, nkv, dh, group);

        let attn_shape = shape::attention_shape(gb.shape(q_rope));
        let attn = gb.attention_kind(q_rope, k_rep, v_rep, nh, dh, MaskKind::Causal, attn_shape);
        let attn_out = gb.mm(attn, o_w);
        let post_attn = gb.add(skip, attn_out);
        let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta_h, spec.eps);

        let gate = gb.mm(normed_post, gate_w);
        let up = gb.mm(normed_post, up_w);
        let gate_act = gb.silu(gate);
        let swiglu = gb.mul(gate_act, up);
        let ffn_out = gb.mm(swiglu, down_w);
        let out = gb.add(post_attn, ffn_out);

        Ok(Some(ctx.wrap(out, spec.hidden_shape.clone())))
    }
}

pub(crate) fn per_head_rms(
    gb: &mut HirMut,
    x: rlx_ir::HirNodeId,
    gamma: rlx_ir::HirNodeId,
    beta: rlx_ir::HirNodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> rlx_ir::HirNodeId {
    let flat = (batch * seq * heads) as i64;
    let dh = head_dim as i64;
    let r = gb.reshape_(x, vec![flat, dh]);
    let n = gb.rms_norm(r, gamma, beta, eps);
    gb.reshape_(n, vec![batch as i64, seq as i64, (heads * head_dim) as i64])
}
