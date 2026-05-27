// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;

use std::sync::{Arc, Mutex};

use super::BlockStage;
use super::qwen3_decoder::per_head_rms;
use super::self_attn::repeat_kv;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct Qwen3DecodeLayerSpec {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_group_size: usize,
    pub eps: f32,
    pub use_custom_mask: bool,
    pub hidden_shape: rlx_ir::Shape,
    pub batch: usize,
}

#[derive(Debug, Clone)]
pub struct Qwen3DecodeLayerStage {
    pub layer_prefix: String,
    pub spec: Qwen3DecodeLayerSpec,
    pub layer_idx: usize,
    pub kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
}

impl Qwen3DecodeLayerStage {
    pub fn layer(
        layer_idx: usize,
        spec: Qwen3DecodeLayerSpec,
        kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            spec,
            layer_idx,
            kv_out,
        }
    }
}

impl BlockStage for Qwen3DecodeLayerStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let decode = ctx
            .state
            .decode
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Qwen3DecodeLayer requires BindDecodeInputs"))?;
        let zero_beta_h = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("Qwen3DecodeLayer requires ZeroBeta"))?;
        let zero_beta_dh = ctx
            .state
            .named
            .get("zero_beta.head")
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Qwen3DecodeLayer requires zero_beta.head"))?;

        let lp = &self.layer_prefix;
        let spec = &self.spec;
        let nh = spec.num_heads;
        let nkv = spec.num_kv_heads;
        let dh = spec.head_dim;
        let batch = spec.batch;

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

        let past_k = decode.past_k[self.layer_idx];
        let past_v = decode.past_v[self.layer_idx];

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
            batch,
            1,
            nh,
            dh,
            spec.eps,
        );
        let k_normed = per_head_rms(
            &mut gb,
            k,
            k_norm_g,
            zero_beta_dh,
            batch,
            1,
            nkv,
            dh,
            spec.eps,
        );

        let q_rope = gb.rope(q_normed, decode.cos, decode.sin, dh);
        let k_rope = gb.rope(k_normed, decode.cos, decode.sin, dh);

        let new_k = gb.concat_(vec![past_k, k_rope], 1);
        let new_v = gb.concat_(vec![past_v, v], 1);
        self.kv_out.lock().expect("kv out").push(new_k);
        self.kv_out.lock().expect("kv out").push(new_v);

        let k_rep = repeat_kv(&mut gb, new_k, nkv, dh, spec.kv_group_size);
        let v_rep = repeat_kv(&mut gb, new_v, nkv, dh, spec.kv_group_size);

        let attn_shape = shape::attention_shape(gb.shape(q_rope));
        let attn = if spec.use_custom_mask {
            let mask = decode
                .mask
                .ok_or_else(|| anyhow::anyhow!("custom mask requested but not bound"))?;
            gb.attention(q_rope, k_rep, v_rep, mask, nh, dh, attn_shape)
        } else {
            gb.attention_kind(q_rope, k_rep, v_rep, nh, dh, MaskKind::Causal, attn_shape)
        };

        let attn_out = gb.mm(attn, o_w);
        let post_attn = gb.add(skip, attn_out);
        let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta_h, spec.eps);
        let gate = gb.mm(normed_post, gate_w);
        let up = gb.mm(normed_post, up_w);
        let gate_act = gb.silu(gate);
        let swiglu = gb.mul(gate_act, up);
        let ffn_out = gb.mm(swiglu, down_w);
        let h_id = gb.add(post_attn, ffn_out);

        Ok(Some(ctx.wrap(h_id, spec.hidden_shape.clone())))
    }
}
