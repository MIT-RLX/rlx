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
    pub qk_norm: bool,
    pub attention_bias: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen3DecodeLayerStage {
    pub layer_prefix: String,
    pub spec: Qwen3DecodeLayerSpec,
    pub layer_idx: usize,
    pub kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    pub qk_out: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
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
            qk_out: None,
        }
    }

    pub fn layer_with_qk(
        layer_idx: usize,
        spec: Qwen3DecodeLayerSpec,
        kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
        qk_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            spec,
            layer_idx,
            kv_out,
            qk_out: Some(qk_out),
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
        let o_w = ctx.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
        let post_ln_g = ctx.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
        let gate_w = ctx.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
        let up_w = ctx.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
        let down_w = ctx.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;
        let (q_bias, k_bias, v_bias) = if spec.attention_bias {
            (
                Some(ctx.load_param(&format!("{lp}.self_attn.q_proj.bias"), false)?),
                Some(ctx.load_param(&format!("{lp}.self_attn.k_proj.bias"), false)?),
                Some(ctx.load_param(&format!("{lp}.self_attn.v_proj.bias"), false)?),
            )
        } else {
            (None, None, None)
        };
        let (q_norm_g, k_norm_g) = if spec.qk_norm {
            (
                Some(ctx.load_param(&format!("{lp}.self_attn.q_norm.weight"), false)?),
                Some(ctx.load_param(&format!("{lp}.self_attn.k_norm.weight"), false)?),
            )
        } else {
            (None, None)
        };

        let past_k = decode.past_k[self.layer_idx];
        let past_v = decode.past_v[self.layer_idx];

        let mut gb = HirMut::new(ctx.hir());
        let skip = input.id;
        let normed_in = gb.rms_norm(skip, in_ln_g, zero_beta_h, spec.eps);
        let mut q = gb.mm(normed_in, q_w);
        let mut k = gb.mm(normed_in, k_w);
        let mut v = gb.mm(normed_in, v_w);

        if let (Some(qb), Some(kb), Some(vb)) = (q_bias, k_bias, v_bias) {
            q = gb.add(q, qb);
            k = gb.add(k, kb);
            v = gb.add(v, vb);
        }

        let (q_rope_in, k_rope_in) = if let (Some(qng), Some(kng)) = (q_norm_g, k_norm_g) {
            let q_normed = per_head_rms(&mut gb, q, qng, zero_beta_dh, batch, 1, nh, dh, spec.eps);
            let k_normed = per_head_rms(&mut gb, k, kng, zero_beta_dh, batch, 1, nkv, dh, spec.eps);
            (q_normed, k_normed)
        } else {
            (q, k)
        };

        let q_rope = gb.rope(q_rope_in, decode.cos, decode.sin, dh);
        let k_rope = gb.rope(k_rope_in, decode.cos, decode.sin, dh);

        let new_k = gb.concat_(vec![past_k, k_rope], 1);
        let new_v = gb.concat_(vec![past_v, v], 1);
        self.kv_out.lock().expect("kv out").push(new_k);
        self.kv_out.lock().expect("kv out").push(new_v);

        let k_rep = repeat_kv(&mut gb, new_k, nkv, dh, spec.kv_group_size);
        let v_rep = repeat_kv(&mut gb, new_v, nkv, dh, spec.kv_group_size);
        if let Some(ref sink) = self.qk_out {
            sink.lock().expect("qwen3 decode qk out").push(q_rope);
            sink.lock().expect("qwen3 decode qk out").push(k_rep);
        }

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
