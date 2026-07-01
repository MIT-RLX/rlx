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

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;
#[derive(Debug, Clone)]
pub struct LlamaDecodeLayerSpec {
    pub num_heads: usize,
    pub head_dim: usize,
    /// Leading per-head dims that get rotary-rotated (`head_dim` unless
    /// partial RoPE — Phi-3 / long-context variants).
    pub n_rot: usize,
    pub num_kv_heads: usize,
    pub kv_group_size: usize,
    pub eps: f32,
    pub use_custom_mask: bool,
    pub hidden_shape: rlx_ir::Shape,
    /// RoPE pairing flavor. GGUF Llama weights need [`rlx_ir::RopeStyle::GptJ`];
    /// HF-safetensors checkpoints use [`rlx_ir::RopeStyle::NeoX`] (default).
    pub rope_style: rlx_ir::RopeStyle,
}

#[derive(Debug, Clone)]
pub struct LlamaDecodeLayerStage {
    pub layer_prefix: String,
    pub spec: LlamaDecodeLayerSpec,
    pub layer_idx: usize,
    pub kv_out: std::sync::Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>,
    /// Optional EAGLE3-style tap for the pre-attention-norm layer
    /// input. Mirrors the field on
    /// [`crate::blocks::GemmaDecodeLayerStage`]; see that doc for
    /// semantics and push-order guarantees.
    pub aux_in_out: Option<std::sync::Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>>,
}

impl LlamaDecodeLayerStage {
    pub fn layer(
        layer_idx: usize,
        spec: LlamaDecodeLayerSpec,
        kv_out: std::sync::Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            spec,
            layer_idx,
            kv_out,
            aux_in_out: None,
        }
    }

    pub fn with_aux_input_tap(
        mut self,
        sink: std::sync::Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        self.aux_in_out = Some(sink);
        self
    }
}

impl BlockStage for LlamaDecodeLayerStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        if let Some(sink) = self.aux_in_out.as_ref() {
            sink.lock().expect("aux in out").push(input.id);
        }

        let decode = ctx
            .state
            .decode
            .clone()
            .ok_or_else(|| anyhow::anyhow!("LlamaDecodeLayer requires BindDecodeInputs"))?;
        let zero_beta = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("LlamaDecodeLayer requires ZeroBeta"))?;

        let lp = &self.layer_prefix;
        let spec = &self.spec;
        let in_ln_g = ctx.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
        let q_w = ctx.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?;
        let k_w = ctx.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
        let v_w = ctx.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;
        let o_w = ctx.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
        let post_ln_g = ctx.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
        let gate_w = ctx.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
        let up_w = ctx.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
        let down_w = ctx.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;

        let past_k = decode.past_k.get(self.layer_idx);
        let past_v = decode.past_v.get(self.layer_idx);

        let mut gb = HirMut::new(ctx.hir());
        let normed_in = gb.rms_norm(input.id, in_ln_g, zero_beta, spec.eps);
        let q = gb.mm(normed_in, q_w);
        let k = gb.mm(normed_in, k_w);
        let v = gb.mm(normed_in, v_w);

        let q_rope = gb.rope_n_styled(
            q,
            decode.cos,
            decode.sin,
            spec.head_dim,
            spec.n_rot,
            spec.rope_style,
        );
        let k_rope = gb.rope_n_styled(
            k,
            decode.cos,
            decode.sin,
            spec.head_dim,
            spec.n_rot,
            spec.rope_style,
        );

        let (new_k, new_v) = match (past_k, past_v) {
            (Some(past_k), Some(past_v)) => (
                gb.concat_(vec![*past_k, k_rope], 1),
                gb.concat_(vec![*past_v, v], 1),
            ),
            _ => (k_rope, v),
        };
        self.kv_out.lock().expect("kv out").push(new_k);
        self.kv_out.lock().expect("kv out").push(new_v);

        let k_rep = super::self_attn::repeat_kv(
            &mut gb,
            new_k,
            spec.num_kv_heads,
            spec.head_dim,
            spec.kv_group_size,
        );
        let v_rep = super::self_attn::repeat_kv(
            &mut gb,
            new_v,
            spec.num_kv_heads,
            spec.head_dim,
            spec.kv_group_size,
        );

        let attn_shape = shape::attention_shape(gb.shape(q_rope));
        let attn = if spec.use_custom_mask {
            let mask = decode
                .mask
                .ok_or_else(|| anyhow::anyhow!("custom mask requested but not bound"))?;
            gb.attention(
                q_rope,
                k_rep,
                v_rep,
                mask,
                spec.num_heads,
                spec.head_dim,
                attn_shape,
            )
        } else {
            gb.attention_kind(
                q_rope,
                k_rep,
                v_rep,
                spec.num_heads,
                spec.head_dim,
                MaskKind::Causal,
                attn_shape,
            )
        };

        let attn_out = gb.mm(attn, o_w);
        let post_attn = gb.add(input.id, attn_out);
        let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta, spec.eps);
        let gate = gb.mm(normed_post, gate_w);
        let up = gb.mm(normed_post, up_w);
        let gate_act = gb.silu(gate);
        let swiglu = gb.mul(gate_act, up);
        let ffn_out = gb.mm(swiglu, down_w);
        let h_id = gb.add(post_attn, ffn_out);

        Ok(Some(ctx.wrap(h_id, spec.hidden_shape.clone())))
    }
}
