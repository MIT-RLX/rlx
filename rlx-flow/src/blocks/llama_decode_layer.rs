// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

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
    pub num_kv_heads: usize,
    pub kv_group_size: usize,
    pub eps: f32,
    pub use_custom_mask: bool,
    pub hidden_shape: rlx_ir::Shape,
}

#[derive(Debug, Clone)]
pub struct LlamaDecodeLayerStage {
    pub layer_prefix: String,
    pub spec: LlamaDecodeLayerSpec,
    pub layer_idx: usize,
    pub kv_out: std::sync::Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>,
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
        }
    }
}

impl BlockStage for LlamaDecodeLayerStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
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

        let past_k = decode.past_k[self.layer_idx];
        let past_v = decode.past_v[self.layer_idx];

        let mut gb = HirMut::new(ctx.hir());
        let normed_in = gb.rms_norm(input.id, in_ln_g, zero_beta, spec.eps);
        let q = gb.mm(normed_in, q_w);
        let k = gb.mm(normed_in, k_w);
        let v = gb.mm(normed_in, v_w);

        let q_rope = gb.rope(q, decode.cos, decode.sin, spec.head_dim);
        let k_rope = gb.rope(k, decode.cos, decode.sin, spec.head_dim);

        let new_k = gb.concat_(vec![past_k, k_rope], 1);
        let new_v = gb.concat_(vec![past_v, v], 1);
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
