// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;
use rlx_ir::{DType, Shape};

use std::sync::{Arc, Mutex};

use super::{BlockStage, GemmaLayerStyle};
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct GemmaDecodeLayerSpec {
    pub style: GemmaLayerStyle,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub kv_group_size: usize,
    pub eps: f32,
    pub use_custom_mask: bool,
    pub hidden_shape: rlx_ir::Shape,
    pub mask: MaskKind,
    pub score_scale: Option<f32>,
    pub attn_logit_softcap: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct GemmaDecodeLayerStage {
    pub layer_prefix: String,
    pub spec: GemmaDecodeLayerSpec,
    pub layer_idx: usize,
    pub kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
}

impl GemmaDecodeLayerStage {
    pub fn layer(
        layer_idx: usize,
        spec: GemmaDecodeLayerSpec,
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

impl BlockStage for GemmaDecodeLayerStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let decode = ctx
            .state
            .decode
            .clone()
            .ok_or_else(|| anyhow::anyhow!("GemmaDecodeLayer requires BindDecodeInputs"))?;
        let zero_beta = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("GemmaDecodeLayer requires ZeroBeta"))?;

        let lp = &self.layer_prefix;
        let spec = &self.spec;
        let style = spec.style;

        let in_ln_w = ctx.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
        let in_ln_len = norm_len(ctx, in_ln_w)?;
        let in_ln_ones = ctx.synth_param(
            &format!("{lp}.input_layernorm.ones"),
            vec![1.0f32; in_ln_len],
            Shape::new(&[in_ln_len], DType::F32),
        );

        let pre_ffn_key = if matches!(
            style,
            GemmaLayerStyle::Gemma2 | GemmaLayerStyle::Gemma3 | GemmaLayerStyle::Gemma4
        ) {
            format!("{lp}.pre_feedforward_layernorm")
        } else {
            format!("{lp}.post_attention_layernorm")
        };
        let pre_ffn_w = ctx.load_param(&format!("{pre_ffn_key}.weight"), false)?;
        let pre_ffn_len = norm_len(ctx, pre_ffn_w)?;
        let pre_ffn_ones = ctx.synth_param(
            &format!("{pre_ffn_key}.ones"),
            vec![1.0f32; pre_ffn_len],
            Shape::new(&[pre_ffn_len], DType::F32),
        );

        let post_ffn = if matches!(
            style,
            GemmaLayerStyle::Gemma2 | GemmaLayerStyle::Gemma3 | GemmaLayerStyle::Gemma4
        ) {
            let post_key = format!("{lp}.post_feedforward_layernorm");
            let w = ctx.load_param(&format!("{post_key}.weight"), false)?;
            let len = norm_len(ctx, w)?;
            let ones = ctx.synth_param(
                &format!("{post_key}.ones"),
                vec![1.0f32; len],
                Shape::new(&[len], DType::F32),
            );
            Some((w, ones))
        } else {
            None
        };

        let q_w = ctx.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?;
        let k_w = ctx.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
        let v_w = ctx.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;
        let o_w = ctx.load_param(&format!("{lp}.self_attn.o_proj.weight"), true)?;
        let gate_w = ctx.load_param(&format!("{lp}.mlp.gate_proj.weight"), true)?;
        let up_w = ctx.load_param(&format!("{lp}.mlp.up_proj.weight"), true)?;
        let down_w = ctx.load_param(&format!("{lp}.mlp.down_proj.weight"), true)?;

        let past_k = decode.past_k[self.layer_idx];
        let past_v = decode.past_v[self.layer_idx];

        let mut gb = HirMut::new(ctx.hir());
        let in_gamma = gb.add(in_ln_ones, in_ln_w);
        let normed_in = gb.rms_norm(input.id, in_gamma, zero_beta, spec.eps);
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
            gb.attention_(q_rope, k_rep, v_rep, mask, spec.num_heads, spec.head_dim)
        } else {
            gb.attention_kind_opts(
                q_rope,
                k_rep,
                v_rep,
                spec.num_heads,
                spec.head_dim,
                spec.mask,
                attn_shape,
                spec.score_scale,
                spec.attn_logit_softcap,
            )
        };

        let attn_out = gb.mm(attn, o_w);
        let post_attn = gb.add(input.id, attn_out);

        let pre_gamma = gb.add(pre_ffn_ones, pre_ffn_w);
        let mut h = gb.rms_norm(post_attn, pre_gamma, zero_beta, spec.eps);
        let gate = gb.mm(h, gate_w);
        let up = gb.mm(h, up_w);
        let gate_act = gb.gelu_approx(gate);
        h = gb.mul(gate_act, up);
        h = gb.mm(h, down_w);

        if let Some((post_w, post_ones)) = post_ffn {
            let post_gamma = gb.add(post_ones, post_w);
            h = gb.rms_norm(h, post_gamma, zero_beta, spec.eps);
        }

        let out_id = gb.add(post_attn, h);
        Ok(Some(ctx.wrap(out_id, spec.hidden_shape.clone())))
    }
}

fn norm_len(ctx: &FlowCtx<'_>, weight: rlx_ir::HirNodeId) -> Result<usize> {
    match ctx.node_shape(weight)?.dims().last() {
        Some(rlx_ir::shape::Dim::Static(n)) => Ok(*n),
        _ => Ok(0),
    }
}
