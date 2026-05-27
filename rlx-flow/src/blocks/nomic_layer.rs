// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct NomicEncoderLayerSpec {
    pub layer_prefix: String,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub attention_mask_input: String,
}

impl NomicEncoderLayerSpec {
    pub fn hf(
        layer_prefix: impl Into<String>,
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Self {
        Self {
            layer_prefix: layer_prefix.into(),
            hidden_size,
            num_heads,
            head_dim,
            eps,
            attention_mask_input: "attention_mask".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NomicEncoderLayerStage {
    pub spec: NomicEncoderLayerSpec,
}

impl NomicEncoderLayerStage {
    pub fn new(spec: NomicEncoderLayerSpec) -> Self {
        Self { spec }
    }
}

impl BlockStage for NomicEncoderLayerStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let spec = &self.spec;
        let h = spec.hidden_size;
        let nh = spec.num_heads;
        let dh = spec.head_dim;
        let lp = &spec.layer_prefix;

        let cos = ctx
            .state
            .rope_cos
            .ok_or_else(|| anyhow::anyhow!("NomicEncoderLayer requires RopeTables"))?;
        let sin = ctx
            .state
            .rope_sin
            .ok_or_else(|| anyhow::anyhow!("NomicEncoderLayer requires RopeTables"))?;
        let mask_id = ctx
            .state
            .inputs
            .get(&spec.attention_mask_input)
            .map(|(id, _)| *id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "NomicEncoderLayer requires input `{}`",
                    spec.attention_mask_input
                )
            })?;

        let qkv_w = ctx.load_param(&format!("{lp}.attn.Wqkv.weight"), true)?;
        let out_w = ctx.load_param(&format!("{lp}.attn.out_proj.weight"), true)?;
        let ln1_g = ctx.load_param(&format!("{lp}.norm1.weight"), false)?;
        let ln1_b = ctx.load_param(&format!("{lp}.norm1.bias"), false)?;
        let fc11_w = ctx.load_param(&format!("{lp}.mlp.fc11.weight"), true)?;
        let fc12_w = ctx.load_param(&format!("{lp}.mlp.fc12.weight"), true)?;
        let fc2_w = ctx.load_param(&format!("{lp}.mlp.fc2.weight"), true)?;
        let ln2_g = ctx.load_param(&format!("{lp}.norm2.weight"), false)?;
        let ln2_b = ctx.load_param(&format!("{lp}.norm2.bias"), false)?;

        let mut gb = HirMut::new(ctx.hir());
        let skip = input.id;

        let qkv = gb.mm(skip, qkv_w);
        let last_ax = gb.shape(qkv).rank() - 1;
        let q = gb.narrow_(qkv, last_ax, 0, h);
        let k = gb.narrow_(qkv, last_ax, h, h);
        let v = gb.narrow_(qkv, last_ax, 2 * h, h);
        let q_rope = gb.rope(q, cos, sin, dh);
        let k_rope = gb.rope(k, cos, sin, dh);
        let attn = gb.attention_(q_rope, k_rope, v, mask_id, nh, dh);

        let attn_out = gb.mm(attn, out_w);
        let res1 = gb.add(attn_out, skip);
        let normed1 = gb.ln(res1, ln1_g, ln1_b, spec.eps);

        let up = gb.mm(normed1, fc11_w);
        let gate_mm = gb.mm(normed1, fc12_w);
        let gate = gb.silu(gate_mm);
        let swiglu = gb.mul(up, gate);
        let ffn_out = gb.mm(swiglu, fc2_w);

        let res2 = gb.add(ffn_out, normed1);
        let out = gb.ln(res2, ln2_g, ln2_b, spec.eps);

        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
