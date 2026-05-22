// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::HirGraphExt;

use super::attn_mask::ATTN_MASK;
use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct VitSelfAttnSpec {
    pub qkv_weight: String,
    pub qkv_bias: String,
    pub out_weight: String,
    pub out_bias: String,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct VitSelfAttnStage {
    pub spec: VitSelfAttnSpec,
}

impl VitSelfAttnStage {
    pub fn new(spec: VitSelfAttnSpec) -> Self {
        Self { spec }
    }

    pub fn dinov2(layer_prefix: impl Into<String>, hidden_size: usize, num_heads: usize) -> Self {
        let p = layer_prefix.into();
        Self::new(VitSelfAttnSpec {
            qkv_weight: format!("{p}.attn.qkv.weight"),
            qkv_bias: format!("{p}.attn.qkv.bias"),
            out_weight: format!("{p}.attn.proj.weight"),
            out_bias: format!("{p}.attn.proj.bias"),
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads,
        })
    }

    pub fn nomic_vision(layer_prefix: impl Into<String>, hidden_size: usize, num_heads: usize) -> Self {
        let p = layer_prefix.into();
        Self::new(VitSelfAttnSpec {
            qkv_weight: format!("{p}.attn.Wqkv.weight"),
            qkv_bias: format!("{p}.attn.Wqkv.bias"),
            out_weight: format!("{p}.attn.out_proj.weight"),
            out_bias: format!("{p}.attn.out_proj.bias"),
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads,
        })
    }
}

impl BlockStage for VitSelfAttnStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let mask = ctx
            .state
            .named
            .get(ATTN_MASK)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("VitSelfAttn requires AttnMaskStage"))?;

        let spec = &self.spec;
        let h = spec.hidden_size;
        let nh = spec.num_heads;
        let dh = spec.head_dim;

        let qkv_w = ctx.load_param(&spec.qkv_weight, true)?;
        let qkv_b = ctx.load_param(&spec.qkv_bias, false)?;
        let out_w = ctx.load_param(&spec.out_weight, true)?;
        let out_b = ctx.load_param(&spec.out_bias, false)?;

        let mut gb = HirMut::new(ctx.hir());
        let qkv_mm = gb.mm(input.id, qkv_w);
        let qkv = gb.add(qkv_mm, qkv_b);

        let last_ax = gb.shape(qkv).rank() - 1;
        let q = gb.narrow_(qkv, last_ax, 0, h);
        let k = gb.narrow_(qkv, last_ax, h, h);
        let v = gb.narrow_(qkv, last_ax, 2 * h, h);

        let attn = gb.attention_(q, k, v, mask, nh, dh);
        let proj_mm = gb.mm(attn, out_w);
        let out = gb.add(proj_mm, out_b);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
