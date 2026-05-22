// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;
use rlx_ir::HirGraphExt;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct SelfAttnPrefillSpec {
    pub q_key: String,
    pub k_key: String,
    pub v_key: String,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub mask: MaskKind,
}

impl SelfAttnPrefillSpec {
    pub fn hf_layer(prefix: impl Into<String>, num_heads: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        let p = prefix.into();
        Self {
            q_key: format!("{p}.self_attn.q_proj.weight"),
            k_key: format!("{p}.self_attn.k_proj.weight"),
            v_key: format!("{p}.self_attn.v_proj.weight"),
            num_heads,
            head_dim,
            num_kv_heads,
            mask: MaskKind::Causal,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SelfAttnPrefillStage {
    pub spec: SelfAttnPrefillSpec,
}

impl SelfAttnPrefillStage {
    pub fn new(spec: SelfAttnPrefillSpec) -> Self {
        Self { spec }
    }
}

impl BlockStage for SelfAttnPrefillStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let cos = ctx
            .state
            .rope_cos
            .ok_or_else(|| anyhow::anyhow!("SelfAttnPrefill requires RopeTables"))?;
        let sin = ctx
            .state
            .rope_sin
            .ok_or_else(|| anyhow::anyhow!("SelfAttnPrefill requires RopeTables"))?;

        let spec = &self.spec;
        let q_w = ctx.load_param(&spec.q_key, true)?;
        let k_w = ctx.load_param(&spec.k_key, true)?;
        let v_w = ctx.load_param(&spec.v_key, true)?;

        let mut gb = HirMut::new(ctx.hir());
        let q = gb.mm(input.id, q_w);
        let k = gb.mm(input.id, k_w);
        let v = gb.mm(input.id, v_w);
        let q_rope = gb.rope(q, cos, sin, spec.head_dim);
        let k_rope = gb.rope(k, cos, sin, spec.head_dim);

        let group = spec.num_heads / spec.num_kv_heads;
        let k_rep = repeat_kv(&mut gb, k_rope, spec.num_kv_heads, spec.head_dim, group);
        let v_rep = repeat_kv(&mut gb, v, spec.num_kv_heads, spec.head_dim, group);

        let attn_shape = shape::attention_shape(gb.shape(q_rope));
        let attn = gb.attention_kind(
            q_rope,
            k_rep,
            v_rep,
            spec.num_heads,
            spec.head_dim,
            spec.mask,
            attn_shape,
        );
        Ok(Some(ctx.wrap(attn, input.shape.clone())))
    }
}

pub(crate) fn repeat_kv(
    g: &mut HirMut,
    x: rlx_ir::HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> rlx_ir::HirNodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}
