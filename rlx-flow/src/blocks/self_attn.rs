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
pub struct SelfAttnPrefillSpec {
    pub q_key: String,
    pub k_key: String,
    pub v_key: String,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    /// Number of leading per-head dimensions that get rotary-rotated.
    /// Equals `head_dim` for plain RoPE; smaller values express
    /// partial RoPE (Gemma 4 full-attention p-RoPE with
    /// `partial_rotary_factor=0.25`, Qwen3.5 MRoPE, etc.). Backends
    /// already handle `n_rot < head_dim` via [`rlx_ir::op::Op::Rope`].
    pub n_rot: usize,
    /// Optional named RoPE table to use instead of the default
    /// (`state.rope_cos` / `state.rope_sin`). When set, the emitter
    /// looks up `state.named["{name}_cos"]` / `…_sin`. Populate via
    /// [`crate::blocks::NamedRopeTablesStage`].
    pub rope_table: Option<String>,
    /// Reuse the K projection as V (Gemma 4 `attention_k_eq_v`).
    /// When true, `v_key` is ignored and V is sourced from K
    /// **before** RoPE rotation.
    pub k_eq_v: bool,
    pub mask: MaskKind,
    pub score_scale: Option<f32>,
    pub attn_logit_softcap: Option<f32>,
}

impl SelfAttnPrefillSpec {
    pub fn hf_layer(
        prefix: impl Into<String>,
        num_heads: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Self {
        let p = prefix.into();
        Self {
            q_key: format!("{p}.self_attn.q_proj.weight"),
            k_key: format!("{p}.self_attn.k_proj.weight"),
            v_key: format!("{p}.self_attn.v_proj.weight"),
            num_heads,
            head_dim,
            num_kv_heads,
            n_rot: head_dim,
            rope_table: None,
            k_eq_v: false,
            mask: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        }
    }

    /// Builder-style override for partial RoPE. Pass `n_rot < head_dim`
    /// for layers that only rotate the leading slice.
    pub fn with_n_rot(mut self, n_rot: usize) -> Self {
        self.n_rot = n_rot;
        self
    }

    /// Switch this layer to a named RoPE table (see
    /// [`crate::blocks::NamedRopeTablesStage`]).
    pub fn with_rope_table(mut self, name: impl Into<String>) -> Self {
        self.rope_table = Some(name.into());
        self
    }

    /// Enable Gemma 4 `attention_k_eq_v`: reuse the K projection
    /// output as V (pre-RoPE), skipping the V matmul + V weight load.
    pub fn with_k_eq_v(mut self) -> Self {
        self.k_eq_v = true;
        self
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
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let spec = &self.spec;
        let (cos, sin) = resolve_rope_handles(ctx, spec.rope_table.as_deref())?;

        let q_w = ctx.load_param(&spec.q_key, true)?;
        let k_w = ctx.load_param(&spec.k_key, true)?;
        let v_w = if spec.k_eq_v {
            None
        } else {
            Some(ctx.load_param(&spec.v_key, true)?)
        };

        let mut gb = HirMut::new(ctx.hir());
        let q = gb.mm(input.id, q_w);
        let k = gb.mm(input.id, k_w);
        // V is the pre-RoPE K projection when `attention_k_eq_v` —
        // RoPE only affects positional info, which V doesn't need.
        let v = match v_w {
            Some(w) => gb.mm(input.id, w),
            None => k,
        };
        let q_rope = gb.rope_n(q, cos, sin, spec.head_dim, spec.n_rot);
        let k_rope = gb.rope_n(k, cos, sin, spec.head_dim, spec.n_rot);

        let group = spec.num_heads / spec.num_kv_heads;
        let k_rep = repeat_kv(&mut gb, k_rope, spec.num_kv_heads, spec.head_dim, group);
        let v_rep = repeat_kv(&mut gb, v, spec.num_kv_heads, spec.head_dim, group);

        let attn_shape = shape::attention_shape(gb.shape(q_rope));
        let attn = gb.attention_kind_opts(
            q_rope,
            k_rep,
            v_rep,
            spec.num_heads,
            spec.head_dim,
            spec.mask,
            attn_shape,
            spec.score_scale,
            spec.attn_logit_softcap,
        );
        Ok(Some(ctx.wrap(attn, input.shape.clone())))
    }
}

/// Resolve `(cos, sin)` HIR handles, honouring an optional named RoPE
/// table. Defaults to `state.rope_cos`/`state.rope_sin` when no name
/// is supplied — i.e. preserves the single-table contract every
/// pre-Gemma-4 model relies on.
pub(crate) fn resolve_rope_handles(
    ctx: &FlowCtx<'_>,
    name: Option<&str>,
) -> Result<(rlx_ir::HirNodeId, rlx_ir::HirNodeId)> {
    if let Some(slot) = name {
        let cos_key = format!("{slot}_cos");
        let sin_key = format!("{slot}_sin");
        let cos = ctx.state.named.get(&cos_key).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "self-attn requested RoPE table `{slot}` but `{cos_key}` is missing from state.named — \
                 emit a NamedRopeTablesStage before the layer block"
            )
        })?;
        let sin = ctx.state.named.get(&sin_key).copied().ok_or_else(|| {
            anyhow::anyhow!("self-attn requested RoPE table `{slot}` but `{sin_key}` is missing")
        })?;
        return Ok((cos, sin));
    }
    let cos = ctx
        .state
        .rope_cos
        .ok_or_else(|| anyhow::anyhow!("self-attn requires RopeTables"))?;
    let sin = ctx
        .state
        .rope_sin
        .ok_or_else(|| anyhow::anyhow!("self-attn requires RopeTables"))?;
    Ok((cos, sin))
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
