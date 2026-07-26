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

//! Multi-head Latent Attention (MLA) — the DeepSeek-V2/V3 attention block.
//!
//! MLA shrinks the KV footprint by attending through a low-rank *latent* rather
//! than full per-head K/V. Each token is down-projected to a compressed KV latent
//! `c_KV` (plus a single shared RoPE key `k_R`), and per-head keys/values are
//! reconstructed by up-projection. Queries take the same down/up path through
//! their own latent `c_Q`. RoPE is *decoupled*: only a narrow `qk_rope_head_dim`
//! slice is rotated; the wider `qk_nope_head_dim` slice is not.
//!
//! This is the prefill (materialized) form, the twin of
//! [`super::self_attn::SelfAttnPrefillStage`]: it emits ordinary primitives
//! (matmul / RMSNorm / narrow / concat / pad / RoPE / attention), so it fuses and
//! runs on every backend with no dedicated kernel. It returns the raw attention
//! output `[B, S, num_heads · v_head_dim]`; the surrounding layer applies `o_proj`.
//!
//! ## The head-dim asymmetry
//!
//! MLA's score dim (`qk_nope_head_dim + qk_rope_head_dim`) differs from its value
//! dim (`v_head_dim`), but [`rlx_ir::op::Op::Attention`] carries a *single*
//! `head_dim`. We bridge this by zero-padding V up to the QK head-dim: the padded
//! columns contribute nothing to the scores (which only see Q·K) and produce zero
//! output columns, which we narrow back off. Correct on every backend, and it
//! keeps the fused/flash SDPA path. A dedicated asymmetric-head attention op is
//! the Tier-2 perf follow-up.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::{MaskKind, PadMode, RopeStyle};
use rlx_ir::shape;

use super::BlockStage;
use super::self_attn::{repeat_kv, resolve_rope_handles};
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Load a batch of weights into `let` bindings, cutting the projection-key
/// boilerplate. Each row is `binding = "suffix" @ transpose`; the key is
/// `"{prefix}.{suffix}"`. `@ true` transposes (matmul weights, stored
/// `[out, in]`), `@ false` leaves it (norm gammas). Expands to one
/// `ctx.load_param(..)?` per row, so it must run where `?` is valid.
macro_rules! load_params {
    ($ctx:expr, $prefix:expr; $( $name:ident = $suffix:literal @ $transpose:literal );+ $(;)? ) => {
        $( let $name = $ctx.load_param(&format!("{}.{}", $prefix, $suffix), $transpose)?; )+
    };
}

/// Resolved query-projection weight handles: the DeepSeek-V2/V3 LoRA pair or
/// the LoRA-less V2-Lite single projection.
enum QProj {
    Lora {
        q_a_w: rlx_ir::HirNodeId,
        q_a_ln: rlx_ir::HirNodeId,
        q_b_w: rlx_ir::HirNodeId,
        zb_q: rlx_ir::HirNodeId,
    },
    Direct(rlx_ir::HirNodeId),
}

#[derive(Debug, Clone)]
pub struct MlaAttnPrefillSpec {
    /// Weight-key prefix, e.g. `"model.layers.0"`. Suffixes follow the
    /// DeepSeek-V2/V3 HF submodule names (`self_attn.q_a_proj.weight`, …).
    pub prefix: String,
    pub num_heads: usize,
    /// Query down-projection rank (`q_lora_rank`). `Some` selects the
    /// DeepSeek-V2/V3 LoRA query path (`q_a_proj` → RMSNorm → `q_b_proj`);
    /// `None` selects the LoRA-less variant (DeepSeek-V2-Lite), a single
    /// `q_proj` straight from the hidden state.
    pub q_lora_rank: Option<usize>,
    /// KV down-projection rank (`kv_lora_rank`) — the size of the cached latent.
    pub kv_lora_rank: usize,
    /// Per-head non-rotated score dim.
    pub qk_nope_head_dim: usize,
    /// Per-head decoupled-RoPE score dim (the only slice RoPE touches).
    pub qk_rope_head_dim: usize,
    /// Per-head value dim.
    pub v_head_dim: usize,
    /// Epsilon shared by both latent RMSNorms (`q_a_layernorm` / `kv_a_layernorm`).
    pub eps: f32,
    /// Attention mask — `Causal` for standard decoder MLA.
    pub mask: MaskKind,
    /// Softmax scale; `None` ⇒ `1/√(qk_nope_head_dim + qk_rope_head_dim)`.
    /// Set for YaRN `mscale` variants.
    pub score_scale: Option<f32>,
    /// RoPE pairing flavor. DeepSeek HF checkpoints are [`RopeStyle::NeoX`].
    pub rope_style: RopeStyle,
    /// Optional named RoPE table (see [`super::self_attn::resolve_rope_handles`]).
    pub rope_table: Option<String>,
}

impl MlaAttnPrefillSpec {
    /// Standard DeepSeek-V2/V3 layer with the canonical `{prefix}.self_attn.*`
    /// key layout.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_layer(
        prefix: impl Into<String>,
        num_heads: usize,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        eps: f32,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            num_heads,
            q_lora_rank: Some(q_lora_rank),
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            eps,
            mask: MaskKind::Causal,
            score_scale: None,
            rope_style: RopeStyle::NeoX,
            rope_table: None,
        }
    }

    /// DeepSeek-V2-Lite layer: no query LoRA. The query comes from a single
    /// `self_attn.q_proj.weight` instead of the `q_a_proj`/`q_b_proj` pair.
    #[allow(clippy::too_many_arguments)]
    pub fn deepseek_lite_layer(
        prefix: impl Into<String>,
        num_heads: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        eps: f32,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            num_heads,
            q_lora_rank: None,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            eps,
            mask: MaskKind::Causal,
            score_scale: None,
            rope_style: RopeStyle::NeoX,
            rope_table: None,
        }
    }

    /// Per-head score dim (`qk_nope_head_dim + qk_rope_head_dim`) — the padded
    /// head-dim the underlying [`rlx_ir::op::Op::Attention`] sees.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Builder-style override for the RoPE pairing flavor.
    pub fn with_rope_style(mut self, style: RopeStyle) -> Self {
        self.rope_style = style;
        self
    }

    /// Builder-style override for the softmax scale (YaRN `mscale`).
    pub fn with_score_scale(mut self, scale: f32) -> Self {
        self.score_scale = Some(scale);
        self
    }

    /// Switch this layer to a named RoPE table.
    pub fn with_rope_table(mut self, name: impl Into<String>) -> Self {
        self.rope_table = Some(name.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct MlaAttnPrefillStage {
    pub spec: MlaAttnPrefillSpec,
}

impl MlaAttnPrefillStage {
    pub fn new(spec: MlaAttnPrefillSpec) -> Self {
        Self { spec }
    }
}

impl BlockStage for MlaAttnPrefillStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let spec = &self.spec;
        let nh = spec.num_heads;
        let nope = spec.qk_nope_head_dim;
        let rope = spec.qk_rope_head_dim;
        let qk = nope + rope; // per-head score dim
        let vh = spec.v_head_dim;
        let kv_lora = spec.kv_lora_rank;

        // MLA's per-head reshapes need the batch/seq split explicit; the block
        // is prefill, so both are static.
        let b = input.shape.dim(0).unwrap_static();
        let s = input.shape.dim(1).unwrap_static();
        let (bi, si) = (b as i64, s as i64);

        let (cos, sin) = resolve_rope_handles(ctx, spec.rope_table.as_deref())?;

        let lp = &spec.prefix;
        load_params!(ctx, lp;
            kv_a_w  = "self_attn.kv_a_proj_with_mqa.weight" @ true;
            kv_a_ln = "self_attn.kv_a_layernorm.weight"     @ false;
            kv_b_w  = "self_attn.kv_b_proj.weight"          @ true;
        );
        // Query weights: the V2/V3 LoRA pair (down → RMSNorm → up) or the
        // LoRA-less V2-Lite `q_proj`. The zero-beta slot must match the
        // normalized latent width; the flow's shared `zero_beta` is hidden-sized.
        let q_proj = match spec.q_lora_rank {
            Some(r) => QProj::Lora {
                q_a_w: ctx.load_param(&format!("{lp}.self_attn.q_a_proj.weight"), true)?,
                q_a_ln: ctx.load_param(&format!("{lp}.self_attn.q_a_layernorm.weight"), false)?,
                q_b_w: ctx.load_param(&format!("{lp}.self_attn.q_b_proj.weight"), true)?,
                zb_q: zero_beta_for(ctx, r),
            },
            None => QProj::Direct(ctx.load_param(&format!("{lp}.self_attn.q_proj.weight"), true)?),
        };
        let zb_kv = zero_beta_for(ctx, kv_lora);

        let mut gb = HirMut::new(ctx.hir());

        // ── Query: LoRA down→norm→up, or a direct projection (V2-Lite) ──
        let q = match q_proj {
            QProj::Lora {
                q_a_w,
                q_a_ln,
                q_b_w,
                zb_q,
            } => {
                let c_q = gb.mm(input.id, q_a_w);
                let c_q = gb.rms_norm(c_q, q_a_ln, zb_q, spec.eps);
                gb.mm(c_q, q_b_w)
            }
            QProj::Direct(q_w) => gb.mm(input.id, q_w),
        }; // [B, S, nh·qk]
        let q4 = gb.reshape_(q, vec![bi, si, nh as i64, qk as i64]);
        let q_nope = gb.narrow_(q4, 3, 0, nope); // [B, S, nh, nope]
        let q_rope = gb.narrow_(q4, 3, nope, rope); // [B, S, nh, rope]
        // RoPE runs on the flattened [B, S, nh·rope] layout (its seq-indexing
        // contract), then folds back to per-head rank-4.
        let q_rope = gb.reshape_(q_rope, vec![bi, si, (nh * rope) as i64]);
        let q_rope = gb.rope_n_styled(q_rope, cos, sin, rope, rope, spec.rope_style);
        let q_rope = gb.reshape_(q_rope, vec![bi, si, nh as i64, rope as i64]);
        let q_full = gb.concat_(vec![q_nope, q_rope], 3); // [B, S, nh, qk]
        let q_full = gb.reshape_(q_full, vec![bi, si, (nh * qk) as i64]);

        // ── KV: joint down-projection (compressed latent ++ shared RoPE key) ──
        let kv_a = gb.mm(input.id, kv_a_w); // [B, S, kv_lora + rope]
        let c_kv = gb.narrow_(kv_a, 2, 0, kv_lora); // [B, S, kv_lora]
        let k_rope = gb.narrow_(kv_a, 2, kv_lora, rope); // [B, S, rope] — shared
        let c_kv = gb.rms_norm(c_kv, kv_a_ln, zb_kv, spec.eps);
        let kv_b = gb.mm(c_kv, kv_b_w); // [B, S, nh·(nope + vh)]
        let kv_b4 = gb.reshape_(kv_b, vec![bi, si, nh as i64, (nope + vh) as i64]);
        let k_nope = gb.narrow_(kv_b4, 3, 0, nope); // [B, S, nh, nope]
        let v = gb.narrow_(kv_b4, 3, nope, vh); // [B, S, nh, vh]

        // Shared RoPE key: rotate once, then broadcast across all query heads.
        let k_rope = gb.rope_n_styled(k_rope, cos, sin, rope, rope, spec.rope_style);
        let k_rope = repeat_kv(&mut gb, k_rope, 1, rope, nh); // [B, S, nh·rope]
        let k_rope = gb.reshape_(k_rope, vec![bi, si, nh as i64, rope as i64]);
        let k_full = gb.concat_(vec![k_nope, k_rope], 3); // [B, S, nh, qk]
        let k_full = gb.reshape_(k_full, vec![bi, si, (nh * qk) as i64]);

        // V rides the symmetric Op::Attention by zero-padding its head-dim vh→qk;
        // the extra columns yield zero outputs we narrow back off after attention.
        let v_pad = gb.pad_(
            v,
            vec![[0, 0], [0, 0], [0, 0], [0, qk - vh]],
            PadMode::Constant(0.0),
        ); // [B, S, nh, qk]
        let v_pad = gb.reshape_(v_pad, vec![bi, si, (nh * qk) as i64]);

        // ── Scaled dot-product attention over the padded (qk-wide) heads ──
        let attn_shape = shape::attention_shape(gb.shape(q_full));
        let attn_raw = gb.attention_kind_opts(
            q_full,
            k_full,
            v_pad,
            nh,
            qk,
            spec.mask,
            attn_shape,
            spec.score_scale,
            None,
        ); // [B, S, nh·qk]
        let attn = gb.reshape_(attn_raw, vec![bi, si, nh as i64, qk as i64]);
        let attn = gb.narrow_(attn, 3, 0, vh); // drop padded value columns
        let out = gb.reshape_(attn, vec![bi, si, (nh * vh) as i64]);

        let out_shape = rlx_ir::Shape::new(&[b, s, nh * vh], rlx_ir::DType::F32);
        Ok(Some(ctx.wrap(out, out_shape)))
    }
}

/// A cached rank-1 zero vector of length `len`, for an RMSNorm beta slot. Deduped
/// across layers by length in `state.named` so N MLA layers share one node per
/// distinct latent rank.
fn zero_beta_for(ctx: &mut FlowCtx<'_>, len: usize) -> rlx_ir::HirNodeId {
    let key = format!("__mla_zero_beta.{len}");
    if let Some(&id) = ctx.state.named.get(&key) {
        return id;
    }
    let id = ctx.synth_zeros(&key, len);
    ctx.state.named.insert(key, id);
    id
}
