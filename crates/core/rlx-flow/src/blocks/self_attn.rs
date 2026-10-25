// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    /// `crate::blocks::NamedRopeTablesStage`.
    pub rope_table: Option<String>,
    /// Reuse the K projection as V (Gemma 4 `attention_k_eq_v`).
    /// When true, `v_key` is ignored and V is sourced from K
    /// **before** RoPE rotation.
    pub k_eq_v: bool,
    pub mask: MaskKind,
    pub score_scale: Option<f32>,
    pub attn_logit_softcap: Option<f32>,
    /// RoPE pairing flavor. Defaults to NeoX (HF rotate-half); set
    /// [`rlx_ir::op::RopeStyle::GptJ`] for GGUF-Llama-style interleaved pairs so
    /// prefill matches the (already style-aware) decode path.
    pub rope_style: rlx_ir::op::RopeStyle,
    /// Per-head RMS norms on Q/K (and optionally V) before RoPE — Gemma 3/4.
    /// `None` for families without them.
    pub head_norm: Option<HeadNormSpec>,
    /// Name of a declared flow input holding an additive attention bias
    /// (`[batch, heads?, seq, seq]`), used with [`MaskKind::Bias`] — e.g. a
    /// media-span mask that lets image tokens attend bidirectionally.
    pub bias_input: Option<String>,
}

/// Per-head RMS norms applied to the Q/K projections (and, for Gemma 4, a
/// weightless one on V) before RoPE.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadNormSpec {
    pub q_key: String,
    pub k_key: String,
    /// Gemma's `x / rms(x) * (1 + w)` gain; plain `w` when false.
    pub unit_offset: bool,
    /// Weightless RMS norm on V (Gemma 4).
    pub v_norm: bool,
    pub eps: f32,
}

impl HeadNormSpec {
    /// HF Gemma names under `prefix` (`model.layers.N`), `(1 + w)` gains.
    pub fn gemma(prefix: &str, v_norm: bool, eps: f32) -> Self {
        Self {
            q_key: format!("{prefix}.self_attn.q_norm.weight"),
            k_key: format!("{prefix}.self_attn.k_norm.weight"),
            unit_offset: true,
            v_norm,
            eps,
        }
    }
}

/// [`HeadNormSpec`] with its gains loaded into the graph.
pub(crate) struct LoadedHeadNorms {
    q: rlx_ir::HirNodeId,
    k: rlx_ir::HirNodeId,
    q_ones: Option<rlx_ir::HirNodeId>,
    k_ones: Option<rlx_ir::HirNodeId>,
    v_ones: Option<rlx_ir::HirNodeId>,
    beta: rlx_ir::HirNodeId,
    eps: f32,
    head_dim: usize,
}

impl LoadedHeadNorms {
    pub(crate) fn load(
        ctx: &mut FlowCtx<'_>,
        spec: &HeadNormSpec,
        head_dim: usize,
    ) -> Result<Self> {
        let q = ctx.load_param(&spec.q_key, false)?;
        let k = ctx.load_param(&spec.k_key, false)?;
        let ones = |ctx: &mut FlowCtx<'_>, name: String| {
            synth_once(ctx, &name, vec![1.0; head_dim], head_dim)
        };
        let (q_ones, k_ones) = if spec.unit_offset {
            (
                Some(ones(ctx, format!("{}.ones", spec.q_key))),
                Some(ones(ctx, format!("{}.ones", spec.k_key))),
            )
        } else {
            (None, None)
        };
        let v_ones = spec
            .v_norm
            .then(|| ones(ctx, format!("head_norm.v.ones.{head_dim}")));
        let beta = synth_once(
            ctx,
            &format!("head_norm.zero_beta.{head_dim}"),
            vec![0.0; head_dim],
            head_dim,
        );
        Ok(Self {
            q,
            k,
            q_ones,
            k_ones,
            v_ones,
            beta,
            eps: spec.eps,
            head_dim,
        })
    }

    pub(crate) fn q(&self, gb: &mut HirMut, x: rlx_ir::HirNodeId) -> rlx_ir::HirNodeId {
        let g = gain(gb, self.q, self.q_ones);
        per_head_rms_any(gb, x, g, self.beta, self.head_dim, self.eps)
    }

    pub(crate) fn k(&self, gb: &mut HirMut, x: rlx_ir::HirNodeId) -> rlx_ir::HirNodeId {
        let g = gain(gb, self.k, self.k_ones);
        per_head_rms_any(gb, x, g, self.beta, self.head_dim, self.eps)
    }

    /// V unchanged unless the spec asks for the weightless V norm.
    pub(crate) fn v(&self, gb: &mut HirMut, x: rlx_ir::HirNodeId) -> rlx_ir::HirNodeId {
        match self.v_ones {
            Some(ones) => per_head_rms_any(gb, x, ones, self.beta, self.head_dim, self.eps),
            None => x,
        }
    }
}

fn gain(
    gb: &mut HirMut,
    w: rlx_ir::HirNodeId,
    ones: Option<rlx_ir::HirNodeId>,
) -> rlx_ir::HirNodeId {
    match ones {
        Some(o) => gb.add(o, w),
        None => w,
    }
}

/// A synthesized `[len]` param registered once per name: several stages of a
/// layer (KV tap, attention) ask for the same constant, and two `Param` nodes
/// with one name would be two independent values to the runtime.
pub(crate) fn synth_once(
    ctx: &mut FlowCtx<'_>,
    name: &str,
    data: Vec<f32>,
    len: usize,
) -> rlx_ir::HirNodeId {
    let cache_key = format!("synth:{name}");
    if let Some(&id) = ctx.state.loaded_params.get(&cache_key) {
        return id;
    }
    let id = ctx.synth_param(name, data, rlx_ir::Shape::new(&[len], rlx_ir::DType::F32));
    ctx.state.loaded_params.insert(cache_key, id);
    id
}

/// RMS-normalize each `head_dim`-wide head of `x` (`[…, heads·head_dim]`),
/// whatever its leading shape; one leading dim may be dynamic.
fn per_head_rms_any(
    gb: &mut HirMut,
    x: rlx_ir::HirNodeId,
    gamma: rlx_ir::HirNodeId,
    beta: rlx_ir::HirNodeId,
    head_dim: usize,
    eps: f32,
) -> rlx_ir::HirNodeId {
    let sh = gb.shape(x);
    let rank = sh.rank();
    let lead: Vec<i64> = (0..rank - 1)
        .map(|i| {
            let d = sh.dim(i);
            if d.is_static() {
                d.unwrap_static() as i64
            } else {
                -1
            }
        })
        .collect();
    let width = sh.dim(rank - 1).unwrap_static();
    let heads = (width / head_dim) as i64;
    let split: Vec<i64> = lead
        .iter()
        .copied()
        .chain([heads, head_dim as i64])
        .collect();
    let merged: Vec<i64> = lead.iter().copied().chain([width as i64]).collect();
    let r = gb.reshape_(x, split);
    let n = gb.rms_norm(r, gamma, beta, eps);
    gb.reshape_(n, merged)
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
            rope_style: rlx_ir::op::RopeStyle::NeoX,
            head_norm: None,
            bias_input: None,
        }
    }

    /// Builder-style override for the RoPE pairing flavor (GGUF Llama → GptJ).
    pub fn with_rope_style(mut self, style: rlx_ir::op::RopeStyle) -> Self {
        self.rope_style = style;
        self
    }

    /// Builder-style override for partial RoPE. Pass `n_rot < head_dim`
    /// for layers that only rotate the leading slice.
    pub fn with_n_rot(mut self, n_rot: usize) -> Self {
        self.n_rot = n_rot;
        self
    }

    /// Switch this layer to a named RoPE table (see
    /// `crate::blocks::NamedRopeTablesStage`).
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

    /// Apply per-head Q/K (and optionally V) RMS norms before RoPE.
    pub fn with_head_norm(mut self, norm: HeadNormSpec) -> Self {
        self.head_norm = Some(norm);
        self
    }

    /// Mask with the additive bias in flow input `name` ([`MaskKind::Bias`]).
    pub fn with_bias_input(mut self, name: impl Into<String>) -> Self {
        self.mask = MaskKind::Bias;
        self.bias_input = Some(name.into());
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

        let q_w = ctx.load_matrix(&spec.q_key, true)?;
        let k_w = ctx.load_matrix(&spec.k_key, true)?;
        let v_w = if spec.k_eq_v {
            None
        } else {
            Some(ctx.load_matrix(&spec.v_key, true)?)
        };

        let norms = match &spec.head_norm {
            Some(n) => Some(LoadedHeadNorms::load(ctx, n, spec.head_dim)?),
            None => None,
        };
        let bias =
            match &spec.bias_input {
                Some(name) => Some(ctx.state.inputs.get(name).map(|(id, _)| *id).ok_or_else(
                    || anyhow::anyhow!("self-attn bias input `{name}` not declared"),
                )?),
                None => None,
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
        let (q, k, v) = match &norms {
            Some(n) => (n.q(&mut gb, q), n.k(&mut gb, k), n.v(&mut gb, v)),
            None => (q, k, v),
        };
        let q_rope = gb.rope_n_styled(q, cos, sin, spec.head_dim, spec.n_rot, spec.rope_style);
        let k_rope = gb.rope_n_styled(k, cos, sin, spec.head_dim, spec.n_rot, spec.rope_style);

        let group = spec.num_heads / spec.num_kv_heads;
        let k_rep = repeat_kv(&mut gb, k_rope, spec.num_kv_heads, spec.head_dim, group);
        let v_rep = repeat_kv(&mut gb, v, spec.num_kv_heads, spec.head_dim, group);

        let attn_shape = shape::attention_shape(gb.shape(q_rope));
        let attn = match bias {
            Some(bias) => gb.inner().mir(
                rlx_ir::ops::attention::attention_kind_op(
                    spec.num_heads,
                    spec.head_dim,
                    None,
                    MaskKind::Bias,
                    spec.score_scale,
                    spec.attn_logit_softcap,
                ),
                vec![q_rope, k_rep, v_rep, bias],
                attn_shape,
            ),
            None => gb.attention_kind_opts(
                q_rope,
                k_rep,
                v_rep,
                spec.num_heads,
                spec.head_dim,
                spec.mask,
                attn_shape,
                spec.score_scale,
                spec.attn_logit_softcap,
            ),
        };
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
    // GQA KV-head replication as a single broadcast, NOT narrow+concat. Repeating
    // each of `num_kv_heads` contiguous head-blocks `group` times is exactly
    // reshape([…, nkv, 1, dh]) → expand([…, nkv, group, dh]) → reshape([…, nkv*group*dh]),
    // preserving the interleaved [kv0,kv0,…,kv1,kv1,…] order the old code built.
    // The old form emitted `num_kv_heads` Narrow ops plus a many-input Concat that
    // lowers to ONE kernel launch PER piece (num_kv_heads*group of them) — ~48
    // launches/layer of pure data movement that dominate a launch-bound decode.
    // Expand is one native kernel on every backend; the reshapes are views.
    let sh = g.shape(x);
    let rank = sh.rank();
    let last_ax = rank - 1;
    let lead: Vec<i64> = (0..rank - 1)
        .map(|i| sh.dim(i))
        .collect::<Vec<_>>()
        .iter()
        .map(|d| {
            // Fall back to the old path if a leading dim is dynamic (can't build a
            // concrete reshape target); decode/prefill shapes are static.
            if d.is_static() {
                d.unwrap_static() as i64
            } else {
                -1
            }
        })
        .collect();
    if lead.iter().any(|&d| d < 0) {
        let mut pieces = Vec::with_capacity(num_kv_heads * group);
        for h in 0..num_kv_heads {
            let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
            for _ in 0..group {
                pieces.push(slice);
            }
        }
        return g.concat_(pieces, last_ax);
    }
    let (nkv, dh, grp) = (num_kv_heads as i64, head_dim as i64, group as i64);
    let split: Vec<i64> = lead.iter().copied().chain([nkv, 1, dh]).collect();
    let expanded: Vec<i64> = lead.iter().copied().chain([nkv, grp, dh]).collect();
    let merged: Vec<i64> = lead.iter().copied().chain([nkv * grp * dh]).collect();
    let r = g.reshape_(x, split);
    let e = g.expand_(r, expanded);
    g.reshape_(e, merged)
}
