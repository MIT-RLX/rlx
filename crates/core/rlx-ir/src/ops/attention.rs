// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attention builders: SDPA with custom or kernel-synthesized
//! masks (plan #53).

use crate::op::MaskKind;
use crate::{Graph, NodeId, Op, Shape};

/// Build an [`Op::Attention`] with optional score scale and logit softcap.
/// `v_head_dim = None` ⇒ V/output width equals `head_dim` (the common case).
pub fn attention_kind_op(
    num_heads: usize,
    head_dim: usize,
    v_head_dim: Option<usize>,
    mask_kind: MaskKind,
    score_scale: Option<f32>,
    attn_logit_softcap: Option<f32>,
) -> Op {
    Op::Attention {
        num_heads,
        head_dim,
        v_head_dim,
        mask_kind,
        score_scale,
        attn_logit_softcap,
    }
}

impl Graph {
    /// Scaled dot-product attention with a custom (caller-supplied) mask.
    /// Equivalent to `attention_kind(.., MaskKind::Custom, ..)`.
    pub fn attention(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        mask: NodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
    ) -> NodeId {
        self.attention_opts(q, k, v, mask, num_heads, head_dim, shape, None, None)
    }

    /// Like [`Self::attention`] with optional score scale and logit softcap.
    pub fn attention_opts(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        mask: NodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    ) -> NodeId {
        self.push(
            attention_kind_op(
                num_heads,
                head_dim,
                None,
                MaskKind::Custom,
                score_scale,
                attn_logit_softcap,
            ),
            vec![q, k, v, mask],
            shape,
            None,
        )
    }

    /// Scaled dot-product attention with a kernel-synthesized mask
    /// (`None` / `Causal` / `SlidingWindow`). Inputs are Q, K, V only —
    /// no mask tensor is allocated or read in the inner loop. Use
    /// `MaskKind::None` for a single un-padded sequence.
    pub fn attention_kind(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
    ) -> NodeId {
        self.attention_kind_opts(q, k, v, num_heads, head_dim, mask_kind, shape, None, None)
    }

    /// Like [`Self::attention_kind`] with optional score scale and logit softcap.
    pub fn attention_kind_opts(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: usize,
        head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    ) -> NodeId {
        debug_assert!(
            !matches!(mask_kind, MaskKind::Custom | MaskKind::Bias),
            "attention_kind() requires a non-tensor MaskKind; use attention() for Custom or attention_bias() for Bias"
        );
        self.push(
            attention_kind_op(
                num_heads,
                head_dim,
                None,
                mask_kind,
                score_scale,
                attn_logit_softcap,
            ),
            vec![q, k, v],
            shape,
            None,
        )
    }

    /// Scaled dot-product attention with an additive bias tensor of shape
    /// `[batch, num_heads, query_len, key_len]` added to the
    /// `QK^T · scale` scores before softmax. Lets boxRPB / per-query
    /// position biases reuse the fast `Op::Attention` kernel path.
    pub fn attention_bias(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        bias: NodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
    ) -> NodeId {
        self.attention_bias_opts(q, k, v, bias, num_heads, head_dim, shape, None, None)
    }

    /// Like [`Self::attention_bias`] with optional score scale and logit softcap
    /// (e.g. Gemma's `score_scale = 1.0` + final-attn-logit softcap, which the
    /// bias-less variant would silently drop).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_bias_opts(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        bias: NodeId,
        num_heads: usize,
        head_dim: usize,
        shape: Shape,
        score_scale: Option<f32>,
        attn_logit_softcap: Option<f32>,
    ) -> NodeId {
        self.push(
            attention_kind_op(
                num_heads,
                head_dim,
                None,
                MaskKind::Bias,
                score_scale,
                attn_logit_softcap,
            ),
            vec![q, k, v, bias],
            shape,
            None,
        )
    }

    /// Asymmetric SDPA where V/output per-head width (`v_head_dim`) differs from
    /// the Q/K score width (`head_dim`). Q/K/V only (kernel-synthesized mask).
    /// Used by MLA to skip zero-padding V up to `head_dim`. `shape` is the
    /// output `[.., num_heads * v_head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_kind_vdim(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: usize,
        head_dim: usize,
        v_head_dim: usize,
        mask_kind: MaskKind,
        shape: Shape,
    ) -> NodeId {
        debug_assert!(
            !matches!(mask_kind, MaskKind::Custom | MaskKind::Bias),
            "attention_kind_vdim() requires a non-tensor MaskKind"
        );
        self.push(
            attention_kind_op(num_heads, head_dim, Some(v_head_dim), mask_kind, None, None),
            vec![q, k, v],
            shape,
            None,
        )
    }
}
