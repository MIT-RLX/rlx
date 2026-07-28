// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `attention` — Microsoft contrib fused attention ops.
//!
//! `GroupQueryAttention` (com.microsoft) is the fused decoder-attention block
//! emitted by the `transformers.js` / ONNX Runtime LM exporters (ChatterBox,
//! Phi, Llama, Qwen …). It bundles: packed-QKV split, RoPE on Q/K, KV-cache
//! append, grouped-query head repeat, scaled causal attention, and returns the
//! present K/V. We decompose it into the primitives rlx already has —
//! `narrow`/`rope`/`attention` — mirroring the native `Qwen3DecodeLayer` build.
//!
//! KV-cache note: the native import path drives the LM by *re-prefilling* the
//! growing sequence each AR step (the moss playbook), so `past_key`/`past_value`
//! are always empty and RoPE positions are the static `0..S`. The `present.*`
//! outputs are therefore the current-step K/V; they are only meaningful when the
//! caller requests them (they are pruned as dead when the import requests only
//! `logits`).

#![allow(unused_imports)]

use anyhow::{Result, bail};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{MaskKind, RopeStyle};
use rlx_ir::{DType, HirGraphExt, Op, Shape};

use super::*;

/// Lower `com.microsoft::GroupQueryAttention`.
pub(super) fn lower_group_query_attention(
    m: &mut HirMut<'_>,
    ctx: &mut LowerCtx<'_>,
    node: &BundleNode,
) -> Result<bool> {
    let num_heads = node
        .attrs
        .get("num_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as usize;
    let mut kv_num_heads = node
        .attrs
        .get("kv_num_heads")
        .and_then(|v| v.as_i64())
        .unwrap_or(num_heads as i64) as usize;
    if kv_num_heads == 0 {
        kv_num_heads = num_heads;
    }
    if num_heads == 0 {
        ctx.unsupported("GroupQueryAttention(num_heads=0)");
        return Ok(false);
    }
    let do_rotary = node
        .attrs
        .get("do_rotary")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let interleaved = node
        .attrs
        .get("rotary_interleaved")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        != 0;
    let local_window = node
        .attrs
        .get("local_window_size")
        .and_then(|v| v.as_i64())
        .unwrap_or(-1);
    let softcap = node
        .attrs
        .get("softcap")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let scale_attr = node
        .attrs
        .get("scale")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32);

    // Q/K/V — either packed into input[0] (key/value empty) or supplied
    // separately in inputs[1]/[2].
    let key_name = node.inputs.get(1).map(String::as_str).unwrap_or("");
    let val_name = node.inputs.get(2).map(String::as_str).unwrap_or("");
    let packed = key_name.is_empty() || val_name.is_empty();

    let (q, k, v, head_dim) = if packed {
        let qkv = ctx.tensor(&node.inputs[0])?;
        let last = m.shape(qkv).rank() - 1;
        let width = match m.shape(qkv).dim(last) {
            rlx_ir::Dim::Static(n) => n,
            _ => {
                ctx.unsupported("GroupQueryAttention(dynamic qkv width)");
                return Ok(false);
            }
        };
        let denom = num_heads + 2 * kv_num_heads;
        if denom == 0 || width % denom != 0 {
            ctx.unsupported("GroupQueryAttention(packed width mismatch)");
            return Ok(false);
        }
        let head_dim = width / denom;
        let q_w = num_heads * head_dim;
        let kv_w = kv_num_heads * head_dim;
        let q = m.narrow_(qkv, last, 0, q_w);
        let k = m.narrow_(qkv, last, q_w, kv_w);
        let v = m.narrow_(qkv, last, q_w + kv_w, kv_w);
        (q, k, v, head_dim)
    } else {
        let q = ctx.tensor(&node.inputs[0])?;
        let k = ctx.tensor(key_name)?;
        let v = ctx.tensor(val_name)?;
        let last = m.shape(q).rank() - 1;
        let q_w = match m.shape(q).dim(last) {
            rlx_ir::Dim::Static(n) => n,
            _ => {
                ctx.unsupported("GroupQueryAttention(dynamic q width)");
                return Ok(false);
            }
        };
        if q_w % num_heads != 0 {
            ctx.unsupported("GroupQueryAttention(q width mismatch)");
            return Ok(false);
        }
        (q, k, v, q_w / num_heads)
    };

    // RoPE on Q and K (V is not rotated). cos/sin caches are inputs[7]/[8].
    let (q, k) = if do_rotary {
        let cos_name = node.inputs.get(7).map(String::as_str).unwrap_or("");
        let sin_name = node.inputs.get(8).map(String::as_str).unwrap_or("");
        if cos_name.is_empty() || sin_name.is_empty() {
            ctx.unsupported("GroupQueryAttention(do_rotary without cos/sin cache)");
            return Ok(false);
        }
        let cos = ctx.tensor(cos_name)?;
        let sin = ctx.tensor(sin_name)?;
        let style = if interleaved {
            RopeStyle::GptJ
        } else {
            RopeStyle::NeoX
        };
        let q_r = m.rope_styled(q, cos, sin, head_dim, style);
        let k_r = m.rope_styled(k, cos, sin, head_dim, style);
        (q_r, k_r)
    } else {
        (q, k)
    };

    // present.key / present.value (outputs 1, 2): the current-step K/V. Bind the
    // packed [B, S, kv_heads·head_dim] tensors directly — the native re-prefill
    // path does not consume them, so they stay as dead outputs unless requested.
    if let Some(pk) = node.outputs.get(1).filter(|n| !n.is_empty()) {
        ctx.env.insert(pk.clone(), k);
    }
    if let Some(pv) = node.outputs.get(2).filter(|n| !n.is_empty()) {
        ctx.env.insert(pv.clone(), v);
    }

    // Grouped-query head repeat so K/V match the Q head count (no-op when
    // kv_num_heads == num_heads).
    let group = num_heads / kv_num_heads.max(1);
    let k_rep = repeat_kv_last(m, k, kv_num_heads, head_dim, group);
    let v_rep = repeat_kv_last(m, v, kv_num_heads, head_dim, group);

    // Scaled causal (or sliding-window) attention over the packed layout.
    let mask_kind = if local_window > 0 {
        MaskKind::SlidingWindow(local_window as usize)
    } else {
        MaskKind::Causal
    };
    let score_scale = scale_attr;
    let softcap_opt = if softcap > 0.0 { Some(softcap) } else { None };
    let attn_shape = m.shape(q).clone();
    let attn = m.attention_kind_opts(
        q,
        k_rep,
        v_rep,
        num_heads,
        head_dim,
        mask_kind,
        attn_shape,
        score_scale,
        softcap_opt,
    );
    ctx.env.insert(node.outputs[0].clone(), attn);
    Ok(true)
}

/// Repeat each KV head `group` times along the packed last axis
/// (`[B, S, kv·D] → [B, S, kv·group·D]`). Identity when `group == 1`.
fn repeat_kv_last(
    m: &mut HirMut<'_>,
    x: HirNodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> HirNodeId {
    if group <= 1 {
        return x;
    }
    let last = m.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = m.narrow_(x, last, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    m.concat_(pieces, last)
}
