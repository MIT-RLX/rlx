// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// (license header truncated — see workspace root.)

//! Qwen3 graph builder — prefill-only (no KV cache yet).
//!
//! Emits the IR for a single forward pass over `[batch, seq]` token
//! ids and returns the final hidden states after `model.norm`. The
//! lm_head projection, sampling, and incremental decode land in the
//! next slice (see plan in repo notes).
//!
//! Qwen3 specifics handled here:
//!   - **GQA** via graph-level KV head repetition (narrow + concat).
//!     The existing `Op::Attention` only carries `num_heads`, so K/V
//!     are widened from `num_kv_heads * head_dim` to
//!     `num_heads * head_dim` before the attention call. A real GQA
//!     op replaces this in Phase 2 alongside the KV-cache kernels.
//!   - **QK-norm**: per-head RMSNorm on Q and K (before RoPE), via a
//!     reshape to `[B*S*heads, head_dim]` so the existing RMSNorm
//!     kernel sees its canonical 2D layout.
//!   - **Causal attention** via `MaskKind::Causal` — no mask tensor.
//!   - **Sliding window** is parsed from config but Phase 1 treats
//!     every layer as full causal; the per-layer SWA wiring lands
//!     when backend kernels confirm `MaskKind::SlidingWindow` support.

use crate::qwen3::config::Qwen3Config;
use crate::weight_loader::WeightLoader;
use anyhow::{Result, anyhow};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;
use rlx_ir::*;
use std::collections::HashMap;

/// Build a Qwen3 causal-LM IR graph.
///
/// When `with_lm_head` is `false`, the output is the post-norm hidden
/// state `[batch, seq, hidden_size]` (useful for embedding-style
/// pooling or for inserting a custom head). When `true`, the output is
/// logits `[batch, seq, vocab_size]`.
///
/// When `with_kv_outputs` is `true`, each layer's post-RoPE K and post-projection
/// V tensors (both shape `[batch, seq, kv_proj_dim]`, pre-GQA-repeat) are appended
/// to the graph outputs in order `[main, k_0, v_0, k_1, v_1, ..., k_{N-1}, v_{N-1}]`.
/// Used to seed the KV cache for decode mode.
///
/// Tied embeddings (`cfg.tie_word_embeddings = true`) are handled by
/// reusing the `model.embed_tokens.weight` parameter node via a
/// graph-level transpose — no data duplication, one extra `Transpose`
/// op per model.
pub fn build_qwen3_graph_sized(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_qwen3_graph_sized_impl(
        cfg,
        weights,
        batch,
        seq,
        with_lm_head,
        with_kv_outputs,
        /*last_logits_only*/ false,
    )
}

/// Build a Qwen3 prefill graph that projects only the last sequence
/// position through `lm_head`.
///
/// The output shape is `[batch, 1, vocab_size]`. This is the generation
/// hot path: prompt prefill needs every layer's hidden state to build
/// attention/KV state, but sampling only consumes the final logits row.
pub fn build_qwen3_graph_sized_last_logits(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_kv_outputs: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_qwen3_graph_sized_impl(
        cfg,
        weights,
        batch,
        seq,
        /*with_lm_head*/ true,
        with_kv_outputs,
        /*last_logits_only*/ true,
    )
}

fn build_qwen3_graph_sized_impl(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    with_kv_outputs: bool,
    last_logits_only: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(anyhow!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        ));
    }
    if cfg.attention_bias {
        return Err(anyhow!(
            "attention_bias=true not yet wired (Qwen3 dense ships with attention_bias=false)"
        ));
    }

    let mut g = Graph::new("qwen3");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;

    // Synthetic zero-beta params for every RMSNorm shape we touch.
    // The Op::RmsNorm kernel takes (x, gamma, beta); Qwen3 weights
    // have only gamma, so we synthesise a zero beta once per shape.
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "qwen3.zero_beta.hidden", h);
    let zero_beta_headdim = synth_zero(&mut g, &mut params, "qwen3.zero_beta.head_dim", dh);

    // ── RoPE cos/sin cache ─────────────────────────────────────────
    let half = dh / 2;
    let mut cos_data = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin_data = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    let cos_id = g.param(
        "rope.cos",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );
    params.insert("rope.cos".into(), cos_data);
    let sin_id = g.param(
        "rope.sin",
        Shape::new(&[cfg.max_position_embeddings, half], f),
    );
    params.insert("rope.sin".into(), sin_data);

    // ── Inputs ─────────────────────────────────────────────────────
    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));

    // ── Token embedding ────────────────────────────────────────────
    let embed_w = load_p(
        &mut g,
        &mut params,
        weights,
        "model.embed_tokens.weight",
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    // KV-cache output collector. When `with_kv_outputs` is true, each layer
    // appends (k_rope, v) — both [B, S, kv_proj_dim], pre-GQA-repeat — so
    // the host can seed the cache for decode mode.
    let mut kv_outputs: Vec<NodeId> = Vec::with_capacity(2 * cfg.num_hidden_layers);

    // ── Decoder layers ─────────────────────────────────────────────
    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer_idx}");

        // input_layernorm (RMS)
        let in_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            false,
        )?;
        let normed_in = g.rms_norm(h_id, in_ln_g, zero_beta_hidden, eps);

        // Q/K/V projections (HF stores as [out, in]; rlx wants [in, out])
        let q_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
            true,
        )?;
        let k_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
            true,
        )?;
        let v_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
            true,
        )?;
        let q = g.mm(normed_in, q_w); // [B, S, q_dim]
        let k = g.mm(normed_in, k_w); // [B, S, kv_dim]
        let v = g.mm(normed_in, v_w); // [B, S, kv_dim]

        // QK-norm: per-head RMSNorm on Q and K, gamma shape [head_dim].
        let q_norm_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.q_norm.weight"),
            false,
        )?;
        let k_norm_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.k_norm.weight"),
            false,
        )?;
        let q_normed = per_head_rms(
            &mut g,
            q,
            q_norm_g,
            zero_beta_headdim,
            batch,
            seq,
            nh,
            dh,
            eps,
        );
        let k_normed = per_head_rms(
            &mut g,
            k,
            k_norm_g,
            zero_beta_headdim,
            batch,
            seq,
            nkv,
            dh,
            eps,
        );

        // RoPE on Q and K (post-QK-norm).
        let q_rope = g.rope(q_normed, cos_id, sin_id, dh);
        let k_rope = g.rope(k_normed, cos_id, sin_id, dh);

        if with_kv_outputs {
            kv_outputs.push(k_rope);
            kv_outputs.push(v);
        }

        // GQA: repeat each KV head `group` times so K/V match Q's head count.
        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);

        // Causal SDPA.
        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = g.attention_kind(q_rope, k_rep, v_rep, nh, dh, MaskKind::Causal, attn_shape);

        // o_proj
        let o_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
            true,
        )?;
        let attn_out = g.mm(attn, o_w);

        // Residual.
        let post_attn = g.add(h_id, attn_out);

        // post_attention_layernorm (RMS)
        let post_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            false,
        )?;
        let normed_post = g.rms_norm(post_attn, post_ln_g, zero_beta_hidden, eps);

        // SwiGLU MLP: down(silu(gate(x)) * up(x))
        let gate_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.gate_proj.weight"),
            true,
        )?;
        let up_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.up_proj.weight"),
            true,
        )?;
        let down_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.down_proj.weight"),
            true,
        )?;
        let gate = g.mm(normed_post, gate_w);
        let up = g.mm(normed_post, up_w);
        let gate_act = g.silu(gate);
        let swiglu = g.mul(gate_act, up);
        let ffn_out = g.mm(swiglu, down_w);

        // Residual.
        h_id = g.add(post_attn, ffn_out);
    }

    // ── Final norm ─────────────────────────────────────────────────
    // Narrow before the norm when only last-token logits are wanted —
    // skips a [seq-1, n_embd] rms_norm in addition to the
    // [seq-1, vocab] matmul. Mirrors the llama.cpp #23198 idea of
    // running per-token-discardable work only on rows we'll emit.
    let final_ln_g = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
    let h_for_norm = if with_lm_head && last_logits_only {
        g.narrow_(h_id, 1, seq - 1, 1)
    } else {
        h_id
    };
    let hidden = g.rms_norm(h_for_norm, final_ln_g, zero_beta_hidden, eps);

    // ── Optional lm_head ───────────────────────────────────────────
    let out = if with_lm_head {
        let head_input = hidden;
        // Tied: pre-transpose `embed_w` (shape [vocab, hidden]) once at
        // build time into a distinct param of shape [hidden, vocab].
        // The earlier scheme materialized this transpose every forward
        // pass via `g.transpose_(embed_w, [1,0])` — a 600MB scalar
        // strided copy that dominated the LM head step on CPU. The
        // cost is one extra parameter (memory) for one fewer hot-loop
        // op (compute + bandwidth).
        // Untied: load lm_head.weight directly with the safetensors →
        // rlx transpose convention.
        let lm_head_w = if cfg.tie_word_embeddings {
            let embed = params
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
            let vocab = cfg.vocab_size;
            let hidden_size = cfg.hidden_size;
            let mut transposed = vec![0f32; embed.len()];
            for v in 0..vocab {
                for hi in 0..hidden_size {
                    transposed[hi * vocab + v] = embed[v * hidden_size + hi];
                }
            }
            let name = "qwen3.lm_head.tied_t";
            let id = g.param(name, Shape::new(&[hidden_size, vocab], DType::F32));
            params.insert(name.to_string(), transposed);
            id
        } else {
            load_p(&mut g, &mut params, weights, "lm_head.weight", true)?
        };
        // F16 LM head path (opt-in via RLX_QWEN3_F16_LM_HEAD=1): emit
        // cast(input)→cast(weight)→matmul→cast(out) to push the
        // dominant prefill matmul onto Apple AMX's f16 throughput.
        // Measured net result on the qwen3 matrix is mixed — the
        // extra cast nodes add MPSGraph dispatch overhead that beats
        // the AMX gain on the smallest cells (B≤2, L≤32). Pre-casting
        // the weight at param-upload time would flip this (avoid the
        // per-call weight cast) but needs the typed-param plumbing
        // that's still TODO. Keep opt-in so callers running big
        // batches can take the win.
        let use_f16 = std::env::var("RLX_QWEN3_F16_LM_HEAD").is_ok();
        if use_f16 {
            let h_f16 = g.cast(head_input, DType::F16);
            let w_f16 = g.cast(lm_head_w, DType::F16);
            let logits_f16 = g.mm(h_f16, w_f16);
            g.cast(logits_f16, DType::F32)
        } else {
            g.mm(head_input, lm_head_w)
        }
    } else {
        hidden
    };

    let mut outputs = Vec::with_capacity(1 + kv_outputs.len());
    outputs.push(out);
    outputs.extend(kv_outputs);
    g.set_outputs(outputs);
    Ok((g, params))
}

/// Per-head RMSNorm via reshape to `[B*S*heads, head_dim]`, norm with
/// `gamma=[head_dim]`, then reshape back to `[B, S, heads*head_dim]`.
/// Keeps the kernel on its canonical 2D contract.
#[allow(clippy::too_many_arguments)]
fn per_head_rms(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> NodeId {
    let flat = (batch * seq * heads) as i64;
    let dh = head_dim as i64;
    let r = g.reshape_(x, vec![flat, dh]);
    let n = g.rms_norm(r, gamma, beta, eps);
    g.reshape_(n, vec![batch as i64, seq as i64, (heads * head_dim) as i64])
}

/// GQA repeat: widen `[B, S, num_kv_heads * head_dim]` to
/// `[B, S, num_kv_heads * group * head_dim]` by emitting each KV head
/// `group` times in sequence. Uses narrow + concat; cheap in op count
/// (one narrow per KV head, one concat per layer per K/V) and avoids
/// touching `Op::Attention`. Replaced by a true GQA op in Phase 2.
fn repeat_kv(
    g: &mut Graph,
    x: NodeId,
    num_kv_heads: usize,
    head_dim: usize,
    group: usize,
) -> NodeId {
    if group == 1 {
        return x;
    }
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces: Vec<NodeId> = Vec::with_capacity(num_kv_heads * group);
    for h in 0..num_kv_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..group {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

/// Load a weight by key and register it as a `Param` node. When
/// `transpose` is set, the safetensors `[out, in]` layout is swapped
/// to rlx's `[in, out]` matmul convention.
fn load_p(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut dyn WeightLoader,
    key: &str,
    transpose: bool,
) -> Result<NodeId> {
    let (data, shape) = if transpose {
        weights.take_transposed(key)?
    } else {
        weights.take(key)?
    };
    let ir_shape = Shape::new(&shape, DType::F32);
    let id = g.param(key, ir_shape);
    params.insert(key.to_string(), data);
    Ok(id)
}

/// Register a zero-valued param of the given length. Used to supply
/// the `beta` argument to RMSNorm calls (Qwen3 has no bias term but
/// the IR op signature requires one).
fn synth_zero(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    len: usize,
) -> NodeId {
    let id = g.param(name, Shape::new(&[len], DType::F32));
    params.insert(name.to_string(), vec![0f32; len]);
    id
}

/// Build a Qwen3 decode-mode IR graph for a single new token given
/// a cached past of `past_seq` tokens. Inputs are:
///
///   - `input_ids` shape `[batch, 1]`
///   - `rope_cos` shape `[1, head_dim/2]` — host pre-narrows the full
///     cos table at the new token's absolute position (= `past_seq`).
///   - `rope_sin` shape `[1, head_dim/2]` — likewise.
///   - For each layer `i` in `0..num_hidden_layers`:
///     - `past_k_{i}` shape `[batch, past_seq, kv_proj_dim]`
///     - `past_v_{i}` shape `[batch, past_seq, kv_proj_dim]`
///
/// Outputs in order:
///
///   - `logits` shape `[batch, 1, vocab_size]`
///   - For each layer `i`: `new_k_{i}`, `new_v_{i}` — both
///     `[batch, past_seq + 1, kv_proj_dim]` — the cache to feed back
///     in on the next decode step.
///
/// The IR's `Op::Attention` with `MaskKind::Causal` correctly handles
/// `Lq=1` vs `Lk=past_seq+1` after the kernel fix in
/// `rlx-cpu/src/executor.rs` (Q's absolute position = `past_seq`, so
/// all `Lk` positions ≤ past_seq are attended; the upper-triangular
/// fill becomes a no-op).
pub fn build_qwen3_decode_graph_sized(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_qwen3_decode_graph_sized_ext(cfg, weights, batch, past_seq, false)
}

/// Extended decode-mode builder.
///
/// `use_custom_mask`:
///   - `false` (default): use `MaskKind::Causal`, no mask input. Behavior
///     identical to [`build_qwen3_decode_graph_sized`]. Graph shape is
///     specialized to the exact `past_seq`.
///   - `true`: take a `mask` input of shape `[batch, past_seq + 1]` and
///     apply it via `MaskKind::Custom`. Lets a bucketed compile cache
///     pad `past_k`/`past_v` up to the bucket's upper bound and mask
///     the padded positions so they don't contribute to attention. The
///     graph is then reusable for any actual past length ≤ `past_seq`.
pub fn build_qwen3_decode_graph_sized_ext(
    cfg: &Qwen3Config,
    weights: &mut dyn WeightLoader,
    batch: usize,
    past_seq: usize,
    use_custom_mask: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(anyhow!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        ));
    }
    if cfg.attention_bias {
        return Err(anyhow!("attention_bias=true not yet wired"));
    }

    let mut g = Graph::new("qwen3_decode");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let group = cfg.kv_group_size();
    let kv_dim = cfg.kv_proj_dim();
    let eps = cfg.rms_norm_eps as f32;
    let half = dh / 2;
    let new_seq = past_seq + 1;

    let h = cfg.hidden_size;
    let zero_beta_hidden = synth_zero(&mut g, &mut params, "qwen3.zero_beta.hidden", h);
    let zero_beta_headdim = synth_zero(&mut g, &mut params, "qwen3.zero_beta.head_dim", dh);

    // ── Inputs ─────────────────────────────────────────────────────
    let input_ids = g.input("input_ids", Shape::new(&[batch, 1], DType::F32));
    let cos_id = g.input("rope_cos", Shape::new(&[1, half], f));
    let sin_id = g.input("rope_sin", Shape::new(&[1, half], f));
    // Optional custom mask input — shape [batch, past_seq + 1]. 1.0 for
    // valid K positions, 0.0 for padded positions. Only declared and
    // wired into attention when `use_custom_mask` is true.
    let mask_id = if use_custom_mask {
        Some(g.input("mask", Shape::new(&[batch, past_seq + 1], f)))
    } else {
        None
    };
    let mut past_k_ids: Vec<NodeId> = Vec::with_capacity(cfg.num_hidden_layers);
    let mut past_v_ids: Vec<NodeId> = Vec::with_capacity(cfg.num_hidden_layers);
    for layer_idx in 0..cfg.num_hidden_layers {
        let pk = g.input(
            &format!("past_k_{layer_idx}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        );
        let pv = g.input(
            &format!("past_v_{layer_idx}"),
            Shape::new(&[batch, past_seq, kv_dim], f),
        );
        past_k_ids.push(pk);
        past_v_ids.push(pv);
    }

    // ── Token embedding ────────────────────────────────────────────
    let embed_w = load_p(
        &mut g,
        &mut params,
        weights,
        "model.embed_tokens.weight",
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    let mut new_k_outputs: Vec<NodeId> = Vec::with_capacity(cfg.num_hidden_layers);
    let mut new_v_outputs: Vec<NodeId> = Vec::with_capacity(cfg.num_hidden_layers);

    // ── Decoder layers ─────────────────────────────────────────────
    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer_idx}");

        let in_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            false,
        )?;
        let normed_in = g.rms_norm(h_id, in_ln_g, zero_beta_hidden, eps);

        let q_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
            true,
        )?;
        let k_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
            true,
        )?;
        let v_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
            true,
        )?;
        let q = g.mm(normed_in, q_w); // [B, 1, q_dim]
        let k = g.mm(normed_in, k_w); // [B, 1, kv_dim]
        let v = g.mm(normed_in, v_w); // [B, 1, kv_dim]

        let q_norm_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.q_norm.weight"),
            false,
        )?;
        let k_norm_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.k_norm.weight"),
            false,
        )?;
        let q_normed = per_head_rms(
            &mut g,
            q,
            q_norm_g,
            zero_beta_headdim,
            batch,
            1,
            nh,
            dh,
            eps,
        );
        let k_normed = per_head_rms(
            &mut g,
            k,
            k_norm_g,
            zero_beta_headdim,
            batch,
            1,
            nkv,
            dh,
            eps,
        );

        // RoPE at position `past_seq` — cos/sin slice is 1 row, so the
        // existing Rope kernel applies it to the single new-token row.
        let q_rope = g.rope(q_normed, cos_id, sin_id, dh);
        let k_rope = g.rope(k_normed, cos_id, sin_id, dh);

        // Append new K/V to the cached past. New shape: [B, past_seq+1, kv_dim].
        let new_k = g.concat_(vec![past_k_ids[layer_idx], k_rope], 1);
        let new_v = g.concat_(vec![past_v_ids[layer_idx], v], 1);
        new_k_outputs.push(new_k);
        new_v_outputs.push(new_v);

        // GQA: widen K/V from num_kv_heads to num_heads.
        let k_rep = repeat_kv(&mut g, new_k, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, new_v, nkv, dh, group);

        // SDPA. Causal path: Lq=1, Lk=past+1, mask is a no-op (Q
        // attends to all keys). Custom-mask path: host pads past_k/v
        // and passes a per-step mask that zeros the padded positions.
        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = match mask_id {
            Some(mask) => g.attention(q_rope, k_rep, v_rep, mask, nh, dh, attn_shape),
            None => g.attention_kind(q_rope, k_rep, v_rep, nh, dh, MaskKind::Causal, attn_shape),
        };

        let o_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
            true,
        )?;
        let attn_out = g.mm(attn, o_w);
        let post_attn = g.add(h_id, attn_out);

        let post_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            false,
        )?;
        let normed_post = g.rms_norm(post_attn, post_ln_g, zero_beta_hidden, eps);

        let gate_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.gate_proj.weight"),
            true,
        )?;
        let up_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.up_proj.weight"),
            true,
        )?;
        let down_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.down_proj.weight"),
            true,
        )?;
        let gate = g.mm(normed_post, gate_w);
        let up = g.mm(normed_post, up_w);
        let gate_act = g.silu(gate);
        let swiglu = g.mul(gate_act, up);
        let ffn_out = g.mm(swiglu, down_w);

        h_id = g.add(post_attn, ffn_out);
    }

    let final_ln_g = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
    let hidden = g.rms_norm(h_id, final_ln_g, zero_beta_hidden, eps);

    let lm_head_w = if cfg.tie_word_embeddings {
        let embed = params
            .get("model.embed_tokens.weight")
            .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
        let vocab = cfg.vocab_size;
        let hidden_size = cfg.hidden_size;
        let mut transposed = vec![0f32; embed.len()];
        for v in 0..vocab {
            for hi in 0..hidden_size {
                transposed[hi * vocab + v] = embed[v * hidden_size + hi];
            }
        }
        let name = "qwen3.lm_head.tied_t";
        let id = g.param(name, Shape::new(&[hidden_size, vocab], DType::F32));
        params.insert(name.to_string(), transposed);
        id
    } else {
        load_p(&mut g, &mut params, weights, "lm_head.weight", true)?
    };
    let logits = g.mm(hidden, lm_head_w);

    // Outputs: [logits, k_0, v_0, k_1, v_1, ..., k_{N-1}, v_{N-1}]
    let mut outputs = Vec::with_capacity(1 + 2 * cfg.num_hidden_layers);
    outputs.push(logits);
    for i in 0..cfg.num_hidden_layers {
        outputs.push(new_k_outputs[i]);
        outputs.push(new_v_outputs[i]);
    }
    g.set_outputs(outputs);
    let _ = new_seq; // documents the resulting cache length
    Ok((g, params))
}

// ────────────────────────────────────────────────────────────────
// Packed-weights mode — Op::DequantMatMul for the big projections.
//
// The default builder dequants every K-quant tensor to F32 at load
// (~7-9× memory expansion). For models that won't fit in unified
// memory after that — Qwen3-14B+, Qwen3.6-27B-MTP, etc. — this
// alternate builder keeps the per-layer + LM-head matmul weights
// as packed bytes in the arena and emits `Op::DequantMatMul` so
// the kernel dequants per matmul invocation.
//
// Trade-off: 7-9× less load memory, 2-4× slower per matmul on CPU
// (each call re-dequants the weight to a scratch buffer before
// sgemm). A future tile-streaming kernel would close the compute
// gap; today it's "loadable but slow."
// ────────────────────────────────────────────────────────────────

/// Companion to [`build_qwen3_graph_sized`] that keeps K-quant
/// weights packed in the arena. Pass an empty `packed` HashMap;
/// the function fills it with the GGUF-packed bytes for every
/// per-layer matmul weight + the LM-head weight. Non-K-quant
/// tensors (F32 norms, F16/BF16, Q4_0/Q5_0/Q8_0 legacy formats)
/// stay in `params` as F32.
///
/// Used together with [`Qwen3Generator::from_path_packed`] and
/// [`Qwen3RunnerBuilder::packed_weights`] for the low-memory
/// inference path.
#[allow(clippy::too_many_arguments)]
pub fn build_qwen3_graph_sized_packed(
    cfg: &Qwen3Config,
    weights: &mut crate::weight_loader::GgufLoader,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    packed: &mut HashMap<String, (Vec<u8>, rlx_ir::quant::QuantScheme, Vec<usize>)>,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    use rlx_ir::quant::QuantScheme;

    if cfg.num_attention_heads % cfg.num_key_value_heads != 0 {
        return Err(anyhow!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        ));
    }
    let mut g = Graph::new("qwen3_packed");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let dh = cfg.head_dim;
    let group = cfg.kv_group_size();
    let eps = cfg.rms_norm_eps as f32;

    let zero_beta_hidden = synth_zero(&mut g, &mut params, "qwen3.zero_beta.hidden", h);
    let zero_beta_headdim = synth_zero(&mut g, &mut params, "qwen3.zero_beta.head_dim", dh);

    // RoPE caches stay F32 (small).
    let half = dh / 2;
    let mut cos_data = vec![0f32; cfg.max_position_embeddings * half];
    let mut sin_data = vec![0f32; cfg.max_position_embeddings * half];
    for pos in 0..cfg.max_position_embeddings {
        for i in 0..half {
            let freq = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dh as f64);
            let angle = pos as f64 * freq;
            let (s, c) = angle.sin_cos();
            cos_data[pos * half + i] = c as f32;
            sin_data[pos * half + i] = s as f32;
        }
    }
    let cos_id = g.param("rope.cos", Shape::new(&[cfg.max_position_embeddings, half], f));
    params.insert("rope.cos".into(), cos_data);
    let sin_id = g.param("rope.sin", Shape::new(&[cfg.max_position_embeddings, half], f));
    params.insert("rope.sin".into(), sin_data);

    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));

    // Embedding stays F32 — gather op needs dequant'd table.
    let embed_w = load_p(
        &mut g,
        &mut params,
        weights,
        "model.embed_tokens.weight",
        false,
    )?;
    let mut h_id = g.gather_(embed_w, input_ids, 0);

    // Helper closure: load a matmul weight either packed (K-quant)
    // or F32 (anything else). Returns (NodeId, Option<scheme>) —
    // `Some(scheme)` means caller emits `Op::DequantMatMul`,
    // `None` means caller emits regular `Op::MatMul`.
    //
    // The shape semantics: for the packed case, the param is a
    // U8 byte tensor; the *output* shape of the matmul is `[..., n]`
    // where n is what GGUF reports as the innermost dim (=
    // out_features in HF). Caller passes that n.
    fn load_proj(
        g: &mut Graph,
        params: &mut HashMap<String, Vec<f32>>,
        packed: &mut HashMap<String, (Vec<u8>, QuantScheme, Vec<usize>)>,
        weights: &mut crate::weight_loader::GgufLoader,
        key: &str,
    ) -> Result<(NodeId, Option<QuantScheme>, Vec<usize>)> {
        // Probe packed first; if not a K-quant, fall back to F32.
        if let Some((bytes, scheme, shape)) = weights.take_packed(key)? {
            let id = g.param(key, Shape::new(&[bytes.len()], DType::U8));
            packed.insert(key.to_string(), (bytes, scheme, shape.clone()));
            // GGUF shape (after the safetensors-style reverse) is
            // `[out, in]` — same as HF safetensors `[out_features,
            // in_features]`. The matmul output dim is `out` = shape[0].
            Ok((id, Some(scheme), shape))
        } else {
            let nid = load_p(g, params, weights, key, /*transpose*/ true)?;
            // load_p with transpose=true gives back [in, out].
            let shape = params
                .get(key)
                .map(|_| Vec::<usize>::new()) // dummy; unused for the F32 path
                .unwrap_or_default();
            Ok((nid, None, shape))
        }
    }

    // Emit either DequantMatMul or MatMul depending on scheme.
    fn emit_proj(
        g: &mut Graph,
        input: NodeId,
        w: NodeId,
        scheme: Option<QuantScheme>,
        out_shape: Shape,
    ) -> NodeId {
        match scheme {
            Some(s) => g.add_node(
                Op::DequantMatMul { scheme: s },
                vec![input, w],
                out_shape,
            ),
            None => g.mm(input, w),
        }
    }

    let kv_outputs: Vec<NodeId> = Vec::new();
    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("model.layers.{layer_idx}");

        let in_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.input_layernorm.weight"),
            false,
        )?;
        let normed_in = g.rms_norm(h_id, in_ln_g, zero_beta_hidden, eps);

        let q_dim = nh * dh;
        let kv_dim = nkv * dh;
        let (q_w, q_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.q_proj.weight"),
        )?;
        let (k_w, k_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.k_proj.weight"),
        )?;
        let (v_w, v_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.v_proj.weight"),
        )?;
        let q = emit_proj(&mut g, normed_in, q_w, q_s, Shape::new(&[batch, seq, q_dim], f));
        let k = emit_proj(&mut g, normed_in, k_w, k_s, Shape::new(&[batch, seq, kv_dim], f));
        let v = emit_proj(&mut g, normed_in, v_w, v_s, Shape::new(&[batch, seq, kv_dim], f));

        let q_norm_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.q_norm.weight"),
            false,
        )?;
        let k_norm_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.self_attn.k_norm.weight"),
            false,
        )?;
        let q_normed = per_head_rms(&mut g, q, q_norm_g, zero_beta_headdim, batch, seq, nh, dh, eps);
        let k_normed = per_head_rms(&mut g, k, k_norm_g, zero_beta_headdim, batch, seq, nkv, dh, eps);

        let q_rope = g.rope(q_normed, cos_id, sin_id, dh);
        let k_rope = g.rope(k_normed, cos_id, sin_id, dh);

        let k_rep = repeat_kv(&mut g, k_rope, nkv, dh, group);
        let v_rep = repeat_kv(&mut g, v, nkv, dh, group);

        let attn_shape = shape::attention_shape(g.shape(q_rope));
        let attn = g.attention_kind(q_rope, k_rep, v_rep, nh, dh, MaskKind::Causal, attn_shape);

        let (o_w, o_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.self_attn.o_proj.weight"),
        )?;
        let attn_out = emit_proj(&mut g, attn, o_w, o_s, Shape::new(&[batch, seq, h], f));
        let post_attn = g.add(h_id, attn_out);

        let post_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.post_attention_layernorm.weight"),
            false,
        )?;
        let normed_post = g.rms_norm(post_attn, post_ln_g, zero_beta_hidden, eps);

        let (gate_w, gate_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.gate_proj.weight"),
        )?;
        let (up_w, up_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.up_proj.weight"),
        )?;
        let (down_w, down_s, _) = load_proj(
            &mut g,
            &mut params,
            packed,
            weights,
            &format!("{lp}.mlp.down_proj.weight"),
        )?;
        let inter = cfg.intermediate_size;
        let gate = emit_proj(&mut g, normed_post, gate_w, gate_s, Shape::new(&[batch, seq, inter], f));
        let up = emit_proj(&mut g, normed_post, up_w, up_s, Shape::new(&[batch, seq, inter], f));
        let gate_act = g.silu(gate);
        let swiglu = g.mul(gate_act, up);
        let ffn_out = emit_proj(&mut g, swiglu, down_w, down_s, Shape::new(&[batch, seq, h], f));
        h_id = g.add(post_attn, ffn_out);
        let _ = kv_outputs.len(); // silence unused for now
    }

    let final_ln_g = load_p(&mut g, &mut params, weights, "model.norm.weight", false)?;
    let hidden = g.rms_norm(h_id, final_ln_g, zero_beta_hidden, eps);

    let out = if with_lm_head {
        let head_input = if last_logits_only {
            g.narrow_(hidden, 1, seq - 1, 1)
        } else {
            hidden
        };
        // Tied: try to reuse the embed weight in packed form. If it's
        // a K-quant in the GGUF, register a SECOND packed param under
        // a distinct name (lm_head reads it; gather reads the F32
        // dequant'd version already in `params`).
        let (lm_head_w, lm_head_scheme) = if cfg.tie_word_embeddings {
            // Re-open the file to grab the packed bytes for embed.
            // The loader has already taken the F32 dequant copy for
            // gather; take_packed by-name would fail (already taken).
            // Cheapest fix: build the f32→packed copy NOT — that's
            // hard. Instead: fall back to F32 tied path (which is
            // what we did before — pre-transpose once at build time).
            let embed = params
                .get("model.embed_tokens.weight")
                .ok_or_else(|| anyhow!("missing model.embed_tokens.weight for tied lm_head"))?;
            let vocab = cfg.vocab_size;
            let hidden_size = cfg.hidden_size;
            let mut transposed = vec![0f32; embed.len()];
            for v in 0..vocab {
                for hi in 0..hidden_size {
                    transposed[hi * vocab + v] = embed[v * hidden_size + hi];
                }
            }
            let name = "qwen3.lm_head.tied_t";
            let id = g.param(name, Shape::new(&[hidden_size, vocab], DType::F32));
            params.insert(name.to_string(), transposed);
            (id, None)
        } else {
            let (id, scheme, _) = load_proj(
                &mut g,
                &mut params,
                packed,
                weights,
                "lm_head.weight",
            )?;
            (id, scheme)
        };
        emit_proj(
            &mut g,
            head_input,
            lm_head_w,
            lm_head_scheme,
            Shape::new(
                &[batch, if last_logits_only { 1 } else { seq }, cfg.vocab_size],
                f,
            ),
        )
    } else {
        hidden
    };

    g.set_outputs(vec![out]);
    Ok((g, params))
}
