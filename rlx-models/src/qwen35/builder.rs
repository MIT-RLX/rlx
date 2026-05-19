// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! Qwen3.5 / Qwen3.6 forward graph builder.
//!
//! End-to-end prefill graph composing the gated-DeltaNet "linear
//! attention" trunk layers, the every-`full_attention_interval`
//! standard attention layers, and (optionally) the MTP head. Mirror
//! of `llama.cpp / src/models/qwen35.cpp` translated into RLX IR.
//!
//! **Status (this slice):**
//! - Trunk linear-attn block: full forward (norm → joint qkv +
//!   gate split → α/β/dt → softplus gate → depthwise conv (k=4)
//!   manually unrolled → SiLU → q/k/v split → L2-norm → GQA
//!   repeat → [`Op::GatedDeltaNet`] → SiLU(z)-gated norm →
//!   `ssm_out` → residual → post-norm → SwiGLU FFN → residual).
//! - Trunk full-attn block: joint Q+gate projection (Qwen3-Next
//!   style) → Q/K norm → standard RoPE → causal attention →
//!   sigmoid-gate multiply → `attn_output` → SwiGLU FFN.
//! - MTP head: enorm(token_embd) ++ hnorm(h_pre) → eh_proj →
//!   one full-attn block → shared head LM.
//!
//! **Deviations from the reference (documented for the parity
//! oracle to verify against):**
//! - RoPE is applied as standard per-axis rotation over the first
//!   `rope_dim_count` dims; the multi-section MRoPE (with
//!   `rope_sections`) is approximated as plain RoPE. This is
//!   correct for single-modality text generation where all
//!   sections rotate at the same base frequency.
//! - `ssm_conv1d` is unrolled into `k = 4` narrow + mul + add
//!   instead of using `Op::Conv` (avoids NHWC/NCHW reshuffling and
//!   matches the bytewise math llama.cpp uses for k≤4 SSM convs).
//! - State for the gated-DeltaNet scan resets per batch (the
//!   kernel doesn't expose decode-time state caching yet — prefill
//!   only). Decode-mode KV/state cache is the follow-up slice.
//!
//! Memory footprint: every K-quant weight is dequantized to F32 at
//! load time. For Qwen3.5-0.8B Q4_K_M (~0.4 GB packed) that's
//! ~1.5 GB. For Qwen3.6-27B Q4_K_M (~16 GB packed) that's ~65 GB,
//! which won't fit on commodity Macs — the packed-weights path
//! (`Op::DequantMatMul`) lands in a follow-up.

use crate::qwen35::config::Qwen35Config;
use crate::qwen35::weights::{
    MatWeight, Qwen35FullAttnLayer, Qwen35LinearLayer, Qwen35MtpLayer, Qwen35TrunkLayer,
    Qwen35Weights,
};
use anyhow::{Result, anyhow};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, MaskKind};
use rlx_ir::quant::QuantScheme;
use rlx_ir::*;
use std::collections::HashMap;

/// Side channel for packed K-quant weights. `build_qwen35_graph_sized`
/// populates this when a `MatWeight::Packed` source is encountered:
/// `param_name → (loader_key, scheme, [out, in])`. The runner uses
/// `loader_key` to fetch bytes from the still-alive `GgufLoader` via
/// `tensor_bytes_borrowed`, then `compiled.set_param_typed(param_name,
/// bytes, DType::U8)`.
pub type PackedParams = HashMap<String, (String, QuantScheme, Vec<usize>)>;

/// Build the Qwen3.5 forward IR.
///
/// * `with_lm_head` — emit the final `lm_head` projection (logits
///   over vocab). Disable when only the hidden state is needed.
/// * `last_logits_only` — gather only the last-token logits before
///   the LM head, saving a `[seq, vocab]` matmul on long prompts.
/// * `enable_mtp_head` — also emit the MTP head's logits as a
///   second output. The MTP head is fed from the trunk's final
///   pre-norm hidden state; its output is `[batch, 1, vocab]`
///   regardless of `seq`.
///
/// Returns the graph plus a `HashMap` of `name → f32 bytes` for
/// every loaded parameter. Caller uploads via `compiled.set_param`.
pub fn build_qwen35_graph_sized(
    cfg: &Qwen35Config,
    weights: Qwen35Weights,
    batch: usize,
    seq: usize,
    with_lm_head: bool,
    last_logits_only: bool,
    enable_mtp_head: bool,
) -> Result<(Graph, HashMap<String, Vec<f32>>, PackedParams)> {
    let n_embd = cfg.hidden_size;
    let n_vocab = if weights.token_embd.is_empty() {
        cfg.vocab_size
    } else {
        weights.token_embd.len() / n_embd
    };
    if n_vocab == 0 {
        return Err(anyhow!("qwen35: vocab_size could not be inferred"));
    }

    let mut g = Graph::new("qwen35");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let mut packed: PackedParams = HashMap::new();

    // ── Input: token ids ───────────────────────────────────────
    // Use F32 dtype on the host I/O surface (the gather kernel
    // accepts F32 indices via index-cast internally). This mirrors
    // the qwen3 builder's convention.
    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));

    // Embedding table param.
    let embed_w = register_param(
        &mut g,
        &mut params,
        "token_embd.weight",
        weights.token_embd.clone(),
        Shape::new(&[n_vocab, n_embd], DType::F32),
    );

    // Hidden states `[batch, seq, n_embd]`.
    let mut h = g.gather_(embed_w, input_ids, 0);

    // ── Trunk layers ───────────────────────────────────────────
    for (il, layer) in weights.trunk_layers.iter().enumerate() {
        match layer {
            Qwen35TrunkLayer::Linear(lin) => {
                h = build_linear_layer(&mut g, &mut params, &mut packed, cfg, il, lin,
                    batch, seq, h)?;
            }
            Qwen35TrunkLayer::FullAttn(fa) => {
                h = build_full_attn_layer(&mut g, &mut params, &mut packed, cfg, il, fa,
                    batch, seq, h)?;
            }
        }
    }

    // Snapshot pre-norm hidden state (MTP input) before the final
    // RMS norm. Kept at full `[batch, seq, n_embd]` so the MTP head
    // sees every token's hidden state — mirrors llama.cpp #23198,
    // which moved the output-row gather to *after* `h_pre_norm` so
    // the MTP draft path keeps a dense pre-norm even when the LM
    // head only emits last-token logits.
    let h_pre_norm = h;

    // Final RMS norm — applied to the (possibly narrowed) hidden
    // state. Narrowing here saves the [seq-1, n_embd] norm and the
    // [seq-1, vocab] matmul that would otherwise run for outputs we
    // discard. The MTP head is unaffected because it consumes
    // `h_pre_norm` (still full seq) above.
    let h_for_norm = if with_lm_head && last_logits_only {
        g.narrow_(h, 1, seq - 1, 1)
    } else {
        h
    };
    let out_norm = register_param(
        &mut g,
        &mut params,
        "output_norm.weight",
        weights.output_norm.clone(),
        Shape::new(&[n_embd], DType::F32),
    );
    let out_norm_beta = synth_zero(&mut g, &mut params, "output_norm.beta", n_embd);
    let h_norm = g.rms_norm(h_for_norm, out_norm, out_norm_beta, cfg.rms_norm_eps as f32);
    let h_logits_in = h_norm;

    let mut outputs = Vec::new();

    if with_lm_head {
        // LM head: tied to token_embd if no separate `output` weight.
        // Packed-aware when `weights.output` carries K-quant bytes.
        let logit_rows = if last_logits_only { 1 } else { seq };
        let logit_shape = Shape::new(&[batch, logit_rows, n_vocab], DType::F32);
        let logits = match &weights.output {
            Some(w) => {
                let head = proj_mat(
                    &mut g,
                    &mut params,
                    &mut packed,
                    "output.weight",
                    w,
                    n_embd,
                    n_vocab,
                );
                emit_proj(&mut g, h_logits_in, head, w, logit_shape)
            }
            None => {
                let embed_t = transpose_2d(&weights.token_embd, n_vocab, n_embd);
                let tied = register_param(
                    &mut g,
                    &mut params,
                    "lm_head.tied_t",
                    embed_t,
                    Shape::new(&[n_embd, n_vocab], DType::F32),
                );
                g.mm(h_logits_in, tied)
            }
        };
        outputs.push(logits);
    }

    // ── MTP head (optional) ────────────────────────────────────
    if enable_mtp_head {
        let mtp_layer = weights
            .mtp_layers
            .first()
            .ok_or_else(|| anyhow!("qwen35: MTP requested but no MTP layers loaded"))?;
        let mtp_il = cfg.num_hidden_layers - cfg.nextn_predict_layers;
        let mtp_logits = build_mtp_head(
            &mut g,
            &mut params,
            &mut packed,
            cfg,
            mtp_il,
            mtp_layer,
            batch,
            seq,
            input_ids,
            h_pre_norm,
            &weights.token_embd,
            n_vocab,
        )?;
        outputs.push(mtp_logits);
    }

    if outputs.is_empty() {
        outputs.push(h_norm);
    }
    g.set_outputs(outputs);
    Ok((g, params, packed))
}

// ── Trunk linear-attention (gated DeltaNet) layer ──────────────
fn build_linear_layer(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    lin: &Qwen35LinearLayer,
    batch: usize,
    seq: usize,
    h_in: NodeId,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let n_ff = cfg.intermediate_size;
    let n_state = cfg.ssm_state_size;
    let n_k_heads = cfg.ssm_group_count;
    let n_v_heads = cfg.ssm_time_step_rank;
    let key_dim = n_state * n_k_heads;
    let value_dim = n_state * n_v_heads;
    let conv_channels = key_dim * 2 + value_dim;
    let k_conv = cfg.ssm_conv_kernel;

    // Declare bias / 1-D-shape params up front. The
    // `FuseMatMulBiasAct` pass walks nodes in declaration order and
    // tries to fuse `MatMul → Add(rank1_bias) → Activation`; if the
    // bias is declared *after* the matmul it's not yet in the
    // rewriter's id_map and the pass panics. Declaring rank-1
    // params first is the contract every other builder in this
    // crate follows.
    let dt_bias = param(
        g,
        params,
        &name(il, "ssm_dt.bias"),
        &lin.ssm_dt_bias,
        &[n_v_heads],
    );
    let ssm_a_p = param(g, params, &name(il, "ssm_a"), &lin.ssm_a, &[n_v_heads]);

    // attn_norm (pre-norm)
    let attn_norm_w = param(g, params, &name(il, "attn_norm.weight"), &lin.attn_norm,
        &[n_embd]);
    let attn_norm_b = synth_zero(g, params, &name(il, "attn_norm.beta"), n_embd);
    let x = g.rms_norm(h_in, attn_norm_w, attn_norm_b, cfg.rms_norm_eps as f32);

    // Reshape to 2D for matmul: [batch*seq, n_embd].
    let x_2d = g.reshape_(x, vec![(batch * seq) as i64, n_embd as i64]);

    let rows = batch * seq;
    // Fused qkv projection (key_dim*2 + value_dim channels).
    let qkv_w = proj_mat(g, params, packed, &name(il, "attn_qkv.weight"),
        &lin.attn_qkv, n_embd, conv_channels);
    let qkv = emit_proj(g, x_2d, qkv_w, &lin.attn_qkv,
        Shape::new(&[rows, conv_channels], DType::F32));
    // → [batch*seq, conv_channels]

    // Gate projection z.
    let gate_w = proj_mat(g, params, packed, &name(il, "attn_gate.weight"),
        &lin.attn_gate, n_embd, value_dim);
    let z = emit_proj(g, x_2d, gate_w, &lin.attn_gate,
        Shape::new(&[rows, value_dim], DType::F32));
    // → [batch*seq, value_dim]

    // alpha = ssm_alpha @ x ; shape [batch*seq, n_v_heads]
    let alpha_w = proj_mat(g, params, packed, &name(il, "ssm_alpha.weight"),
        &lin.ssm_alpha, n_embd, n_v_heads);
    let alpha = emit_proj(g, x_2d, alpha_w, &lin.ssm_alpha,
        Shape::new(&[rows, n_v_heads], DType::F32));

    // beta = sigmoid(ssm_beta @ x)
    let beta_w = proj_mat(g, params, packed, &name(il, "ssm_beta.weight"),
        &lin.ssm_beta, n_embd, n_v_heads);
    let beta_pre = emit_proj(g, x_2d, beta_w, &lin.ssm_beta,
        Shape::new(&[rows, n_v_heads], DType::F32));
    let beta = activation(g, Activation::Sigmoid, beta_pre);

    // gate_g = softplus(alpha + ssm_dt_bias) * ssm_a
    //   ssm_dt_bias: [n_v_heads], broadcast over [batch*seq, n_v_heads].
    let alpha_biased = g.add(alpha, dt_bias);
    let alpha_softplus = softplus(g, alpha_biased);
    let gate_g = g.mul(alpha_softplus, ssm_a_p);

    // Reshape gate/beta to [batch, seq, n_v_heads] for the
    // GatedDeltaNet kernel signature.
    let gate_g_3d =
        g.reshape_(gate_g, vec![batch as i64, seq as i64, n_v_heads as i64]);
    let beta_3d =
        g.reshape_(beta, vec![batch as i64, seq as i64, n_v_heads as i64]);

    // Depthwise 1-D conv (manually unrolled for k=4). The conv
    // input is qkv [batch*seq, conv_channels]; we view it as
    // [batch, seq, conv_channels] and apply causal left-padding
    // of (k-1) zeros.
    let qkv_3d = g.reshape_(
        qkv,
        vec![batch as i64, seq as i64, conv_channels as i64],
    );
    let conv_out = depthwise_conv1d_causal(
        g,
        params,
        &name(il, "ssm_conv1d.weight"),
        &lin.ssm_conv1d,
        qkv_3d,
        batch,
        seq,
        conv_channels,
        k_conv,
    )?;
    let conv_silu = g.silu(conv_out);
    // → [batch, seq, conv_channels]

    // Split convolved channels into q_conv, k_conv, v_conv.
    let q_part = g.narrow_(conv_silu, 2, 0, key_dim);
    let k_part = g.narrow_(conv_silu, 2, key_dim, key_dim);
    let v_part = g.narrow_(conv_silu, 2, key_dim * 2, value_dim);

    // Reshape into per-head: [batch, seq, n_*_heads, n_state].
    let q_heads = g.reshape_(
        q_part,
        vec![
            batch as i64,
            seq as i64,
            n_k_heads as i64,
            n_state as i64,
        ],
    );
    let k_heads = g.reshape_(
        k_part,
        vec![
            batch as i64,
            seq as i64,
            n_k_heads as i64,
            n_state as i64,
        ],
    );
    let v_heads = g.reshape_(
        v_part,
        vec![
            batch as i64,
            seq as i64,
            n_v_heads as i64,
            n_state as i64,
        ],
    );

    // L2 normalize Q and K along the last (state) dim.
    let q_l2 = l2_norm(g, q_heads, cfg.rms_norm_eps as f32);
    let k_l2 = l2_norm(g, k_heads, cfg.rms_norm_eps as f32);

    // GQA-repeat k from n_k_heads to n_v_heads if needed.
    let (q_rep, k_rep) = if n_k_heads == n_v_heads {
        (q_l2, k_l2)
    } else {
        let factor = n_v_heads / n_k_heads;
        if factor * n_k_heads != n_v_heads {
            return Err(anyhow!(
                "qwen35 layer {il}: n_v_heads={n_v_heads} must be a multiple \
                 of n_k_heads={n_k_heads} (gqa)"
            ));
        }
        (
            repeat_heads(g, q_l2, batch, seq, n_k_heads, n_state, factor),
            repeat_heads(g, k_l2, batch, seq, n_k_heads, n_state, factor),
        )
    };

    // GatedDeltaNet scan: returns [batch, seq, n_v_heads, n_state].
    let scan_out_shape = Shape::new(
        &[batch, seq, n_v_heads, n_state],
        DType::F32,
    );
    let scan_out =
        g.gated_delta_net(q_rep, k_rep, v_heads, gate_g_3d, beta_3d, n_state, scan_out_shape);

    // Gated norm: ssm_norm over the last (state) dim, multiplied
    // by silu(z) per element. z was [batch*seq, value_dim] = per
    // (head, state). Reshape both to [batch, seq, n_v_heads,
    // n_state] and apply.
    let z_4d = g.reshape_(
        z,
        vec![
            batch as i64,
            seq as i64,
            n_v_heads as i64,
            n_state as i64,
        ],
    );
    let z_silu = g.silu(z_4d);

    let ssm_norm_w = param(g, params, &name(il, "ssm_norm.weight"), &lin.ssm_norm,
        &[n_state]);
    let ssm_norm_b = synth_zero(g, params, &name(il, "ssm_norm.beta"), n_state);
    let scan_normed = g.rms_norm(scan_out, ssm_norm_w, ssm_norm_b, cfg.rms_norm_eps as f32);
    let scan_gated = g.mul(scan_normed, z_silu);

    // Reshape back to [batch*seq, value_dim] for ssm_out.
    let scan_flat = g.reshape_(
        scan_gated,
        vec![(batch * seq) as i64, value_dim as i64],
    );

    // ssm_out: [value_dim → n_embd]. Packed-aware.
    let ssm_out_w = proj_mat(g, params, packed, &name(il, "ssm_out.weight"),
        &lin.ssm_out, value_dim, n_embd);
    let attn_out_2d = emit_proj(g, scan_flat, ssm_out_w, &lin.ssm_out,
        Shape::new(&[rows, n_embd], DType::F32));
    let attn_out =
        g.reshape_(attn_out_2d, vec![batch as i64, seq as i64, n_embd as i64]);

    // Residual.
    let h_post_attn = g.add(h_in, attn_out);

    // Post-attention norm + SwiGLU FFN + residual.
    let h_ffn = build_ffn(g, params, cfg, il, h_post_attn, batch, seq,
        &lin.attn_post_norm, &lin.ffn_gate, &lin.ffn_up, &lin.ffn_down, n_ff, packed)?;

    Ok(h_ffn)
}

// ── Trunk full-attention (every full_attention_interval) layer ─
fn build_full_attn_layer(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    fa: &Qwen35FullAttnLayer,
    batch: usize,
    seq: usize,
    h_in: NodeId,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let n_ff = cfg.intermediate_size;
    let n_head = cfg.num_attention_heads;
    let n_kv_head = cfg.num_key_value_heads;
    let head_dim = cfg.key_length;
    let q_gate_cols = n_head * head_dim * 2;
    let kv_cols = n_kv_head * head_dim;

    // pre-norm
    let attn_norm_w = param(g, params, &name(il, "attn_norm.weight"), &fa.attn_norm,
        &[n_embd]);
    let attn_norm_b = synth_zero(g, params, &name(il, "attn_norm.beta"), n_embd);
    let x = g.rms_norm(h_in, attn_norm_w, attn_norm_b, cfg.rms_norm_eps as f32);
    let x_2d = g.reshape_(x, vec![(batch * seq) as i64, n_embd as i64]);

    let rows = batch * seq;
    // Joint Q + gate projection (Qwen3-Next).
    let q_gate_w = proj_mat(g, params, packed, &name(il, "attn_q.weight"),
        &fa.attn_q_gate, n_embd, q_gate_cols);
    let q_gate = emit_proj(g, x_2d, q_gate_w, &fa.attn_q_gate,
        Shape::new(&[rows, q_gate_cols], DType::F32));
    // Layout per qwen35.cpp ggml_view_3d: the n_head*2 axis is
    // (gate, q) interleaved per head, but ggml's strides imply
    // Q is at offset 0 and gate at offset n_embd_head_k. Equivalent
    // to splitting [batch*seq, n_head, head_dim*2] into [...,
    // head_dim] (q) and [..., head_dim] (gate).
    let q_gate_3d = g.reshape_(
        q_gate,
        vec![
            (batch * seq) as i64,
            n_head as i64,
            (head_dim * 2) as i64,
        ],
    );
    let q_view = g.narrow_(q_gate_3d, 2, 0, head_dim);
    let gate_view = g.narrow_(q_gate_3d, 2, head_dim, head_dim);

    // K, V projections.
    let k_w = proj_mat(g, params, packed, &name(il, "attn_k.weight"),
        &fa.attn_k, n_embd, kv_cols);
    let k_proj = emit_proj(g, x_2d, k_w, &fa.attn_k,
        Shape::new(&[rows, kv_cols], DType::F32));
    let v_w = proj_mat(g, params, packed, &name(il, "attn_v.weight"),
        &fa.attn_v, n_embd, kv_cols);
    let v_proj = emit_proj(g, x_2d, v_w, &fa.attn_v,
        Shape::new(&[rows, kv_cols], DType::F32));

    // Per-head Q norm: reshape to [batch*seq*n_head, head_dim].
    let q_for_norm = g.reshape_(
        q_view,
        vec![(batch * seq * n_head) as i64, head_dim as i64],
    );
    let q_norm_w = param(g, params, &name(il, "attn_q_norm.weight"),
        &fa.attn_q_norm, &[head_dim]);
    let q_norm_b = synth_zero(g, params, &name(il, "attn_q_norm.beta"), head_dim);
    let q_normed = g.rms_norm(q_for_norm, q_norm_w, q_norm_b, cfg.rms_norm_eps as f32);

    // Per-head K norm.
    let k_for_norm = g.reshape_(
        k_proj,
        vec![(batch * seq * n_kv_head) as i64, head_dim as i64],
    );
    let k_norm_w = param(g, params, &name(il, "attn_k_norm.weight"),
        &fa.attn_k_norm, &[head_dim]);
    let k_norm_b = synth_zero(g, params, &name(il, "attn_k_norm.beta"), head_dim);
    let k_normed = g.rms_norm(k_for_norm, k_norm_w, k_norm_b, cfg.rms_norm_eps as f32);

    // Reshape Q to [batch, seq, n_head, head_dim] and K to [batch,
    // seq, n_kv_head, head_dim] for RoPE.
    let q_4d = g.reshape_(
        q_normed,
        vec![batch as i64, seq as i64, n_head as i64, head_dim as i64],
    );
    let k_4d = g.reshape_(
        k_normed,
        vec![
            batch as i64,
            seq as i64,
            n_kv_head as i64,
            head_dim as i64,
        ],
    );

    // Standard RoPE (deviation from MRoPE noted in module docs).
    // Cos/sin tables are host-supplied at runtime.
    let half_d = cfg.rope_dim_count / 2;
    let cos_in = g.input(
        &format!("rope_cos_l{il}"),
        Shape::new(&[1, half_d], DType::F32),
    );
    let sin_in = g.input(
        &format!("rope_sin_l{il}"),
        Shape::new(&[1, half_d], DType::F32),
    );
    let q_for_rope = g.reshape_(
        q_4d,
        vec![(batch * seq * n_head) as i64, head_dim as i64],
    );
    let k_for_rope = g.reshape_(
        k_4d,
        vec![
            (batch * seq * n_kv_head) as i64,
            head_dim as i64,
        ],
    );
    let q_rot = g.rope(q_for_rope, cos_in, sin_in, head_dim);
    let k_rot = g.rope(k_for_rope, cos_in, sin_in, head_dim);

    // GQA repeat: widen K/V from n_kv_head to n_head along head dim.
    let group = n_head / n_kv_head;
    let k_rot_3d = g.reshape_(
        k_rot,
        vec![
            batch as i64,
            seq as i64,
            n_kv_head as i64,
            head_dim as i64,
        ],
    );
    let v_3d = g.reshape_(
        v_proj,
        vec![
            batch as i64,
            seq as i64,
            n_kv_head as i64,
            head_dim as i64,
        ],
    );
    let k_full = if group == 1 {
        k_rot_3d
    } else {
        repeat_heads(g, k_rot_3d, batch, seq, n_kv_head, head_dim, group)
    };
    let v_full = if group == 1 {
        v_3d
    } else {
        repeat_heads(g, v_3d, batch, seq, n_kv_head, head_dim, group)
    };

    // Reshape into the attention-op canonical [batch, seq, kv_dim].
    let kv_dim = n_head * head_dim;
    let q_attn = g.reshape_(
        q_rot,
        vec![batch as i64, seq as i64, kv_dim as i64],
    );
    let k_attn = g.reshape_(
        k_full,
        vec![batch as i64, seq as i64, kv_dim as i64],
    );
    let v_attn = g.reshape_(
        v_full,
        vec![batch as i64, seq as i64, kv_dim as i64],
    );

    // Causal attention. MaskKind::Causal synthesizes the mask in
    // the kernel — only Q, K, V are graph-level inputs.
    let attn_out = g.add_node(
        Op::Attention {
            num_heads: n_head,
            head_dim,
            mask_kind: MaskKind::Causal,
        },
        vec![q_attn, k_attn, v_attn],
        Shape::new(&[batch, seq, kv_dim], DType::F32),
    );

    // sigmoid(gate) * attn_out (reshape gate to [batch, seq, kv_dim]).
    let gate_flat = g.reshape_(
        gate_view,
        vec![batch as i64, seq as i64, kv_dim as i64],
    );
    let gate_sig = activation(g, Activation::Sigmoid, gate_flat);
    let attn_gated = g.mul(attn_out, gate_sig);

    // Output projection.
    let attn_gated_2d = g.reshape_(
        attn_gated,
        vec![(batch * seq) as i64, kv_dim as i64],
    );
    let out_w = proj_mat(g, params, packed, &name(il, "attn_output.weight"),
        &fa.attn_output, kv_dim, n_embd);
    let attn_out_proj = emit_proj(g, attn_gated_2d, out_w, &fa.attn_output,
        Shape::new(&[rows, n_embd], DType::F32));
    let attn_out_3d = g.reshape_(
        attn_out_proj,
        vec![batch as i64, seq as i64, n_embd as i64],
    );

    // Residual.
    let h_post_attn = g.add(h_in, attn_out_3d);

    // FFN.
    let h_ffn = build_ffn(g, params, cfg, il, h_post_attn, batch, seq,
        &fa.attn_post_norm, &fa.ffn_gate, &fa.ffn_up, &fa.ffn_down, n_ff, packed)?;

    Ok(h_ffn)
}

// ── MTP head (NextN) ───────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn build_mtp_head(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    cfg: &Qwen35Config,
    il: usize,
    mtp: &Qwen35MtpLayer,
    batch: usize,
    seq: usize,
    input_ids: NodeId,
    h_pre_norm: NodeId,
    trunk_token_embd: &[f32],
    n_vocab: usize,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;

    // hnorm(h_pre_norm) and enorm(tok_embd(input_ids)). The MTP
    // head consumes the **trunk's pre-norm** hidden state alongside
    // a fresh embedding of the input ids.
    let hnorm_w = param(g, params, &name(il, "nextn.hnorm.weight"), &mtp.hnorm,
        &[n_embd]);
    let hnorm_b = synth_zero(g, params, &name(il, "nextn.hnorm.beta"), n_embd);
    let h_normed = g.rms_norm(h_pre_norm, hnorm_w, hnorm_b, cfg.rms_norm_eps as f32);

    // Embedding table for the MTP head: the optional override may
    // ship packed K-quant bytes, but gather needs dense rows, so we
    // materialize to F32 either way.
    let embed_bytes: Vec<f32> = match &mtp.embed_tokens {
        Some(MatWeight::F32(v)) => v.clone(),
        Some(MatWeight::Packed { .. }) => {
            // GGUF gather on packed bytes is a separate kernel and
            // not wired today. Fall back to the trunk's already-
            // dequantized embed table — same vocab, so functionally
            // equivalent unless the MTP head was tuned with a
            // different embedding.
            trunk_token_embd.to_vec()
        }
        None => trunk_token_embd.to_vec(),
    };
    let embed_w = register_param(
        g,
        params,
        &name(il, "nextn.embed_tokens.weight"),
        embed_bytes,
        Shape::new(&[n_vocab, n_embd], DType::F32),
    );
    let tok_embd = g.gather_(embed_w, input_ids, 0);

    let enorm_w = param(g, params, &name(il, "nextn.enorm.weight"), &mtp.enorm,
        &[n_embd]);
    let enorm_b = synth_zero(g, params, &name(il, "nextn.enorm.beta"), n_embd);
    let e_normed = g.rms_norm(tok_embd, enorm_w, enorm_b, cfg.rms_norm_eps as f32);

    // Concat [e, h] along the embedding dim → [batch, seq, 2*n_embd].
    let concat = g.concat_(vec![e_normed, h_normed], 2);

    // eh_proj: [2*n_embd → n_embd]. Map back to hidden dim.
    let concat_2d = g.reshape_(
        concat,
        vec![(batch * seq) as i64, (2 * n_embd) as i64],
    );
    let eh_w = proj_mat(g, params, packed, &name(il, "nextn.eh_proj.weight"),
        &mtp.eh_proj, 2 * n_embd, n_embd);
    let cur_2d = emit_proj(g, concat_2d, eh_w, &mtp.eh_proj,
        Shape::new(&[batch * seq, n_embd], DType::F32));
    let cur = g.reshape_(cur_2d, vec![batch as i64, seq as i64, n_embd as i64]);

    // Single full-attn-style block on the projected hidden state.
    let cur_after = build_full_attn_layer(g, params, packed, cfg, il, &mtp.base,
        batch, seq, cur)?;

    // Shared head norm + LM head (with fallback to trunk's
    // output_norm / output if the NextN-specific tensors are
    // absent).
    let head_norm_w = if let Some(w) = &mtp.shared_head_norm {
        param(
            g,
            params,
            &name(il, "nextn.shared_head_norm.weight"),
            w,
            &[n_embd],
        )
    } else {
        // Fall back to trunk's output_norm (already registered).
        // Re-register a duplicate under a head-specific name to
        // avoid graph-side aliasing surprises.
        let copy = mtp
            .shared_head_norm
            .clone()
            .unwrap_or_else(|| vec![1.0f32; n_embd]); // unreachable
        let _ = copy;
        // Resolve through `params` map.
        match params.get("output_norm.weight").cloned() {
            Some(w) => register_param(
                g,
                params,
                &name(il, "nextn.shared_head_norm.weight_fallback"),
                w,
                Shape::new(&[n_embd], DType::F32),
            ),
            None => synth_zero(g, params, &name(il, "nextn.shared_head_norm.placeholder"),
                n_embd),
        }
    };
    let head_norm_b = synth_zero(g, params, &name(il, "nextn.shared_head_norm.beta"),
        n_embd);

    // Last-token narrow for the MTP head — it predicts only one
    // step ahead.
    let last = g.narrow_(cur_after, 1, seq - 1, 1);
    let last_norm = g.rms_norm(last, head_norm_w, head_norm_b, cfg.rms_norm_eps as f32);

    let logits = if let Some(w) = &mtp.shared_head_head {
        let head_w = proj_mat(g, params, packed,
            &name(il, "nextn.shared_head_head.weight"),
            w, n_embd, n_vocab);
        emit_proj(g, last_norm, head_w, w, Shape::new(&[batch, 1, n_vocab], DType::F32))
    } else {
        // Tied head: reuse the embedding table (always dense).
        let bytes = transpose_2d(trunk_token_embd, n_vocab, n_embd);
        let head = register_param(
            g,
            params,
            &name(il, "nextn.shared_head_head.tied_t"),
            bytes,
            Shape::new(&[n_embd, n_vocab], DType::F32),
        );
        g.mm(last_norm, head)
    };
    Ok(logits)
}

// ── Helpers ────────────────────────────────────────────────────

fn build_ffn(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    cfg: &Qwen35Config,
    il: usize,
    h_in: NodeId,
    batch: usize,
    seq: usize,
    attn_post_norm: &[f32],
    ffn_gate: &MatWeight,
    ffn_up: &MatWeight,
    ffn_down: &MatWeight,
    n_ff: usize,
    packed: &mut PackedParams,
) -> Result<NodeId> {
    let n_embd = cfg.hidden_size;
    let post_norm_w = param(g, params, &name(il, "post_attention_norm.weight"),
        attn_post_norm, &[n_embd]);
    let post_norm_b = synth_zero(g, params, &name(il, "post_attention_norm.beta"), n_embd);
    let h_normed = g.rms_norm(h_in, post_norm_w, post_norm_b, cfg.rms_norm_eps as f32);
    let h_2d = g.reshape_(h_normed, vec![(batch * seq) as i64, n_embd as i64]);

    let gate_w = proj_mat(g, params, packed, &name(il, "ffn_gate.weight"),
        ffn_gate, n_embd, n_ff);
    let up_w = proj_mat(g, params, packed, &name(il, "ffn_up.weight"),
        ffn_up, n_embd, n_ff);
    let down_w = proj_mat(g, params, packed, &name(il, "ffn_down.weight"),
        ffn_down, n_ff, n_embd);

    let rows = (batch * seq) as usize;
    let gate = emit_proj(g, h_2d, gate_w, ffn_gate, Shape::new(&[rows, n_ff], DType::F32));
    let up = emit_proj(g, h_2d, up_w, ffn_up, Shape::new(&[rows, n_ff], DType::F32));
    let gate_silu = g.silu(gate);
    let swiglu = g.mul(gate_silu, up);
    let down = emit_proj(g, swiglu, down_w, ffn_down, Shape::new(&[rows, n_embd], DType::F32));
    let ffn_out = g.reshape_(down, vec![batch as i64, seq as i64, n_embd as i64]);
    Ok(g.add(h_in, ffn_out))
}

/// Depthwise causal 1-D conv with kernel size `k`, unrolled into
/// `k` left-shifted slices summed elementwise. Input
/// `[batch, seq, channels]`; output same shape. Kernel weights
/// `[k, channels]` in the on-disk layout (innermost = channels per
/// gguf).
fn depthwise_conv1d_causal(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    weight: &[f32],
    input: NodeId,
    batch: usize,
    seq: usize,
    channels: usize,
    k: usize,
) -> Result<NodeId> {
    // Pre-pad input on the seq dim with (k-1) zeros so the conv is
    // causal.
    let pad_shape = Shape::new(&[batch, k - 1, channels], DType::F32);
    let pad_name = format!("{name}.causal_pad");
    let pad_data = vec![0f32; batch * (k - 1) * channels];
    let pad = register_param(g, params, &pad_name, pad_data, pad_shape);
    let padded = g.concat_(vec![pad, input], 1);
    // padded: [batch, seq + k - 1, channels]

    // Register kernel: shape [k, channels] (no transpose needed —
    // we index along axis 0 directly).
    let weight_p = param(g, params, name, weight, &[k, channels]);

    // Sum_{i=0..k} padded.narrow(seq, i, seq) * weight[i, :].
    let mut acc: Option<NodeId> = None;
    for i in 0..k {
        let slice_i = g.narrow_(padded, 1, i, seq);
        let kern_i_row = g.narrow_(weight_p, 0, i, 1);
        // Reshape kern_i_row from [1, channels] → [channels] for
        // broadcast across [batch, seq, channels].
        let kern_i = g.reshape_(kern_i_row, vec![channels as i64]);
        let prod = g.mul(slice_i, kern_i);
        acc = Some(match acc {
            None => prod,
            Some(a) => g.add(a, prod),
        });
    }
    Ok(acc.expect("k >= 1"))
}

/// L2 normalize along the last dim:
/// `out = x / sqrt(sum(x², axis=-1, keepdim) + eps)`.
fn l2_norm(g: &mut Graph, x: NodeId, eps: f32) -> NodeId {
    let rank = g.shape(x).rank();
    let last = rank - 1;
    let sq = g.mul(x, x);
    let sumsq = g.sum(sq, vec![last], true);
    // sumsq + eps
    let eps_p = scalar_const(g, eps);
    let sumsq_eps = g.add(sumsq, eps_p);
    let denom = g.sqrt(sumsq_eps);
    g.div(x, denom)
}

/// `softplus(x) = log(1 + exp(x))`.
fn softplus(g: &mut Graph, x: NodeId) -> NodeId {
    let ex = activation(g, Activation::Exp, x);
    let one = scalar_const(g, 1.0);
    let sum = g.add(ex, one);
    activation(g, Activation::Log, sum)
}

/// Repeat each head `factor` times along the head axis (axis = 2,
/// for a [b, s, h, d] tensor). Concatenates `factor` narrows for
/// each source head.
fn repeat_heads(
    g: &mut Graph,
    x: NodeId,
    batch: usize,
    seq: usize,
    in_heads: usize,
    head_dim: usize,
    factor: usize,
) -> NodeId {
    let _ = (batch, seq, head_dim);
    let mut pieces = Vec::with_capacity(in_heads * factor);
    for h in 0..in_heads {
        let slice = g.narrow_(x, 2, h, 1);
        for _ in 0..factor {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, 2)
}

fn activation(g: &mut Graph, kind: Activation, x: NodeId) -> NodeId {
    let s = g.shape(x).clone();
    g.activation(kind, x, s)
}

/// Register a `MatWeight` as a graph param. F32 takes the same path
/// as `param()` (with a [in, out] transpose so the matmul convention
/// matches). Packed registers as a U8 byte tensor + records the
/// scheme/shape in `packed`. Returns `(node, scheme_or_none, in, out)`.
fn proj_mat(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    packed: &mut PackedParams,
    name: &str,
    weight: &MatWeight,
    expected_in: usize,
    expected_out: usize,
) -> NodeId {
    match weight {
        MatWeight::F32(data) => {
            assert_eq!(
                data.len(),
                expected_in * expected_out,
                "proj_mat F32 {name}: len {} != in {expected_in} * out {expected_out}",
                data.len()
            );
            param(
                g,
                params,
                name,
                &transpose_2d(data, expected_out, expected_in),
                &[expected_in, expected_out],
            )
        }
        MatWeight::Packed {
            key,
            scheme,
            shape,
        } => {
            // Total byte count = elements × bytes-per-block /
            // block-size. We can compute it from scheme + shape.
            let n_elements: usize = shape.iter().product();
            let bytes_per_block = scheme.gguf_block_bytes() as usize;
            let block_size = scheme.gguf_block_size() as usize;
            assert!(
                n_elements.is_multiple_of(block_size),
                "proj_mat packed {name}: {n_elements} elems not aligned to \
                 block {block_size} for {scheme:?}"
            );
            let n_blocks = n_elements / block_size;
            let total_bytes = n_blocks * bytes_per_block;
            let id = g.param(name, Shape::new(&[total_bytes], DType::U8));
            packed.insert(
                name.to_string(),
                (key.clone(), *scheme, vec![expected_out, expected_in]),
            );
            id
        }
    }
}

/// Emit either `MatMul` (F32 weights) or `DequantMatMul` (packed),
/// based on whether `proj_mat` saw a packed source for `weight_node`.
/// The caller passes the *expected* out_shape (post-matmul); the
/// packed path needs it because DequantMatMul shape can't be
/// inferred from the U8 bytes alone.
fn emit_proj(
    g: &mut Graph,
    input: NodeId,
    weight_node: NodeId,
    weight_src: &MatWeight,
    out_shape: Shape,
) -> NodeId {
    match weight_src {
        MatWeight::F32(_) => g.mm(input, weight_node),
        MatWeight::Packed { scheme, .. } => g.add_node(
            Op::DequantMatMul { scheme: *scheme },
            vec![input, weight_node],
            out_shape,
        ),
    }
}

fn scalar_const(g: &mut Graph, value: f32) -> NodeId {
    // Encode the scalar f32 as a 4-byte Constant payload.
    let bytes = value.to_le_bytes().to_vec();
    g.add_node(
        Op::Constant { data: bytes },
        vec![],
        Shape::new(&[1], DType::F32),
    )
}

fn param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: &[f32],
    shape: &[usize],
) -> NodeId {
    register_param(g, params, name, data.to_vec(), Shape::new(shape, DType::F32))
}

fn register_param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    data: Vec<f32>,
    shape: Shape,
) -> NodeId {
    let id = g.param(name, shape);
    params.insert(name.to_string(), data);
    id
}

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

fn name(il: usize, suffix: &str) -> String {
    format!("blk.{il}.{suffix}")
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        data.len(),
        rows * cols,
        "transpose_2d: len {} != rows {rows} * cols {cols}",
        data.len()
    );
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
