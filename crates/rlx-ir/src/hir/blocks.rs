// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Composite HIR block lowering — expands block ops into MIR primitives.

use crate::dynamic::sym;
use crate::hir::LowerError;
use crate::infer::GraphExt;
use crate::op::{Activation, MaskKind};
use crate::shape::{self, Dim};
use crate::{DType, Graph, NodeId, Shape};

/// Lower [`super::HirOp::LlamaDecoderBlock`].
pub fn lower_llama_decoder_block(
    g: &mut Graph,
    inputs: &[NodeId],
    num_heads: usize,
    head_dim: usize,
    num_kv_heads: usize,
    eps: f32,
    mask: MaskKind,
    out_shape: Shape,
) -> Result<NodeId, LowerError> {
    let need_mask = matches!(mask, MaskKind::Custom | MaskKind::Bias);
    let expected = if need_mask { 15 } else { 14 };
    if inputs.len() != expected {
        return Err(LowerError::WrongInputCount {
            op: "LlamaDecoderBlock",
            expected: if need_mask { "15" } else { "14" },
            got: inputs.len(),
        });
    }
    let x = inputs[0];
    let ln1_g = inputs[1];
    let ln1_b = inputs[2];
    let q_w = inputs[3];
    let k_w = inputs[4];
    let v_w = inputs[5];
    let o_w = inputs[6];
    let ln2_g = inputs[7];
    let ln2_b = inputs[8];
    let gate_w = inputs[9];
    let up_w = inputs[10];
    let down_w = inputs[11];
    let cos = inputs[12];
    let sin = inputs[13];

    let normed_in = g.rms_norm(x, ln1_g, ln1_b, eps);
    let q = g.mm(normed_in, q_w);
    let k = g.mm(normed_in, k_w);
    let v = g.mm(normed_in, v_w);
    let q_rope = g.rope(q, cos, sin, head_dim);
    let k_rope = g.rope(k, cos, sin, head_dim);

    let group = num_heads / num_kv_heads;
    let k_rep = repeat_kv(g, k_rope, num_kv_heads, head_dim, group);
    let v_rep = repeat_kv(g, v, num_kv_heads, head_dim, group);

    let attn_shape = shape::attention_shape(g.shape(q_rope));
    let attn = match mask {
        MaskKind::Custom => g.attention(
            q_rope, k_rep, v_rep, inputs[14], num_heads, head_dim, attn_shape,
        ),
        MaskKind::Bias => g.attention_bias(
            q_rope, k_rep, v_rep, inputs[14], num_heads, head_dim, attn_shape,
        ),
        other => g.attention_kind(q_rope, k_rep, v_rep, num_heads, head_dim, other, attn_shape),
    };
    let attn_out = g.mm(attn, o_w);
    let post_attn = g.add(x, attn_out);

    let normed_post = g.rms_norm(post_attn, ln2_g, ln2_b, eps);
    let gate = g.mm(normed_post, gate_w);
    let up = g.mm(normed_post, up_w);
    let gate_act = g.silu(gate);
    let swiglu = g.mul(gate_act, up);
    let ffn_out = g.mm(swiglu, down_w);
    let h = g.add(post_attn, ffn_out);
    debug_assert_eq!(g.shape(h), &out_shape);
    Ok(h)
}

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

/// Lower [`super::HirOp::Qwen35MtpHead`].
///
/// Inputs (29): `h_pre_norm`, `input_ids`, `cos`, `sin`, `last_token_idx`,
/// `embed_w`, `hnorm_{w,b}`, `enorm_{w,b}`, `eh_w`, full-attn
/// `{attn_norm, q_gate, k, v, q_norm, k_norm, o, post_norm, ffn gate/up/down}`,
/// `head_norm_{w,b}`, `lm_head_w`.
#[allow(clippy::too_many_arguments)]
pub fn lower_qwen35_mtp_head(
    g: &mut Graph,
    inputs: &[NodeId],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    n_embd: usize,
    n_ff: usize,
    mtp_vocab: usize,
    eps: f32,
    out_shape: Shape,
) -> Result<NodeId, LowerError> {
    const EXPECTED: usize = 29;
    if inputs.len() != EXPECTED {
        return Err(LowerError::WrongInputCount {
            op: "Qwen35MtpHead",
            expected: "29",
            got: inputs.len(),
        });
    }
    let h_pre_norm = inputs[0];
    let input_ids = inputs[1];
    let cos = inputs[2];
    let sin = inputs[3];
    let last_token_idx = inputs[4];
    let embed_w = inputs[5];
    let hnorm_w = inputs[6];
    let hnorm_b = inputs[7];
    let enorm_w = inputs[8];
    let enorm_b = inputs[9];
    let eh_w = inputs[10];
    let attn_norm_w = inputs[11];
    let attn_norm_b = inputs[12];
    let q_gate_w = inputs[13];
    let k_w = inputs[14];
    let v_w = inputs[15];
    let q_norm_w = inputs[16];
    let q_norm_b = inputs[17];
    let k_norm_w = inputs[18];
    let k_norm_b = inputs[19];
    let o_w = inputs[20];
    let post_norm_w = inputs[21];
    let post_norm_b = inputs[22];
    let gate_w = inputs[23];
    let up_w = inputs[24];
    let down_w = inputs[25];
    let head_norm_w = inputs[26];
    let head_norm_b = inputs[27];
    let lm_head_w = inputs[28];

    let in_shape = g.shape(h_pre_norm);
    let batch = in_shape.dims()[0].unwrap_static();
    let seq_dim = in_shape.dims()[1];
    let dynamic_seq = !seq_dim.is_static();
    let seq = if dynamic_seq {
        0
    } else {
        seq_dim.unwrap_static()
    };
    let f = DType::F32;
    let _q_gate_cols = num_heads * head_dim * 2;
    let kv_cols = num_kv_heads * head_dim;
    let kv_dim = num_heads * head_dim;
    let rows = batch * seq.max(1);

    let h_normed = g.rms_norm(h_pre_norm, hnorm_w, hnorm_b, eps);
    let tok_embd = g.gather_(embed_w, input_ids, 0);
    let e_normed = g.rms_norm(tok_embd, enorm_w, enorm_b, eps);
    let concat = g.concat_(vec![e_normed, h_normed], 2);
    let concat_2d = g.reshape_(
        concat,
        if dynamic_seq {
            vec![-1, (2 * n_embd) as i64]
        } else {
            vec![rows as i64, (2 * n_embd) as i64]
        },
    );
    let cur_2d = g.mm(concat_2d, eh_w);
    let cur = g.reshape_(
        cur_2d,
        if dynamic_seq {
            vec![batch as i64, -1, n_embd as i64]
        } else {
            vec![batch as i64, seq as i64, n_embd as i64]
        },
    );

    // Full-attn block (causal, no KV cache).
    let x = g.rms_norm(cur, attn_norm_w, attn_norm_b, eps);
    let x_2d = g.reshape_(
        x,
        if dynamic_seq {
            vec![-1, n_embd as i64]
        } else {
            vec![rows as i64, n_embd as i64]
        },
    );
    let q_gate = g.mm(x_2d, q_gate_w);
    let q_gate_4d = g.reshape_(
        q_gate,
        if dynamic_seq {
            vec![batch as i64, -1, num_heads as i64, (head_dim * 2) as i64]
        } else {
            vec![
                batch as i64,
                seq as i64,
                num_heads as i64,
                (head_dim * 2) as i64,
            ]
        },
    );
    let q_heads = g.narrow_(q_gate_4d, 3, 0, head_dim);
    let gate_heads = g.narrow_(q_gate_4d, 3, head_dim, head_dim);
    let q_packed = g.reshape_(
        q_heads,
        if dynamic_seq {
            vec![batch as i64, -1, kv_dim as i64]
        } else {
            vec![batch as i64, seq as i64, kv_dim as i64]
        },
    );
    let gate_packed = g.reshape_(
        gate_heads,
        if dynamic_seq {
            vec![batch as i64, -1, kv_dim as i64]
        } else {
            vec![batch as i64, seq as i64, kv_dim as i64]
        },
    );

    let k_proj = g.mm(x_2d, k_w);
    let v_proj = g.mm(x_2d, v_w);
    let k_packed = g.reshape_(
        k_proj,
        if dynamic_seq {
            vec![batch as i64, -1, kv_cols as i64]
        } else {
            vec![batch as i64, seq as i64, kv_cols as i64]
        },
    );
    let v_packed = g.reshape_(
        v_proj,
        if dynamic_seq {
            vec![batch as i64, -1, kv_cols as i64]
        } else {
            vec![batch as i64, seq as i64, kv_cols as i64]
        },
    );

    let q_normed = per_head_rms_graph(
        g,
        q_packed,
        q_norm_w,
        q_norm_b,
        batch,
        seq,
        num_heads,
        head_dim,
        eps,
        dynamic_seq,
    );
    let k_normed = per_head_rms_graph(
        g,
        k_packed,
        k_norm_w,
        k_norm_b,
        batch,
        seq,
        num_kv_heads,
        head_dim,
        eps,
        dynamic_seq,
    );
    let q_rot = g.rope_n(q_normed, cos, sin, head_dim, n_rot);
    let k_rot = g.rope_n(k_normed, cos, sin, head_dim, n_rot);

    let group = num_heads / num_kv_heads;
    let k_full = if group == 1 {
        k_rot
    } else {
        repeat_heads_packed_graph(g, k_rot, batch, seq, num_kv_heads, head_dim, group)
    };
    let v_full = if group == 1 {
        v_packed
    } else {
        repeat_heads_packed_graph(g, v_packed, batch, seq, num_kv_heads, head_dim, group)
    };

    let gate_sig = g.activation(
        Activation::Sigmoid,
        gate_packed,
        g.shape(gate_packed).clone(),
    );
    let attn_shape = if dynamic_seq {
        Shape::from_dims(
            &[
                Dim::Static(batch),
                Dim::Dynamic(sym::SEQ),
                Dim::Static(kv_dim),
            ],
            f,
        )
    } else {
        Shape::new(&[batch, seq, kv_dim], f)
    };
    let attn_out = g.add_node(
        crate::ops::attention::attention_kind_op(num_heads, head_dim, MaskKind::Causal, None, None),
        vec![q_rot, k_full, v_full],
        attn_shape,
    );
    let attn_gated = g.mul(attn_out, gate_sig);
    let attn_gated_2d = g.reshape_(
        attn_gated,
        if dynamic_seq {
            vec![-1, kv_dim as i64]
        } else {
            vec![rows as i64, kv_dim as i64]
        },
    );
    let attn_out_proj = g.mm(attn_gated_2d, o_w);
    let attn_out_3d = g.reshape_(
        attn_out_proj,
        if dynamic_seq {
            vec![batch as i64, -1, n_embd as i64]
        } else {
            vec![batch as i64, seq as i64, n_embd as i64]
        },
    );
    let h_post_attn = g.add(cur, attn_out_3d);

    let h_ffn = swiglu_ffn_graph(
        g,
        h_post_attn,
        post_norm_w,
        post_norm_b,
        gate_w,
        up_w,
        down_w,
        batch,
        seq,
        n_embd,
        n_ff,
        eps,
        dynamic_seq,
    );

    let idx_2d = g.reshape_(last_token_idx, vec![batch as i64, 1]);
    let last = g.gather_(h_ffn, idx_2d, 1);
    let last_norm = g.rms_norm(last, head_norm_w, head_norm_b, eps);
    let logits = g.mm(last_norm, lm_head_w);
    debug_assert_eq!(g.shape(logits), &out_shape);
    let _ = mtp_vocab;
    Ok(logits)
}

fn per_head_rms_graph(
    g: &mut Graph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    eps: f32,
    dynamic_seq: bool,
) -> NodeId {
    let r = g.reshape_(
        x,
        if dynamic_seq {
            vec![batch as i64, -1, heads as i64, head_dim as i64]
        } else {
            vec![batch as i64, seq as i64, heads as i64, head_dim as i64]
        },
    );
    let n = g.rms_norm(r, gamma, beta, eps);
    g.reshape_(
        n,
        if dynamic_seq {
            vec![batch as i64, -1, (heads * head_dim) as i64]
        } else {
            vec![batch as i64, seq as i64, (heads * head_dim) as i64]
        },
    )
}

fn repeat_heads_packed_graph(
    g: &mut Graph,
    x: NodeId,
    _batch: usize,
    _seq: usize,
    in_heads: usize,
    head_dim: usize,
    factor: usize,
) -> NodeId {
    let last_ax = g.shape(x).rank() - 1;
    let mut pieces = Vec::with_capacity(in_heads * factor);
    for h in 0..in_heads {
        let slice = g.narrow_(x, last_ax, h * head_dim, head_dim);
        for _ in 0..factor {
            pieces.push(slice);
        }
    }
    g.concat_(pieces, last_ax)
}

fn swiglu_ffn_graph(
    g: &mut Graph,
    h_in: NodeId,
    norm_w: NodeId,
    norm_b: NodeId,
    gate_w: NodeId,
    up_w: NodeId,
    down_w: NodeId,
    batch: usize,
    seq: usize,
    n_embd: usize,
    _n_ff: usize,
    eps: f32,
    dynamic_seq: bool,
) -> NodeId {
    let _f = DType::F32;
    let rows = batch * seq.max(1);
    let normed = g.rms_norm(h_in, norm_w, norm_b, eps);
    let h_2d = g.reshape_(
        normed,
        if dynamic_seq {
            vec![-1, n_embd as i64]
        } else {
            vec![rows as i64, n_embd as i64]
        },
    );
    let gate = g.mm(h_2d, gate_w);
    let up = g.mm(h_2d, up_w);
    let gate_act = g.silu(gate);
    let swiglu = g.mul(gate_act, up);
    let down = g.mm(swiglu, down_w);
    let ffn_out = g.reshape_(
        down,
        if dynamic_seq {
            vec![batch as i64, -1, n_embd as i64]
        } else {
            vec![batch as i64, seq as i64, n_embd as i64]
        },
    );
    g.add(h_in, ffn_out)
}
