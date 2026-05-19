// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! IR-lowered detector decoder.
//!
//! The per-layer compute (self-attn, text cross-attn, image cross-attn
//! with explicit boxRPB add, FFN) is expressed as an `rlx_ir::Graph`,
//! compiled once per layer on the requested device. Box-refinement /
//! sineembed / presence-token concatenation stay in Rust because they're
//! iterative and small.
//!
//! Custom mask in the IR's `Op::Attention` only supports key-padding
//! masks. Image cross-attention with boxRPB adds a per-head, per-query,
//! per-key bias, so we lower it manually as matmul → scale → add → softmax
//! → matmul instead of routing through `Op::Attention`.

use super::detector_decoder::{Mlp2, Mlp3, Sam3DecoderLayerWeights, Sam3DecoderWeights};
use anyhow::{Result, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::shape;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

const D_MODEL: usize = 256;
const DIM_FF: usize = 2048;
const N_HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / N_HEADS;
const NUM_QUERIES: usize = 200;
const N_LAYERS: usize = 6;

fn split_qkv(w_t: &[f32], e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut wq = vec![0f32; e * e];
    let mut wk = vec![0f32; e * e];
    let mut wv = vec![0f32; e * e];
    for i in 0..e {
        for j in 0..e {
            wq[i * e + j] = w_t[i * 3 * e + j];
            wk[i * e + j] = w_t[i * 3 * e + e + j];
            wv[i * e + j] = w_t[i * 3 * e + 2 * e + j];
        }
    }
    (wq, wk, wv)
}

fn add_w(g: &mut Graph, params: &mut HashMap<String, Vec<f32>>, name: &str, data: Vec<f32>, shape: Shape) -> NodeId {
    let id = g.param(name, shape);
    params.insert(name.to_string(), data);
    id
}

fn linear_bias(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: NodeId,
    w: Vec<f32>,
    b: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
) -> NodeId {
    let f = DType::F32;
    let w_id = add_w(g, params, &format!("{name}.w"), w, Shape::new(&[in_dim, out_dim], f));
    let b_id = add_w(g, params, &format!("{name}.b"), b, Shape::new(&[out_dim], f));
    let cur = g.shape(input).clone();
    let mut out_dims: Vec<usize> = cur.dims().iter().map(|d| d.unwrap_static()).collect();
    *out_dims.last_mut().unwrap() = out_dim;
    g.fused_matmul_bias_act(input, w_id, b_id, None, Shape::new(&out_dims, f))
}

/// Build the boxRPB MLP + outer-add subgraph. Takes log-normed deltas
/// (cheap geometry computed on host) and emits the `[B, H, nq+1, hw]`
/// additive log-bias used by the image cross-attention. Replaces the
/// host-side `boxrpb_log_full_into` for the per-layer hot path.
///
/// `deltas_x: [B, nq, w, 2]`, `deltas_y: [B, nq, h, 2]` → bias.
#[allow(clippy::too_many_arguments)]
fn build_boxrpb_subgraph(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    boxrpb_x: &Mlp2,
    boxrpb_y: &Mlp2,
    deltas_x: NodeId,
    deltas_y: NodeId,
    batch: usize,
    nq: usize,
    nh: usize,
    h: usize,
    w: usize,
) -> NodeId {
    use rlx_ir::infer::GraphExt;
    let f = DType::F32;
    let hidden_x = boxrpb_x.hidden;
    let hidden_y = boxrpb_y.hidden;
    assert_eq!(boxrpb_x.in_dim, 2);
    assert_eq!(boxrpb_y.in_dim, 2);
    assert_eq!(boxrpb_x.out_dim, nh);
    assert_eq!(boxrpb_y.out_dim, nh);

    // X branch: [B, nq, w, 2] → flat [B*nq*w, 2] → MLP → [B, nq, w, nh] → [B, nh, nq, w]
    let dx_flat = g.reshape_(deltas_x, vec![(batch * nq * w) as i64, 2]);
    let dx_h_w = add_w(g, params, "boxrpb_x.w0", boxrpb_x.w0_t.clone(),
                       Shape::new(&[2, hidden_x], f));
    let dx_h_b = add_w(g, params, "boxrpb_x.b0", boxrpb_x.b0.clone(),
                       Shape::new(&[hidden_x], f));
    let dx_h = g.fused_matmul_bias_act(
        dx_flat, dx_h_w, dx_h_b, Some(Activation::Relu),
        Shape::new(&[batch * nq * w, hidden_x], f),
    );
    let dx_o_w = add_w(g, params, "boxrpb_x.w1", boxrpb_x.w1_t.clone(),
                       Shape::new(&[hidden_x, nh], f));
    let dx_o_b = add_w(g, params, "boxrpb_x.b1", boxrpb_x.b1.clone(),
                       Shape::new(&[nh], f));
    let dx_o = g.fused_matmul_bias_act(
        dx_h, dx_o_w, dx_o_b, None,
        Shape::new(&[batch * nq * w, nh], f),
    );
    // [B*nq*w, nh] → [B, nq, w, nh] → [B, nh, nq, w]
    let dx_4d = g.reshape_(dx_o, vec![batch as i64, nq as i64, w as i64, nh as i64]);
    let dx_perm = g.transpose_(dx_4d, vec![0, 3, 1, 2]);
    // Reshape to [B, nh, nq, 1, w] for broadcast outer-add.
    let dx_bc = g.reshape_(dx_perm, vec![batch as i64, nh as i64, nq as i64, 1, w as i64]);

    // Y branch (symmetric).
    let dy_flat = g.reshape_(deltas_y, vec![(batch * nq * h) as i64, 2]);
    let dy_h_w = add_w(g, params, "boxrpb_y.w0", boxrpb_y.w0_t.clone(),
                       Shape::new(&[2, hidden_y], f));
    let dy_h_b = add_w(g, params, "boxrpb_y.b0", boxrpb_y.b0.clone(),
                       Shape::new(&[hidden_y], f));
    let dy_h = g.fused_matmul_bias_act(
        dy_flat, dy_h_w, dy_h_b, Some(Activation::Relu),
        Shape::new(&[batch * nq * h, hidden_y], f),
    );
    let dy_o_w = add_w(g, params, "boxrpb_y.w1", boxrpb_y.w1_t.clone(),
                       Shape::new(&[hidden_y, nh], f));
    let dy_o_b = add_w(g, params, "boxrpb_y.b1", boxrpb_y.b1.clone(),
                       Shape::new(&[nh], f));
    let dy_o = g.fused_matmul_bias_act(
        dy_h, dy_o_w, dy_o_b, None,
        Shape::new(&[batch * nq * h, nh], f),
    );
    let dy_4d = g.reshape_(dy_o, vec![batch as i64, nq as i64, h as i64, nh as i64]);
    let dy_perm = g.transpose_(dy_4d, vec![0, 3, 1, 2]);
    // Reshape to [B, nh, nq, h, 1] for broadcast outer-add.
    let dy_bc = g.reshape_(dy_perm, vec![batch as i64, nh as i64, nq as i64, h as i64, 1]);

    // Outer add: [B,nh,nq,1,w] + [B,nh,nq,h,1] → [B,nh,nq,h,w]
    let rpb_q = g.binary(
        BinaryOp::Add, dx_bc, dy_bc,
        Shape::new(&[batch, nh, nq, h, w], f),
    );
    let rpb_q_flat = g.reshape_(rpb_q, vec![batch as i64, nh as i64, nq as i64, (h * w) as i64]);

    // Prepend the zero presence row → [B, nh, lq, hw].
    let hw = h * w;
    let lq = nq + 1;
    let zero_pres = add_w(
        g, params,
        "rpb_zero_presence",
        vec![0f32; batch * nh * hw],
        Shape::new(&[batch, nh, 1, hw], f),
    );
    g.concat(vec![zero_pres, rpb_q_flat], 2, Shape::new(&[batch, nh, lq, hw], f))
}

/// Build the per-layer decoder graph.
///
/// Inputs:
///   * `tgt`          `[B, nq, D]`
///   * `query_pos`    `[B, nq, D]`         (sineembed(ref_boxes) → ref_point_head)
///   * `presence`     `[B, 1, D]`
///   * `memory`       `[B, hw, D]`
///   * `memory_pos`   `[B, hw, D]`
///   * `text`         `[B, seq, D]`
///   * `text_kpm_inv` `[B, seq]`           (1 = valid, 0 = PAD)
///   * `deltas_x`     `[B, nq, w, 2]`      (log-normed coord deltas, host-cheap)
///   * `deltas_y`     `[B, nq, h, 2]`
///
/// The boxRPB MLPs + outer-add live inside the graph so per-call host
/// work shrinks from 5×(MLP forward + outer-add over 16M floats) to
/// 5×(geometry-only delta computation over ~30K floats), and the
/// per-layer upload drops from a 16MB rpb_bias tensor to two ~115KB
/// delta tensors.
///
/// Outputs (in this order):
///   * `new_tgt`      `[B, nq, D]`
///   * `new_presence` `[B, 1, D]`
///   * `out_norm`     `[B, nq, D]` — `norm(new_tgt)`, used by host for box delta + intermediate.
#[allow(clippy::too_many_arguments)]
fn build_layer_graph(
    layer: &Sam3DecoderLayerWeights,
    boxrpb_x: &Mlp2,
    boxrpb_y: &Mlp2,
    norm_w: &[f32],
    norm_b: &[f32],
    batch: usize,
    h: usize,
    w: usize,
    seq: usize,
    use_bias_attn: bool,
    boxrpb_in_ir: bool,
) -> (Graph, HashMap<String, Vec<f32>>) {
    let hw = h * w;
    let mut g = Graph::new("sam3_dec_layer");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let d = D_MODEL;
    let nh = N_HEADS;
    let dh = HEAD_DIM;
    let nq = NUM_QUERIES;
    let lq = nq + 1;

    let tgt = g.input("tgt", Shape::new(&[batch, nq, d], f));
    let query_pos = g.input("query_pos", Shape::new(&[batch, nq, d], f));
    let presence = g.input("presence", Shape::new(&[batch, 1, d], f));
    let memory = g.input("memory", Shape::new(&[batch, hw, d], f));
    let memory_pos = g.input("memory_pos", Shape::new(&[batch, hw, d], f));
    let text = g.input("text", Shape::new(&[batch, seq, d], f));
    let text_kpm_inv = g.input("text_kpm_inv", Shape::new(&[batch, seq], f));
    // boxRPB feeds either as in-graph MLP (GPU backends — host saves a
    // 16MB upload per layer × 5 layers per call) or as a pre-computed
    // `rpb_bias` input (CPU — host BLAS beats the IR's K=2 GEMM for the
    // tiny boxRPB MLP shapes).
    let (rpb_bias, _deltas_x_id, _deltas_y_id) = if boxrpb_in_ir {
        let dx = g.input("deltas_x", Shape::new(&[batch, nq, w, 2], f));
        let dy = g.input("deltas_y", Shape::new(&[batch, nq, h, 2], f));
        let r = build_boxrpb_subgraph(
            &mut g, &mut params,
            boxrpb_x, boxrpb_y,
            dx, dy,
            batch, nq, nh, h, w,
        );
        (r, Some(dx), Some(dy))
    } else {
        let r = g.input("rpb_bias", Shape::new(&[batch, nh, lq, hw], f));
        (r, None, None)
    };

    // ── Self-attention (prepend presence) ─────────────────────────────
    // Concat presence + tgt → sa_x  [B, nq+1, D]
    // Pos: zeros for presence + query_pos for nq → sa_pos [B, nq+1, D]
    let sa_x = g.concat(vec![presence, tgt], 1, Shape::new(&[batch, lq, d], f));
    // Zero pos for presence: build a zero tensor [B, 1, D].
    // We don't have a Zero op; use a constant via param with zeroed data.
    let zero_pos = add_w(
        &mut g,
        &mut params,
        "zero_presence_pos",
        vec![0f32; batch * d],
        Shape::new(&[batch, 1, d], f),
    );
    let sa_pos = g.concat(vec![zero_pos, query_pos], 1, Shape::new(&[batch, lq, d], f));
    let sa_qk = g.binary(BinaryOp::Add, sa_x, sa_pos, Shape::new(&[batch, lq, d], f));

    let (wq, wk, wv) = split_qkv(&layer.self_attn_in_w_t, d);
    let bq = layer.self_attn_in_b[0..d].to_vec();
    let bk = layer.self_attn_in_b[d..2 * d].to_vec();
    let bv = layer.self_attn_in_b[2 * d..3 * d].to_vec();
    let q_sa = linear_bias(&mut g, &mut params, "sa.q", sa_qk, wq, bq, d, d);
    let k_sa = linear_bias(&mut g, &mut params, "sa.k", sa_qk, wk, bk, d, d);
    let v_sa = linear_bias(&mut g, &mut params, "sa.v", sa_x, wv, bv, d, d);
    let sa_attn = g.attention_kind(
        q_sa, k_sa, v_sa,
        nh, dh, MaskKind::None,
        shape::attention_shape(g.shape(q_sa)),
    );
    let sa_proj = linear_bias(
        &mut g, &mut params, "sa.out",
        sa_attn,
        layer.self_attn_out_w_t.clone(),
        layer.self_attn_out_b.clone(),
        d, d,
    );
    let sa_res = g.binary(BinaryOp::Add, sa_x, sa_proj, Shape::new(&[batch, lq, d], f));
    // norm2 (post self-attn norm).
    let n2_w = add_w(&mut g, &mut params, "norm2.w", layer.norm2_w.clone(), Shape::new(&[d], f));
    let n2_b = add_w(&mut g, &mut params, "norm2.b", layer.norm2_b.clone(), Shape::new(&[d], f));
    let sa_normed = g.layer_norm(sa_res, n2_w, n2_b, -1, 1e-5, Shape::new(&[batch, lq, d], f));
    // Match host decoder topology: split presence off here, run text-CA
    // on the queries only, then re-concat for image-CA. The host gets
    // intermediate parity at 2.4e-6 with this — diverging from upstream
    // which keeps presence in text-CA, but matching the production path.
    let presence_after_sa = g.narrow_(sa_normed, 1, 0, 1);
    let queries_after_sa = g.narrow_(sa_normed, 1, 1, nq);

    // ── Text cross-attention (queries only) ───────────────────────────
    let q_text_in = g.binary(BinaryOp::Add, queries_after_sa, query_pos, Shape::new(&[batch, nq, d], f));
    let (wqc, wkc, wvc) = split_qkv(&layer.ca_text_in_w_t, d);
    let bqc = layer.ca_text_in_b[0..d].to_vec();
    let bkc = layer.ca_text_in_b[d..2 * d].to_vec();
    let bvc = layer.ca_text_in_b[2 * d..3 * d].to_vec();
    let q_text = linear_bias(&mut g, &mut params, "ca_text.q", q_text_in, wqc, bqc, d, d);
    let k_text = linear_bias(&mut g, &mut params, "ca_text.k", text, wkc, bkc, d, d);
    let v_text = linear_bias(&mut g, &mut params, "ca_text.v", text, wvc, bvc, d, d);
    let ca_text_attn = g.attention(
        q_text, k_text, v_text, text_kpm_inv,
        nh, dh,
        shape::attention_shape(g.shape(q_text)),
    );
    let ca_text_proj = linear_bias(
        &mut g, &mut params, "ca_text.out",
        ca_text_attn,
        layer.ca_text_out_w_t.clone(),
        layer.ca_text_out_b.clone(),
        d, d,
    );
    let after_ca_text_res = g.binary(BinaryOp::Add, queries_after_sa, ca_text_proj, Shape::new(&[batch, nq, d], f));
    let cat_w = add_w(&mut g, &mut params, "catext_norm.w", layer.catext_norm_w.clone(), Shape::new(&[d], f));
    let cat_b = add_w(&mut g, &mut params, "catext_norm.b", layer.catext_norm_b.clone(), Shape::new(&[d], f));
    let after_ca_text = g.layer_norm(after_ca_text_res, cat_w, cat_b, -1, 1e-5, Shape::new(&[batch, nq, d], f));

    // ── Image cross-attention (re-concat presence) ────────────────────
    let ca_in = g.concat(vec![presence_after_sa, after_ca_text], 1, Shape::new(&[batch, lq, d], f));
    let ca_q_in = g.binary(BinaryOp::Add, ca_in, sa_pos, Shape::new(&[batch, lq, d], f));
    let k_mem_in = g.binary(BinaryOp::Add, memory, memory_pos, Shape::new(&[batch, hw, d], f));

    let (wqi, wki, wvi) = split_qkv(&layer.cross_attn_in_w_t, d);
    let bqi = layer.cross_attn_in_b[0..d].to_vec();
    let bki = layer.cross_attn_in_b[d..2 * d].to_vec();
    let bvi = layer.cross_attn_in_b[2 * d..3 * d].to_vec();
    let q_img = linear_bias(&mut g, &mut params, "ca_img.q", ca_q_in, wqi, bqi, d, d);
    let k_img = linear_bias(&mut g, &mut params, "ca_img.k", k_mem_in, wki, bki, d, d);
    let v_img = linear_bias(&mut g, &mut params, "ca_img.v", memory, wvi, bvi, d, d);

    // CPU/MLX: route through Op::Attention(MaskKind::Bias) — the additive
    // boxRPB tensor is added inside the kernel's per-head loop, sharing
    // the fast NEON/par_for path. Metal's MSL SDPA kernel doesn't
    // accept a bias tensor yet, so for Metal we fall back to the manual
    // matmul+scale+add+softmax+matmul decomposition.
    let attn_flat = if use_bias_attn {
        g.attention_bias(
            q_img, k_img, v_img, rpb_bias,
            nh, dh,
            shape::attention_shape(g.shape(q_img)),
        )
    } else {
        let q_4d = g.reshape_(q_img, vec![batch as i64, lq as i64, nh as i64, dh as i64]);
        let q_perm = g.transpose_(q_4d, vec![0, 2, 1, 3]);
        let k_4d = g.reshape_(k_img, vec![batch as i64, hw as i64, nh as i64, dh as i64]);
        let k_perm = g.transpose_(k_4d, vec![0, 2, 1, 3]);
        let v_4d = g.reshape_(v_img, vec![batch as i64, hw as i64, nh as i64, dh as i64]);
        let v_perm = g.transpose_(v_4d, vec![0, 2, 1, 3]);
        let k_t = g.transpose_(k_perm, vec![0, 1, 3, 2]);
        let scores = g.matmul(q_perm, k_t, Shape::new(&[batch, nh, lq, hw], f));
        let scale_val = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let scale_node = add_w(&mut g, &mut params, "img.scale", vec![scale_val], Shape::new(&[1], f));
        let scores_scaled = g.binary(BinaryOp::Mul, scores, scale_node, Shape::new(&[batch, nh, lq, hw], f));
        let scores_biased = g.binary(BinaryOp::Add, scores_scaled, rpb_bias, Shape::new(&[batch, nh, lq, hw], f));
        let probs = g.softmax(scores_biased, -1, Shape::new(&[batch, nh, lq, hw], f));
        let attn_out = g.matmul(probs, v_perm, Shape::new(&[batch, nh, lq, dh], f));
        let attn_perm = g.transpose_(attn_out, vec![0, 2, 1, 3]);
        g.reshape_(attn_perm, vec![batch as i64, lq as i64, d as i64])
    };
    let ca_img_proj = linear_bias(
        &mut g, &mut params, "ca_img.out",
        attn_flat,
        layer.cross_attn_out_w_t.clone(),
        layer.cross_attn_out_b.clone(),
        d, d,
    );
    let ca_img_res = g.binary(BinaryOp::Add, ca_in, ca_img_proj, Shape::new(&[batch, lq, d], f));
    let n1_w = add_w(&mut g, &mut params, "norm1.w", layer.norm1_w.clone(), Shape::new(&[d], f));
    let n1_b = add_w(&mut g, &mut params, "norm1.b", layer.norm1_b.clone(), Shape::new(&[d], f));
    let after_ca_img = g.layer_norm(ca_img_res, n1_w, n1_b, -1, 1e-5, Shape::new(&[batch, lq, d], f));

    // ── FFN ───────────────────────────────────────────────────────────
    let ff1 = linear_bias(
        &mut g, &mut params, "ffn.fc1",
        after_ca_img,
        layer.linear1_w_t.clone(),
        layer.linear1_b.clone(),
        d, DIM_FF,
    );
    let relud = g.activation(Activation::Relu, ff1, Shape::new(&[batch, lq, DIM_FF], f));
    let ff2 = linear_bias(
        &mut g, &mut params, "ffn.fc2",
        relud,
        layer.linear2_w_t.clone(),
        layer.linear2_b.clone(),
        DIM_FF, d,
    );
    let ffn_res = g.binary(BinaryOp::Add, after_ca_img, ff2, Shape::new(&[batch, lq, d], f));
    let n3_w = add_w(&mut g, &mut params, "norm3.w", layer.norm3_w.clone(), Shape::new(&[d], f));
    let n3_b = add_w(&mut g, &mut params, "norm3.b", layer.norm3_b.clone(), Shape::new(&[d], f));
    let after_ffn = g.layer_norm(ffn_res, n3_w, n3_b, -1, 1e-5, Shape::new(&[batch, lq, d], f));

    // Split outputs.
    let new_presence = g.narrow_(after_ffn, 1, 0, 1);
    let new_tgt = g.narrow_(after_ffn, 1, 1, nq);

    // Compute out_norm = norm(new_tgt) for box refinement.
    let dec_norm_w = add_w(&mut g, &mut params, "dec.norm.w", norm_w.to_vec(), Shape::new(&[d], f));
    let dec_norm_b = add_w(&mut g, &mut params, "dec.norm.b", norm_b.to_vec(), Shape::new(&[d], f));
    let out_norm = g.layer_norm(new_tgt, dec_norm_w, dec_norm_b, -1, 1e-5, Shape::new(&[batch, nq, d], f));

    g.set_outputs(vec![new_tgt, new_presence, out_norm]);
    let _ = (q_img, k_img, v_img, ca_img_proj);
    (g, params)
}

/// Compile-once-per-layer decoder, runnable across many frames.
pub struct Sam3CompiledDecoder {
    layers: Vec<CompiledGraph>,
    bbox_embed: Mlp3,
    ref_point_head: Mlp2,
    boxrpb_x: Mlp2,
    boxrpb_y: Mlp2,
    initial_query_embed: Vec<f32>,
    initial_reference_points: Vec<f32>,
    cached_layer0_query_pos: Vec<f32>,
    /// Layer-0 cached inputs for the boxRPB IR subgraph path (GPU
    /// backends). Constant geometry, ~115KB each.
    cached_layer0_deltas_x: Option<Vec<f32>>,
    cached_layer0_deltas_y: Option<Vec<f32>>,
    /// Layer-0 cached `rpb_bias [B, H, lq, hw]` for the host-MLP path
    /// (CPU backend). Constant boxRPB tensor, ~66MB.
    cached_layer0_rpb: Option<Vec<f32>>,
    cached_initial_ref_boxes: Vec<f32>,
    boxrpb_in_ir: bool,
    presence_token: Vec<f32>,
    presence_head: Mlp3,
    presence_norm_w: Vec<f32>,
    presence_norm_b: Vec<f32>,
    /// Per-call delta scratch (geometry only — the MLP forward and
    /// outer-add run inside the IR graph for GPU backends).
    scratch_deltas_x: Vec<f32>,
    scratch_deltas_y: Vec<f32>,
    /// Host-MLP path scratch buffers (CPU backend). Allocated only
    /// when boxrpb_in_ir is false.
    scratch_rpb: Option<Vec<f32>>,
    scratch_dx_thq: Option<Vec<f32>>,
    scratch_dy_thq: Option<Vec<f32>>,
    scratch_boxrpb_x_hidden: Option<Vec<f32>>,
    scratch_boxrpb_y_hidden: Option<Vec<f32>>,
    scratch_boxrpb_x_feats: Option<Vec<f32>>,
    scratch_boxrpb_y_feats: Option<Vec<f32>>,
    /// Scratch for ref_point_head sineembed + MLP intermediates.
    scratch_sine: Vec<f32>,
    scratch_rph_hidden: Vec<f32>,
    /// Output of ref_point_head MLP for layers 1..N (layer 0 uses cache).
    scratch_query_pos: Vec<f32>,
    /// Scratch for box-refinement `mlp3_forward(bbox_embed)`.
    scratch_bbox_h0: Vec<f32>,
    scratch_bbox_h1: Vec<f32>,
    scratch_bbox_out: Vec<f32>,
    pub batch: usize,
    pub hw: usize,
    pub seq: usize,
}

impl Sam3CompiledDecoder {
    pub fn new(
        weights: &Sam3DecoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        ensure!(weights.loaded, "decoder weights not loaded");
        let nq = NUM_QUERIES;
        let d = D_MODEL;
        let h_w = (hw as f64).sqrt().round() as usize;
        ensure!(
            h_w * h_w == hw,
            "boxRPB cache requires square spatial grid; got hw={hw}"
        );
        let mut layers = Vec::with_capacity(N_LAYERS);
        // Metal: opt-in to bias-mask SDPA via env var. The default
        // routes Metal through the MPSGraph manual-decomp because the
        // bias-aware SDPA kernels (sdpa_long, sdpa_fa_f32) currently
        // produce incorrect output — needs debugging.
        let use_bias_attn = if matches!(device, Device::Metal) {
            std::env::var("RLX_SAM3_METAL_BIAS_SDPA").is_ok()
        } else {
            true
        };
        // MLX: do boxRPB in the IR graph (saves the 16MB×5 per-call
        // rpb_bias upload — verified ~3% gain to 89.9ms, matching
        // PyTorch MPS). CPU: host BLAS beats the IR's K=2 GEMM for
        // the tiny boxRPB MLP. Metal: the 5D broadcast outer-add in
        // the boxRPB subgraph currently produces incorrect output on
        // the Metal IR backend — keep it on the host until that's
        // debugged.
        let boxrpb_in_ir = matches!(device, Device::Mlx);
        for layer in &weights.layers {
            let (g, params) = build_layer_graph(
                layer,
                &weights.boxrpb_x, &weights.boxrpb_y,
                &weights.norm_w, &weights.norm_b,
                batch, h_w, h_w, seq, use_bias_attn,
                boxrpb_in_ir,
            );
            let session = Session::new(device);
            let mut compiled = session.compile(g);
            for (name, data) in &params {
                compiled.set_param(name, data);
            }
            layers.push(compiled);
        }
        // Precompute layer-0 inputs that depend only on constant model
        // weights (initial reference_points → cached query_pos and
        // boxRPB deltas). The boxRPB MLP+outer-add now runs inside the
        // graph, so we only cache the cheap geometry deltas.
        let mut cached_initial_ref_boxes = vec![0f32; batch * nq * 4];
        for b in 0..batch {
            for q in 0..nq {
                for k in 0..4 {
                    let v = weights.reference_points[q * 4 + k];
                    cached_initial_ref_boxes[(b * nq + q) * 4 + k] = sigmoid(v);
                }
            }
        }
        let sine = sineembed_4d(&cached_initial_ref_boxes, batch, nq, d);
        let cached_layer0_query_pos =
            mlp2_forward(&weights.ref_point_head, &sine, batch * nq)?;
        let lq = nq + 1;
        let nh = N_HEADS;
        let (cached_layer0_deltas_x, cached_layer0_deltas_y, cached_layer0_rpb) = if boxrpb_in_ir {
            let mut dx = vec![0f32; batch * nq * h_w * 2];
            let mut dy = vec![0f32; batch * nq * h_w * 2];
            compute_deltas_into(&cached_initial_ref_boxes, batch, nq, h_w, h_w, &mut dx, &mut dy);
            (Some(dx), Some(dy), None)
        } else {
            let rpb = boxrpb_log_full(
                &weights.boxrpb_x, &weights.boxrpb_y,
                &cached_initial_ref_boxes,
                batch, nq, h_w, h_w,
            )?;
            (None, None, Some(rpb))
        };
        Ok(Self {
            layers,
            bbox_embed: weights.bbox_embed.clone(),
            ref_point_head: weights.ref_point_head.clone(),
            boxrpb_x: weights.boxrpb_x.clone(),
            boxrpb_y: weights.boxrpb_y.clone(),
            initial_query_embed: weights.query_embed.clone(),
            initial_reference_points: weights.reference_points.clone(),
            cached_layer0_query_pos,
            cached_layer0_deltas_x,
            cached_layer0_deltas_y,
            cached_layer0_rpb,
            cached_initial_ref_boxes,
            boxrpb_in_ir,
            presence_token: weights.presence_token.clone(),
            presence_head: weights.presence_token_head.clone(),
            presence_norm_w: weights.presence_token_out_norm_w.clone(),
            presence_norm_b: weights.presence_token_out_norm_b.clone(),
            scratch_deltas_x: if boxrpb_in_ir { vec![0f32; batch * nq * h_w * 2] } else { Vec::new() },
            scratch_deltas_y: if boxrpb_in_ir { vec![0f32; batch * nq * h_w * 2] } else { Vec::new() },
            scratch_rpb: (!boxrpb_in_ir).then(|| vec![0f32; batch * nh * lq * hw]),
            scratch_dx_thq: (!boxrpb_in_ir).then(|| vec![0f32; nh * nq * h_w]),
            scratch_dy_thq: (!boxrpb_in_ir).then(|| vec![0f32; nh * nq * h_w]),
            scratch_boxrpb_x_hidden: (!boxrpb_in_ir).then(|| vec![0f32; nq * h_w * weights.boxrpb_x.hidden]),
            scratch_boxrpb_y_hidden: (!boxrpb_in_ir).then(|| vec![0f32; nq * h_w * weights.boxrpb_y.hidden]),
            scratch_boxrpb_x_feats: (!boxrpb_in_ir).then(|| vec![0f32; nq * h_w * weights.boxrpb_x.out_dim]),
            scratch_boxrpb_y_feats: (!boxrpb_in_ir).then(|| vec![0f32; nq * h_w * weights.boxrpb_y.out_dim]),
            scratch_sine: vec![0f32; batch * nq * 2 * d],
            scratch_rph_hidden: vec![0f32; batch * nq * weights.ref_point_head.hidden],
            scratch_query_pos: vec![0f32; batch * nq * weights.ref_point_head.out_dim],
            scratch_bbox_h0: vec![0f32; batch * nq * weights.bbox_embed.hidden],
            scratch_bbox_h1: vec![0f32; batch * nq * weights.bbox_embed.hidden],
            scratch_bbox_out: vec![0f32; batch * nq * weights.bbox_embed.out_dim],
            batch,
            hw,
            seq,
        })
    }

    /// Run the decoder. Inputs are batch-first: `memory [B, hw, D]`,
    /// `memory_pos [B, hw, D]`, `text [B, seq, D]` (note: text is
    /// batch-first here, not seq-first), `text_kpm` (1 = PAD).
    pub fn run(
        &mut self,
        memory: &[f32],
        memory_pos: &[f32],
        text_seq_first: &[f32],
        text_kpm: &[u8],
        h: usize,
        w: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let hw = h * w;
        ensure!(hw == self.hw);
        let batch = self.batch;
        let nq = NUM_QUERIES;
        let d = D_MODEL;
        let nh = N_HEADS;
        let lq = nq + 1;
        let seq = self.seq;

        // Initial tgt = query_embed expanded to batch.
        let mut tgt = vec![0f32; batch * nq * d];
        for b in 0..batch {
            tgt[b * nq * d..(b + 1) * nq * d].copy_from_slice(&self.initial_query_embed);
        }
        // Initial ref_boxes = sigmoid(reference_points).
        let mut ref_boxes = vec![0f32; batch * nq * 4];
        for b in 0..batch {
            for q in 0..nq {
                for k in 0..4 {
                    let v = self.initial_reference_points[q * 4 + k];
                    ref_boxes[(b * nq + q) * 4 + k] = sigmoid(v);
                }
            }
        }
        let mut presence = vec![0f32; batch * d];
        for b in 0..batch {
            presence[b * d..(b + 1) * d].copy_from_slice(&self.presence_token);
        }

        // Text → batch-first.
        let mut text_bf = vec![0f32; batch * seq * d];
        for b in 0..batch {
            for l in 0..seq {
                let s = (l * batch + b) * d;
                let dst = (b * seq + l) * d;
                text_bf[dst..dst + d].copy_from_slice(&text_seq_first[s..s + d]);
            }
        }
        let text_kpm_inv: Vec<f32> = text_kpm.iter().map(|&v| if v == 0 { 1.0 } else { 0.0 }).collect();

        let mut intermediate = Vec::with_capacity(N_LAYERS);
        let mut intermediate_ref_boxes = Vec::with_capacity(N_LAYERS);
        intermediate_ref_boxes.push(ref_boxes.clone());
        let mut presence_logits = Vec::with_capacity(N_LAYERS);

        let profile = std::env::var("RLX_SAM3_PROFILE").is_ok();
        let mut t_qpos = 0u128;
        let mut t_rpb = 0u128;
        let mut t_graph = 0u128;
        let mut t_box = 0u128;
        let mut t_other = 0u128;
        for li in 0..N_LAYERS {
            let tq = std::time::Instant::now();
            // Compute query_pos = ref_point_head(sineembed(ref_boxes)).
            // Layer 0's ref_boxes is the constant `sigmoid(reference_points)`,
            // so its query_pos and boxRPB are precomputed once at
            // construction and reused per call.
            // For layer 0, use the cached query_pos slice directly and
            // the cached rpb buffer. For other layers, recompute into
            // pre-allocated scratch buffers so we don't malloc 33MB/layer.
            let query_pos_slice: &[f32];
            let rpb_slice: &[f32];
            let deltas_x_slice: &[f32];
            let deltas_y_slice: &[f32];
            if li == 0 {
                query_pos_slice = &self.cached_layer0_query_pos;
                if self.boxrpb_in_ir {
                    deltas_x_slice = self.cached_layer0_deltas_x.as_ref().unwrap();
                    deltas_y_slice = self.cached_layer0_deltas_y.as_ref().unwrap();
                    rpb_slice = &[];
                } else {
                    rpb_slice = self.cached_layer0_rpb.as_ref().unwrap();
                    deltas_x_slice = &[];
                    deltas_y_slice = &[];
                }
            } else {
                sineembed_4d_into(&ref_boxes, batch, nq, d, &mut self.scratch_sine);
                mlp2_forward_into(
                    &self.ref_point_head,
                    &self.scratch_sine,
                    batch * nq,
                    &mut self.scratch_rph_hidden,
                    &mut self.scratch_query_pos,
                );
                query_pos_slice = &self.scratch_query_pos;
                if self.boxrpb_in_ir {
                    compute_deltas_into(
                        &ref_boxes, batch, nq, h, w,
                        &mut self.scratch_deltas_x,
                        &mut self.scratch_deltas_y,
                    );
                    deltas_x_slice = &self.scratch_deltas_x;
                    deltas_y_slice = &self.scratch_deltas_y;
                    rpb_slice = &[];
                } else {
                    let mut host_deltas_x = vec![0f32; nq * w * 2];
                    let mut host_deltas_y = vec![0f32; nq * h * 2];
                    boxrpb_log_full_into(
                        &self.boxrpb_x, &self.boxrpb_y, &ref_boxes,
                        batch, nq, h, w,
                        self.scratch_rpb.as_mut().unwrap(),
                        self.scratch_dx_thq.as_mut().unwrap(),
                        self.scratch_dy_thq.as_mut().unwrap(),
                        &mut host_deltas_x,
                        &mut host_deltas_y,
                        self.scratch_boxrpb_x_hidden.as_mut().unwrap(),
                        self.scratch_boxrpb_y_hidden.as_mut().unwrap(),
                        self.scratch_boxrpb_x_feats.as_mut().unwrap(),
                        self.scratch_boxrpb_y_feats.as_mut().unwrap(),
                    )?;
                    rpb_slice = self.scratch_rpb.as_ref().unwrap();
                    deltas_x_slice = &[];
                    deltas_y_slice = &[];
                }
            }
            if profile { t_qpos += tq.elapsed().as_micros(); }

            let tr = std::time::Instant::now();
            if profile { t_rpb += tr.elapsed().as_micros(); }

            // Run graph.
            let tg = std::time::Instant::now();
            let outputs = if self.boxrpb_in_ir {
                self.layers[li].run(&[
                    ("tgt", tgt.as_slice()),
                    ("query_pos", query_pos_slice),
                    ("presence", presence.as_slice()),
                    ("memory", memory),
                    ("memory_pos", memory_pos),
                    ("text", text_bf.as_slice()),
                    ("text_kpm_inv", text_kpm_inv.as_slice()),
                    ("deltas_x", deltas_x_slice),
                    ("deltas_y", deltas_y_slice),
                ])
            } else {
                self.layers[li].run(&[
                    ("tgt", tgt.as_slice()),
                    ("query_pos", query_pos_slice),
                    ("presence", presence.as_slice()),
                    ("memory", memory),
                    ("memory_pos", memory_pos),
                    ("text", text_bf.as_slice()),
                    ("text_kpm_inv", text_kpm_inv.as_slice()),
                    ("rpb_bias", rpb_slice),
                ])
            };
            if profile { t_graph += tg.elapsed().as_micros(); }
            ensure!(outputs.len() == 3, "decoder layer expected 3 outputs");
            tgt = outputs[0].clone();
            presence = outputs[1].clone();
            let out_norm = outputs[2].clone();

            let tb = std::time::Instant::now();
            // Box refinement: delta = bbox_embed(out_norm); ref = sigmoid(inv_sig(ref) + delta).
            mlp3_forward_into(
                &self.bbox_embed, &out_norm, batch * nq,
                &mut self.scratch_bbox_h0,
                &mut self.scratch_bbox_h1,
                &mut self.scratch_bbox_out,
            );
            let delta: &[f32] = &self.scratch_bbox_out;
            if profile { t_box += tb.elapsed().as_micros(); }
            let to = std::time::Instant::now();
            let _ = to;
            let _ = &mut t_other;
            let mut new_ref = vec![0f32; batch * nq * 4];
            for q in 0..nq {
                for b in 0..batch {
                    let cur = &ref_boxes[(b * nq + q) * 4..(b * nq + q + 1) * 4];
                    let dl = &delta[(b * nq + q) * 4..(b * nq + q + 1) * 4];
                    for k in 0..4 {
                        new_ref[(b * nq + q) * 4 + k] = sigmoid(inv_sigmoid(cur[k]) + dl[k]);
                    }
                }
            }
            ref_boxes = new_ref;
            if li != N_LAYERS - 1 {
                intermediate_ref_boxes.push(ref_boxes.clone());
            }

            // Intermediate output in seq-first convention.
            let mut out_seq_first = vec![0f32; nq * batch * d];
            for q in 0..nq {
                for b in 0..batch {
                    let src = (b * nq + q) * d;
                    let dst = (q * batch + b) * d;
                    out_seq_first[dst..dst + d].copy_from_slice(&out_norm[src..src + d]);
                }
            }
            intermediate.push(out_seq_first);

            // Presence logits.
            let p_norm = layer_norm_host(&presence, &self.presence_norm_w, &self.presence_norm_b, d);
            let p_logit = mlp3_forward(&self.presence_head, &p_norm, batch)?;
            presence_logits.push(p_logit);
        }
        if profile {
            let to_ms = |us: u128| us as f32 / 1000.0;
            eprintln!(
                "  decoder per-stage (6 layers total): qpos={:.1}ms  rpb={:.1}ms  graph={:.1}ms  box={:.1}ms",
                to_ms(t_qpos), to_ms(t_rpb), to_ms(t_graph), to_ms(t_box)
            );
        }

        // Stack.
        let mut int_stack = vec![0f32; N_LAYERS * nq * batch * d];
        for (li, l) in intermediate.iter().enumerate() {
            int_stack[li * nq * batch * d..(li + 1) * nq * batch * d].copy_from_slice(l);
        }
        let mut ref_stack = vec![0f32; N_LAYERS * nq * batch * 4];
        for (li, r) in intermediate_ref_boxes.iter().enumerate() {
            ref_stack[li * nq * batch * 4..(li + 1) * nq * batch * 4].copy_from_slice(r);
        }
        let mut presence_stack = vec![0f32; N_LAYERS * batch];
        for (li, p) in presence_logits.iter().enumerate() {
            for b in 0..batch {
                presence_stack[li * batch + b] = p[b];
            }
        }
        let _ = nh;
        let _ = lq;
        Ok((int_stack, ref_stack, presence_stack, presence))
    }
}

// ── Host helpers (sineembed, boxRPB, mlp, sigmoid) ─────────────────────

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn inv_sigmoid(x: f32) -> f32 {
    let eps = 1e-3f32;
    let x = x.clamp(0.0, 1.0).max(eps).min(1.0 - eps);
    (x / (1.0 - x)).ln()
}

fn layer_norm_host(x: &[f32], gamma: &[f32], beta: &[f32], dim: usize) -> Vec<f32> {
    let rows = x.len() / dim;
    let mut out = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (*v - mean).powi(2)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for c in 0..dim {
            out[r * dim + c] = (row[c] - mean) * inv * gamma[c] + beta[c];
        }
    }
    out
}

fn mlp2_forward(mlp: &Mlp2, x: &[f32], rows: usize) -> Result<Vec<f32>> {
    let h = matmul_bias_relu(x, &mlp.w0_t, &mlp.b0, rows, mlp.in_dim, mlp.hidden);
    Ok(matmul_bias(&h, &mlp.w1_t, &mlp.b1, rows, mlp.hidden, mlp.out_dim))
}

/// In-place mlp2: `out = w1·relu(w0·x + b0) + b1`. Caller provides the
/// hidden scratch buffer and output buffer — no allocation in the hot
/// path. First layer uses fused matmul+bias+relu epilogue.
fn mlp2_forward_into(
    mlp: &Mlp2,
    x: &[f32],
    rows: usize,
    hidden: &mut [f32],
    out: &mut [f32],
) {
    rlx_cpu::blas::sgemm_bias_epilogue(
        x, &mlp.w0_t, &mlp.b0, hidden,
        rows, mlp.in_dim, mlp.hidden,
        |v| if v < 0.0 { 0.0 } else { v },
    );
    rlx_cpu::blas::sgemm_bias(hidden, &mlp.w1_t, &mlp.b1, out, rows, mlp.hidden, mlp.out_dim);
}

fn mlp3_forward(mlp: &Mlp3, x: &[f32], rows: usize) -> Result<Vec<f32>> {
    let h = matmul_bias_relu(x, &mlp.w0_t, &mlp.b0, rows, mlp.in_dim, mlp.hidden);
    let h = matmul_bias_relu(&h, &mlp.w1_t, &mlp.b1, rows, mlp.hidden, mlp.hidden);
    Ok(matmul_bias(&h, &mlp.w2_t, &mlp.b2, rows, mlp.hidden, mlp.out_dim))
}

fn mlp3_forward_into(
    mlp: &Mlp3,
    x: &[f32],
    rows: usize,
    h0: &mut [f32],
    h1: &mut [f32],
    out: &mut [f32],
) {
    let relu = |v: f32| if v < 0.0 { 0.0 } else { v };
    rlx_cpu::blas::sgemm_bias_epilogue(
        x, &mlp.w0_t, &mlp.b0, h0,
        rows, mlp.in_dim, mlp.hidden, relu,
    );
    rlx_cpu::blas::sgemm_bias_epilogue(
        h0, &mlp.w1_t, &mlp.b1, h1,
        rows, mlp.hidden, mlp.hidden, relu,
    );
    rlx_cpu::blas::sgemm_bias(h1, &mlp.w2_t, &mlp.b2, out, rows, mlp.hidden, mlp.out_dim);
}

fn matmul_bias(x: &[f32], w_t: &[f32], b: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * n];
    rlx_cpu::blas::sgemm_bias(x, w_t, b, &mut out, rows, k, n);
    out
}

fn matmul_bias_relu(x: &[f32], w_t: &[f32], b: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = matmul_bias(x, w_t, b, rows, k, n);
    for v in out.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    out
}

fn sineembed_4d(pos: &[f32], batch: usize, nq: usize, d_model: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * nq * 2 * d_model];
    sineembed_4d_into(pos, batch, nq, d_model, &mut out);
    out
}

fn sineembed_4d_into(pos: &[f32], batch: usize, nq: usize, d_model: usize, out: &mut [f32]) {
    let half = d_model / 2;
    let scale = 2.0 * std::f32::consts::PI;
    let mut dim_t = vec![0.0f32; half];
    for i in 0..half {
        let exp = 2.0 * ((i / 2) as f32) / half as f32;
        dim_t[i] = 10000.0f32.powf(exp);
    }
    debug_assert_eq!(out.len(), batch * nq * 2 * d_model);
    for b in 0..batch {
        for q in 0..nq {
            let p = &pos[(b * nq + q) * 4..(b * nq + q + 1) * 4];
            let vals = [p[1] * scale, p[0] * scale, p[2] * scale, p[3] * scale];
            let base = (b * nq + q) * 2 * d_model;
            for axis in 0..4 {
                let slot = base + axis * half;
                for i in 0..half {
                    let theta = vals[axis] / dim_t[i];
                    out[slot + i] = if i % 2 == 0 { theta.sin() } else { theta.cos() };
                }
            }
        }
    }
}

/// Owning version that allocates; kept for the construct-time cache
/// where we run it once. Hot path uses `boxrpb_log_full_into` with a
/// pre-allocated scratch buffer.
fn boxrpb_log_full(
    boxrpb_x: &Mlp2,
    boxrpb_y: &Mlp2,
    reference_boxes: &[f32],
    batch: usize,
    nq: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>> {
    let nh = N_HEADS;
    let lq = nq + 1;
    let mut out = vec![0f32; batch * nh * lq * h * w];
    let mut dx_thq = vec![0f32; nh * nq * w];
    let mut dy_thq = vec![0f32; nh * nq * h];
    let mut deltas_x = vec![0f32; nq * w * 2];
    let mut deltas_y = vec![0f32; nq * h * 2];
    let mut hidden_x = vec![0f32; nq * w * boxrpb_x.hidden];
    let mut hidden_y = vec![0f32; nq * h * boxrpb_y.hidden];
    let mut feats_x = vec![0f32; nq * w * boxrpb_x.out_dim];
    let mut feats_y = vec![0f32; nq * h * boxrpb_y.out_dim];
    boxrpb_log_full_into(
        boxrpb_x, boxrpb_y, reference_boxes,
        batch, nq, h, w,
        &mut out, &mut dx_thq, &mut dy_thq,
        &mut deltas_x, &mut deltas_y,
        &mut hidden_x, &mut hidden_y,
        &mut feats_x, &mut feats_y,
    )?;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn boxrpb_log_full_into(
    boxrpb_x: &Mlp2,
    boxrpb_y: &Mlp2,
    reference_boxes: &[f32],
    batch: usize,
    nq: usize,
    h: usize,
    w: usize,
    out: &mut [f32],
    dx_thq: &mut [f32],
    dy_thq: &mut [f32],
    deltas_x: &mut [f32],
    deltas_y: &mut [f32],
    hidden_x: &mut [f32],
    hidden_y: &mut [f32],
    feats_x: &mut [f32],
    feats_y: &mut [f32],
) -> Result<()> {
    let nh = N_HEADS;
    let lq = nq + 1;
    debug_assert_eq!(out.len(), batch * nh * lq * h * w);
    debug_assert_eq!(dx_thq.len(), nh * nq * w);
    debug_assert_eq!(dy_thq.len(), nh * nq * h);
    debug_assert_eq!(deltas_x.len(), nq * w * 2);
    debug_assert_eq!(deltas_y.len(), nq * h * 2);
    debug_assert_eq!(hidden_x.len(), nq * w * boxrpb_x.hidden);
    debug_assert_eq!(hidden_y.len(), nq * h * boxrpb_y.hidden);
    debug_assert_eq!(feats_x.len(), nq * w * boxrpb_x.out_dim);
    debug_assert_eq!(feats_y.len(), nq * h * boxrpb_y.out_dim);
    // Zero the presence rows once — non-presence rows get overwritten.
    for head in 0..nh {
        for b in 0..batch {
            let off = b * nh * lq * h * w + head * lq * h * w;
            // Presence row at lq=0
            for i in 0..h * w {
                out[off + i] = 0.0;
            }
        }
    }
    let coords_h: Vec<f32> = (0..h).map(|y| y as f32 / h as f32).collect();
    let coords_w: Vec<f32> = (0..w).map(|x| x as f32 / w as f32).collect();

    for b in 0..batch {
        for q in 0..nq {
            let p = &reference_boxes[(b * nq + q) * 4..(b * nq + q + 1) * 4];
            let (cx, cy, bw, bh) = (p[0], p[1], p[2], p[3]);
            let x0 = cx - 0.5 * bw;
            let x1 = cx + 0.5 * bw;
            let y0 = cy - 0.5 * bh;
            let y1 = cy + 0.5 * bh;
            for xi in 0..w {
                let dx0 = (coords_w[xi] - x0) * 8.0;
                let dx1 = (coords_w[xi] - x1) * 8.0;
                deltas_x[(q * w + xi) * 2] = log_norm(dx0);
                deltas_x[(q * w + xi) * 2 + 1] = log_norm(dx1);
            }
            for yi in 0..h {
                let dy0 = (coords_h[yi] - y0) * 8.0;
                let dy1 = (coords_h[yi] - y1) * 8.0;
                deltas_y[(q * h + yi) * 2] = log_norm(dy0);
                deltas_y[(q * h + yi) * 2 + 1] = log_norm(dy1);
            }
        }
        mlp2_forward_into(boxrpb_x, deltas_x, nq * w, hidden_x, feats_x);
        mlp2_forward_into(boxrpb_y, deltas_y, nq * h, hidden_y, feats_y);
        let dx_feats: &[f32] = feats_x;
        let dy_feats: &[f32] = feats_y;
        // Transpose dx/dy from [pos, head] to [head, q, pos] so the
        // outer-add per (head, q) reads contiguous slices.
        for q in 0..nq {
            for xi in 0..w {
                let src_base = (q * w + xi) * nh;
                for head in 0..nh {
                    dx_thq[(head * nq + q) * w + xi] = dx_feats[src_base + head];
                }
            }
            for yi in 0..h {
                let src_base = (q * h + yi) * nh;
                for head in 0..nh {
                    dy_thq[(head * nq + q) * h + yi] = dy_feats[src_base + head];
                }
            }
        }
        let base = b * nh * lq * h * w;
        let total = nh * nq;
        let out_ptr = out.as_mut_ptr() as usize;
        let dx_ptr = dx_thq.as_ptr() as usize;
        let dy_ptr = dy_thq.as_ptr() as usize;
        rlx_cpu::pool::par_for(total, 8, &|off, cnt| unsafe {
            for idx in off..off + cnt {
                let head = idx / nq;
                let q = idx % nq;
                let dst = (out_ptr as *mut f32)
                    .add(base + (head * lq + 1 + q) * h * w);
                let dx_row = std::slice::from_raw_parts(
                    (dx_ptr as *const f32).add((head * nq + q) * w),
                    w,
                );
                let dy_row = std::slice::from_raw_parts(
                    (dy_ptr as *const f32).add((head * nq + q) * h),
                    h,
                );
                for y in 0..h {
                    let dy = dy_row[y];
                    let row_dst = dst.add(y * w);
                    for x in 0..w {
                        *row_dst.add(x) = dy + dx_row[x];
                    }
                }
            }
        });
    }
    Ok(())
}

fn log_norm(v: f32) -> f32 {
    let s = if v < 0.0 { -1.0 } else { 1.0 };
    s * (v.abs() + 1.0).log2() / 8.0f32.log2()
}

/// Geometry-only delta computation. Output layout matches the IR
/// `deltas_x [B, nq, w, 2]` / `deltas_y [B, nq, h, 2]` inputs that
/// feed the boxRPB subgraph: per (batch, query, spatial-coord), pair
/// of `log_norm((coord - left)*8)` and `log_norm((coord - right)*8)`.
fn compute_deltas_into(
    reference_boxes: &[f32],
    batch: usize,
    nq: usize,
    h: usize,
    w: usize,
    deltas_x: &mut [f32],
    deltas_y: &mut [f32],
) {
    debug_assert_eq!(deltas_x.len(), batch * nq * w * 2);
    debug_assert_eq!(deltas_y.len(), batch * nq * h * 2);
    let coords_h: Vec<f32> = (0..h).map(|y| y as f32 / h as f32).collect();
    let coords_w: Vec<f32> = (0..w).map(|x| x as f32 / w as f32).collect();
    for b in 0..batch {
        for q in 0..nq {
            let p = &reference_boxes[(b * nq + q) * 4..(b * nq + q + 1) * 4];
            let (cx, cy, bw, bh) = (p[0], p[1], p[2], p[3]);
            let x0 = cx - 0.5 * bw;
            let x1 = cx + 0.5 * bw;
            let y0 = cy - 0.5 * bh;
            let y1 = cy + 0.5 * bh;
            let dx_off = ((b * nq + q) * w) * 2;
            for xi in 0..w {
                let dx0 = (coords_w[xi] - x0) * 8.0;
                let dx1 = (coords_w[xi] - x1) * 8.0;
                deltas_x[dx_off + xi * 2] = log_norm(dx0);
                deltas_x[dx_off + xi * 2 + 1] = log_norm(dx1);
            }
            let dy_off = ((b * nq + q) * h) * 2;
            for yi in 0..h {
                let dy0 = (coords_h[yi] - y0) * 8.0;
                let dy1 = (coords_h[yi] - y1) * 8.0;
                deltas_y[dy_off + yi * 2] = log_norm(dy0);
                deltas_y[dy_off + yi * 2 + 1] = log_norm(dy1);
            }
        }
    }
}
