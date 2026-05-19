// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! SAM v1 ViT image encoder graph builder.
//!
//! Mirrors `candle-transformers/src/models/segment_anything/image_encoder.rs`.
//! Decomposes attention into primitives (rlx-ir's `attention_` op is a
//! black box and can't host the inline rel-pos add SAM uses).
//!
//! Two attention modes:
//!   - **Global** (window_size == 0): full S = hw·hw attention. Used by
//!     blocks listed in `global_attn_indexes`.
//!   - **Windowed** (window_size > 0): pad spatial dims to a multiple
//!     of `window_size` via concat-with-zeros, reshape into
//!     `[B·nW, ws, ws, C]`, attention within each window, reverse the
//!     reshape, narrow off the padding.
//!
//! The neck (Conv2d 1×1 + LN2d + Conv2d 3×3 + LN2d → `[B, 256, hw, hw]`)
//! is *not* part of the graph yet — it lives in [`apply_neck_host`]
//! because rlx-ir has no f32 forward Conv2d. The neck is tiny compared
//! to the 12-layer encoder body, so the host-side overhead is
//! negligible.

use super::config::{SAM_EMBED_HW, SamEncoderConfig};
use super::preprocess::{SamPreprocessWeights, extract_preprocess_weights};
use crate::weight_map::WeightMap;
use anyhow::{Result, anyhow, ensure};
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// Build the SAM ViT-B/L/H image-encoder graph (body only — neck
/// runs host-side via [`apply_neck_host`]).
///
/// Input: `"hidden"` shape `[1, hw·hw, embed_dim]` — the caller flattens
/// the BHWC patch embeddings emitted by
/// [`crate::sam::preprocess::assemble_patch_tokens`].
///
/// Output: `[1, hw·hw, embed_dim]` — the post-block representation,
/// pre-neck. The neck takes this through 1×1 Conv → LN2d → 3×3 Conv →
/// LN2d on the host side.
pub fn build_sam_encoder_graph(
    cfg: &SamEncoderConfig,
    weights: &mut WeightMap,
) -> Result<(
    Graph,
    HashMap<String, Vec<f32>>,
    SamPreprocessWeights,
    NeckWeights,
)> {
    let mut g = Graph::new("sam_image_encoder");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    // Host-side preprocess weights (patch projection + abs pos embed).
    // Drain these *before* iterating blocks so the keys are gone when
    // we later assert the WeightMap is empty.
    let preprocess = extract_preprocess_weights(weights, cfg)?;

    let e = cfg.embed_dim;
    let nh = cfg.num_heads;
    let dh = cfg.head_dim();
    let scale = 1.0 / (dh as f32).sqrt();
    let eps = cfg.layer_norm_eps as f32;
    let hw = SAM_EMBED_HW;
    let s = hw * hw; // 64·64 = 4096

    // Input: pre-assembled patch tokens [1, 4096, E].
    let hidden_input = g.input("hidden", Shape::new(&[1, s, e], f));

    let mut x = hidden_input;
    for layer_idx in 0..cfg.depth {
        let lp = format!("image_encoder.blocks.{layer_idx}");
        let is_global = cfg.global_attn_indexes.contains(&layer_idx);
        let ws = if is_global { 0 } else { cfg.window_size };

        // ── Pre-LN1 ──
        let n1_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm1.weight"),
            false,
        )?;
        let n1_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm1.bias"),
            false,
        )?;
        let normed = g.ln(x, n1_g, n1_b, eps);

        // ── Attention (windowed or global) ──
        let attn_out = if ws == 0 {
            attention_global(
                &mut g,
                &mut params,
                weights,
                &lp,
                normed,
                e,
                nh,
                dh,
                scale,
                hw,
                cfg.use_rel_pos,
                cfg.qkv_bias,
            )?
        } else {
            attention_windowed(
                &mut g,
                &mut params,
                weights,
                &lp,
                normed,
                e,
                nh,
                dh,
                scale,
                hw,
                ws,
                cfg.use_rel_pos,
                cfg.qkv_bias,
            )?
        };

        // Residual
        x = g.add(x, attn_out);

        // ── Pre-LN2 + MLP (4× expansion, plain GELU) ──
        let n2_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm2.weight"),
            false,
        )?;
        let n2_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm2.bias"),
            false,
        )?;
        let normed2 = g.ln(x, n2_g, n2_b, eps);

        let fc1_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.lin1.weight"),
            true,
        )?;
        let fc1_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.lin1.bias"),
            false,
        )?;
        let fc2_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.lin2.weight"),
            true,
        )?;
        let fc2_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.lin2.bias"),
            false,
        )?;

        let up_mm = g.mm(normed2, fc1_w);
        let up = g.add(up_mm, fc1_b);
        // candle's `Activation::Gelu` dispatches to `Tensor::gelu_erf()`
        // — the exact erf form — for SAM's MlpBlock. Use the matching
        // erf kernel here.
        let act = g.gelu(up);
        let down_mm = g.mm(act, fc2_w);
        let ffn = g.add(down_mm, fc2_b);

        x = g.add(x, ffn);
    }

    g.set_outputs(vec![x]);

    // Pull the host-side neck weights so the caller can run them after.
    let neck_w = extract_neck_weights(weights, cfg)?;

    Ok((g, params, preprocess, neck_w))
}

/// Global-attention block: full self-attention over all `hw·hw` tokens.
#[allow(clippy::too_many_arguments)]
fn attention_global(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut WeightMap,
    lp: &str,
    x: NodeId, // [1, S, E]
    e: usize,
    nh: usize,
    dh: usize,
    scale: f32,
    hw: usize,
    use_rel_pos: bool,
    qkv_bias: bool,
) -> Result<NodeId> {
    let s = hw * hw;
    decomposed_attention(
        g,
        params,
        w,
        lp,
        x,
        e,
        nh,
        dh,
        scale,
        hw,
        hw,
        s,
        1,
        use_rel_pos,
        qkv_bias,
    )
}

/// Windowed-attention block: pad → partition into `nW = (hw_p/ws)²`
/// windows → attention within each window → reverse partition → crop.
#[allow(clippy::too_many_arguments)]
fn attention_windowed(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut WeightMap,
    lp: &str,
    x: NodeId, // [1, S, E] flat (= [1, hw, hw, E] BHWC, flattened)
    e: usize,
    nh: usize,
    dh: usize,
    scale: f32,
    hw: usize,
    ws: usize,
    use_rel_pos: bool,
    qkv_bias: bool,
) -> Result<NodeId> {
    // Restore spatial: [1, S, E] → [1, hw, hw, E]
    let bhwc = g.reshape_(x, vec![1, hw as i64, hw as i64, e as i64]);

    let pad = (ws - hw % ws) % ws;
    let hw_p = hw + pad;
    let n_win_per_side = hw_p / ws;
    let n_win = n_win_per_side * n_win_per_side;

    // Pad with concat-zeros along axes 1, 2.
    let padded = if pad > 0 {
        let z_h = pad_zero_param(g, params, &format!("{lp}.attn._pad_h"), &[1, pad, hw, e]);
        let p1 = g.concat_(vec![bhwc, z_h], 1); // [1, hw_p, hw, E]
        let z_w = pad_zero_param(g, params, &format!("{lp}.attn._pad_w"), &[1, hw_p, pad, e]);
        g.concat_(vec![p1, z_w], 2) // [1, hw_p, hw_p, E]
    } else {
        bhwc
    };

    // [1, hw_p, hw_p, E] → [1, nw, ws, nw, ws, E] → transpose(2,3)
    //   → [1, nw, nw, ws, ws, E] → reshape [nw², ws, ws, E]
    let reshaped = g.reshape_(
        padded,
        vec![
            1,
            n_win_per_side as i64,
            ws as i64,
            n_win_per_side as i64,
            ws as i64,
            e as i64,
        ],
    );
    let transposed = g.transpose_(reshaped, vec![0, 1, 3, 2, 4, 5]);
    let windowed = g.reshape_(
        transposed,
        vec![n_win as i64, ws as i64, ws as i64, e as i64],
    );
    // Flatten spatial for the attention: [nw², ws², E]
    let win_flat = g.reshape_(windowed, vec![n_win as i64, (ws * ws) as i64, e as i64]);

    // Run decomposed attention. Window has spatial dims (ws, ws);
    // sequence length S = ws·ws; batch dim = n_win.
    let attn_out = decomposed_attention(
        g,
        params,
        w,
        lp,
        win_flat,
        e,
        nh,
        dh,
        scale,
        ws,
        ws,
        ws * ws,
        n_win,
        use_rel_pos,
        qkv_bias,
    )?;
    // attn_out: [nw², ws·ws, E]

    // Reverse: [nw², ws², E] → [nw², ws, ws, E] → [1, nw, nw, ws, ws, E]
    //   → transpose(2,3) → [1, nw, ws, nw, ws, E] → [1, hw_p, hw_p, E]
    let un = g.reshape_(attn_out, vec![n_win as i64, ws as i64, ws as i64, e as i64]);
    let un = g.reshape_(
        un,
        vec![
            1,
            n_win_per_side as i64,
            n_win_per_side as i64,
            ws as i64,
            ws as i64,
            e as i64,
        ],
    );
    let un = g.transpose_(un, vec![0, 1, 3, 2, 4, 5]);
    let un = g.reshape_(un, vec![1, hw_p as i64, hw_p as i64, e as i64]);
    // Crop off the padding
    let un = if pad > 0 {
        let cropped_h = g.narrow_(un, 1, 0, hw);
        g.narrow_(cropped_h, 2, 0, hw)
    } else {
        un
    };
    // Flatten back to [1, S, E]
    Ok(g.reshape_(un, vec![1, (hw * hw) as i64, e as i64]))
}

/// Decomposed multi-head attention with optional decomposed rel_pos.
/// Input `[B, S, E]`; output `[B, S, E]`.
///
/// `h, w` are the spatial dims of the attention window (S = h·w).
/// For windowed attention `B = n_win`, `h = w = ws`. For global,
/// `B = 1`, `h = w = hw`.
#[allow(clippy::too_many_arguments)]
fn decomposed_attention(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    w: &mut WeightMap,
    lp: &str,
    x: NodeId, // [B, S, E]
    e: usize,
    nh: usize,
    dh: usize,
    scale: f32,
    h: usize,
    w_dim: usize,
    s: usize, // = h * w_dim
    b: usize,
    use_rel_pos: bool,
    qkv_bias: bool,
) -> Result<NodeId> {
    // 1) QKV projection. Bias param is loaded *before* the mm so its
    //    NodeId is lower — `FuseMatMulBiasAct` walks nodes in topo
    //    order and assumes the bias has been copied into the new id
    //    map before the matmul is rewritten.
    let qkv_w_node = load_p(g, params, w, &format!("{lp}.attn.qkv.weight"), true)?;
    let qkv_b_node = if qkv_bias {
        Some(load_p(g, params, w, &format!("{lp}.attn.qkv.bias"), false)?)
    } else {
        None
    };
    let qkv_mm = g.mm(x, qkv_w_node); // [B, S, 3E]
    let qkv = if let Some(b) = qkv_b_node {
        g.add(qkv_mm, b)
    } else {
        qkv_mm
    };

    // 2) Reshape & permute to [3, B·nh, S, dh].
    //    [B, S, 3E] → [B, S, 3, nh, dh] → permute(2,0,3,1,4) → [3, B, nh, S, dh]
    //    → reshape [3, B·nh, S, dh].
    let qkv5 = g.reshape_(qkv, vec![b as i64, s as i64, 3, nh as i64, dh as i64]);
    let qkv_perm = g.transpose_(qkv5, vec![2, 0, 3, 1, 4]); // [3, B, nh, S, dh]
    let qkv_flat = g.reshape_(qkv_perm, vec![3, (b * nh) as i64, s as i64, dh as i64]);
    let q = g.narrow_(qkv_flat, 0, 0, 1);
    let q = g.reshape_(q, vec![(b * nh) as i64, s as i64, dh as i64]);
    let k = g.narrow_(qkv_flat, 0, 1, 1);
    let k = g.reshape_(k, vec![(b * nh) as i64, s as i64, dh as i64]);
    let v = g.narrow_(qkv_flat, 0, 2, 1);
    let v = g.reshape_(v, vec![(b * nh) as i64, s as i64, dh as i64]);

    // 3) attn = (q * scale) @ k.T   shape [B·nh, S, S]
    let scale_node = scalar_param(g, params, &format!("{lp}.attn._scale"), scale);
    let q_scaled = g.mul(q, scale_node);
    let k_t = g.transpose_(k, vec![0, 2, 1]); // [B·nh, dh, S]
    let scores = g.mm(q_scaled, k_t); // [B·nh, S, S]

    // 4) Optionally add decomposed rel_pos.
    let scores = if use_rel_pos {
        // rel_pos_h: [2h-1, dh]  rel_pos_w: [2w-1, dh]
        // We pre-resolve get_rel_pos() host-side into r_h: [h, h, dh] and
        // r_w: [w, w, dh] indexed buffers (cheap, ≤ 27×27×64 elements).
        let (mut r_h_data, mut r_w_data) = extract_rel_pos(w, lp, h, w_dim, dh)?;
        // Bisect helpers:
        //   RLX_SAM_DEBUG_ZERO_RELPOS=1  zero both r_h and r_w
        //   RLX_SAM_DEBUG_ZERO_RELH=1    zero only r_h (keep rel_w)
        //   RLX_SAM_DEBUG_ZERO_RELW=1    zero only r_w (keep rel_h)
        if std::env::var("RLX_SAM_DEBUG_ZERO_RELPOS").is_ok() {
            r_h_data.iter_mut().for_each(|v| *v = 0.0);
            r_w_data.iter_mut().for_each(|v| *v = 0.0);
        }
        if std::env::var("RLX_SAM_DEBUG_ZERO_RELH").is_ok() {
            r_h_data.iter_mut().for_each(|v| *v = 0.0);
        }
        if std::env::var("RLX_SAM_DEBUG_ZERO_RELW").is_ok() {
            r_w_data.iter_mut().for_each(|v| *v = 0.0);
        }
        let r_h_node = const_param(
            g,
            params,
            &format!("{lp}.attn._rel_h_indexed"),
            &[h, h, dh],
            r_h_data,
        );
        let r_w_node = const_param(
            g,
            params,
            &format!("{lp}.attn._rel_w_indexed"),
            &[w_dim, w_dim, dh],
            r_w_data,
        );
        add_decomposed_rel_pos(g, scores, q, r_h_node, r_w_node, b, nh, h, w_dim, dh)?
    } else {
        scores
    };

    // 5) softmax over last axis
    let attn_w = g.sm(scores, -1);

    // 6) attn @ V → [B·nh, S, dh]
    let attn_v = g.mm(attn_w, v);

    // 7) Reverse the head split: [B·nh, S, dh] → [B, nh, S, dh] → [B, S, nh, dh] → [B, S, E]
    let reshaped = g.reshape_(attn_v, vec![b as i64, nh as i64, s as i64, dh as i64]);
    let perm = g.transpose_(reshaped, vec![0, 2, 1, 3]); // [B, S, nh, dh]
    let merged = g.reshape_(perm, vec![b as i64, s as i64, e as i64]);

    // 8) Output projection (always biased).
    let proj_w = load_p(g, params, w, &format!("{lp}.attn.proj.weight"), true)?;
    let proj_b = load_p(g, params, w, &format!("{lp}.attn.proj.bias"), false)?;
    let proj_mm = g.mm(merged, proj_w);
    Ok(g.add(proj_mm, proj_b))
}

/// Add decomposed relative positional bias to attention scores.
///
/// Math (per the SAM paper, candle's `add_decomposed_rel_pos`):
///   r_q = q.reshape(B·nh, h, w, dh)
///   rel_h[bhw,c] = sum_c r_q[bhw,c] · r_h_indexed[hq, hk, c]    → [B·nh, h, w, h]
///   rel_w[bhw,c] = sum_c r_q[bhw,c] · r_w_indexed[wq, wk, c]    → [B·nh, h, w, w]
///   scores += rel_h.unsqueeze(4) + rel_w.unsqueeze(3)           → [B·nh, h, w, h, w]
///   scores.reshape(B·nh, h·w, h·w)
#[allow(clippy::too_many_arguments)]
fn add_decomposed_rel_pos(
    g: &mut Graph,
    scores: NodeId, // [B·nh, S, S]
    q: NodeId,      // [B·nh, S, dh]
    r_h: NodeId,    // [h, h, dh]  (pre-indexed)
    r_w: NodeId,    // [w, w, dh]
    b: usize,
    nh: usize,
    h: usize,
    w: usize,
    dh: usize,
) -> Result<NodeId> {
    let bh = b * nh;
    // r_q: [bh, h, w, dh]
    let r_q = g.reshape_(q, vec![bh as i64, h as i64, w as i64, dh as i64]);

    // rel_h: "bhwc, hkc -> bhwk".
    // Unrolled-per-h_q: rlx-cpu's batched 3-D matmul gives subtly wrong
    // results in this exact shape regime, so we lower the einsum to
    // `h` independent 2-D matmuls (one per h_q index) and `g.concat_`
    // them back. Each per-h_q matmul is `[bh, w, dh] @ [dh, h_k]`,
    // which uses the well-tested flat sgemm path (rhs has no batch
    // dim, only the lhs does — that's the case the Sgemm flatten
    // trick was designed for).
    let mut rel_h_slices: Vec<NodeId> = Vec::with_capacity(h);
    for h_q in 0..h {
        // r_q at h_q: narrow axis 1, then squeeze.
        let rq_slice = g.narrow_(r_q, 1, h_q, 1); // [bh, 1, w, dh]
        let rq_slice = g.reshape_(rq_slice, vec![bh as i64, w as i64, dh as i64]);
        // r_h at h_q: narrow axis 0, then squeeze + transpose to [dh, h].
        let rh_slice = g.narrow_(r_h, 0, h_q, 1); // [1, h, dh]
        let rh_slice = g.reshape_(rh_slice, vec![h as i64, dh as i64]); // [h_k, dh]
        let rh_t = g.transpose_(rh_slice, vec![1, 0]); // [dh, h_k]
        let mm = g.mm(rq_slice, rh_t); // [bh, w, h_k]
        // Add a leading length-1 axis so we can concat into [bh, h, w, h_k].
        let mm5 = g.reshape_(mm, vec![bh as i64, 1, w as i64, h as i64]);
        rel_h_slices.push(mm5);
    }
    let rel_h_4d = g.concat_(rel_h_slices, 1); // [bh, h, w, h]

    // rel_w: same idea, w_q as the unrolled axis.
    let mut rel_w_slices: Vec<NodeId> = Vec::with_capacity(w);
    for w_q in 0..w {
        let rq_slice = g.narrow_(r_q, 2, w_q, 1); // [bh, h, 1, dh]
        let rq_slice = g.reshape_(rq_slice, vec![bh as i64, h as i64, dh as i64]);
        let rw_slice = g.narrow_(r_w, 0, w_q, 1); // [1, w, dh]
        let rw_slice = g.reshape_(rw_slice, vec![w as i64, dh as i64]); // [w_k, dh]
        let rw_t = g.transpose_(rw_slice, vec![1, 0]); // [dh, w_k]
        let mm = g.mm(rq_slice, rw_t); // [bh, h, w_k]
        let mm5 = g.reshape_(mm, vec![bh as i64, h as i64, 1, w as i64]);
        rel_w_slices.push(mm5);
    }
    let rel_w_4d = g.concat_(rel_w_slices, 2); // [bh, h, w, w]

    // Broadcast-add into the [bh, h, w, h, w] view of scores.
    //
    // History: rlx-cpu's BiasAdd misroute for mid-shape singletons is
    // now fixed (`is_trailing_bias_broadcast`), so CPU uses simple
    // unsqueeze+add. The rlx-metal BinaryBroadcast MSL kernel exists
    // but produces wrong results on the SAM rel_pos pattern (suspect:
    // setBytes alignment of inline `constant uint*` for ranks > 4 —
    // needs focused debugging). Until then, materialise both rel
    // tensors to the full output shape via `concat`-tile so the add
    // is a same-shape op and works on every backend.
    let scores_5d = g.reshape_(
        scores,
        vec![bh as i64, h as i64, w as i64, h as i64, w as i64],
    );
    let rel_h_5d = g.reshape_(rel_h_4d, vec![bh as i64, h as i64, w as i64, h as i64, 1]);
    let rel_h_tiled = {
        let mut copies = Vec::with_capacity(w);
        for _ in 0..w {
            copies.push(rel_h_5d);
        }
        g.concat_(copies, 4) // [bh, h, w, h, w]
    };
    let rel_w_5d = g.reshape_(rel_w_4d, vec![bh as i64, h as i64, w as i64, 1, w as i64]);
    let rel_w_tiled = {
        let mut copies = Vec::with_capacity(h);
        for _ in 0..h {
            copies.push(rel_w_5d);
        }
        g.concat_(copies, 3) // [bh, h, w, h, w]
    };
    let s1 = g.add(scores_5d, rel_h_tiled);
    let s2 = g.add(s1, rel_w_tiled);
    Ok(g.reshape_(s2, vec![bh as i64, (h * w) as i64, (h * w) as i64]))
}

/// Resolve candle's `get_rel_pos()` host-side into per-axis bias
/// tables of shape `[q_size, k_size, dh]` (here q_size == k_size).
///
/// Stored `rel_pos_h` has shape `[2·max(q,k)-1, dh]`; we gather along
/// axis 0 using `relative_coords[i,j] = i - j + (k-1)` (since q==k,
/// scale factors collapse to 1).
fn extract_rel_pos(
    weights: &mut WeightMap,
    lp: &str,
    h: usize,
    w: usize,
    dh: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let (rel_h_raw, rh_shape) = weights.take(&format!("{lp}.attn.rel_pos_h"))?;
    let (rel_w_raw, rw_shape) = weights.take(&format!("{lp}.attn.rel_pos_w"))?;
    ensure!(
        rh_shape == vec![2 * h - 1, dh],
        "{lp}.attn.rel_pos_h expected [{}, {dh}], got {rh_shape:?}",
        2 * h - 1
    );
    ensure!(
        rw_shape == vec![2 * w - 1, dh],
        "{lp}.attn.rel_pos_w expected [{}, {dh}], got {rw_shape:?}",
        2 * w - 1
    );

    let mut r_h = vec![0f32; h * h * dh];
    for q in 0..h {
        for k in 0..h {
            let idx = (q as isize - k as isize + (h as isize - 1)) as usize;
            let src = &rel_h_raw[idx * dh..(idx + 1) * dh];
            let dst = &mut r_h[(q * h + k) * dh..(q * h + k + 1) * dh];
            dst.copy_from_slice(src);
        }
    }
    let mut r_w = vec![0f32; w * w * dh];
    for q in 0..w {
        for k in 0..w {
            let idx = (q as isize - k as isize + (w as isize - 1)) as usize;
            let src = &rel_w_raw[idx * dh..(idx + 1) * dh];
            let dst = &mut r_w[(q * w + k) * dh..(q * w + k + 1) * dh];
            dst.copy_from_slice(src);
        }
    }
    Ok((r_h, r_w))
}

// ─── Neck (Conv2d 1×1 + LN2d + Conv2d 3×3 + LN2d) host-side ────────

/// Weights for the four neck layers, kept on the host because rlx-ir
/// doesn't have f32 forward Conv2d (and 3×3 padding=1 doesn't reduce
/// to matmul).
pub struct NeckWeights {
    pub conv1_w: Vec<f32>, // [out_chans, embed_dim] (1×1 conv = per-channel linear)
    pub ln1_g: Vec<f32>,   // [out_chans]
    pub ln1_b: Vec<f32>,
    pub conv2_w: Vec<f32>, // [out_chans, out_chans, 3, 3]
    pub ln2_g: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub embed_dim: usize,
    pub out_chans: usize,
    pub eps: f32,
}

fn extract_neck_weights(weights: &mut WeightMap, cfg: &SamEncoderConfig) -> Result<NeckWeights> {
    let (conv1_w_raw, c1_shape) = weights.take("image_encoder.neck.0.weight")?;
    ensure!(
        c1_shape == vec![cfg.out_chans, cfg.embed_dim, 1, 1],
        "neck.0.weight expected [{}, {}, 1, 1], got {c1_shape:?}",
        cfg.out_chans,
        cfg.embed_dim
    );
    let conv1_w = conv1_w_raw; // [out_chans, embed_dim] after flattening last two singleton dims
    let (ln1_g, _) = weights.take("image_encoder.neck.1.weight")?;
    let (ln1_b, _) = weights.take("image_encoder.neck.1.bias")?;
    let (conv2_w, c2_shape) = weights.take("image_encoder.neck.2.weight")?;
    ensure!(
        c2_shape == vec![cfg.out_chans, cfg.out_chans, 3, 3],
        "neck.2.weight expected [{}, {}, 3, 3], got {c2_shape:?}",
        cfg.out_chans,
        cfg.out_chans
    );
    let (ln2_g, _) = weights.take("image_encoder.neck.3.weight")?;
    let (ln2_b, _) = weights.take("image_encoder.neck.3.bias")?;
    Ok(NeckWeights {
        conv1_w,
        ln1_g,
        ln1_b,
        conv2_w,
        ln2_g,
        ln2_b,
        embed_dim: cfg.embed_dim,
        out_chans: cfg.out_chans,
        eps: cfg.layer_norm_eps as f32,
    })
}

/// Run the encoder neck on the host. `body_out` is the encoder body's
/// output reshaped to `[hw·hw, embed_dim]` (BHWC flattened). Returns
/// `[out_chans, hw, hw]` NCHW image embeddings.
pub fn apply_neck_host(neck: &NeckWeights, body_out: &[f32], hw: usize) -> Vec<f32> {
    let e = neck.embed_dim;
    let oc = neck.out_chans;
    let eps = neck.eps;

    // 1) Conv 1×1: per-pixel linear projection from embed_dim → out_chans.
    //    body_out is BHWC; treat as [hw·hw, embed_dim] and matmul by
    //    conv1_w.T (i.e. `out[s, oc] = sum_e body_out[s, e] * conv1_w[oc, e]`).
    let s = hw * hw;
    let mut feat = vec![0f32; s * oc]; // BHWC: [hw·hw, oc]
    for si in 0..s {
        for oi in 0..oc {
            let mut acc = 0f32;
            for ei in 0..e {
                acc += body_out[si * e + ei] * neck.conv1_w[oi * e + ei];
            }
            feat[si * oc + oi] = acc;
        }
    }

    // 2) LN2d: normalize over channel dim (per spatial position).
    layernorm2d_inplace(&mut feat, s, oc, &neck.ln1_g, &neck.ln1_b, eps);

    // 3) Conv 3×3 padding=1, stride=1. We compute it in NCHW. The input
    //    is currently BHWC = [hw·hw, oc]; convert to NCHW = [oc, hw, hw].
    let mut nchw = vec![0f32; oc * hw * hw];
    for y in 0..hw {
        for x in 0..hw {
            for c in 0..oc {
                nchw[c * hw * hw + y * hw + x] = feat[(y * hw + x) * oc + c];
            }
        }
    }
    let conv2_out = conv2d_3x3_pad1(&nchw, oc, oc, hw, hw, &neck.conv2_w);

    // 4) LN2d again. Convert back to BHWC for the LN, then back to NCHW.
    let mut bhwc = vec![0f32; s * oc];
    for c in 0..oc {
        for y in 0..hw {
            for x in 0..hw {
                bhwc[(y * hw + x) * oc + c] = conv2_out[c * hw * hw + y * hw + x];
            }
        }
    }
    layernorm2d_inplace(&mut bhwc, s, oc, &neck.ln2_g, &neck.ln2_b, eps);

    let mut out_nchw = vec![0f32; oc * hw * hw];
    for y in 0..hw {
        for x in 0..hw {
            for c in 0..oc {
                out_nchw[c * hw * hw + y * hw + x] = bhwc[(y * hw + x) * oc + c];
            }
        }
    }
    out_nchw
}

/// LN over channel dim of BHWC `[S, C]` (matches candle's LayerNorm2d).
fn layernorm2d_inplace(data: &mut [f32], s: usize, c: usize, g: &[f32], b: &[f32], eps: f32) {
    for si in 0..s {
        let row = &mut data[si * c..(si + 1) * c];
        let mean: f32 = row.iter().sum::<f32>() / c as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for k in 0..c {
            row[k] = (row[k] - mean) * inv * g[k] + b[k];
        }
    }
}

/// 3×3 Conv2d with stride=1, padding=1, no bias. NCHW in, NCHW out.
/// Reference implementation — not vectorized, fine for the SAM neck
/// (1 call per inference, 64×64×256).
fn conv2d_3x3_pad1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [out_c, in_c, 3, 3]
) -> Vec<f32> {
    let mut out = vec![0f32; out_c * h * w];
    for oc in 0..out_c {
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0f32;
                for ic in 0..in_c {
                    for ky in 0..3 {
                        let iy = y as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = x as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            let v = input[ic * h * w + iy as usize * w + ix as usize];
                            let wi = ((oc * in_c + ic) * 3 + ky) * 3 + kx;
                            acc += v * weight[wi];
                        }
                    }
                }
                out[oc * h * w + y * w + x] = acc;
            }
        }
    }
    out
}

// ─── Small builder helpers ─────────────────────────────────────────

fn load_p(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    key: &str,
    transpose: bool,
) -> Result<NodeId> {
    let (data, shape) = if transpose {
        weights
            .take_transposed(key)
            .map_err(|e| anyhow!("transpose-load `{key}`: {e}"))?
    } else {
        weights
            .take(key)
            .map_err(|e| anyhow!("load `{key}`: {e}"))?
    };
    let name = key.to_string();
    let id = g.param(&name, Shape::new(&shape, DType::F32));
    params.insert(name, data);
    Ok(id)
}

#[allow(dead_code)]
fn scalar_param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    value: f32,
) -> NodeId {
    let id = g.param(name, Shape::new(&[1], DType::F32));
    params.insert(name.to_string(), vec![value]);
    id
}

fn const_param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    shape: &[usize],
    data: Vec<f32>,
) -> NodeId {
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), data);
    id
}

fn pad_zero_param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    shape: &[usize],
) -> NodeId {
    let n: usize = shape.iter().product();
    let id = g.param(name, Shape::new(shape, DType::F32));
    params.insert(name.to_string(), vec![0f32; n]);
    id
}
