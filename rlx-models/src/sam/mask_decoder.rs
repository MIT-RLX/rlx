// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! SAM v1 mask decoder — host-side.
//!
//! Two-way transformer over (point tokens, image embeddings) → mask
//! token outputs → ConvTranspose2d ×2 upscaling → hypernetwork MLPs
//! → mask logits + IoU predictions. Mirrors candle's
//! `mask_decoder.rs`.

use super::config::SAM_EMBED_HW;
use super::prompt_encoder::gelu_erf_inplace;
use super::transformer::{
    TwoWayTransformerWeights, extract_two_way_transformer_weights, layer_norm_last, linear,
    two_way_transformer_forward,
};
use crate::weight_map::WeightMap;
use anyhow::{Result, ensure};

pub struct MaskDecoderWeights {
    pub iou_token: Vec<f32>,   // [1, transformer_dim]
    pub mask_tokens: Vec<f32>, // [num_mask_tokens, transformer_dim]
    pub transformer: TwoWayTransformerWeights,

    /// ConvTranspose2d: in=transformer_dim, out=transformer_dim/4,
    /// kernel=2, stride=2. Weight shape `[in, out, 2, 2]`.
    pub upscale_conv1_w: Vec<f32>,
    pub upscale_conv1_b: Vec<f32>,
    /// LayerNorm2d on the upscaled feature.
    pub upscale_ln_g: Vec<f32>,
    pub upscale_ln_b: Vec<f32>,
    /// ConvTranspose2d: in=transformer_dim/4, out=transformer_dim/8.
    pub upscale_conv2_w: Vec<f32>,
    pub upscale_conv2_b: Vec<f32>,

    /// `num_mask_tokens` × 3-layer ReLU MLPs (`transformer_dim → transformer_dim
    /// → transformer_dim → transformer_dim/8`). Each MLP's flat
    /// weights+biases stored sequentially in `hyper_mlps_*`.
    pub hyper_mlps: Vec<HypernetMlp>,

    /// IoU prediction head: 3-layer ReLU MLP `transformer_dim →
    /// iou_head_hidden_dim → iou_head_hidden_dim → num_mask_tokens`.
    pub iou_head: HypernetMlp,

    pub transformer_dim: usize,
    pub num_mask_tokens: usize,
}

pub struct HypernetMlp {
    pub layers: Vec<MlpLayer>,
}

pub struct MlpLayer {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub in_d: usize,
    pub out_d: usize,
}

pub(super) fn extract_mask_decoder_weights(
    weights: &mut WeightMap,
    transformer_dim: usize,
    num_mask_tokens: usize,
    iou_head_depth: usize,
    iou_head_hidden_dim: usize,
    transformer_depth: usize,
    transformer_num_heads: usize,
    transformer_mlp_dim: usize,
) -> Result<MaskDecoderWeights> {
    let (iou_token, sh) = weights.take("mask_decoder.iou_token.weight")?;
    ensure!(
        sh == vec![1, transformer_dim],
        "iou_token shape {sh:?} not [1, {transformer_dim}]"
    );
    let (mask_tokens, sh) = weights.take("mask_decoder.mask_tokens.weight")?;
    ensure!(
        sh == vec![num_mask_tokens, transformer_dim],
        "mask_tokens shape {sh:?} not [{num_mask_tokens}, {transformer_dim}]"
    );

    // ConvTranspose2d weight convention in PyTorch: [in, out, kH, kW].
    let q4 = transformer_dim / 4;
    let q8 = transformer_dim / 8;
    let (upscale_conv1_w, sh) = weights.take("mask_decoder.output_upscaling.0.weight")?;
    ensure!(
        sh == vec![transformer_dim, q4, 2, 2],
        "output_upscaling.0.weight shape {sh:?} not [{transformer_dim}, {q4}, 2, 2]"
    );
    let (upscale_conv1_b, _) = weights.take("mask_decoder.output_upscaling.0.bias")?;
    let (upscale_ln_g, _) = weights.take("mask_decoder.output_upscaling.1.weight")?;
    let (upscale_ln_b, _) = weights.take("mask_decoder.output_upscaling.1.bias")?;
    let (upscale_conv2_w, sh) = weights.take("mask_decoder.output_upscaling.3.weight")?;
    ensure!(
        sh == vec![q4, q8, 2, 2],
        "output_upscaling.3.weight shape {sh:?} not [{q4}, {q8}, 2, 2]"
    );
    let (upscale_conv2_b, _) = weights.take("mask_decoder.output_upscaling.3.bias")?;

    // Each hypernetwork MLP: 3-layer (transformer_dim → transformer_dim
    // → transformer_dim → transformer_dim/8).
    let mut hyper_mlps = Vec::with_capacity(num_mask_tokens);
    for i in 0..num_mask_tokens {
        let mlp = extract_mlp(
            weights,
            &format!("mask_decoder.output_hypernetworks_mlps.{i}"),
            transformer_dim,
            transformer_dim,
            q8,
            3,
        )?;
        hyper_mlps.push(mlp);
    }

    let iou_head = extract_mlp(
        weights,
        "mask_decoder.iou_prediction_head",
        transformer_dim,
        iou_head_hidden_dim,
        num_mask_tokens,
        iou_head_depth,
    )?;

    let transformer = extract_two_way_transformer_weights(
        weights,
        transformer_dim,
        transformer_depth,
        transformer_num_heads,
        transformer_mlp_dim,
    )?;

    Ok(MaskDecoderWeights {
        iou_token,
        mask_tokens,
        transformer,
        upscale_conv1_w,
        upscale_conv1_b,
        upscale_ln_g,
        upscale_ln_b,
        upscale_conv2_w,
        upscale_conv2_b,
        hyper_mlps,
        iou_head,
        transformer_dim,
        num_mask_tokens,
    })
}

fn extract_mlp(
    weights: &mut WeightMap,
    prefix: &str,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    num_layers: usize,
) -> Result<HypernetMlp> {
    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let in_d = if i == 0 { input_dim } else { hidden_dim };
        let out_d = if i + 1 == num_layers {
            output_dim
        } else {
            hidden_dim
        };
        let (w, sh) = weights.take(&format!("{prefix}.layers.{i}.weight"))?;
        ensure!(
            sh == vec![out_d, in_d],
            "{prefix}.layers.{i}.weight shape {sh:?} not [{out_d}, {in_d}]"
        );
        let (b, _) = weights.take(&format!("{prefix}.layers.{i}.bias"))?;
        layers.push(MlpLayer { w, b, in_d, out_d });
    }
    Ok(HypernetMlp { layers })
}

/// Forward through a ReLU MLP. Input `[rows, layer0.in_d]`, output
/// `[rows, last_layer.out_d]`. The final layer is NOT followed by ReLU.
pub fn mlp_forward(mlp: &HypernetMlp, x: &[f32], rows: usize) -> Vec<f32> {
    let mut cur = x.to_vec();
    let n = mlp.layers.len();
    for (i, layer) in mlp.layers.iter().enumerate() {
        cur = linear(&cur, &layer.w, &layer.b, rows, layer.in_d, layer.out_d);
        if i + 1 < n {
            for v in cur.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
    cur
}

/// Forward through the mask decoder, returning (masks, iou_pred).
///
/// `image_embeddings`: NCHW `[1, C=256, hw, hw]`.
/// `image_pe`: NCHW `[1, C=256, hw, hw]`.
/// `sparse_prompt_embeddings`: `[1, num_sparse, E]` (may have 0 sparse tokens).
/// `dense_prompt_embeddings`: `[1, E, hw, hw]`.
///
/// `multimask_output`: if true, return masks[..., 1:4] (3 candidates);
/// else return masks[..., 0:1] (the single "best" output).
///
/// Output shapes:
///   - masks: `[1, num_masks, 4·hw, 4·hw]`
///     (num_masks = 3 if multimask_output else 1).
///   - iou_pred: `[1, num_masks]`.
pub fn mask_decoder_forward(
    w: &MaskDecoderWeights,
    image_embeddings: &[f32],
    image_pe: &[f32],
    sparse_prompt_embeddings: &[f32],
    num_sparse_tokens: usize,
    dense_prompt_embeddings: &[f32],
    multimask_output: bool,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
    let e = w.transformer_dim;
    let hw = SAM_EMBED_HW;
    ensure!(
        image_embeddings.len() == e * hw * hw,
        "image_embeddings len {} ≠ E·hw·hw ({e}·{hw}·{hw})",
        image_embeddings.len()
    );
    ensure!(
        image_pe.len() == e * hw * hw,
        "image_pe len {} ≠ E·hw·hw",
        image_pe.len()
    );
    ensure!(
        dense_prompt_embeddings.len() == e * hw * hw,
        "dense_prompt_embeddings len {} ≠ E·hw·hw",
        dense_prompt_embeddings.len()
    );
    ensure!(
        sparse_prompt_embeddings.len() == num_sparse_tokens * e,
        "sparse_prompt_embeddings len {} ≠ num_sparse·E ({num_sparse_tokens}·{e})",
        sparse_prompt_embeddings.len()
    );

    // ── Build `tokens` = cat(iou_token, mask_tokens, sparse_prompts) ──
    // Output tokens (iou + mask): shape [1 + num_mask_tokens, E]
    let nm = w.num_mask_tokens;
    let n_out_tokens = 1 + nm;
    let q_n = n_out_tokens + num_sparse_tokens;
    let mut tokens = Vec::with_capacity(q_n * e);
    tokens.extend_from_slice(&w.iou_token); // [1, E]
    tokens.extend_from_slice(&w.mask_tokens); // [nm, E]
    tokens.extend_from_slice(sparse_prompt_embeddings); // [num_sparse, E]
    // Shape [1, q_n, E].

    // ── src = image_embeddings + dense_prompt_embeddings ──
    let mut src = image_embeddings.to_vec();
    for i in 0..src.len() {
        src[i] += dense_prompt_embeddings[i];
    }
    let pos_src = image_pe.to_vec();

    // ── Run the two-way transformer ──
    let (hs, src_post) = two_way_transformer_forward(
        &w.transformer,
        &src,
        &pos_src,
        &tokens,
        /*b=*/ 1,
        e,
        hw,
        hw,
        q_n,
    );
    // hs: [1, q_n, E]; src_post: [1, hw*hw, E]

    // iou_token_out = hs[:, 0, :] → [1, E]
    let iou_token_out: Vec<f32> = hs[..e].to_vec();
    // mask_tokens_out = hs[:, 1..1+nm, :] → [1, nm, E]
    let mask_tokens_out = &hs[e..e * (1 + nm)];

    // src reshape to [B, C, H, W] (BCHW). src_post is [1, hw*hw, E];
    // transpose to [1, E, hw*hw] then reshape to [1, E, hw, hw].
    let mut src_nchw = vec![0f32; e * hw * hw];
    for s in 0..hw * hw {
        for c in 0..e {
            src_nchw[c * hw * hw + s] = src_post[s * e + c];
        }
    }

    // ── Upscaling: ConvTranspose2d → LN2d → GELU → ConvTranspose2d → GELU ──
    let q4 = e / 4;
    let q8 = e / 8;
    let h1 = hw * 2;
    let w1 = hw * 2;
    let mut up1 = conv_transpose2d_stride2_k2(
        &src_nchw,
        e,
        q4,
        hw,
        hw,
        &w.upscale_conv1_w,
        &w.upscale_conv1_b,
    );
    layernorm2d_nchw(&mut up1, q4, h1, w1, &w.upscale_ln_g, &w.upscale_ln_b, 1e-6);
    gelu_erf_inplace(&mut up1);

    let h2 = h1 * 2;
    let w2 = w1 * 2;
    let mut up2 =
        conv_transpose2d_stride2_k2(&up1, q4, q8, h1, w1, &w.upscale_conv2_w, &w.upscale_conv2_b);
    gelu_erf_inplace(&mut up2);
    // up2 is [q8, h2, w2] = [transformer_dim/8, 4·hw, 4·hw].

    // ── Per-mask hypernetwork MLPs → [nm, q8] ──
    let mut hyper_in = vec![0f32; nm * q8];
    for i in 0..nm {
        let token = &mask_tokens_out[i * e..(i + 1) * e];
        let h = mlp_forward(&w.hyper_mlps[i], token, 1);
        hyper_in[i * q8..(i + 1) * q8].copy_from_slice(&h);
    }
    // hyper_in: [nm, q8]. up2 flat [q8, spat].
    // masks = hyper_in @ up2   shape [nm, spat]. BLAS-backed.
    let spat = h2 * w2;
    let mut masks_all = vec![0f32; nm * spat];
    rlx_cpu::blas::sgemm_auto(&hyper_in, &up2, &mut masks_all, nm, q8, spat);

    // ── IoU prediction head ──
    let iou_pred_all = mlp_forward(&w.iou_head, &iou_token_out, 1); // [1, nm]

    // ── Slice for multimask vs single ──
    let (masks, iou_pred, num_masks) = if multimask_output {
        // [1, 1..nm, h2, w2] = [1, nm-1, h2, w2] (3 masks for nm=4)
        let mut masks = vec![0f32; (nm - 1) * spat];
        masks.copy_from_slice(&masks_all[spat..]);
        let mut iou = vec![0f32; nm - 1];
        iou.copy_from_slice(&iou_pred_all[1..]);
        (masks, iou, nm - 1)
    } else {
        let masks = masks_all[..spat].to_vec();
        let iou = iou_pred_all[..1].to_vec();
        (masks, iou, 1)
    };

    Ok((masks, iou_pred, num_masks, h2))
}

// ─── Host-side ConvTranspose2d (k=2, s=2, pad=0) ────────────────

/// PyTorch ConvTranspose2d with kernel=2, stride=2, padding=0.
/// Weight shape `[in_c, out_c, 2, 2]`. Bias `[out_c]`.
/// Each input pixel scatters a 2×2 patch into the output (non-overlapping).
fn conv_transpose2d_stride2_k2(
    input: &[f32], // [in_c, h, w]
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32], // [in_c, out_c, 2, 2]
    bias: &[f32],   // [out_c]
) -> Vec<f32> {
    let out_h = h * 2;
    let out_w = w * 2;
    let mut out = vec![0f32; out_c * out_h * out_w];
    // Pre-fill with bias broadcast per channel.
    for oc in 0..out_c {
        let b = bias[oc];
        let plane = &mut out[oc * out_h * out_w..(oc + 1) * out_h * out_w];
        for v in plane.iter_mut() {
            *v = b;
        }
    }
    for ic in 0..in_c {
        for iy in 0..h {
            for ix in 0..w {
                let v = input[ic * h * w + iy * w + ix];
                if v == 0.0 {
                    continue;
                }
                for ky in 0..2 {
                    let oy = iy * 2 + ky;
                    for kx in 0..2 {
                        let ox = ix * 2 + kx;
                        for oc in 0..out_c {
                            let w_idx = ((ic * out_c + oc) * 2 + ky) * 2 + kx;
                            out[oc * out_h * out_w + oy * out_w + ox] += v * weight[w_idx];
                        }
                    }
                }
            }
        }
    }
    out
}

/// LayerNorm over channel axis of NCHW (per spatial pos). Same as in
/// `prompt_encoder.rs`; duplicated to keep the file self-contained.
fn layernorm2d_nchw(
    data: &mut [f32],
    c: usize,
    h: usize,
    w: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) {
    let n = h * w;
    for i in 0..n {
        let mut mean = 0f32;
        for k in 0..c {
            mean += data[k * n + i];
        }
        mean /= c as f32;
        let mut var = 0f32;
        for k in 0..c {
            let d = data[k * n + i] - mean;
            var += d * d;
        }
        var /= c as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for k in 0..c {
            let v = (data[k * n + i] - mean) * inv;
            data[k * n + i] = v * gamma[k] + beta[k];
        }
    }
}

#[allow(dead_code)]
fn _silence_layer_norm_last_import(x: &mut [f32], g: &[f32], b: &[f32]) {
    layer_norm_last(x, 1, 1, g, b, 1e-5);
}
