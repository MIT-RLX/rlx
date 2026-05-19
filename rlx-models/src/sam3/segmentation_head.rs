// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! Native SAM3 segmentation head + dot-product scoring.
//!
//! Mirrors `sam3.model.maskformer_segmentation.UniversalSegmentationHead`
//! and `sam3.model.model_misc.DotProductScoring` as configured in
//! `model_builder._create_segmentation_head` / `_create_dot_product_scoring`.

use super::detector::Sam3DetectorOutput;
use super::detector_decoder::{Mlp2, Mlp3};
use super::sam3::Sam3ImagePrediction;
use super::tensor::{layer_norm, linear, multihead_attention};
use crate::weight_map::WeightMap;
use anyhow::{Result, ensure};

const D_MODEL: usize = 256;
const N_HEADS: usize = 8;

#[derive(Clone, Default)]
pub struct Sam3SegmentationHeadWeights {
    pub loaded: bool,
    pub cross_attn_norm_w: Vec<f32>,
    pub cross_attn_norm_b: Vec<f32>,
    pub cross_attend_in_w_t: Vec<f32>,
    pub cross_attend_in_b: Vec<f32>,
    pub cross_attend_out_w_t: Vec<f32>,
    pub cross_attend_out_b: Vec<f32>,
    pub pixel_conv_w: Vec<Vec<f32>>,
    pub pixel_conv_b: Vec<Vec<f32>>,
    pub pixel_gn_w: Vec<Vec<f32>>,
    pub pixel_gn_b: Vec<Vec<f32>>,
    pub inst_w: Vec<f32>,
    pub inst_b: Vec<f32>,
    pub sem_w: Vec<f32>,
    pub sem_b: Vec<f32>,
    pub mask_embed: Mlp3,
}

#[derive(Clone, Default)]
pub struct Sam3DotProductScoringWeights {
    pub loaded: bool,
    pub prompt_mlp: Mlp2,
    pub prompt_mlp_out_norm_w: Vec<f32>,
    pub prompt_mlp_out_norm_b: Vec<f32>,
    pub prompt_proj_w_t: Vec<f32>,
    pub prompt_proj_b: Vec<f32>,
    pub hs_proj_w_t: Vec<f32>,
    pub hs_proj_b: Vec<f32>,
}

pub fn extract_segmentation_head_weights(
    weights: &mut WeightMap,
) -> Result<Sam3SegmentationHeadWeights> {
    let base = "detector.segmentation_head";

    let (cross_attn_norm_w, _) = weights.take(&format!("{base}.cross_attn_norm.weight"))?;
    let (cross_attn_norm_b, _) = weights.take(&format!("{base}.cross_attn_norm.bias"))?;
    let (cross_attend_in_w_t, _) =
        weights.take_transposed(&format!("{base}.cross_attend_prompt.in_proj_weight"))?;
    let (cross_attend_in_b, _) =
        weights.take(&format!("{base}.cross_attend_prompt.in_proj_bias"))?;
    let (cross_attend_out_w_t, _) =
        weights.take_transposed(&format!("{base}.cross_attend_prompt.out_proj.weight"))?;
    let (cross_attend_out_b, _) =
        weights.take(&format!("{base}.cross_attend_prompt.out_proj.bias"))?;

    let mut pixel_conv_w = Vec::new();
    let mut pixel_conv_b = Vec::new();
    let mut pixel_gn_w = Vec::new();
    let mut pixel_gn_b = Vec::new();
    for i in 0..3 {
        let (cw, cs) = weights.take(&format!("{base}.pixel_decoder.conv_layers.{i}.weight"))?;
        ensure!(
            cs == vec![D_MODEL, D_MODEL, 3, 3],
            "pixel_decoder conv {i} shape {cs:?}"
        );
        let (cb, _) = weights.take(&format!("{base}.pixel_decoder.conv_layers.{i}.bias"))?;
        let (nw, _) = weights.take(&format!("{base}.pixel_decoder.norms.{i}.weight"))?;
        let (nb, _) = weights.take(&format!("{base}.pixel_decoder.norms.{i}.bias"))?;
        pixel_conv_w.push(cw);
        pixel_conv_b.push(cb);
        pixel_gn_w.push(nw);
        pixel_gn_b.push(nb);
    }

    let (inst_w, ins) = weights.take(&format!("{base}.instance_seg_head.weight"))?;
    ensure!(
        ins == vec![D_MODEL, D_MODEL, 1, 1],
        "instance_seg_head shape {ins:?}"
    );
    let (inst_b, _) = weights.take(&format!("{base}.instance_seg_head.bias"))?;
    let (sem_w, ss) = weights.take(&format!("{base}.semantic_seg_head.weight"))?;
    ensure!(ss == vec![1, D_MODEL, 1, 1], "semantic_seg_head shape {ss:?}");
    let (sem_b, _) = weights.take(&format!("{base}.semantic_seg_head.bias"))?;

    let (m0_t, _) = weights
        .take_transposed(&format!("{base}.mask_predictor.mask_embed.layers.0.weight"))?;
    let (m0_b, _) = weights.take(&format!("{base}.mask_predictor.mask_embed.layers.0.bias"))?;
    let (m1_t, _) = weights
        .take_transposed(&format!("{base}.mask_predictor.mask_embed.layers.1.weight"))?;
    let (m1_b, _) = weights.take(&format!("{base}.mask_predictor.mask_embed.layers.1.bias"))?;
    let (m2_t, _) = weights
        .take_transposed(&format!("{base}.mask_predictor.mask_embed.layers.2.weight"))?;
    let (m2_b, _) = weights.take(&format!("{base}.mask_predictor.mask_embed.layers.2.bias"))?;
    let mask_embed = Mlp3 {
        w0_t: m0_t,
        b0: m0_b,
        w1_t: m1_t,
        b1: m1_b,
        w2_t: m2_t,
        b2: m2_b,
        in_dim: D_MODEL,
        hidden: D_MODEL,
        out_dim: D_MODEL,
    };

    Ok(Sam3SegmentationHeadWeights {
        loaded: true,
        cross_attn_norm_w,
        cross_attn_norm_b,
        cross_attend_in_w_t,
        cross_attend_in_b,
        cross_attend_out_w_t,
        cross_attend_out_b,
        pixel_conv_w,
        pixel_conv_b,
        pixel_gn_w,
        pixel_gn_b,
        inst_w,
        inst_b,
        sem_w,
        sem_b,
        mask_embed,
    })
}

pub fn extract_dot_product_scoring_weights(
    weights: &mut WeightMap,
) -> Result<Sam3DotProductScoringWeights> {
    let base = "detector.dot_prod_scoring";
    let (pm0_t, _) = weights.take_transposed(&format!("{base}.prompt_mlp.layers.0.weight"))?;
    let (pm0_b, _) = weights.take(&format!("{base}.prompt_mlp.layers.0.bias"))?;
    let (pm1_t, _) = weights.take_transposed(&format!("{base}.prompt_mlp.layers.1.weight"))?;
    let (pm1_b, _) = weights.take(&format!("{base}.prompt_mlp.layers.1.bias"))?;
    let prompt_mlp = Mlp2 {
        w0_t: pm0_t,
        b0: pm0_b,
        w1_t: pm1_t,
        b1: pm1_b,
        in_dim: D_MODEL,
        hidden: 2048,
        out_dim: D_MODEL,
    };
    let (pm_norm_w, _) = weights.take(&format!("{base}.prompt_mlp.out_norm.weight"))?;
    let (pm_norm_b, _) = weights.take(&format!("{base}.prompt_mlp.out_norm.bias"))?;
    let (pp_t, _) = weights.take_transposed(&format!("{base}.prompt_proj.weight"))?;
    let (pp_b, _) = weights.take(&format!("{base}.prompt_proj.bias"))?;
    let (hs_t, _) = weights.take_transposed(&format!("{base}.hs_proj.weight"))?;
    let (hs_b, _) = weights.take(&format!("{base}.hs_proj.bias"))?;
    Ok(Sam3DotProductScoringWeights {
        loaded: true,
        prompt_mlp,
        prompt_mlp_out_norm_w: pm_norm_w,
        prompt_mlp_out_norm_b: pm_norm_b,
        prompt_proj_w_t: pp_t,
        prompt_proj_b: pp_b,
        hs_proj_w_t: hs_t,
        hs_proj_b: hs_b,
    })
}

#[derive(Debug, Clone, Default)]
pub struct Sam3SegmentationOutput {
    pub mask_pred: Vec<f32>,
    pub semantic_seg: Vec<f32>,
    pub h_out: usize,
    pub w_out: usize,
    pub num_queries: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn forward_segmentation(
    weights: &Sam3SegmentationHeadWeights,
    enc_memory_bf: &[f32],
    backbone_fpn: &[Vec<f32>],
    backbone_shapes: &[(usize, usize)],
    obj_queries_last_bf: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    enc_h: usize,
    enc_w: usize,
    num_queries: usize,
    seq_len: usize,
) -> Result<Sam3SegmentationOutput> {
    ensure!(weights.loaded, "SAM3 segmentation head not loaded");
    ensure!(batch == 1, "batch > 1 not supported yet");
    ensure!(backbone_fpn.len() == 3, "expected 3 FPN levels (after scalp)");

    let hw = enc_h * enc_w;
    let norm_mem = layer_norm(
        enc_memory_bf,
        &weights.cross_attn_norm_w,
        &weights.cross_attn_norm_b,
        D_MODEL,
        1e-5,
    )?;
    let mut prompt_bf = vec![0f32; batch * seq_len * D_MODEL];
    for b in 0..batch {
        for l in 0..seq_len {
            let s = (l * batch + b) * D_MODEL;
            let d = (b * seq_len + l) * D_MODEL;
            prompt_bf[d..d + D_MODEL].copy_from_slice(&prompt_seq_first[s..s + D_MODEL]);
        }
    }
    let ca = multihead_attention(
        &norm_mem,
        &prompt_bf,
        &prompt_bf,
        &weights.cross_attend_in_w_t,
        &weights.cross_attend_in_b,
        &weights.cross_attend_out_w_t,
        &weights.cross_attend_out_b,
        batch,
        hw,
        seq_len,
        D_MODEL,
        N_HEADS,
        Some(prompt_kpm),
    )?;
    let mut enc_refined = enc_memory_bf.to_vec();
    for i in 0..enc_refined.len() {
        enc_refined[i] += ca[i];
    }
    let mut enc_visual = vec![0f32; batch * D_MODEL * hw];
    for b in 0..batch {
        for y in 0..enc_h {
            for xc in 0..enc_w {
                for c in 0..D_MODEL {
                    enc_visual[((b * D_MODEL + c) * enc_h + y) * enc_w + xc] =
                        enc_refined[(b * hw + y * enc_w + xc) * D_MODEL + c];
                }
            }
        }
    }

    let mut levels = backbone_fpn.to_vec();
    levels[2] = enc_visual;
    let mut shapes = backbone_shapes.to_vec();
    shapes[2] = (enc_h, enc_w);

    let mut prev = levels.pop().unwrap();
    let (mut ph, mut pw) = shapes.pop().unwrap();
    for (i, (curr, (ch, cw))) in levels.iter().rev().zip(shapes.iter().rev()).enumerate() {
        let up = nearest_upsample_nchw(&prev, D_MODEL, ph, pw, *ch, *cw);
        let mut combined = vec![0f32; curr.len()];
        for j in 0..combined.len() {
            combined[j] = curr[j] + up[j];
        }
        let conv = conv2d_3x3_pad1(
            &combined,
            D_MODEL,
            *ch,
            *cw,
            &weights.pixel_conv_w[i],
            &weights.pixel_conv_b[i],
        );
        let mut relud = group_norm(
            &conv,
            batch,
            D_MODEL,
            *ch,
            *cw,
            8,
            &weights.pixel_gn_w[i],
            &weights.pixel_gn_b[i],
        );
        for v in relud.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        prev = relud;
        ph = *ch;
        pw = *cw;
    }
    let pixel_embed = prev;

    let inst = conv2d_1x1(
        &pixel_embed,
        D_MODEL,
        D_MODEL,
        ph,
        pw,
        &weights.inst_w,
        &weights.inst_b,
    );

    let mask_embed_out = mlp3_forward(&weights.mask_embed, obj_queries_last_bf, batch * num_queries)?;
    let mut mask_pred = vec![0f32; batch * num_queries * ph * pw];
    for b in 0..batch {
        for q in 0..num_queries {
            for c in 0..D_MODEL {
                let qcoeff = mask_embed_out[(b * num_queries + q) * D_MODEL + c];
                if qcoeff == 0.0 {
                    continue;
                }
                let plane = &inst[((b * D_MODEL + c) * ph * pw)
                    ..((b * D_MODEL + c) * ph * pw + ph * pw)];
                let dst = &mut mask_pred
                    [(b * num_queries + q) * ph * pw..(b * num_queries + q + 1) * ph * pw];
                for p in 0..ph * pw {
                    dst[p] += qcoeff * plane[p];
                }
            }
        }
    }

    let semantic_seg = conv2d_1x1(&pixel_embed, D_MODEL, 1, ph, pw, &weights.sem_w, &weights.sem_b);

    Ok(Sam3SegmentationOutput {
        mask_pred,
        semantic_seg,
        h_out: ph,
        w_out: pw,
        num_queries,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn forward_dot_prod_scoring(
    weights: &Sam3DotProductScoringWeights,
    hs_bf: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    num_layers: usize,
    batch: usize,
    num_queries: usize,
    seq_len: usize,
) -> Result<Vec<f32>> {
    ensure!(weights.loaded, "SAM3 dot product scoring not loaded");
    let rows = seq_len * batch;
    let pm = &weights.prompt_mlp;
    let mut h = linear(prompt_seq_first, rows, pm.in_dim, &pm.w0_t, pm.hidden, &pm.b0)?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let mut h = linear(&h, rows, pm.hidden, &pm.w1_t, pm.out_dim, &pm.b1)?;
    for i in 0..h.len() {
        h[i] += prompt_seq_first[i];
    }
    let h = layer_norm(
        &h,
        &weights.prompt_mlp_out_norm_w,
        &weights.prompt_mlp_out_norm_b,
        D_MODEL,
        1e-5,
    )?;

    let mut pooled = vec![0f32; batch * D_MODEL];
    let mut counts = vec![0.0f32; batch];
    for b in 0..batch {
        for l in 0..seq_len {
            if prompt_kpm[b * seq_len + l] == 0 {
                let src = (l * batch + b) * D_MODEL;
                let dst = b * D_MODEL;
                for c in 0..D_MODEL {
                    pooled[dst + c] += h[src + c];
                }
                counts[b] += 1.0;
            }
        }
    }
    for b in 0..batch {
        let denom = counts[b].max(1.0);
        for c in 0..D_MODEL {
            pooled[b * D_MODEL + c] /= denom;
        }
    }

    let proj_pooled = linear(
        &pooled,
        batch,
        D_MODEL,
        &weights.prompt_proj_w_t,
        D_MODEL,
        &weights.prompt_proj_b,
    )?;
    let proj_hs = linear(
        hs_bf,
        num_layers * batch * num_queries,
        D_MODEL,
        &weights.hs_proj_w_t,
        D_MODEL,
        &weights.hs_proj_b,
    )?;

    let scale = 1.0f32 / (D_MODEL as f32).sqrt();
    let clamp = 12.0f32;
    let mut scores = vec![0f32; num_layers * batch * num_queries];
    for l in 0..num_layers {
        for b in 0..batch {
            let pp = &proj_pooled[b * D_MODEL..(b + 1) * D_MODEL];
            for q in 0..num_queries {
                let row = &proj_hs[((l * batch + b) * num_queries + q) * D_MODEL
                    ..((l * batch + b) * num_queries + q + 1) * D_MODEL];
                let mut acc = 0.0f32;
                for c in 0..D_MODEL {
                    acc += row[c] * pp[c];
                }
                let s = (acc * scale).clamp(-clamp, clamp);
                scores[(l * batch + b) * num_queries + q] = s;
            }
        }
    }
    Ok(scores)
}

fn mlp3_forward(mlp: &Mlp3, x: &[f32], rows: usize) -> Result<Vec<f32>> {
    let mut h = linear(x, rows, mlp.in_dim, &mlp.w0_t, mlp.hidden, &mlp.b0)?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let mut h = linear(&h, rows, mlp.hidden, &mlp.w1_t, mlp.hidden, &mlp.b1)?;
    for v in h.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    linear(&h, rows, mlp.hidden, &mlp.w2_t, mlp.out_dim, &mlp.b2)
}

fn nearest_upsample_nchw(
    x: &[f32],
    c: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; c * dst_h * dst_w];
    for cc in 0..c {
        let inp = &x[cc * src_h * src_w..(cc + 1) * src_h * src_w];
        let oup = &mut out[cc * dst_h * dst_w..(cc + 1) * dst_h * dst_w];
        for y in 0..dst_h {
            let sy = y * src_h / dst_h;
            for x in 0..dst_w {
                let sx = x * src_w / dst_w;
                oup[y * dst_w + x] = inp[sy * src_w + sx];
            }
        }
    }
    out
}

fn conv2d_3x3_pad1(
    input: &[f32],
    c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut out = vec![0f32; c * h * w];
    for oc in 0..c {
        let b = bias[oc];
        let oup = &mut out[oc * h * w..(oc + 1) * h * w];
        for v in oup.iter_mut() {
            *v = b;
        }
    }
    for oc in 0..c {
        for ic in 0..c {
            let w_oi = &weight[((oc * c + ic) * 9)..((oc * c + ic) * 9 + 9)];
            let inp = &input[ic * h * w..(ic + 1) * h * w];
            let oup = &mut out[oc * h * w..(oc + 1) * h * w];
            for oy in 0..h {
                for ox in 0..w {
                    let mut acc = 0.0f32;
                    for ky in 0..3 {
                        let iy = oy as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let ix = ox as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            acc += inp[iy as usize * w + ix as usize] * w_oi[ky * 3 + kx];
                        }
                    }
                    oup[oy * w + ox] += acc;
                }
            }
        }
    }
    out
}

fn conv2d_1x1(
    input: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let n = h * w;
    let mut out = vec![0f32; out_c * n];
    rlx_cpu::blas::sgemm(weight, input, &mut out, out_c, in_c, n);
    for oc in 0..out_c {
        let b = bias[oc];
        let row = &mut out[oc * n..(oc + 1) * n];
        for v in row {
            *v += b;
        }
    }
    out
}

fn group_norm(
    x: &[f32],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    num_groups: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    assert!(channels % num_groups == 0);
    let cpg = channels / num_groups;
    let spatial = h * w;
    let mut out = vec![0f32; batch * channels * spatial];
    for b in 0..batch {
        for g in 0..num_groups {
            let c0 = g * cpg;
            let n = (cpg * spatial) as f32;
            let mut mean = 0.0f32;
            for c in 0..cpg {
                let plane = &x[((b * channels + c0 + c) * spatial)
                    ..((b * channels + c0 + c + 1) * spatial)];
                for v in plane {
                    mean += *v;
                }
            }
            mean /= n;
            let mut var = 0.0f32;
            for c in 0..cpg {
                let plane = &x[((b * channels + c0 + c) * spatial)
                    ..((b * channels + c0 + c + 1) * spatial)];
                for v in plane {
                    let d = *v - mean;
                    var += d * d;
                }
            }
            var /= n;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for c in 0..cpg {
                let src = &x[((b * channels + c0 + c) * spatial)
                    ..((b * channels + c0 + c + 1) * spatial)];
                let dst = &mut out[((b * channels + c0 + c) * spatial)
                    ..((b * channels + c0 + c + 1) * spatial)];
                let g_ = gamma[c0 + c];
                let bias = beta[c0 + c];
                for (s, d) in src.iter().zip(dst.iter_mut()) {
                    *d = (*s - mean) * inv * g_ + bias;
                }
            }
        }
    }
    out
}

/// Legacy stub used by the not-yet-finished `Sam3::predict_image` path.
pub fn segmentation_forward_native(
    _weights: &Sam3SegmentationHeadWeights,
    detector: &Sam3DetectorOutput,
    h_out: usize,
    w_out: usize,
) -> Sam3ImagePrediction {
    Sam3ImagePrediction {
        masks: vec![0.0; detector.num_queries * h_out * w_out],
        mask_shape: vec![detector.num_queries, h_out, w_out],
        boxes: vec![0.0; detector.num_queries * 4],
        boxes_shape: vec![detector.num_queries, 4],
        scores: vec![0.0; detector.num_queries],
        scores_shape: vec![detector.num_queries],
        num_instances: detector.num_queries,
        h_out,
        w_out,
    }
}
