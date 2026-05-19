// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! SAM3 top-level API.
//!
//! This module owns the shipped checkpoint-facing surface for base SAM3.
//! The native surface exposes SAM3 preprocessing and ViT patch embeddings.
//! Full image/video inference runs through native Rust modules. The Python
//! helper in this directory is kept only as a parity oracle.

use super::config::Sam3Config;
use super::detector::{Sam3DetectorWeights, detector_forward_native};
use super::detector_decoder::{
    Sam3DecoderOutput, Sam3DecoderWeights, extract_decoder_weights, forward_decoder,
};
use super::detector_encoder::{Sam3EncoderWeights, extract_encoder_weights, forward_encoder};
use super::detector_encoder_ir::{forward_encoder_ir, forward_encoder_ir_on};
use super::geometry::{Sam3GeometryWeights, encode_geometry_native};
use super::neck::{Sam3NeckWeights, apply_neck_native, extract_neck_weights};
use super::preprocess::{assemble_patch_tokens, preprocess_image};
use super::segmentation_head::{
    Sam3DotProductScoringWeights, Sam3SegmentationHeadWeights, Sam3SegmentationOutput,
    extract_dot_product_scoring_weights, extract_segmentation_head_weights,
    forward_dot_prod_scoring, forward_segmentation, segmentation_forward_native,
};
use super::text_encoder::{
    Sam3TextEncoded, Sam3TextEncoderWeights, encode_text_native, encode_tokens,
    extract_text_encoder_weights,
};
use super::tracker::{Sam3TrackerWeights, extract_tracker_weights, tracker_forward_native};
use super::vision_encoder::{
    Sam3VisionEncoderWeights, encode_image_native, extract_vision_encoder_weights,
};
use crate::weight_map::WeightMap;
use anyhow::{Context, Result, bail, ensure};
use rlx_runtime::Device;

#[derive(Debug, Clone)]
pub struct Sam3EncodedImage {
    /// `[grid * grid, embed_dim]` flattened row-major patch tokens.
    pub patch_tokens: Vec<f32>,
    pub grid: usize,
    pub embed_dim: usize,
    pub resized_hw: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct Sam3ImagePrediction {
    /// Mask logits/probabilities flattened in row-major order. The shape
    /// is available in `mask_shape`.
    pub masks: Vec<f32>,
    pub mask_shape: Vec<usize>,
    pub boxes: Vec<f32>,
    pub boxes_shape: Vec<usize>,
    pub scores: Vec<f32>,
    pub scores_shape: Vec<usize>,
    pub num_instances: usize,
    pub h_out: usize,
    pub w_out: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Sam3VideoState {
    pub frame_index: usize,
    pub memory_tokens: Vec<Vec<f32>>,
    pub last_prediction: Option<Sam3ImagePrediction>,
}

#[derive(Debug, Clone)]
pub struct Sam3VideoFramePrediction {
    pub frame_index: usize,
    pub image: Sam3ImagePrediction,
    pub memory_len: usize,
}

pub struct Sam3 {
    cfg: Sam3Config,
    vision: Option<Sam3VisionEncoderWeights>,
    neck: Sam3NeckWeights,
    text: Sam3TextEncoderWeights,
    geometry: Sam3GeometryWeights,
    detector: Sam3DetectorWeights,
    encoder: Sam3EncoderWeights,
    decoder: Sam3DecoderWeights,
    seg_head: Sam3SegmentationHeadWeights,
    scoring: Sam3DotProductScoringWeights,
    seg: Sam3SegmentationHeadWeights,
    tracker: Sam3TrackerWeights,
    device: Device,
}

impl Sam3 {
    /// Load a SAM3 checkpoint for native inference.
    ///
    /// Native Rust inference consumes safetensors. Convert upstream `.pt`
    /// checkpoints with `tests/sam3_parity_helpers/pt_to_safetensors.py`.
    pub fn from_checkpoint(weights_path: &str, cfg: Sam3Config) -> Result<Self> {
        Self::from_checkpoint_on(weights_path, cfg, Device::Cpu)
    }

    pub fn from_checkpoint_on(weights_path: &str, cfg: Sam3Config, device: Device) -> Result<Self> {
        Self::from_safetensors_on(weights_path, cfg, device)
    }

    pub fn from_safetensors(weights_path: &str, cfg: Sam3Config) -> Result<Self> {
        Self::from_safetensors_on(weights_path, cfg, Device::Cpu)
    }

    pub fn from_safetensors_on(
        weights_path: &str,
        cfg: Sam3Config,
        device: Device,
    ) -> Result<Self> {
        if !weights_path.ends_with(".safetensors") {
            bail!(
                "SAM3 native loader expects safetensors; convert .pt with tests/sam3_parity_helpers/pt_to_safetensors.py"
            );
        }

        let mut wm = WeightMap::from_file(weights_path)?;
        let vision = extract_vision_encoder_weights(&mut wm, &cfg.vit)?;
        let neck = extract_neck_weights(&mut wm)?;
        let text = extract_text_encoder_weights(&mut wm, &cfg.text)?;
        let encoder = extract_encoder_weights(&mut wm)?;
        let decoder = extract_decoder_weights(&mut wm)?;
        let seg_head = extract_segmentation_head_weights(&mut wm)?;
        let scoring = extract_dot_product_scoring_weights(&mut wm)?;
        let tracker = extract_tracker_weights(&mut wm)?;
        Ok(Self {
            cfg,
            vision: Some(vision),
            neck,
            text,
            geometry: Sam3GeometryWeights::default(),
            detector: Sam3DetectorWeights::default(),
            encoder,
            seg: Sam3SegmentationHeadWeights::default(),
            tracker,
            decoder,
            seg_head,
            scoring,
            device,
        })
    }

    pub fn config(&self) -> &Sam3Config {
        &self.cfg
    }

    /// Returns the loaded tracker weights (used by the video smoke test
    /// to confirm checkpoint coverage).
    pub fn tracker_weights(&self) -> &Sam3TrackerWeights {
        &self.tracker
    }

    pub fn encoder_weights(&self) -> &Sam3EncoderWeights {
        &self.encoder
    }

    pub fn decoder_weights(&self) -> &Sam3DecoderWeights {
        &self.decoder
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn encode_image(
        &self,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
    ) -> Result<Sam3EncodedImage> {
        let vision = self
            .vision
            .as_ref()
            .context("SAM3 encode_image requires native vision weights")?;
        let (image_nchw, resized_hw) = preprocess_image(image_u8, h_in, w_in);
        let encoded = encode_image_native(vision, &self.cfg.vit, &image_nchw)?;
        Ok(Sam3EncodedImage {
            patch_tokens: encoded.tokens,
            grid: encoded.grid,
            embed_dim: encoded.dim,
            resized_hw,
        })
    }

    /// Run the vision trunk + 4-scale neck and return per-level
    /// `[channels, h, w]` feature maps with matching sinusoidal positional
    /// encodings. Used by the detector and as a parity gate.
    /// End-to-end image inference with a pre-tokenized text prompt
    /// (`tokens` has length `seq_len` == decoder context length, usually
    /// 32). Returns the same 3-tuple the public `Sam3Processor.set_text_
    /// prompt` exposes — without NMS / score thresholding, which we leave
    /// to callers so parity tests can compare raw model outputs.
    pub fn predict_image_text(
        &self,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
        tokens: &[u32],
    ) -> Result<Sam3ImagePrediction> {
        let cfg = &self.cfg;
        let nq = 200;
        let seq_len = tokens.len();

        // Vision + neck.
        let vision = self
            .vision
            .as_ref()
            .context("predict_image_text requires native vision weights")?;
        let (image_nchw, resized_hw) = preprocess_image(image_u8, h_in, w_in);
        let vision_out =
            super::vision_encoder::encode_image_native(vision, &cfg.vit, &image_nchw)?;
        let levels = apply_neck_native(&self.neck, &vision_out)?;
        // Drop the last (scale=0.5) level per scalp=1.
        let kept = &levels[..3];
        let backbone_fpn: Vec<Vec<f32>> = kept.iter().map(|l| l.features.clone()).collect();
        let backbone_shapes: Vec<(usize, usize)> = kept.iter().map(|l| (l.h, l.w)).collect();
        // Encoder input = level scale=1.0 (index 2).
        let src_level = &kept[2];
        let h = src_level.h;
        let w = src_level.w;
        let batch = 1;

        // Text encoder.
        let text_out = encode_tokens(&self.text, tokens, batch, seq_len)?;

        // Detector encoder (single-level fusion).
        let memory_bf = forward_encoder(
            &self.encoder,
            &src_level.features,
            &src_level.pos,
            &text_out.text_memory_resized,
            &text_out.attention_mask,
            batch,
            h,
            w,
            seq_len,
        )?;
        // Convert FPN pos to batch-first for the decoder.
        let mut memory_pos = vec![0f32; batch * h * w * 256];
        for b in 0..batch {
            for y in 0..h {
                for xc in 0..w {
                    for c in 0..256 {
                        memory_pos[(b * h * w + y * w + xc) * 256 + c] =
                            src_level.pos[((b * 256 + c) * h + y) * w + xc];
                    }
                }
            }
        }

        // Detector decoder.
        let dec = forward_decoder(
            &self.decoder,
            &memory_bf,
            &memory_pos,
            &text_out.text_memory_resized,
            &text_out.attention_mask,
            batch,
            h,
            w,
            seq_len,
        )?;

        // Last-layer queries batch-first.
        let num_layers = dec.num_layers;
        let mut queries_last_bf = vec![0f32; batch * nq * 256];
        let li = num_layers - 1;
        for q in 0..nq {
            for b in 0..batch {
                let src = ((li * nq + q) * batch + b) * 256;
                let dst = (b * nq + q) * 256;
                queries_last_bf[dst..dst + 256]
                    .copy_from_slice(&dec.intermediate[src..src + 256]);
            }
        }

        // Last layer's refined reference boxes.
        let mut ref_last_bf = vec![0f32; batch * nq * 4];
        for q in 0..nq {
            for b in 0..batch {
                let src = ((li * nq + q) * batch + b) * 4;
                let dst = (b * nq + q) * 4;
                ref_last_bf[dst..dst + 4]
                    .copy_from_slice(&dec.intermediate_ref_boxes[src..src + 4]);
            }
        }

        // Final boxes: sigmoid(inv_sigmoid(ref_last) + bbox_embed(queries_last)).
        let delta = super::detector_decoder::bbox_embed_forward(
            &self.decoder,
            &queries_last_bf,
            batch * nq,
        )?;
        let mut final_boxes_cxcywh = vec![0f32; batch * nq * 4];
        for q in 0..nq {
            for b in 0..batch {
                let rb = &ref_last_bf[(b * nq + q) * 4..(b * nq + q + 1) * 4];
                let d = &delta[(b * nq + q) * 4..(b * nq + q + 1) * 4];
                let out_off = (b * nq + q) * 4;
                for k in 0..4 {
                    let inv = if rb[k] <= 0.0 {
                        ((1e-3f32 as f32) / (1.0 - 1e-3)).ln()
                    } else if rb[k] >= 1.0 {
                        ((1.0 - 1e-3) / 1e-3f32).ln()
                    } else {
                        (rb[k].max(1e-3) / (1.0 - rb[k]).max(1e-3)).ln()
                    };
                    let s = inv + d[k];
                    final_boxes_cxcywh[out_off + k] = 1.0 / (1.0 + (-s).exp());
                }
            }
        }
        // Convert to xyxy.
        let mut boxes_xyxy = vec![0f32; batch * nq * 4];
        for i in 0..(batch * nq) {
            let cx = final_boxes_cxcywh[i * 4];
            let cy = final_boxes_cxcywh[i * 4 + 1];
            let bw = final_boxes_cxcywh[i * 4 + 2];
            let bh = final_boxes_cxcywh[i * 4 + 3];
            boxes_xyxy[i * 4] = cx - 0.5 * bw;
            boxes_xyxy[i * 4 + 1] = cy - 0.5 * bh;
            boxes_xyxy[i * 4 + 2] = cx + 0.5 * bw;
            boxes_xyxy[i * 4 + 3] = cy + 0.5 * bh;
        }

        // Scores: dot product scoring, last layer.
        let mut hs_bf = vec![0f32; num_layers * batch * nq * 256];
        for l in 0..num_layers {
            for q in 0..nq {
                for b in 0..batch {
                    let src = ((l * nq + q) * batch + b) * 256;
                    let dst = ((l * batch + b) * nq + q) * 256;
                    hs_bf[dst..dst + 256]
                        .copy_from_slice(&dec.intermediate[src..src + 256]);
                }
            }
        }
        let all_scores = forward_dot_prod_scoring(
            &self.scoring,
            &hs_bf,
            &text_out.text_memory_resized,
            &text_out.attention_mask,
            num_layers,
            batch,
            nq,
            seq_len,
        )?;
        let last_scores =
            all_scores[(num_layers - 1) * batch * nq..num_layers * batch * nq].to_vec();

        // Segmentation.
        let seg = forward_segmentation(
            &self.seg_head,
            &memory_bf,
            &backbone_fpn,
            &backbone_shapes,
            &queries_last_bf,
            &text_out.text_memory_resized,
            &text_out.attention_mask,
            batch,
            h,
            w,
            nq,
            seq_len,
        )?;

        Ok(Sam3ImagePrediction {
            masks: seg.mask_pred,
            mask_shape: vec![batch, nq, seg.h_out, seg.w_out],
            boxes: boxes_xyxy,
            boxes_shape: vec![batch, nq, 4],
            scores: last_scores,
            scores_shape: vec![batch, nq],
            num_instances: nq,
            h_out: resized_hw.0,
            w_out: resized_hw.1,
        })
    }

    /// Forward the segmentation head: cross-attend encoder memory to the
    /// text prompt, run the pixel decoder, and emit per-query mask logits
    /// plus the semantic mask.
    #[allow(clippy::too_many_arguments)]
    pub fn run_segmentation(
        &self,
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
        forward_segmentation(
            &self.seg_head,
            enc_memory_bf,
            backbone_fpn,
            backbone_shapes,
            obj_queries_last_bf,
            prompt_seq_first,
            prompt_kpm,
            batch,
            enc_h,
            enc_w,
            num_queries,
            seq_len,
        )
    }

    /// Compute per-query, per-layer scores via mean-pooled text + linear
    /// projections + dot product.
    #[allow(clippy::too_many_arguments)]
    pub fn run_dot_prod_scoring(
        &self,
        hs_bf: &[f32],
        prompt_seq_first: &[f32],
        prompt_kpm: &[u8],
        num_layers: usize,
        batch: usize,
        num_queries: usize,
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        forward_dot_prod_scoring(
            &self.scoring,
            hs_bf,
            prompt_seq_first,
            prompt_kpm,
            num_layers,
            batch,
            num_queries,
            seq_len,
        )
    }

    /// Run the detector decoder. Inputs are the encoder memory in
    /// batch-first flat `[batch, h*w, 256]` plus matching positional
    /// encoding, and the text memory in seq-first `[seq, batch, 256]`.
    /// Returns intermediate layer outputs, refined boxes, and presence
    /// logits — the same triple the upstream model uses to derive scores
    /// and final box predictions.
    #[allow(clippy::too_many_arguments)]
    pub fn run_decoder(
        &self,
        memory: &[f32],
        memory_pos: &[f32],
        memory_text: &[f32],
        text_attention_mask: &[u8],
        batch: usize,
        h: usize,
        w: usize,
        seq_len: usize,
    ) -> Result<Sam3DecoderOutput> {
        forward_decoder(
            &self.decoder,
            memory,
            memory_pos,
            memory_text,
            text_attention_mask,
            batch,
            h,
            w,
            seq_len,
        )
    }

    /// Run the detector encoder fusion on a single FPN level + text
    /// prompt. Returns the encoded image memory in batch-first flat
    /// `[batch, h*w, 256]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_encoder(
        &self,
        src_bchw: &[f32],
        src_pos_bchw: &[f32],
        prompt_seq_first: &[f32],
        prompt_kpm: &[u8],
        batch: usize,
        src_h: usize,
        src_w: usize,
        prompt_len: usize,
    ) -> Result<Vec<f32>> {
        // Backend selection:
        //   RLX_SAM3_ENCODER_HOST=1 → host-side per-head sgemm (legacy).
        //   RLX_SAM3_ENCODER_DEVICE=metal → IR on Metal (default Cpu).
        if std::env::var("RLX_SAM3_ENCODER_HOST").is_ok() {
            return forward_encoder(
                &self.encoder,
                src_bchw,
                src_pos_bchw,
                prompt_seq_first,
                prompt_kpm,
                batch,
                src_h,
                src_w,
                prompt_len,
            );
        }
        let dev = match std::env::var("RLX_SAM3_ENCODER_DEVICE").ok().as_deref() {
            Some("metal") => Device::Metal,
            Some("mlx") => Device::Mlx,
            _ => Device::Cpu,
        };
        let _ = forward_encoder_ir; // silence unused if always _on
        forward_encoder_ir_on(
            &self.encoder,
            src_bchw,
            src_pos_bchw,
            prompt_seq_first,
            prompt_kpm,
            batch,
            src_h,
            src_w,
            prompt_len,
            dev,
        )
    }

    /// Run the text encoder on already-tokenized inputs. Returns the
    /// resized memory the detector consumes.
    pub fn encode_text_tokens(
        &self,
        tokens: &[u32],
        batch: usize,
        seq_len: usize,
    ) -> Result<Sam3TextEncoded> {
        encode_tokens(&self.text, tokens, batch, seq_len)
    }

    pub fn predict_neck(
        &self,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
    ) -> Result<Vec<super::neck::Sam3FeatureLevel>> {
        let vision = self
            .vision
            .as_ref()
            .context("SAM3 predict_neck requires native vision weights")?;
        let (image_nchw, _) = preprocess_image(image_u8, h_in, w_in);
        let vision_out = super::vision_encoder::encode_image_native(vision, &self.cfg.vit, &image_nchw)?;
        apply_neck_native(&self.neck, &vision_out)
    }

    pub fn patch_embed_image(
        &self,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
    ) -> Result<Sam3EncodedImage> {
        let vision = self
            .vision
            .as_ref()
            .context("SAM3 patch_embed_image requires native vision weights")?;
        let (image_nchw, resized_hw) = preprocess_image(image_u8, h_in, w_in);
        let patch_tokens = assemble_patch_tokens(&vision.pre, &image_nchw)?;
        Ok(Sam3EncodedImage {
            patch_tokens,
            grid: vision.pre.grid,
            embed_dim: vision.pre.embed_dim,
            resized_hw,
        })
    }

    pub fn predict_image(
        &mut self,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
        text_prompt: Option<&str>,
        boxes: Option<&[f32]>,
        points: Option<(&[f32], &[f32])>,
    ) -> Result<Sam3ImagePrediction> {
        self.predict_image_native(image_u8, h_in, w_in, text_prompt, boxes, points)
    }

    pub fn predict_video_frame(
        &mut self,
        state: &mut Sam3VideoState,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
        text_prompt: Option<&str>,
    ) -> Result<Sam3VideoFramePrediction> {
        let pred = self.predict_image_native(image_u8, h_in, w_in, text_prompt, None, None)?;
        Ok(tracker_forward_native(&self.tracker, state, pred))
    }

    fn predict_image_native(
        &mut self,
        image_u8: &[u8],
        h_in: usize,
        w_in: usize,
        text_prompt: Option<&str>,
        boxes: Option<&[f32]>,
        points: Option<(&[f32], &[f32])>,
    ) -> Result<Sam3ImagePrediction> {
        ensure!(
            image_u8.len() == h_in * w_in * 3,
            "SAM3 image must be RGB u8 with len {} (got {})",
            h_in * w_in * 3,
            image_u8.len()
        );
        let vision = self
            .vision
            .as_ref()
            .context("SAM3 predict_image requires native vision weights")?;
        let (image_nchw, resized_hw) = preprocess_image(image_u8, h_in, w_in);
        let vision_out = encode_image_native(vision, &self.cfg.vit, &image_nchw)?;
        let levels = apply_neck_native(&self.neck, &vision_out)?;
        let text = encode_text_native(&self.text, &self.cfg.text, text_prompt)?;
        let geometry = encode_geometry_native(&self.geometry, boxes, points);
        let det = detector_forward_native(
            &self.detector,
            &self.cfg.detector,
            &levels,
            &text,
            &geometry,
        )?;
        Ok(segmentation_forward_native(
            &self.seg,
            &det,
            resized_hw.0,
            resized_hw.1,
        ))
    }
}
