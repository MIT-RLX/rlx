// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! SAM v1 — Meta's Segment Anything image-segmentation model.
//!
//! ## Phasing
//!
//! Phase 1 (this commit) lands the **image encoder** end-to-end:
//!   - Host-side preprocessing (resize-to-1024, ImageNet pixel
//!     normalization, zero-pad to 1024×1024, patch embedding via
//!     Conv2d-as-matmul).
//!   - IR graph for the 12 encoder blocks with **windowed + global**
//!     attention, **decomposed relative position embeddings**, plain
//!     GELU-tanh MLPs, pre-norm residual structure.
//!   - Host-side neck (Conv2d 1×1 → LN2d → Conv2d 3×3 padding=1 →
//!     LN2d → `[256, 64, 64]` image embeddings).
//!
//! **Phase 1 status:** 100% numerical parity with candle's
//! `ImageEncoderViT::forward()` on real `sam_vit_b_01ec64.safetensors`
//! weights — `max |Δ| = 7.15e-6` on the 1×256×64×64 image embeddings
//! (full 12-layer ViT-B at 1024×1024 input). Phase-1 bisect env vars
//! remain in `tests/sam_parity.rs` for future debugging:
//!   - `RLX_SAM_DEBUG_DEPTH=N` — run only the first N encoder blocks
//!   - `RLX_SAM_DEBUG_NO_RELPOS=1` — disable decomposed relative pos
//!   - `RLX_SAM_DEBUG_FORCE_GLOBAL=1` — force every block to use global attn
//!   - `RLX_SAM_DEBUG_ZERO_RELH=1` / `RLX_SAM_DEBUG_ZERO_RELW=1` — zero
//!     a single rel_pos axis (data only — the matmul + add still execute)
//!
//! Phase 2 (next commit) lands the **prompt encoder** + **mask decoder**:
//!   - Random Fourier positional encoding, point/box/mask embeddings.
//!   - Two-way transformer between prompt tokens and image embeddings.
//!   - ConvTranspose2d upscaling (likely host-side) + hypernetwork
//!     MLPs for mask + IoU output.
//!
//! Weight key convention matches Meta / candle exactly so the
//! `lmz/candle-sam` safetensors checkpoints load with no remapping.

pub mod config;
pub mod image_encoder;
pub mod mask_decoder;
pub mod preprocess;
pub mod prompt_encoder;
#[allow(clippy::module_inception)]
pub mod sam;
pub mod transformer;

pub use config::{
    EncoderKind, SAM_EMBED_HW, SAM_IMG_SIZE, SAM_PATCH_SIZE, SAM_PIXEL_MEAN, SAM_PIXEL_STD,
    SAM_PROMPT_EMBED_DIM, SamConfig, SamDecoderConfig, SamEncoderConfig,
};
pub use image_encoder::{NeckWeights, apply_neck_host, build_sam_encoder_graph};
pub use mask_decoder::{MaskDecoderWeights, mask_decoder_forward};
pub use preprocess::{SamPreprocessWeights, assemble_patch_tokens, preprocess_image};
pub use prompt_encoder::{PromptEncoderOutput, PromptEncoderWeights, prompt_encoder_forward};
pub use sam::{MaskPrediction, SAM_MASK_IN_CHANS, Sam, sam_vit_b_config};

/// Re-export `Device` so callers can construct it without depending
/// on `rlx-runtime` themselves.
pub use rlx_runtime::Device;
pub use transformer::{TwoWayTransformerWeights, attention_forward, two_way_transformer_forward};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_map::WeightMap;
    use std::collections::HashMap;

    /// Build a synthetic ViT-B WeightMap so we can verify the encoder
    /// graph builds without panicking. Real numerical parity needs the
    /// safetensors checkpoint — see `tests/sam_parity.rs`.
    fn synthetic_vit_b_weights() -> WeightMap {
        let cfg = SamEncoderConfig::vit_b();
        let e = cfg.embed_dim;
        let dh = cfg.head_dim();
        let int_dim = e * 4;
        let hw = SAM_EMBED_HW;
        let ws = cfg.window_size;
        let ps = SAM_PATCH_SIZE;
        let pd = 3 * ps * ps;

        let mut t: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let z = |n: usize| vec![0.0f32; n];

        t.insert(
            "image_encoder.patch_embed.proj.weight".into(),
            (z(e * pd), vec![e, 3, ps, ps]),
        );
        t.insert(
            "image_encoder.patch_embed.proj.bias".into(),
            (z(e), vec![e]),
        );
        t.insert(
            "image_encoder.pos_embed".into(),
            (z(hw * hw * e), vec![1, hw, hw, e]),
        );

        for i in 0..cfg.depth {
            let lp = format!("image_encoder.blocks.{i}");
            let is_global = cfg.global_attn_indexes.contains(&i);
            let rel_size = if is_global { hw } else { ws };

            t.insert(format!("{lp}.norm1.weight"), (z(e), vec![e]));
            t.insert(format!("{lp}.norm1.bias"), (z(e), vec![e]));
            t.insert(
                format!("{lp}.attn.qkv.weight"),
                (z(3 * e * e), vec![3 * e, e]),
            );
            t.insert(format!("{lp}.attn.qkv.bias"), (z(3 * e), vec![3 * e]));
            t.insert(format!("{lp}.attn.proj.weight"), (z(e * e), vec![e, e]));
            t.insert(format!("{lp}.attn.proj.bias"), (z(e), vec![e]));
            t.insert(
                format!("{lp}.attn.rel_pos_h"),
                (z((2 * rel_size - 1) * dh), vec![2 * rel_size - 1, dh]),
            );
            t.insert(
                format!("{lp}.attn.rel_pos_w"),
                (z((2 * rel_size - 1) * dh), vec![2 * rel_size - 1, dh]),
            );
            t.insert(format!("{lp}.norm2.weight"), (z(e), vec![e]));
            t.insert(format!("{lp}.norm2.bias"), (z(e), vec![e]));
            t.insert(
                format!("{lp}.mlp.lin1.weight"),
                (z(int_dim * e), vec![int_dim, e]),
            );
            t.insert(format!("{lp}.mlp.lin1.bias"), (z(int_dim), vec![int_dim]));
            t.insert(
                format!("{lp}.mlp.lin2.weight"),
                (z(e * int_dim), vec![e, int_dim]),
            );
            t.insert(format!("{lp}.mlp.lin2.bias"), (z(e), vec![e]));
        }
        // Neck
        t.insert(
            "image_encoder.neck.0.weight".into(),
            (z(cfg.out_chans * e), vec![cfg.out_chans, e, 1, 1]),
        );
        t.insert(
            "image_encoder.neck.1.weight".into(),
            (z(cfg.out_chans), vec![cfg.out_chans]),
        );
        t.insert(
            "image_encoder.neck.1.bias".into(),
            (z(cfg.out_chans), vec![cfg.out_chans]),
        );
        t.insert(
            "image_encoder.neck.2.weight".into(),
            (
                z(cfg.out_chans * cfg.out_chans * 9),
                vec![cfg.out_chans, cfg.out_chans, 3, 3],
            ),
        );
        t.insert(
            "image_encoder.neck.3.weight".into(),
            (z(cfg.out_chans), vec![cfg.out_chans]),
        );
        t.insert(
            "image_encoder.neck.3.bias".into(),
            (z(cfg.out_chans), vec![cfg.out_chans]),
        );

        WeightMap::from_tensors(t)
    }

    #[test]
    fn encoder_graph_builds() {
        let cfg = SamEncoderConfig::vit_b();
        let mut wm = synthetic_vit_b_weights();
        let (g, _params, _pre, _neck) = build_sam_encoder_graph(&cfg, &mut wm).unwrap();
        assert_eq!(g.outputs.len(), 1);
        // [1, hw·hw, embed_dim]
        let s = g.shape(g.outputs[0]);
        let dims: Vec<usize> = s.dims().iter().map(|d| d.unwrap_static()).collect();
        assert_eq!(dims, vec![1, SAM_EMBED_HW * SAM_EMBED_HW, cfg.embed_dim]);
        // All non-preprocess weights must be drained.
        let leftovers: Vec<&str> = wm.keys().collect();
        assert!(leftovers.is_empty(), "leftover weights: {leftovers:?}");
    }

    #[test]
    fn preprocess_round_trip_shapes() {
        // 100×80 RGB image → padded to 1024×1024 NCHW; new_h, new_w
        // preserve aspect ratio with long side = 1024.
        let img = vec![128u8; 100 * 80 * 3];
        let (nchw, (h, w)) = preprocess_image(&img, 100, 80);
        assert_eq!(nchw.len(), 3 * 1024 * 1024);
        assert_eq!(h, 1024);
        assert_eq!(w, (80.0_f32 * (1024.0 / 100.0)).round() as usize);
    }
}
