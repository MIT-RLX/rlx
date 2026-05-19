// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! DINOv2 graph builder — pre-norm ViT with LayerScale and plain GELU MLP.
//!
//! Two output modes (controlled by `cfg.num_classes`):
//!   - `> 0` → ImageNet classifier head matching candle's `forward`:
//!     `cat(cls_pooled, mean(patch_tokens))` → linear → `[B, num_classes]`.
//!   - `= 0` → encoder-only: emit `[B, seq, H]` of final-LN'd tokens.
//!
//! Weight key convention matches Meta / candle exactly so safetensors
//! checkpoints load without remapping.

use super::config::DinoV2Config;
use super::preprocess::DinoV2PreprocessWeights;
use crate::weight_map::WeightMap;
use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// Build the DINOv2 IR graph.
///
/// Input: `"hidden"` `[batch, seq, hidden_size]` — caller assembles
///   `[CLS, register_tokens…, projected_patches] + pos_embed`.
///   (See [`crate::dinov2::preprocess::assemble_hidden`].)
///
/// Output:
///   - `cfg.num_classes > 0`: classifier logits `[batch, num_classes]`
///   - `cfg.num_classes == 0`: final-LN'd token sequence `[batch, seq, hidden_size]`
pub fn build_dinov2_graph_sized(
    cfg: &DinoV2Config,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, DinoV2PreprocessWeights)> {
    let mut g = Graph::new("dinov2");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = h / nh;
    let eps = cfg.layer_norm_eps as f32;
    let seq = cfg.seq_len();

    // ── Preprocess weights (host-side patchify + token assembly) ──
    let preprocess = super::preprocess::extract_preprocess_weights(weights, cfg)?;

    // ── All-ones mask (vision has no padding tokens) ──
    let mask_data = vec![1.0f32; batch * seq];
    let mask_id = g.param("attn_mask", Shape::new(&[batch, seq], f));
    params.insert("attn_mask".into(), mask_data);

    // ── Input: pre-assembled [batch, seq, H] ──
    let hidden_input = g.input("hidden", Shape::new(&[batch, seq, h], f));

    // Per-layer debug bisection (RLX_DINOV2_DEBUG_LAYER=N).
    let debug_layer: Option<usize> = std::env::var("RLX_DINOV2_DEBUG_LAYER")
        .ok()
        .and_then(|s| s.parse().ok());
    let mut debug_outs: Vec<NodeId> = Vec::new();
    let mut debug_labels: Vec<&'static str> = Vec::new();
    let dbg = |outs: &mut Vec<NodeId>,
               labels: &mut Vec<&'static str>,
               li: usize,
               label: &'static str,
               id: NodeId| {
        if Some(li) == debug_layer {
            outs.push(id);
            labels.push(label);
        }
    };

    // ── Encoder blocks ──
    let mut x = hidden_input;
    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("blocks.{layer_idx}");

        // Pre-norm + attention
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
        let normed1 = g.ln(x, n1_g, n1_b, eps);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "01_ln1",
            normed1,
        );

        // Fused QKV (candle key: attn.qkv, with bias)
        let qkv_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.qkv.weight"),
            true,
        )?;
        let qkv_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.qkv.bias"),
            false,
        )?;
        let qkv_mm = g.mm(normed1, qkv_w);
        let qkv = g.add(qkv_mm, qkv_b);
        dbg(&mut debug_outs, &mut debug_labels, layer_idx, "02_qkv", qkv);

        let last_ax = g.shape(qkv).rank() - 1;
        let q = g.narrow_(qkv, last_ax, 0, h);
        let k = g.narrow_(qkv, last_ax, h, h);
        let v = g.narrow_(qkv, last_ax, 2 * h, h);

        let attn = g.attention_(q, k, v, mask_id, nh, dh);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "03_attn",
            attn,
        );

        // Output projection (candle key: attn.proj, with bias)
        let p_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.proj.weight"),
            true,
        )?;
        let p_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.proj.bias"),
            false,
        )?;
        let proj_mm = g.mm(attn, p_w);
        let attn_out = g.add(proj_mm, p_b);

        // LayerScale 1
        let ls1_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ls1.gamma"),
            false,
        )?;
        let scaled1 = g.mul(attn_out, ls1_g);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "04_ls1",
            scaled1,
        );

        x = g.add(x, scaled1);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "05_resid1",
            x,
        );

        // Pre-norm + MLP (plain GELU)
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
            &format!("{lp}.mlp.fc1.weight"),
            true,
        )?;
        let fc1_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc1.bias"),
            false,
        )?;
        let fc2_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc2.weight"),
            true,
        )?;
        let fc2_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc2.bias"),
            false,
        )?;

        let up_mm = g.mm(normed2, fc1_w);
        let up = g.add(up_mm, fc1_b);
        // Tanh-approximation GELU to match candle's `Tensor::gelu` —
        // the default PyTorch GELU formula. Using exact erf-GELU
        // accumulates ~5e-2 logit drift across 12 ViT layers.
        let act = g.gelu_approx(up);
        let down_mm = g.mm(act, fc2_w);
        let ffn = g.add(down_mm, fc2_b);
        dbg(&mut debug_outs, &mut debug_labels, layer_idx, "06_ffn", ffn);

        // LayerScale 2
        let ls2_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.ls2.gamma"),
            false,
        )?;
        let scaled2 = g.mul(ffn, ls2_g);

        x = g.add(x, scaled2);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "07_resid2",
            x,
        );
    }

    // ── Final LayerNorm ──
    let fn_g = load_p(&mut g, &mut params, weights, "norm.weight", false)?;
    let fn_b = load_p(&mut g, &mut params, weights, "norm.bias", false)?;
    let encoded = g.ln(x, fn_g, fn_b, eps);

    // ── Optional ImageNet classifier head (matches candle's forward) ──
    let final_output = if cfg.num_classes > 0 {
        // candle: cls = xs[:, 0, :]; mean = xs[:, 1:, :].mean(1); cat(cls, mean); head(x)
        // We skip register tokens (between CLS and patches) when pooling
        // patch tokens, mirroring candle's `dinov2reg4.rs`.
        let cls_slice = g.narrow_(encoded, 1, 0, 1); // [B, 1, H]
        let cls_flat = g.reshape_(cls_slice, vec![batch as i64, h as i64]);

        let patch_start = 1 + cfg.num_register_tokens;
        let patch_tokens = g.narrow_(encoded, 1, patch_start, cfg.num_patches()); // [B, np, H]
        let mean_patches = g.mean(patch_tokens, vec![1], false); // [B, H]

        let features = g.concat_(vec![cls_flat, mean_patches], 1); // [B, 2H]

        let head_w = load_p(&mut g, &mut params, weights, "head.weight", true)?;
        let head_b = load_p(&mut g, &mut params, weights, "head.bias", false)?;
        let logits_mm = g.mm(features, head_w);
        g.add(logits_mm, head_b)
    } else {
        encoded
    };

    let mut outputs = vec![final_output];
    outputs.extend(debug_outs);
    if !debug_labels.is_empty() {
        eprintln!("[dinov2] debug outputs: head/encoded + {:?}", debug_labels);
    }
    g.set_outputs(outputs);

    Ok((g, params, preprocess))
}

fn load_p(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    key: &str,
    transpose: bool,
) -> Result<NodeId> {
    let (data, shape) = if transpose {
        weights.take_transposed(key)?
    } else {
        weights.take(key)?
    };
    let name = key.to_string();
    let ir_shape = Shape::new(&shape, DType::F32);
    let id = g.param(&name, ir_shape);
    params.insert(name, data);
    Ok(id)
}
