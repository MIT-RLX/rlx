// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! NomicVision graph builder — patch embedding, pre-norm, biased SwiGLU + intermediate LN.

use crate::config::NomicVisionConfig;
use crate::weight_map::WeightMap;
use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// Build a NomicVision encoder IR graph.
///
/// Single input: "hidden" [batch, seq, hidden_size]
/// — caller assembles CLS + projected_patches + pos_embed before calling.
///
/// Output: CLS embedding [batch, hidden_size]
///
/// Also returns projection weights + CLS token + pos_embed for the caller
/// to use during preprocessing.
pub fn build_vision_graph_sized(
    cfg: &NomicVisionConfig,
    weights: &mut WeightMap,
    batch: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>, VisionPreprocessWeights)> {
    let mut g = Graph::new("nomic_vision");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = h / nh;
    let _int_dim = cfg.intermediate_size();
    let eps = cfg.layer_norm_eps() as f32;
    let ps = cfg.patch_size;
    let np = (cfg.img_size / ps) * (cfg.img_size / ps);
    let seq = np + 1; // +1 for CLS token
    let _patch_dim = 3 * ps * ps;

    // ── Extract preprocessing weights (returned to caller) ──────
    let (proj_w_data, proj_w_shape) = weights.take_transposed("embeddings.proj.weight")?;
    let (proj_b_data, _) = weights.take("embeddings.proj.bias")?;
    let (cls_token_data, _) = weights.take("embeddings.cls_token")?;
    let (pos_embed_data, _) = weights.take("embeddings.pos_embed")?;

    let preprocess = VisionPreprocessWeights {
        proj_w: proj_w_data,
        proj_w_cols: proj_w_shape.last().copied().unwrap_or(h),
        proj_b: proj_b_data,
        cls_token: cls_token_data,
        pos_embed: pos_embed_data,
    };

    // ── Constant all-ones mask (vision has no padding) ──────────
    let mask_data = vec![1.0f32; batch * seq];
    let mask_id = g.param("attn_mask", Shape::new(&[batch, seq], f));
    params.insert("attn_mask".into(), mask_data);

    // ── Input: pre-assembled [batch, seq, H] ────────────────────
    let hidden_input = g.input("hidden", Shape::new(&[batch, seq, h], f));

    // RLX_VISION_DEBUG_LAYER=N — append every intermediate of layer N
    // to graph outputs. Used for per-step CPU-vs-WGPU diff bisection.
    let debug_layer: Option<usize> = std::env::var("RLX_VISION_DEBUG_LAYER")
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

    // ── Encoder layers (pre-norm, biased, SwiGLU + intermediate LN) ──
    let mut h_id = hidden_input;

    for layer_idx in 0..cfg.num_hidden_layers {
        let lp = format!("layers.{layer_idx}");

        // Pre-norm: LN1 BEFORE attention
        let ln1_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm1.weight"),
            false,
        )?;
        let ln1_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm1.bias"),
            false,
        )?;
        let normed1 = g.ln(h_id, ln1_g, ln1_b, eps);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "01_ln1",
            normed1,
        );

        // QKV (WITH bias)
        let qkv_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.Wqkv.weight"),
            true,
        )?;
        let qkv_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.Wqkv.bias"),
            false,
        )?;
        let qkv_mm = g.mm(normed1, qkv_w);
        let qkv = g.add(qkv_mm, qkv_b);
        dbg(&mut debug_outs, &mut debug_labels, layer_idx, "02_qkv", qkv);

        // Split Q/K/V
        let last_ax = g.shape(qkv).rank() - 1;
        let q = g.narrow_(qkv, last_ax, 0, h);
        let k = g.narrow_(qkv, last_ax, h, h);
        let v = g.narrow_(qkv, last_ax, 2 * h, h);
        dbg(&mut debug_outs, &mut debug_labels, layer_idx, "03_q", q);
        dbg(&mut debug_outs, &mut debug_labels, layer_idx, "04_k", k);
        dbg(&mut debug_outs, &mut debug_labels, layer_idx, "05_v", v);

        // Attention (constant all-ones mask — all patches valid)
        let attn = g.attention_(q, k, v, mask_id, nh, dh);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "06_attn",
            attn,
        );

        // Output projection (WITH bias)
        let out_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.out_proj.weight"),
            true,
        )?;
        let out_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attn.out_proj.bias"),
            false,
        )?;
        let attn_mm = g.mm(attn, out_w);
        let attn_out = g.add(attn_mm, out_b);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "07_attn_out",
            attn_out,
        );

        // Residual: h += attn_out
        h_id = g.add(h_id, attn_out);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "08_resid1",
            h_id,
        );

        // Pre-norm: LN2 BEFORE FFN
        let ln2_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm2.weight"),
            false,
        )?;
        let ln2_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.norm2.bias"),
            false,
        )?;
        let normed2 = g.ln(h_id, ln2_g, ln2_b, eps);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "09_ln2",
            normed2,
        );

        // SwiGLU with bias + intermediate LN: fc2(norm(fc11(x) * silu(fc12(x))))
        let fc11_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc11.weight"),
            true,
        )?;
        let fc11_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc11.bias"),
            false,
        )?;
        let fc12_w = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc12.weight"),
            true,
        )?;
        let fc12_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.fc12.bias"),
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
        let mlp_ln_g = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.norm.weight"),
            false,
        )?;
        let mlp_ln_b = load_p(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.mlp.norm.bias"),
            false,
        )?;

        let up_mm = g.mm(normed2, fc11_w);
        let up = g.add(up_mm, fc11_b);
        dbg(&mut debug_outs, &mut debug_labels, layer_idx, "10_up", up);
        let gate_mm = g.mm(normed2, fc12_w);
        let gate_bias = g.add(gate_mm, fc12_b);
        let gate = g.silu(gate_bias);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "11_silu_gate",
            gate,
        );
        let swiglu = g.mul(up, gate);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "12_swiglu",
            swiglu,
        );

        // Intermediate LayerNorm (unique to vision SwiGLU)
        let normed_swiglu = g.ln(swiglu, mlp_ln_g, mlp_ln_b, eps);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "13_int_ln",
            normed_swiglu,
        );

        // Down projection
        let down_mm = g.mm(normed_swiglu, fc2_w);
        let ffn_out = g.add(down_mm, fc2_b);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "14_ffn_out",
            ffn_out,
        );

        // Residual: h += ffn_out
        h_id = g.add(h_id, ffn_out);
        dbg(
            &mut debug_outs,
            &mut debug_labels,
            layer_idx,
            "15_resid2",
            h_id,
        );
    }

    // Final LayerNorm — detect key name
    let final_ln_key = if weights.has("norm.weight") {
        "norm"
    } else if weights.has("selector.norm1.weight") {
        "selector.norm1"
    } else {
        "encoder.norm"
    };
    let fln_g = load_p(
        &mut g,
        &mut params,
        weights,
        &format!("{final_ln_key}.weight"),
        false,
    )?;
    let fln_b = load_p(
        &mut g,
        &mut params,
        weights,
        &format!("{final_ln_key}.bias"),
        false,
    )?;
    let final_out = g.ln(h_id, fln_g, fln_b, eps);

    // CLS pooling: extract first token
    let cls_out = g.narrow_(final_out, 1, 0, 1);
    let cls_flat = g.reshape_(cls_out, vec![batch as i64, h as i64]);

    let mut outputs = vec![cls_flat];
    outputs.extend(debug_outs);
    if !debug_labels.is_empty() {
        eprintln!("[vision] debug outputs: cls + {:?}", debug_labels);
    }
    g.set_outputs(outputs);
    Ok((g, params, preprocess))
}

/// Preprocessing weights extracted from safetensors for the caller to
/// assemble the "hidden" input before graph execution.
pub struct VisionPreprocessWeights {
    /// Patch projection weight [patch_dim, H] (pre-transposed for sgemm)
    pub proj_w: Vec<f32>,
    /// Number of columns in proj_w (= hidden_size)
    pub proj_w_cols: usize,
    /// Patch projection bias \[H\]
    pub proj_b: Vec<f32>,
    /// CLS token \[H\] (or [1, 1, H] flattened)
    pub cls_token: Vec<f32>,
    /// Position embeddings [1+np, H] (or [1, 1+np, H] flattened)
    pub pos_embed: Vec<f32>,
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
