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

//! BERT graph builder — constructs RLX IR from config + weights.

use crate::config::BertConfig;
use crate::weight_map::WeightMap;
use anyhow::Result;
use rlx_ir::infer::GraphExt;
use rlx_ir::*;
use std::collections::HashMap;

/// Build a BERT encoder IR graph from config and weights.
///
/// Returns the graph and a map of param_name → weight data.
/// The graph expects inputs: `input_ids [B,S]`, `attention_mask [B,S]`, `token_type_ids [B,S]`.
/// Output: `hidden_states [B, S, H]`.
/// Build a BERT encoder IR graph.
///
/// `batch` and `seq` are the concrete dimensions for this compilation.
/// The graph will be compiled for exactly these dimensions.
/// Call again with different dims to recompile for a different size.
pub fn build_bert_graph(
    cfg: &BertConfig,
    weights: &mut WeightMap,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    build_bert_graph_sized(cfg, weights, 1, 1)
}

pub fn build_bert_graph_sized(
    cfg: &BertConfig,
    weights: &mut WeightMap,
    batch: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("bert");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let dh = cfg.head_dim();
    let int_dim = cfg.intermediate_size;
    let eps = cfg.layer_norm_eps as f32;

    // Detect key prefix
    let prefix = if weights.has("bert.embeddings.word_embeddings.weight") {
        "bert."
    } else {
        ""
    };

    // ── Embedding params ────────────────────────────────────────
    let word_emb = load_param(
        &mut g,
        &mut params,
        weights,
        &format!("{prefix}embeddings.word_embeddings.weight"),
        &[cfg.vocab_size, h],
        false,
    )?;
    let pos_emb = load_param(
        &mut g,
        &mut params,
        weights,
        &format!("{prefix}embeddings.position_embeddings.weight"),
        &[cfg.max_position_embeddings, h],
        false,
    )?;
    let tt_emb = load_param(
        &mut g,
        &mut params,
        weights,
        &format!("{prefix}embeddings.token_type_embeddings.weight"),
        &[cfg.type_vocab_size, h],
        false,
    )?;
    let emb_ln_g = load_param(
        &mut g,
        &mut params,
        weights,
        &format!("{prefix}embeddings.LayerNorm.weight"),
        &[h],
        false,
    )?;
    let emb_ln_b = load_param(
        &mut g,
        &mut params,
        weights,
        &format!("{prefix}embeddings.LayerNorm.bias"),
        &[h],
        false,
    )?;

    // ── Inputs (concrete batch × seq) ─────────────────────────
    let input_ids = g.input("input_ids", Shape::new(&[batch, seq], DType::F32));
    let _attention_mask = g.input("attention_mask", Shape::new(&[batch, seq], f));
    let token_type_ids = g.input("token_type_ids", Shape::new(&[batch, seq], DType::F32));
    let pos_ids = g.input("position_ids", Shape::new(&[batch, seq], DType::F32));

    // ── Embedding lookup → [batch, seq, H] ──────────────────
    let word_out = g.gather_(word_emb, input_ids, 0);
    let pos_out = g.gather_(pos_emb, pos_ids, 0);
    let tt_out = g.gather_(tt_emb, token_type_ids, 0);

    let wp = g.add(word_out, pos_out);
    let emb_sum = g.add(wp, tt_out);
    let hidden = g.ln(emb_sum, emb_ln_g, emb_ln_b, eps);

    // ── Encoder layers ──────────────────────────────────────────
    let mut h_id = hidden;

    // Detect attention key style
    let bert_style = weights.has(&format!(
        "{prefix}encoder.layer.0.attention.self.query.weight"
    ));

    for layer_idx in 0..cfg.num_hidden_layers {
        // Both BERT-style and non-BERT-style HF checkpoints publish the
        // per-layer prefix as `encoder.layer.{i}` — the structural
        // difference shows up further down (fused-vs-split QKV).
        let lp = format!("{prefix}encoder.layer.{layer_idx}");

        // ── QKV projection (fused) ──────────────────────────
        let (qkv_w, qkv_b) = if bert_style {
            load_fused_qkv(&mut g, &mut params, weights, &lp, h, nh, dh)?
        } else {
            // mpnet style
            load_fused_qkv_mpnet(&mut g, &mut params, weights, &lp, h, nh, dh)?
        };

        let qkv_mm = g.mm(h_id, qkv_w);
        let qkv = g.add(qkv_mm, qkv_b);

        // Split Q/K/V along last axis
        let last_ax = g.shape(qkv).rank() - 1;
        let q = g.narrow_(qkv, last_ax, 0, h);
        let k = g.narrow_(qkv, last_ax, h, h);
        let v = g.narrow_(qkv, last_ax, 2 * h, h);

        // Attention
        let attn = g.attention_(q, k, v, _attention_mask, nh, dh);

        // Output projection
        let out_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attention.output.dense.weight"),
            &[h, h],
            true,
        )?;
        let out_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attention.output.dense.bias"),
            &[h],
            false,
        )?;
        let attn_mm = g.mm(attn, out_w);
        let attn_out = g.add(attn_mm, out_b);

        // Residual + LN1
        let ln1_g = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attention.output.LayerNorm.weight"),
            &[h],
            false,
        )?;
        let ln1_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.attention.output.LayerNorm.bias"),
            &[h],
            false,
        )?;
        let res1 = g.add(attn_out, h_id);
        let normed1 = g.ln(res1, ln1_g, ln1_b, eps);

        // FFN intermediate
        let int_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.intermediate.dense.weight"),
            &[h, int_dim],
            true,
        )?;
        let int_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.intermediate.dense.bias"),
            &[int_dim],
            false,
        )?;
        let int_mm = g.mm(normed1, int_w);
        let int_add = g.add(int_mm, int_b);
        let ffn_int = g.gelu(int_add);

        // FFN output
        let out2_w = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.output.dense.weight"),
            &[int_dim, h],
            true,
        )?;
        let out2_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.output.dense.bias"),
            &[h],
            false,
        )?;
        let out2_mm = g.mm(ffn_int, out2_w);
        let ffn_out = g.add(out2_mm, out2_b);

        // Residual + LN2
        let ln2_g = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.output.LayerNorm.weight"),
            &[h],
            false,
        )?;
        let ln2_b = load_param(
            &mut g,
            &mut params,
            weights,
            &format!("{lp}.output.LayerNorm.bias"),
            &[h],
            false,
        )?;
        let res2 = g.add(ffn_out, normed1);
        h_id = g.ln(res2, ln2_g, ln2_b, eps);
    }

    g.set_outputs(vec![h_id]);
    Ok((g, params))
}

/// Load a parameter: register in graph + store weight data.
fn load_param(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    key: &str,
    _expected_shape: &[usize],
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

/// Fuse Q/K/V weights into single [H, 3H] matrix (BERT-style keys).
fn load_fused_qkv(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    layer_prefix: &str,
    h: usize,
    _nh: usize,
    _dh: usize,
) -> Result<(NodeId, NodeId)> {
    let (wq, _) =
        weights.take_transposed(&format!("{layer_prefix}.attention.self.query.weight"))?;
    let (wk, _) = weights.take_transposed(&format!("{layer_prefix}.attention.self.key.weight"))?;
    let (wv, _) =
        weights.take_transposed(&format!("{layer_prefix}.attention.self.value.weight"))?;

    let bq = weights
        .take(&format!("{layer_prefix}.attention.self.query.bias"))?
        .0;
    let bk = weights
        .take(&format!("{layer_prefix}.attention.self.key.bias"))?
        .0;
    let bv = weights
        .take(&format!("{layer_prefix}.attention.self.value.bias"))?
        .0;

    // Concatenate: [H, H] + [H, H] + [H, H] → [H, 3H]
    let mut fused_w = vec![0f32; h * 3 * h];
    let mut fused_b = vec![0f32; 3 * h];
    for row in 0..h {
        fused_w[row * 3 * h..row * 3 * h + h].copy_from_slice(&wq[row * h..(row + 1) * h]);
        fused_w[row * 3 * h + h..row * 3 * h + 2 * h].copy_from_slice(&wk[row * h..(row + 1) * h]);
        fused_w[row * 3 * h + 2 * h..row * 3 * h + 3 * h]
            .copy_from_slice(&wv[row * h..(row + 1) * h]);
    }
    fused_b[..h].copy_from_slice(&bq);
    fused_b[h..2 * h].copy_from_slice(&bk);
    fused_b[2 * h..].copy_from_slice(&bv);

    let w_name = format!("{layer_prefix}.attention.qkv.weight");
    let b_name = format!("{layer_prefix}.attention.qkv.bias");
    let w_id = g.param(&w_name, Shape::new(&[h, 3 * h], DType::F32));
    let b_id = g.param(&b_name, Shape::new(&[3 * h], DType::F32));
    params.insert(w_name, fused_w);
    params.insert(b_name, fused_b);

    Ok((w_id, b_id))
}

/// mpnet-style QKV fusion (different key names).
fn load_fused_qkv_mpnet(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    weights: &mut WeightMap,
    layer_prefix: &str,
    h: usize,
    nh: usize,
    dh: usize,
) -> Result<(NodeId, NodeId)> {
    // Try mpnet keys
    let q_key = format!("{layer_prefix}.attention.attn.q.weight");
    if weights.has(&q_key) {
        let (wq, _) = weights.take_transposed(&q_key)?;
        let (wk, _) =
            weights.take_transposed(&format!("{layer_prefix}.attention.attn.k.weight"))?;
        let (wv, _) =
            weights.take_transposed(&format!("{layer_prefix}.attention.attn.v.weight"))?;
        let bq = weights
            .take(&format!("{layer_prefix}.attention.attn.q.bias"))?
            .0;
        let bk = weights
            .take(&format!("{layer_prefix}.attention.attn.k.bias"))?
            .0;
        let bv = weights
            .take(&format!("{layer_prefix}.attention.attn.v.bias"))?
            .0;

        let mut fused_w = vec![0f32; h * 3 * h];
        let mut fused_b = vec![0f32; 3 * h];
        for row in 0..h {
            fused_w[row * 3 * h..row * 3 * h + h].copy_from_slice(&wq[row * h..(row + 1) * h]);
            fused_w[row * 3 * h + h..row * 3 * h + 2 * h]
                .copy_from_slice(&wk[row * h..(row + 1) * h]);
            fused_w[row * 3 * h + 2 * h..row * 3 * h + 3 * h]
                .copy_from_slice(&wv[row * h..(row + 1) * h]);
        }
        fused_b[..h].copy_from_slice(&bq);
        fused_b[h..2 * h].copy_from_slice(&bk);
        fused_b[2 * h..].copy_from_slice(&bv);

        let w_name = format!("{layer_prefix}.attention.qkv.weight");
        let b_name = format!("{layer_prefix}.attention.qkv.bias");
        let w_id = g.param(&w_name, Shape::new(&[h, 3 * h], DType::F32));
        let b_id = g.param(&b_name, Shape::new(&[3 * h], DType::F32));
        params.insert(w_name, fused_w);
        params.insert(b_name, fused_b);
        return Ok((w_id, b_id));
    }

    // Fallback: already-fused QKV
    let fused_key = format!("{layer_prefix}.attention.self.qkv.weight");
    if weights.has(&fused_key) {
        let (data, _) = weights.take_transposed(&fused_key)?;
        let bias = weights
            .take(&format!("{layer_prefix}.attention.self.qkv.bias"))?
            .0;
        let w_name = format!("{layer_prefix}.attention.qkv.weight");
        let b_name = format!("{layer_prefix}.attention.qkv.bias");
        let w_id = g.param(&w_name, Shape::new(&[h, 3 * h], DType::F32));
        let b_id = g.param(&b_name, Shape::new(&[3 * h], DType::F32));
        params.insert(w_name, data);
        params.insert(b_name, bias);
        return Ok((w_id, b_id));
    }

    // Fallback to BERT style
    load_fused_qkv(g, params, weights, layer_prefix, h, nh, dh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_tiny_bert_graph() {
        // Create a minimal config
        let cfg = BertConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 256,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
        };

        // Create fake weights
        let h = cfg.hidden_size;
        let int = cfg.intermediate_size;
        let mut tensors = HashMap::new();
        let add = |m: &mut HashMap<String, (Vec<f32>, Vec<usize>)>, k: &str, shape: Vec<usize>| {
            let size: usize = shape.iter().product();
            m.insert(k.to_string(), (vec![0.01f32; size], shape));
        };

        // Embeddings
        add(
            &mut tensors,
            "embeddings.word_embeddings.weight",
            vec![100, h],
        );
        add(
            &mut tensors,
            "embeddings.position_embeddings.weight",
            vec![32, h],
        );
        add(
            &mut tensors,
            "embeddings.token_type_embeddings.weight",
            vec![2, h],
        );
        add(&mut tensors, "embeddings.LayerNorm.weight", vec![h]);
        add(&mut tensors, "embeddings.LayerNorm.bias", vec![h]);

        // Layer 0 — attention
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.query.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.query.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.key.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.key.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.value.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.self.value.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.dense.weight",
            vec![h, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.dense.bias",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.LayerNorm.weight",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.attention.output.LayerNorm.bias",
            vec![h],
        );

        // Layer 0 — FFN
        add(
            &mut tensors,
            "encoder.layer.0.intermediate.dense.weight",
            vec![int, h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.intermediate.dense.bias",
            vec![int],
        );
        add(
            &mut tensors,
            "encoder.layer.0.output.dense.weight",
            vec![h, int],
        );
        add(&mut tensors, "encoder.layer.0.output.dense.bias", vec![h]);
        add(
            &mut tensors,
            "encoder.layer.0.output.LayerNorm.weight",
            vec![h],
        );
        add(
            &mut tensors,
            "encoder.layer.0.output.LayerNorm.bias",
            vec![h],
        );

        let mut wm = WeightMap::from_tensors(tensors);
        let (graph, params) = build_bert_graph(&cfg, &mut wm).unwrap();

        println!("{graph}");
        println!("Nodes: {}, Params: {}", graph.len(), params.len());

        // Verify graph is valid
        let errors = rlx_ir::verify::verify(&graph);
        assert!(errors.is_empty(), "verification errors: {errors:?}");

        // Should have params for all weights
        assert!(
            params.len() >= 15,
            "expected 15+ params, got {}",
            params.len()
        );

        // Output should exist
        assert!(!graph.outputs.is_empty());
    }
}
