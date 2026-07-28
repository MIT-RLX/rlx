// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Llama-like mlx-lm graph construction from config + packed Linears.
//!
//! Prefill includes NeoX RoPE on Q/K. Decode concatenates past K/V
//! (caller maintains the KV cache across steps).

use anyhow::{Result, bail};
use rlx_ir::op::{Activation, BinaryOp, MaskKind, RopeStyle};
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

use crate::config::MlxArchConfig;
use crate::graph::PackedLinearBinding;
use crate::rope::{build_default_tables, f32_le_bytes};

fn dq(g: &mut Graph, x: NodeId, b: &PackedLinearBinding, batch_seq: usize) -> Result<NodeId> {
    let (n, _k) = (b.packed.out_shape[0], b.packed.out_shape[1]);
    let n_groups = b.packed.n_groups().max(1);
    let w = g.param(
        format!("{}.weight", b.name),
        Shape::new(&[b.packed.w_q.len()], DType::U8),
    );
    let s = g.param(
        format!("{}.scales", b.name),
        Shape::new(&[n, n_groups], b.packed.scale_dtype()),
    );
    let z = g.param(
        format!("{}.biases", b.name),
        Shape::new(&[n, n_groups], b.packed.bias_dtype()),
    );
    Ok(g.add_node(
        Op::DequantMatMul {
            scheme: b.packed.scheme,
        },
        vec![x, w, s, z],
        Shape::new(&[batch_seq, n], DType::F32),
    ))
}

fn find<'a>(linears: &'a [PackedLinearBinding], name: &str) -> Result<&'a PackedLinearBinding> {
    linears
        .iter()
        .find(|b| b.name == name)
        .ok_or_else(|| anyhow::anyhow!("missing packed linear {name}"))
}

fn rope_tables(g: &mut Graph, arch: &MlxArchConfig, seq: usize) -> (NodeId, NodeId) {
    let hd = arch.head_dim();
    let half = hd / 2;
    let (cos, sin) = build_default_tables(arch.rope_theta as f64, hd, seq);
    let cos_n = g.add_node(
        Op::Constant {
            data: f32_le_bytes(&cos),
        },
        vec![],
        Shape::new(&[seq, half], DType::F32),
    );
    let sin_n = g.add_node(
        Op::Constant {
            data: f32_le_bytes(&sin),
        },
        vec![],
        Shape::new(&[seq, half], DType::F32),
    );
    (cos_n, sin_n)
}

fn apply_rope(
    g: &mut Graph,
    x4: NodeId,
    cos: NodeId,
    sin: NodeId,
    batch: usize,
    seq: usize,
    n_heads: usize,
    hd: usize,
) -> NodeId {
    let flat = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, seq as i64, (n_heads * hd) as i64],
        },
        vec![x4],
        Shape::new(&[batch, seq, n_heads * hd], DType::F32),
    );
    let rot = g.add_node(
        Op::Rope {
            head_dim: hd,
            n_rot: hd,
            style: RopeStyle::NeoX,
        },
        vec![flat, cos, sin],
        Shape::new(&[batch, seq, n_heads * hd], DType::F32),
    );
    g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, seq as i64, n_heads as i64, hd as i64],
        },
        vec![rot],
        Shape::new(&[batch, seq, n_heads, hd], DType::F32),
    )
}

/// Build a single mlx-lm Llama-style decoder layer (prefill, no KV cache).
pub fn build_llama_decoder_layer(
    g: &mut Graph,
    arch: &MlxArchConfig,
    layer_idx: usize,
    linears: &[PackedLinearBinding],
    hidden: NodeId,
    batch: usize,
    seq: usize,
    cos: NodeId,
    sin: NodeId,
) -> Result<NodeId> {
    let h = arch.hidden_size;
    let batch_seq = batch * seq;
    let prefix = format!("model.layers.{layer_idx}");
    let eps = arch.rms_norm_eps;

    let ln1_g = g.param(
        format!("{prefix}.input_layernorm.weight"),
        Shape::new(&[h], DType::F32),
    );
    let ln1_b = g.param(
        format!("{prefix}.input_layernorm.bias_zero"),
        Shape::new(&[h], DType::F32),
    );
    let n1 = g.add_node(
        Op::RmsNorm { axis: -1, eps },
        vec![hidden, ln1_g, ln1_b],
        Shape::new(&[batch_seq, h], DType::F32),
    );

    let q = dq(
        g,
        n1,
        find(linears, &format!("{prefix}.self_attn.q_proj"))?,
        batch_seq,
    )?;
    let k = dq(
        g,
        n1,
        find(linears, &format!("{prefix}.self_attn.k_proj"))?,
        batch_seq,
    )?;
    let v = dq(
        g,
        n1,
        find(linears, &format!("{prefix}.self_attn.v_proj"))?,
        batch_seq,
    )?;

    let nh = arch.num_attention_heads;
    let nkv = arch.num_key_value_heads;
    let hd = arch.head_dim();
    let q4 = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, seq as i64, nh as i64, hd as i64],
        },
        vec![q],
        Shape::new(&[batch, seq, nh, hd], DType::F32),
    );
    let k4 = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, seq as i64, nkv as i64, hd as i64],
        },
        vec![k],
        Shape::new(&[batch, seq, nkv, hd], DType::F32),
    );
    let v4 = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, seq as i64, nkv as i64, hd as i64],
        },
        vec![v],
        Shape::new(&[batch, seq, nkv, hd], DType::F32),
    );
    let q_r = apply_rope(g, q4, cos, sin, batch, seq, nh, hd);
    let k_r = apply_rope(g, k4, cos, sin, batch, seq, nkv, hd);
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q_r, k_r, v4],
        Shape::new(&[batch, seq, nh, hd], DType::F32),
    );
    let attn_flat = g.add_node(
        Op::Reshape {
            new_shape: vec![batch_seq as i64, (nh * hd) as i64],
        },
        vec![attn],
        Shape::new(&[batch_seq, nh * hd], DType::F32),
    );
    let o = dq(
        g,
        attn_flat,
        find(linears, &format!("{prefix}.self_attn.o_proj"))?,
        batch_seq,
    )?;
    let h1 = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![hidden, o],
        Shape::new(&[batch_seq, h], DType::F32),
    );

    let ln2_g = g.param(
        format!("{prefix}.post_attention_layernorm.weight"),
        Shape::new(&[h], DType::F32),
    );
    let ln2_b = g.param(
        format!("{prefix}.post_attention_layernorm.bias_zero"),
        Shape::new(&[h], DType::F32),
    );
    let n2 = g.add_node(
        Op::RmsNorm { axis: -1, eps },
        vec![h1, ln2_g, ln2_b],
        Shape::new(&[batch_seq, h], DType::F32),
    );

    let gate = dq(
        g,
        n2,
        find(linears, &format!("{prefix}.mlp.gate_proj"))?,
        batch_seq,
    )?;
    let up = dq(
        g,
        n2,
        find(linears, &format!("{prefix}.mlp.up_proj"))?,
        batch_seq,
    )?;
    let gate_s = g.add_node(
        Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[batch_seq, arch.intermediate_size], DType::F32),
    );
    let ff = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![gate_s, up],
        Shape::new(&[batch_seq, arch.intermediate_size], DType::F32),
    );
    let down = dq(
        g,
        ff,
        find(linears, &format!("{prefix}.mlp.down_proj"))?,
        batch_seq,
    )?;
    Ok(g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![h1, down],
        Shape::new(&[batch_seq, h], DType::F32),
    ))
}

/// One decode layer: `seq=1`, concat past K/V, return `(hidden, new_k, new_v)`.
fn build_llama_decode_layer(
    g: &mut Graph,
    arch: &MlxArchConfig,
    layer_idx: usize,
    linears: &[PackedLinearBinding],
    hidden: NodeId,
    batch: usize,
    past_len: usize,
    past_k: NodeId,
    past_v: NodeId,
    cos: NodeId,
    sin: NodeId,
) -> Result<(NodeId, NodeId, NodeId)> {
    let h = arch.hidden_size;
    let seq = 1usize;
    let batch_seq = batch * seq;
    let prefix = format!("model.layers.{layer_idx}");
    let eps = arch.rms_norm_eps;
    let nh = arch.num_attention_heads;
    let nkv = arch.num_key_value_heads;
    let hd = arch.head_dim();
    let kv_len = past_len + 1;

    let ln1_g = g.param(
        format!("{prefix}.input_layernorm.weight"),
        Shape::new(&[h], DType::F32),
    );
    let ln1_b = g.param(
        format!("{prefix}.input_layernorm.bias_zero"),
        Shape::new(&[h], DType::F32),
    );
    let n1 = g.add_node(
        Op::RmsNorm { axis: -1, eps },
        vec![hidden, ln1_g, ln1_b],
        Shape::new(&[batch_seq, h], DType::F32),
    );

    let q = dq(
        g,
        n1,
        find(linears, &format!("{prefix}.self_attn.q_proj"))?,
        batch_seq,
    )?;
    let k = dq(
        g,
        n1,
        find(linears, &format!("{prefix}.self_attn.k_proj"))?,
        batch_seq,
    )?;
    let v = dq(
        g,
        n1,
        find(linears, &format!("{prefix}.self_attn.v_proj"))?,
        batch_seq,
    )?;

    let q4 = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, 1, nh as i64, hd as i64],
        },
        vec![q],
        Shape::new(&[batch, 1, nh, hd], DType::F32),
    );
    let k4 = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, 1, nkv as i64, hd as i64],
        },
        vec![k],
        Shape::new(&[batch, 1, nkv, hd], DType::F32),
    );
    let v4 = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, 1, nkv as i64, hd as i64],
        },
        vec![v],
        Shape::new(&[batch, 1, nkv, hd], DType::F32),
    );
    let q_r = apply_rope(g, q4, cos, sin, batch, 1, nh, hd);
    let k_r = apply_rope(g, k4, cos, sin, batch, 1, nkv, hd);
    let new_k = g.add_node(
        Op::Concat { axis: 1 },
        vec![past_k, k_r],
        Shape::new(&[batch, kv_len, nkv, hd], DType::F32),
    );
    let new_v = g.add_node(
        Op::Concat { axis: 1 },
        vec![past_v, v4],
        Shape::new(&[batch, kv_len, nkv, hd], DType::F32),
    );
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            mask_kind: MaskKind::Causal,
            score_scale: None,
            attn_logit_softcap: None,
        },
        vec![q_r, new_k, new_v],
        Shape::new(&[batch, 1, nh, hd], DType::F32),
    );
    let attn_flat = g.add_node(
        Op::Reshape {
            new_shape: vec![batch_seq as i64, (nh * hd) as i64],
        },
        vec![attn],
        Shape::new(&[batch_seq, nh * hd], DType::F32),
    );
    let o = dq(
        g,
        attn_flat,
        find(linears, &format!("{prefix}.self_attn.o_proj"))?,
        batch_seq,
    )?;
    let h1 = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![hidden, o],
        Shape::new(&[batch_seq, h], DType::F32),
    );

    let ln2_g = g.param(
        format!("{prefix}.post_attention_layernorm.weight"),
        Shape::new(&[h], DType::F32),
    );
    let ln2_b = g.param(
        format!("{prefix}.post_attention_layernorm.bias_zero"),
        Shape::new(&[h], DType::F32),
    );
    let n2 = g.add_node(
        Op::RmsNorm { axis: -1, eps },
        vec![h1, ln2_g, ln2_b],
        Shape::new(&[batch_seq, h], DType::F32),
    );
    let gate = dq(
        g,
        n2,
        find(linears, &format!("{prefix}.mlp.gate_proj"))?,
        batch_seq,
    )?;
    let up = dq(
        g,
        n2,
        find(linears, &format!("{prefix}.mlp.up_proj"))?,
        batch_seq,
    )?;
    let gate_s = g.add_node(
        Op::Activation(Activation::Silu),
        vec![gate],
        Shape::new(&[batch_seq, arch.intermediate_size], DType::F32),
    );
    let ff = g.add_node(
        Op::Binary(BinaryOp::Mul),
        vec![gate_s, up],
        Shape::new(&[batch_seq, arch.intermediate_size], DType::F32),
    );
    let down = dq(
        g,
        ff,
        find(linears, &format!("{prefix}.mlp.down_proj"))?,
        batch_seq,
    )?;
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![h1, down],
        Shape::new(&[batch_seq, h], DType::F32),
    );
    Ok((out, new_k, new_v))
}

fn lm_head_logits(
    g: &mut Graph,
    arch: &MlxArchConfig,
    flat: NodeId,
    batch: usize,
    seq: usize,
) -> NodeId {
    let h = arch.hidden_size;
    let batch_seq = batch * seq;
    let fn_g = g.param("model.norm.weight", Shape::new(&[h], DType::F32));
    let fn_b = g.param("model.norm.bias_zero", Shape::new(&[h], DType::F32));
    let normed = g.add_node(
        Op::RmsNorm {
            axis: -1,
            eps: arch.rms_norm_eps,
        },
        vec![flat, fn_g, fn_b],
        Shape::new(&[batch_seq, h], DType::F32),
    );
    let head = g.param(
        "lm_head.weight",
        Shape::new(&[arch.vocab_size, h], DType::F32),
    );
    let head_t = g.add_node(
        Op::Transpose { perm: vec![1, 0] },
        vec![head],
        Shape::new(&[h, arch.vocab_size], DType::F32),
    );
    let logits = g.add_node(
        Op::MatMul,
        vec![normed, head_t],
        Shape::new(&[batch_seq, arch.vocab_size], DType::F32),
    );
    g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, seq as i64, arch.vocab_size as i64],
        },
        vec![logits],
        Shape::new(&[batch, seq, arch.vocab_size], DType::F32),
    )
}

/// Prefill: `tokens` I32 `[batch, seq]` → embed → N layers (RoPE) → logits.
pub fn build_llama_like_prefill(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    seq: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    let layers = num_layers
        .unwrap_or(arch.num_hidden_layers)
        .min(arch.num_hidden_layers);
    if layers == 0 {
        bail!("num_hidden_layers is 0");
    }
    let h = arch.hidden_size;
    let batch_seq = batch * seq;
    let mut g = Graph::new(graph_name);
    let (cos, sin) = rope_tables(&mut g, arch, seq);
    let tokens = g.input("tokens", Shape::new(&[batch, seq], DType::I32));
    let emb_w = g.param(
        "model.embed_tokens.weight",
        Shape::new(&[arch.vocab_size, h], DType::F32),
    );
    let flat_tok = g.add_node(
        Op::Reshape {
            new_shape: vec![batch_seq as i64],
        },
        vec![tokens],
        Shape::new(&[batch_seq], DType::I32),
    );
    let emb_flat = g.add_node(
        Op::Gather { axis: 0 },
        vec![emb_w, flat_tok],
        Shape::new(&[batch_seq, h], DType::F32),
    );
    let mut flat = emb_flat;
    for i in 0..layers {
        flat = build_llama_decoder_layer(&mut g, arch, i, linears, flat, batch, seq, cos, sin)?;
    }
    let out = lm_head_logits(&mut g, arch, flat, batch, seq);
    g.set_outputs(vec![out]);
    Ok(g)
}

/// Decode step: `token` I32 `[batch, 1]` + past K/V per layer → logits + new K/V.
///
/// Inputs: `token`, then for each layer `past_k_{i}`, `past_v_{i}` with shape
/// `[batch, past_len, n_kv, head_dim]`. Cos/sin tables cover the **new**
/// position only (`[1, head_dim/2]`); pass position via rebuilding with
/// `position` offset baked into the constant tables.
///
/// Outputs: `logits [batch,1,V]`, then `new_k_0, new_v_0, …`.
pub fn build_llama_like_decode(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    past_len: usize,
    position: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    let layers = num_layers
        .unwrap_or(arch.num_hidden_layers)
        .min(arch.num_hidden_layers);
    if layers == 0 {
        bail!("num_hidden_layers is 0");
    }
    let h = arch.hidden_size;
    let nkv = arch.num_key_value_heads;
    let hd = arch.head_dim();
    let mut g = Graph::new(graph_name);
    // Single-row rope tables at absolute `position`.
    let (cos_full, sin_full) = build_default_tables(arch.rope_theta as f64, hd, position + 1);
    let half = hd / 2;
    let cos_row = &cos_full[position * half..(position + 1) * half];
    let sin_row = &sin_full[position * half..(position + 1) * half];
    let cos = g.add_node(
        Op::Constant {
            data: f32_le_bytes(cos_row),
        },
        vec![],
        Shape::new(&[1, half], DType::F32),
    );
    let sin = g.add_node(
        Op::Constant {
            data: f32_le_bytes(sin_row),
        },
        vec![],
        Shape::new(&[1, half], DType::F32),
    );

    let token = g.input("token", Shape::new(&[batch, 1], DType::I32));
    let emb_w = g.param(
        "model.embed_tokens.weight",
        Shape::new(&[arch.vocab_size, h], DType::F32),
    );
    let flat_tok = g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64],
        },
        vec![token],
        Shape::new(&[batch], DType::I32),
    );
    let mut flat = g.add_node(
        Op::Gather { axis: 0 },
        vec![emb_w, flat_tok],
        Shape::new(&[batch, h], DType::F32),
    );

    let mut kv_outs = Vec::with_capacity(layers * 2);
    for i in 0..layers {
        let past_k = g.input(
            format!("past_k_{i}"),
            Shape::new(&[batch, past_len, nkv, hd], DType::F32),
        );
        let past_v = g.input(
            format!("past_v_{i}"),
            Shape::new(&[batch, past_len, nkv, hd], DType::F32),
        );
        let (h_out, nk, nv) = build_llama_decode_layer(
            &mut g, arch, i, linears, flat, batch, past_len, past_k, past_v, cos, sin,
        )?;
        flat = h_out;
        kv_outs.push(nk);
        kv_outs.push(nv);
    }
    let logits = lm_head_logits(&mut g, arch, flat, batch, 1);
    let mut outs = vec![logits];
    outs.extend(kv_outs);
    g.set_outputs(outs);
    Ok(g)
}

/// Load an mlx-community dir, collect packed Linears, build a Llama-like prefill.
pub fn build_llama_like_from_dir(
    path: impl AsRef<std::path::Path>,
    batch: usize,
    seq: usize,
    num_layers: Option<usize>,
) -> Result<(Graph, Vec<PackedLinearBinding>, MlxArchConfig)> {
    let path = path.as_ref();
    let mut weights = crate::load::load_path(path)?;
    let arch = weights
        .config
        .arch
        .clone()
        .ok_or_else(|| anyhow::anyhow!("config.json missing Llama-like arch fields"))?;
    let linears = crate::graph::collect_packed_linears(&mut weights)?;
    let gname = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mlx_llama");
    let g = build_llama_like_prefill(gname, &arch, &linears, batch, seq, num_layers)?;
    Ok((g, linears, arch))
}
