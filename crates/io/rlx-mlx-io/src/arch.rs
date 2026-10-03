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

/// Per-head RMSNorm over `[batch, seq, heads, head_dim]`, normalizing the
/// head dimension. Qwen3 applies this to Q and K before RoPE.
fn apply_head_norm(
    g: &mut Graph,
    x: NodeId,
    weight_name: &str,
    batch: usize,
    seq: usize,
    heads: usize,
    hd: usize,
    eps: f32,
) -> NodeId {
    let w = g.param(weight_name, Shape::new(&[hd], DType::F32));
    let b = g.param(
        format!("{weight_name}.bias_zero"),
        Shape::new(&[hd], DType::F32),
    );
    g.add_node(
        Op::RmsNorm { axis: -1, eps },
        vec![x, w, b],
        Shape::new(&[batch, seq, heads, hd], DType::F32),
    )
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
    build_llama_decoder_layer_kv(g, arch, layer_idx, linears, hidden, batch, seq, cos, sin)
        .map(|(out, _, _)| out)
}

/// Like [`build_llama_decoder_layer`] but also returns this layer's rope'd
/// `K` and `V`, each `[batch, seq, n_kv, head_dim]`.
///
/// Those are exactly the rows a KV cache holds, so a prefill graph that
/// forwards them can seed a decode loop in one pass instead of replaying the
/// prompt a token at a time.
#[allow(clippy::too_many_arguments)]
pub fn build_llama_decoder_layer_kv(
    g: &mut Graph,
    arch: &MlxArchConfig,
    layer_idx: usize,
    linears: &[PackedLinearBinding],
    hidden: NodeId,
    batch: usize,
    seq: usize,
    cos: NodeId,
    sin: NodeId,
) -> Result<(NodeId, NodeId, NodeId)> {
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
    let (q4, k4) = if arch.uses_qk_norm() {
        (
            apply_head_norm(
                g,
                q4,
                &format!("{prefix}.self_attn.q_norm.weight"),
                batch,
                seq,
                nh,
                hd,
                eps,
            ),
            apply_head_norm(
                g,
                k4,
                &format!("{prefix}.self_attn.k_norm.weight"),
                batch,
                seq,
                nkv,
                hd,
                eps,
            ),
        )
    } else {
        (q4, k4)
    };
    let q_r = apply_rope(g, q4, cos, sin, batch, seq, nh, hd);
    let k_r = apply_rope(g, k4, cos, sin, batch, seq, nkv, hd);
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
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
    let out = g.add_node(
        Op::Binary(BinaryOp::Add),
        vec![h1, down],
        Shape::new(&[batch_seq, h], DType::F32),
    );
    Ok((out, k_r, v4))
}

/// One decode layer: `seq=1`, concat past K/V, return `(hidden, new_k, new_v)`.
#[allow(clippy::too_many_arguments)]
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
    mask: Option<NodeId>,
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
    let (q4, k4) = if arch.uses_qk_norm() {
        (
            apply_head_norm(
                g,
                q4,
                &format!("{prefix}.self_attn.q_norm.weight"),
                batch,
                1,
                nh,
                hd,
                eps,
            ),
            apply_head_norm(
                g,
                k4,
                &format!("{prefix}.self_attn.k_norm.weight"),
                batch,
                1,
                nkv,
                hd,
                eps,
            ),
        )
    } else {
        (q4, k4)
    };
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
    // With a keep-mask the cached rows may be padding, so masking has to be
    // explicit; without one every cached key is real and causal masking over
    // a single query row is equivalent.
    let (mask_kind, attn_inputs) = match mask {
        Some(m) => (MaskKind::Custom, vec![q_r, new_k, new_v, m]),
        None => (MaskKind::Causal, vec![q_r, new_k, new_v]),
    };
    let attn = g.add_node(
        Op::Attention {
            num_heads: nh,
            head_dim: hd,
            v_head_dim: None,
            mask_kind,
            score_scale: None,
            attn_logit_softcap: None,
        },
        attn_inputs,
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

/// Final norm + LM head.
///
/// The head is the single largest tensor in a small model — for a 0.6B with a
/// 152k vocabulary it is more than half the parameters. When the checkpoint
/// ships it quantized (including the tied case, where the caller passes the
/// embedding's packed form under this name) it stays packed and goes through
/// `Op::DequantMatMul`, which also avoids transposing `[vocab, hidden]` on
/// every decode step. Dequantizing it instead costs ~6x the bytes and a
/// full-size transpose per token, which dominates decode.
fn lm_head_logits(
    g: &mut Graph,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    flat: NodeId,
    batch: usize,
    seq: usize,
) -> Result<NodeId> {
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
    let logits = match find(linears, "lm_head") {
        Ok(b) => dq(g, normed, b, batch_seq)?,
        Err(_) => {
            let head = g.param(
                "lm_head.weight",
                Shape::new(&[arch.vocab_size, h], DType::F32),
            );
            let head_t = g.add_node(
                Op::Transpose { perm: vec![1, 0] },
                vec![head],
                Shape::new(&[h, arch.vocab_size], DType::F32),
            );
            g.add_node(
                Op::MatMul,
                vec![normed, head_t],
                Shape::new(&[batch_seq, arch.vocab_size], DType::F32),
            )
        }
    };
    Ok(g.add_node(
        Op::Reshape {
            new_shape: vec![batch as i64, seq as i64, arch.vocab_size as i64],
        },
        vec![logits],
        Shape::new(&[batch, seq, arch.vocab_size], DType::F32),
    ))
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
    build_prefill_inner(
        graph_name,
        arch,
        linears,
        batch,
        seq,
        num_layers,
        false,
        EmbedSource::Table,
    )
}

/// Like [`build_llama_like_prefill`] but also outputs each layer's K and V,
/// so a prompt seeds the KV cache in a single pass.
///
/// Outputs: `logits`, then `k_0, v_0, k_1, v_1, …`, each K/V shaped
/// `[batch, seq, n_kv, head_dim]`. Without this a generation loop has to
/// replay the prompt one token at a time through the decode graph, which
/// costs a compile per prompt token.
pub fn build_llama_like_prefill_kv(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    seq: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    build_prefill_inner(
        graph_name,
        arch,
        linears,
        batch,
        seq,
        num_layers,
        true,
        EmbedSource::Table,
    )
}

#[allow(clippy::too_many_arguments)]
/// Prefill over pre-looked-up embeddings.
///
/// Same graph as `build_llama_like_prefill_kv` but with `embeddings`
/// `[batch*seq, hidden]` as an input instead of a token-id input plus the
/// embedding table as a parameter. See [`EmbedSource`] for why that matters.
pub fn build_llama_like_prefill_kv_embedded(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    seq: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    build_prefill_inner(
        graph_name,
        arch,
        linears,
        batch,
        seq,
        num_layers,
        true,
        EmbedSource::Rows,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_prefill_inner(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    seq: usize,
    num_layers: Option<usize>,
    with_kv: bool,
    embed: EmbedSource,
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
    if embed == EmbedSource::Rows {
        let rows = g.input("embeddings", Shape::new(&[batch_seq, h], DType::F32));
        return build_prefill_body(
            g, arch, linears, batch, seq, layers, with_kv, rows, cos, sin,
        );
    }
    // Token ids are F32-encoded: the CPU/Metal arena is f32-aliased and
    // `Op::Gather` reads its index operand as f32 unless the dtype is I64,
    // so declaring I32 here would make every nonzero id gather row 0.
    let tokens = g.input("tokens", Shape::new(&[batch, seq], DType::F32));
    let emb_w = g.param(
        "model.embed_tokens.weight",
        Shape::new(&[arch.vocab_size, h], DType::F32),
    );
    let flat_tok = g.add_node(
        Op::Reshape {
            new_shape: vec![batch_seq as i64],
        },
        vec![tokens],
        Shape::new(&[batch_seq], DType::F32),
    );
    let emb_flat = g.add_node(
        Op::Gather { axis: 0 },
        vec![emb_w, flat_tok],
        Shape::new(&[batch_seq, h], DType::F32),
    );
    build_prefill_body(
        g, arch, linears, batch, seq, layers, with_kv, emb_flat, cos, sin,
    )
}

/// Where a graph's token embeddings come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedSource {
    /// Gather rows from a dense `model.embed_tokens.weight` parameter, given
    /// token ids. Self-contained, but the table is a parameter of every graph:
    /// at a 152k vocabulary that is ~600 MB per compiled graph, host and device.
    Table,
    /// Take the already-looked-up rows as an `embeddings` input. The caller owns
    /// the table — one copy, in whatever form it likes — and the graph carries
    /// none of it.
    Rows,
}

/// Layers + LM head over already-embedded tokens. Shared by both
/// `EmbedSource` paths so they cannot drift.
#[allow(clippy::too_many_arguments)]
fn build_prefill_body(
    mut g: Graph,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    seq: usize,
    layers: usize,
    with_kv: bool,
    embedded: NodeId,
    cos: NodeId,
    sin: NodeId,
) -> Result<Graph> {
    let mut flat = embedded;
    let mut kv: Vec<NodeId> = Vec::new();
    for i in 0..layers {
        let (next, k, v) =
            build_llama_decoder_layer_kv(&mut g, arch, i, linears, flat, batch, seq, cos, sin)?;
        flat = next;
        if with_kv {
            kv.push(k);
            kv.push(v);
        }
    }
    let out = lm_head_logits(&mut g, arch, linears, flat, batch, seq)?;
    let mut outputs = vec![out];
    outputs.extend(kv);
    g.set_outputs(outputs);
    Ok(g)
}

/// Single-row RoPE tables for absolute `position`, to feed
/// [`build_llama_like_decode_dyn`]'s `rope_cos` / `rope_sin` inputs.
pub fn decode_rope_row(arch: &MlxArchConfig, position: usize) -> (Vec<f32>, Vec<f32>) {
    let hd = arch.head_dim();
    let half = hd / 2;
    let (cos_full, sin_full) = build_default_tables(arch.rope_theta as f64, hd, position + 1);
    (
        cos_full[position * half..(position + 1) * half].to_vec(),
        sin_full[position * half..(position + 1) * half].to_vec(),
    )
}

/// Decode step: `token` I32 `[batch, 1]` + past K/V per layer → logits + new K/V.
///
/// Inputs: `token`, then for each layer `past_k_{i}`, `past_v_{i}` with shape
/// `[batch, past_len, n_kv, head_dim]`. Cos/sin tables cover the **new**
/// position only (`[1, head_dim/2]`); pass position via rebuilding with
/// `position` offset baked into the constant tables.
///
/// Outputs: `logits [batch,1,V]`, then `new_k_0, new_v_0, …`.
///
/// Baking the position in means a fresh graph — and so a fresh compile — for
/// every generated token. For a generation loop prefer
/// [`build_llama_like_decode_dyn`], which takes the RoPE row as an input so
/// one compiled graph serves every position.
pub fn build_llama_like_decode(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    past_len: usize,
    position: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    build_decode_inner(
        graph_name,
        arch,
        linears,
        batch,
        past_len,
        Some(position),
        num_layers,
        false,
        EmbedSource::Table,
    )
}

/// Like [`build_llama_like_decode`] but with the RoPE row as graph inputs
/// `rope_cos` / `rope_sin` (`[1, head_dim/2]` each) instead of constants.
///
/// The graph is then independent of the token's position, so a generation
/// loop compiles once per KV length rather than once per token — the
/// difference between seconds per token and milliseconds. Feed the row from
/// [`decode_rope_row`].
pub fn build_llama_like_decode_dyn(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    past_len: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    build_decode_inner(
        graph_name,
        arch,
        linears,
        batch,
        past_len,
        None,
        num_layers,
        false,
        EmbedSource::Table,
    )
}

/// Decode with the RoPE row *and* a keep-mask as inputs.
///
/// `past_len` becomes a **capacity** rather than the true cache length: pad
/// the past K/V to it and pass a `[batch, past_len + 1]` mask that is 1.0 for
/// real keys and 0.0 for padding (the final slot, the token being decoded, is
/// always real). One compiled graph then serves every cache length up to that
/// capacity, so a generation loop compiles once per bucket instead of once
/// per token. Build the mask with [`decode_keep_mask`].
pub fn build_llama_like_decode_masked(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    capacity: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    build_decode_inner(
        graph_name,
        arch,
        linears,
        batch,
        capacity,
        None,
        num_layers,
        true,
        EmbedSource::Table,
    )
}

/// Masked padded decode over a pre-looked-up embedding row.
///
/// Same graph as [`build_llama_like_decode_masked`] but taking `embeddings`
/// `[batch, hidden]` as an input instead of a token id plus the embedding
/// table as a parameter. See [`EmbedSource`].
pub fn build_llama_like_decode_masked_embedded(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    capacity: usize,
    num_layers: Option<usize>,
) -> Result<Graph> {
    build_decode_inner(
        graph_name,
        arch,
        linears,
        batch,
        capacity,
        None,
        num_layers,
        true,
        EmbedSource::Rows,
    )
}

/// Keep-mask for [`build_llama_like_decode_masked`]: `1.0` for the `real`
/// cached keys and for the new token's own key at the final slot, `0.0` for
/// the padding between them.
pub fn decode_keep_mask(real: usize, capacity: usize) -> Vec<f32> {
    (0..=capacity)
        .map(|i| if i < real || i == capacity { 1.0 } else { 0.0 })
        .collect()
}

/// Shared body. `position = Some(p)` bakes the RoPE row in as constants;
/// `None` exposes it as `rope_cos` / `rope_sin` inputs.
#[allow(clippy::too_many_arguments)]
fn build_decode_inner(
    graph_name: &str,
    arch: &MlxArchConfig,
    linears: &[PackedLinearBinding],
    batch: usize,
    past_len: usize,
    position: Option<usize>,
    num_layers: Option<usize>,
    masked: bool,
    embed: EmbedSource,
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
    let half = hd / 2;
    let (cos, sin) = match position {
        Some(position) => {
            let (cos_row, sin_row) = decode_rope_row(arch, position);
            let cos = g.add_node(
                Op::Constant {
                    data: f32_le_bytes(&cos_row),
                },
                vec![],
                Shape::new(&[1, half], DType::F32),
            );
            let sin = g.add_node(
                Op::Constant {
                    data: f32_le_bytes(&sin_row),
                },
                vec![],
                Shape::new(&[1, half], DType::F32),
            );
            (cos, sin)
        }
        None => (
            g.input("rope_cos", Shape::new(&[1, half], DType::F32)),
            g.input("rope_sin", Shape::new(&[1, half], DType::F32)),
        ),
    };

    // `[batch, past_len + 1]` keep-mask: 1.0 for a real key, 0.0 for
    // padding. Present only on the masked variant, where `past_len` is a
    // bucket size rather than the true cache length.
    let mask = masked.then(|| g.input("mask", Shape::new(&[batch, past_len + 1], DType::F32)));
    let mut flat = if embed == EmbedSource::Rows {
        g.input("embeddings", Shape::new(&[batch, h], DType::F32))
    } else {
        // F32-encoded token id — see the note in `build_prefill_inner`.
        let token = g.input("token", Shape::new(&[batch, 1], DType::F32));
        let emb_w = g.param(
            "model.embed_tokens.weight",
            Shape::new(&[arch.vocab_size, h], DType::F32),
        );
        let flat_tok = g.add_node(
            Op::Reshape {
                new_shape: vec![batch as i64],
            },
            vec![token],
            Shape::new(&[batch], DType::F32),
        );
        g.add_node(
            Op::Gather { axis: 0 },
            vec![emb_w, flat_tok],
            Shape::new(&[batch, h], DType::F32),
        )
    };

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
            &mut g, arch, i, linears, flat, batch, past_len, past_k, past_v, cos, sin, mask,
        )?;
        flat = h_out;
        kv_outs.push(nk);
        kv_outs.push(nv);
    }
    let logits = lm_head_logits(&mut g, arch, linears, flat, batch, 1)?;
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
