// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// (license header truncated — see workspace root.)

//! IR-lowered detector encoder fusion.
//!
//! Expresses the same 6-layer SAM3 encoder as a `rlx_ir::Graph` so the
//! attention path goes through `Op::Attention` (BLAS-batched per-head on
//! CPU; MPS / MLX on GPU backends) instead of the per-head sgemm loop in
//! [`super::detector_encoder::forward_encoder`].
//!
//! Inputs (graph) — all f32 batch-first:
//!   * `src`         `[B, hw, D]`
//!   * `src_pos`     `[B, hw, D]`
//!   * `prompt`      `[B, L, D]`
//!   * `prompt_kpm_inv` `[B, L]` — `1.0` = valid, `0.0` = PAD (already
//!     inverted from the upstream PyTorch convention so it matches the
//!     IR `MaskKind::Custom` semantics).
//!
//! Output: encoder memory `[B, hw, D]`.

use super::detector_encoder::Sam3EncoderWeights;
use anyhow::{Result, ensure};
use rlx_ir::op::{Activation, BinaryOp, MaskKind};
use rlx_ir::shape;
use rlx_ir::{DType, Graph, NodeId, Shape};
use rlx_runtime::{CompiledGraph, Device, Session};
use std::collections::HashMap;

/// Compiled encoder graph + uploaded parameters, ready for many `run`s.
/// Use this when you want to amortise the graph compile + param upload
/// cost across multiple frames (e.g. video, batched eval, benchmarks).
pub struct Sam3CompiledEncoder {
    pub compiled: CompiledGraph,
    pub batch: usize,
    pub hw: usize,
    pub seq: usize,
    pub d: usize,
}

impl Sam3CompiledEncoder {
    /// Compile the encoder graph for `(batch, hw, seq)` on the given device
    /// and upload all parameters from `weights`.
    pub fn new(
        weights: &Sam3EncoderWeights,
        batch: usize,
        hw: usize,
        seq: usize,
        device: Device,
    ) -> Result<Self> {
        let (graph, params) = build_graph(weights, batch, hw, seq)?;
        let session = Session::new(device);
        let mut compiled = session.compile(graph);
        for (name, data) in &params {
            compiled.set_param(name, data);
        }
        Ok(Self {
            compiled,
            batch,
            hw,
            seq,
            d: D_MODEL,
        })
    }

    /// Run on a single frame's pre-arranged batch-first inputs.
    /// `src` and `src_pos` are NCHW; `prompt` is seq-first; `prompt_kpm`
    /// is the upstream PyTorch convention (1 = PAD).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        src_bchw: &[f32],
        src_pos_bchw: &[f32],
        prompt_seq_first: &[f32],
        prompt_kpm: &[u8],
        src_h: usize,
        src_w: usize,
    ) -> Result<Vec<f32>> {
        let hw = src_h * src_w;
        ensure!(hw == self.hw, "compiled encoder expects hw={}, got {hw}", self.hw);
        // Convert NCHW → [B, hw, D].
        let mut src_bhwc = vec![0f32; self.batch * hw * self.d];
        let mut pos_bhwc = vec![0f32; self.batch * hw * self.d];
        for b in 0..self.batch {
            for s in 0..hw {
                for c in 0..self.d {
                    src_bhwc[(b * hw + s) * self.d + c] =
                        src_bchw[((b * self.d + c) * hw) + s];
                    pos_bhwc[(b * hw + s) * self.d + c] =
                        src_pos_bchw[((b * self.d + c) * hw) + s];
                }
            }
        }
        let mut prompt_bf = vec![0f32; self.batch * self.seq * self.d];
        for b in 0..self.batch {
            for l in 0..self.seq {
                let s = (l * self.batch + b) * self.d;
                let dst = (b * self.seq + l) * self.d;
                prompt_bf[dst..dst + self.d]
                    .copy_from_slice(&prompt_seq_first[s..s + self.d]);
            }
        }
        let prompt_kpm_inv: Vec<f32> = prompt_kpm
            .iter()
            .map(|&v| if v == 0 { 1.0 } else { 0.0 })
            .collect();
        let outputs = self.compiled.run(&[
            ("src", src_bhwc.as_slice()),
            ("src_pos", pos_bhwc.as_slice()),
            ("prompt", prompt_bf.as_slice()),
            ("prompt_kpm_inv", prompt_kpm_inv.as_slice()),
        ]);
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("encoder graph produced no outputs"))
    }
}

const D_MODEL: usize = 256;
const DIM_FF: usize = 2048;
const N_HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / N_HEADS;

/// Split a transposed `[E, 3*E]` in_proj weight into three `[E, E]`
/// slabs for Q, K, V.
fn split_qkv(w_t: &[f32], e: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut wq = vec![0f32; e * e];
    let mut wk = vec![0f32; e * e];
    let mut wv = vec![0f32; e * e];
    for i in 0..e {
        for j in 0..e {
            wq[i * e + j] = w_t[i * 3 * e + j];
            wk[i * e + j] = w_t[i * 3 * e + e + j];
            wv[i * e + j] = w_t[i * 3 * e + 2 * e + j];
        }
    }
    (wq, wk, wv)
}

// The `_data` ownership is kept on the signature so callers can hand
// us the param blob in the same call shape they use elsewhere; the
// IR-level `g.param` only needs the shape (the data is uploaded
// separately via `CompiledGraph::set_param` after compile).
fn add_param(g: &mut Graph, name: &str, _data: Vec<f32>, shape: Shape) -> NodeId {
    g.param(name, shape)
}

/// Build the encoder graph + collect the parameter blobs that need to be
/// uploaded via `CompiledGraph::set_param`.
fn build_graph(
    weights: &Sam3EncoderWeights,
    batch: usize,
    hw: usize,
    seq: usize,
) -> Result<(Graph, HashMap<String, Vec<f32>>)> {
    let mut g = Graph::new("sam3_detector_encoder");
    let mut params: HashMap<String, Vec<f32>> = HashMap::new();
    let f = DType::F32;

    let d = D_MODEL;
    let nh = N_HEADS;
    let dh = HEAD_DIM;
    let dim_ff = DIM_FF;

    // Inputs.
    let src = g.input("src", Shape::new(&[batch, hw, d], f));
    let src_pos = g.input("src_pos", Shape::new(&[batch, hw, d], f));
    let prompt = g.input("prompt", Shape::new(&[batch, seq, d], f));
    let prompt_kpm_inv = g.input("prompt_kpm_inv", Shape::new(&[batch, seq], f));

    let mut tgt = src;

    for (li, layer) in weights.layers.iter().enumerate() {
        // ── Pre-LN 1 (self-attn) ───────────────────────────────────
        let n1_w = add_param(
            &mut g,
            &format!("l{li}.norm1.w"),
            layer.norm1_w.clone(),
            Shape::new(&[d], f),
        );
        params.insert(format!("l{li}.norm1.w"), layer.norm1_w.clone());
        let n1_b = add_param(
            &mut g,
            &format!("l{li}.norm1.b"),
            layer.norm1_b.clone(),
            Shape::new(&[d], f),
        );
        params.insert(format!("l{li}.norm1.b"), layer.norm1_b.clone());
        let n1 = g.layer_norm(tgt, n1_w, n1_b, -1, 1e-5, Shape::new(&[batch, hw, d], f));

        // q = n1 + pos; k = q; v = n1
        let qk_in = g.binary(
            BinaryOp::Add,
            n1,
            src_pos,
            Shape::new(&[batch, hw, d], f),
        );

        // Self-attn QKV projections (split the [E, 3E] in_proj_w).
        let (wq, wk, wv) = split_qkv(&layer.self_attn_in_w_t, d);
        let bq = layer.self_attn_in_b[0..d].to_vec();
        let bk = layer.self_attn_in_b[d..2 * d].to_vec();
        let bv = layer.self_attn_in_b[2 * d..3 * d].to_vec();
        let q_node = qkv_linear(&mut g, &mut params, &format!("l{li}.sa.q"), qk_in, wq, bq, batch, hw, d);
        let k_node = qkv_linear(&mut g, &mut params, &format!("l{li}.sa.k"), qk_in, wk, bk, batch, hw, d);
        let v_node = qkv_linear(&mut g, &mut params, &format!("l{li}.sa.v"), n1, wv, bv, batch, hw, d);

        let sa_attn = g.attention_kind(
            q_node,
            k_node,
            v_node,
            nh,
            dh,
            MaskKind::None,
            shape::attention_shape(g.shape(q_node)),
        );
        // Output projection (linear + bias).
        let sa_out = linear_with_bias(
            &mut g,
            &mut params,
            &format!("l{li}.sa.proj"),
            sa_attn,
            layer.self_attn_out_w_t.clone(),
            layer.self_attn_out_b.clone(),
            batch * hw,
            d,
            d,
        );
        tgt = g.binary(BinaryOp::Add, tgt, sa_out, Shape::new(&[batch, hw, d], f));

        // ── Pre-LN 2 (text cross-attn) ─────────────────────────────
        let n2_w = add_param(
            &mut g,
            &format!("l{li}.norm2.w"),
            layer.norm2_w.clone(),
            Shape::new(&[d], f),
        );
        params.insert(format!("l{li}.norm2.w"), layer.norm2_w.clone());
        let n2_b = add_param(
            &mut g,
            &format!("l{li}.norm2.b"),
            layer.norm2_b.clone(),
            Shape::new(&[d], f),
        );
        params.insert(format!("l{li}.norm2.b"), layer.norm2_b.clone());
        let n2 = g.layer_norm(tgt, n2_w, n2_b, -1, 1e-5, Shape::new(&[batch, hw, d], f));

        let (wqc, wkc, wvc) = split_qkv(&layer.cross_attn_in_w_t, d);
        let bqc = layer.cross_attn_in_b[0..d].to_vec();
        let bkc = layer.cross_attn_in_b[d..2 * d].to_vec();
        let bvc = layer.cross_attn_in_b[2 * d..3 * d].to_vec();
        let qc = qkv_linear(&mut g, &mut params, &format!("l{li}.ca.q"), n2, wqc, bqc, batch, hw, d);
        let kc = qkv_linear(&mut g, &mut params, &format!("l{li}.ca.k"), prompt, wkc, bkc, batch, seq, d);
        let vc = qkv_linear(&mut g, &mut params, &format!("l{li}.ca.v"), prompt, wvc, bvc, batch, seq, d);

        let ca_attn = g.attention(
            qc,
            kc,
            vc,
            prompt_kpm_inv,
            nh,
            dh,
            shape::attention_shape(g.shape(qc)),
        );
        let ca_out = linear_with_bias(
            &mut g,
            &mut params,
            &format!("l{li}.ca.proj"),
            ca_attn,
            layer.cross_attn_out_w_t.clone(),
            layer.cross_attn_out_b.clone(),
            batch * hw,
            d,
            d,
        );
        tgt = g.binary(BinaryOp::Add, tgt, ca_out, Shape::new(&[batch, hw, d], f));

        // ── Pre-LN 3 (FFN) ─────────────────────────────────────────
        let n3_w = add_param(
            &mut g,
            &format!("l{li}.norm3.w"),
            layer.norm3_w.clone(),
            Shape::new(&[d], f),
        );
        params.insert(format!("l{li}.norm3.w"), layer.norm3_w.clone());
        let n3_b = add_param(
            &mut g,
            &format!("l{li}.norm3.b"),
            layer.norm3_b.clone(),
            Shape::new(&[d], f),
        );
        params.insert(format!("l{li}.norm3.b"), layer.norm3_b.clone());
        let n3 = g.layer_norm(tgt, n3_w, n3_b, -1, 1e-5, Shape::new(&[batch, hw, d], f));

        let ff1 = linear_with_bias(
            &mut g,
            &mut params,
            &format!("l{li}.ffn.fc1"),
            n3,
            layer.linear1_w_t.clone(),
            layer.linear1_b.clone(),
            batch * hw,
            d,
            dim_ff,
        );
        let relud = g.activation(Activation::Relu, ff1, Shape::new(&[batch, hw, dim_ff], f));
        let ff2 = linear_with_bias(
            &mut g,
            &mut params,
            &format!("l{li}.ffn.fc2"),
            relud,
            layer.linear2_w_t.clone(),
            layer.linear2_b.clone(),
            batch * hw,
            dim_ff,
            d,
        );
        tgt = g.binary(BinaryOp::Add, tgt, ff2, Shape::new(&[batch, hw, d], f));
    }

    g.set_outputs(vec![tgt]);
    Ok((g, params))
}

fn qkv_linear(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: NodeId,
    w: Vec<f32>,
    b: Vec<f32>,
    batch: usize,
    seq: usize,
    d: usize,
) -> NodeId {
    let f = DType::F32;
    let w_name = format!("{name}.w");
    let b_name = format!("{name}.b");
    let w_id = g.param(&w_name, Shape::new(&[d, d], f));
    params.insert(w_name, w);
    let b_id = g.param(&b_name, Shape::new(&[d], f));
    params.insert(b_name, b);
    let out_shape = Shape::new(&[batch, seq, d], f);
    g.fused_matmul_bias_act(input, w_id, b_id, None, out_shape)
}

fn linear_with_bias(
    g: &mut Graph,
    params: &mut HashMap<String, Vec<f32>>,
    name: &str,
    input: NodeId,
    w: Vec<f32>,
    b: Vec<f32>,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> NodeId {
    let f = DType::F32;
    let w_name = format!("{name}.w");
    let b_name = format!("{name}.b");
    let w_id = g.param(&w_name, Shape::new(&[in_dim, out_dim], f));
    params.insert(w_name, w);
    let b_id = g.param(&b_name, Shape::new(&[out_dim], f));
    params.insert(b_name, b);
    // Reshape input to [rows, in_dim] equivalent — the IR's MatMul
    // already handles batch broadcasting, so leave it as-is and trust
    // the executor.
    let cur_shape = g.shape(input).clone();
    let mut out_dims: Vec<usize> = cur_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    *out_dims.last_mut().unwrap() = out_dim;
    let _ = rows;
    let _ = in_dim;
    g.fused_matmul_bias_act(input, w_id, b_id, None, Shape::new(&out_dims, f))
}

/// Build, compile and execute the encoder graph on the requested device.
#[allow(clippy::too_many_arguments)]
pub fn forward_encoder_ir_on(
    weights: &Sam3EncoderWeights,
    src_bchw: &[f32],
    src_pos_bchw: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    src_h: usize,
    src_w: usize,
    prompt_len: usize,
    device: Device,
) -> Result<Vec<f32>> {
    ensure!(weights.loaded, "SAM3 detector encoder not loaded");
    let hw = src_h * src_w;
    ensure!(
        src_bchw.len() == batch * D_MODEL * hw,
        "encoder src shape mismatch"
    );
    ensure!(
        prompt_seq_first.len() == prompt_len * batch * D_MODEL,
        "encoder prompt shape mismatch"
    );

    // Convert NCHW → [B, hw, D] for the IR.
    let mut src_bhwc = vec![0f32; batch * hw * D_MODEL];
    let mut pos_bhwc = vec![0f32; batch * hw * D_MODEL];
    for b in 0..batch {
        for s in 0..hw {
            for c in 0..D_MODEL {
                src_bhwc[(b * hw + s) * D_MODEL + c] = src_bchw[((b * D_MODEL + c) * hw) + s];
                pos_bhwc[(b * hw + s) * D_MODEL + c] = src_pos_bchw[((b * D_MODEL + c) * hw) + s];
            }
        }
    }

    // Convert seq-first prompt → batch-first.
    let mut prompt_bf = vec![0f32; batch * prompt_len * D_MODEL];
    for b in 0..batch {
        for l in 0..prompt_len {
            let s = (l * batch + b) * D_MODEL;
            let dst = (b * prompt_len + l) * D_MODEL;
            prompt_bf[dst..dst + D_MODEL].copy_from_slice(&prompt_seq_first[s..s + D_MODEL]);
        }
    }
    // Invert key padding mask: IR Custom expects 1.0 = valid, 0.0 = PAD.
    let prompt_kpm_inv: Vec<f32> = prompt_kpm
        .iter()
        .map(|&v| if v == 0 { 1.0 } else { 0.0 })
        .collect();

    let (graph, params) = build_graph(weights, batch, hw, prompt_len)?;
    let session = Session::new(device);
    let mut compiled = session.compile(graph);
    for (name, data) in &params {
        compiled.set_param(name, data);
    }
    let outputs = compiled.run(&[
        ("src", src_bhwc.as_slice()),
        ("src_pos", pos_bhwc.as_slice()),
        ("prompt", prompt_bf.as_slice()),
        ("prompt_kpm_inv", prompt_kpm_inv.as_slice()),
    ]);
    let out = outputs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("encoder graph produced no outputs"))?;
    Ok(out)
}

/// Convenience wrapper that runs on CPU.
#[allow(clippy::too_many_arguments)]
pub fn forward_encoder_ir(
    weights: &Sam3EncoderWeights,
    src_bchw: &[f32],
    src_pos_bchw: &[f32],
    prompt_seq_first: &[f32],
    prompt_kpm: &[u8],
    batch: usize,
    src_h: usize,
    src_w: usize,
    prompt_len: usize,
) -> Result<Vec<f32>> {
    forward_encoder_ir_on(
        weights,
        src_bchw,
        src_pos_bchw,
        prompt_seq_first,
        prompt_kpm,
        batch,
        src_h,
        src_w,
        prompt_len,
        Device::Cpu,
    )
}
