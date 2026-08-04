// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Build an rlx graph for a NeMo checkpoint's architecture.
//!
//! The dominant `.nemo` architecture is the **Conformer / FastConformer
//! encoder** (Parakeet, Canary, and every modern NeMo ASR model). This module
//! reconstructs that encoder as primitive rlx ops, binding the real checkpoint
//! weights by name so RLXP can compile and run the whole trunk — not just a
//! single Linear.
//!
//! What is reconstructed, faithfully to the NeMo reference
//! (`nemo/collections/asr/parts/submodules/{conformer_modules,multi_head_attention}.py`):
//!
//! * per-layer **Macaron** feed-forwards (half-step residual, Swish/SiLU),
//! * **relative-position multi-head self-attention** (Transformer-XL style,
//!   with `pos_bias_u`/`pos_bias_v`, `linear_pos`, and the `rel_shift` trick),
//! * the **convolution module** (pointwise → GLU → depthwise → norm → Swish →
//!   pointwise), with BatchNorm or LayerNorm auto-detected from the state dict,
//! * all four `LayerNorm`s + residuals and the final `norm_out`,
//! * the `xscale = √d_model` input scaling NeMo applies before the layers,
//! * optionally, the `striding` / `dw_striding` **conv subsampling** front-end
//!   (`EncoderOpts::mel_frames`), so the graph runs straight from mel features.
//!
//! The graph is **shape-specialized** to a `(batch, seq_len)` — rlx graphs are
//! static-shape — with the relative sinusoidal position table baked as a
//! constant for that `seq_len`. When the checkpoint is not a recognizable
//! Conformer, [`build_nemo_probe_graph`] falls back to the historical
//! single-Linear probe so callers (e.g. `rlx-pkg`) always get a valid graph.
//!
//! Numerics are structurally faithful but **parity against a reference NeMo
//! forward pass is not yet asserted here** (no bundled checkpoint + expected
//! output); the tests verify structure, shapes, and end-to-end shape inference.

use crate::{NemoConfig, NemoModel};
use anyhow::{Result, anyhow};
use rlx_ir::infer::GraphExt;
use rlx_ir::op::PadMode;
use rlx_ir::{DType, Graph, NodeId, Op, Shape};

/// LayerNorm / BatchNorm epsilon (NeMo default).
const DEFAULT_EPS: f32 = 1e-5;
/// Macaron half-step feed-forward residual scale (`ConformerLayer.fc_factor`).
const FC_FACTOR: f32 = 0.5;
/// Base of the sinusoidal relative-position encoding (NeMo `INF_VAL`).
const POS_ENC_BASE: f64 = 10_000.0;

/// Read-only access to a checkpoint's tensor shapes by name — the only thing
/// the graph builder needs from the weights. Implemented by [`NemoModel`] and,
/// for tests, by plain name→shape maps, so the architecture mapping can be
/// exercised without a multi-gigabyte `.nemo` on disk.
pub trait TensorShapes {
    /// Shape of the named tensor, or `None` if absent.
    fn shape_of(&self, name: &str) -> Option<&[usize]>;
    /// Whether the named tensor is present.
    fn has(&self, name: &str) -> bool {
        self.shape_of(name).is_some()
    }
}

impl TensorShapes for NemoModel {
    fn shape_of(&self, name: &str) -> Option<&[usize]> {
        NemoModel::shape_of(self, name)
    }
}

impl TensorShapes for std::collections::BTreeMap<String, Vec<usize>> {
    fn shape_of(&self, name: &str) -> Option<&[usize]> {
        self.get(name).map(Vec::as_slice)
    }
}

impl TensorShapes for std::collections::HashMap<String, Vec<usize>> {
    fn shape_of(&self, name: &str) -> Option<&[usize]> {
        self.get(name).map(Vec::as_slice)
    }
}

/// Knobs for [`build_nemo_encoder_graph`]. The encoder is shape-specialized to
/// these; defaults give a single sequence of `seq_len` hidden states.
#[derive(Debug, Clone)]
pub struct EncoderOpts {
    /// Graph name.
    pub name: String,
    /// Batch dimension of the encoder input.
    pub batch: usize,
    /// Encoder (post-subsampling) sequence length `T` — the length of the
    /// hidden-state input and the width of the baked relative-position table.
    pub seq_len: usize,
    /// Apply NeMo's `xscale = √d_model` multiply to the input hidden states
    /// (ConformerEncoder default `xscaling=True`).
    pub xscale: bool,
    /// LayerNorm / BatchNorm epsilon (NeMo default `1e-5`).
    pub eps: f32,
    /// When `Some(frames)` and the checkpoint carries a `striding`/`dw_striding`
    /// `pre_encode` front-end (and `preprocessor.features` is known), prepend the
    /// conv subsampling: the graph input becomes mel features
    /// `[batch, frames, feat_in]` and the post-subsampling length `T` (which
    /// replaces `seq_len`) is derived from the conv strides. When `None`, the
    /// input is the already-subsampled hidden state `[batch, seq_len, d_model]`.
    pub mel_frames: Option<usize>,
}

impl Default for EncoderOpts {
    fn default() -> Self {
        Self {
            name: "nemo_conformer".into(),
            batch: 1,
            seq_len: 128,
            xscale: true,
            eps: DEFAULT_EPS,
            mel_frames: None,
        }
    }
}

/// Static hyper-parameters the builder needs, resolved from the YAML config
/// with fall-backs derived from actual weight shapes (so a stripped config
/// still works).
#[derive(Debug, Clone, Copy)]
struct ConformerDims {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    head_dim: usize,
    conv_kernel: usize,
    /// Conv-module norm: `true` = BatchNorm1d (has running stats), `false` =
    /// LayerNorm.
    conv_batch_norm: bool,
}

fn cfg_usize(cfg: &NemoConfig, keys: &[&str]) -> Option<usize> {
    keys.iter().find_map(|k| cfg.get_usize(k))
}

/// `"…linear1.weight"` → `"…linear1.bias"`.
fn bias_name(weight: &str) -> String {
    match weight.strip_suffix("weight") {
        Some(prefix) => format!("{prefix}bias"),
        None => format!("{weight}.bias"),
    }
}

/// Detect + resolve the Conformer geometry, or `None` if the state dict does
/// not look like a NeMo Conformer encoder (no `encoder.layers.0.self_attn`).
fn resolve_dims(cfg: &NemoConfig, shapes: &impl TensorShapes) -> Option<ConformerDims> {
    // The relative-position attention weights are the signature of the arch.
    let q_w = shapes.shape_of("encoder.layers.0.self_attn.linear_q.weight")?;
    if q_w.len() != 2 {
        return None;
    }
    let d_model = cfg_usize(cfg, &["encoder.d_model", "cfg.encoder.d_model"]).unwrap_or(q_w[0]);

    // Heads: config, else back out from pos_bias_u = [H, d_k].
    let n_heads = cfg_usize(cfg, &["encoder.n_heads", "cfg.encoder.n_heads"])
        .or_else(|| {
            shapes
                .shape_of("encoder.layers.0.self_attn.pos_bias_u")
                .filter(|s| s.len() == 2)
                .map(|s| s[0])
        })
        .unwrap_or(8);
    if n_heads == 0 || !d_model.is_multiple_of(n_heads) {
        return None;
    }

    // Layers: config, else count present `norm_out` weights.
    let n_layers =
        cfg_usize(cfg, &["encoder.n_layers", "cfg.encoder.n_layers"]).unwrap_or_else(|| {
            let mut n = 0;
            while shapes.has(&format!("encoder.layers.{n}.norm_out.weight")) {
                n += 1;
            }
            n.max(1)
        });

    // Depthwise conv kernel: weight is [d_model, 1, k].
    let conv_kernel = shapes
        .shape_of("encoder.layers.0.conv.depthwise_conv.weight")
        .filter(|s| s.len() == 3)
        .map(|s| s[2])
        .unwrap_or(9);

    Some(ConformerDims {
        d_model,
        n_layers,
        n_heads,
        head_dim: d_model / n_heads,
        conv_kernel,
        // Conv-module norm flavor: BatchNorm keeps running stats.
        conv_batch_norm: shapes.has("encoder.layers.0.conv.batch_norm.running_mean"),
    })
}

/// Stateful builder for one encoder graph: owns the graph plus the invariant
/// context (`shapes`, resolved `dims`, `batch`, current length `seq`, `eps`) so
/// the block/attention/conv methods take only the tensors and names that vary.
struct Encoder<'a, S: TensorShapes> {
    g: Graph,
    shapes: &'a S,
    dims: ConformerDims,
    batch: usize,
    /// Current sequence length `T`; updated by [`Encoder::input`] when a
    /// subsampling front-end downsamples the mel frames.
    seq: usize,
    eps: f32,
}

impl<'a, S: TensorShapes> Encoder<'a, S> {
    fn new(
        name: &str,
        shapes: &'a S,
        dims: ConformerDims,
        batch: usize,
        seq: usize,
        eps: f32,
    ) -> Self {
        Self {
            g: Graph::new(name),
            shapes,
            dims,
            batch,
            seq,
            eps,
        }
    }

    fn finish(mut self, out: NodeId) -> Graph {
        self.g.set_outputs(vec![out]);
        self.g
    }

    // ── Leaf helpers ────────────────────────────────────────────────────────

    /// `x + bias` for `{weight}`'s companion bias, broadcast over the last axis;
    /// a no-op when the bias tensor is absent.
    fn add_bias(&mut self, y: NodeId, weight_name: &str) -> NodeId {
        let name = bias_name(weight_name);
        match self.shapes.shape_of(&name) {
            Some(sh) => {
                let b = self.g.param(&name, Shape::new(sh, DType::F32));
                self.g.add(y, b)
            }
            None => y,
        }
    }

    /// `x + bias` reshaped for a channel axis that is not last (`reshape` is the
    /// broadcast shape, e.g. `[1, C, 1]` or `[1, C, 1, 1]`); no-op when absent.
    fn add_channel_bias(&mut self, y: NodeId, bias: &str, reshape: Vec<i64>) -> NodeId {
        match self.shapes.shape_of(bias) {
            Some(sh) => {
                let b = self.g.param(bias, Shape::new(sh, DType::F32));
                let br = self.g.reshape_(b, reshape);
                self.g.add(y, br)
            }
            None => y,
        }
    }

    /// A NeMo Linear: weight is `[out, in]`, MatMul wants `[in, out]`, so
    /// transpose the param, then `x·Wᵀ (+ bias)`.
    fn linear(&mut self, x: NodeId, weight: &str, bias: bool) -> Result<NodeId> {
        let w_sh = self
            .shapes
            .shape_of(weight)
            .ok_or_else(|| anyhow!("missing {weight}"))?;
        if w_sh.len() != 2 {
            return Err(anyhow!(
                "{weight}: expected 2-D linear weight, got {w_sh:?}"
            ));
        }
        let w = self.g.param(weight, Shape::new(w_sh, DType::F32));
        let wt = self.g.transpose_(w, vec![1, 0]);
        let y = self.g.mm(x, wt);
        Ok(if bias { self.add_bias(y, weight) } else { y })
    }

    /// A NeMo pointwise Conv1d (`kernel_size = 1`) applied as a channel-mixing
    /// linear on a channel-last `[.., C_in]` tensor. Weight is `[C_out, C_in, 1]`.
    fn pointwise(&mut self, x: NodeId, weight: &str, bias: bool) -> Result<NodeId> {
        let w_sh = self
            .shapes
            .shape_of(weight)
            .ok_or_else(|| anyhow!("missing {weight}"))?;
        if w_sh.len() != 3 || w_sh[2] != 1 {
            return Err(anyhow!(
                "{weight}: expected [Cout, Cin, 1] pointwise, got {w_sh:?}"
            ));
        }
        let (c_out, c_in) = (w_sh[0] as i64, w_sh[1] as i64);
        let w = self.g.param(weight, Shape::new(w_sh, DType::F32));
        // [Cout, Cin, 1] → [Cout, Cin] → [Cin, Cout].
        let w2 = self.g.reshape_(w, vec![c_out, c_in]);
        let wt = self.g.transpose_(w2, vec![1, 0]);
        let y = self.g.mm(x, wt);
        Ok(if bias { self.add_bias(y, weight) } else { y })
    }

    /// LayerNorm over the last axis, binding `{prefix}.weight` / `{prefix}.bias`.
    fn layer_norm(&mut self, x: NodeId, prefix: &str) -> NodeId {
        let d = self
            .shapes
            .shape_of(&format!("{prefix}.weight"))
            .map_or(self.dims.d_model, |s| s[0]);
        let gamma = self
            .g
            .param(format!("{prefix}.weight"), Shape::new(&[d], DType::F32));
        let beta = self
            .g
            .param(format!("{prefix}.bias"), Shape::new(&[d], DType::F32));
        self.g.ln(x, gamma, beta, self.eps)
    }

    /// Elementwise multiply by a scalar constant.
    fn scale(&mut self, x: NodeId, s: f32) -> NodeId {
        let c = self.g.constant(s as f64, DType::F32);
        self.g.mul(x, c)
    }

    /// `[B, T, D]` → per-head `[B, H, T, dk]`.
    fn split_heads(&mut self, y: NodeId) -> NodeId {
        let (b, t, h, dk) = (self.batch, self.seq, self.dims.n_heads, self.dims.head_dim);
        let r = self
            .g
            .reshape_(y, vec![b as i64, t as i64, h as i64, dk as i64]);
        self.g.transpose_(r, vec![0, 2, 1, 3])
    }

    /// Transformer-XL relative shift: `[B, H, T, 2T-1]` scored against every
    /// relative offset → `[B, H, T, T]` aligned to absolute keys. Pad last axis
    /// by (1,0), reshape to `[B,H,2T,T]`, drop the first row, reshape back, keep
    /// the first `T` columns.
    fn rel_shift(&mut self, x: NodeId, p: usize) -> NodeId {
        let (b, h, t) = (self.batch, self.dims.n_heads, self.seq);
        let padded = self.g.pad_(
            x,
            vec![[0, 0], [0, 0], [0, 0], [1, 0]],
            PadMode::Constant(0.0),
        );
        let viewed = self
            .g
            .reshape_(padded, vec![b as i64, h as i64, (p + 1) as i64, t as i64]);
        let dropped = self.g.narrow_(viewed, 2, 1, p);
        let back = self
            .g
            .reshape_(dropped, vec![b as i64, h as i64, t as i64, p as i64]);
        self.g.narrow_(back, 3, 0, t)
    }

    // ── Conformer sub-blocks ────────────────────────────────────────────────

    /// Macaron feed-forward: `linear2(swish(linear1(x)))`.
    fn ffn(&mut self, x: NodeId, prefix: &str) -> Result<NodeId> {
        let h = self.linear(x, &format!("{prefix}.linear1.weight"), true)?;
        let h = self.g.silu(h);
        self.linear(h, &format!("{prefix}.linear2.weight"), true)
    }

    /// Relative-position multi-head self-attention (Transformer-XL), built from
    /// primitives so the `pos_bias_u/v` + `rel_shift` bias survives. `pos_emb`
    /// is the shared `[1, 2T-1, d_model]` sinusoidal constant.
    fn attention(&mut self, x: NodeId, pos_emb: NodeId, prefix: &str) -> Result<NodeId> {
        let (b, t, h, dk, d) = (
            self.batch,
            self.seq,
            self.dims.n_heads,
            self.dims.head_dim,
            self.dims.d_model,
        );
        let p = 2 * t - 1;

        let q = self.linear(x, &format!("{prefix}.linear_q.weight"), true)?;
        let k = self.linear(x, &format!("{prefix}.linear_k.weight"), true)?;
        let v = self.linear(x, &format!("{prefix}.linear_v.weight"), true)?;
        let qh = self.split_heads(q);
        let kh = self.split_heads(k);
        let vh = self.split_heads(v);

        // pos_bias_u/v: [H, dk] → [1, H, 1, dk], added to q.
        let u = self.g.param(
            format!("{prefix}.pos_bias_u"),
            Shape::new(&[h, dk], DType::F32),
        );
        let vb = self.g.param(
            format!("{prefix}.pos_bias_v"),
            Shape::new(&[h, dk], DType::F32),
        );
        let u4 = self.g.reshape_(u, vec![1, h as i64, 1, dk as i64]);
        let v4 = self.g.reshape_(vb, vec![1, h as i64, 1, dk as i64]);
        let q_u = self.g.add(qh, u4);
        let q_v = self.g.add(qh, v4);

        // matrix_ac = (q+u) · kᵀ → [B, H, T, T].
        let kt = self.g.transpose_(kh, vec![0, 1, 3, 2]);
        let ac = self.g.mm(q_u, kt);

        // p = linear_pos(pos_emb) → [1, P, D] → [1, H, P, dk] → [1, H, dk, P].
        let pos_p = self.linear(pos_emb, &format!("{prefix}.linear_pos.weight"), false)?;
        let pos_r = self
            .g
            .reshape_(pos_p, vec![1, p as i64, h as i64, dk as i64]);
        let pos_h = self.g.transpose_(pos_r, vec![0, 2, 1, 3]);
        let pos_t = self.g.transpose_(pos_h, vec![0, 1, 3, 2]);

        // matrix_bd = rel_shift((q+v) · pᵀ)[.., :T] → [B, H, T, T].
        let bd_full = self.g.mm(q_v, pos_t);
        let bd = self.rel_shift(bd_full, p);

        // scores = (ac + bd) / √dk ; softmax ; · v.
        let scores = self.g.add(ac, bd);
        let scores = self.scale(scores, 1.0 / (dk as f32).sqrt());
        let attn = self.g.sm(scores, -1);
        let ctx = self.g.mm(attn, vh); // [B, H, T, dk]
        let ctx = self.g.transpose_(ctx, vec![0, 2, 1, 3]); // [B, T, H, dk]
        let ctx = self.g.reshape_(ctx, vec![b as i64, t as i64, d as i64]);

        self.linear(ctx, &format!("{prefix}.linear_out.weight"), true)
    }

    /// Conv-module norm over the channel axis (last, on `[B, T, D]`): BatchNorm
    /// with running stats, else LayerNorm.
    fn conv_norm(&mut self, x: NodeId, prefix: &str) -> NodeId {
        if !self.dims.conv_batch_norm {
            return self.layer_norm(x, &format!("{prefix}.batch_norm"));
        }
        let d = self.dims.d_model;
        let bn = format!("{prefix}.batch_norm");
        let gamma = self
            .g
            .param(format!("{bn}.weight"), Shape::new(&[d], DType::F32));
        let beta = self
            .g
            .param(format!("{bn}.bias"), Shape::new(&[d], DType::F32));
        let mean = self
            .g
            .param(format!("{bn}.running_mean"), Shape::new(&[d], DType::F32));
        let var = self
            .g
            .param(format!("{bn}.running_var"), Shape::new(&[d], DType::F32));
        self.g
            .batch_norm_inference(x, gamma, beta, mean, var, self.eps)
    }

    /// Conformer convolution module on channel-last `[B, T, D]`:
    /// pointwise → GLU → depthwise(time) → norm → Swish → pointwise.
    fn conv_module(&mut self, x: NodeId, prefix: &str) -> Result<NodeId> {
        let (b, t, d, k) = (
            self.batch,
            self.seq,
            self.dims.d_model,
            self.dims.conv_kernel,
        );

        // pointwise_conv1: [B,T,D] → [B,T,2D], then GLU over the channel halves.
        let pw1 = self.pointwise(x, &format!("{prefix}.pointwise_conv1.weight"), true)?;
        let a = self.g.narrow_(pw1, 2, 0, d);
        let gate = self.g.narrow_(pw1, 2, d, d);
        let sig = self.g.sigmoid(gate);
        let glu = self.g.mul(a, sig); // [B, T, D]

        // depthwise_conv over time: [B,T,D] → [B,D,T] → [B,D,1,T] conv → [B,D,T].
        let xt = self.g.transpose_(glu, vec![0, 2, 1]);
        let x4 = self.g.reshape_(xt, vec![b as i64, d as i64, 1, t as i64]);
        let dw_name = format!("{prefix}.depthwise_conv.weight");
        let dw_sh = self
            .shapes
            .shape_of(&dw_name)
            .ok_or_else(|| anyhow!("missing {dw_name}"))?; // [D, 1, k]
        let dw = self.g.param(&dw_name, Shape::new(dw_sh, DType::F32));
        let dw4 = self.g.reshape_(dw, vec![d as i64, 1, 1, k as i64]);
        let conv = self
            .g
            .conv2d(x4, dw4, [1, k], [1, 1], [0, (k - 1) / 2], [1, 1], d);
        let conv = self.g.reshape_(conv, vec![b as i64, d as i64, t as i64]);
        let conv = self.add_channel_bias(
            conv,
            &format!("{prefix}.depthwise_conv.bias"),
            vec![1, d as i64, 1],
        );
        let cl = self.g.transpose_(conv, vec![0, 2, 1]); // back to [B, T, D]

        let normed = self.conv_norm(cl, prefix);
        let act = self.g.silu(normed);
        self.pointwise(act, &format!("{prefix}.pointwise_conv2.weight"), true)
    }

    /// One full Conformer block (Macaron FFN → MHSA → Conv → Macaron FFN → norm),
    /// each sub-block pre-normed and residual-added.
    fn layer(&mut self, x: NodeId, i: usize, pos_emb: NodeId) -> Result<NodeId> {
        let pfx = format!("encoder.layers.{i}");

        let y = self.layer_norm(x, &format!("{pfx}.norm_feed_forward1"));
        let y = self.ffn(y, &format!("{pfx}.feed_forward1"))?;
        let y = self.scale(y, FC_FACTOR);
        let mut r = self.g.add(x, y);

        let y = self.layer_norm(r, &format!("{pfx}.norm_self_att"));
        let y = self.attention(y, pos_emb, &format!("{pfx}.self_attn"))?;
        r = self.g.add(r, y);

        let y = self.layer_norm(r, &format!("{pfx}.norm_conv"));
        let y = self.conv_module(y, &format!("{pfx}.conv"))?;
        r = self.g.add(r, y);

        let y = self.layer_norm(r, &format!("{pfx}.norm_feed_forward2"));
        let y = self.ffn(y, &format!("{pfx}.feed_forward2"))?;
        let y = self.scale(y, FC_FACTOR);
        r = self.g.add(r, y);

        Ok(self.layer_norm(r, &format!("{pfx}.norm_out")))
    }

    /// Bake the `[1, 2T-1, d_model]` relative sinusoidal position table (NeMo
    /// `RelPositionalEncoding.pe`) as an `Op::Constant`. Row `r` carries the
    /// relative offset `(T-1) - r`, so row 0 is `+(T-1)` and the last `-(T-1)`.
    fn pos_table(&mut self) -> NodeId {
        let (t, d) = (self.seq, self.dims.d_model);
        let p = 2 * t - 1;
        let mut data = vec![0f32; p * d];
        for r in 0..p {
            let pos = (t as f64) - 1.0 - (r as f64);
            for j in 0..d / 2 {
                let angle = pos * POS_ENC_BASE.powf(-(2.0 * j as f64) / d as f64);
                data[r * d + 2 * j] = angle.sin() as f32;
                data[r * d + 2 * j + 1] = angle.cos() as f32;
            }
        }
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.g.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(&[1, p, d], DType::F32),
        )
    }

    /// The encoder input: either the mel `pre_encode` conv subsampling (which
    /// downsamples and sets [`Encoder::seq`] to the resulting `T`), or the
    /// already-subsampled hidden state `[B, seq, D]`.
    fn input(&mut self, cfg: &NemoConfig, mel_frames: Option<usize>) -> Result<NodeId> {
        if let Some(frames) = mel_frames {
            if let Some(hidden) = self.subsampling(cfg, frames)? {
                return Ok(hidden);
            }
        }
        Ok(self.g.input(
            "x",
            Shape::new(&[self.batch, self.seq, self.dims.d_model], DType::F32),
        ))
    }

    /// Prepend the NeMo `ConvSubsampling` front-end (`striding` / `dw_striding`),
    /// returning the encoder hidden state `[B, T, d_model]` and setting
    /// [`Encoder::seq`] to `T`, or `None` when there is no usable conv front-end.
    ///
    /// Conv layers are discovered from `pre_encode.conv.{j}.weight` and each is
    /// classified by weight shape: the input conv, a depthwise conv
    /// (`Cin/groups == 1`), or a pointwise `1×1` (stride 1). ReLU follows the
    /// input conv and each pointwise conv. The downsampled `T`/`F` come straight
    /// from the conv shape inference (which matches NeMo's `calc_length` floor
    /// formula), so no length arithmetic is duplicated here.
    fn subsampling(&mut self, cfg: &NemoConfig, frames: usize) -> Result<Option<NodeId>> {
        if !self.shapes.has("pre_encode.conv.0.weight") || !self.shapes.has("pre_encode.out.weight")
        {
            return Ok(None);
        }
        let Some(feat_in) = cfg_usize(
            cfg,
            &[
                "preprocessor.features",
                "preprocessor.feat_in",
                "encoder.feat_in",
                "cfg.preprocessor.features",
            ],
        ) else {
            return Ok(None);
        };

        // Conv weights in Sequential order (ReLU indices carry no weights).
        let convs: Vec<(usize, &[usize])> = (0..64)
            .filter_map(|j| {
                self.shapes
                    .shape_of(&format!("pre_encode.conv.{j}.weight"))
                    .filter(|s| s.len() == 4)
                    .map(|s| (j, s))
            })
            .collect();
        if convs.is_empty() {
            return Ok(None);
        }

        // Mel [B, frames, feat_in] → NCHW image [B, 1, frames, feat_in].
        let x = self
            .g
            .input("x", Shape::new(&[self.batch, frames, feat_in], DType::F32));
        let mut cur = self
            .g
            .reshape_(x, vec![self.batch as i64, 1, frames as i64, feat_in as i64]);

        let mut first = true;
        for (j, w_sh) in convs {
            let (c_out, c_in_pg, kh, kw) = (w_sh[0], w_sh[1], w_sh[2], w_sh[3]);
            let pointwise = kh == 1 && kw == 1;
            let depthwise = !pointwise && !first && c_in_pg == 1;
            let groups = if depthwise { c_out } else { 1 };
            let stride = if pointwise { [1, 1] } else { [2, 2] };
            let pad = if pointwise {
                [0, 0]
            } else {
                [(kh - 1) / 2, (kw - 1) / 2]
            };
            let weight = format!("pre_encode.conv.{j}.weight");
            let w = self.g.param(&weight, Shape::new(w_sh, DType::F32));
            cur = self.g.conv2d(cur, w, [kh, kw], stride, pad, [1, 1], groups);
            cur = self.add_channel_bias(
                cur,
                &format!("pre_encode.conv.{j}.bias"),
                vec![1, c_out as i64, 1, 1],
            );
            if first || pointwise {
                cur = self.g.relu(cur);
            }
            first = false;
        }

        // Conv output [B, C, T', F'] — read the downsampled dims off the graph.
        let out: Vec<usize> = self
            .g
            .shape(cur)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        if out.len() != 4 {
            return Ok(None);
        }
        let (c, tp, fp) = (out[1], out[2], out[3]);
        // [B, C, T', F'] → [B, T', C, F'] → [B, T', C·F'].
        let tr = self.g.transpose_(cur, vec![0, 2, 1, 3]);
        let flat = self
            .g
            .reshape_(tr, vec![self.batch as i64, tp as i64, (c * fp) as i64]);
        let hidden = self.linear(flat, "pre_encode.out.weight", true)?;
        self.seq = tp;
        Ok(Some(hidden))
    }
}

// ── Public builders ─────────────────────────────────────────────────────────

/// Build the full Conformer encoder trunk over hidden states `[batch, seq_len,
/// d_model]`, or `None` when the checkpoint is not a recognizable Conformer.
///
/// The returned graph has one input `x` (the subsampled encoder hidden states,
/// or the mel features when [`EncoderOpts::mel_frames`] is set) and one output
/// (the encoder features after `norm_out` of the last layer).
pub fn build_nemo_encoder_graph(
    cfg: &NemoConfig,
    shapes: &impl TensorShapes,
    opts: &EncoderOpts,
) -> Result<Option<Graph>> {
    let Some(dims) = resolve_dims(cfg, shapes) else {
        return Ok(None);
    };
    if opts.batch == 0 || opts.seq_len == 0 {
        return Err(anyhow!("encoder batch/seq_len must be non-zero"));
    }

    let mut enc = Encoder::new(&opts.name, shapes, dims, opts.batch, opts.seq_len, opts.eps);
    let hidden = enc.input(cfg, opts.mel_frames)?;
    // NeMo scales the subsampled hidden states by √d_model before the layers.
    let mut h = if opts.xscale {
        enc.scale(hidden, (dims.d_model as f32).sqrt())
    } else {
        hidden
    };
    let pos = enc.pos_table();
    for i in 0..dims.n_layers {
        h = enc.layer(h, i, pos)?;
    }
    Ok(Some(enc.finish(h)))
}

/// Build a graph for a `.nemo` checkpoint.
///
/// Prefers the full [Conformer encoder](build_nemo_encoder_graph); if the
/// checkpoint is not a Conformer, falls back to a single-Linear probe that
/// still binds a real weight so RLXP can compile/run a slice.
pub fn build_nemo_probe_graph(model: &NemoModel, name: &str) -> Result<Option<Graph>> {
    let opts = EncoderOpts {
        name: name.to_string(),
        ..EncoderOpts::default()
    };
    if let Some(g) = build_nemo_encoder_graph(model.config(), model, &opts)? {
        return Ok(Some(g));
    }
    Ok(Some(build_linear_probe(model, name)))
}

/// The historical fallback: bind one real Linear weight (or an identity input
/// sized to the model's feature dim) so there is always a valid graph.
fn build_linear_probe(model: &NemoModel, name: &str) -> Graph {
    let cfg = model.config();
    let d_model = cfg_usize(
        cfg,
        &[
            "encoder.d_model",
            "model.encoder.d_model",
            "cfg.encoder.d_model",
        ],
    )
    .unwrap_or(1);
    let features =
        cfg_usize(cfg, &["preprocessor.features", "cfg.preprocessor.features"]).unwrap_or(d_model);

    let chosen = [
        "encoder.layers.0.feed_forward.linear1.weight",
        "encoder.layers.0.fc1.weight",
        "encoder.layers.0.linear1.weight",
        "encoder.layers.0.self_attn.linear_q.weight",
    ]
    .into_iter()
    .find_map(|n| {
        model
            .shape_of(n)
            .filter(|sh| sh.len() == 2)
            .map(|sh| (n.to_string(), sh.to_vec()))
    });

    let mut g = Graph::new(name);
    if let Some((weight, shape)) = chosen {
        let in_dim = shape[1];
        let x = g.input("x", Shape::new(&[1, in_dim], DType::F32));
        let w = g.param(weight, Shape::new(&shape, DType::F32));
        let wt = g.transpose_(w, vec![1, 0]);
        let y = g.mm(x, wt);
        g.set_outputs(vec![y]);
    } else {
        let dim = features.min(d_model).max(1);
        let x = g.input("x", Shape::new(&[1, dim], DType::F32));
        g.set_outputs(vec![x]);
    }
    g
}

/// Summarize config fields useful for graph builders.
pub fn nemo_arch_summary(cfg: &NemoConfig) -> String {
    format!(
        "d_model={:?} n_layers={:?} n_heads={:?} features={:?}",
        cfg.get_usize("encoder.d_model"),
        cfg.get_usize("encoder.n_layers"),
        cfg.get_usize("encoder.n_heads"),
        cfg.get_usize("preprocessor.features"),
    )
}
