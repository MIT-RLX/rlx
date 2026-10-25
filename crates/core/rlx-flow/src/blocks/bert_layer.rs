// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::Shape;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::{FlowCtx, FusedProj, LinearWeight};
use crate::value::FlowValue;

/// QKV weight layout for BERT-family encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertQkvStyle {
    /// `attention.self.{query,key,value}` (BERT / MiniLM).
    Bert,
    /// `attention.attn.{q,k,v}` (mpnet-style).
    Mpnet,
}

#[derive(Debug, Clone)]
pub struct BertEncoderLayerSpec {
    pub layer_prefix: String,
    pub qkv_style: BertQkvStyle,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub attention_mask_input: String,
}

impl BertEncoderLayerSpec {
    pub fn hf(
        layer_prefix: impl Into<String>,
        qkv_style: BertQkvStyle,
        hidden_size: usize,
        num_heads: usize,
        eps: f32,
    ) -> Self {
        Self {
            layer_prefix: layer_prefix.into(),
            qkv_style,
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads,
            eps,
            attention_mask_input: "attention_mask".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BertEncoderLayerStage {
    pub spec: BertEncoderLayerSpec,
}

impl BertEncoderLayerStage {
    pub fn new(spec: BertEncoderLayerSpec) -> Self {
        Self { spec }
    }
}

impl BlockStage for BertEncoderLayerStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let spec = &self.spec;
        let h = spec.hidden_size;
        let nh = spec.num_heads;
        let dh = spec.head_dim;
        let lp = &spec.layer_prefix;

        // Packed (int8 / GGUF) projections run as fused `DequantMatMul`s over
        // their quant blobs: no f32 copy of the weights, and none transposed.
        // Dense weights keep the host-fused QKV graph exactly as before.
        let (wq_key, wk_key, wv_key) = qkv_weight_keys(lp, spec.qkv_style);
        let packed = ctx.weights.has_packed(&wq_key);
        let (qkv_w, qkv_b) = if packed {
            let w = match ctx.resolve_linear_fused(
                &[&wq_key, &wk_key, &wv_key],
                true,
                rlx_ir::DType::F32,
            )? {
                FusedProj::Fused { weight, .. } => weight,
                FusedProj::Separate(_) => {
                    anyhow::bail!("BertEncoderLayer: q/k/v of {lp} are packed in different schemes")
                }
            };
            (w, load_fused_qkv_bias(ctx, lp, h, spec.qkv_style)?)
        } else {
            let (w, b) = load_fused_qkv(ctx, lp, h, spec.qkv_style)?;
            (LinearWeight::Dense(w), b)
        };
        let linear = |ctx: &mut FlowCtx<'_>, key: String| -> Result<LinearWeight> {
            if packed {
                ctx.resolve_linear(&key, true, rlx_ir::DType::F32)
            } else {
                Ok(LinearWeight::Dense(ctx.load_matrix(&key, true)?))
            }
        };

        let out_w = linear(ctx, format!("{lp}.attention.output.dense.weight"))?;
        let out_b = ctx.load_param(&format!("{lp}.attention.output.dense.bias"), false)?;
        let ln1_g = ctx.load_param(&format!("{lp}.attention.output.LayerNorm.weight"), false)?;
        let ln1_b = ctx.load_param(&format!("{lp}.attention.output.LayerNorm.bias"), false)?;
        let ln2_g = ctx.load_param(&format!("{lp}.output.LayerNorm.weight"), false)?;
        let ln2_b = ctx.load_param(&format!("{lp}.output.LayerNorm.bias"), false)?;
        let int_w = linear(ctx, format!("{lp}.intermediate.dense.weight"))?;
        let int_b = ctx.load_param(&format!("{lp}.intermediate.dense.bias"), false)?;
        let out2_w = linear(ctx, format!("{lp}.output.dense.weight"))?;
        let out2_b = ctx.load_param(&format!("{lp}.output.dense.bias"), false)?;

        let mask_id = ctx
            .state
            .inputs
            .get(&spec.attention_mask_input)
            .map(|(id, _)| *id)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "BertEncoderLayer requires input `{}`",
                    spec.attention_mask_input
                )
            })?;

        let mut gb = HirMut::new(ctx.hir());
        let skip = input.id;

        let qkv_mm = qkv_w.emit(&mut gb, skip);
        let qkv = gb.add(qkv_mm, qkv_b);
        let last_ax = gb.shape(qkv).rank() - 1;
        let q = gb.narrow_(qkv, last_ax, 0, h);
        let k = gb.narrow_(qkv, last_ax, h, h);
        let v = gb.narrow_(qkv, last_ax, 2 * h, h);
        let attn = gb.attention_(q, k, v, mask_id, nh, dh);

        let attn_mm = out_w.emit(&mut gb, attn);
        let attn_out = gb.add(attn_mm, out_b);
        let res1 = gb.add(attn_out, skip);
        let normed1 = gb.ln(res1, ln1_g, ln1_b, spec.eps);

        let int_mm = int_w.emit(&mut gb, normed1);
        let int_add = gb.add(int_mm, int_b);
        let ffn_int = gb.gelu(int_add);
        let out2_mm = out2_w.emit(&mut gb, ffn_int);
        let ffn_out = gb.add(out2_mm, out2_b);
        let res2 = gb.add(ffn_out, normed1);
        let out = gb.ln(res2, ln2_g, ln2_b, spec.eps);

        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}

/// The q/k/v weight and bias keys of one layer: `[wq, wk, wv, bq, bk, bv]`.
fn qkv_keys(layer_prefix: &str, style: BertQkvStyle) -> [String; 6] {
    let names = match style {
        BertQkvStyle::Bert => [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
        ],
        BertQkvStyle::Mpnet => ["attention.attn.q", "attention.attn.k", "attention.attn.v"],
    };
    let [q, k, v] = names.map(|n| format!("{layer_prefix}.{n}"));
    [
        format!("{q}.weight"),
        format!("{k}.weight"),
        format!("{v}.weight"),
        format!("{q}.bias"),
        format!("{k}.bias"),
        format!("{v}.bias"),
    ]
}

fn qkv_weight_keys(layer_prefix: &str, style: BertQkvStyle) -> (String, String, String) {
    let [q, k, v, ..] = qkv_keys(layer_prefix, style);
    (q, k, v)
}

/// The q, k, v biases concatenated, for a fused (packed) QKV projection.
fn load_fused_qkv_bias(
    ctx: &mut FlowCtx<'_>,
    layer_prefix: &str,
    h: usize,
    style: BertQkvStyle,
) -> Result<rlx_ir::HirNodeId> {
    let [_, _, _, bq_key, bk_key, bv_key] = qkv_keys(layer_prefix, style);
    let mut fused_b = Vec::with_capacity(3 * h);
    for key in [&bq_key, &bk_key, &bv_key] {
        fused_b.extend(ctx.weights.take(key, false)?.0);
    }
    let b_name = format!("{layer_prefix}.attention.qkv.bias");
    let b_id = ctx
        .hir()
        .param(&b_name, Shape::new(&[3 * h], rlx_ir::DType::F32));
    ctx.params.insert(b_name, fused_b);
    Ok(b_id)
}

fn load_fused_qkv(
    ctx: &mut FlowCtx<'_>,
    layer_prefix: &str,
    h: usize,
    style: BertQkvStyle,
) -> Result<(rlx_ir::HirNodeId, rlx_ir::HirNodeId)> {
    let [wq_key, wk_key, wv_key, bq_key, bk_key, bv_key] = qkv_keys(layer_prefix, style);

    let wq_data = ctx.weights.take(&wq_key, true)?;
    let wk_data = ctx.weights.take(&wk_key, true)?;
    let wv_data = ctx.weights.take(&wv_key, true)?;
    let (bq_data, _) = ctx.weights.take(&bq_key, false)?;
    let (bk_data, _) = ctx.weights.take(&bk_key, false)?;
    let (bv_data, _) = ctx.weights.take(&bv_key, false)?;

    let w_name = format!("{layer_prefix}.attention.qkv.weight");
    let b_name = format!("{layer_prefix}.attention.qkv.bias");

    let (wq, _) = wq_data;
    let (wk, _) = wk_data;
    let (wv, _) = wv_data;

    let mut fused_w = vec![0f32; h * 3 * h];
    let mut fused_b = vec![0f32; 3 * h];
    for row in 0..h {
        fused_w[row * 3 * h..row * 3 * h + h].copy_from_slice(&wq[row * h..(row + 1) * h]);
        fused_w[row * 3 * h + h..row * 3 * h + 2 * h].copy_from_slice(&wk[row * h..(row + 1) * h]);
        fused_w[row * 3 * h + 2 * h..row * 3 * h + 3 * h]
            .copy_from_slice(&wv[row * h..(row + 1) * h]);
    }
    fused_b[..h].copy_from_slice(&bq_data);
    fused_b[h..2 * h].copy_from_slice(&bk_data);
    fused_b[2 * h..].copy_from_slice(&bv_data);

    let w_id = ctx
        .hir()
        .param(&w_name, Shape::new(&[h, 3 * h], rlx_ir::DType::F32));
    let b_id = ctx
        .hir()
        .param(&b_name, Shape::new(&[3 * h], rlx_ir::DType::F32));
    ctx.params.insert(w_name, fused_w);
    ctx.params.insert(b_name, fused_b);
    Ok((w_id, b_id))
}
