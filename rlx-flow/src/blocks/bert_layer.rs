// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::HirGraphExt;
use rlx_ir::Shape;

use super::BlockStage;
use crate::context::FlowCtx;
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
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let spec = &self.spec;
        let h = spec.hidden_size;
        let nh = spec.num_heads;
        let dh = spec.head_dim;
        let lp = &spec.layer_prefix;

        let (qkv_w, qkv_b) = load_fused_qkv(ctx, lp, h, spec.qkv_style)?;

        let out_w = ctx.load_param(&format!("{lp}.attention.output.dense.weight"), true)?;
        let out_b = ctx.load_param(&format!("{lp}.attention.output.dense.bias"), false)?;
        let ln1_g = ctx.load_param(&format!("{lp}.attention.output.LayerNorm.weight"), false)?;
        let ln1_b = ctx.load_param(&format!("{lp}.attention.output.LayerNorm.bias"), false)?;
        let ln2_g = ctx.load_param(&format!("{lp}.output.LayerNorm.weight"), false)?;
        let ln2_b = ctx.load_param(&format!("{lp}.output.LayerNorm.bias"), false)?;
        let int_w = ctx.load_param(&format!("{lp}.intermediate.dense.weight"), true)?;
        let int_b = ctx.load_param(&format!("{lp}.intermediate.dense.bias"), false)?;
        let out2_w = ctx.load_param(&format!("{lp}.output.dense.weight"), true)?;
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

        let qkv_mm = gb.mm(skip, qkv_w);
        let qkv = gb.add(qkv_mm, qkv_b);
        let last_ax = gb.shape(qkv).rank() - 1;
        let q = gb.narrow_(qkv, last_ax, 0, h);
        let k = gb.narrow_(qkv, last_ax, h, h);
        let v = gb.narrow_(qkv, last_ax, 2 * h, h);
        let attn = gb.attention_(q, k, v, mask_id, nh, dh);

        let attn_mm = gb.mm(attn, out_w);
        let attn_out = gb.add(attn_mm, out_b);
        let res1 = gb.add(attn_out, skip);
        let normed1 = gb.ln(res1, ln1_g, ln1_b, spec.eps);

        let int_mm = gb.mm(normed1, int_w);
        let int_add = gb.add(int_mm, int_b);
        let ffn_int = gb.gelu(int_add);
        let out2_mm = gb.mm(ffn_int, out2_w);
        let ffn_out = gb.add(out2_mm, out2_b);
        let res2 = gb.add(ffn_out, normed1);
        let out = gb.ln(res2, ln2_g, ln2_b, spec.eps);

        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}

fn load_fused_qkv(
    ctx: &mut FlowCtx<'_>,
    layer_prefix: &str,
    h: usize,
    style: BertQkvStyle,
) -> Result<(rlx_ir::HirNodeId, rlx_ir::HirNodeId)> {
    let (wq_key, wk_key, wv_key, bq_key, bk_key, bv_key) = match style {
        BertQkvStyle::Bert => (
            format!("{layer_prefix}.attention.self.query.weight"),
            format!("{layer_prefix}.attention.self.key.weight"),
            format!("{layer_prefix}.attention.self.value.weight"),
            format!("{layer_prefix}.attention.self.query.bias"),
            format!("{layer_prefix}.attention.self.key.bias"),
            format!("{layer_prefix}.attention.self.value.bias"),
        ),
        BertQkvStyle::Mpnet => (
            format!("{layer_prefix}.attention.attn.q.weight"),
            format!("{layer_prefix}.attention.attn.k.weight"),
            format!("{layer_prefix}.attention.attn.v.weight"),
            format!("{layer_prefix}.attention.attn.q.bias"),
            format!("{layer_prefix}.attention.attn.k.bias"),
            format!("{layer_prefix}.attention.attn.v.bias"),
        ),
    };

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
