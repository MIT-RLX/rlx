// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::MaskKind;
use rlx_ir::shape;

use std::sync::{Arc, Mutex};

use super::BlockStage;
use super::qwen3_decoder::per_head_rms;
use super::self_attn::repeat_kv;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct Qwen3DecodeLayerSpec {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_group_size: usize,
    pub eps: f32,
    pub use_custom_mask: bool,
    pub hidden_shape: rlx_ir::Shape,
    pub batch: usize,
    pub qk_norm: bool,
    pub attention_bias: bool,
}

#[derive(Debug, Clone)]
pub struct Qwen3DecodeLayerStage {
    pub layer_prefix: String,
    pub spec: Qwen3DecodeLayerSpec,
    pub layer_idx: usize,
    pub kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    pub qk_out: Option<Arc<Mutex<Vec<rlx_ir::HirNodeId>>>>,
}

impl Qwen3DecodeLayerStage {
    pub fn layer(
        layer_idx: usize,
        spec: Qwen3DecodeLayerSpec,
        kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            spec,
            layer_idx,
            kv_out,
            qk_out: None,
        }
    }

    pub fn layer_with_qk(
        layer_idx: usize,
        spec: Qwen3DecodeLayerSpec,
        kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
        qk_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            spec,
            layer_idx,
            kv_out,
            qk_out: Some(qk_out),
        }
    }
}

impl BlockStage for Qwen3DecodeLayerStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let decode = ctx
            .state
            .decode
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Qwen3DecodeLayer requires BindDecodeInputs"))?;
        let zero_beta_h = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("Qwen3DecodeLayer requires ZeroBeta"))?;
        let zero_beta_dh = ctx
            .state
            .named
            .get("zero_beta.head")
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Qwen3DecodeLayer requires zero_beta.head"))?;

        let lp = &self.layer_prefix;
        let spec = &self.spec;
        let nh = spec.num_heads;
        let nkv = spec.num_kv_heads;
        let dh = spec.head_dim;
        let batch = spec.batch;

        // Decode is weight-read-bandwidth bound (batch=1 GEMV): storing the
        // projection weights F16-resident halves the bytes read per token.
        // Opt-in `RLX_QWEN3_F16_WEIGHTS`; norms/biases stay F32.
        let w_dt = if rlx_ir::env::flag("RLX_QWEN3_F16_WEIGHTS") {
            rlx_ir::DType::F16
        } else {
            rlx_ir::DType::F32
        };
        let in_ln_g = ctx.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
        // Projections route through `resolve_linear`: a packed `WeightSource`
        // (K-quant GGUF) yields a fused `DequantMatMul` over the U8 blob (no F32
        // weight residency — the low-memory packed decode path); an F32 source
        // falls back to a dense `mm`, byte-identical to the default runner.
        // Fused QKV (opt-in `RLX_QWEN3_FUSED_QKV`, packed only): one DequantMatMul
        // over the concatenated q/k/v weights instead of 3 GEMVs → 2 fewer kernel
        // dispatches per layer. `None` on dense/non-packed → per-key path below.
        let use_fused_qkv = rlx_ir::env::flag("RLX_QWEN3_FUSED_QKV");
        let qkv_fused = if use_fused_qkv {
            Some(ctx.resolve_linear_fused(
                &[
                    &format!("{lp}.self_attn.q_proj.weight"),
                    &format!("{lp}.self_attn.k_proj.weight"),
                    &format!("{lp}.self_attn.v_proj.weight"),
                ],
                true,
                w_dt,
            )?)
        } else {
            None
        };
        let (q_w, k_w, v_w) = if use_fused_qkv {
            (None, None, None)
        } else {
            (
                Some(ctx.resolve_linear(&format!("{lp}.self_attn.q_proj.weight"), true, w_dt)?),
                Some(ctx.resolve_linear(&format!("{lp}.self_attn.k_proj.weight"), true, w_dt)?),
                Some(ctx.resolve_linear(&format!("{lp}.self_attn.v_proj.weight"), true, w_dt)?),
            )
        };
        let o_w = ctx.resolve_linear(&format!("{lp}.self_attn.o_proj.weight"), true, w_dt)?;
        let post_ln_g = ctx.load_param(&format!("{lp}.post_attention_layernorm.weight"), false)?;
        let gate_w = ctx.resolve_linear(&format!("{lp}.mlp.gate_proj.weight"), true, w_dt)?;
        let up_w = ctx.resolve_linear(&format!("{lp}.mlp.up_proj.weight"), true, w_dt)?;
        let down_w = ctx.resolve_linear(&format!("{lp}.mlp.down_proj.weight"), true, w_dt)?;
        let (q_bias, k_bias, v_bias) = if spec.attention_bias {
            (
                Some(ctx.load_param(&format!("{lp}.self_attn.q_proj.bias"), false)?),
                Some(ctx.load_param(&format!("{lp}.self_attn.k_proj.bias"), false)?),
                Some(ctx.load_param(&format!("{lp}.self_attn.v_proj.bias"), false)?),
            )
        } else {
            (None, None, None)
        };
        let (q_norm_g, k_norm_g) = if spec.qk_norm {
            (
                Some(ctx.load_param(&format!("{lp}.self_attn.q_norm.weight"), false)?),
                Some(ctx.load_param(&format!("{lp}.self_attn.k_norm.weight"), false)?),
            )
        } else {
            (None, None)
        };

        let past_k = decode.past_k[self.layer_idx];
        let past_v = decode.past_v[self.layer_idx];

        let mut gb = HirMut::new(ctx.hir());
        let skip = input.id;
        let normed_in = gb.rms_norm(skip, in_ln_g, zero_beta_h, spec.eps);
        use crate::context::FusedProj;
        let (mut q, mut k, mut v) = match &qkv_fused {
            Some(FusedProj::Fused { weight, dims }) => {
                // One GEMV → [.., Σ out], then narrow_-split into q ‖ k ‖ v (the
                // combined weight rows are stacked in that order).
                let qkv = weight.emit(&mut gb, normed_in);
                let la = gb.shape(qkv).rank() - 1;
                let q = gb.narrow_(qkv, la, 0, dims[0]);
                let k = gb.narrow_(qkv, la, dims[0], dims[1]);
                let v = gb.narrow_(qkv, la, dims[0] + dims[1], dims[2]);
                (q, k, v)
            }
            Some(FusedProj::Separate(ws)) => (
                ws[0].emit(&mut gb, normed_in),
                ws[1].emit(&mut gb, normed_in),
                ws[2].emit(&mut gb, normed_in),
            ),
            None => (
                q_w.as_ref().unwrap().emit(&mut gb, normed_in),
                k_w.as_ref().unwrap().emit(&mut gb, normed_in),
                v_w.as_ref().unwrap().emit(&mut gb, normed_in),
            ),
        };

        if let (Some(qb), Some(kb), Some(vb)) = (q_bias, k_bias, v_bias) {
            q = gb.add(q, qb);
            k = gb.add(k, kb);
            v = gb.add(v, vb);
        }

        let (q_rope_in, k_rope_in) = if let (Some(qng), Some(kng)) = (q_norm_g, k_norm_g) {
            let q_normed = per_head_rms(&mut gb, q, qng, zero_beta_dh, batch, 1, nh, dh, spec.eps);
            let k_normed = per_head_rms(&mut gb, k, kng, zero_beta_dh, batch, 1, nkv, dh, spec.eps);
            (q_normed, k_normed)
        } else {
            (q, k)
        };

        let q_rope = gb.rope(q_rope_in, decode.cos, decode.sin, dh);
        let k_rope = gb.rope(k_rope_in, decode.cos, decode.sin, dh);

        // F16-resident KV cache (opt-in `RLX_QWEN3_F16_KV`): store K/V half-sized
        // so decode attention reads half the KV bytes. Q, softmax accumulation,
        // and the attention output stay F32 (`sdpa_decode_m1_f16kv`). `past_k/v`
        // are declared F16 to match (see the decode graph builder).
        let (k_rope, v) = if rlx_ir::env::flag("RLX_QWEN3_F16_KV") {
            (
                gb.cast(k_rope, rlx_ir::DType::F16),
                gb.cast(v, rlx_ir::DType::F16),
            )
        } else {
            (k_rope, v)
        };

        // In-place KV append (opt-in `RLX_QWEN3_INPLACE_KV`): write the new row
        // into `past_k/v` at index `past_seq` instead of concat-copying the whole
        // O(context) cache. Requires the flow to declare `past_k/v` one row
        // larger (`[batch, past_seq+1, kv_dim]`) so the aliased output fits.
        let (new_k, new_v) = if rlx_ir::env::flag("RLX_QWEN3_INPLACE_KV") {
            let pos = gb.shape(past_k).dim(1).unwrap_static() - 1;
            (
                gb.kv_append(past_k, k_rope, 1, pos),
                gb.kv_append(past_v, v, 1, pos),
            )
        } else {
            (
                gb.concat_(vec![past_k, k_rope], 1),
                gb.concat_(vec![past_v, v], 1),
            )
        };
        self.kv_out.lock().expect("kv out").push(new_k);
        self.kv_out.lock().expect("kv out").push(new_v);

        // GQA-native (opt-in `RLX_QWEN3_GQA_NATIVE`): the SDPA kernels index K/V
        // by shared kv head internally, so pass the un-expanded nkv-head K/V and
        // skip `repeat_kv`'s Expand — which writes 2× the KV that attention then
        // reads (~75MB) vs ~30MB reading the base KV in place (shared-head
        // re-reads hit L2). Measured FASTER on Metal decode: ~11.4→10.4ms wait,
        // 82→89 tok/s, token-identical (M4 Pro, with RLX_QWEN3_BAKE_WEIGHTS).
        let (k_attn, v_attn) = if rlx_ir::env::flag("RLX_QWEN3_GQA_NATIVE") {
            (new_k, new_v)
        } else {
            (
                repeat_kv(&mut gb, new_k, nkv, dh, spec.kv_group_size),
                repeat_kv(&mut gb, new_v, nkv, dh, spec.kv_group_size),
            )
        };
        if let Some(ref sink) = self.qk_out {
            sink.lock().expect("qwen3 decode qk out").push(q_rope);
            sink.lock().expect("qwen3 decode qk out").push(k_attn);
        }

        let attn_shape = shape::attention_shape(gb.shape(q_rope));
        let attn = if spec.use_custom_mask {
            let mask = decode
                .mask
                .ok_or_else(|| anyhow::anyhow!("custom mask requested but not bound"))?;
            gb.attention(q_rope, k_attn, v_attn, mask, nh, dh, attn_shape)
        } else {
            gb.attention_kind(q_rope, k_attn, v_attn, nh, dh, MaskKind::Causal, attn_shape)
        };

        let attn_out = o_w.emit(&mut gb, attn);
        let post_attn = gb.add(skip, attn_out);
        let normed_post = gb.rms_norm(post_attn, post_ln_g, zero_beta_h, spec.eps);
        let gate = gate_w.emit(&mut gb, normed_post);
        let up = up_w.emit(&mut gb, normed_post);
        let gate_act = gb.silu(gate);
        let swiglu = gb.mul(gate_act, up);
        let ffn_out = down_w.emit(&mut gb, swiglu);
        let h_id = gb.add(post_attn, ffn_out);

        Ok(Some(ctx.wrap(h_id, spec.hidden_shape.clone())))
    }
}
