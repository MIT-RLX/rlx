// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use std::sync::{Arc, Mutex};

use crate::context::FlowCtx;
use crate::value::FlowValue;
/// Export RoPE(K) and V projections as side outputs for KV-cache prefill.
#[derive(Debug, Clone)]
pub struct LlamaKvTapStage {
    pub layer_prefix: String,
    pub head_dim: usize,
    pub eps: f32,
    pub outputs: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
}

impl LlamaKvTapStage {
    pub fn layer(
        layer_idx: usize,
        head_dim: usize,
        eps: f32,
        sink: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            head_dim,
            eps,
            outputs: sink,
        }
    }

    pub fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<()> {
        let lp = &self.layer_prefix;
        let zero_beta = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("LlamaKvTap requires ZeroBeta"))?;
        let cos = ctx
            .state
            .rope_cos
            .ok_or_else(|| anyhow::anyhow!("LlamaKvTap requires RopeTables"))?;
        let sin = ctx
            .state
            .rope_sin
            .ok_or_else(|| anyhow::anyhow!("LlamaKvTap requires RopeTables"))?;

        let in_ln_g = ctx.load_param(&format!("{lp}.input_layernorm.weight"), false)?;
        let k_w = ctx.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
        let v_w = ctx.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;

        let mut gb = HirMut::new(ctx.hir());
        let normed_in = gb.rms_norm(input.id, in_ln_g, zero_beta, self.eps);
        let k = gb.mm(normed_in, k_w);
        let v = gb.mm(normed_in, v_w);
        let k_rope = gb.rope(k, cos, sin, self.head_dim);

        self.outputs.lock().expect("kv tap sink").push(k_rope);
        self.outputs.lock().expect("kv tap sink").push(v);
        Ok(())
    }
}
