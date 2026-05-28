// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use std::sync::{Arc, Mutex};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Export RoPE(K) and V after Gemma input RMSNorm (prefill KV-cache fill).
#[derive(Debug, Clone)]
pub struct GemmaKvTapStage {
    pub layer_prefix: String,
    pub head_dim: usize,
    pub outputs: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
}

impl GemmaKvTapStage {
    pub fn layer(
        layer_idx: usize,
        head_dim: usize,
        _eps: f32,
        sink: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            head_dim,
            outputs: sink,
        }
    }
}

impl BlockStage for GemmaKvTapStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let lp = &self.layer_prefix;
        let cos = ctx
            .state
            .rope_cos
            .ok_or_else(|| anyhow::anyhow!("GemmaKvTap requires RopeTables"))?;
        let sin = ctx
            .state
            .rope_sin
            .ok_or_else(|| anyhow::anyhow!("GemmaKvTap requires RopeTables"))?;

        let k_w = ctx.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
        let v_w = ctx.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?;

        let mut gb = HirMut::new(ctx.hir());
        let k = gb.mm(input.id, k_w);
        let v = gb.mm(input.id, v_w);
        let k_rope = gb.rope(k, cos, sin, self.head_dim);

        self.outputs.lock().expect("kv tap sink").push(k_rope);
        self.outputs.lock().expect("kv tap sink").push(v);
        Ok(Some(input))
    }
}
