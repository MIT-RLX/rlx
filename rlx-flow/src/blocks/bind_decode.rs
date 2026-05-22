// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Bind decode graph inputs (RoPE slice, past K/V, optional mask).

use anyhow::Result;
use rlx_ir::hir::{HirModule, HirNodeId, HirOp};

use crate::context::{DecodeBindings, FlowCtx};
use crate::weight::WeightSource;

#[derive(Debug, Clone)]
pub struct BindDecodeInputsStage {
    pub num_layers: usize,
    pub use_custom_mask: bool,
}

impl BindDecodeInputsStage {
    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let cos = find_input(ctx.hir(), "rope_cos")?;
        let sin = find_input(ctx.hir(), "rope_sin")?;
        let mask = if self.use_custom_mask {
            Some(find_input(ctx.hir(), "mask")?)
        } else {
            None
        };
        let mut past_k = Vec::with_capacity(self.num_layers);
        let mut past_v = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            past_k.push(find_input(ctx.hir(), &format!("past_k_{i}"))?);
            past_v.push(find_input(ctx.hir(), &format!("past_v_{i}"))?);
        }
        ctx.state.decode = Some(DecodeBindings {
            cos,
            sin,
            mask,
            past_k,
            past_v,
        });
        Ok(())
    }
}

fn find_input(hir: &HirModule, name: &str) -> Result<HirNodeId> {
    for node in hir.nodes() {
        if let HirOp::Input { name: n } = &node.op {
            if n == name {
                return Ok(node.id);
            }
        }
    }
    Err(anyhow::anyhow!("decode flow missing input: {name}"))
}
