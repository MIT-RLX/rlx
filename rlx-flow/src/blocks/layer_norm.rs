// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::HirGraphExt;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct LayerNormStage {
    pub gamma_key: String,
    pub beta_key: String,
    pub eps: f32,
}

impl LayerNormStage {
    pub fn new(
        gamma_key: impl Into<String>,
        beta_key: impl Into<String>,
        eps: f32,
    ) -> Self {
        Self {
            gamma_key: gamma_key.into(),
            beta_key: beta_key.into(),
            eps,
        }
    }

    pub fn hf(gamma_key: impl Into<String>, beta_key: impl Into<String>, eps: f32) -> Self {
        Self::new(gamma_key, beta_key, eps)
    }
}

impl BlockStage for LayerNormStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let gamma = ctx.load_param(&self.gamma_key, false)?;
        let beta = ctx.load_param(&self.beta_key, false)?;
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.ln(input.id, gamma, beta, self.eps);
        Ok(Some(ctx.wrap(id, input.shape.clone())))
    }
}
