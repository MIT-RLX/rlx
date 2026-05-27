// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;
#[derive(Debug, Clone)]
pub struct RmsNormStage {
    pub weight_key: String,
    pub eps: f32,
}

impl RmsNormStage {
    pub fn new(weight_key: impl Into<String>, eps: f32) -> Self {
        Self {
            weight_key: weight_key.into(),
            eps,
        }
    }
}

impl BlockStage for RmsNormStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let gamma = ctx.load_param(&self.weight_key, false)?;
        let zero_beta = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("RmsNorm requires ZeroBeta stage"))?;
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.rms_norm(input.id, gamma, zero_beta, self.eps);
        Ok(Some(ctx.wrap(id, input.shape.clone())))
    }
}
