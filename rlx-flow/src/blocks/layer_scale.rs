// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct LayerScaleStage {
    pub gamma_key: String,
}

impl LayerScaleStage {
    pub fn new(gamma_key: impl Into<String>) -> Self {
        Self {
            gamma_key: gamma_key.into(),
        }
    }
}

impl BlockStage for LayerScaleStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let gamma = ctx.load_param(&self.gamma_key, false)?;
        let mut gb = HirMut::new(ctx.hir());
        let out = gb.mul(input.id, gamma);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
