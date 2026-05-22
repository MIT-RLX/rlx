// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::HirGraphExt;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct LinearStage {
    pub weight_key: String,
    pub transpose: bool,
}

impl LinearStage {
    pub fn new(weight_key: impl Into<String>, transpose: bool) -> Self {
        Self {
            weight_key: weight_key.into(),
            transpose,
        }
    }
}

impl BlockStage for LinearStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let w = ctx.load_param(&self.weight_key, self.transpose)?;
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.mm(input.id, w);
        let out_shape = gb.shape(id).clone();
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
