// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Extract CLS token `[batch, 1, hidden]` → `[batch, hidden]`.
#[derive(Debug, Clone)]
pub struct ClsTokenPoolStage {
    pub batch: usize,
    pub hidden: usize,
}

impl ClsTokenPoolStage {
    pub fn new(batch: usize, hidden: usize) -> Self {
        Self { batch, hidden }
    }
}

impl BlockStage for ClsTokenPoolStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let mut gb = HirMut::new(ctx.hir());
        let cls = gb.narrow_(input.id, 1, 0, 1);
        let flat = gb.reshape_(cls, vec![self.batch as i64, self.hidden as i64]);
        Ok(Some(ctx.wrap(
            flat,
            rlx_ir::Shape::new(&[self.batch, self.hidden], DType::F32),
        )))
    }
}
