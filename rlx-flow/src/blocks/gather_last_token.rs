// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, Shape};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;
#[derive(Debug, Clone)]
pub struct GatherLastTokenStage {
    pub batch: usize,
    pub input_name: String,
}

impl GatherLastTokenStage {
    pub fn dynamic(batch: usize) -> Self {
        Self {
            batch,
            input_name: "last_token_idx".into(),
        }
    }

    pub fn static_last(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            input_name: format!("__static_last_{seq}"),
        }
    }
}

impl BlockStage for GatherLastTokenStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let id = if self.input_name.starts_with("__static_last_") {
            let seq: usize = self
                .input_name
                .strip_prefix("__static_last_")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| anyhow::anyhow!("invalid static last token stage"))?;
            let mut gb = HirMut::new(ctx.hir());
            gb.narrow_(input.id, 1, seq - 1, 1)
        } else {
            let idx = ctx.input(&self.input_name, Shape::new(&[self.batch], DType::F32));
            let mut gb = HirMut::new(ctx.hir());
            let idx_2d = gb.reshape_(idx, vec![self.batch as i64, 1]);
            gb.gather_(input.id, idx_2d, 1)
        };
        let out_shape = if input.shape.rank() >= 2 {
            let batch = input.shape.dim(0).unwrap_static();
            let hidden = input.shape.dim(2).unwrap_static();
            Shape::new(&[batch, 1, hidden], input.shape.dtype())
        } else {
            input.shape.clone()
        };
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
