// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, Shape};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Multiply token embeddings by `sqrt(hidden_size)` (Gemma / Gemma 2).
#[derive(Debug, Clone)]
pub struct EmbedScaleStage {
    pub hidden_size: usize,
}

impl EmbedScaleStage {
    pub fn new(hidden_size: usize) -> Self {
        Self { hidden_size }
    }
}

impl BlockStage for EmbedScaleStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let scale = (self.hidden_size as f32).sqrt();
        let name = format!("gemma.embed_scale.{}", self.hidden_size);
        let scale_id = ctx.synth_param(&name, vec![scale], Shape::new(&[1], DType::F32));
        let mut gb = HirMut::new(ctx.hir());
        let out = gb.mul(input.id, scale_id);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
