// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::HirGraphExt;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;
use crate::weight::WeightSource;

#[derive(Debug, Clone)]
pub struct EmbedStage {
    pub weight_key: String,
    pub axis: usize,
}

impl EmbedStage {
    pub fn token(weight_key: impl Into<String>) -> Self {
        Self {
            weight_key: weight_key.into(),
            axis: 0,
        }
    }
}

impl BlockStage for EmbedStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let embed_w = ctx.load_param(&self.weight_key, false)?;
        ctx.state.embed_weight = Some(embed_w);
        let out_shape = {
            let w_shape = ctx.hir().node(embed_w).shape.clone();
            let mut dims: Vec<rlx_ir::Dim> = input.shape.dims().to_vec();
            dims.push(w_shape.dim(1));
            rlx_ir::Shape::from_dims(&dims, input.shape.dtype())
        };
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.gather_(embed_w, input.id, self.axis);
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
