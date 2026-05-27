// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Gather rows from an embedding table using a named graph input as indices.
#[derive(Debug, Clone)]
pub struct GatherFromInputStage {
    pub input_name: String,
    pub weight_key: String,
    pub axis: usize,
}

impl GatherFromInputStage {
    pub fn new(input_name: impl Into<String>, weight_key: impl Into<String>, axis: usize) -> Self {
        Self {
            input_name: input_name.into(),
            weight_key: weight_key.into(),
            axis,
        }
    }
}

impl BlockStage for GatherFromInputStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, _input: FlowValue) -> Result<Option<FlowValue>> {
        let (indices_id, indices_shape) = ctx
            .state
            .inputs
            .get(&self.input_name)
            .ok_or_else(|| anyhow::anyhow!("GatherFromInput missing input `{}`", self.input_name))?
            .clone();
        let embed_w = ctx.load_param(&self.weight_key, false)?;
        let w_shape = ctx.hir().node(embed_w).shape.clone();
        let mut dims: Vec<rlx_ir::Dim> = indices_shape.dims().to_vec();
        dims.push(w_shape.dim(1));
        let out_shape = rlx_ir::Shape::from_dims(&dims, indices_shape.dtype());

        let mut gb = HirMut::new(ctx.hir());
        let id = gb.gather_(embed_w, indices_id, self.axis);
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}

/// Add a gather-from-input embedding to the active hidden tensor.
#[derive(Debug, Clone)]
pub struct GatherAddStage {
    pub input_name: String,
    pub weight_key: String,
    pub axis: usize,
}

impl GatherAddStage {
    pub fn new(input_name: impl Into<String>, weight_key: impl Into<String>, axis: usize) -> Self {
        Self {
            input_name: input_name.into(),
            weight_key: weight_key.into(),
            axis,
        }
    }
}

impl BlockStage for GatherAddStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let (indices_id, indices_shape) = ctx
            .state
            .inputs
            .get(&self.input_name)
            .ok_or_else(|| anyhow::anyhow!("GatherAdd missing input `{}`", self.input_name))?
            .clone();
        let embed_w = ctx.load_param(&self.weight_key, false)?;
        let w_shape = ctx.hir().node(embed_w).shape.clone();
        let mut dims: Vec<rlx_ir::Dim> = indices_shape.dims().to_vec();
        dims.push(w_shape.dim(1));
        let out_shape = rlx_ir::Shape::from_dims(&dims, indices_shape.dtype());

        let mut gb = HirMut::new(ctx.hir());
        let gathered = gb.gather_(embed_w, indices_id, self.axis);
        let id = gb.add(input.id, gathered);
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
