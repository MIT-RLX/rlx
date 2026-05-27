// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::{DType, Shape};

use crate::context::FlowCtx;
#[derive(Debug, Clone)]
pub struct RopeTablesStage {
    pub cos_key: String,
    pub sin_key: String,
    pub max_positions: usize,
    pub half_dim: usize,
    pub cos_data: Vec<f32>,
    pub sin_data: Vec<f32>,
}

impl RopeTablesStage {
    pub fn param(
        max_positions: usize,
        half_dim: usize,
        cos_data: Vec<f32>,
        sin_data: Vec<f32>,
    ) -> Self {
        Self {
            cos_key: "rope.cos".into(),
            sin_key: "rope.sin".into(),
            max_positions,
            half_dim,
            cos_data,
            sin_data,
        }
    }

    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let f = DType::F32;
        let cos_shape = Shape::new(&[self.max_positions, self.half_dim], f);
        let sin_shape = Shape::new(&[self.max_positions, self.half_dim], f);
        let cos_id = ctx.synth_param(&self.cos_key, self.cos_data.clone(), cos_shape);
        let sin_id = ctx.synth_param(&self.sin_key, self.sin_data.clone(), sin_shape);
        ctx.state.rope_cos = Some(cos_id);
        ctx.state.rope_sin = Some(sin_id);
        Ok(())
    }
}
