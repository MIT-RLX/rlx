// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use anyhow::Result;
use rlx_ir::{DType, Shape};

use crate::context::FlowCtx;

/// Bake single-step decode RoPE cos/sin tables into flow state (static past length).
#[derive(Debug, Clone)]
pub struct DecodeRopeParamsStage {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half_dim: usize,
}

impl DecodeRopeParamsStage {
    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let f = DType::F32;
        let shape = Shape::new(&[1, self.half_dim], f);
        let cos_id = ctx.synth_param("decode.rope.cos", self.cos.clone(), shape.clone());
        let sin_id = ctx.synth_param("decode.rope.sin", self.sin.clone(), shape);
        ctx.state.rope_cos = Some(cos_id);
        ctx.state.rope_sin = Some(sin_id);
        Ok(())
    }
}
