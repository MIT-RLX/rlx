// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;

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
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        // Delegate to `FlowCtx::linear` so packed (GGUF/MLX) weights lower to a
        // fused `DequantMatMul` and F32 weights to a plain matmul — one code
        // path for every model that uses this stage.
        Ok(Some(ctx.linear(
            &input,
            &self.weight_key,
            self.transpose,
        )?))
    }
}
