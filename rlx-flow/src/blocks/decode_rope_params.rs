// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

use anyhow::Result;
use rlx_ir::{DType, Shape};

use crate::context::FlowCtx;

/// Bake single-step decode RoPE cos/sin tables into flow state (static past length).
#[derive(Debug, Clone)]
pub struct DecodeRopeParamsStage {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half_dim: usize,
    /// Named slot — see [`crate::blocks::RopeTablesStage::param_named`].
    pub named_slot: Option<String>,
}

impl DecodeRopeParamsStage {
    /// Bake the default decode rope row (sets `state.rope_cos/sin`).
    pub fn new(cos: Vec<f32>, sin: Vec<f32>, half_dim: usize) -> Self {
        Self {
            cos,
            sin,
            half_dim,
            named_slot: None,
        }
    }

    /// Bake an extra decode rope row under a named slot.
    pub fn named(slot: impl Into<String>, cos: Vec<f32>, sin: Vec<f32>, half_dim: usize) -> Self {
        Self {
            cos,
            sin,
            half_dim,
            named_slot: Some(slot.into()),
        }
    }

    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let f = DType::F32;
        let shape = Shape::new(&[1, self.half_dim], f);
        let (cos_key, sin_key) = match &self.named_slot {
            Some(slot) => (
                format!("decode.rope.{slot}.cos"),
                format!("decode.rope.{slot}.sin"),
            ),
            None => ("decode.rope.cos".into(), "decode.rope.sin".into()),
        };
        let cos_id = ctx.synth_param(&cos_key, self.cos.clone(), shape.clone());
        let sin_id = ctx.synth_param(&sin_key, self.sin.clone(), shape);
        match &self.named_slot {
            Some(slot) => {
                ctx.state.named.insert(format!("{slot}_cos"), cos_id);
                ctx.state.named.insert(format!("{slot}_sin"), sin_id);
            }
            None => {
                ctx.state.rope_cos = Some(cos_id);
                ctx.state.rope_sin = Some(sin_id);
            }
        }
        Ok(())
    }
}
