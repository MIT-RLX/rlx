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
#[derive(Debug, Clone)]
pub struct RopeTablesStage {
    pub cos_key: String,
    pub sin_key: String,
    pub max_positions: usize,
    pub half_dim: usize,
    pub cos_data: Vec<f32>,
    pub sin_data: Vec<f32>,
    /// When `Some(slot)`, push the (cos, sin) HIR ids into
    /// `state.named["{slot}_cos"]` / `"{slot}_sin"` instead of the
    /// default `state.rope_cos`/`state.rope_sin`. Self-attention
    /// blocks opt into the named slot via
    /// `SelfAttnPrefillSpec::rope_table`. Used by Gemma 4 which
    /// ships split sliding/full RoPE thetas.
    pub named_slot: Option<String>,
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
            named_slot: None,
        }
    }

    /// Variant that publishes the tables under a named slot (for
    /// per-layer RoPE) rather than the default flow-state handles.
    pub fn param_named(
        slot: impl Into<String>,
        max_positions: usize,
        half_dim: usize,
        cos_data: Vec<f32>,
        sin_data: Vec<f32>,
    ) -> Self {
        let slot = slot.into();
        Self {
            cos_key: format!("rope.{slot}.cos"),
            sin_key: format!("rope.{slot}.sin"),
            max_positions,
            half_dim,
            cos_data,
            sin_data,
            named_slot: Some(slot),
        }
    }

    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let f = DType::F32;
        let cos_shape = Shape::new(&[self.max_positions, self.half_dim], f);
        let sin_shape = Shape::new(&[self.max_positions, self.half_dim], f);
        let cos_id = ctx.synth_param(&self.cos_key, self.cos_data.clone(), cos_shape);
        let sin_id = ctx.synth_param(&self.sin_key, self.sin_data.clone(), sin_shape);
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
