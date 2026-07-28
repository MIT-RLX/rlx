// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

    /// Bake per-token MRoPE (Qwen-VL / Qwen3.5) cos/sin tables from explicit
    /// 3-D positions and publish them like any other RoPE table. `positions[t]
    /// = [pt, ph, pw, pe]` (len = seq, or batch·seq for per-batch-distinct
    /// positions). Feeds the existing per-token [`rlx_ir::op::Op::Rope`] path —
    /// no MRoPE-specific op is needed. See [`crate::rope::build_mrope_tables`].
    #[allow(clippy::too_many_arguments)]
    pub fn mrope(
        rope_theta: f64,
        head_dim: usize,
        n_rot: usize,
        sections: [usize; 4],
        positions: &[[usize; 4]],
        interleaved: bool,
        named_slot: Option<String>,
    ) -> Self {
        let (cos_data, sin_data) = crate::rope::build_mrope_tables(
            rope_theta,
            head_dim,
            n_rot,
            sections,
            positions,
            interleaved,
        );
        let max_positions = positions.len();
        let half_dim = head_dim / 2;
        match named_slot {
            Some(slot) => Self::param_named(slot, max_positions, half_dim, cos_data, sin_data),
            None => Self::param(max_positions, half_dim, cos_data, sin_data),
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
