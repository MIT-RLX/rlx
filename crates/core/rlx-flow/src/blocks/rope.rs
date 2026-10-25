// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use anyhow::Result;
use rlx_ir::{DType, Shape};

use crate::context::{FlowCtx, f64_format};
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
    /// The tables in f64, from [`Self::with_f64`] when this process computes
    /// in f64; emitted through [`FlowCtx::synth_param_f64`] so they stay exact
    /// once the graph is promoted.
    pub f64_data: Option<(Vec<f64>, Vec<f64>)>,
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
            f64_data: None,
        }
    }

    /// `[max_positions, inv_freq.len()]` tables, `cos(pos · inv_freq[i])`,
    /// carrying their f64 residual (see [`Self::with_f64`]).
    pub fn from_inv_freq(max_positions: usize, inv_freq: &[f64]) -> Self {
        let (cos, sin): (Vec<f64>, Vec<f64>) = (0..max_positions)
            .flat_map(|p| inv_freq.iter().map(move |&f| (p as f64 * f).sin_cos()))
            .map(|(s, c)| (c, s))
            .unzip();
        Self::param(max_positions, inv_freq.len(), Vec::new(), Vec::new()).with_f64(&cos, &sin)
    }

    /// Publish under a named slot, as [`Self::param_named`] does.
    pub fn named(mut self, slot: impl Into<String>) -> Self {
        let slot = slot.into();
        self.cos_key = format!("rope.{slot}.cos");
        self.sin_key = format!("rope.{slot}.sin");
        self.named_slot = Some(slot);
        self
    }

    /// Tables from f64 values: f32 as usual and, when this process computes
    /// in f64, the f64 values too (see [`Self::f64_data`]).
    pub fn with_f64(mut self, cos: &[f64], sin: &[f64]) -> Self {
        self.cos_data = cos.iter().map(|&x| x as f32).collect();
        self.sin_data = sin.iter().map(|&x| x as f32).collect();
        self.f64_data = f64_format().then(|| (cos.to_vec(), sin.to_vec()));
        self
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
        // Table row stride now matches `build_mrope_tables` (n_rot/2), which the
        // RoPE kernel indexes by. Equals head_dim/2 for full rope.
        let half_dim = n_rot / 2;
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
            f64_data: None,
        }
    }

    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let f = DType::F32;
        let cos_shape = Shape::new(&[self.max_positions, self.half_dim], f);
        let sin_shape = Shape::new(&[self.max_positions, self.half_dim], f);
        let (cos_id, sin_id) = match &self.f64_data {
            Some((cos, sin)) => (
                ctx.synth_param_f64(&self.cos_key, cos, cos_shape),
                ctx.synth_param_f64(&self.sin_key, sin, sin_shape),
            ),
            None => (
                ctx.synth_param(&self.cos_key, self.cos_data.clone(), cos_shape),
                ctx.synth_param(&self.sin_key, self.sin_data.clone(), sin_shape),
            ),
        };
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
