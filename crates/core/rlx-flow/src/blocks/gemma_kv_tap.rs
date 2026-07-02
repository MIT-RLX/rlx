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
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use std::sync::{Arc, Mutex};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Export RoPE(K) and V after Gemma input RMSNorm (prefill KV-cache fill).
///
/// Configuration mirrors [`crate::blocks::SelfAttnPrefillSpec`] for the
/// fields that affect K/V shape and rotation:
/// - `head_dim` — per-head width (per layer; Gemma 4 12B uses 256 for
///   sliding layers, 512 for full-attention layers).
/// - `n_rot` — number of leading per-head dims that get rotated
///   (partial rotary; `head_dim` for plain RoPE).
/// - `rope_table` — optional named slot in `state.named` (the
///   `"global"` slot when Gemma 4 full-attention layers ship distinct
///   `rope_theta` / `partial_rotary_factor`).
/// - `k_eq_v` — reuse the K projection as V (skip the V matmul +
///   `v_proj.weight` load).
#[derive(Debug, Clone)]
pub struct GemmaKvTapStage {
    pub layer_prefix: String,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_table: Option<String>,
    pub k_eq_v: bool,
    pub outputs: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
}

impl GemmaKvTapStage {
    /// Backwards-compatible default: full rotation, default RoPE
    /// table, no K=V aliasing. Matches Gemma 1 / 2 / 3.
    pub fn layer(
        layer_idx: usize,
        head_dim: usize,
        _eps: f32,
        sink: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    ) -> Self {
        Self {
            layer_prefix: format!("model.layers.{layer_idx}"),
            head_dim,
            n_rot: head_dim,
            rope_table: None,
            k_eq_v: false,
            outputs: sink,
        }
    }

    /// Set partial rotary (Gemma 4 full-attention p-RoPE).
    pub fn with_n_rot(mut self, n_rot: usize) -> Self {
        self.n_rot = n_rot;
        self
    }

    /// Bind to a named secondary rope table (`"global"`).
    pub fn with_rope_table(mut self, name: impl Into<String>) -> Self {
        self.rope_table = Some(name.into());
        self
    }

    /// Skip the V projection and alias V to the K projection output
    /// (pre-RoPE). Matches the SelfAttn emit's `k_eq_v` semantics so
    /// the cache holds the same V the attention block computes.
    pub fn with_k_eq_v(mut self) -> Self {
        self.k_eq_v = true;
        self
    }
}

impl BlockStage for GemmaKvTapStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let lp = &self.layer_prefix;
        let (cos, sin) = super::self_attn::resolve_rope_handles(ctx, self.rope_table.as_deref())?;

        let k_w = ctx.load_param(&format!("{lp}.self_attn.k_proj.weight"), true)?;
        let v_w = if self.k_eq_v {
            None
        } else {
            Some(ctx.load_param(&format!("{lp}.self_attn.v_proj.weight"), true)?)
        };

        let mut gb = HirMut::new(ctx.hir());
        let k = gb.mm(input.id, k_w);
        // V = K (pre-RoPE) when k_eq_v; otherwise its own matmul.
        let v = match v_w {
            Some(w) => gb.mm(input.id, w),
            None => k,
        };
        let k_rope = gb.rope_n(k, cos, sin, self.head_dim, self.n_rot);

        self.outputs.lock().expect("kv tap sink").push(k_rope);
        self.outputs.lock().expect("kv tap sink").push(v);
        Ok(Some(input))
    }
}
