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
use rlx_ir::{DType, Shape};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Gemma RMSNorm: `x / rms(x) * (1 + weight)` (HF layout).
#[derive(Debug, Clone)]
pub struct GemmaRmsNormStage {
    pub weight_key: String,
    pub eps: f32,
}

impl GemmaRmsNormStage {
    pub fn hf_layer(prefix: impl Into<String>, eps: f32) -> Self {
        let p = prefix.into();
        Self {
            weight_key: format!("{p}.weight"),
            eps,
        }
    }
}

impl BlockStage for GemmaRmsNormStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let w = ctx.load_param(&self.weight_key, false)?;
        let zero_beta = ctx
            .state
            .zero_beta
            .ok_or_else(|| anyhow::anyhow!("GemmaRmsNorm requires ZeroBeta stage"))?;
        let w_shape = ctx.node_shape(w)?;
        let hidden = norm_weight_len(&w_shape);
        let ones_name = format!("{}.ones", self.weight_key);
        let ones = ctx.synth_param(
            &ones_name,
            vec![1.0f32; hidden],
            Shape::new(&[hidden], DType::F32),
        );
        let mut gb = HirMut::new(ctx.hir());
        let one_plus_w = gb.add(ones, w);
        let normed = gb.rms_norm(input.id, one_plus_w, zero_beta, self.eps);
        Ok(Some(ctx.wrap(normed, input.shape.clone())))
    }
}

fn norm_weight_len(shape: &rlx_ir::Shape) -> usize {
    match shape.dims().last() {
        Some(rlx_ir::shape::Dim::Static(n)) => *n,
        _ => 0,
    }
}
