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
use rlx_ir::op::Activation;
use rlx_ir::{DType, Shape};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Gemma 2 final logit softcap: `cap * tanh(logits / cap)`.
#[derive(Debug, Clone)]
pub struct LogitSoftcapStage {
    pub cap: f32,
}

impl LogitSoftcapStage {
    pub fn new(cap: f32) -> Self {
        Self { cap }
    }
}

impl BlockStage for LogitSoftcapStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let cap = self.cap;
        let inv_name = format!("gemma.logit_softcap.inv.{cap}");
        let inv = ctx.synth_param(&inv_name, vec![1.0 / cap], Shape::new(&[1], DType::F32));
        let cap_name = format!("gemma.logit_softcap.cap.{cap}");
        let cap_id = ctx.synth_param(&cap_name, vec![cap], Shape::new(&[1], DType::F32));
        let mut gb = HirMut::new(ctx.hir());
        let scaled = gb.mul(input.id, inv);
        let t = gb.activation(Activation::Tanh, scaled, gb.shape(scaled).clone());
        let out = gb.mul(t, cap_id);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
