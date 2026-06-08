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

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

#[derive(Debug, Clone)]
pub struct SwiGluStage {
    pub gate_key: String,
    pub up_key: String,
    pub down_key: String,
}

impl SwiGluStage {
    pub fn new(
        gate_key: impl Into<String>,
        up_key: impl Into<String>,
        down_key: impl Into<String>,
    ) -> Self {
        Self {
            gate_key: gate_key.into(),
            up_key: up_key.into(),
            down_key: down_key.into(),
        }
    }

    pub fn hf_mlp(prefix: impl Into<String>) -> Self {
        let p = prefix.into();
        Self::new(
            format!("{p}.gate_proj.weight"),
            format!("{p}.up_proj.weight"),
            format!("{p}.down_proj.weight"),
        )
    }
}

impl BlockStage for SwiGluStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let gate_w = ctx.load_param(&self.gate_key, true)?;
        let up_w = ctx.load_param(&self.up_key, true)?;
        let down_w = ctx.load_param(&self.down_key, true)?;
        let mut gb = HirMut::new(ctx.hir());
        let gate = gb.mm(input.id, gate_w);
        let up = gb.mm(input.id, up_w);
        let gate_act = gb.silu(gate);
        let swiglu = gb.mul(gate_act, up);
        let id = gb.mm(swiglu, down_w);
        let out_shape = gb.shape(id).clone();
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
