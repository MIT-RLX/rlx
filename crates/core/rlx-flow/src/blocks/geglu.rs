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

/// Gemma GeGLU FFN: `down( gelu_tanh(gate(x)) * up(x) )`.
#[derive(Debug, Clone)]
pub struct GeGluStage {
    pub gate_key: String,
    pub up_key: String,
    pub down_key: String,
}

impl GeGluStage {
    pub fn hf_mlp(prefix: impl Into<String>) -> Self {
        let p = prefix.into();
        Self {
            gate_key: format!("{p}.mlp.gate_proj.weight"),
            up_key: format!("{p}.mlp.up_proj.weight"),
            down_key: format!("{p}.mlp.down_proj.weight"),
        }
    }
}

impl BlockStage for GeGluStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let gate_w = ctx.load_param(&self.gate_key, true)?;
        let up_w = ctx.load_param(&self.up_key, true)?;
        let down_w = ctx.load_param(&self.down_key, true)?;
        let mut gb = HirMut::new(ctx.hir());
        let gate = gb.mm(input.id, gate_w);
        let up = gb.mm(input.id, up_w);
        let gate_act = gb.gelu_approx(gate);
        let gated = gb.mul(gate_act, up);
        let id = gb.mm(gated, down_w);
        let out_shape = gb.shape(id).clone();
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
