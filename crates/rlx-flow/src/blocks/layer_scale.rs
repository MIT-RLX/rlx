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
pub struct LayerScaleStage {
    pub gamma_key: String,
}

impl LayerScaleStage {
    pub fn new(gamma_key: impl Into<String>) -> Self {
        Self {
            gamma_key: gamma_key.into(),
        }
    }
}

impl BlockStage for LayerScaleStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let gamma = ctx.load_param(&self.gamma_key, false)?;
        let mut gb = HirMut::new(ctx.hir());
        let out = gb.mul(input.id, gamma);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}
