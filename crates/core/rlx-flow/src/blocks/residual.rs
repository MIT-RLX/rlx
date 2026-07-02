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

/// Save tensor for a later [`ResidualAddStage`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidualSaveStage;

impl BlockStage for ResidualSaveStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        ctx.state.residual_skip = Some(input.id);
        ctx.state.residual_shape = Some(input.shape.clone());
        Ok(Some(input))
    }
}

/// Add the tensor saved by [`ResidualSaveStage`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ResidualAddStage;

impl BlockStage for ResidualAddStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let skip = ctx
            .state
            .residual_skip
            .ok_or_else(|| anyhow::anyhow!("ResidualAdd requires ResidualSave"))?;
        let shape = ctx
            .state
            .residual_shape
            .clone()
            .ok_or_else(|| anyhow::anyhow!("ResidualAdd missing saved shape"))?;
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.add(input.id, skip);
        ctx.state.residual_skip = None;
        ctx.state.residual_shape = None;
        Ok(Some(ctx.wrap(id, shape)))
    }
}
