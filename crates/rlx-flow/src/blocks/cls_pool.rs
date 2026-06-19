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
use rlx_ir::hir::HirMut;
use rlx_ir::{DType, HirGraphExt};

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Extract CLS token `[batch, 1, hidden]` → `[batch, hidden]`.
#[derive(Debug, Clone)]
pub struct ClsTokenPoolStage {
    pub batch: usize,
    pub hidden: usize,
}

impl ClsTokenPoolStage {
    pub fn new(batch: usize, hidden: usize) -> Self {
        Self { batch, hidden }
    }
}

impl BlockStage for ClsTokenPoolStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let mut gb = HirMut::new(ctx.hir());
        let cls = gb.narrow_(input.id, 1, 0, 1);
        let flat = gb.reshape_(cls, vec![self.batch as i64, self.hidden as i64]);
        Ok(Some(ctx.wrap(
            flat,
            rlx_ir::Shape::new(&[self.batch, self.hidden], DType::F32),
        )))
    }
}
