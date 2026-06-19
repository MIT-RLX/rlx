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

//! Gather one RoPE cos/sin row from prefill-style param tables for decode.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::{HirModule, HirMut, HirNodeId, HirOp};

use crate::context::FlowCtx;

#[derive(Debug, Clone)]
pub struct GatherDecodeRopeStage {
    pub position_input: String,
    pub half_dim: usize,
}

impl GatherDecodeRopeStage {
    pub fn new(half_dim: usize) -> Self {
        Self {
            position_input: "position".into(),
            half_dim,
        }
    }

    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let cos_table = ctx.state.rope_cos.ok_or_else(|| {
            anyhow::anyhow!("GatherDecodeRope requires RopeTablesStage before bind")
        })?;
        let sin_table = ctx.state.rope_sin.ok_or_else(|| {
            anyhow::anyhow!("GatherDecodeRope requires RopeTablesStage before bind")
        })?;
        let position = find_input(ctx.hir(), &self.position_input)?;

        let mut gb = HirMut::new(ctx.hir());
        let idx = gb.reshape_(position, vec![1, 1]);
        let cos_row = gb.gather_(cos_table, idx, 0);
        let sin_row = gb.gather_(sin_table, idx, 0);
        let cos_row = gb.reshape_(cos_row, vec![1, self.half_dim as i64]);
        let sin_row = gb.reshape_(sin_row, vec![1, self.half_dim as i64]);

        ctx.state.rope_cos = Some(cos_row);
        ctx.state.rope_sin = Some(sin_row);
        Ok(())
    }
}

fn find_input(hir: &HirModule, name: &str) -> Result<HirNodeId> {
    for node in hir.nodes() {
        if let HirOp::Input { name: n } = &node.op {
            if n == name {
                return Ok(node.id);
            }
        }
    }
    Err(anyhow::anyhow!("decode flow missing input: {name}"))
}
