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

//! Bind or synthesize a vision-style attention mask (all-ones = no padding).

use anyhow::Result;
use rlx_ir::hir::{HirModule, HirOp};
use rlx_ir::{DType, Shape};

use crate::context::FlowCtx;

pub const ATTN_MASK: &str = "attn_mask";

#[derive(Debug, Clone)]
pub struct AttnMaskStage {
    pub batch: usize,
    pub seq: usize,
    /// When set, reuse a flow input instead of synthesizing all-ones.
    pub input_name: Option<String>,
}

impl AttnMaskStage {
    pub fn ones(batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            input_name: None,
        }
    }

    pub fn from_input(name: impl Into<String>, batch: usize, seq: usize) -> Self {
        Self {
            batch,
            seq,
            input_name: Some(name.into()),
        }
    }

    pub fn emit(&self, ctx: &mut FlowCtx<'_>) -> Result<()> {
        let id = if let Some(name) = &self.input_name {
            find_input(ctx.hir(), name)?
        } else {
            let data = vec![1.0f32; self.batch * self.seq];
            ctx.synth_param(
                ATTN_MASK,
                data,
                Shape::new(&[self.batch, self.seq], DType::F32),
            )
        };
        ctx.state.named.insert(ATTN_MASK.to_string(), id);
        Ok(())
    }
}

fn find_input(hir: &HirModule, name: &str) -> Result<rlx_ir::HirNodeId> {
    for node in hir.nodes() {
        if let HirOp::Input { name: n } = &node.op {
            if n == name {
                return Ok(node.id);
            }
        }
    }
    Err(anyhow::anyhow!("attn mask flow missing input: {name}"))
}
