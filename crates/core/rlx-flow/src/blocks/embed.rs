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
pub struct EmbedStage {
    pub weight_key: String,
    pub axis: usize,
}

impl EmbedStage {
    pub fn token(weight_key: impl Into<String>) -> Self {
        Self {
            weight_key: weight_key.into(),
            axis: 0,
        }
    }
}

impl BlockStage for EmbedStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let embed_w = ctx.load_param(&self.weight_key, false)?;
        ctx.state.embed_weight = Some(embed_w);
        let out_shape = {
            let w_shape = ctx.hir().node(embed_w).shape.clone();
            let mut dims: Vec<rlx_ir::Dim> = input.shape.dims().to_vec();
            dims.push(w_shape.dim(1));
            rlx_ir::Shape::from_dims(&dims, input.shape.dtype())
        };
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.gather_(embed_w, input.id, self.axis);
        Ok(Some(ctx.wrap(id, out_shape)))
    }
}
