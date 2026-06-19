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

//! Block interface traits — generics with associated types (Slang interface bounds).

use anyhow::Result;
use rlx_ir::Shape;

use crate::context::FlowCtx;
use crate::stage_contract::{LayerStage, StageArtifacts};
use crate::value::FlowValue;

/// KV cache tensor shapes exposed by attention blocks (associated type stand-in).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheContract {
    pub k: Shape,
    pub v: Shape,
}

/// Attention block interface: hidden in, hidden out, plus cache contract.
pub trait AttentionStage: LayerStage {
    fn cache_contract(&self, ctx: &FlowCtx<'_>, hidden: &Shape) -> KvCacheContract;

    fn emit_attention(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<(FlowValue, StageArtifacts, KvCacheContract)> {
        let contract = self.cache_contract(ctx, &input.shape);
        let (value, artifacts) = self.emit_layer(ctx, input)?;
        Ok((value, artifacts, contract))
    }
}

/// FFN block interface (SwiGLU / MLP).
pub trait FfnStage: LayerStage {
    /// Intermediate projection width (associated type as shape).
    fn intermediate_shape(&self, ctx: &FlowCtx<'_>, hidden: &Shape) -> Shape;
}

/// Normalization block interface.
pub trait NormStage: LayerStage {
    fn eps(&self) -> f32;
}
