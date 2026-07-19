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

//! DiT modulation stages — adaLN-Zero + gated residual.

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;
use rlx_ir::op::AdaNormKind;

use super::{BlockStage, LinearStage};
use crate::context::FlowCtx;
use crate::stage::FlowStage;
use crate::value::FlowValue;

/// Apply [`Op::AdaLayerNorm`] using named modulation inputs already bound in
/// [`FlowCtx`] state (`scale_input` / `shift_input`).
#[derive(Debug, Clone)]
pub struct AdaLayerNormStage {
    pub scale_input: String,
    pub shift_input: String,
    pub norm: AdaNormKind,
    pub eps: f32,
}

impl AdaLayerNormStage {
    pub fn new(
        scale_input: impl Into<String>,
        shift_input: impl Into<String>,
        norm: AdaNormKind,
        eps: f32,
    ) -> Self {
        Self {
            scale_input: scale_input.into(),
            shift_input: shift_input.into(),
            norm,
            eps,
        }
    }
}

impl BlockStage for AdaLayerNormStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let (scale, _) = ctx
            .state
            .inputs
            .get(&self.scale_input)
            .ok_or_else(|| anyhow::anyhow!("AdaLayerNorm missing input `{}`", self.scale_input))?
            .clone();
        let (shift, _) = ctx
            .state
            .inputs
            .get(&self.shift_input)
            .ok_or_else(|| anyhow::anyhow!("AdaLayerNorm missing input `{}`", self.shift_input))?
            .clone();
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.ada_layer_norm(input.id, scale, shift, self.norm, self.eps);
        Ok(Some(ctx.wrap(id, input.shape.clone())))
    }
}

/// Apply [`Op::GatedResidual`]: `residual + gate · y` where `y` is the active
/// tensor and `residual` was saved via [`super::ResidualSaveStage`].
#[derive(Debug, Clone)]
pub struct GatedResidualStage {
    pub gate_input: String,
}

impl GatedResidualStage {
    pub fn new(gate_input: impl Into<String>) -> Self {
        Self {
            gate_input: gate_input.into(),
        }
    }
}

impl BlockStage for GatedResidualStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let skip = ctx
            .state
            .residual_skip
            .ok_or_else(|| anyhow::anyhow!("GatedResidual requires ResidualSave"))?;
        let shape = ctx
            .state
            .residual_shape
            .clone()
            .ok_or_else(|| anyhow::anyhow!("GatedResidual missing saved shape"))?;
        let (gate, _) = ctx
            .state
            .inputs
            .get(&self.gate_input)
            .ok_or_else(|| anyhow::anyhow!("GatedResidual missing input `{}`", self.gate_input))?
            .clone();
        let mut gb = HirMut::new(ctx.hir());
        let id = gb.gated_residual(skip, input.id, gate);
        ctx.state.residual_skip = None;
        ctx.state.residual_shape = None;
        Ok(Some(ctx.wrap(id, shape)))
    }
}

/// One DiT-style sublayer: residual-save → adaLN → linear → gated residual.
///
/// Modulation tensors must already be bound as graph inputs named
/// `{prefix}.scale`, `{prefix}.shift`, `{prefix}.gate`.
pub fn dit_ada_gated_linear(
    prefix: impl Into<String>,
    weight_key: impl Into<String>,
    norm: AdaNormKind,
    eps: f32,
) -> FlowStage {
    let prefix = prefix.into();
    let scale = format!("{prefix}.scale");
    let shift = format!("{prefix}.shift");
    let gate = format!("{prefix}.gate");
    FlowStage::Sequence(vec![
        FlowStage::ResidualSave(super::ResidualSaveStage),
        FlowStage::AdaLayerNorm(AdaLayerNormStage::new(scale, shift, norm, eps)),
        FlowStage::Linear(LinearStage::new(weight_key, true)),
        FlowStage::GatedResidual(GatedResidualStage::new(gate)),
    ])
}
