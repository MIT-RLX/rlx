// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Gated DeltaNet scan — generic op wrapper (Qwen3.5 trunk, …).

use anyhow::Result;
use rlx_ir::hir::HirMut;
use rlx_ir::HirGraphExt;
use rlx_ir::Shape;

use super::BlockStage;
use crate::context::FlowCtx;
use crate::value::FlowValue;

/// Q/K/V/G/Beta tensors must already be shaped `[batch, seq, heads, state]`.
#[derive(Debug, Clone)]
pub struct GdnScanStage {
    pub state_size: usize,
    pub out_shape: Shape,
    pub carry_state: bool,
    pub state_key: Option<String>,
}

impl GdnScanStage {
    pub fn prefill(state_size: usize, out_shape: Shape) -> Self {
        Self {
            state_size,
            out_shape,
            carry_state: false,
            state_key: None,
        }
    }

    pub fn with_carry(mut self, state_key: impl Into<String>) -> Self {
        self.carry_state = true;
        self.state_key = Some(state_key.into());
        self
    }
}

impl BlockStage for GdnScanStage {
    fn emit(
        &self,
        ctx: &mut FlowCtx<'_>,
        input: FlowValue,
    ) -> Result<Option<FlowValue>> {
        let slots = ctx
            .state
            .gdn
            .clone()
            .ok_or_else(|| anyhow::anyhow!("GdnScan requires gdn inputs in FlowState"))?;
        let carry_state = if self.carry_state {
            let key = self
                .state_key
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("GdnScan carry requires state_key"))?;
            Some(
                *ctx.state
                    .named
                    .get(key)
                    .ok_or_else(|| anyhow::anyhow!("GdnScan missing carry state `{key}`"))?,
            )
        } else {
            None
        };
        let mut gb = HirMut::new(ctx.hir());
        let id = if let Some(state) = carry_state {
            gb.gated_delta_net_carry(
                slots.q,
                slots.k,
                slots.v,
                slots.g,
                slots.beta,
                state,
                self.state_size,
                self.out_shape.clone(),
            )
        } else {
            gb.gated_delta_net(
                slots.q,
                slots.k,
                slots.v,
                slots.g,
                slots.beta,
                self.state_size,
                self.out_shape.clone(),
            )
        };
        let _ = input;
        Ok(Some(ctx.wrap(id, self.out_shape.clone())))
    }
}
