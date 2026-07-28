// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
    /// When `Some(hidden)`, do NOT load the (potentially multi-GiB) F32
    /// embedding table or emit a `gather` — instead declare an
    /// `inputs_embeds` `[..input dims.., hidden]` F32 input that the caller
    /// fills with host-gathered rows. A large-vocab model (e.g. Bonsai-27B's
    /// `token_embd` is `[248320, 5120]` F32 = 4.7 GiB) otherwise keeps the
    /// whole table resident on the device just to look up the prompt's few
    /// rows; gathering host-side keeps it off the accelerator entirely.
    pub host_hidden: Option<usize>,
}

impl EmbedStage {
    pub fn token(weight_key: impl Into<String>) -> Self {
        Self {
            weight_key: weight_key.into(),
            axis: 0,
            host_hidden: None,
        }
    }

    /// Host-gathered variant — see [`Self::host_hidden`].
    pub fn token_host(weight_key: impl Into<String>, hidden: usize) -> Self {
        Self {
            weight_key: weight_key.into(),
            axis: 0,
            host_hidden: Some(hidden),
        }
    }
}

impl BlockStage for EmbedStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        if let Some(hidden) = self.host_hidden {
            // `inputs_embeds` [..input dims.., hidden] fed by the caller; the
            // embedding table never touches the device.
            let mut dims: Vec<rlx_ir::Dim> = input.shape.dims().to_vec();
            dims.push(rlx_ir::Dim::Static(hidden));
            let out_shape = rlx_ir::Shape::from_dims(&dims, rlx_ir::DType::F32);
            let id = ctx.input("inputs_embeds", out_shape.clone());
            return Ok(Some(ctx.wrap(id, out_shape)));
        }
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
