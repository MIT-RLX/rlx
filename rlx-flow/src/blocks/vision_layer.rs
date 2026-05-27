// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use std::sync::Arc;

use anyhow::Result;
use rlx_ir::HirGraphExt;
use rlx_ir::hir::HirMut;

use super::{BlockStage, VitSelfAttnStage};
use crate::context::FlowCtx;
use crate::layer::LayerStack;
use crate::stage::FlowStage;
use crate::value::FlowValue;

/// NomicVision SwiGLU FFN with intermediate LayerNorm.
#[derive(Debug, Clone)]
pub struct VisionSwiGluFfnStage {
    pub layer_prefix: String,
    pub eps: f32,
}

impl VisionSwiGluFfnStage {
    pub fn new(layer_prefix: impl Into<String>, eps: f32) -> Self {
        Self {
            layer_prefix: layer_prefix.into(),
            eps,
        }
    }
}

impl BlockStage for VisionSwiGluFfnStage {
    fn emit(&self, ctx: &mut FlowCtx<'_>, input: FlowValue) -> Result<Option<FlowValue>> {
        let lp = &self.layer_prefix;
        let fc11_w = ctx.load_param(&format!("{lp}.mlp.fc11.weight"), true)?;
        let fc11_b = ctx.load_param(&format!("{lp}.mlp.fc11.bias"), false)?;
        let fc12_w = ctx.load_param(&format!("{lp}.mlp.fc12.weight"), true)?;
        let fc12_b = ctx.load_param(&format!("{lp}.mlp.fc12.bias"), false)?;
        let fc2_w = ctx.load_param(&format!("{lp}.mlp.fc2.weight"), true)?;
        let fc2_b = ctx.load_param(&format!("{lp}.mlp.fc2.bias"), false)?;
        let mlp_ln_g = ctx.load_param(&format!("{lp}.mlp.norm.weight"), false)?;
        let mlp_ln_b = ctx.load_param(&format!("{lp}.mlp.norm.bias"), false)?;

        let mut gb = HirMut::new(ctx.hir());
        let up_mm = gb.mm(input.id, fc11_w);
        let up = gb.add(up_mm, fc11_b);
        let gate_mm = gb.mm(input.id, fc12_w);
        let gate_bias = gb.add(gate_mm, fc12_b);
        let gate = gb.silu(gate_bias);
        let swiglu = gb.mul(up, gate);
        let normed = gb.ln(swiglu, mlp_ln_g, mlp_ln_b, self.eps);
        let down_mm = gb.mm(normed, fc2_w);
        let out = gb.add(down_mm, fc2_b);
        Ok(Some(ctx.wrap(out, input.shape.clone())))
    }
}

/// Fused NomicVision encoder block.
pub fn nomic_vision_layer_fused(
    layer_idx: usize,
    hidden_size: usize,
    num_heads: usize,
    eps: f32,
) -> FlowStage {
    let lp = format!("layers.{layer_idx}");
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(
            LayerStack::named(lp.clone())
                .residual_save()
                .layer_norm(
                    format!("{lp}.norm1.weight"),
                    format!("{lp}.norm1.bias"),
                    eps,
                )
                .stage(FlowStage::VitSelfAttn(VitSelfAttnStage::nomic_vision(
                    &lp,
                    hidden_size,
                    num_heads,
                )))
                .residual_add()
                .residual_save()
                .layer_norm(
                    format!("{lp}.norm2.weight"),
                    format!("{lp}.norm2.bias"),
                    eps,
                )
                .stage(FlowStage::VisionSwiGluFfn(VisionSwiGluFfnStage::new(
                    &lp, eps,
                )))
                .residual_add()
                .build()
                .unwrap_sequence(),
        ),
    }
}

trait UnwrapSequence {
    fn unwrap_sequence(self) -> FlowStage;
}

impl UnwrapSequence for FlowStage {
    fn unwrap_sequence(self) -> FlowStage {
        match self {
            FlowStage::Named { inner, .. } => (*inner).clone(),
            other => other,
        }
    }
}
