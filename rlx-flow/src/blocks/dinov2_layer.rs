// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use std::sync::Arc;

use super::{GeluFfnStage, LayerScaleStage, VitSelfAttnStage};
use crate::layer::LayerStack;
use crate::stage::FlowStage;

/// Fused DINOv2 ViT encoder block (pre-norm + LayerScale + tanh-approx GELU FFN).
pub fn dinov2_layer_fused(layer_idx: usize, hidden_size: usize, num_heads: usize, eps: f32) -> FlowStage {
    let lp = format!("blocks.{layer_idx}");
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
                .stage(FlowStage::VitSelfAttn(VitSelfAttnStage::dinov2(
                    &lp,
                    hidden_size,
                    num_heads,
                )))
                .stage(FlowStage::LayerScale(LayerScaleStage::new(format!(
                    "{lp}.ls1.gamma"
                ))))
                .residual_add()
                .residual_save()
                .layer_norm(
                    format!("{lp}.norm2.weight"),
                    format!("{lp}.norm2.bias"),
                    eps,
                )
                .stage(FlowStage::GeluFfn(GeluFfnStage::dinov2(&lp)))
                .stage(FlowStage::LayerScale(LayerScaleStage::new(format!(
                    "{lp}.ls2.gamma"
                ))))
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
