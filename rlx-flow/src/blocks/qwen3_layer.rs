// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

use std::sync::{Arc, Mutex};

use super::{Qwen3DecodeLayerSpec, Qwen3DecodeLayerStage, Qwen3DecoderSpec, Qwen3DecoderStage};
use crate::stage::FlowStage;

pub fn qwen3_prefill_layer_fused(layer_idx: usize, spec: Qwen3DecoderSpec) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::Qwen3Decoder(Qwen3DecoderStage::layer(
            layer_idx, spec,
        ))),
    }
}

pub fn qwen3_prefill_layer_fused_kv(
    layer_idx: usize,
    spec: Qwen3DecoderSpec,
    kv_sink: Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>,
) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::Qwen3Decoder(Qwen3DecoderStage::layer_with_kv(
            layer_idx, spec, kv_sink,
        ))),
    }
}

/// KV-cache decode layer (QK-norm + concat past K/V + causal/custom attention).
pub fn qwen3_decode_layer_fused(
    layer_idx: usize,
    spec: Qwen3DecodeLayerSpec,
    kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::Qwen3DecodeLayer(Qwen3DecodeLayerStage::layer(
            layer_idx, spec, kv_out,
        ))),
    }
}
