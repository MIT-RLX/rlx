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

use std::sync::{Arc, Mutex};

use super::{Qwen3DecodeLayerSpec, Qwen3DecodeLayerStage, Qwen3DecoderSpec, Qwen3DecoderStage};
use crate::side::SideOutputs;
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

pub fn qwen3_prefill_layer_fused_kv_qk(
    layer_idx: usize,
    spec: Qwen3DecoderSpec,
    kv_sink: Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>,
    qk_sink: Arc<std::sync::Mutex<Vec<rlx_ir::HirNodeId>>>,
) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::Qwen3Decoder(
            Qwen3DecoderStage::layer_with_kv_qk(layer_idx, spec, kv_sink, qk_sink),
        )),
    }
}

/// Prefill layer with optional KV and/or Q/K side taps (AIF probe).
pub fn qwen3_prefill_layer_side(
    layer_idx: usize,
    spec: Qwen3DecoderSpec,
    kv_sink: &SideOutputs,
    qk_sink: &SideOutputs,
    export_kv: bool,
    export_qk: bool,
) -> FlowStage {
    if export_qk {
        qwen3_prefill_layer_fused_kv_qk(layer_idx, spec, kv_sink.inner(), qk_sink.inner())
    } else if export_kv {
        qwen3_prefill_layer_fused_kv(layer_idx, spec, kv_sink.inner())
    } else {
        qwen3_prefill_layer_fused(layer_idx, spec)
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

pub fn qwen3_decode_layer_fused_qk(
    layer_idx: usize,
    spec: Qwen3DecodeLayerSpec,
    kv_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
    qk_out: Arc<Mutex<Vec<rlx_ir::HirNodeId>>>,
) -> FlowStage {
    FlowStage::Named {
        name: format!("layer{layer_idx}"),
        inner: Arc::new(FlowStage::Qwen3DecodeLayer(
            Qwen3DecodeLayerStage::layer_with_qk(layer_idx, spec, kv_out, qk_out),
        )),
    }
}

/// Decode layer with optional Q/K side taps (AIF decode-step probe).
pub fn qwen3_decode_layer_side(
    layer_idx: usize,
    spec: Qwen3DecodeLayerSpec,
    kv_out: &SideOutputs,
    qk_out: &SideOutputs,
    export_qk: bool,
) -> FlowStage {
    if export_qk {
        qwen3_decode_layer_fused_qk(layer_idx, spec, kv_out.inner(), qk_out.inner())
    } else {
        qwen3_decode_layer_fused(layer_idx, spec, kv_out.inner())
    }
}
