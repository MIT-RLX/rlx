// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `compile` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    AdaLayerNormBackwardParams, AdaLayerNormParams, ArgmaxParams, AttentionBwdParams,
    AttentionParams, BatchElementwiseRegionParams, BinaryParams, Conv1dParams, Conv2dParams,
    Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams, DequantMatmulParams,
    ElementwiseRegionParams, ExpandParams, FmaParams, FusedResidualLnParams,
    FusedResidualLnTeeParams, FusedResidualRmsNormParams, GatedDeltaNetParams,
    GatedResidualBackwardParams, GatedResidualParams, GatherAxisParams, GatherBwdParams,
    GatherParams, GroupedMatmulParams, GruParams, Im2Col2dParams, Kernel, LayerNormBwdParams,
    LayerNormParams, Mamba2Params, MatmulParams, MatmulQkvParams, NarrowConcatParams, Pool1dParams,
    Pool2dParams, Pool3dParams, ReduceParams, RmsNormBwdParams, RnnParams, RopeBwdParams,
    RopeParams, SampleParams, ScatterAddParams, SceParams, SelectiveScanParams, SoftmaxParams,
    TopKParams, TransposeParams, UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams,
    ada_layer_norm_backward_kernel, ada_layer_norm_kernel, argmax_kernel, attention_bwd_kernel,
    attention_kernel, batch_elementwise_region_kernel, binary_kernel, cast_f32_to_f16_kernel,
    compare_kernel, concat_kernel, conv1d_kernel, conv1d_tiled_kernel, conv2d_kernel,
    conv3d_kernel, copy_kernel, cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel,
    elementwise_region_kernel, elementwise_region_spatial_kernel, expand_kernel, fma_kernel,
    fused_residual_ln_kernel, fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel,
    gated_delta_net_kernel, gated_residual_backward_kernel, gated_residual_kernel,
    gather_axis_kernel, gather_backward_acc_kernel, gather_backward_zero_kernel, gather_kernel,
    gather_split_kernel, grouped_matmul_kernel, gru_kernel, im2col2d_kernel,
    layer_norm_backward_gamma_partial_kernel, layer_norm_backward_gamma_reduce_kernel,
    layer_norm_backward_input_kernel, layernorm_kernel, lead_pack_uniform, mamba2_kernel,
    matmul_coop_f16_vulkan_active_kernel, matmul_coop_f16_vulkan_kernel,
    matmul_coop_f32_active_kernel, matmul_coop16_kernel, matmul_f16_compute_kernel,
    matmul_f16w_kernel, matmul_kernel, matmul_qkv_coop_f16_vk_active_kernel,
    matmul_qkv_coop_f16_vk_kernel, matmul_qkv_coop_f32_kernel, matmul_qkv_kernel,
    matmul_wide_active_kernel, matmul_wide_kernel, narrow_kernel, pool1d_kernel, pool2d_kernel,
    pool3d_kernel, reduce_kernel, rms_norm_backward_kernel, rms_norm_backward_param_kernel,
    rnn_kernel, rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_kernel,
    selective_scan_kernel, softmax_cross_entropy_kernel, softmax_kernel, topk_kernel,
    transpose_kernel, umap_knn_kernel, unary_f16_mirror_kernel, unary_kernel,
    welch_peaks_gpu_kernel, where_kernel,
};
use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, AdaNormKind, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op, ada_modulation_launch, ada_modulation_lead_pack};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::*;

mod lower;

impl WgpuExecutable {
    /// Compile against an explicit `DimBinding`. Each `Dim::Dynamic`
    /// in the graph that maps to a symbol in `bindings` is replaced
    /// with `Dim::Static(size)` before the standard compile runs.
    /// Symbols not in the binding stay dynamic — and then `compile`
    /// will panic with the usual diagnostic.
    pub fn compile_with_bindings(graph: Graph, bindings: &DimBinding) -> Self {
        if bindings.is_empty() {
            return Self::compile(graph);
        }
        // Walk the graph and bind every node's shape.
        let mut fresh = Graph::new(&graph.name);
        for node in graph.nodes() {
            let bound = node.shape.bind(bindings);
            fresh.add_node(node.op.clone(), node.inputs.clone(), bound);
        }
        fresh.set_outputs(graph.outputs.clone());
        Self::compile(fresh)
    }

    pub fn compile(graph: Graph) -> Self {
        Self::compile_rng(graph, rlx_ir::RngOptions::default())
    }

    pub fn compile_rng(graph: Graph, rng: rlx_ir::RngOptions) -> Self {
        use rlx_opt::pass::Pass as _;
        // Match Session `WgpuBackend::compile`: rewrite If/While before lower.
        let graph = rlx_opt::LowerControlFlow.run(graph);
        let rng = std::sync::Arc::new(std::sync::RwLock::new(rng));
        if has_dynamic_dims(&graph) {
            return Self::deferred(graph, rng);
        }
        Self::compile_static_inner(graph, rng)
    }

    pub(crate) fn compile_static_inner(
        graph: Graph,
        rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
    ) -> Self {
        lower::compile_static_inner(graph, rng)
    }
}
