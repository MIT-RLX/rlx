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

//! `test` — extracted from the `backend` module for navigability (see `mod.rs`).

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
    FusedResidualLnTeeParams, FusedResidualRmsNormParams, GatedResidualBackwardParams,
    GatedResidualParams, GatherAxisParams, GatherBwdParams, GatherParams, GroupedMatmulParams,
    GruParams, Kernel, LayerNormBwdParams, LayerNormParams, Mamba2Params, MatmulParams,
    MatmulQkvParams, NarrowConcatParams, Pool1dParams, Pool2dParams, Pool3dParams, ReduceParams,
    RmsNormBwdParams, RnnParams, RopeBwdParams, RopeParams, SampleParams, ScatterAddParams,
    SceParams, SelectiveScanParams, SoftmaxParams, TopKParams, TransposeParams, UmapKnnParams,
    UnaryParams, WelchPeaksGpuParams, WhereParams, argmax_kernel, attention_bwd_kernel,
    attention_kernel, batch_elementwise_region_kernel, binary_kernel, cast_f32_to_f16_kernel,
    compare_kernel, concat_kernel, conv1d_kernel, conv2d_kernel, conv3d_kernel, copy_kernel,
    cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel, elementwise_region_kernel,
    elementwise_region_spatial_kernel, expand_kernel, fma_kernel, fused_residual_ln_kernel,
    fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel, gather_axis_kernel,
    gather_backward_acc_kernel, gather_backward_zero_kernel, gather_kernel, gather_split_kernel,
    grouped_matmul_kernel, gru_kernel, layer_norm_backward_gamma_partial_kernel,
    layer_norm_backward_gamma_reduce_kernel, layer_norm_backward_input_kernel, layernorm_kernel,
    mamba2_kernel, matmul_coop_f16_vulkan_active_kernel, matmul_coop_f16_vulkan_kernel,
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
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::*;

impl WgpuExecutable {
    /// Test hook: first `Step::Attention` Q sequence stride (600 = packed QKV).
    #[doc(hidden)]
    pub fn test_attn_q_seq_stride(&self) -> Option<u32> {
        self.schedule.iter().find_map(|s| {
            if let Step::Attention { params, .. } = s {
                Some(params.q_seq_stride)
            } else {
                None
            }
        })
    }

    /// Test hook: `(q_off, k_off, v_off, q_seq_stride)` for the first attention step.
    #[doc(hidden)]
    pub fn test_attn_offsets_and_stride(&self) -> Option<(u32, u32, u32, u32)> {
        self.schedule.iter().find_map(|s| {
            if let Step::Attention { params, .. } = s {
                Some((
                    params.q_off,
                    params.k_off,
                    params.v_off,
                    params.q_seq_stride,
                ))
            } else {
                None
            }
        })
    }

    /// Global arena offset in f32 elements (not bind-window-local).
    #[doc(hidden)]
    pub fn test_arena_offset_elems(&self, id: NodeId) -> u32 {
        (self.arena.offset(id) / 4) as u32
    }
}
