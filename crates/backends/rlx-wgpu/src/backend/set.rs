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

//! `set` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    ArgmaxParams, AttentionBwdParams, AttentionParams, BatchElementwiseRegionParams, BinaryParams,
    Conv1dParams, Conv2dParams, Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams,
    DequantMatmulParams, ElementwiseRegionParams, ExpandParams, FmaParams, FusedResidualLnParams,
    FusedResidualLnTeeParams, FusedResidualRmsNormParams, GatherAxisParams, GatherBwdParams,
    GatherParams, GroupedMatmulParams, GruParams, Kernel, LayerNormBwdParams, LayerNormParams,
    Mamba2Params, MatmulParams, MatmulQkvParams, NarrowConcatParams, Pool1dParams, Pool2dParams,
    Pool3dParams, ReduceParams, RmsNormBwdParams, RnnParams, RopeBwdParams, RopeParams,
    SampleParams, ScatterAddParams, SceParams, SelectiveScanParams, SoftmaxParams, TopKParams,
    TransposeParams, UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams, argmax_kernel,
    attention_bwd_kernel, attention_kernel, batch_elementwise_region_kernel, binary_kernel,
    cast_f32_to_f16_kernel, compare_kernel, concat_kernel, conv1d_kernel, conv2d_kernel,
    conv3d_kernel, copy_kernel, cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel,
    elementwise_region_kernel, elementwise_region_spatial_kernel, expand_kernel, fma_kernel,
    fused_residual_ln_kernel, fused_residual_ln_tee_kernel, fused_residual_rms_norm_kernel,
    gather_axis_kernel, gather_backward_acc_kernel, gather_backward_zero_kernel, gather_kernel,
    gather_split_kernel, grouped_matmul_kernel, gru_kernel,
    layer_norm_backward_gamma_partial_kernel, layer_norm_backward_gamma_reduce_kernel,
    layer_norm_backward_input_kernel, layernorm_kernel, mamba2_kernel,
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
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::*;

impl WgpuExecutable {
    /// Override RNG policy for in-graph random ops without recompiling.
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        *self.rng.write().expect("rng lock") = rng;
    }

    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every Step is in the safe set. See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        const STASH_MAX_BYTES: usize = 16 * 1024 * 1024;
        if data.len() * 4 <= STASH_MAX_BYTES {
            self.stashed_params.insert(name.to_string(), data.to_vec());
        }
        if self.coop_f16_vk {
            crate::coop_f16_vk::refresh_wide_b_flag(&mut self.coop_f16_vk_wide_b, name, data);
        }
        if self.unresolved.is_some() {
            self.pending_params.insert(name.to_string(), data.to_vec());
            return;
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            self.arena.write_f32(&dev.queue, id, data);
        }
    }

    /// Upload raw bytes for a Param. The bytes land tight-packed at
    /// the param's slot offset — no f32 round-trip. Used for quantized
    /// weights (int8 / int4) where the kernel reads the byte stream
    /// via `bitcast<u32>` from the f32-typed arena.
    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if self.unresolved.is_some() {
            self.pending_param_bytes
                .insert(name.to_string(), data.to_vec());
            return;
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            dev.queue
                .write_buffer(&self.arena.buffer, self.arena.offset(id) as u64, data);
        }
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }
}
