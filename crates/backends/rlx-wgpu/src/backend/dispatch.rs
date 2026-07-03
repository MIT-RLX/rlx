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

//! `dispatch` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};
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

use super::*;

impl WgpuExecutable {
    pub(crate) fn dispatch_arena_copy_bytes(
        &self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
        src_id: NodeId,
        dst_id: NodeId,
        nbytes: usize,
    ) {
        if nbytes == 0 {
            return;
        }
        let src = self.arena.offset(src_id) as u64;
        let dst = self.arena.offset(dst_id) as u64;
        let nbytes = nbytes
            .min(self.arena.len_of(src_id))
            .min(self.arena.len_of(dst_id)) as u64;
        let elems = (nbytes / 4).max(1) as u32;
        let lo = src.min(dst);
        let hi = src.saturating_add(nbytes).max(dst.saturating_add(nbytes));
        let max_binding = dev.device.limits().max_storage_buffer_binding_size;
        let mut size = hi.saturating_sub(lo).div_ceil(256) * 256;
        size = size.max(256).min(max_binding);
        let mut base = (lo / 256) * 256;
        if base.saturating_add(size) > self.arena.size as u64 {
            base = (self.arena.size as u64).saturating_sub(size);
            base = (base / 256) * 256;
        }
        let p = CopyParams {
            n: elems,
            in_off: (src.saturating_sub(base) / 4) as u32,
            out_off: (dst.saturating_sub(base) / 4) as u32,
            _p0: 0,
            _p1: 0,
            _p2: 0,
            _p3: 0,
            _p4: 0,
        };
        let ck = copy_kernel(&dev.device);
        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu kv_feed_copy uniform"),
            size: std::mem::size_of::<CopyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
        let bg = bind_two_buf0_window(&dev.device, ck, &self.arena.buffer, base, size, &u);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rlx-wgpu kv_feed_copy pass"),
            ..Default::default()
        });
        pass.set_pipeline(&ck.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let (gx, gy, gz) = dispatch_dims(elems, 64);
        pass.dispatch_workgroups(gx, gy, gz);
    }


    #[allow(dead_code)]
    pub(crate) fn dispatch_arena_copy_between_nodes(
        &self,
        dev: &crate::device::WgpuDevice,
        enc: &mut wgpu::CommandEncoder,
        src_id: NodeId,
        dst_id: NodeId,
    ) {
        let nbytes = self.arena.len_of(src_id).min(self.arena.len_of(dst_id));
        self.dispatch_arena_copy_bytes(dev, enc, src_id, dst_id, nbytes);
    }

}
