// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `dispatch` — extracted from the `backend` module for navigability (see `mod.rs`).

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
        // The window must span from `base` to `hi`, NOT from `lo` to `hi`.
        //
        // `base` is `lo` floored to the 256-byte storage-binding alignment, so
        // it sits at or below `lo`; sizing the window as `hi - lo` leaves it
        // short by exactly that flooring, and whichever operand is at the far
        // end falls outside the binding. The shader's clamped access then reads
        // or writes nothing, silently.
        //
        // Measured: resident training feeding `b` (src=240, dst=1216) got
        // base=0 with size=1024, so the destination at 1216..1232 was outside
        // the window and the parameter never updated. The same off-by-`lo & 255`
        // put `w__v`'s SOURCE (768..832) outside a 512-byte window, so Adam's
        // second moment read garbage and exploded. Compute `base` first, then
        // size from it.
        let base = (lo / 256) * 256;
        let mut size = hi.saturating_sub(base).div_ceil(256) * 256;
        size = size.max(256);
        // Shrink rather than slide: `hi` is inside the arena by construction, so
        // clamping the far end keeps both operands covered, whereas moving
        // `base` down would uncover `hi`.
        if base.saturating_add(size) > self.arena.size as u64 {
            size = (self.arena.size as u64).saturating_sub(base);
        }
        assert!(
            size <= max_binding,
            "rlx-wgpu arena copy spans {size} B between offsets {src} and {dst}, over this \
             adapter's {max_binding} B storage-binding limit. Widening the window is not \
             possible here; the copy would have to be split or routed through \
             copy_buffer_to_buffer."
        );
        debug_assert!(
            src >= base
                && src.saturating_add(nbytes) <= base + size
                && dst >= base
                && dst.saturating_add(nbytes) <= base + size,
            "arena copy window [{base}, {}) does not cover src {src}..{} and dst {dst}..{}",
            base + size,
            src + nbytes,
            dst + nbytes
        );
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
        if rlx_ir::env::flag("RLX_WGPU_FEED_TRACE") {
            eprintln!(
                "[copy] src={src} dst={dst} nbytes={nbytes} elems={elems} base={base} size={size} \
                 in_off={} out_off={} src_end_rel={} dst_end_rel={} window_elems={} arena={}",
                p.in_off,
                p.out_off,
                (src + nbytes).saturating_sub(base),
                (dst + nbytes).saturating_sub(base),
                size / 4,
                self.arena.size,
            );
        }
        let ck = copy_kernel(&dev.device);
        let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rlx-wgpu kv_feed_copy uniform"),
            size: std::mem::size_of::<CopyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
        let bg = bind_arena_window(&dev.device, ck, &self.arena, base, size, &u);
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
