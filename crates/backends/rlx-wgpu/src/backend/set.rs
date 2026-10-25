// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `set` — extracted from the `backend` module for navigability (see `mod.rs`).

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

    /// Un-arm the static-weight-pack skip: a param write makes every pack that
    /// consumes it stale.
    ///
    /// Without this, `run(); set_param(w, ..); run()` keeps the FIRST run's
    /// fused pack and the consuming GEMM silently computes with the old
    /// weights — no error, just a wrong answer. That is the shape of every
    /// weight-swap workload: a training step, a LoRA merge, quantisation
    /// re-binding, a sweep harness reusing one executable across weight sets.
    ///
    /// Clearing the flag wholesale (rather than tracking which packs consume
    /// `name`) costs one run's worth of re-packing and cannot be wrong. Weight
    /// writes are rare next to inference steps.
    fn invalidate_static_weight_packs(&mut self) {
        self.static_once_done = false;
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        self.invalidate_static_weight_packs();
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
        self.invalidate_static_weight_packs();
        if self.unresolved.is_some() {
            self.pending_param_bytes
                .insert(name.to_string(), data.to_vec());
            return;
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        if let Some(&id) = self.param_offsets.get(name)
            && self.arena.has(id)
        {
            // Chunked + shard-safe (weight buffer or arena). Large F5 tiles
            // truncated on a single `queue.write_buffer` under Metal/wgpu.
            self.arena
                .write_bytes_range(&dev.queue, self.arena.offset(id), data);
        }
    }

    /// Upload a raw BF16 weight byte stream (`ne*2` bytes, native LE) for a
    /// param the compiled arena stores PACKED in `bf16_weight_buffer`. Returns
    /// `true` iff handled here (param is packed) — no host bf16→f32 widen and
    /// no ne*4 arena write. Returns `false` (caller falls back to the f32-widen
    /// path) when the graph is still unresolved or the param is not packed.
    /// The `Arena::write_f32` mirror covers the deferred-replay case.
    pub fn set_param_bf16_packed(&mut self, name: &str, data: &[u8]) -> bool {
        self.invalidate_static_weight_packs();
        if self.unresolved.is_some() {
            return false;
        }
        let Some(&id) = self.param_offsets.get(name) else {
            return false;
        };
        if self.arena.has(id) && self.arena.is_bf16_packed(id) {
            let dev = wgpu_device().expect("rlx-wgpu: device gone");
            self.arena.write_bf16_packed(&dev.queue, id, data);
            true
        } else {
            false
        }
    }

    /// Adopt `other`'s weight buffer instead of keeping our own copy: one GPU
    /// copy of the packed weights then backs both executables (decode rungs,
    /// prefill and decode). Only when both park exactly the same params at the
    /// same offsets — executables compiled from one param set. Arena-resident
    /// params are not touched; the caller sets those as usual. `other` must
    /// already hold its weights. Returns false (upload normally) otherwise.
    pub fn share_weights_from(&mut self, other: &Self) -> bool {
        if self.unresolved.is_some() || other.unresolved.is_some() {
            return false;
        }
        let (dst, src) = (&self.arena, &other.arena);
        let (Some(dst_buf), Some(src_buf)) =
            (dst.weight_buffer.as_ref(), src.weight_buffer.as_ref())
        else {
            return false;
        };
        if dst.weight_shard_size != src.weight_shard_size
            || dst_buf.size() != src_buf.size()
            || dst.weight_extra_shards.len() != src.weight_extra_shards.len()
            || dst
                .weight_extra_shards
                .iter()
                .zip(&src.weight_extra_shards)
                .any(|(a, b)| a.size() != b.size())
        {
            return false;
        }
        fn parked(exe: &WgpuExecutable) -> Option<HashMap<&str, usize>> {
            let mut out = HashMap::new();
            for (name, id) in &exe.param_offsets {
                if let Some(&off) = exe.arena.weight_offsets.get(id) {
                    out.insert(name.as_str(), off);
                }
            }
            (!out.is_empty()).then_some(out)
        }
        match (parked(self), parked(other)) {
            (Some(a), Some(b)) if a == b => {}
            _ => return false,
        }
        if rlx_ir::env::flag("RLX_WGPU_SHARD_LOG") {
            eprintln!(
                "[rlx-wgpu] adopted a shared weight buffer ({:.1} MiB)",
                src_buf.size() as f64 / (1u64 << 20) as f64
            );
        }
        self.arena.weight_buffer = Some(src_buf.clone());
        self.arena.weight_extra_shards = src.weight_extra_shards.clone();
        self.gemv_plans
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.invalidate_static_weight_packs();
        true
    }

    /// Copy the named inputs' device contents aside, on the GPU, so
    /// [`Self::restore_inputs`] can put them back: the recurrent state a
    /// multi-token step updates in place, kept for when only part of the step
    /// is to stand. Replaces an earlier save. False for an unknown input.
    pub fn save_inputs(&mut self, names: &[&str]) -> bool {
        if self.unresolved.is_some() || self.arena.is_sharded() {
            return false;
        }
        let mut spans = Vec::with_capacity(names.len());
        let mut total = 0u64;
        for name in names {
            let Some(&id) = self.input_offsets.get(*name) else {
                return false;
            };
            let off = self.arena.offset(id);
            if crate::buffer::is_weight_off(off) {
                return false;
            }
            let len = (*self.arena.lens.get(&id).unwrap_or(&0) as u64).div_ceil(4) * 4;
            spans.push((off, len, total));
            total += len.div_ceil(16) * 16;
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        let reuse = matches!(&self.saved_inputs, Some((b, _)) if b.size() >= total.max(16));
        if !reuse {
            let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rlx-wgpu saved inputs"),
                size: total.max(16),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.saved_inputs = Some((buf, Vec::new()));
        }
        let (buf, saved) = self.saved_inputs.as_mut().expect("saved inputs");
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlx-wgpu save inputs"),
            });
        for &(off, len, at) in &spans {
            let (src, local) = self.arena.resolve_act(off);
            enc.copy_buffer_to_buffer(src, local as u64, buf, at, len);
        }
        dev.queue.submit(std::iter::once(enc.finish()));
        *saved = spans;
        true
    }

    /// Put back the inputs [`Self::save_inputs`] copied aside. False when
    /// nothing was saved.
    pub fn restore_inputs(&mut self) -> bool {
        let Some((buf, spans)) = self.saved_inputs.as_ref() else {
            return false;
        };
        if spans.is_empty() {
            return false;
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlx-wgpu restore inputs"),
            });
        for &(off, len, at) in spans {
            let (dst, local) = self.arena.resolve_act(off);
            enc.copy_buffer_to_buffer(buf, at, dst, local as u64, len);
        }
        dev.queue.submit(std::iter::once(enc.finish()));
        true
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Mark an already-staged graph input as device-resident so subsequent
    /// `run` calls skip host re-upload (ORT-style I/O binding for constants).
    pub fn prepare_resident_gpu_handle(&mut self, name: &str) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.insert(name.to_string());
        self.gpu_handles.entry(name.to_string()).or_default();
        true
    }
}
