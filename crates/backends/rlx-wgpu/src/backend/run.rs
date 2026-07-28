// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `run` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::buffer::{
    Arena, ReadbackLayout, ReadbackStaging, TinyReadbackStaging, decode_mapped_readback_f32,
    decode_tiny_mapped_f32, encode_readback_copies, plan_f32_uniform, read_f32_many_pooled,
    schedule_readback_map, use_tiny_readback, wait_readback_map,
};
use crate::device::wgpu_device;
use crate::kernels::{
    ActivationBackwardParams, AdaLayerNormBackwardParams, AdaLayerNormParams, ArgmaxParams,
    AttentionBwdParams, AttentionParams, BatchElementwiseRegionParams, BinaryParams, Conv1dParams,
    Conv2dParams, Conv3dParams, CopyParams, CumsumBwdParams, CumsumParams, DequantMatmulParams,
    ElementwiseRegionParams, ExpandParams, FmaParams, FusedResidualLnParams,
    FusedResidualLnTeeParams, FusedResidualRmsNormParams, GatedDeltaNetParams,
    GatedResidualBackwardParams, GatedResidualParams, GatherAxisParams, GatherBwdParams,
    GatherParams, GroupedMatmulParams, GruParams, Kernel, LayerNormBwdParams, LayerNormParams,
    Mamba2Params, MatmulParams, MatmulQkvParams, NarrowConcatParams, Pool1dParams, Pool2dParams,
    Pool3dParams, ReduceParams, RmsNormBwdParams, RnnParams, RopeBwdParams, RopeParams,
    SampleParams, ScatterAddParams, SceBwdParams, SceParams, SelectiveScanParams, SoftmaxParams,
    TopKParams, TransposeParams, UmapKnnParams, UnaryParams, WelchPeaksGpuParams, WhereParams,
    activation_backward_kernel, ada_layer_norm_backward_kernel, ada_layer_norm_kernel,
    argmax_kernel, attention_bwd_kernel, attention_kernel, axial_rope2d_kernel,
    batch_elementwise_region_kernel, binary_c64_kernel, binary_kernel, cast_f32_to_f16_kernel,
    cast_kernel, compare_kernel, complex_cast_kernel, complex_norm_sq_backward_kernel,
    complex_norm_sq_kernel, concat_kernel, conjugate_c64_kernel, conv_transpose3d_kernel,
    conv1d_kernel, conv1d_tiled_kernel, conv2d_kernel, conv3d_backward_input_kernel,
    conv3d_backward_weight_kernel, conv3d_kernel, copy_kernel, cum_scan_kernel,
    cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel, dequant_matmul_mlx_kernel,
    elementwise_region_kernel, elementwise_region_spatial_kernel, expand_kernel,
    fake_quantize_fixed_kernel, fake_quantize_perbatch_kernel, fft_butterfly_stage_kernel,
    fma_kernel, fused_residual_ln_kernel, fused_residual_ln_tee_kernel,
    fused_residual_rms_norm_kernel, gated_delta_net_kernel, gated_residual_backward_kernel,
    gated_residual_kernel, gather_axis_kernel, gather_backward_acc_kernel,
    gather_backward_zero_kernel, gather_kernel, gather_split_kernel,
    group_norm_backward_beta_kernel, group_norm_backward_gamma_kernel,
    group_norm_backward_input_kernel, grouped_matmul_kernel, gru_kernel, im2col2d_kernel,
    layer_norm_backward_gamma_partial_kernel, layer_norm_backward_gamma_reduce_kernel,
    layer_norm_backward_input_kernel, layernorm_kernel, mamba2_kernel,
    matmul_coop_f16_vulkan_active_kernel, matmul_coop_f16_vulkan_kernel,
    matmul_coop_f32_active_kernel, matmul_coop16_kernel, matmul_f16_compute_kernel,
    matmul_f16w_kernel, matmul_kernel, matmul_qkv_coop_f16_vk_active_kernel,
    matmul_qkv_coop_f16_vk_kernel, matmul_qkv_coop_f32_kernel, matmul_qkv_kernel,
    matmul_wide_active_kernel, matmul_wide_kernel, maxpool2d_backward_kernel,
    maxpool3d_backward_kernel, narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel,
    reduce_kernel, rms_norm_backward_kernel, rms_norm_backward_param_kernel, rnn_kernel,
    rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_kernel, selective_scan_kernel,
    softmax_cross_entropy_backward_kernel, softmax_cross_entropy_kernel,
    softmax_cross_entropy_with_logits_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    umap_knn_kernel, unary_f16_mirror_kernel, unary_kernel, welch_peaks_gpu_kernel, where_kernel,
};
use rlx_ir::dynamic::{bind_graph, has_dynamic_dims, infer_bindings_from_f32_inputs, same_binding};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::shape::DimBinding;
use rlx_ir::{Graph, NodeId, Op};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

use super::*;

impl WgpuExecutable {
    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.pending_read_indices = read_indices.map(|s| s.to_vec());
        let outs = self.run_inner(inputs);
        self.pending_read_indices = None;
        outs
    }

    /// Async sibling of [`Self::run`] for the browser, where GPU→CPU readback
    /// cannot block the event loop. All compute is dispatched + submitted
    /// synchronously (via the normal `run_inner` in dispatch-only mode); the
    /// outputs are then read back from the arena asynchronously.
    ///
    /// Supports pure feed-forward graphs only — graphs with host-executed ops
    /// (GGUF dequant, LSTM/GRU, GatedDeltaNet, FFT-host, …) map intermediate
    /// results back mid-graph, which would block the browser. Such graphs
    /// panic with a clear message rather than hang.
    #[cfg(target_arch = "wasm32")]
    pub async fn run_async(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        assert!(
            !self
                .schedule
                .iter()
                .any(|s| step_runs_on_host(s) || step_is_tail_host(s)),
            "rlx-wgpu run_async: graph contains host-executed ops unsupported on wasm"
        );

        // 1. Dispatch + submit all compute, skipping the blocking readback.
        self.dispatch_only = true;
        let _ = self.run_inner(inputs);
        self.dispatch_only = false;

        let dev =
            wgpu_device().expect("rlx-wgpu: device not initialized (call init_wgpu_device first)");

        // 2. Read the selected outputs back from the arena asynchronously.
        let plan = self.readback_plan();
        let out_ids_all: Vec<_> = self.graph.outputs.clone();
        let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
        let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
        if layout.regions.is_empty() {
            return Vec::new();
        }
        ReadbackStaging::prepare(&dev.device, &mut self.readback_staging, layout.total_bytes);
        let staging_buf = self
            .readback_staging
            .as_ref()
            .expect("readback staging")
            .buffer()
            .clone();

        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rlx-wgpu async readback"),
            });
        encode_readback_copies(&mut enc, &self.arena, &staging_buf, &out_ids, &layout);
        dev.queue.submit(std::iter::once(enc.finish()));

        crate::buffer::wasm_async::map_read_async(&staging_buf, layout.total_bytes)
            .await
            .expect("rlx-wgpu async buffer map failed");

        let partial = decode_mapped_readback_f32(&staging_buf, &layout);
        self.pack_readback_outputs(&plan, partial)
    }

    /// True when the compiled schedule contains host-executed ops (GGUF
    /// dequant, LSTM/GRU, collectives, FFT-host, …). Such graphs cannot
    /// run on browser WebGPU (`run_async` rejects them).
    pub fn requires_host_execution(&self) -> bool {
        self.schedule
            .iter()
            .any(|s| super::step_runs_on_host(s) || super::step_is_tail_host(s))
    }

    pub(crate) fn run_tail_host_audio_ops(&self, dev: &crate::device::WgpuDevice) {
        if !self.schedule.iter().any(step_is_tail_host) {
            return;
        }
        for step in &self.schedule {
            if !step_is_tail_host(step) {
                continue;
            }
            match step {
                Step::WelchPeaksHost {
                    spec_byte_off,
                    dst_byte_off,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                } => {
                    crate::welch_peaks_host::run_welch_peaks(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *dst_byte_off as usize,
                        *welch_batch as usize,
                        *n_fft as usize,
                        *n_segments as usize,
                        *k as usize,
                    );
                }
                Step::LogMelHost {
                    spec_byte_off,
                    filt_byte_off,
                    dst_byte_off,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    crate::log_mel_host::run_log_mel(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                    );
                }
                Step::LogMelBackwardHost {
                    spec_byte_off,
                    filt_byte_off,
                    dy_byte_off,
                    dst_byte_off,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    crate::log_mel_host::run_log_mel_backward(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dy_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                    );
                }
                _ => {}
            }
        }
    }

    pub(crate) fn run_inner(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        // Lazy compile path: if we deferred compile waiting for shapes,
        // infer the binding from input data lengths now and compile.
        if self.unresolved.is_some() {
            self.lazy_compile_for_inputs(inputs);
        }
        let dev = wgpu_device().expect("rlx-wgpu: device gone");
        self.stage_gpu_handle_inputs(dev, inputs);
        // Always re-stage graph inputs. Arena liveness reuse may overwrite
        // input slots once the input dies mid-graph; a hash-based skip across
        // runs would leave those slots holding stale activations (empty/garbage
        // outputs on Conformer-CTC and similar). `RLX_WGPU_FORCE_INPUT_UPLOAD`
        // is retained as a documented no-op for callers that set it.
        let _ = rlx_ir::env::flag("RLX_WGPU_FORCE_INPUT_UPLOAD");
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                self.arena.write_f32(&dev.queue, id, data);
            }
        }
        for &(act_id, act, ref src_name) in &self.coop_f16_host_activations {
            let src =
                host_tensor_f32(src_name, inputs, &self.stashed_params).unwrap_or_else(|| {
                    panic!("rlx-wgpu CoopF16Vk host activation: missing tensor {src_name:?}")
                });
            let mirrored = apply_activation_host(act, src);
            self.arena.write_f32(&dev.queue, act_id, &mirrored);
        }
        if !self.coop_f16_host_activations.is_empty() {
            // Ensure host staging writes are visible before CoopF16Vk reads f16.
            let flush = dev
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("rlx-wgpu host mirror flush"),
                });
            dev.queue.submit(std::iter::once(flush.finish()));
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        }

        // Active-extent (PLAN L1): scale safe Steps' primary dim by
        // actual/upper. Used in BOTH the uniform-write loop (so the
        // kernel sees the scaled count) AND the dispatch loop (so the
        // workgroup grid is shrunk).
        let active = self.active_extent.filter(|_| self.all_safe_for_active());
        if rlx_ir::env::flag("RLX_WGPU_ACTIVE_TRACE")
            && self.active_extent.is_some()
            && active.is_none()
        {
            let bad: Vec<_> = self
                .schedule
                .iter()
                .filter(|s| !s.safe_for_active_extent())
                .map(step_name)
                .collect();
            eprintln!(
                "[wgpu-active] extent={:?} DISABLED by unsafe steps: {bad:?}",
                self.active_extent
            );
        }
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // Stage uniform writes — but skip the loop entirely when the
        // bytes already in the uniforms match this run's active extent.
        // BERT inference at fixed batch hits this path: 100+ tiny
        // queue.write_buffer calls (one per Step) collapse to zero,
        // saving milliseconds of staging-copy overhead.
        let need_uniform_writes = self.uniforms_active_extent != Some(active);
        if need_uniform_writes {
            let mut gpu_ui = 0usize;
            for (step_i, step) in self.schedule.iter().enumerate() {
                if step_runs_on_host(step) {
                    continue;
                }
                if self.static_once_done && self.static_once_steps.contains(&step_i) {
                    if !matches!(step, Step::FftGpu { .. }) {
                        gpu_ui += 1;
                    }
                    continue;
                }
                if std::env::var("RLX_DBG_STEP").is_ok() {
                    let usz = self.uniforms.get(gpu_ui).map(|u| u.size()).unwrap_or(0);
                    eprintln!("[wgpu-uni gpu_ui={gpu_ui} ubuf={usz}B] {}", step_name(step));
                }
                match step {
                    Step::CastF32ToF16 { .. } => {
                        // Params are static for this step (offset+len), so the
                        // pre-pass write at compile time is sufficient. No
                        // active-extent scaling — len is the full element count.
                    }
                    Step::Matmul {
                        m,
                        k,
                        n,
                        a_off_f32,
                        b_off_f32,
                        c_off_f32,
                        batch,
                        a_batch_stride,
                        b_batch_stride,
                        c_batch_stride,
                        has_bias,
                        bias_off_f32,
                        act_id,
                        b_is_param: _,
                        compute_precision: _,
                    } => {
                        // PLAN L1 (safe at any batch — c_batch_stride is
                        // pre-baked at compile time at FULL m, so scaling
                        // params.m only changes per-thread bound checks).
                        let m_scaled = scale(*m);
                        let p = MatmulParams {
                            m: m_scaled,
                            k: *k,
                            n: *n,
                            a_off: *a_off_f32,
                            b_off: *b_off_f32,
                            c_off: *c_off_f32,
                            batch: *batch,
                            a_batch_stride: *a_batch_stride,
                            b_batch_stride: *b_batch_stride,
                            c_batch_stride: *c_batch_stride,
                            has_bias: *has_bias,
                            bias_off: *bias_off_f32,
                            act_id: *act_id,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        };
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Binary { params } | Step::Compare { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Unary { params, .. } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Where { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Fma { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ReluBackward { params } | Step::ActivationBackward { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Reduce { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Softmax { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SoftmaxCrossEntropy { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SoftmaxCrossEntropyWithLogits { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SoftmaxCrossEntropyBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::RmsNormBackwardInput { params }
                    | Step::RmsNormBackwardGamma { params }
                    | Step::RmsNormBackwardBeta { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardInput { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardGammaPartial { params, .. } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::LayerNormBackwardGammaReduce { params } => {
                        // `outer` here is the partial chunk count (not
                        // a batch dim) — do NOT apply active-extent
                        // scaling.
                        dev.queue.write_buffer(
                            &self.uniforms[gpu_ui],
                            0,
                            bytemuck::bytes_of(params),
                        );
                    }
                    Step::CumsumBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::RopeBackward { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatherBackward { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Cumsum { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::CumScan { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FftGpu { .. } => {}
                    Step::Copy { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Cast { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ComplexCast { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::BinaryC64 { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ComplexNormSq { params }
                    | Step::ComplexNormSqBackward { params }
                    | Step::ConjugateC64 { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FftButterflyStage { params } => {
                        let mut p = *params;
                        p.batch = scale(p.batch);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::BufferCopy { .. } => {}
                    Step::ElementwiseRegion { params } => {
                        // Active-extent: scale element count.
                        let mut p = *params;
                        p.len = scale(p.len);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::BatchElementwiseRegion { params } => {
                        let mut p = *params;
                        p.slice_len = scale(p.slice_len);
                        p.num_batch = scale(p.num_batch);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Transpose { params, .. } => {
                        // PLAN L1: when bucket_outermost == 1, scale
                        // `out_total` proportional to scaling `out_dim_0`.
                        // Other transposes leave out_total at full extent
                        // (predicate prevents the active-extent path).
                        let mut p = *params;
                        if p.bucket_outermost == 1 && p.out_dim_0 > 0 {
                            let scaled_d0 = scale(p.out_dim_0);
                            let inner = p.out_total / p.out_dim_0;
                            p.out_total = scaled_d0 * inner;
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Narrow { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Concat { params } => {
                        let mut p = *params;
                        p.total = scale(p.total);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Gather { params } => {
                        let mut p = *params;
                        p.n_out = scale(p.n_out);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatherAxis { params } => {
                        let mut p = *params;
                        // Active-extent trims the LEADING (outer) dim only; the
                        // gathered axis + trailing are not seq-proportional.
                        // Scaling `total` directly would drop trailing lanes for
                        // a seq-collapsing gather (e.g. last-token select, outer=1).
                        p.total = scale(p.outer) * p.num_idx * p.trailing;
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Attention { params, .. } => {
                        // PLAN L1: scale seq_q + seq_k. Stride fields
                        // (seq_q_stride / seq_k_stride) stay at the
                        // compile-time full extent, so per-(batch, head)
                        // offset math in the WGSL stays correct.
                        let mut p = *params;
                        p.seq_q = scale(p.seq_q);
                        p.seq_k = scale(p.seq_k);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AttentionBackward { params, .. } => {
                        let mut p = *params;
                        if p.wrt == 0 {
                            p.seq_q = scale(p.seq_q);
                        } else {
                            p.seq_k = scale(p.seq_k);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Rope { params } => {
                        // PLAN L1: scale `seq` and `n_total` proportionally.
                        // `seq_stride` and `batch` stay at compile-time
                        // values; the WGSL kernel uses them for buffer
                        // offsets while `seq` / `n_total` are loop bounds.
                        let mut p = *params;
                        let s_active = scale(p.seq);
                        p.seq = s_active;
                        p.n_total = p.batch * s_active * p.last_dim;
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Expand { params, .. } => {
                        // PLAN L1: same pattern as Transpose.
                        let mut p = *params;
                        if p.bucket_outermost == 1 && p.out_dim_0 > 0 {
                            let scaled_d0 = scale(p.out_dim_0);
                            let inner = p.out_total / p.out_dim_0;
                            p.out_total = scaled_d0 * inner;
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Argmax { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool2d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::MaxPool2dBackward { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::MaxPool3dBackward { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv3dBackwardInput { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv3dBackwardWeight { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GroupNormBackwardInput { params }
                    | Step::GroupNormBackwardGamma { params }
                    | Step::GroupNormBackwardBeta { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AxialRope2d { params } => {
                        let mut p = *params;
                        p.batch = scale(p.batch);
                        p.n_total = p.batch * p.seq * p.hidden;
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FakeQuantizeFixed { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        // axis=None: inner tracks full extent with n.
                        if p.chan_dim <= 1 {
                            p.inner = p.n.max(1);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FakeQuantizePerBatch { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        if p.chan_dim <= 1 {
                            p.inner = p.n.max(1);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv2d { params } | Step::Conv2dTiled { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool1d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Pool3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv1d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Conv3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ConvTranspose3d { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::ScatterAdd { params } => {
                        // Two-phase: phase 0 zeros the FULL output (preserves
                        // accumulator semantics); phase 1 scatters first
                        // num_updates_active updates only.
                        let mut p = *params;
                        if p.op == 1 {
                            p.num_updates = scale(p.num_updates);
                        }
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::TopK { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::WelchPeaksGpu { params } => {
                        let mut p = *params;
                        p.welch_batch = scale(p.welch_batch);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::UmapKnn { params } => {
                        let mut p = *params;
                        p.n = scale(p.n);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GroupedMatmul { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Sample { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::SelectiveScan { params } => {
                        // Predicate-gated to batch=1: scale seq.
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Mamba2 { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Gru { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::Rnn { params } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatedDeltaNet {
                        params,
                        use_gpu: true,
                        ..
                    } => {
                        let mut p = *params;
                        p.seq = scale(p.seq);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::DequantMatmul { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::DequantMatmulMlx { params } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    // Im2ColGpu params are static (written once at compile);
                    // no active-extent scaling.
                    Step::Im2ColGpu { .. }
                    | Step::GatherSplit { .. }
                    | Step::DequantMatmulGguf { .. }
                    | Step::DequantMatmulInt8Host { .. }
                    | Step::DequantMatmulMlxHost { .. }
                    | Step::Conv2dHost { .. }
                    | Step::DequantGroupedMatmulGguf { .. }
                    | Step::DequantGroupedMatmulMlxHost { .. }
                    | Step::GatedDeltaNet { use_gpu: false, .. }
                    | Step::Lstm { .. }
                    | Step::ConvTranspose2d { .. }
                    | Step::ConvTranspose3dHost { .. }
                    | Step::GroupNormHost { .. }
                    | Step::LayerNorm2dHost { .. }
                    | Step::ResizeNearest2xHost { .. }
                    | Step::ReverseHost { .. }
                    | Step::ArgReduceHost { .. }
                    | Step::AxialRope2dHost { .. }
                    | Step::GruHost { .. }
                    | Step::RnnHost { .. }
                    | Step::Llada2GroupLimitedGate { .. }
                    | Step::UmapKnnHost { .. }
                    | Step::MsDeformAttnHost { .. }
                    | Step::CollectiveHost { .. }
                    | Step::CustomHost { .. }
                    | Step::FftHost { .. }
                    | Step::ScanHost { .. }
                    | Step::HostOp { .. }
                    | Step::CpuIndexing { .. }
                    | Step::ConcatHost { .. }
                    | Step::ConcatHostPieces { .. }
                    | Step::TransposeHost { .. }
                    | Step::NarrowHost { .. }
                    | Step::ExpandHost { .. }
                    | Step::SpdHost { .. }
                    | Step::Im2ColHost { .. }
                    | Step::Conv2dBackwardWeightHost { .. }
                    | Step::Conv2dBackwardInputHost { .. }
                    | Step::RngNormalHost { .. }
                    | Step::RngUniformHost { .. }
                    | Step::WelchPeaksHost { .. }
                    | Step::LogMelHost { .. }
                    | Step::LogMelBackwardHost { .. } => {}
                    Step::FusedResidualLn { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FusedResidualLnTee { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::FusedResidualRmsNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AdaLayerNorm { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatedResidual { params } => {
                        let mut p = *params;
                        p.outer = scale(p.outer);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::AdaLayerNormBackward { params } => {
                        let mut p = *params;
                        p.mod_rows = scale(p.mod_rows);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::GatedResidualBackward { params } => {
                        let mut p = *params;
                        p.mod_rows = scale(p.mod_rows);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    Step::MatmulQkv { params, kind: _ } => {
                        let mut p = *params;
                        p.m = scale(p.m);
                        dev.queue
                            .write_buffer(&self.uniforms[gpu_ui], 0, bytemuck::bytes_of(&p));
                    }
                    // Raw-GPU custom op: its storage params buffer is written at
                    // compile time (mapped_at_creation) with static, window-
                    // relative offsets — no active-extent scaling. No-op here so
                    // `gpu_ui` still advances for its uniform slot.
                    Step::WgpuGpuKernel { .. } => {}
                    #[cfg(feature = "splat")]
                    Step::GaussianSplatRender { .. }
                    | Step::GaussianSplatRenderBackward { .. }
                    | Step::GaussianSplatPrepare { .. }
                    | Step::GaussianSplatRasterize { .. } => {}
                }
                if !matches!(step, Step::FftGpu { .. }) {
                    gpu_ui += 1;
                }
            }
            self.uniforms_active_extent = Some(active);
        }

        // Encode + submit.
        let mm_k = matmul_kernel(&dev.device);
        let mm_w_active = matmul_wide_active_kernel(&dev.device);
        let mm_f16w = matmul_f16w_kernel(&dev.device);
        let mm_f16c = matmul_f16_compute_kernel(&dev.device);
        let mm_coop = matmul_coop16_kernel(&dev.device);
        let mm_coop_f16_vk = matmul_coop_f16_vulkan_kernel(&dev.device);
        let mm_coop_f32 = matmul_coop_f32_active_kernel(&dev.device);
        let mm_cast = cast_f32_to_f16_kernel(&dev.device);
        let bk = binary_kernel(&dev.device);
        let uk = unary_kernel(&dev.device);
        let ck = compare_kernel(&dev.device);
        let wk = where_kernel(&dev.device);
        let fk = fma_kernel(&dev.device);
        let abk = activation_backward_kernel(&dev.device);
        // One dispatch per compute pass ⇒ WebGPU inserts a full memory barrier
        // between every op, eliminating intra-pass hazard races on aliased arena
        // slots (the source of nondeterministic deep-graph results on wgpu).
        let one_op_per_pass = rlx_ir::env::flag("RLX_WGPU_ONE_OP_PER_PASS");
        let mut step_i = 0;
        let mut gpu_bi = 0usize;
        let mut fft_i = 0usize;
        let mut host_cache = rlx_gpu_host::HostTensorCache::new();
        while step_i < self.schedule.len() {
            // Host→Host streaks: skip empty compute encodes/submits between
            // HostOps (was dominating Kitten discrete wall time alongside D2H).
            let starting_on_host = step_runs_on_host(&self.schedule[step_i])
                || step_is_tail_host(&self.schedule[step_i]);
            let mut pass_dispatched = false;
            if !starting_on_host {
                let mut enc = dev
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("rlx-wgpu run"),
                    });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rlx-wgpu compute pass"),
                        timestamp_writes: None,
                    });
                    while step_i < self.schedule.len() {
                        if step_is_tail_host(&self.schedule[step_i]) {
                            step_i += 1;
                            continue;
                        }
                        if self.static_once_done && self.static_once_steps.contains(&step_i) {
                            let step = &self.schedule[step_i];
                            if step_runs_on_host(step) {
                                step_i += 1;
                                continue;
                            }
                            if !matches!(step, Step::FftGpu { .. }) {
                                gpu_bi += 1;
                            }
                            step_i += 1;
                            continue;
                        }
                        if step_runs_on_host(&self.schedule[step_i]) {
                            break;
                        }
                        // Vulkan/DX12: end the pass after unary/cast so f32→f16
                        // mirrors are visible to the next step. Only split once
                        // we've dispatched in *this* pass — otherwise the step that
                        // needs the flush would never run (infinite empty passes).
                        if pass_dispatched
                            && step_i > 0
                            && step_needs_pass_flush(
                                &self.schedule[step_i],
                                &self.schedule[step_i - 1],
                            )
                        {
                            break;
                        }
                        let step = &self.schedule[step_i];
                        // PLAN L3: per-step Perfetto trace span; no-op when
                        // env var RLX_TRACE_PERFETTO unset.
                        let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "wgpu");
                        if std::env::var("RLX_DBG_STEP").is_ok() {
                            eprintln!("[wgpu-step] {}", step_name(step));
                        }
                        match step {
                            Step::CastF32ToF16 { params } => {
                                // Pre-pass for matmul_coop16: mirror f32 arena
                                // region into f16 shadow buffer so the matmul
                                // kernel can read A as f16. One thread per
                                // element; 64-thread workgroups.
                                if let Some(cast_k) = mm_cast {
                                    pass.set_pipeline(&cast_k.pipeline);
                                    pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                    let (gx, gy, gz) = dispatch_dims(params.len, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                }
                            }
                            Step::Matmul {
                                m,
                                n,
                                batch,
                                b_off_f32,
                                b_is_param,
                                compute_precision,
                                ..
                            } =>
                            // The dispatch branches below use a chain of
                            // `is_some() && …unwrap()` to pick a pipeline
                            // because each variant cares about a different
                            // Option<Pipeline>. `if let Some(p) = …` chains
                            // would require nesting per variant; the flat
                            // form is the readable shape here.
                            {
                                #[allow(clippy::unnecessary_unwrap)]
                                // Safe at any batch (see safe_for_active_extent
                                // comment); scale m, output rows past m_s per
                                // batch retain prior values via c_batch_stride.
                                let m_s = scale(*m);
                                if m_s == 0 {
                                    continue;
                                }
                                let coop_f16_wide = mm_coop_f16_vk.is_some()
                                    && *compute_precision == MatmulCompute::CoopF16Vk
                                    && crate::coop_f16_vk::use_wide_matmul(
                                        *b_off_f32,
                                        *n,
                                        &self.coop_f16_b_param,
                                        &self.coop_f16_vk_wide_b,
                                    );
                                pass.set_bind_group(
                                    0,
                                    coop_f16_vk_bind_group(self, gpu_bi, coop_f16_wide),
                                    &[],
                                );
                                // Kernel selection priority:
                                //   1. compute_precision == F16 + b_is_param +
                                //      SHADER_F16 → matmul_f16_compute
                                //      (f16 multiply, f32 acc — 2× ALU on Apple)
                                //   2. legacy RLX_WGPU_F16_WEIGHTS opt-in →
                                //      matmul_f16w (storage-only f16; experimental,
                                //      currently regresses on Apple)
                                //   3. wide-N (m≥32, n≥64)   → matmul_wide
                                //   4. otherwise            → matmul (small/skinny)
                                let f16w_opt_in = rlx_ir::env::flag("RLX_WGPU_F16_WEIGHTS");
                                if let Some(coop) = mm_coop.as_ref()
                                    && *b_is_param
                                    && *compute_precision == MatmulCompute::Coop16
                                {
                                    // Hardware GEMM via simdgroup_matrix /
                                    // KHR_cooperative_matrix. 32×32 output tile
                                    // per workgroup (16 hardware-GEMM ops with
                                    // shared A/B loads). Caller guaranteed m, n,
                                    // k are multiples of 32/32/8.
                                    pass.set_pipeline(&coop.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                } else if mm_coop_f16_vk.is_some()
                                    && *compute_precision == MatmulCompute::CoopF16Vk
                                {
                                    if coop_f16_wide {
                                        dispatch_wide_f32_matmul(
                                            &mut pass,
                                            mm_w_active,
                                            mm_k,
                                            m_s,
                                            *n,
                                            *batch,
                                        );
                                    } else {
                                        let n_eff = scale(*n);
                                        let coop_vk = matmul_coop_f16_vulkan_active_kernel(
                                            &dev.device,
                                            n_eff,
                                        )
                                        .expect("coop f16 vk kernel missing");
                                        pass.set_pipeline(&coop_vk.pipeline);
                                        pass.dispatch_workgroups(
                                            m_s.div_ceil(16),
                                            n.div_ceil(16),
                                            *batch,
                                        );
                                    }
                                } else if let Some(coop_f32) = mm_coop_f32.as_ref()
                                    && *b_is_param
                                    && *compute_precision == MatmulCompute::CoopF32
                                {
                                    // CoopF32: Metal uses 32×32 simdgroup tiles;
                                    // Vulkan uses 8×8 coopLoadT portable kernel.
                                    pass.set_pipeline(&coop_f32.pipeline);
                                    let backend = wgpu_device()
                                        .map(|d| d.backend)
                                        .unwrap_or(wgpu::Backend::Noop);
                                    let (gx, gy) = if backend == wgpu::Backend::Metal {
                                        (n.div_ceil(32), m_s.div_ceil(32))
                                    } else {
                                        (m_s.div_ceil(8), n.div_ceil(8))
                                    };
                                    pass.dispatch_workgroups(gx, gy, *batch);
                                } else if let Some(f16c) = mm_f16c.as_ref()
                                    && *b_is_param
                                    && *compute_precision == MatmulCompute::F16
                                {
                                    pass.set_pipeline(&f16c.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                } else if let Some(f16w) = mm_f16w.as_ref()
                                    && *b_is_param
                                    && f16w_opt_in
                                {
                                    pass.set_pipeline(&f16w.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                } else if m_s >= 32 && *n >= 64 {
                                    pass.set_pipeline(&mm_w_active.pipeline);
                                    let backend = wgpu_device()
                                        .map(|d| d.backend)
                                        .unwrap_or(wgpu::Backend::Noop);
                                    let (gx, gy) = if matches!(
                                        backend,
                                        wgpu::Backend::Vulkan | wgpu::Backend::Dx12
                                    ) {
                                        (n.div_ceil(64), m_s.div_ceil(64))
                                    } else {
                                        (n.div_ceil(64), m_s.div_ceil(32))
                                    };
                                    pass.dispatch_workgroups(gx, gy, *batch);
                                } else {
                                    pass.set_pipeline(&mm_k.pipeline);
                                    pass.dispatch_workgroups(
                                        n.div_ceil(32),
                                        m_s.div_ceil(32),
                                        *batch,
                                    );
                                }
                            }
                            Step::Binary { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                pass.set_pipeline(&bk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Compare { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Unary { params, f16_mirror } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                if *f16_mirror {
                                    if let Some(uk_f16) = unary_f16_mirror_kernel(&dev.device) {
                                        pass.set_pipeline(&uk_f16.pipeline);
                                    } else {
                                        pass.set_pipeline(&uk.pipeline);
                                    }
                                } else {
                                    pass.set_pipeline(&uk.pipeline);
                                }
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Where { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                pass.set_pipeline(&wk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Fma { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                pass.set_pipeline(&fk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ReluBackward { params } | Step::ActivationBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                pass.set_pipeline(&abk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Reduce { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let rk = reduce_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total_out = outer_s.saturating_mul(params.inner);
                                if params.reduce_dim <= 64 {
                                    // Fast path: 1 thread per output cell.
                                    let (gx, gy, gz) = dispatch_dims(total_out, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                } else {
                                    // Tree-reduce path: 1 workgroup (64
                                    // threads) per output cell, parallel
                                    // reduction with shared scratch.
                                    let (gx, gy, gz) = dispatch_dims(total_out, 1);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                }
                            }
                            Step::Softmax { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let sk = softmax_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::SoftmaxCrossEntropy { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let sk = softmax_cross_entropy_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::SoftmaxCrossEntropyWithLogits { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let sk = softmax_cross_entropy_with_logits_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::SoftmaxCrossEntropyBackward { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let sk = softmax_cross_entropy_backward_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::LayerNorm { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let lk = layernorm_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::RmsNormBackwardInput { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let rk = rms_norm_backward_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(outer_s, 1, 1);
                            }
                            Step::RmsNormBackwardGamma { params }
                            | Step::RmsNormBackwardBeta { params } => {
                                if params.inner == 0 {
                                    continue;
                                }
                                let rk = rms_norm_backward_param_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::LayerNormBackwardInput { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let lk = layer_norm_backward_input_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(outer_s, 1, 1);
                            }
                            Step::LayerNormBackwardGammaPartial {
                                params,
                                num_workgroups,
                            } => {
                                if params.inner == 0 || *num_workgroups == 0 {
                                    continue;
                                }
                                let lk = layer_norm_backward_gamma_partial_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(*num_workgroups, 1, 1);
                            }
                            Step::LayerNormBackwardGammaReduce { params } => {
                                if params.inner == 0 {
                                    continue;
                                }
                                let lk = layer_norm_backward_gamma_reduce_kernel(&dev.device);
                                pass.set_pipeline(&lk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::CumsumBackward { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let ck = cumsum_backward_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::RopeBackward { params } => {
                                let seq_s = scale(params.seq);
                                if seq_s == 0 {
                                    continue;
                                }
                                let rk = rope_backward_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * seq_s * params.hidden;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatherBackward { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let total = outer_s * params.axis_dim * params.trailing;
                                if total > 0 {
                                    let zk = gather_backward_zero_kernel(&dev.device);
                                    pass.set_pipeline(&zk.pipeline);
                                    pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                    let (gx, _, _) = dispatch_dims(total, 256);
                                    pass.dispatch_workgroups(gx, 1, 1);
                                }
                                let ak = gather_backward_acc_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(outer_s, 1, 1);
                            }
                            Step::Cumsum { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let ck2 = cumsum_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::CumScan { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let ck2 = cum_scan_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FftGpu {
                                src_off,
                                dst_off,
                                outer,
                                n,
                                inverse,
                                norm_scale,
                            } => {
                                let res = &self.fft_gpu_steps[fft_i];
                                fft_i += 1;
                                crate::fft_dispatch::dispatch_fft_gpu_in_pass(
                                    &dev.device,
                                    &dev.queue,
                                    &mut pass,
                                    res,
                                    *src_off,
                                    *dst_off,
                                    *outer,
                                    *n,
                                    *inverse != 0,
                                    *norm_scale,
                                );
                            }
                            Step::Copy { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let ck2 = copy_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Cast { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let cast_k = cast_kernel(&dev.device);
                                pass.set_pipeline(&cast_k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ComplexCast { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let k = complex_cast_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::BinaryC64 { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let k = binary_c64_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ComplexNormSq { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let k = complex_norm_sq_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ComplexNormSqBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let k = complex_norm_sq_backward_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ConjugateC64 { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let k = conjugate_c64_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FftButterflyStage { params } => {
                                let batch_s = scale(params.batch);
                                let n = batch_s.saturating_mul(params.half);
                                if n == 0 {
                                    continue;
                                }
                                let k = fft_butterfly_stage_kernel(&dev.device);
                                pass.set_pipeline(&k.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::BufferCopy { .. } => {
                                // Host step: `copy_buffer_to_buffer` runs outside compute passes.
                            }
                            Step::ElementwiseRegion { params } => {
                                let len_s = scale(params.len);
                                if len_s == 0 {
                                    continue;
                                }
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                if params.prologue == rlx_ir::REGION_PROLOGUE_RESIZE_NEAREST_2X_NCHW
                                {
                                    let ek = elementwise_region_spatial_kernel(&dev.device);
                                    pass.set_pipeline(&ek.pipeline);
                                    let (gx, gy, gz) = dispatch_prologue_nchw(
                                        params.out_w,
                                        params.out_h,
                                        params.out_n * params.out_c,
                                    );
                                    pass.dispatch_workgroups(gx, gy, gz);
                                } else {
                                    let ek = elementwise_region_kernel(&dev.device);
                                    pass.set_pipeline(&ek.pipeline);
                                    let (gx, gy, gz) = dispatch_dims(len_s, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                }
                            }
                            Step::BatchElementwiseRegion { params } => {
                                let slice_len_s = scale(params.slice_len);
                                let num_batch_s = scale(params.num_batch);
                                if slice_len_s == 0 || num_batch_s == 0 {
                                    continue;
                                }
                                let ek = batch_elementwise_region_kernel(&dev.device);
                                pass.set_pipeline(&ek.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, _) = dispatch_dims(slice_len_s, 64);
                                pass.dispatch_workgroups(gx, gy, num_batch_s);
                            }
                            Step::Transpose { params, .. } => {
                                // Compute scaled grid count to match the
                                // uniform's scaled out_total when bucket axis
                                // is outermost.
                                let total_s =
                                    if params.bucket_outermost == 1 && params.out_dim_0 > 0 {
                                        let scaled_d0 = scale(params.out_dim_0);
                                        let inner = params.out_total / params.out_dim_0;
                                        scaled_d0 * inner
                                    } else {
                                        params.out_total
                                    };
                                if total_s == 0 {
                                    continue;
                                }
                                let tk = transpose_kernel(&dev.device);
                                pass.set_pipeline(&tk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Narrow { params } => {
                                let total_s = scale(params.total);
                                if total_s == 0 {
                                    continue;
                                }
                                let nk = narrow_kernel(&dev.device);
                                pass.set_pipeline(&nk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Concat { params } => {
                                let total_s = scale(params.total);
                                if total_s == 0 {
                                    continue;
                                }
                                let cck = concat_kernel(&dev.device);
                                pass.set_pipeline(&cck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Gather { params } => {
                                let n_out_s = scale(params.n_out);
                                if n_out_s == 0 {
                                    continue;
                                }
                                let gk = gather_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_out_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatherAxis { params } => {
                                // Trim only the leading (outer) dim; see the
                                // uniform-write path above.
                                let total_s =
                                    scale(params.outer) * params.num_idx * params.trailing;
                                if total_s == 0 {
                                    continue;
                                }
                                let gk = gather_axis_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Attention { params, .. } => {
                                // Scale seq_q for grid dim; per-head strides
                                // come from seq_q_stride / seq_k_stride (full
                                // extent) inside the WGSL.
                                let seq_q_s = scale(params.seq_q);
                                if seq_q_s == 0 {
                                    continue;
                                }
                                let ak = attention_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * params.heads * seq_q_s;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::AttentionBackward { params, .. } => {
                                let axis = if params.wrt == 0 {
                                    params.seq_q
                                } else {
                                    params.seq_k
                                };
                                let axis_s = scale(axis);
                                if axis_s == 0 {
                                    continue;
                                }
                                let ak = attention_bwd_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * params.heads * axis_s;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Rope { params } => {
                                // Multi-batch via stride-field WGSL fix:
                                // iterate `batch * scaled_seq * last_dim` items.
                                let s_active = scale(params.seq);
                                let total_s = params.batch * s_active * params.last_dim;
                                if total_s == 0 {
                                    continue;
                                }
                                let rk = rope_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Expand { params, .. } => {
                                let total_s =
                                    if params.bucket_outermost == 1 && params.out_dim_0 > 0 {
                                        let scaled_d0 = scale(params.out_dim_0);
                                        let inner = params.out_total / params.out_dim_0;
                                        scaled_d0 * inner
                                    } else {
                                        params.out_total
                                    };
                                if total_s == 0 {
                                    continue;
                                }
                                let ek = expand_kernel(&dev.device);
                                pass.set_pipeline(&ek.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Argmax { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let amk = argmax_kernel(&dev.device);
                                pass.set_pipeline(&amk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Pool2d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = pool2d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::MaxPool2dBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = maxpool2d_backward_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let gx = params.w.div_ceil(8);
                                let gy = params.h.div_ceil(8);
                                let gz = n_s * params.c;
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::MaxPool3dBackward { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = maxpool3d_backward_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c * params.d * params.h * params.w;
                                let (gx, gy, gz) = dispatch_dims(total, 256);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv3dBackwardInput { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = conv3d_backward_input_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c_in * params.d * params.h * params.w;
                                let (gx, gy, gz) = dispatch_dims(total, 256);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv3dBackwardWeight { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = conv3d_backward_weight_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let c_in_per_g = params.c_in / params.groups.max(1);
                                let total =
                                    params.c_out * c_in_per_g * params.kd * params.kh * params.kw;
                                let (gx, gy, gz) = dispatch_dims(total, 256);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GroupNormBackwardInput { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = group_norm_backward_input_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(n_s * params.num_groups, 1, 1);
                            }
                            Step::GroupNormBackwardGamma { .. } => {
                                let pk = group_norm_backward_gamma_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::GroupNormBackwardBeta { .. } => {
                                let pk = group_norm_backward_beta_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            Step::AxialRope2d { params } => {
                                let batch_s = scale(params.batch);
                                if batch_s == 0 {
                                    continue;
                                }
                                let total = batch_s * params.seq * params.hidden;
                                let pk = axial_rope2d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FakeQuantizeFixed { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = fake_quantize_fixed_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FakeQuantizePerBatch { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 || params.chan_dim == 0 {
                                    continue;
                                }
                                let pk = fake_quantize_perbatch_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(params.chan_dim, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv2d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let ck2 = conv2d_kernel(&dev.device);
                                pass.set_pipeline(&ck2.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // conv2d.wgsl tiles `CONV2D_TILE` output spatial
                                // positions per thread (const must match the kernel).
                                let spatial = params.h_out * params.w_out;
                                let sp_tiles = spatial.div_ceil(CONV2D_TILE);
                                let total = n_s * params.c_out * sp_tiles;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv2dTiled { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let ck = conv1d_tiled_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // conv1d_tiled.wgsl: one thread per output element
                                // (c_out * l_out), N == 1, w_out == 1.
                                let total = n_s * params.c_out * params.h_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Im2ColGpu { params } => {
                                // One thread per col element (k_total * spatial);
                                // the following Step::Matmul consumes the col matrix.
                                let imk = im2col2d_kernel(&dev.device);
                                pass.set_pipeline(&imk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.k_total.saturating_mul(params.spatial);
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Pool1d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = pool1d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c * params.l_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Pool3d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let pk = pool3d_kernel(&dev.device);
                                pass.set_pipeline(&pk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total =
                                    n_s * params.c * params.d_out * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv1d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let ck = conv1d_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = n_s * params.c_out * params.l_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Conv3d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let ck = conv3d_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total =
                                    n_s * params.c_out * params.d_out * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ConvTranspose3d { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let ck = conv_transpose3d_kernel(&dev.device);
                                pass.set_pipeline(&ck.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total =
                                    n_s * params.c_out * params.d_out * params.h_out * params.w_out;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::ScatterAdd { params } => {
                                let sk = scatter_add_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // Phase 0 zeros the FULL output (preserves
                                // accumulator semantics). Phase 1 scatters first
                                // num_updates_active updates only; serial single
                                // workgroup either way (atomic CAS unsupported in
                                // naga's MSL emitter — see scatter_add.wgsl).
                                if params.op == 0 {
                                    let (gx, gy, gz) = dispatch_dims(params.out_total, 64);
                                    pass.dispatch_workgroups(gx, gy, gz);
                                } else {
                                    pass.dispatch_workgroups(1, 1, 1);
                                }
                            }
                            Step::TopK { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let tk = topk_kernel(&dev.device);
                                pass.set_pipeline(&tk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::WelchPeaksGpu { params } => {
                                let batch_s = scale(params.welch_batch);
                                if batch_s == 0 {
                                    continue;
                                }
                                let wk = welch_peaks_gpu_kernel(&dev.device);
                                pass.set_pipeline(&wk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(batch_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::UmapKnn { params } => {
                                let n_s = scale(params.n);
                                if n_s == 0 {
                                    continue;
                                }
                                let uk = umap_knn_kernel(&dev.device);
                                pass.set_pipeline(&uk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(n_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::WgpuGpuKernel { name, workgroups } => {
                                // Raw-GPU custom op: fetch the cached pipeline by name
                                // and dispatch against the compile-time bind group.
                                let gk = crate::wgpu_gpu_custom::lookup(name).expect(
                                "WgpuGpuKernel vanished from the registry between compile and run",
                            );
                                let kernel = crate::wgpu_gpu_custom::get_or_build_pipeline(
                                    &dev.device,
                                    &*gk,
                                );
                                pass.set_pipeline(&kernel.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = *workgroups;
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GroupedMatmul { params } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    continue;
                                }
                                let gk = grouped_matmul_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.n.div_ceil(8), m_s.div_ceil(8), 1);
                            }
                            Step::Sample { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let sk = sample_kernel(&dev.device);
                                pass.set_pipeline(&sk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::SelectiveScan { params } => {
                                // Predicate-gated to batch=1; the seq scaling
                                // happens inside the kernel (uniform sees scaled
                                // seq). Dispatch grid here is per-(batch, hidden);
                                // unaffected by seq scaling.
                                let ssk = selective_scan_kernel(&dev.device);
                                pass.set_pipeline(&ssk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * params.hidden;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatedDeltaNet {
                                params,
                                use_gpu: true,
                                ..
                            } => {
                                // One workgroup per (batch, head); workgroup_size=128.
                                let gk = gated_delta_net_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.batch * params.heads, 1, 1);
                            }
                            Step::Mamba2 { params } => {
                                let mk = mamba2_kernel(&dev.device);
                                pass.set_pipeline(&mk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let total = params.batch * params.heads * params.head_dim;
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::Gru { params } => {
                                // One workgroup per batch item (workgroup_size=256).
                                let gk = gru_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.batch, 1, 1);
                            }
                            Step::Rnn { params } => {
                                let rk = rnn_kernel(&dev.device);
                                pass.set_pipeline(&rk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.batch, 1, 1);
                            }
                            Step::DequantMatmul { params } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    continue;
                                }
                                let dk = dequant_matmul_kernel(&dev.device);
                                pass.set_pipeline(&dk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(params.n.div_ceil(8), m_s.div_ceil(8), 1);
                            }
                            Step::DequantMatmulMlx { params } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    continue;
                                }
                                let dk = dequant_matmul_mlx_kernel(&dev.device);
                                pass.set_pipeline(&dk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                // One workgroup per (col, row_tile); local size 256
                                // splits K and stages X in workgroup memory.
                                let n_row_tiles = m_s.div_ceil(8);
                                pass.dispatch_workgroups(params.n * n_row_tiles, 1, 1);
                            }
                            Step::FusedResidualLn { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let frk = fused_residual_ln_kernel(&dev.device);
                                pass.set_pipeline(&frk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FusedResidualLnTee { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let frtk = fused_residual_ln_tee_kernel(&dev.device);
                                pass.set_pipeline(&frtk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::FusedResidualRmsNorm { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let frk = fused_residual_rms_norm_kernel(&dev.device);
                                pass.set_pipeline(&frk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::AdaLayerNorm { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let ak = ada_layer_norm_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(outer_s, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::GatedResidual { params } => {
                                let outer_s = scale(params.outer);
                                if outer_s == 0 {
                                    continue;
                                }
                                let total = outer_s.saturating_mul(params.inner);
                                let gk = gated_residual_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                let (gx, gy, gz) = dispatch_dims(total, 64);
                                pass.dispatch_workgroups(gx, gy, gz);
                            }
                            Step::AdaLayerNormBackward { params } => {
                                let mod_rows_s = scale(params.mod_rows);
                                if mod_rows_s == 0 {
                                    continue;
                                }
                                let ak = ada_layer_norm_backward_kernel(&dev.device);
                                pass.set_pipeline(&ak.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(mod_rows_s, 1, 1);
                            }
                            Step::GatedResidualBackward { params } => {
                                let mod_rows_s = scale(params.mod_rows);
                                if mod_rows_s == 0 {
                                    continue;
                                }
                                let gk = gated_residual_backward_kernel(&dev.device);
                                pass.set_pipeline(&gk.pipeline);
                                pass.set_bind_group(0, &self.bind_groups[gpu_bi], &[]);
                                pass.dispatch_workgroups(mod_rows_s, 1, 1);
                            }
                            Step::MatmulQkv { params, kind } => {
                                let m_s = scale(params.m);
                                if m_s == 0 {
                                    continue;
                                }
                                let qkv_coop_wide = matches!(kind, MatmulQkvKind::CoopF16Vk)
                                    && crate::coop_f16_vk::use_wide_matmul(
                                        params.b_off,
                                        params.n,
                                        &self.coop_f16_b_param,
                                        &self.coop_f16_vk_wide_b,
                                    );
                                pass.set_bind_group(
                                    0,
                                    coop_f16_vk_bind_group(self, gpu_bi, qkv_coop_wide),
                                    &[],
                                );
                                match kind {
                                    MatmulQkvKind::CoopF16Vk => {
                                        if qkv_coop_wide {
                                            pass.set_pipeline(
                                                &matmul_qkv_kernel(&dev.device).pipeline,
                                            );
                                            pass.dispatch_workgroups(
                                                params.n.div_ceil(32),
                                                m_s.div_ceil(32),
                                                1,
                                            );
                                        } else {
                                            let n_eff = scale(params.n);
                                            let mqk = matmul_qkv_coop_f16_vk_active_kernel(
                                                &dev.device,
                                                n_eff,
                                            )
                                            .expect("coop f16 matmul_qkv kernel missing");
                                            pass.set_pipeline(&mqk.pipeline);
                                            pass.dispatch_workgroups(
                                                m_s.div_ceil(16),
                                                params.n.div_ceil(16),
                                                1,
                                            );
                                        }
                                    }
                                    MatmulQkvKind::CoopF32 => {
                                        pass.set_pipeline(
                                            &matmul_qkv_coop_f32_kernel(&dev.device)
                                                .expect("coop matmul_qkv kernel missing")
                                                .pipeline,
                                        );
                                        pass.dispatch_workgroups(
                                            params.n.div_ceil(32),
                                            m_s.div_ceil(32),
                                            1,
                                        );
                                    }
                                    MatmulQkvKind::F32 => {
                                        pass.set_pipeline(&matmul_qkv_kernel(&dev.device).pipeline);
                                        pass.dispatch_workgroups(
                                            params.n.div_ceil(32),
                                            m_s.div_ceil(32),
                                            1,
                                        );
                                    }
                                }
                            }
                            Step::GatherSplit { .. }
                            | Step::DequantMatmulGguf { .. }
                            | Step::DequantMatmulInt8Host { .. }
                            | Step::DequantMatmulMlxHost { .. }
                            | Step::Conv2dHost { .. }
                            | Step::DequantGroupedMatmulGguf { .. }
                            | Step::DequantGroupedMatmulMlxHost { .. }
                            | Step::GatedDeltaNet { use_gpu: false, .. }
                            | Step::Lstm { .. }
                            | Step::ConvTranspose2d { .. }
                            | Step::ConvTranspose3dHost { .. }
                            | Step::GroupNormHost { .. }
                            | Step::LayerNorm2dHost { .. }
                            | Step::ResizeNearest2xHost { .. }
                            | Step::ReverseHost { .. }
                            | Step::ArgReduceHost { .. }
                            | Step::AxialRope2dHost { .. }
                            | Step::GruHost { .. }
                            | Step::RnnHost { .. }
                            | Step::Llada2GroupLimitedGate { .. }
                            | Step::UmapKnnHost { .. }
                            | Step::MsDeformAttnHost { .. }
                            | Step::CollectiveHost { .. }
                            | Step::CustomHost { .. }
                            | Step::FftHost { .. }
                            | Step::ScanHost { .. }
                            | Step::HostOp { .. }
                            | Step::CpuIndexing { .. }
                            | Step::ConcatHost { .. }
                            | Step::ConcatHostPieces { .. }
                            | Step::TransposeHost { .. }
                            | Step::NarrowHost { .. }
                            | Step::ExpandHost { .. }
                            | Step::SpdHost { .. }
                            | Step::Im2ColHost { .. }
                            | Step::Conv2dBackwardWeightHost { .. }
                            | Step::Conv2dBackwardInputHost { .. }
                            | Step::RngNormalHost { .. }
                            | Step::RngUniformHost { .. }
                            | Step::WelchPeaksHost { .. }
                            | Step::LogMelHost { .. }
                            | Step::LogMelBackwardHost { .. } => {}
                            #[cfg(feature = "splat")]
                            Step::GaussianSplatRender { .. }
                            | Step::GaussianSplatRenderBackward { .. }
                            | Step::GaussianSplatPrepare { .. }
                            | Step::GaussianSplatRasterize { .. } => {}
                        }
                        if !matches!(step, Step::FftGpu { .. }) {
                            gpu_bi += 1;
                        }
                        step_i += 1;
                        pass_dispatched = true;
                        if one_op_per_pass {
                            break; // end the pass after this op ⇒ barrier before next
                        }
                    }
                }
                let needs_f16_drain = step_i < self.schedule.len()
                    && !step_runs_on_host(&self.schedule[step_i])
                    && step_i > 0
                    && step_needs_pass_flush(&self.schedule[step_i], &self.schedule[step_i - 1]);
                let gpu_schedule_done = step_i >= self.schedule.len();
                let skip_readback =
                    rlx_ir::env::flag("RLX_BENCH_DISPATCH_ONLY") || self.dispatch_only;
                let defer_tail = gpu_schedule_done && self.schedule.iter().any(step_is_tail_host);
                let mut fused_readback: Option<(
                    ReadbackLayout,
                    std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
                    Vec<usize>,
                )> = None;
                if gpu_schedule_done && !skip_readback && !defer_tail {
                    if !self.gpu_handle_feeds.is_empty() {
                        self.propagate_gpu_handle_feeds_on_gpu(dev, &mut enc);
                    }
                    let plan = self.readback_plan();
                    let out_ids_all: Vec<_> = self.graph.outputs.clone();
                    let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
                    let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
                    if use_tiny_readback(&layout, out_ids.len()) && plan == vec![0] {
                        if self.tiny_readback.is_none() {
                            self.tiny_readback = Some(TinyReadbackStaging::new(&dev.device));
                        }
                        let tiny = self.tiny_readback.as_ref().expect("tiny readback");
                        encode_readback_copies(
                            &mut enc,
                            &self.arena,
                            tiny.buffer(),
                            &out_ids,
                            &layout,
                        );
                        let map_rx = schedule_readback_map(&mut enc, tiny.buffer(), &layout);
                        let sub = dev.queue.submit(std::iter::once(enc.finish()));
                        wait_readback_map(&dev.device, sub, &map_rx, layout.total_bytes);
                        map_rx.recv().unwrap().unwrap();
                        return self.pack_readback_outputs(
                            &plan,
                            vec![decode_tiny_mapped_f32(tiny.buffer(), layout.total_bytes)],
                        );
                    }
                    ReadbackStaging::prepare(
                        &dev.device,
                        &mut self.readback_staging,
                        layout.total_bytes,
                    );
                    if let Some(staging) = self.readback_staging.as_ref() {
                        encode_readback_copies(
                            &mut enc,
                            &self.arena,
                            staging.buffer(),
                            &out_ids,
                            &layout,
                        );
                        let map_rx = schedule_readback_map(&mut enc, staging.buffer(), &layout);
                        fused_readback = Some((layout, map_rx, plan));
                    }
                }
                let main_submission = dev.queue.submit(std::iter::once(enc.finish()));
                if defer_tail {
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    self.run_tail_host_audio_ops(dev);
                    if !skip_readback {
                        let mut rb_enc =
                            dev.device
                                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: Some("rlx-wgpu readback after tail-host"),
                                });
                        if !self.gpu_handle_feeds.is_empty() {
                            self.propagate_gpu_handle_feeds_on_gpu(dev, &mut rb_enc);
                        }
                        let plan = self.readback_plan();
                        let out_ids_all: Vec<_> = self.graph.outputs.clone();
                        let out_ids: Vec<_> = plan.iter().map(|&i| out_ids_all[i]).collect();
                        let layout = ReadbackLayout::for_nodes(&self.arena, &out_ids);
                        if use_tiny_readback(&layout, out_ids.len()) && plan == vec![0] {
                            if self.tiny_readback.is_none() {
                                self.tiny_readback = Some(TinyReadbackStaging::new(&dev.device));
                            }
                            let tiny = self.tiny_readback.as_ref().expect("tiny readback");
                            encode_readback_copies(
                                &mut rb_enc,
                                &self.arena,
                                tiny.buffer(),
                                &out_ids,
                                &layout,
                            );
                            let map_rx = schedule_readback_map(&mut rb_enc, tiny.buffer(), &layout);
                            let sub = dev.queue.submit(std::iter::once(rb_enc.finish()));
                            wait_readback_map(&dev.device, sub, &map_rx, layout.total_bytes);
                            map_rx.recv().unwrap().unwrap();
                            return self.pack_readback_outputs(
                                &plan,
                                vec![decode_tiny_mapped_f32(tiny.buffer(), layout.total_bytes)],
                            );
                        }
                        ReadbackStaging::prepare(
                            &dev.device,
                            &mut self.readback_staging,
                            layout.total_bytes,
                        );
                        if let Some(staging) = self.readback_staging.as_ref() {
                            encode_readback_copies(
                                &mut rb_enc,
                                &self.arena,
                                staging.buffer(),
                                &out_ids,
                                &layout,
                            );
                            let map_rx =
                                schedule_readback_map(&mut rb_enc, staging.buffer(), &layout);
                            let sub = dev.queue.submit(std::iter::once(rb_enc.finish()));
                            wait_readback_map(&dev.device, sub, &map_rx, layout.total_bytes);
                            map_rx.recv().unwrap().unwrap();
                            self.dump_node_stats_if_requested(dev);
                            let partial = decode_mapped_readback_f32(staging.buffer(), &layout);
                            return self.pack_readback_outputs(&plan, partial);
                        }
                    }
                }
                if needs_f16_drain {
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                }
                let need_host_sync =
                    step_i < self.schedule.len() && step_runs_on_host(&self.schedule[step_i]);
                if need_host_sync {
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    // Only invalidate after real GPU work. Host→Host chains share
                    // one outer loop iteration's empty encode; clearing there made
                    // HostTensorCache always miss (full D2H every HostOp).
                    if pass_dispatched {
                        host_cache.clear();
                    }
                }
                if gpu_schedule_done {
                    if skip_readback || defer_tail {
                        if !self.static_once_done && !self.static_once_steps.is_empty() {
                            self.static_once_done = true;
                        }
                        return self
                            .graph
                            .outputs
                            .iter()
                            .map(|&id| {
                                let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                                vec![0.0; n]
                            })
                            .collect();
                    }
                    if let (Some((layout, map_rx, plan)), Some(staging)) =
                        (fused_readback, self.readback_staging.as_ref())
                    {
                        wait_readback_map(
                            &dev.device,
                            main_submission,
                            &map_rx,
                            layout.total_bytes,
                        );
                        map_rx.recv().unwrap().unwrap();
                        self.dump_node_stats_if_requested(dev);
                        let partial = decode_mapped_readback_f32(staging.buffer(), &layout);
                        if !self.static_once_done && !self.static_once_steps.is_empty() {
                            self.static_once_done = true;
                        }
                        return self.pack_readback_outputs(&plan, partial);
                    }
                    break;
                }
            } // !starting_on_host — GPU encode/submit/readback path

            // Skip already-baked static weight packing (Concat/Expand of Params).
            while step_i < self.schedule.len()
                && self.static_once_done
                && self.static_once_steps.contains(&step_i)
                && step_runs_on_host(&self.schedule[step_i])
            {
                step_i += 1;
            }
            if step_i >= self.schedule.len() {
                break;
            }
            if !step_runs_on_host(&self.schedule[step_i]) {
                continue;
            }
            // Deferred HostOp/Conv/structure outputs live only in the mirror
            // until a device-reading host step or GPU pass needs them.
            if host_cache.has_deferred_writes()
                && !matches!(
                    &self.schedule[step_i],
                    Step::HostOp { .. }
                        | Step::Conv2dHost { .. }
                        | Step::ExpandHost { .. }
                        | Step::NarrowHost { .. }
                        | Step::TransposeHost { .. }
                        | Step::ConcatHost { .. }
                        | Step::BufferCopy { .. }
                )
            {
                let mut a = crate::host_stage::WgpuArena {
                    arena: &self.arena,
                    device: &dev.device,
                    queue: &dev.queue,
                    size_bytes: 0,
                };
                host_cache.flush_to_device(&mut a);
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            }
            match &self.schedule[step_i] {
                Step::BufferCopy {
                    src_byte_off,
                    dst_byte_off,
                    bytes,
                } => {
                    // wgpu forbids `copy_buffer_to_buffer` on the same buffer;
                    // use the generic copy compute kernel instead.
                    let src = *src_byte_off;
                    let dst = *dst_byte_off;
                    let nbytes = *bytes as u64;
                    let elems = (nbytes / 4).max(1) as u32;
                    let max_binding = dev.device.limits().max_storage_buffer_binding_size;
                    // wgpu forbids `copy_buffer_to_buffer` on the same buffer;
                    // use the generic copy compute kernel instead.
                    // Weight-tagged sources live in a separate buffer — always
                    // host-round-trip (or distinct-buffer GPU copy). Also when
                    // src/dst are more than `max_binding` apart on a large arena.
                    let src_is_weight = crate::buffer::is_weight_off(src as usize);
                    let dst_is_weight = crate::buffer::is_weight_off(dst as usize);
                    let lo = if src_is_weight || dst_is_weight {
                        0u64
                    } else {
                        src.min(dst)
                    };
                    let hi = if src_is_weight || dst_is_weight {
                        max_binding.saturating_add(1) // force host / resolve_w path
                    } else {
                        src.saturating_add(nbytes).max(dst.saturating_add(nbytes))
                    };
                    // Same for sharded arenas when src/dst live on different stripes.
                    // Virtually-sharded NVIDIA arenas: GPU copy kernels corrupt
                    // Kitten NSF — host those. Unsharded discrete can use the
                    // GPU copy kernel (was blanket-hosted via coop_discrete and
                    // dominated Kitten wall time). Force host with
                    // RLX_WGPU_HOST_BUFFER_COPY=1.
                    let cross_shard =
                        self.arena.is_sharded() && !src_is_weight && !dst_is_weight && {
                            let s = self.arena.shard_size as u64;
                            let src_hi = src.saturating_add(nbytes.saturating_sub(1));
                            let dst_hi = dst.saturating_add(nbytes.saturating_sub(1));
                            src / s != src_hi / s || dst / s != dst_hi / s || src / s != dst / s
                        };
                    if hi.saturating_sub(lo) > max_binding
                        || cross_shard
                        || self.arena.is_sharded()
                        || src_is_weight
                        || dst_is_weight
                        || host_cache.has_deferred_writes()
                        || crate::device::coop_discrete_backend()
                        || rlx_ir::env::flag("RLX_WGPU_HOST_BUFFER_COPY")
                    {
                        let n_f32 = (nbytes as usize) / 4;
                        let src_u = src as usize;
                        let dst_u = dst as usize;
                        let data = if let Some(hit) = host_cache.get_arc_covering(src_u, n_f32) {
                            hit[..n_f32].to_vec()
                        } else {
                            if host_cache.is_dirty(src_u) {
                                let mut a = crate::host_stage::WgpuArena {
                                    arena: &self.arena,
                                    device: &dev.device,
                                    queue: &dev.queue,
                                    size_bytes: 0,
                                };
                                host_cache.flush_offset(&mut a, src_u);
                            }
                            let bytes_host = self.arena.read_bytes_range(
                                &dev.device,
                                &dev.queue,
                                src_u,
                                nbytes as usize,
                            );
                            bytemuck::cast_slice(&bytes_host).to_vec()
                        };
                        let defer = !rlx_ir::env::flag("RLX_WGPU_HOST_EAGER_H2D");
                        if !defer {
                            self.arena.write_bytes_range(
                                &dev.queue,
                                dst_u,
                                bytemuck::cast_slice(data.as_slice()),
                            );
                            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                        }
                        host_cache.insert(dst_u, data, defer);
                        step_i += 1;
                        continue;
                    }
                    // GPU copy — deferred mirrors must be on device first.
                    if host_cache.has_deferred_writes() {
                        let mut a = crate::host_stage::WgpuArena {
                            arena: &self.arena,
                            device: &dev.device,
                            queue: &dev.queue,
                            size_bytes: 0,
                        };
                        host_cache.flush_to_device(&mut a);
                        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    }
                    let mut base = (lo / 256) * 256;
                    // The bind window must cover [base, hi]. `base` is floored to 256B
                    // alignment (so base ≤ lo); measuring the size from `lo` instead of
                    // `base` clips the window when the copy straddles a 256B boundary,
                    // silently dropping the tail (e.g. a 92B Bool→F32 mask copy whose
                    // dst sat just past the window → x_mask stayed 0).
                    let span = hi.saturating_sub(base).max(1);
                    let mut size = span.div_ceil(256) * 256;
                    size = size.max(256).min(max_binding);
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
                        label: Some("rlx-wgpu arena_copy uniform"),
                        size: std::mem::size_of::<CopyParams>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    dev.queue.write_buffer(&u, 0, bytemuck::bytes_of(&p));
                    let bg = bind_arena_window(&dev.device, ck, &self.arena, base, size, &u);
                    let mut enc =
                        dev.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("rlx-wgpu arena_copy"),
                            });
                    {
                        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("rlx-wgpu arena_copy pass"),
                            ..Default::default()
                        });
                        pass.set_pipeline(&ck.pipeline);
                        pass.set_bind_group(0, &bg, &[]);
                        let (gx, gy, gz) = dispatch_dims(elems, 64);
                        pass.dispatch_workgroups(gx, gy, gz);
                    }
                    dev.queue.submit(std::iter::once(enc.finish()));
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    host_cache.invalidate(dst as usize);
                }
                Step::GatherSplit {
                    n_out,
                    n_idx,
                    dim,
                    vocab,
                    table_byte_off,
                    idx_byte_off,
                    out_byte_off,
                } => {
                    run_gather_split(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *n_out,
                        *n_idx,
                        *dim,
                        *vocab,
                        *table_byte_off as usize,
                        *idx_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::DequantMatmulGguf {
                    m,
                    k,
                    n,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    out_byte_off,
                } => {
                    // Active-extent: scale m (seq·batch) like Metal/other matmuls.
                    let mm = scale(*m) as usize;
                    let kk = *k as usize;
                    let nn = *n as usize;
                    // Scratch-free windowed GEMV for Q4_K / Q6_K / Q1_0.
                    // - m==1 (decode): always OK for supported schemes.
                    // - Q1_0 + m>1: tiled GEMM (or per-row GEMV) inside gemv_rows.
                    // - Other schemes + m>1: per-row GEMV only when k is an
                    //   integer number of GGUF blocks — flat pack is block-linear
                    //   over n*k, so unaligned k makes row slices invalid and used
                    //   to force the host D2H/H2D fallback for *all* m>1 (very slow).
                    let use_gemv = if !crate::gguf_gpu::gemv_supports_scheme(*scheme_id) {
                        false
                    } else if mm == 1 || *scheme_id == 24 {
                        true
                    } else {
                        let be =
                            crate::gguf_host::scheme_from_id(*scheme_id).gguf_block_size() as usize;
                        kk.is_multiple_of(be.max(1))
                    };
                    if use_gemv {
                        crate::gguf_gpu::run_dequant_matmul_gguf_gemv_rows(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            mm,
                            kk,
                            nn,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    } else if self.dequant_scratch_off > 0 {
                        crate::gguf_gpu::run_dequant_matmul_gguf_gpu(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            mm,
                            kk,
                            nn,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            self.dequant_scratch_off,
                            *out_byte_off as usize,
                        );
                    } else {
                        crate::gguf_host::run_dequant_matmul_gguf(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            mm,
                            kk,
                            nn,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    }
                }
                Step::DequantMatmulInt8Host {
                    m,
                    k,
                    n,
                    block_size,
                    is_asymmetric,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    zp_byte_off,
                    out_byte_off,
                } => {
                    let mm = scale(*m) as usize;
                    crate::int8_host::run_dequant_matmul_int8(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        mm,
                        *k as usize,
                        *n as usize,
                        *block_size as usize,
                        *is_asymmetric,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *scale_byte_off as usize,
                        *zp_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::DequantMatmulMlxHost {
                    m,
                    k,
                    n,
                    scheme,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    zp_byte_off,
                    out_byte_off,
                } => {
                    let mm = scale(*m) as usize;
                    crate::gguf_host::run_dequant_matmul_mlx(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        mm,
                        *k as usize,
                        *n as usize,
                        *scheme,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *scale_byte_off as usize,
                        *zp_byte_off as usize,
                        *out_byte_off as usize,
                    );
                }
                Step::Conv2dHost {
                    n,
                    c_in,
                    c_out,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                    in_byte_off,
                    w_byte_off,
                    out_byte_off,
                } => {
                    crate::conv_host::run_conv2d_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *n as usize,
                        *c_in as usize,
                        *c_out as usize,
                        *h as usize,
                        *w as usize,
                        *h_out as usize,
                        *w_out as usize,
                        *kh as usize,
                        *kw as usize,
                        *sh as usize,
                        *sw as usize,
                        *ph as usize,
                        *pw as usize,
                        *dh as usize,
                        *dw as usize,
                        *groups as usize,
                        *in_byte_off as usize,
                        *w_byte_off as usize,
                        *out_byte_off as usize,
                        Some(&mut host_cache),
                    );
                }
                Step::DequantGroupedMatmulGguf {
                    m,
                    k,
                    n,
                    num_experts,
                    scheme_id,
                    x_byte_off,
                    w_byte_off,
                    idx_byte_off,
                    out_byte_off,
                } => {
                    if self.dequant_scratch_off > 0 {
                        crate::gguf_gpu::run_dequant_grouped_matmul_gguf_gpu(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *num_experts as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *idx_byte_off as usize,
                            self.dequant_scratch_off,
                            *out_byte_off as usize,
                        );
                    } else {
                        crate::gguf_host::run_dequant_grouped_matmul_gguf(
                            &self.arena,
                            &dev.device,
                            &dev.queue,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *num_experts as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *idx_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    }
                }
                Step::DequantGroupedMatmulMlxHost {
                    m,
                    k,
                    n,
                    num_experts,
                    scheme,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    zp_byte_off,
                    idx_byte_off,
                    out_byte_off,
                    scale_bf16,
                } => {
                    crate::gguf_host::run_dequant_grouped_matmul_mlx(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        scale(*m) as usize,
                        *k as usize,
                        *n as usize,
                        *num_experts as usize,
                        *scheme,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *scale_byte_off as usize,
                        *zp_byte_off as usize,
                        *idx_byte_off as usize,
                        *out_byte_off as usize,
                        *scale_bf16,
                    );
                }
                Step::GatedDeltaNet {
                    params,
                    q_byte_off,
                    k_byte_off,
                    v_byte_off,
                    g_byte_off,
                    beta_byte_off,
                    state_byte_off,
                    dst_byte_off,
                    use_gpu: false,
                } => {
                    crate::gdn_host::run_gated_delta_net(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *q_byte_off as usize,
                        *k_byte_off as usize,
                        *v_byte_off as usize,
                        *g_byte_off as usize,
                        *beta_byte_off as usize,
                        *state_byte_off as usize,
                        *dst_byte_off as usize,
                        params.batch as usize,
                        scale(params.seq) as usize,
                        params.heads as usize,
                        params.state_size as usize,
                        params.use_carry != 0,
                    );
                }
                Step::GatedDeltaNet { use_gpu: true, .. } => {
                    unreachable!("rlx-wgpu: GPU GatedDeltaNet handled in compute pass");
                }
                Step::Lstm {
                    x_byte_off,
                    w_ih_byte_off,
                    w_hh_byte_off,
                    bias_byte_off,
                    h0_byte_off,
                    c0_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    crate::lstm_host::run_lstm(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x_byte_off as usize,
                        *w_ih_byte_off as usize,
                        *w_hh_byte_off as usize,
                        *bias_byte_off as usize,
                        *h0_byte_off as usize,
                        *c0_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *carry,
                    );
                }
                Step::ConvTranspose2d {
                    src_byte_off,
                    weight_byte_off,
                    dst_byte_off,
                    n,
                    c_in,
                    h,
                    w_in,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw,
                    groups,
                } => {
                    crate::conv_transpose2d_host::run_conv_transpose2d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *weight_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c_in as usize,
                        *h as usize,
                        *w_in as usize,
                        *c_out as usize,
                        *h_out as usize,
                        *w_out as usize,
                        *kh as usize,
                        *kw as usize,
                        *sh as usize,
                        *sw as usize,
                        *ph as usize,
                        *pw as usize,
                        *dh as usize,
                        *dw as usize,
                        *groups as usize,
                    );
                }
                Step::ConvTranspose3dHost {
                    src_byte_off,
                    weight_byte_off,
                    dst_byte_off,
                    n,
                    c_in,
                    d,
                    h,
                    w_in,
                    c_out,
                    d_out,
                    h_out,
                    w_out,
                    kd,
                    kh,
                    kw,
                    sd,
                    sh,
                    sw,
                    pd,
                    ph,
                    pw,
                    dd,
                    dh,
                    dw,
                    groups,
                } => {
                    crate::conv_transpose3d_host::run_conv_transpose3d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *weight_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c_in as usize,
                        *d as usize,
                        *h as usize,
                        *w_in as usize,
                        *c_out as usize,
                        *d_out as usize,
                        *h_out as usize,
                        *w_out as usize,
                        *kd as usize,
                        *kh as usize,
                        *kw as usize,
                        *sd as usize,
                        *sh as usize,
                        *sw as usize,
                        *pd as usize,
                        *ph as usize,
                        *pw as usize,
                        *dd as usize,
                        *dh as usize,
                        *dw as usize,
                        *groups as usize,
                    );
                }
                Step::GroupNormHost {
                    src_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dst_byte_off,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps,
                } => {
                    crate::vision_host::run_group_norm(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *gamma_byte_off as usize,
                        *beta_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c as usize,
                        *h as usize,
                        *w as usize,
                        *num_groups as usize,
                        *eps,
                    );
                }
                Step::LayerNorm2dHost {
                    src_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dst_byte_off,
                    n,
                    c,
                    h,
                    w,
                    eps,
                } => {
                    crate::vision_host::run_layer_norm2d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *gamma_byte_off as usize,
                        *beta_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c as usize,
                        *h as usize,
                        *w as usize,
                        *eps,
                    );
                }
                Step::ResizeNearest2xHost {
                    src_byte_off,
                    dst_byte_off,
                    n,
                    c,
                    h,
                    w,
                } => {
                    crate::vision_host::run_resize_nearest_2x(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *n as usize,
                        *c as usize,
                        *h as usize,
                        *w as usize,
                    );
                }
                Step::ReverseHost {
                    src_byte_off,
                    dst_byte_off,
                    dims,
                    rev_mask,
                    elem_bytes,
                } => {
                    crate::vision_host::run_reverse(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        dims,
                        rev_mask,
                        *elem_bytes as usize,
                    );
                }
                Step::ArgReduceHost {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    reduced,
                    inner,
                    is_max,
                } => {
                    crate::vision_host::run_argreduce(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *reduced as usize,
                        *inner as usize,
                        *is_max,
                    );
                }
                Step::AxialRope2dHost {
                    src_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    hidden,
                    end_x,
                    end_y,
                    head_dim,
                    num_heads,
                    theta,
                    repeat_factor,
                } => {
                    crate::vision_host::run_axial_rope2d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *hidden as usize,
                        *end_x as usize,
                        *end_y as usize,
                        *head_dim as usize,
                        *num_heads as usize,
                        *theta,
                        *repeat_factor as usize,
                    );
                }
                Step::GruHost {
                    x,
                    w_ih,
                    w_hh,
                    b_ih,
                    b_hh,
                    h0,
                    dst,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    crate::vision_host::run_gru(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x as usize,
                        *w_ih as usize,
                        *w_hh as usize,
                        *b_ih as usize,
                        *b_hh as usize,
                        *h0 as usize,
                        *dst as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *carry,
                    );
                }
                Step::RnnHost {
                    x,
                    w_ih,
                    w_hh,
                    bias,
                    h0,
                    dst,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                    relu,
                } => {
                    crate::vision_host::run_rnn(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x as usize,
                        *w_ih as usize,
                        *w_hh as usize,
                        *bias as usize,
                        *h0 as usize,
                        *dst as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *carry,
                        *relu,
                    );
                }
                Step::Llada2GroupLimitedGate {
                    sig_byte_off,
                    route_byte_off,
                    out_byte_off,
                    n_elems,
                    attrs,
                } => {
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *sig_byte_off as usize,
                        *route_byte_off as usize,
                        *out_byte_off as usize,
                        *n_elems as usize,
                        attrs,
                    );
                }
                Step::UmapKnnHost {
                    pairwise_byte_off,
                    out_byte_off,
                    n,
                    k,
                } => {
                    crate::umap_knn_host::run_umap_knn(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *pairwise_byte_off as usize,
                        *out_byte_off as usize,
                        *n as usize,
                        *k as usize,
                    );
                }
                Step::MsDeformAttnHost {
                    in_offs,
                    out_byte_off,
                    out_bytes,
                    attrs,
                } => {
                    crate::ms_deform_attn::run_ms_deform_attn(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        in_offs,
                        *out_byte_off as usize,
                        *out_bytes as usize,
                        attrs,
                    );
                }
                Step::CollectiveHost {
                    name,
                    in_byte_off,
                    in_bytes,
                    out_byte_off,
                    out_bytes,
                    attrs,
                } => {
                    crate::collective_host::run_collective(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        name,
                        *in_byte_off as usize,
                        *in_bytes as usize,
                        *out_byte_off as usize,
                        *out_bytes as usize,
                        attrs,
                    );
                }
                Step::CustomHost {
                    name,
                    in_specs,
                    out_byte_off,
                    out_shape,
                    attrs,
                } => {
                    crate::custom_host::run_custom_host(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        name,
                        in_specs,
                        *out_byte_off as usize,
                        out_shape,
                        attrs,
                    );
                }
                Step::FftHost {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    dtype_tag,
                } => {
                    crate::fft_host::run_fft1d(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_complex as usize,
                        *inverse,
                        *norm_tag,
                        fft_dtype_from_tag(*dtype_tag),
                    );
                }
                Step::ScanHost { desc } => {
                    crate::scan_host::run_scan(&self.arena, &dev.device, &dev.queue, desc);
                }
                Step::HostOp { desc } => {
                    crate::scan_host::run_host_op_with_cache(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        desc,
                        Some(&mut host_cache),
                    );
                }
                Step::CpuIndexing { thunk } => {
                    crate::scan_host::run_indexing(&self.arena, &dev.device, &dev.queue, thunk);
                }
                Step::ConcatHost {
                    dst_byte_off,
                    outer,
                    inner,
                    total_axis,
                    inputs,
                } => {
                    crate::scan_host::run_concat_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dst_byte_off,
                        *outer,
                        *inner,
                        *total_axis,
                        inputs,
                        Some(&mut host_cache),
                    );
                }
                Step::ConcatHostPieces {
                    dst_byte_off,
                    outer,
                    inner,
                    total_axis,
                    inputs,
                    starts,
                } => {
                    // Partial fill — other columns already on device; flush first.
                    if host_cache.has_deferred_writes() {
                        let mut a = crate::host_stage::WgpuArena {
                            arena: &self.arena,
                            device: &dev.device,
                            queue: &dev.queue,
                            size_bytes: 0,
                        };
                        host_cache.flush_to_device(&mut a);
                        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    }
                    crate::scan_host::run_concat_host_pieces(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dst_byte_off,
                        *outer,
                        *inner,
                        *total_axis,
                        inputs,
                        starts,
                        /*clear=*/ false,
                    );
                }
                Step::TransposeHost {
                    in_byte_off,
                    out_byte_off,
                    in_dims,
                    out_dims,
                    in_strides,
                } => {
                    crate::scan_host::run_transpose_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *in_byte_off,
                        *out_byte_off,
                        in_dims,
                        out_dims,
                        in_strides,
                        Some(&mut host_cache),
                    );
                }
                Step::NarrowHost {
                    in_byte_off,
                    out_byte_off,
                    outer,
                    inner,
                    axis_in_size,
                    start,
                    axis_out_size,
                } => {
                    crate::scan_host::run_narrow_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *in_byte_off,
                        *out_byte_off,
                        *outer,
                        *inner,
                        *axis_in_size,
                        *start,
                        *axis_out_size,
                        Some(&mut host_cache),
                    );
                }
                Step::ExpandHost {
                    in_byte_off,
                    out_byte_off,
                    in_dims,
                    out_dims,
                } => {
                    crate::scan_host::run_expand_host_cached(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *in_byte_off,
                        *out_byte_off,
                        in_dims,
                        out_dims,
                        Some(&mut host_cache),
                    );
                }
                Step::SpdHost {
                    op,
                    inputs,
                    out_shape,
                    out_byte_off,
                } => {
                    crate::spd_host::run_spd(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        op,
                        inputs,
                        out_shape,
                        *out_byte_off,
                    );
                }
                Step::WelchPeaksHost { .. }
                | Step::LogMelHost { .. }
                | Step::LogMelBackwardHost { .. } => {
                    unreachable!("tail-host audio ops run after GPU wait")
                }
                Step::Im2ColHost {
                    x_byte_off,
                    col_byte_off,
                    n,
                    c_in,
                    h,
                    w,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                } => {
                    crate::im2col_host::run_im2col(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x_byte_off as usize,
                        *col_byte_off as usize,
                        *n,
                        *c_in,
                        *h,
                        *w,
                        *h_out,
                        *w_out,
                        *kh,
                        *kw,
                        *sh,
                        *sw,
                        *ph,
                        *pw,
                        *dh,
                        *dw_dil,
                    );
                }
                Step::Conv2dBackwardWeightHost {
                    x_byte_off,
                    dy_byte_off,
                    dw_byte_off,
                    n,
                    c_in,
                    h,
                    w,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                    groups,
                } => {
                    crate::conv_bwd_host::run_conv2d_backward_weight(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *x_byte_off,
                        *dy_byte_off,
                        *dw_byte_off,
                        *n,
                        *c_in,
                        *h,
                        *w,
                        *c_out,
                        *h_out,
                        *w_out,
                        *kh,
                        *kw,
                        *sh,
                        *sw,
                        *ph,
                        *pw,
                        *dh,
                        *dw_dil,
                        *groups,
                    );
                }
                Step::Conv2dBackwardInputHost {
                    dy_byte_off,
                    w_byte_off,
                    dx_byte_off,
                    n,
                    c_in,
                    h,
                    w_in,
                    c_out,
                    h_out,
                    w_out,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    dh,
                    dw_dil,
                    groups,
                } => {
                    crate::conv_bwd_host::run_conv2d_backward_input(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dy_byte_off,
                        *w_byte_off,
                        *dx_byte_off,
                        *n,
                        *c_in,
                        *h,
                        *w_in,
                        *c_out,
                        *h_out,
                        *w_out,
                        *kh,
                        *kw,
                        *sh,
                        *sw,
                        *ph,
                        *pw,
                        *dh,
                        *dw_dil,
                        *groups,
                    );
                }
                Step::RngNormalHost {
                    dst_byte_off,
                    len,
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    crate::rng_host::run_rng_normal(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dst_byte_off as usize,
                        *len as usize,
                        *mean,
                        *scale,
                        *key,
                        *op_seed,
                        opts,
                    );
                }
                Step::RngUniformHost {
                    dst_byte_off,
                    len,
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    crate::rng_host::run_rng_uniform(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *dst_byte_off as usize,
                        *len as usize,
                        *low,
                        *high,
                        *key,
                        *op_seed,
                        opts,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRender {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    dst_byte_off,
                    dst_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_render(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *dst_byte_off as usize,
                        *dst_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatPrepare {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    meta_len,
                    prep_byte_off,
                    prep_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_prepare(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *meta_len as usize,
                        *prep_byte_off as usize,
                        *prep_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRasterize {
                    prep_byte_off,
                    prep_len,
                    meta_byte_off,
                    meta_len,
                    dst_byte_off,
                    dst_len,
                    count,
                    width,
                    height,
                    tile_size,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                } => {
                    crate::splat::run_gaussian_splat_rasterize(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *prep_byte_off as usize,
                        *prep_len as usize,
                        *meta_byte_off as usize,
                        *meta_len as usize,
                        *dst_byte_off as usize,
                        *dst_len as usize,
                        *count as usize,
                        *width,
                        *height,
                        *tile_size,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                }
                #[cfg(feature = "splat")]
                Step::GaussianSplatRenderBackward {
                    positions_byte_off,
                    positions_len,
                    scales_byte_off,
                    scales_len,
                    rotations_byte_off,
                    rotations_len,
                    opacities_byte_off,
                    opacities_len,
                    colors_byte_off,
                    colors_len,
                    sh_coeffs_byte_off,
                    sh_coeffs_len,
                    meta_byte_off,
                    d_loss_byte_off,
                    d_loss_len,
                    packed_byte_off,
                    packed_len,
                    width,
                    height,
                    tile_size,
                    radius_scale,
                    alpha_cutoff,
                    max_splat_steps,
                    transmittance_threshold,
                    max_list_entries,
                    loss_grad_clip,
                    sh_band,
                    max_anisotropy,
                } => {
                    crate::splat::run_gaussian_splat_render_backward(
                        &self.arena,
                        &dev.device,
                        &dev.queue,
                        *positions_byte_off as usize,
                        *positions_len as usize,
                        *scales_byte_off as usize,
                        *scales_len as usize,
                        *rotations_byte_off as usize,
                        *rotations_len as usize,
                        *opacities_byte_off as usize,
                        *opacities_len as usize,
                        *colors_byte_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_byte_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_byte_off as usize,
                        *d_loss_byte_off as usize,
                        *d_loss_len as usize,
                        *packed_byte_off as usize,
                        *packed_len as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                        *loss_grad_clip,
                        *sh_band,
                        *max_anisotropy,
                    );
                }
                _ => break,
            }
            step_i += 1;
            // Host paths stage with `queue.write_buffer` (no submit). Flush
            // deferred mirrors before the next GPU compute pass so kernels see
            // host results on discrete Vulkan/DX12.
            if step_i < self.schedule.len() {
                let next = &self.schedule[step_i];
                if !step_runs_on_host(next) && !step_is_tail_host(next) {
                    if host_cache.has_deferred_writes() {
                        let mut a = crate::host_stage::WgpuArena {
                            arena: &self.arena,
                            device: &dev.device,
                            queue: &dev.queue,
                            size_bytes: 0,
                        };
                        host_cache.flush_to_device(&mut a);
                    }
                    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
                    // Keep mirrors until GPU actually runs (cleared when
                    // `pass_dispatched` on the next host sync).
                }
            } else if host_cache.has_deferred_writes() {
                let mut a = crate::host_stage::WgpuArena {
                    arena: &self.arena,
                    device: &dev.device,
                    queue: &dev.queue,
                    size_bytes: 0,
                };
                host_cache.flush_to_device(&mut a);
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            } else {
                let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
            }
        }

        if host_cache.has_deferred_writes() {
            let mut a = crate::host_stage::WgpuArena {
                arena: &self.arena,
                device: &dev.device,
                queue: &dev.queue,
                size_bytes: 0,
            };
            host_cache.flush_to_device(&mut a);
            let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        }

        self.dump_node_stats_if_requested(dev);

        // NaN/Inf localization (RLX_DEBUG_NANS; RLX_WGPU_NAN_TRACE kept as an
        // alias). wgpu can read every node's buffer back from the arena after
        // the run, so this is a true per-node scan in topological order — the
        // first bad node is the origin, and the shared scanner classifies it as
        // culprit vs propagator with provenance + a fix hint.
        let scanner = {
            let s = rlx_ir::numeric_check::DebugScanner::from_env("wgpu");
            if s.enabled() {
                Some(s)
            } else if rlx_ir::env::flag("RLX_WGPU_NAN_TRACE") {
                Some(rlx_ir::numeric_check::DebugScanner::with_mode(
                    rlx_ir::numeric_check::DebugMode::Warn,
                    "wgpu",
                ))
            } else {
                None
            }
        };
        if let Some(scanner) = scanner {
            let mut found = false;
            for node in self.graph.nodes() {
                if !self.arena.has(node.id)
                    || matches!(
                        node.op,
                        rlx_ir::Op::Input { .. }
                            | rlx_ir::Op::Param { .. }
                            | rlx_ir::Op::Constant { .. }
                    )
                {
                    continue;
                }
                let out = self.arena.read_f32(&dev.device, &dev.queue, node.id);
                // Gather f32 operand buffers for culprit-vs-propagator (skip
                // packed U8/I8 quant weights, which aren't f32).
                let mut owned: Vec<(rlx_ir::NodeId, Vec<f32>)> = Vec::new();
                for &inp in &node.inputs {
                    let ish = self.graph.node(inp).shape.dtype();
                    if self.arena.has(inp) && !matches!(ish, rlx_ir::DType::U8 | rlx_ir::DType::I8)
                    {
                        owned.push((inp, self.arena.read_f32(&dev.device, &dev.queue, inp)));
                    }
                }
                let inrefs: Vec<(rlx_ir::NodeId, &[f32])> =
                    owned.iter().map(|(id, v)| (*id, v.as_slice())).collect();
                if scanner.check(&self.graph, node.id, &out, &inrefs).is_some() {
                    found = true;
                    break; // first bad in topo order is the origin
                }
            }
            if !found {
                eprintln!("rlx nan-check [wgpu]: clean run — no NaN/Inf");
            }
        }

        if rlx_ir::env::flag("RLX_BENCH_DISPATCH_ONLY") {
            if !self.static_once_done && !self.static_once_steps.is_empty() {
                self.static_once_done = true;
            }
            return self
                .graph
                .outputs
                .iter()
                .map(|&id| {
                    let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                    vec![0.0; n]
                })
                .collect();
        }
        if !self.static_once_done && !self.static_once_steps.is_empty() {
            self.static_once_done = true;
        }
        let out_ids: Vec<_> = self.graph.outputs.clone();
        read_f32_many_pooled(
            &self.arena,
            &dev.device,
            &dev.queue,
            &out_ids,
            &mut self.readback_staging,
        )
    }
}
