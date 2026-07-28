// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `CudaExecutable` — lowers an rlx-ir Graph into a sequence of CUDA
//! kernel launches against a pre-allocated device buffer.
//!
//! v2 op coverage: MatMul (tiled SGEMM), Binary, Compare, Activation, Where,
//! Reduce, Softmax, LayerNorm, RmsNorm, FusedResidualLN, Gather, Narrow,
//! Argmax, Reshape/Cast (no-op via slot aliasing), leaf nodes. Anything
//! else panics at compile time with a "fall back to CPU/Metal/MLX/WGPU"
//! diagnostic. Op coverage is grown incrementally — each new op is one
//! `.cu` source + one Step variant + one match arm.

#![allow(unused_imports)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::cudnn::{result as cudnn_result, sys as cudnn_sys};
use cudarc::driver::{CudaContext, DevicePtrMut, LaunchConfig, PushKernelArg};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::rlx_fusion::lower_reduce_axes::LowerNonLastAxisReduce;
use rlx_opt::rlx_fusion::pass::Pass as _;

use crate::arena::{Arena, plan_f32_uniform};
use crate::device::{
    CUBLASLT_WORKSPACE_BYTES, CUDNN_WORKSPACE_BYTES, cuda_blas, cuda_blas_lt_handle,
    cuda_blas_lt_workspace, cuda_context, cuda_dnn_handle, cuda_dnn_workspace,
};
use crate::host_staging::F32HostSlot;
use crate::kernels::{
    ada_layer_norm_backward_kernel, ada_layer_norm_kernel, argmax_kernel, attention_bwd_kernel,
    attention_kernel, attention_row_kernel, batch_elementwise_region_kernel, binary_kernel,
    compare_kernel, concat_kernel, conv_transpose2d_kernel, conv1d_kernel, conv2d_kernel,
    conv3d_kernel, copy_kernel, cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel,
    dispatch_grid_1d, dispatch_grid_prologue_nchw, elementwise_region_kernel, expand_kernel,
    fused_attn_kernel, fused_binary_unary_kernel, fused_residual_ln_kernel,
    fused_residual_rms_norm_kernel, gated_delta_net_kernel, gated_residual_backward_kernel,
    gated_residual_kernel, gather_axis_kernel, gather_backward_kernel, gather_kernel,
    group_norm_kernel, grouped_matmul_kernel, im2col_kernel, layer_norm2d_kernel, layernorm_kernel,
    matmul_epilogue_kernel, matmul_kernel, matmul_wmma_kernel, maxpool2d_backward_kernel,
    narrow_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel, reduce_kernel,
    resize_nearest_2x_kernel, rms_norm_backward_kernel, rms_norm_bwd_zero_kernel,
    rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_acc_kernel,
    scatter_add_zero_kernel, selective_scan_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    unary_kernel, where_kernel,
};

mod compile;
mod fill;
mod output;
mod run;
mod set;

mod bwd_launch;
mod helpers;
mod step;

pub(crate) use bwd_launch::*;
pub(crate) use helpers::*;
pub(crate) use step::*;

/// When kernels turn into PTX device code.
///
/// `Jit` is the default — each kernel NVRTC-compiles on first dispatch,
/// then the cuModule is cached for the rest of the process. `Aot`
/// pre-compiles every kernel at executable construction so the first
/// `run()` doesn't pay any compile latency. The full AOT pass is ~1-3s
/// (10-100ms × 32 kernels) but moves that cost out of the critical path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompileMode {
    #[default]
    Jit,
    Aot,
}

/// How the schedule executes.
///
/// `Stream` (default) launches each Step on the default stream every
/// `run()`. `Graph` captures the full schedule into a CUDA Graph on
/// first run and replays the captured graph on subsequent runs —
/// eliminates per-launch dispatch overhead (~10-20% on small-batch
/// inference). `Eager` is a one-shot helper that compiles + runs +
/// drops the executable in one call; useful for interactive debugging.
/// `MultiStream(n)` allocates a pool of `n` streams and assigns each
/// `Step` to a stream based on data dependencies — independent ops
/// (e.g. unfused Q/K/V projections, FFN gate/up) run in parallel.
/// Cross-stream synchronization uses CUDA events at producer-consumer
/// boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecMode {
    #[default]
    Stream,
    Graph,
    Eager,
    MultiStream(usize),
}

pub struct CudaExecutable {
    ctx: Arc<CudaContext>,
    /// cuBLAS handle bound to the same default stream as `ctx`. Used for
    /// plain matmul (no fused bias/activation); falls back to the custom
    /// kernel when cuBLAS isn't available (e.g., on Mac via the panic-
    /// catch probe).
    blas: Option<Arc<Mutex<CudaBlas>>>,
    /// cuBLASLt handle for fused matmul + bias + activation. Falls back
    /// to plain cuBLAS sgemm + epilogue kernel when unavailable.
    blas_lt: Option<cublaslt_sys::cublasLtHandle_t>,
    /// Shared cuBLASLt scratch — process singleton, only referenced when
    /// the schedule uses cublasLt-fusable matmul.
    blas_lt_workspace: Option<Arc<Mutex<cudarc::driver::CudaSlice<u8>>>>,
    /// cuDNN handle for convolution dispatch (conv1d/2d/3d). Falls back
    /// to the custom direct-convolution kernels when unavailable.
    dnn: Option<cudnn_sys::cudnnHandle_t>,
    /// Shared cuDNN scratch — process singleton, only referenced when the
    /// schedule contains conv steps.
    dnn_workspace: Option<Arc<Mutex<cudarc::driver::CudaSlice<u8>>>>,
    /// Scratch f16 buffer for casting activations on-the-fly when the
    /// matching weight is half-stored. Sized to fit the largest
    /// per-call M·K product seen in matmul dispatch; grown lazily.
    half_act_scratch: Option<cudarc::driver::CudaSlice<u16>>,
    /// Byte offset in the f32 arena for GGUF dequant scratch (max k×n f32).
    dequant_scratch_off: usize,
    /// Byte offset for ephemeral GatedDeltaNet state (`carry_state=false`).
    gdn_scratch_off: usize,
    graph: Graph,
    arena: Arena,
    schedule: Vec<Step>,
    input_offsets: HashMap<String, NodeId>,
    param_offsets: HashMap<String, NodeId>,
    /// Per-step side buffers for kernels that need per-axis u32 metadata
    /// (Transpose, Expand). Indexed via `Step::Transpose.meta_idx` etc.
    meta_buffers: Vec<cudarc::driver::CudaSlice<u32>>,
    exec_mode: ExecMode,
    /// Captured CUDA Graph (built on first `run()` when `exec_mode ==
    /// Graph`). Replayed on subsequent runs to skip per-launch dispatch.
    captured_graph: Option<cudarc::driver::CudaGraph>,
    /// Stream pool for `ExecMode::MultiStream(n)`. Empty for the other
    /// modes (which use the context's default stream).
    streams: Vec<Arc<cudarc::driver::CudaStream>>,
    /// Active-extent hint (`Some((actual, upper))`) for L1 bucketed
    /// dispatch. When set AND every step in `schedule` is in the
    /// safe set, `run` bypasses the captured CUDA Graph (recorded at
    /// full extent) and dispatches per-step with scaled launch dims.
    /// Otherwise full-extent fallback. See PLAN L1.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Reused host output buffers (stable addresses for CUDA Graph dtoh capture).
    output_staging: Vec<F32HostSlot>,
    /// Pinned/pageable host staging for fixed-size graph inputs.
    input_staging: HashMap<String, F32HostSlot>,
    /// cuFFT plan cache + interleaved scratch (only with the `cufft` feature).
    #[cfg(feature = "cufft")]
    cufft_state: crate::cufft_dispatch::CufftState,
    /// Reused event for graph replay completion (avoids full stream sync when possible).
    replay_event: Option<cudarc::driver::CudaEvent>,
    /// Persistent KV inputs (host mirror + device upload each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    gpu_handle_feeds: HashMap<String, usize>,
    /// Row feeds: after decode, copy output row `src_row` into handle row `dst_row`.
    kv_row_feeds: HashMap<String, usize>,
    gpu_handle_resident: std::collections::HashSet<String>,
    /// When set, only these output indices are read back from device (KV feeds stay on GPU).
    pending_read_indices: Option<Vec<usize>>,
    /// Reused sorted/deduped output indices for the current run (avoids alloc in `readback_plan`).
    readback_plan_buf: Vec<usize>,
    /// Output indices baked into the captured CUDA graph (must match on replay).
    captured_readback_plan: Option<Vec<usize>>,
    /// Graph input names in declaration order (parallel to `input_slots`).
    input_slot_names: Vec<String>,
    /// Graph inputs in declaration order: `(arena_byte_offset, max_f32_elems)`.
    input_slots: Vec<(usize, usize)>,
    /// Host readback layout: `(byte_offset_in_host_arena, f32_elems)` per graph output.
    output_slots: Vec<(usize, usize)>,
    /// Pinned/pageable host mirror for `run_slots` / `arena_ptr` (not GPU arena).
    host_arena: Vec<f32>,
    /// Runtime-mutable RNG policy for in-graph random ops.
    rng: std::sync::Arc<std::sync::RwLock<rlx_ir::RngOptions>>,
}

impl CudaExecutable {
    /// Override RNG policy for in-graph random ops without recompiling.
    pub fn set_rng(&mut self, rng: rlx_ir::RngOptions) {
        *self.rng.write().expect("rng lock") = rng;
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.rng.read().expect("rng lock")
    }
}
impl CudaExecutable {
    /// Host buffer base for reading outputs after [`Self::run_slots`].
    /// Offsets in the returned slot pairs are **byte** offsets into this buffer.
    pub fn arena_ptr(&self) -> *const u8 {
        self.host_arena.as_ptr() as *const u8
    }

    fn upload_slot_inputs(&mut self, inputs: &[&[f32]]) {
        let stream = self.ctx.default_stream();
        for (i, data) in inputs.iter().enumerate() {
            let Some(&(byte_off, max_elems)) = self.input_slots.get(i) else {
                break;
            };
            let off_f32 = byte_off / 4;
            let len = data.len().min(max_elems);
            if len == 0 {
                continue;
            }
            let mut slot = self.arena.f32_buf_mut().slice_mut(off_f32..off_f32 + len);
            if let Some(name) = self.input_slot_names.get(i) {
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(data);
                    let _ = host.htod(&stream, &mut slot, len);
                    continue;
                }
            }
            let _ = stream.memcpy_htod(&data[..len], &mut slot);
        }
    }

    fn pack_host_arena(&mut self) {
        self.prepare_readback_plan();
        for &i in &self.readback_plan_buf {
            if i >= self.output_staging.len() || i >= self.output_slots.len() {
                continue;
            }
            let (byte_off, n) = self.output_slots[i];
            if n == 0 {
                continue;
            }
            let start = byte_off / 4;
            let end = start + n;
            if end <= self.host_arena.len() {
                self.output_staging[i].copy_into(&mut self.host_arena[start..end]);
            }
        }
    }

    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.remove(name);
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }

    /// Upload any bound (non-resident) GPU handles from host mirrors into the arena.
    pub fn stage_bound_gpu_handles_to_arena(&mut self) {
        let stream = self.ctx.default_stream();
        self.stage_gpu_handle_inputs(&stream, &[]);
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Register a row feed for resident KV decode (mirrors rlx-vulkan).
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) {
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
    }

    #[allow(dead_code)] // kept for manual stream debugging / future multi-stream sync
    fn sync_all_streams(&self) {
        let _ = self.ctx.default_stream().synchronize();
        for s in &self.streams {
            let _ = s.synchronize();
        }
    }

    /// In-arena f32 copy (element offsets into the unified arena buffer).
    fn copy_arena_f32_range(
        ctx: &Arc<CudaContext>,
        stream: &Arc<cudarc::driver::CudaStream>,
        buffer: &mut cudarc::driver::CudaSlice<f32>,
        src_off: usize,
        dst_off: usize,
        n: usize,
    ) {
        if n == 0 || src_off == dst_off {
            return;
        }
        let kernel = copy_kernel(ctx);
        let count = n as u32;
        let src = src_off as u32;
        let dst = dst_off as u32;
        let (grid, block) = dispatch_grid_1d(count, 64);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launcher = stream.launch_builder(&kernel.function);
        launcher.arg(buffer).arg(&count).arg(&src).arg(&dst);
        unsafe {
            let _ = launcher.launch(cfg);
        }
    }

    /// D2D copy of one KV row from a decode output into its resident handle input.
    /// Syncs the stream so a subsequent bucket rollover read sees the new row.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) {
        if row_elems == 0 {
            return;
        }
        let stream = self.ctx.default_stream();
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in &feeds {
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                continue;
            };
            if *out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[*out_idx];
            if in_id == out_id {
                continue;
            }
            let base_out = self.arena.offset(out_id) / 4;
            let base_in = self.arena.offset(in_id) / 4;
            let rel_src = src_row * row_elems;
            let rel_dst = dst_row * row_elems;
            let cap_in = self.arena.len_of(in_id) / 4;
            let cap_out = self.arena.len_of(out_id) / 4;
            if rel_src + row_elems > cap_out || rel_dst + row_elems > cap_in {
                continue;
            }
            let src_off = base_out + rel_src;
            let dst_off = base_in + rel_dst;
            Self::copy_arena_f32_range(
                &self.ctx,
                &stream,
                self.arena.f32_buf_mut(),
                src_off,
                dst_off,
                row_elems,
            );
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
        let _ = stream.synchronize();
    }

    /// Read one row from a graph output without full-tensor D2H.
    /// Caller must ensure GPU work is complete (`run` / `run_read_outputs` syncs).
    pub fn read_output_row(
        &self,
        out_idx: usize,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        if row_inner == 0 || out_idx >= self.graph.outputs.len() {
            return None;
        }
        let id = self.graph.outputs[out_idx];
        let shape_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
        if shape_elems == 0 {
            return None;
        }
        let rel = row * row_inner;
        if rel + row_inner > shape_elems {
            return None;
        }
        let base = self.arena.offset(id) / 4;
        let off = base + rel;
        let cap_f32 = self.arena.len_of(id) / 4;
        if off + row_inner > base + cap_f32 {
            return None;
        }
        let stream = self.ctx.default_stream();
        let mut host = vec![0f32; row_inner];
        let src = self.arena.f32_buf().slice(off..off + row_inner);
        stream.memcpy_dtoh(&src, &mut host).ok()?;
        Some(host)
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(&out_idx) = self.gpu_handle_feeds.get(name) {
            if out_idx < self.graph.outputs.len() {
                let id = self.graph.outputs[out_idx];
                let stream = self.ctx.default_stream();
                let off_f32 = self.arena.offset(id) / 4;
                let n_f32 = self.arena.len_of(id) / 4;
                let mut host = vec![0f32; n_f32];
                let src = self.arena.f32_buf().slice(off_f32..off_f32 + n_f32);
                if stream.memcpy_dtoh(&src, host.as_mut_slice()).is_ok() {
                    return Some(host);
                }
            }
        }
        if self.gpu_handle_resident.contains(name) {
            if let Some(&id) = self.input_offsets.get(name) {
                let stream = self.ctx.default_stream();
                let off_f32 = self.arena.offset(id) / 4;
                let n_f32 = self.arena.len_of(id) / 4;
                let mut host = vec![0f32; n_f32];
                let src = self.arena.f32_buf().slice(off_f32..off_f32 + n_f32);
                if stream.memcpy_dtoh(&src, host.as_mut_slice()).is_ok() {
                    return Some(host);
                }
            }
        }
        self.gpu_handles.get(name).cloned()
    }

    /// Mark a graph input as device-resident without a host mirror or H2D upload.
    pub fn prepare_resident_gpu_handle(&mut self, name: &str) -> bool {
        if !self.input_offsets.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.insert(name.to_string());
        self.gpu_handles.remove(name);
        true
    }

    #[allow(dead_code)] // kept for future cross-stream device-to-device copies
    fn copy_f32_dtod_between(
        stream: &Arc<cudarc::driver::CudaStream>,
        src: &cudarc::driver::CudaSlice<f32>,
        src_off: usize,
        dst: &mut cudarc::driver::CudaSlice<f32>,
        dst_off: usize,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        let src_slice = src.slice(src_off..src_off + n);
        let mut dst_slice = dst.slice_mut(dst_off..dst_off + n);
        let _ = stream.memcpy_dtod(&src_slice, &mut dst_slice);
    }

    /// Copy a resident K/V prefix from another executable (bucket rollover).
    ///
    /// Rows below `outgoing_upper` are read from the source resident inputs; the
    /// top-of-bucket row (`g == outgoing_upper` when `to_row > outgoing_upper`) is
    /// read from decode outputs because `feed_kv_row` cannot write into the last
    /// resident slot when `dst_row == bucket upper`.
    ///
    /// Values are staged host-side (D2H then H2D) to match the flush path used in
    /// `rlx-llama32` today. Padding rows `[to_row..cap)` are zeroed. A future fast
    /// path may use pure D2D once parity is proven.
    pub fn copy_resident_kv_rows_from(
        &mut self,
        src: &Self,
        from_row: usize,
        to_row: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        if from_row >= to_row || n_layers == 0 || kv_dim == 0 {
            return true;
        }
        let stream = self.ctx.default_stream();
        let need_top = to_row > outgoing_upper;
        let top_global = outgoing_upper;
        if need_top {
            let _ = stream.synchronize();
        }

        for i in 0..n_layers {
            let k_name = format!("past_k_{i}");
            let v_name = format!("past_v_{i}");
            let Some(&dst_k) = self.input_offsets.get(k_name.as_str()) else {
                return false;
            };
            let Some(&dst_v) = self.input_offsets.get(v_name.as_str()) else {
                return false;
            };
            let Some(&src_k) = src.input_offsets.get(k_name.as_str()) else {
                return false;
            };
            let Some(&src_v) = src.input_offsets.get(v_name.as_str()) else {
                return false;
            };
            if !self.arena.has(dst_k)
                || !self.arena.has(dst_v)
                || !src.arena.has(src_k)
                || !src.arena.has(src_v)
            {
                return false;
            }

            self.gpu_handle_resident.insert(k_name.clone());
            self.gpu_handle_resident.insert(v_name.clone());
            self.gpu_handles.remove(&k_name);
            self.gpu_handles.remove(&v_name);

            let dst_k_base = self.arena.offset(dst_k) / 4;
            let dst_v_base = self.arena.offset(dst_v) / 4;
            let k_out = 1 + 2 * i;
            let v_out = 2 + 2 * i;
            if k_out >= src.graph.outputs.len() || v_out >= src.graph.outputs.len() {
                return false;
            }

            for g in from_row..to_row {
                let row_off = g.saturating_mul(kv_dim);
                let from_output = need_top && g == top_global;
                if row_off + kv_dim > self.arena.len_of(dst_k) / 4
                    || row_off + kv_dim > self.arena.len_of(dst_v) / 4
                {
                    return false;
                }
                let (host_k, host_v) = if from_output {
                    let Some(host_k) = src.read_output_row(k_out, top_global, kv_dim) else {
                        return false;
                    };
                    let Some(host_v) = src.read_output_row(v_out, top_global, kv_dim) else {
                        return false;
                    };
                    (host_k, host_v)
                } else {
                    let Some(host_k) = src.read_gpu_handle_row(k_name.as_str(), g, kv_dim) else {
                        return false;
                    };
                    let Some(host_v) = src.read_gpu_handle_row(v_name.as_str(), g, kv_dim) else {
                        return false;
                    };
                    (host_k, host_v)
                };
                let dst_buf = self.arena.f32_buf_mut();
                let mut dst_k_slice =
                    dst_buf.slice_mut(dst_k_base + row_off..dst_k_base + row_off + kv_dim);
                if stream
                    .memcpy_htod(host_k.as_slice(), &mut dst_k_slice)
                    .is_err()
                {
                    return false;
                }
                let dst_buf = self.arena.f32_buf_mut();
                let mut dst_v_slice =
                    dst_buf.slice_mut(dst_v_base + row_off..dst_v_base + row_off + kv_dim);
                if stream
                    .memcpy_htod(host_v.as_slice(), &mut dst_v_slice)
                    .is_err()
                {
                    return false;
                }
            }

            let cap_rows = self.arena.len_of(dst_k) / 4 / kv_dim.max(1);
            if to_row < cap_rows {
                let zeros = vec![0f32; kv_dim];
                for row in to_row..cap_rows {
                    let row_off = row * kv_dim;
                    let dst_buf = self.arena.f32_buf_mut();
                    let mut dst_k_slice =
                        dst_buf.slice_mut(dst_k_base + row_off..dst_k_base + row_off + kv_dim);
                    if stream
                        .memcpy_htod(zeros.as_slice(), &mut dst_k_slice)
                        .is_err()
                    {
                        return false;
                    }
                    let dst_buf = self.arena.f32_buf_mut();
                    let mut dst_v_slice =
                        dst_buf.slice_mut(dst_v_base + row_off..dst_v_base + row_off + kv_dim);
                    if stream
                        .memcpy_htod(zeros.as_slice(), &mut dst_v_slice)
                        .is_err()
                    {
                        return false;
                    }
                }
            }
        }
        let _ = stream.synchronize();
        true
    }

    /// D2D copy of a resident KV prefix from another executable (bucket rollover).
    pub fn seed_resident_kv_prefix_from(
        &mut self,
        src: &Self,
        prefix_tokens: usize,
        outgoing_upper: usize,
        kv_dim: usize,
        n_layers: usize,
    ) -> bool {
        self.copy_resident_kv_rows_from(src, 0, prefix_tokens, outgoing_upper, kv_dim, n_layers)
    }

    /// Read one row from a resident GPU input handle without full-tensor D2H.
    pub fn read_gpu_handle_row(
        &self,
        name: &str,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        if row_inner == 0 {
            return None;
        }
        let &id = self.input_offsets.get(name)?;
        let cap_f32 = self.arena.len_of(id) / 4;
        let rel = row * row_inner;
        if rel + row_inner > cap_f32 {
            return None;
        }
        let base = self.arena.offset(id) / 4;
        let off = base + rel;
        let stream = self.ctx.default_stream();
        let mut host = vec![0f32; row_inner];
        let src = self.arena.f32_buf().slice(off..off + row_inner);
        stream.memcpy_dtoh(&src, &mut host).ok()?;
        Some(host)
    }

    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let mut exe = Self::compile_with_rng(
            self.graph.clone(),
            compile_mode_from_env(),
            exec_mode_from_env(),
            self.rng(),
        );
        for (k, v) in &self.gpu_handles {
            exe.bind_gpu_handle(k, v);
        }
        for (k, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(k, idx);
        }
        for (k, &idx) in &self.kv_row_feeds {
            exe.register_kv_row_feed(k, idx);
        }
        exe.set_active_extent(self.active_extent);
        exe
    }

    /// Build the sorted output readback plan into [`Self::readback_plan_buf`].
    fn prepare_readback_plan(&mut self) {
        self.readback_plan_buf.clear();
        let n = self.graph.outputs.len();
        if let Some(ref want) = self.pending_read_indices {
            self.readback_plan_buf.extend_from_slice(want);
            normalize_read_indices(&mut self.readback_plan_buf);
            return;
        }
        self.readback_plan_buf.extend(0..n);
    }

    fn propagate_gpu_handle_feeds_d2d(&mut self, stream: &Arc<cudarc::driver::CudaStream>) {
        let extent = self.active_extent;
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_offsets.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                let out_bytes = self.arena.len_of(out_id);
                let copy_bytes = match extent {
                    Some((actual, upper)) if upper > 0 => {
                        let stride = (out_bytes / (upper + 1)).max(4);
                        (actual * stride).min(out_bytes)
                    }
                    _ => out_bytes,
                }
                .min(self.arena.len_of(in_id));
                let src_off = self.arena.offset(out_id) / 4;
                let dst_off = self.arena.offset(in_id) / 4;
                let n_f32 = copy_bytes / 4;
                if n_f32 > 0 && src_off != dst_off {
                    let mut tmp = vec![0.0f32; n_f32];
                    let src = self.arena.f32_buf().slice(src_off..src_off + n_f32);
                    if stream.memcpy_dtoh(&src, &mut tmp).is_ok() {
                        let mut dst = self.arena.f32_buf_mut().slice_mut(dst_off..dst_off + n_f32);
                        let _ = stream.memcpy_htod(&tmp, &mut dst);
                    }
                }
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    fn stage_gpu_handle_inputs(
        &mut self,
        stream: &Arc<cudarc::driver::CudaStream>,
        inputs: &[(&str, &[f32])],
    ) {
        for (name, data) in &self.gpu_handles {
            if self.gpu_handle_resident.contains(name) || inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_offsets.get(name.as_str())
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let mut slot = self
                    .arena
                    .f32_buf_mut()
                    .slice_mut(off_f32..off_f32 + data.len());
                if let Some(host) = self.input_staging.get_mut(name.as_str()) {
                    host.copy_from_host(data);
                    let _ = host.htod(stream, &mut slot, data.len());
                } else {
                    let _ = stream.memcpy_htod(data.as_slice(), &mut slot);
                }
            }
        }
    }

    fn refresh_gpu_handles_from_staging(&mut self, plan: &[usize]) {
        if self.pending_read_indices.is_some() {
            return;
        }
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if plan.contains(&out_idx) && out_idx < self.output_staging.len() {
                self.gpu_handles
                    .insert(name.clone(), self.output_staging[out_idx].to_vec());
            }
        }
    }
}
mod tests {
    //! Pure-function tests for the multi-stream scheduler analysis and
    //! the element-wise fusion pass. Both are pure Rust against
    //! synthesized `Vec<Step>` inputs — no CUDA driver needed, so they
    //! run on Mac.
    use super::*;

    #[test]
    fn normalize_read_indices_dedupes() {
        let mut v = vec![3, 1, 2, 1, 0];
        normalize_read_indices(&mut v);
        assert_eq!(v, vec![0, 1, 2, 3]);
    }

    #[test]
    fn step_offsets_binary() {
        let s = Step::Binary {
            n: 8,
            a_off: 100,
            b_off: 200,
            c_off: 300,
            op: 0,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![100, 200]);
        assert_eq!(w, vec![300]);
    }

    #[test]
    fn step_offsets_matmul_with_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 1,
            bias_off_f32: 40,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20, 40]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_matmul_no_bias() {
        let s = Step::Matmul {
            m: 4,
            k: 8,
            n: 4,
            a_off_f32: 10,
            b_off_f32: 20,
            c_off_f32: 30,
            batch: 1,
            a_batch_stride: 0,
            b_batch_stride: 0,
            c_batch_stride: 0,
            has_bias: 0,
            bias_off_f32: 0,
            act_id: 0xFFFF,
        };
        let (r, w) = step_offsets(&s);
        assert_eq!(r, vec![10, 20]);
        assert_eq!(w, vec![30]);
    }

    #[test]
    fn step_offsets_attention_causal_no_mask_arg() {
        let (mb, mh, mq, mk) = rlx_ir::mask_strides_bhsd(1, 8, 8);
        let (qb, qh, qs) = rlx_ir::strides_bhsd(1, 64, 8);
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 1, // causal — mask_off ignored
            scale_bits: 0,
            softcap_bits: 0,
            window: 0,
            seq_q_stride: mq,
            seq_k_stride: mk,
            mask_batch_stride: mb,
            mask_head_stride: mh,
            q_batch_stride: qb,
            q_head_stride: qh,
            q_seq_stride: qs,
            k_batch_stride: qb,
            k_head_stride: qh,
            k_seq_stride: qs,
            v_batch_stride: qb,
            v_head_stride: qh,
            v_seq_stride: qs,
            o_batch_stride: qb,
            o_head_stride: qh,
            o_seq_stride: qs,
        };
        let (r, _) = step_offsets(&s);
        assert!(!r.contains(&9999), "causal mask must not consume mask_off");
        assert_eq!(r, vec![0, 100, 200]);
    }

    #[test]
    fn step_offsets_attention_custom_mask_pulls_mask() {
        let (mb, mh, mq, mk) = rlx_ir::mask_strides_bhsd(1, 8, 8);
        let (qb, qh, qs) = rlx_ir::strides_bhsd(1, 64, 8);
        let s = Step::Attention {
            batch: 1,
            heads: 1,
            seq_q: 8,
            seq_k: 8,
            head_dim: 64,
            q_off: 0,
            k_off: 100,
            v_off: 200,
            out_off: 300,
            mask_off: 9999,
            mask_kind: 2, // custom mask
            scale_bits: 0,
            softcap_bits: 0,
            window: 0,
            seq_q_stride: mq,
            seq_k_stride: mk,
            mask_batch_stride: mb,
            mask_head_stride: mh,
            q_batch_stride: qb,
            q_head_stride: qh,
            q_seq_stride: qs,
            k_batch_stride: qb,
            k_head_stride: qh,
            k_seq_stride: qs,
            v_batch_stride: qb,
            v_head_stride: qh,
            v_seq_stride: qs,
            o_batch_stride: qb,
            o_head_stride: qh,
            o_seq_stride: qs,
        };
        let (r, _) = step_offsets(&s);
        assert!(r.contains(&9999));
    }

    #[test]
    fn step_offsets_scatter_add_acc_marks_out_as_rmw() {
        let s = Step::ScatterAddAcc {
            out_off: 100,
            upd_off: 200,
            idx_off: 300,
            num_updates: 4,
            trailing: 1,
            out_dim: 16,
        };
        let (r, w) = step_offsets(&s);
        // out is read-modify-write, so it appears in BOTH reads and writes
        // — this lets the multi-stream scheduler force the prior
        // ScatterAddZero to complete before the accumulate launches.
        assert!(r.contains(&100));
        assert!(w.contains(&100));
    }

    #[test]
    fn fuse_elementwise_merges_binary_then_unary() {
        let schedule = vec![
            // c = a + b
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            // d = relu(c)
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 1, "expected exactly one fused step");
        match &fused[0] {
            Step::FusedBinaryUnary {
                n,
                a_off,
                b_off,
                out_off,
                bin_op,
                un_op,
            } => {
                assert_eq!(*n, 4);
                assert_eq!(*a_off, 0);
                assert_eq!(*b_off, 4);
                assert_eq!(*out_off, 12);
                assert_eq!(*bin_op, 0);
                assert_eq!(*un_op, 0);
            }
            other => panic!("expected FusedBinaryUnary, got {}", step_name(other)),
        }
    }

    #[test]
    fn fuse_elementwise_skips_when_intermediate_has_two_consumers() {
        // c = a + b
        // d = relu(c)
        // e = c * c   ← second consumer of c, blocks fusion
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
            Step::Binary {
                n: 4,
                a_off: 8,
                b_off: 8,
                c_off: 16,
                op: 2,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 3, "no fusion: c has multiple consumers");
        assert!(matches!(&fused[0], Step::Binary { .. }));
        assert!(matches!(&fused[1], Step::Unary { .. }));
        assert!(matches!(&fused[2], Step::Binary { .. }));
    }

    #[test]
    fn fuse_elementwise_skips_when_n_mismatch() {
        // Different element counts → can't fuse (different launch grid).
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 8,
                in_off: 8,
                out_off: 16,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fuse_elementwise_skips_when_unary_input_isnt_binary_output() {
        // Unary reads a different offset than what Binary wrote.
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 99,
                out_off: 16,
                op: 0,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn fuse_elementwise_handles_multiple_chains() {
        // Two independent Binary→Unary chains in a row — both should fuse.
        let schedule = vec![
            Step::Binary {
                n: 4,
                a_off: 0,
                b_off: 4,
                c_off: 8,
                op: 0,
            },
            Step::Unary {
                n: 4,
                in_off: 8,
                out_off: 12,
                op: 0,
            },
            Step::Binary {
                n: 4,
                a_off: 16,
                b_off: 20,
                c_off: 24,
                op: 2,
            },
            Step::Unary {
                n: 4,
                in_off: 24,
                out_off: 28,
                op: 9,
            },
        ];
        let fused = fuse_elementwise_chains(schedule);
        assert_eq!(fused.len(), 2);
        assert!(matches!(&fused[0], Step::FusedBinaryUnary { .. }));
        assert!(matches!(&fused[1], Step::FusedBinaryUnary { .. }));
    }
}
