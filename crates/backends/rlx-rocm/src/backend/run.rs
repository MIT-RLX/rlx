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

//! `run` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::arena::{Arena, HalfDtype, plan_f32_uniform};
use crate::device::{RocmContext, rocm_blas, rocm_blas_lt, rocm_context, rocm_dnn};
use crate::hip::{HipBuffer, HipDeviceptr};
use crate::hipblas::{
    HipblasComputeType, HipblasContext, HipblasDatatype, HipblasOperation, hipblas_gemm_default,
};
use crate::hipblaslt::HipblasLtContext;
use crate::host_staging::F32HostSlot;
use crate::miopen::MiopenContext;
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use super::*;

impl RocmExecutable {
    /// Fast path: positional inputs, D2H into [`Self::host_arena`], no per-output `Vec`.
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        self.upload_slot_inputs(inputs);
        let _ = self.run_inner(&[]);
        self.pack_host_arena();
        &self.output_slots
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    /// Run and read back only selected outputs (+ GPU handle feed outputs).
    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        self.pending_read_indices = read_indices.map(|s| s.to_vec());
        let outs = self.run_inner(inputs);
        self.pending_read_indices = None;
        // NaN/Inf output-boundary scan (RLX_DEBUG_NANS). ROCm runs op-by-op on
        // the device; per-op D2H would perturb timing, so we scan the outputs
        // here (when reading all of them, where they align with graph.outputs).
        // For internal localization replay the same graph on the CPU backend.
        let scanner = rlx_ir::numeric_check::DebugScanner::from_env("rocm");
        if scanner.enabled() && read_indices.is_none() {
            for (buf, &id) in outs.iter().zip(self.graph.outputs.iter()) {
                scanner.check(&self.graph, id, buf, &[]);
            }
        }
        outs
    }

    pub(crate) fn run_inner(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        use crate::kernels::*;

        let stream = self.ctx.default_stream;
        let arena_base = self.arena.buffer.ptr;

        self.stage_gpu_handle_inputs(inputs);

        // Copy inputs to device. Always done outside any graph capture
        // — inputs change between runs and shouldn't be baked into a
        // captured hipGraph.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let dst = arena_base + (off_f32 as u64) * 4;
                if let Some(host) = self.input_staging.get_mut(name) {
                    host.copy_from_host(data);
                    host.htod(&self.ctx.runtime, dst, data.len())
                        .expect("rlx-rocm: pinned input upload failed");
                } else {
                    unsafe {
                        let _ = (self.ctx.runtime.hip_memcpy_htod)(
                            dst,
                            data.as_ptr() as *const _,
                            std::mem::size_of_val(data),
                        );
                    }
                }
            }
        }

        // Active-extent (PLAN L1): when set + every Step safe, bypass
        // hipGraph capture/replay (recorded at full extent) and dispatch
        // per-step with scaled launch dims via the normal loop.
        let active = self.active_extent.filter(|_| self.all_safe_for_active());
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // hipGraph fast path: replay the previously-captured schedule.
        let graph_eligible = active.is_none()
            && self.exec_mode == ExecMode::Graph
            && schedule_graph_capture_safe(&self.schedule);
        let do_replay = graph_eligible && self.captured_graph.is_some();
        let do_capture = graph_eligible && self.captured_graph.is_none();
        if do_replay {
            unsafe {
                let _ = (self.ctx.runtime.hip_graph_launch)(self.captured_graph.unwrap(), stream);
                let _ = (self.ctx.runtime.hip_stream_sync)(stream);
            }
            self.run_tail_host_audio_ops(false);
            return self.finalize_outputs();
        }
        if do_capture {
            // hipStreamCaptureMode_Relaxed = 2 (matches CUDA value).
            unsafe {
                let _ = (self.ctx.runtime.hip_stream_begin_capture)(stream, 2);
            }
        }

        // Multi-stream scheduler state. When `exec_mode ==
        // MultiStream(n)`, each Step gets assigned to one of `n`
        // pool streams based on producer-consumer dependencies on
        // arena offsets. Independent ops parallelise; producer-
        // consumer chains stay on one stream.
        let multi_stream =
            matches!(self.exec_mode, ExecMode::MultiStream(_)) && !self.streams.is_empty();
        let mut producer_of: HashMap<u32, usize> = HashMap::new();
        let mut last_event: HashMap<usize, crate::hip::HipEvent> = HashMap::new();
        let mut rr_cursor: usize = 0;

        // Dispatch each step on the default stream.
        for step in &self.schedule {
            let _roctx = crate::roctx::scoped_range(step_name(step));
            // PLAN L3: cross-backend Perfetto trace; no-op when env
            // var RLX_TRACE_PERFETTO unset.
            let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "rocm");
            let mut arena_ptr = arena_base;

            // Per-step stream selection (multi-stream mode only).
            let assigned_idx: Option<usize> = if multi_stream {
                let (reads, _) = step_offsets(step);
                let mut producer_streams: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for r in &reads {
                    if let Some(&s) = producer_of.get(r) {
                        producer_streams.insert(s);
                    }
                }
                let chosen = if producer_streams.is_empty() {
                    let s = rr_cursor % self.streams.len();
                    rr_cursor += 1;
                    s
                } else if producer_streams.len() == 1 {
                    *producer_streams.iter().next().unwrap()
                } else {
                    let chosen = *producer_streams.iter().next().unwrap();
                    for s in &producer_streams {
                        if *s != chosen
                            && let Some(evt) = last_event.get(s)
                        {
                            unsafe {
                                let _ = (self.ctx.runtime.hip_stream_wait_event)(
                                    self.streams[chosen],
                                    *evt,
                                    0,
                                );
                            }
                        }
                    }
                    chosen
                };
                Some(chosen)
            } else {
                None
            };
            // Shadow the outer `stream` with the assigned stream.
            #[allow(unused_assignments)]
            let stream = match assigned_idx {
                Some(i) => self.streams[i],
                None => stream,
            };
            // Re-bind hipBLAS handle to the active stream so the
            // hipblasSgemm path's internal kernel launches go to the
            // right queue.
            if multi_stream && let Some(blas) = self.blas.as_ref() {
                let blas = blas.lock().unwrap();
                unsafe {
                    let _ = blas.set_stream(stream);
                }
            }
            match step {
                Step::Binary {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = binary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, a_off, b_off, c_off, op]
                    );
                }
                Step::RocmGpuKernel {
                    name,
                    out_off,
                    out_len,
                    in_offs,
                } => {
                    // Raw-GPU custom op: hipRTC-compile (cached) + launch against
                    // the whole arena. Offsets are scalar args (copied into the
                    // launch params at enqueue, so no async lifetime hazard).
                    let gk = crate::rocm_gpu_kernels::lookup(name)
                        .expect("RocmGpuKernel vanished from the registry between compile and run");
                    let gpu_kernel = crate::rocm_gpu_kernels::get_or_build(&self.ctx, &*gk);
                    let bs = gk.block_size();
                    let (grid, block) = dispatch_grid_1d(*out_len, bs);
                    // Pad (off,len) to MAX_INPUTS with (0,0); `n_inputs` says how
                    // many are real. Matches the fixed 12-arg kernel signature.
                    let n_inputs = in_offs.len() as u32;
                    let mut io = [0u32; crate::rocm_gpu_kernels::MAX_INPUTS * 2];
                    for (i, (o, l)) in in_offs.iter().enumerate() {
                        io[i * 2] = *o;
                        io[i * 2 + 1] = *l;
                    }
                    crate::launch_kernel!(
                        gpu_kernel.as_ref(),
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            out_off,
                            out_len,
                            &n_inputs,
                            &io[0],
                            &io[1],
                            &io[2],
                            &io[3],
                            &io[4],
                            &io[5],
                            &io[6],
                            &io[7]
                        ]
                    );
                }
                Step::Compare {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = compare_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, a_off, b_off, c_off, op]
                    );
                }
                Step::Unary {
                    n,
                    in_off,
                    out_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = unary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, in_off, out_off, op]
                    );
                }
                Step::Where {
                    n,
                    cond_off,
                    x_off,
                    y_off,
                    out_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = where_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, cond_off, x_off, y_off, out_off]
                    );
                }
                Step::FusedBinaryUnary {
                    n,
                    a_off,
                    b_off,
                    out_off,
                    bin_op,
                    un_op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = fused_binary_unary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, a_off, b_off, out_off, bin_op, un_op]
                    );
                }
                Step::ElementwiseRegion {
                    len,
                    num_inputs,
                    num_steps,
                    dst_off,
                    input_offs: _,
                    scalar_input_mask,
                    input_modulus,
                    meta_idx,
                    spatial_prologue,
                    prologue_w,
                    prologue_h,
                    prologue_nc,
                } => {
                    let len_s = scale(*len);
                    if len_s == 0 {
                        continue;
                    }
                    let kernel = elementwise_region_kernel(&self.ctx);
                    let ((gx, gy, gz), (bx, by, bz)) = if *spatial_prologue {
                        dispatch_grid_prologue_nchw(*prologue_w, *prologue_h, *prologue_nc)
                    } else {
                        let (grid, block) = dispatch_grid_1d(len_s, 256);
                        ((grid, 1, 1), (block, 1, 1))
                    };
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (gx, gy, gz),
                        (bx, by, bz),
                        [
                            &mut arena_ptr,
                            &len_s,
                            num_inputs,
                            num_steps,
                            dst_off,
                            &mut meta_ptr,
                            scalar_input_mask,
                            input_modulus
                        ]
                    );
                }
                Step::BatchElementwiseRegion {
                    slice_len,
                    num_batch,
                    num_steps,
                    base_dst_off,
                    slice_elems,
                    batch_offs_idx,
                    meta_idx,
                    scalar_input_mask,
                    input_modulus,
                    ..
                } => {
                    let slice_len_s = scale(*slice_len);
                    let num_batch_s = scale(*num_batch);
                    if slice_len_s == 0 || num_batch_s == 0 {
                        continue;
                    }
                    let kernel = batch_elementwise_region_kernel(&self.ctx);
                    let (grid_x, block_x) = dispatch_grid_1d(slice_len_s, 256);
                    let mut batch_ptr = self.meta_buffers[*batch_offs_idx].ptr;
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid_x, 1, num_batch_s),
                        (block_x, 1, 1),
                        [
                            &mut arena_ptr,
                            &slice_len_s,
                            &num_batch_s,
                            num_steps,
                            base_dst_off,
                            slice_elems,
                            &mut batch_ptr,
                            &mut meta_ptr,
                            scalar_input_mask,
                            input_modulus
                        ]
                    );
                }
                Step::Reduce {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    op,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = reduce_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [&mut arena_ptr, &outer_s, inner, in_off, out_off, op]
                    );
                }
                Step::Softmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = softmax_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [&mut arena_ptr, &outer_s, inner, in_off, out_off]
                    );
                }
                Step::LayerNorm {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    gamma_off,
                    beta_off,
                    eps_bits,
                    op,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = layernorm_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            &outer_s,
                            inner,
                            in_off,
                            out_off,
                            gamma_off,
                            beta_off,
                            eps_bits,
                            op
                        ]
                    );
                }
                Step::FusedResidualLn {
                    outer,
                    inner,
                    in_off,
                    residual_off,
                    bias_off,
                    gamma_off,
                    beta_off,
                    out_off,
                    eps_bits,
                    has_bias,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = fused_residual_ln_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            &outer_s,
                            inner,
                            in_off,
                            residual_off,
                            bias_off,
                            gamma_off,
                            beta_off,
                            out_off,
                            eps_bits,
                            has_bias
                        ]
                    );
                }
                Step::AdaLayerNorm {
                    outer,
                    inner,
                    in_off,
                    scale_off,
                    shift_off,
                    out_off,
                    eps_bits,
                    layer_norm,
                    meta_idx,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = ada_layer_norm_kernel(&self.ctx);
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (outer_s, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            &outer_s,
                            inner,
                            in_off,
                            scale_off,
                            shift_off,
                            out_off,
                            eps_bits,
                            layer_norm,
                            &mut meta_ptr
                        ]
                    );
                }
                Step::GatedResidual {
                    total,
                    inner,
                    x_off,
                    y_off,
                    gate_off,
                    out_off,
                    meta_idx,
                } => {
                    let total_s = scale(*total);
                    if total_s == 0 {
                        continue;
                    }
                    let kernel = gated_residual_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(total_s, 256);
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            &total_s,
                            inner,
                            x_off,
                            y_off,
                            gate_off,
                            out_off,
                            &mut meta_ptr
                        ]
                    );
                }
                Step::AdaLayerNormBackward {
                    mod_rows,
                    seq_per_mod,
                    inner,
                    x_off,
                    scale_off,
                    dy_off,
                    out_off,
                    eps_bits,
                    layer_norm,
                } => {
                    let mod_rows_s = scale(*mod_rows);
                    if mod_rows_s == 0 {
                        continue;
                    }
                    let kernel = ada_layer_norm_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (mod_rows_s, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            &mod_rows_s,
                            seq_per_mod,
                            inner,
                            x_off,
                            scale_off,
                            dy_off,
                            out_off,
                            eps_bits,
                            layer_norm
                        ]
                    );
                }
                Step::GatedResidualBackward {
                    mod_rows,
                    seq_per_mod,
                    inner,
                    y_off,
                    gate_off,
                    dy_off,
                    out_off,
                } => {
                    let mod_rows_s = scale(*mod_rows);
                    if mod_rows_s == 0 {
                        continue;
                    }
                    let kernel = gated_residual_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (mod_rows_s, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            &mod_rows_s,
                            seq_per_mod,
                            inner,
                            y_off,
                            gate_off,
                            dy_off,
                            out_off
                        ]
                    );
                }
                Step::Argmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let kernel = argmax_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, outer, inner, in_off, out_off]
                    );
                }
                Step::Cumsum {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    exclusive,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = cumsum_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(outer_s, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &outer_s, inner, in_off, out_off, exclusive]
                    );
                }
                Step::TopK {
                    outer,
                    inner,
                    k,
                    in_off,
                    out_off,
                } => {
                    let kernel = topk_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, outer, inner, k, in_off, out_off]
                    );
                }
                Step::Sample {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    top_k,
                    top_p_bits,
                    temp_bits,
                    seed_lo,
                    seed_hi,
                } => {
                    let kernel = sample_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            outer,
                            inner,
                            in_off,
                            out_off,
                            top_k,
                            top_p_bits,
                            temp_bits,
                            seed_lo,
                            seed_hi
                        ]
                    );
                }
                Step::RngNormal {
                    dst_byte_off,
                    len,
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    crate::rng_host::run_rng_normal(
                        &self.ctx,
                        &self.arena.buffer,
                        *dst_byte_off as usize,
                        *len as usize,
                        *mean,
                        *scale,
                        *key,
                        *op_seed,
                        opts,
                    );
                }
                Step::RngUniform {
                    dst_byte_off,
                    len,
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let opts = *self.rng.read().expect("rng lock");
                    crate::rng_host::run_rng_uniform(
                        &self.ctx,
                        &self.arena.buffer,
                        *dst_byte_off as usize,
                        *len as usize,
                        *low,
                        *high,
                        *key,
                        *op_seed,
                        opts,
                    );
                }
                Step::Gather {
                    n_out,
                    n_idx,
                    dim,
                    vocab,
                    in_off,
                    idx_off,
                    out_off,
                } => {
                    let kernel = gather_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_out, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n_out,
                            n_idx,
                            dim,
                            vocab,
                            in_off,
                            idx_off,
                            out_off
                        ]
                    );
                }
                Step::GatherAxis {
                    total,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                    table_off,
                    idx_off,
                    out_off,
                } => {
                    let kernel = gather_axis_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            total,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            table_off,
                            idx_off,
                            out_off
                        ]
                    );
                }
                Step::Narrow {
                    total,
                    outer,
                    inner,
                    axis_in_size,
                    axis_out_size,
                    start,
                    in_off,
                    out_off,
                } => {
                    let kernel = narrow_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            total,
                            outer,
                            inner,
                            axis_in_size,
                            axis_out_size,
                            start,
                            in_off,
                            out_off
                        ]
                    );
                }
                Step::Concat {
                    total,
                    outer,
                    inner,
                    axis_in_size,
                    axis_out_size,
                    start,
                    in_off,
                    out_off,
                } => {
                    let kernel = concat_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            total,
                            outer,
                            inner,
                            axis_in_size,
                            axis_out_size,
                            start,
                            in_off,
                            out_off
                        ]
                    );
                }
                Step::Transpose {
                    rank,
                    out_total,
                    in_off,
                    out_off,
                    meta_idx,
                } => {
                    let kernel = transpose_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            rank,
                            out_total,
                            in_off,
                            out_off,
                            &mut meta_ptr
                        ]
                    );
                }
                Step::Expand {
                    rank,
                    out_total,
                    in_off,
                    out_off,
                    meta_idx,
                } => {
                    let kernel = expand_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let mut meta_ptr = self.meta_buffers[*meta_idx].ptr;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            rank,
                            out_total,
                            in_off,
                            out_off,
                            &mut meta_ptr
                        ]
                    );
                }
                Step::Rope {
                    n_total,
                    seq,
                    head_dim,
                    half,
                    in_off,
                    cos_off,
                    sin_off,
                    out_off,
                    last_dim,
                    interleaved,
                } => {
                    let kernel = rope_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n_total,
                            seq,
                            head_dim,
                            half,
                            in_off,
                            cos_off,
                            sin_off,
                            out_off,
                            last_dim,
                            interleaved
                        ]
                    );
                }
                Step::Attention {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim,
                    q_off,
                    k_off,
                    v_off,
                    out_off,
                    mask_off,
                    mask_kind,
                    scale_bits,
                    softcap_bits,
                    window,
                    seq_q_stride,
                    seq_k_stride,
                    mask_batch_stride,
                    mask_head_stride,
                    q_batch_stride,
                    q_head_stride,
                    q_seq_stride,
                    k_batch_stride,
                    k_head_stride,
                    k_seq_stride,
                    v_batch_stride,
                    v_head_stride,
                    v_seq_stride,
                    o_batch_stride,
                    o_head_stride,
                    o_seq_stride,
                } => {
                    let use_row = rlx_ir::attention_dispatch_use_row(
                        *head_dim,
                        "RLX_ROCM_FORCE_ATTENTION_ROW",
                    );
                    if use_row {
                        let total = batch * heads * seq_q;
                        let block = 256u32;
                        crate::launch_kernel!(
                            attention_row_kernel(&self.ctx),
                            stream,
                            (total.div_ceil(block), 1, 1),
                            (block, 1, 1),
                            [
                                &mut arena_ptr,
                                batch,
                                heads,
                                seq_q,
                                seq_k,
                                head_dim,
                                q_off,
                                k_off,
                                v_off,
                                out_off,
                                mask_off,
                                mask_kind,
                                scale_bits,
                                window,
                                seq_q_stride,
                                seq_k_stride,
                                mask_batch_stride,
                                mask_head_stride,
                                q_batch_stride,
                                q_head_stride,
                                q_seq_stride,
                                k_batch_stride,
                                k_head_stride,
                                k_seq_stride,
                                v_batch_stride,
                                v_head_stride,
                                v_seq_stride,
                                o_batch_stride,
                                o_head_stride,
                                o_seq_stride,
                                softcap_bits
                            ]
                        );
                    } else {
                        let q_blocks = (*seq_q).div_ceil(16);
                        crate::launch_kernel!(
                            attention_kernel(&self.ctx),
                            stream,
                            (q_blocks, batch * heads, 1),
                            (128, 1, 1),
                            [
                                &mut arena_ptr,
                                batch,
                                heads,
                                seq_q,
                                seq_k,
                                head_dim,
                                q_off,
                                k_off,
                                v_off,
                                out_off,
                                mask_off,
                                mask_kind,
                                scale_bits,
                                window,
                                seq_q_stride,
                                seq_k_stride,
                                mask_batch_stride,
                                mask_head_stride,
                                q_batch_stride,
                                q_head_stride,
                                q_seq_stride,
                                k_batch_stride,
                                k_head_stride,
                                k_seq_stride,
                                v_batch_stride,
                                v_head_stride,
                                v_seq_stride,
                                o_batch_stride,
                                o_head_stride,
                                o_seq_stride,
                                softcap_bits
                            ]
                        );
                    }
                }
                Step::AttentionBackward {
                    batch,
                    heads,
                    seq_q,
                    seq_k,
                    head_dim,
                    q_off,
                    k_off,
                    v_off,
                    dy_off,
                    out_off,
                    mask_off,
                    mask_kind,
                    scale_bits,
                    window,
                    wrt,
                } => {
                    let kernel = attention_bwd_kernel(&self.ctx);
                    let seq_axis = if *wrt == 0 { *seq_q } else { *seq_k };
                    let y_blocks = seq_axis.div_ceil(256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (batch * heads, y_blocks, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            batch,
                            heads,
                            seq_q,
                            seq_k,
                            head_dim,
                            q_off,
                            k_off,
                            v_off,
                            dy_off,
                            out_off,
                            mask_off,
                            mask_kind,
                            scale_bits,
                            window,
                            wrt
                        ]
                    );
                }
                Step::ScaledQuantScale {
                    x_off_f32,
                    scale_off_f32,
                    n,
                    max_finite,
                } => {
                    let kernel = crate::kernels::scaled_quant_scale_kernel(&self.ctx);
                    let mut arena_ptr = arena_base;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (1, 1, 1),
                        (256, 1, 1),
                        [&mut arena_ptr, x_off_f32, scale_off_f32, n, max_finite]
                    );
                }
                Step::ScaledQuantizeFp8 {
                    x_off_f32,
                    scale_off_f32,
                    out_byte_off,
                    n,
                    e5m2,
                } => {
                    let kernel = crate::kernels::scaled_quantize_fp8_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n, 256);
                    let mut arena_ptr = arena_base;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            x_off_f32,
                            scale_off_f32,
                            out_byte_off,
                            n,
                            e5m2
                        ]
                    );
                }
                Step::ScaledMatMul {
                    m,
                    k,
                    n,
                    lhs_byte_off,
                    rhs_byte_off,
                    lhs_scale_byte_off,
                    rhs_scale_byte_off,
                    out_byte_off,
                    has_bias,
                    bias_byte_off,
                    lhs_e5m2,
                    rhs_e5m2,
                } => {
                    let lt = self
                        .blas_lt
                        .as_ref()
                        .expect("rlx-rocm ScaledMatMul: hipBLASLt required for FP8 GEMM");
                    let workspace = self
                        .blas_lt_workspace
                        .as_ref()
                        .expect("rlx-rocm ScaledMatMul: hipBLASLt workspace required");
                    let r = unsafe {
                        crate::hipblaslt::matmul_fused_fp8(
                            lt,
                            workspace.ptr,
                            HIPBLASLT_WORKSPACE_BYTES,
                            arena_base,
                            *m,
                            *k,
                            *n,
                            *lhs_byte_off as u64,
                            *rhs_byte_off as u64,
                            *lhs_scale_byte_off as u64,
                            *rhs_scale_byte_off as u64,
                            *out_byte_off as u64,
                            *has_bias != 0,
                            *bias_byte_off as u64,
                            *lhs_e5m2 != 0,
                            *rhs_e5m2 != 0,
                            stream,
                        )
                    };
                    r.expect(
                        "rlx-rocm: hipBLASLt FP8 GEMM failed (needs CDNA3+ and verified fp8 \
                         constants — see hipblaslt.rs)",
                    );
                }
                Step::ScaledQuantScaleGeneral {
                    x_off_f32,
                    scale_byte_off,
                    rows,
                    cols,
                    fmt,
                    scale_mode,
                    block,
                } => {
                    let nblk = if *scale_mode == 0 {
                        1
                    } else {
                        cols.div_ceil(*block)
                    };
                    let total = if *scale_mode == 0 { 1 } else { rows * nblk };
                    let kernel = crate::kernels::scaled_quant_scale_general_kernel(&self.ctx);
                    let (grid, blk) = dispatch_grid_1d(total, 128);
                    let mut arena_ptr = arena_base;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (blk, 1, 1),
                        [
                            &mut arena_ptr,
                            x_off_f32,
                            scale_byte_off,
                            rows,
                            cols,
                            fmt,
                            scale_mode,
                            block
                        ]
                    );
                }
                Step::ScaledQuantizeGeneral {
                    x_off_f32,
                    scale_byte_off,
                    out_byte_off,
                    rows,
                    cols,
                    fmt,
                    scale_mode,
                    block,
                } => {
                    let total = rows * cols;
                    let kernel = crate::kernels::scaled_quantize_general_kernel(&self.ctx);
                    let (grid, blk) = dispatch_grid_1d(total, 256);
                    let mut arena_ptr = arena_base;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (blk, 1, 1),
                        [
                            &mut arena_ptr,
                            x_off_f32,
                            scale_byte_off,
                            out_byte_off,
                            rows,
                            cols,
                            fmt,
                            scale_mode,
                            block
                        ]
                    );
                }
                Step::ScaledDequantizeGeneral {
                    codes_byte_off,
                    scale_byte_off,
                    out_off_f32,
                    rows,
                    cols,
                    fmt,
                    scale_mode,
                    block,
                } => {
                    let total = rows * cols;
                    let kernel = crate::kernels::scaled_dequantize_general_kernel(&self.ctx);
                    let (grid, blk) = dispatch_grid_1d(total, 256);
                    let mut arena_ptr = arena_base;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (blk, 1, 1),
                        [
                            &mut arena_ptr,
                            codes_byte_off,
                            scale_byte_off,
                            out_off_f32,
                            rows,
                            cols,
                            fmt,
                            scale_mode,
                            block
                        ]
                    );
                }
                Step::ScaledMatMulDecode {
                    m,
                    k,
                    n,
                    lhs_byte_off,
                    rhs_byte_off,
                    lhs_scale_byte_off,
                    rhs_scale_byte_off,
                    out_off_f32,
                    lhs_fmt,
                    rhs_fmt,
                    scale_mode,
                    block,
                    has_bias,
                    bias_off_f32,
                } => {
                    let kernel = crate::kernels::scaled_matmul_decode_kernel(&self.ctx);
                    let mut arena_ptr = arena_base;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        ((*n).div_ceil(16), (*m).div_ceil(16), 1),
                        (16, 16, 1),
                        [
                            &mut arena_ptr,
                            lhs_byte_off,
                            rhs_byte_off,
                            lhs_scale_byte_off,
                            rhs_scale_byte_off,
                            out_off_f32,
                            m,
                            k,
                            n,
                            lhs_fmt,
                            rhs_fmt,
                            scale_mode,
                            block,
                            has_bias,
                            bias_off_f32
                        ]
                    );
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
                } => {
                    // Tier 0: mixed-precision GemmEx — when B is in
                    // the half-arena, cast activations to f16/bf16
                    // and call hipblasGemmEx with both inputs half +
                    // f32 accumulator. Bias / activation epilogue
                    // runs through the shared matmul_epilogue kernel.
                    let used_mixed = try_mixed_precision_gemm_rocm(
                        &self.ctx,
                        &mut self.arena,
                        &mut self.half_act_scratch,
                        self.blas.as_ref(),
                        *m,
                        *k,
                        *n,
                        *batch,
                        *a_off_f32,
                        *b_off_f32,
                        *c_off_f32,
                    );
                    if used_mixed {
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            crate::launch_kernel!(
                                kernel,
                                stream,
                                (grid, 1, 1),
                                (block, 1, 1),
                                [
                                    &mut arena_ptr,
                                    &total,
                                    n,
                                    c_off_f32,
                                    has_bias,
                                    bias_off_f32,
                                    act_id
                                ]
                            );
                        }
                        continue;
                    }

                    // Tier 1: hipBLASLt fused (matmul + bias + relu/gelu
                    // in one launch). Only when activation is one of
                    // the two natively fusable; other acts fall through
                    // to plain sgemm + epilogue kernel. Strided-batch
                    // is handled via LAYOUT_ATTR_BATCH_COUNT /
                    // STRIDED_BATCH_OFFSET in matmul_fused.
                    let try_lt = self.blas_lt.is_some()
                        && self.blas_lt_workspace.is_some()
                        && crate::hipblaslt::act_supported(*act_id);
                    let used_lt = if try_lt {
                        let lt = self.blas_lt.as_ref().unwrap();
                        let workspace = self.blas_lt_workspace.as_ref().unwrap();
                        let epilogue = crate::hipblaslt::epilogue_for(*act_id, *has_bias != 0)
                            .expect("rlx-rocm: act_supported lied");
                        let r = unsafe {
                            crate::hipblaslt::matmul_fused(
                                lt,
                                workspace.ptr,
                                HIPBLASLT_WORKSPACE_BYTES,
                                arena_base,
                                *m,
                                *k,
                                *n,
                                *a_off_f32,
                                *b_off_f32,
                                *c_off_f32,
                                *has_bias != 0,
                                *bias_off_f32,
                                epilogue,
                                *batch,
                                *a_batch_stride,
                                *b_batch_stride,
                                *c_batch_stride,
                                stream,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("matmul.hipblaslt", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_lt {
                        continue;
                    }

                    // Tier 2: hipBLAS sgemm via raw pointers. Same A↔B
                    // swap trick as the cuBLAS path in rlx-cuda — we
                    // compute the column-major transpose of our row-
                    // major matmul, which gives the right result back.
                    let used_hipblas = if let Some(blas) = self.blas.as_ref() {
                        let blas = blas.lock().unwrap();
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        let a_dev = arena_base + (*a_off_f32 as u64) * 4;
                        let b_dev = arena_base + (*b_off_f32 as u64) * 4;
                        let c_dev = arena_base + (*c_off_f32 as u64) * 4;
                        let result = unsafe {
                            if *batch == 1 {
                                (blas.runtime.sgemm)(
                                    blas.handle,
                                    HipblasOperation::N,
                                    HipblasOperation::N,
                                    *n as i32,
                                    *m as i32,
                                    *k as i32,
                                    &alpha as *const f32,
                                    b_dev as *const f32,
                                    *n as i32,
                                    a_dev as *const f32,
                                    *k as i32,
                                    &beta as *const f32,
                                    c_dev as *mut f32,
                                    *n as i32,
                                )
                            } else {
                                (blas.runtime.sgemm_strided)(
                                    blas.handle,
                                    HipblasOperation::N,
                                    HipblasOperation::N,
                                    *n as i32,
                                    *m as i32,
                                    *k as i32,
                                    &alpha as *const f32,
                                    b_dev as *const f32,
                                    *n as i32,
                                    *b_batch_stride as i64,
                                    a_dev as *const f32,
                                    *k as i32,
                                    *a_batch_stride as i64,
                                    &beta as *const f32,
                                    c_dev as *mut f32,
                                    *n as i32,
                                    *c_batch_stride as i64,
                                    *batch as i32,
                                )
                            }
                        };
                        if let Err(e) = result.ok() {
                            log_fallback("matmul.hipblasSgemm", e);
                            false
                        } else {
                            true
                        }
                    } else {
                        false
                    };
                    if used_hipblas {
                        // Optional bias / activation post-pass via the
                        // matmul_epilogue kernel (same shared kernel
                        // as rlx-cuda's cuBLAS path).
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            crate::launch_kernel!(
                                kernel,
                                stream,
                                (grid, 1, 1),
                                (block, 1, 1),
                                [
                                    &mut arena_ptr,
                                    &total,
                                    n,
                                    c_off_f32,
                                    has_bias,
                                    bias_off_f32,
                                    act_id
                                ]
                            );
                        }
                        continue;
                    }

                    // Tier 3: rocWMMA matrix-core kernel. Opt-in via
                    // `RLX_ROCM_MFMA=1`. f16 multiply / f32 accumulate
                    // — bias / activation run through the shared
                    // matmul_epilogue kernel afterward.
                    if use_mfma() {
                        let kernel = matmul_mfma_kernel(&self.ctx);
                        crate::launch_kernel!(
                            kernel,
                            stream,
                            ((*n).div_ceil(32), (*m).div_ceil(32), *batch),
                            (256, 1, 1),
                            [
                                &mut arena_ptr,
                                m,
                                k,
                                n,
                                a_off_f32,
                                b_off_f32,
                                c_off_f32,
                                batch,
                                a_batch_stride,
                                b_batch_stride,
                                c_batch_stride
                            ]
                        );
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            crate::launch_kernel!(
                                kernel,
                                stream,
                                (grid, 1, 1),
                                (block, 1, 1),
                                [
                                    &mut arena_ptr,
                                    &total,
                                    n,
                                    c_off_f32,
                                    has_bias,
                                    bias_off_f32,
                                    act_id
                                ]
                            );
                        }
                        continue;
                    }

                    // Tier 4: custom 64×64 + 4×4 register-tile kernel.
                    let kernel = matmul_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        ((*n).div_ceil(64), (*m).div_ceil(64), *batch),
                        (16, 16, 1),
                        [
                            &mut arena_ptr,
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
                            act_id
                        ]
                    );
                }
                Step::GroupedMatmul {
                    m,
                    k,
                    n,
                    num_experts,
                    in_off,
                    w_off,
                    idx_off,
                    out_off,
                } => {
                    // Tier 1: sorted-batch dispatch via hipBLAS. Direct
                    // port from rlx-cuda — sync the stream so prior
                    // writes to idx are visible, dtoh-copy the idx
                    // buffer, walk it for runs, issue one
                    // hipblasSgemm per run when run count <= m/4.
                    // Random idx falls back to the per-token kernel.
                    let used_sorted = if let Some(blas) = self.blas.as_ref() {
                        unsafe {
                            let _ = (self.ctx.runtime.hip_stream_sync)(stream);
                        }
                        let mn = *m as usize;
                        let mut idx_host = vec![0.0_f32; mn];
                        let idx_dev = arena_base + (*idx_off as u64) * 4;
                        let dtoh_ok = unsafe {
                            (self.ctx.runtime.hip_memcpy_dtoh)(
                                idx_host.as_mut_ptr() as *mut _,
                                idx_dev,
                                mn * 4,
                            )
                            .ok()
                            .is_ok()
                        };
                        if dtoh_ok {
                            let mut runs: Vec<(u32, u32, u32)> = Vec::new();
                            let mut i = 0usize;
                            while i < mn {
                                let e = idx_host[i] as u32;
                                let mut j = i + 1;
                                while j < mn && (idx_host[j] as u32) == e {
                                    j += 1;
                                }
                                if e < *num_experts {
                                    runs.push((i as u32, j as u32, e));
                                }
                                i = j;
                            }
                            let threshold = (mn / 4).max(2);
                            if !runs.is_empty() && runs.len() <= threshold {
                                let blas = blas.lock().unwrap();
                                let alpha: f32 = 1.0;
                                let beta: f32 = 0.0;
                                let mut all_ok = true;
                                for (lo, hi, e) in &runs {
                                    let rows = hi - lo;
                                    let a_dev = arena_base + ((*in_off + lo * k) as u64) * 4;
                                    let b_dev = arena_base + ((*w_off + e * k * n) as u64) * 4;
                                    let c_dev = arena_base + ((*out_off + lo * n) as u64) * 4;
                                    let r = unsafe {
                                        (blas.runtime.sgemm)(
                                            blas.handle,
                                            HipblasOperation::N,
                                            HipblasOperation::N,
                                            *n as i32,
                                            rows as i32,
                                            *k as i32,
                                            &alpha as *const f32,
                                            b_dev as *const f32,
                                            *n as i32,
                                            a_dev as *const f32,
                                            *k as i32,
                                            &beta as *const f32,
                                            c_dev as *mut f32,
                                            *n as i32,
                                        )
                                    };
                                    if r.ok().is_err() {
                                        log_fallback("grouped_matmul.hipblas", r);
                                        all_ok = false;
                                        break;
                                    }
                                }
                                all_ok
                            } else {
                                false
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    if used_sorted {
                        continue;
                    }

                    // Fallback: per-token expert-lookup kernel.
                    let kernel = grouped_matmul_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        (8, 8, 1),
                        [
                            &mut arena_ptr,
                            m,
                            k,
                            n,
                            num_experts,
                            in_off,
                            w_off,
                            idx_off,
                            out_off
                        ]
                    );
                }
                Step::ScatterAddZero { out_off, out_total } => {
                    let kernel = scatter_add_zero_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, out_off, out_total]
                    );
                }
                Step::ScatterAddAcc {
                    out_off,
                    upd_off,
                    idx_off,
                    num_updates,
                    trailing,
                    out_dim,
                } => {
                    let kernel = scatter_add_acc_kernel(&self.ctx);
                    let total = num_updates * trailing;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            out_off,
                            upd_off,
                            idx_off,
                            num_updates,
                            trailing,
                            out_dim
                        ]
                    );
                }
                Step::DequantMatmul {
                    m,
                    k,
                    n,
                    block_size,
                    scheme_id,
                    x_off,
                    w_off,
                    scale_off,
                    zp_off,
                    out_off,
                } => {
                    let kernel = dequant_matmul_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        (8, 8, 1),
                        [
                            &mut arena_ptr,
                            m,
                            k,
                            n,
                            block_size,
                            scheme_id,
                            x_off,
                            w_off,
                            scale_off,
                            zp_off,
                            out_off
                        ]
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
                    let use_gpu = self.dequant_scratch_off > 0 && self.blas.is_some();
                    if use_gpu {
                        let blas = self.blas.as_ref().unwrap();
                        crate::gguf_gpu::run_dequant_matmul_gguf_gpu(
                            &self.ctx,
                            stream,
                            &self.arena.buffer,
                            blas,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            self.dequant_scratch_off,
                            *out_byte_off as usize,
                        );
                    } else {
                        crate::gguf_host::run_dequant_matmul_gguf(
                            &self.ctx,
                            &self.arena.buffer,
                            *m as usize,
                            *k as usize,
                            *n as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    }
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
                    let use_gpu = self.dequant_scratch_off > 0 && self.blas.is_some();
                    if use_gpu {
                        let blas = self.blas.as_ref().unwrap();
                        unsafe {
                            crate::gguf_gpu::run_dequant_grouped_matmul_gguf_gpu(
                                &self.ctx,
                                stream,
                                &self.arena.buffer,
                                blas,
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
                        }
                    } else {
                        crate::gguf_host::run_dequant_grouped_matmul_gguf(
                            &self.ctx,
                            &self.arena.buffer,
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
                Step::SelectiveScan {
                    batch,
                    seq,
                    hidden,
                    state_size,
                    x_off,
                    delta_off,
                    a_off,
                    b_off,
                    c_off,
                    out_off,
                } => {
                    let kernel = selective_scan_kernel(&self.ctx);
                    let total = batch * hidden;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            batch,
                            seq,
                            hidden,
                            state_size,
                            x_off,
                            delta_off,
                            a_off,
                            b_off,
                            c_off,
                            out_off
                        ]
                    );
                }
                Step::Fft {
                    src_byte_off,
                    dst_byte_off,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    dtype_tag,
                    use_gpu,
                } => {
                    if *use_gpu {
                        let norm = rlx_ir::fft::FftNorm::from_tag(*norm_tag);
                        let scale = norm.output_scale(*n_complex as usize, *inverse) as f32;
                        crate::fft_dispatch::run_fft_gpu(
                            &self.ctx,
                            stream,
                            arena_ptr,
                            *src_byte_off / 4,
                            *dst_byte_off / 4,
                            *outer,
                            *n_complex,
                            *inverse,
                            scale,
                        );
                    } else {
                        crate::fft_host::run_fft1d(
                            &self.ctx,
                            &self.arena.buffer,
                            self.arena.size,
                            *src_byte_off as usize,
                            *dst_byte_off as usize,
                            *outer as usize,
                            *n_complex as usize,
                            *inverse,
                            *norm_tag,
                            rocm_fft_dtype_from_tag(*dtype_tag),
                        );
                    }
                }
                Step::WelchPeaksGpu {
                    spec_off,
                    dst_off,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                    n_bins,
                } => {
                    crate::welch_peaks_dispatch::run_welch_peaks_gpu(
                        &self.ctx,
                        stream,
                        arena_ptr,
                        *spec_off,
                        *dst_off,
                        *welch_batch,
                        *n_fft,
                        *n_segments,
                        *k,
                        *n_bins,
                    );
                }
                Step::LogMelHost { .. }
                | Step::LogMelBackwardHost { .. }
                | Step::WelchPeaksHost { .. } => {}
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
                    use_gpu,
                } => {
                    if *use_gpu {
                        let kernel = im2col_kernel(&self.ctx);
                        let m_dim = *n * *h_out * *w_out;
                        let k_dim = *c_in * *kh * *kw;
                        let total = m_dim * k_dim;
                        let (grid, block) = dispatch_grid_1d(total, 256);
                        let x_off = *x_byte_off / 4;
                        let col_off = *col_byte_off / 4;
                        crate::launch_kernel!(
                            kernel,
                            stream,
                            (grid, 1, 1),
                            (block, 1, 1),
                            [
                                &mut arena_ptr,
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
                                &x_off,
                                &col_off
                            ]
                        );
                    } else {
                        crate::im2col_host::run_im2col(
                            &self.ctx,
                            &self.arena.buffer,
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
                }
                Step::ReverseHost {
                    src_byte_off,
                    dst_byte_off,
                    dims,
                    rev_mask,
                    elem_bytes,
                } => {
                    crate::host_misc::run_reverse(
                        &self.ctx,
                        &self.arena.buffer,
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
                    crate::host_misc::run_argreduce(
                        &self.ctx,
                        &self.arena.buffer,
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
                    crate::host_misc::run_axial_rope2d(
                        &self.ctx,
                        &self.arena.buffer,
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
                Step::GatedDeltaNet {
                    q_byte_off,
                    k_byte_off,
                    v_byte_off,
                    g_byte_off,
                    beta_byte_off,
                    state_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    heads,
                    state_size,
                    use_carry,
                } => {
                    crate::gdn_host::run_gated_delta_net(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *q_byte_off as usize,
                        *k_byte_off as usize,
                        *v_byte_off as usize,
                        *g_byte_off as usize,
                        *beta_byte_off as usize,
                        *state_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *heads as usize,
                        *state_size as usize,
                        *use_carry,
                    );
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
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
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
                Step::ScanHost { desc } => {
                    crate::scan_host::run_scan(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        desc,
                    );
                }
                Step::HostOp { desc } => {
                    crate::scan_host::run_host_op(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        desc,
                    );
                }
                Step::CpuIndexing { thunk } => {
                    crate::scan_host::run_indexing(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        thunk,
                    );
                }
                Step::SpdHost {
                    op,
                    out_off,
                    out_shape,
                    inputs,
                } => {
                    crate::spd_host::run_spd(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        op,
                        *out_off,
                        out_shape,
                        inputs,
                    );
                }
                Step::EighNative {
                    in_off,
                    out_off,
                    n,
                    batch,
                } => {
                    crate::eigh_native::run(
                        &self.ctx,
                        stream,
                        &self.arena.buffer,
                        *in_off,
                        *out_off,
                        *n,
                        *batch,
                    );
                }
                Step::Llada2GroupLimitedGate {
                    sig_off,
                    route_off,
                    out_off,
                    n_elems,
                    attrs,
                } => {
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *sig_off as usize,
                        *route_off as usize,
                        *out_off as usize,
                        *n_elems as usize,
                        attrs,
                    );
                }
                Step::MsDeformAttnHost {
                    in_offs,
                    out_off,
                    out_len,
                    attrs,
                } => {
                    crate::ms_deform_attn_host::run_ms_deform_attn(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        in_offs,
                        *out_off as usize,
                        *out_len as usize,
                        attrs,
                    );
                }
                Step::CollectiveHost {
                    name,
                    in_off,
                    in_len,
                    out_off,
                    out_len,
                    attrs,
                } => {
                    crate::collective_host::run_collective(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        name,
                        *in_off as usize,
                        *in_len as usize,
                        *out_off as usize,
                        *out_len as usize,
                        attrs,
                    );
                }
                Step::UmapKnn {
                    pairwise_off,
                    out_off,
                    n,
                    k,
                } => {
                    crate::umap_knn_host::run_umap_knn(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *pairwise_off as usize,
                        *out_off as usize,
                        *n as usize,
                        *k as usize,
                    );
                }
                Step::GaussianSplatRender {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    dst_off,
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
                    #[cfg(feature = "native-splat")]
                    crate::splat_native::run_gaussian_splat_render_native(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *dst_off as usize,
                        *width,
                        *height,
                        *tile_size,
                        *radius_scale,
                        *alpha_cutoff,
                        *max_splat_steps,
                        *transmittance_threshold,
                        *max_list_entries,
                    );
                    #[cfg(not(feature = "native-splat"))]
                    crate::splat_host::run_gaussian_splat_render(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *dst_off as usize,
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
                Step::GaussianSplatRenderBackward {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    d_loss_off,
                    d_loss_len,
                    packed_off,
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
                    crate::splat_host::run_gaussian_splat_render_backward(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *d_loss_off as usize,
                        *d_loss_len as usize,
                        *packed_off as usize,
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
                Step::GaussianSplatPrepare {
                    positions_off,
                    positions_len,
                    scales_off,
                    scales_len,
                    rotations_off,
                    rotations_len,
                    opacities_off,
                    opacities_len,
                    colors_off,
                    colors_len,
                    sh_coeffs_off,
                    sh_coeffs_len,
                    meta_off,
                    meta_len,
                    prep_off,
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
                    crate::splat_host::run_gaussian_splat_prepare(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *positions_off as usize,
                        *positions_len as usize,
                        *scales_off as usize,
                        *scales_len as usize,
                        *rotations_off as usize,
                        *rotations_len as usize,
                        *opacities_off as usize,
                        *opacities_len as usize,
                        *colors_off as usize,
                        *colors_len as usize,
                        *sh_coeffs_off as usize,
                        *sh_coeffs_len as usize,
                        *meta_off as usize,
                        *meta_len as usize,
                        *prep_off as usize,
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
                Step::GaussianSplatRasterize {
                    prep_off,
                    prep_len,
                    meta_off,
                    meta_len,
                    dst_off,
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
                    crate::splat_host::run_gaussian_splat_rasterize(
                        &self.ctx,
                        &self.arena.buffer,
                        self.arena.size,
                        *prep_off as usize,
                        *prep_len as usize,
                        *meta_off as usize,
                        *meta_len as usize,
                        *dst_off as usize,
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
                Step::RmsNormBackwardInput {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    let x_off = *x_byte_off / 4;
                    let gamma_off = *gamma_byte_off / 4;
                    let beta_off = *beta_byte_off / 4;
                    let dy_off = *dy_byte_off / 4;
                    let dx_off = *dx_byte_off / 4;
                    let wrt = 0u32;
                    let kernel = rms_norm_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*rows, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            rows,
                            h,
                            &x_off,
                            &gamma_off,
                            &beta_off,
                            &dy_off,
                            &dx_off,
                            eps_bits,
                            &wrt
                        ]
                    );
                }
                Step::RmsNormBackwardGamma {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dgamma_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    let x_off = *x_byte_off / 4;
                    let gamma_off = *gamma_byte_off / 4;
                    let beta_off = *beta_byte_off / 4;
                    let dy_off = *dy_byte_off / 4;
                    let dgamma_off = *dgamma_byte_off / 4;
                    let wrt = 1u32;
                    let zk = rms_norm_bwd_zero_kernel(&self.ctx);
                    let (zgrid, zblock) = dispatch_grid_1d(*h, 256);
                    crate::launch_kernel!(
                        zk,
                        stream,
                        (zgrid, 1, 1),
                        (zblock, 1, 1),
                        [&mut arena_ptr, &dgamma_off, h]
                    );
                    let kernel = rms_norm_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*rows, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            rows,
                            h,
                            &x_off,
                            &gamma_off,
                            &beta_off,
                            &dy_off,
                            &dgamma_off,
                            eps_bits,
                            &wrt
                        ]
                    );
                }
                Step::RmsNormBackwardBeta {
                    x_byte_off,
                    gamma_byte_off,
                    beta_byte_off,
                    dy_byte_off,
                    dbeta_byte_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    let x_off = *x_byte_off / 4;
                    let gamma_off = *gamma_byte_off / 4;
                    let beta_off = *beta_byte_off / 4;
                    let dy_off = *dy_byte_off / 4;
                    let dbeta_off = *dbeta_byte_off / 4;
                    let wrt = 2u32;
                    let zk = rms_norm_bwd_zero_kernel(&self.ctx);
                    let (zgrid, zblock) = dispatch_grid_1d(*h, 256);
                    crate::launch_kernel!(
                        zk,
                        stream,
                        (zgrid, 1, 1),
                        (zblock, 1, 1),
                        [&mut arena_ptr, &dbeta_off, h]
                    );
                    let kernel = rms_norm_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*rows, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            rows,
                            h,
                            &x_off,
                            &gamma_off,
                            &beta_off,
                            &dy_off,
                            &dbeta_off,
                            eps_bits,
                            &wrt
                        ]
                    );
                }
                Step::RopeBackward {
                    dy_byte_off,
                    cos_byte_off,
                    sin_byte_off,
                    dx_byte_off,
                    batch,
                    seq,
                    hidden,
                    head_dim,
                    n_rot,
                    cos_len,
                } => {
                    let dy_off = *dy_byte_off / 4;
                    let cos_off = *cos_byte_off / 4;
                    let sin_off = *sin_byte_off / 4;
                    let dx_off = *dx_byte_off / 4;
                    let kernel = rope_backward_kernel(&self.ctx);
                    let total = batch * seq * hidden;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            batch,
                            seq,
                            hidden,
                            head_dim,
                            n_rot,
                            &dy_off,
                            &cos_off,
                            &sin_off,
                            &dx_off,
                            cos_len
                        ]
                    );
                }
                Step::CumsumBackward {
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    cols,
                    exclusive,
                } => {
                    let dy_off = *dy_byte_off / 4;
                    let dx_off = *dx_byte_off / 4;
                    let excl = if *exclusive { 1u32 } else { 0u32 };
                    let kernel = cumsum_backward_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*rows, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, rows, cols, &dy_off, &dx_off, &excl]
                    );
                }
                Step::GatherBackward {
                    dy_byte_off,
                    indices_byte_off,
                    dst_byte_off,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                } => {
                    let dy_off = *dy_byte_off / 4;
                    let idx_off = *indices_byte_off / 4;
                    let dst_off = *dst_byte_off / 4;
                    let total = *outer * *axis_dim * *trailing;
                    if total > 0 {
                        let zk = rms_norm_bwd_zero_kernel(&self.ctx);
                        let (zgrid, zblock) = dispatch_grid_1d(total, 256);
                        crate::launch_kernel!(
                            zk,
                            stream,
                            (zgrid, 1, 1),
                            (zblock, 1, 1),
                            [&mut arena_ptr, &dst_off, &total]
                        );
                    }
                    let kernel = gather_backward_kernel(&self.ctx);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (*outer, (num_idx * trailing).div_ceil(256), 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            outer,
                            axis_dim,
                            num_idx,
                            trailing,
                            &dy_off,
                            &idx_off,
                            &dst_off
                        ]
                    );
                }
                Step::LayerNorm2d {
                    src_off,
                    g_off,
                    b_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                    eps_bits,
                } => {
                    let kernel = layer_norm2d_kernel(&self.ctx);
                    let total = n * h * w;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            src_off,
                            g_off,
                            b_off,
                            dst_off,
                            n,
                            c,
                            h,
                            w,
                            eps_bits
                        ]
                    );
                }
                Step::ConvTranspose2d {
                    src_off,
                    w_off,
                    dst_off,
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
                    let kernel = conv_transpose2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            src_off,
                            w_off,
                            dst_off,
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
                            groups
                        ]
                    );
                }
                Step::GroupNorm {
                    src_off,
                    g_off,
                    b_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps_bits,
                } => {
                    let kernel = group_norm_kernel(&self.ctx);
                    let grid = n * num_groups;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (256, 1, 1),
                        [
                            &mut arena_ptr,
                            src_off,
                            g_off,
                            b_off,
                            dst_off,
                            n,
                            c,
                            h,
                            w,
                            num_groups,
                            eps_bits
                        ]
                    );
                }
                Step::ResizeNearest2x {
                    src_off,
                    dst_off,
                    n,
                    c,
                    h,
                    w,
                } => {
                    let kernel = resize_nearest_2x_kernel(&self.ctx);
                    let total = n * c * h * 2 * w * 2;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, src_off, dst_off, n, c, h, w]
                    );
                }
                Step::ComplexCast {
                    n,
                    in_byte_off,
                    out_byte_off,
                    mode,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = complex_cast_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    // f32-element offsets as u64 — the kernel declares its offset
                    // params `unsigned long long`, so passing a u32 here would
                    // leave the high word as stack garbage → illegal address.
                    let in_off: u64 = (*in_byte_off / 4) as u64;
                    let out_off: u64 = (*out_byte_off / 4) as u64;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, &in_off, &out_off, mode]
                    );
                }
                Step::BinaryC64 {
                    n,
                    a_byte_off,
                    b_byte_off,
                    c_byte_off,
                    op,
                    n_a,
                    n_b,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = binary_c64_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    // f32-element offsets as u64 (kernel params are u64 — see above).
                    let a_off: u64 = (*a_byte_off / 4) as u64;
                    let b_off: u64 = (*b_byte_off / 4) as u64;
                    let c_off: u64 = (*c_byte_off / 4) as u64;
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [&mut arena_ptr, &n_s, &a_off, &b_off, &c_off, op, n_a, n_b]
                    );
                }
                Step::Pool1d {
                    n,
                    c,
                    l,
                    l_out,
                    kl,
                    sl,
                    pl,
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool1d_kernel(&self.ctx);
                    let total = n * c * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n,
                            c,
                            l,
                            l_out,
                            kl,
                            sl,
                            pl,
                            op,
                            in_off,
                            out_off
                        ]
                    );
                }
                Step::Pool2d {
                    n,
                    c,
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
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool2d_kernel(&self.ctx);
                    let total = n * c * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n,
                            c,
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
                            op,
                            in_off,
                            out_off
                        ]
                    );
                }
                Step::Pool3d {
                    n,
                    c,
                    d,
                    h,
                    w,
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
                    op,
                    in_off,
                    out_off,
                } => {
                    let kernel = pool3d_kernel(&self.ctx);
                    let total = n * c * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n,
                            c,
                            d,
                            h,
                            w,
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
                            op,
                            in_off,
                            out_off
                        ]
                    );
                }
                Step::Conv1d {
                    n,
                    c_in,
                    c_out,
                    l,
                    l_out,
                    kl,
                    sl,
                    pl,
                    dl,
                    groups,
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: MIOpen forward conv as a degenerate 2-D
                    // conv (H=kh=sh=1, ph=0, dh=1). Same trick rlx-cuda
                    // uses in conv1d.
                    let used_miopen = if let (Some(dnn), Some(workspace)) =
                        (self.dnn.as_ref(), self.dnn_workspace.as_ref())
                    {
                        let r = unsafe {
                            crate::miopen::conv2d_forward(
                                dnn,
                                workspace.ptr,
                                MIOPEN_WORKSPACE_BYTES,
                                arena_base,
                                *n,
                                *c_in,
                                *c_out,
                                /*h*/ 1,
                                *l,
                                /*h_out*/ 1,
                                *l_out,
                                /*kh*/ 1,
                                *kl,
                                /*sh*/ 1,
                                *sl,
                                /*ph*/ 0,
                                *pl,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv1d.miopen", e);
                        }
                        r.is_ok() && *dl == 1
                    } else {
                        false
                    };
                    if used_miopen {
                        continue;
                    }

                    let kernel = conv1d_kernel(&self.ctx);
                    let total = n * c_out * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n,
                            c_in,
                            c_out,
                            l,
                            l_out,
                            kl,
                            sl,
                            pl,
                            dl,
                            groups,
                            in_off,
                            w_off,
                            out_off
                        ]
                    );
                }
                Step::Conv2d {
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
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: MIOpen forward conv. Bounded to dilation=1
                    // for now since MIOpen's miopenInitConvolutionDescriptor
                    // takes a dilation_h/dilation_w pair (no nd version
                    // here); when dh != 1 || dw != 1 we fall through.
                    let used_miopen = if let (Some(dnn), Some(workspace), 1, 1) =
                        (self.dnn.as_ref(), self.dnn_workspace.as_ref(), *dh, *dw)
                    {
                        let r = unsafe {
                            crate::miopen::conv2d_forward(
                                dnn,
                                workspace.ptr,
                                MIOPEN_WORKSPACE_BYTES,
                                arena_base,
                                *n,
                                *c_in,
                                *c_out,
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
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d.miopen", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_miopen {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
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
                            in_off,
                            w_off,
                            out_off
                        ]
                    );
                }
                Step::Conv3d {
                    n,
                    c_in,
                    c_out,
                    d,
                    h,
                    w,
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
                    in_off,
                    w_off,
                    out_off,
                } => {
                    // Tier 1: MIOpen nd-conv. NCDHW input/output, 5-D
                    // KCDHW filter, 3-D pads/strides/dilations.
                    let used_miopen = if let (Some(dnn), Some(workspace)) =
                        (self.dnn.as_ref(), self.dnn_workspace.as_ref())
                    {
                        let r = unsafe {
                            crate::miopen::conv3d_forward(
                                dnn,
                                workspace.ptr,
                                MIOPEN_WORKSPACE_BYTES,
                                arena_base,
                                *n,
                                *c_in,
                                *c_out,
                                *d,
                                *h,
                                *w,
                                *d_out,
                                *h_out,
                                *w_out,
                                *kd,
                                *kh,
                                *kw,
                                *sd,
                                *sh,
                                *sw,
                                *pd,
                                *ph,
                                *pw,
                                *dd,
                                *dh,
                                *dw,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv3d.miopen", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_miopen {
                        continue;
                    }

                    let kernel = conv3d_kernel(&self.ctx);
                    let total = n * c_out * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    crate::launch_kernel!(
                        kernel,
                        stream,
                        (grid, 1, 1),
                        (block, 1, 1),
                        [
                            &mut arena_ptr,
                            n,
                            c_in,
                            c_out,
                            d,
                            h,
                            w,
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
                            in_off,
                            w_off,
                            out_off
                        ]
                    );
                }
            }

            // Multi-stream tail: record an event so future steps can
            // wait on this one, then update producer_of with the
            // offsets this step wrote.
            if let Some(idx) = assigned_idx {
                let mut evt: crate::hip::HipEvent = std::ptr::null_mut();
                unsafe {
                    if (self.ctx.runtime.hip_event_create)(&mut evt, 0)
                        .ok()
                        .is_ok()
                    {
                        let _ = (self.ctx.runtime.hip_event_record)(evt, stream);
                        // Replace any older event for this stream.
                        if let Some(prev) = last_event.insert(idx, evt) {
                            let _ = (self.ctx.runtime.hip_event_destroy)(prev);
                        }
                    }
                }
                let (_, writes) = step_offsets(step);
                for w in &writes {
                    producer_of.insert(*w, idx);
                }
            }
        }

        // Multi-stream: sync every pool stream + clean up events so
        // output reads see all produced data.
        if multi_stream {
            for s in &self.streams {
                unsafe {
                    let _ = (self.ctx.runtime.hip_stream_sync)(*s);
                }
            }
            for (_, evt) in last_event.drain() {
                unsafe {
                    let _ = (self.ctx.runtime.hip_event_destroy)(evt);
                }
            }
        }

        if do_capture {
            unsafe {
                let mut graph: crate::hip::HipGraph = std::ptr::null_mut();
                let mut graph_exec: crate::hip::HipGraphExec = std::ptr::null_mut();
                if (self.ctx.runtime.hip_stream_end_capture)(stream, &mut graph)
                    .ok()
                    .is_ok()
                    && !graph.is_null()
                {
                    let r = (self.ctx.runtime.hip_graph_instantiate)(
                        &mut graph_exec,
                        graph,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        0,
                    );
                    let _ = (self.ctx.runtime.hip_graph_destroy)(graph);
                    if r.ok().is_ok() {
                        // First-run launch: actually compute outputs.
                        let _ = (self.ctx.runtime.hip_graph_launch)(graph_exec, stream);
                        self.captured_graph = Some(graph_exec);
                    }
                }
            }
        }

        // Sync stream + read outputs.
        unsafe {
            let _ = (self.ctx.runtime.hip_stream_sync)(stream);
        }
        self.run_tail_host_audio_ops(false);
        self.finalize_outputs()
    }

    pub(crate) fn run_tail_host_audio_ops(&self, pre_sync: bool) {
        if !self.schedule.iter().any(step_is_tail_host) {
            return;
        }
        if pre_sync {
            unsafe {
                let _ = (self.ctx.runtime.hip_stream_sync)(self.ctx.default_stream);
            }
        }
        for step in &self.schedule {
            match step {
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
                        &self.ctx,
                        &self.arena.buffer,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                        false,
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
                    crate::log_mel_backward_host::run_log_mel_backward(
                        &self.ctx,
                        &self.arena.buffer,
                        *spec_byte_off as usize,
                        *filt_byte_off as usize,
                        *dy_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *n_fft as usize,
                        *n_bins as usize,
                        *n_mels as usize,
                        false,
                    );
                }
                Step::WelchPeaksHost {
                    spec_byte_off,
                    dst_byte_off,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                } => {
                    crate::welch_peaks_host::run_welch_peaks(
                        &self.ctx,
                        &self.arena.buffer,
                        *spec_byte_off as usize,
                        *dst_byte_off as usize,
                        *welch_batch as usize,
                        *n_fft as usize,
                        *n_segments as usize,
                        *k as usize,
                        false,
                    );
                }
                _ => {}
            }
        }
    }
}
