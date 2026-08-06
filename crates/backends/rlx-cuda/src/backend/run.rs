// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! `run` — extracted from the `backend` module for navigability (see `mod.rs`).
#![allow(unused_imports)]

use crate::arena::{Arena, plan_f32_uniform};
use crate::device::{
    CUBLASLT_WORKSPACE_BYTES, CUDNN_WORKSPACE_BYTES, cuda_blas, cuda_blas_lt_handle,
    cuda_blas_lt_workspace, cuda_context, cuda_dnn_handle, cuda_dnn_workspace,
    record_conv_transpose3d_path, record_conv3d_bwd_path, record_conv3d_path,
};
use crate::host_staging::F32HostSlot;
use crate::kernels::{
    activation_backward_kernel, ada_layer_norm_backward_kernel, ada_layer_norm_kernel,
    argmax_kernel, attention_bwd_kernel, attention_kernel, attention_row_kernel,
    attention_wmma_d128_kernel, attention_wmma_kernel, axial_rope2d_kernel,
    batch_elementwise_region_kernel, batch_norm_inference_bwd_beta_kernel,
    batch_norm_inference_bwd_gamma_kernel, batch_norm_inference_bwd_input_kernel,
    batch_norm_inference_kernel, binary_broadcast_kernel, binary_c64_kernel, binary_kernel,
    compare_kernel, complex_cast_kernel, complex_norm_sq_backward_kernel, complex_norm_sq_kernel,
    concat_kernel, conjugate_c64_kernel, conv_bias_act_epilogue_kernel, conv_transpose2d_kernel,
    conv_transpose3d_kernel, conv1d_kernel, conv2d_backward_input_kernel,
    conv2d_backward_weight_kernel, conv2d_kernel, conv3d_backward_input_kernel,
    conv3d_backward_weight_kernel, conv3d_kernel, copy_kernel, cum_scan_kernel,
    cumsum_backward_kernel, cumsum_kernel, dequant_matmul_kernel, dequant_matmul_mlx_gemm_kernel,
    dequant_matmul_mlx_gemv_kernel, dequant_matmul_mlx_kernel, dequantize_i8_kernel,
    dispatch_grid_1d, dispatch_grid_prologue_nchw, elementwise_region_kernel, expand_kernel,
    fake_quantize_backward_kernel, fake_quantize_ema_kernel, fake_quantize_fixed_kernel,
    fake_quantize_lsq_bwd_scale_kernel, fake_quantize_lsq_bwd_x_kernel,
    fake_quantize_perbatch_kernel, fft_butterfly_stage_kernel, fma_kernel, fused_attn_kernel,
    fused_binary_unary_kernel, fused_residual_ln_kernel, fused_residual_rms_norm_kernel,
    fused_swiglu_kernel, gated_delta_net_kernel, gated_residual_backward_kernel,
    gated_residual_kernel, gather_axis_kernel, gather_backward_kernel, gather_kernel,
    group_norm_bwd_beta_kernel, group_norm_bwd_gamma_kernel, group_norm_bwd_input_kernel,
    group_norm_kernel, grouped_matmul_kernel, im2col_kernel, interpolate3d_kernel,
    kimi_delta_chunk_kernel, layer_norm_bwd_gamma_kernel, layer_norm_bwd_input_kernel,
    layer_norm2d_kernel, layernorm_kernel, matmul_epilogue_kernel, matmul_kernel,
    matmul_tma_kernel, matmul_wmma_kernel, maxpool2d_backward_kernel, maxpool3d_backward_kernel,
    narrow_kernel, pad_kernel, pool1d_kernel, pool2d_kernel, pool3d_kernel, q_conv2d_kernel,
    q_matmul_kernel, quantize_i8_kernel, reduce_kernel, relu_backward_kernel,
    resize_nearest_2x_kernel, rms_norm_backward_kernel, rms_norm_bwd_zero_kernel,
    rope_backward_kernel, rope_kernel, sample_kernel, scatter_add_acc_kernel,
    scatter_add_zero_kernel, selective_scan_kernel, slice_kernel,
    softmax_cross_entropy_backward_kernel, softmax_cross_entropy_kernel,
    softmax_cross_entropy_with_logits_kernel, softmax_kernel, topk_kernel, transpose_kernel,
    unary_kernel, where_kernel,
};
use cudarc::cublas::{CudaBlas, sys as cublas_sys};
use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::cudnn::{result as cudnn_result, sys as cudnn_sys};
use cudarc::driver::{CudaContext, DevicePtrMut, LaunchConfig, PushKernelArg};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::rlx_fusion::lower_reduce_axes::LowerNonLastAxisReduce;
use rlx_opt::rlx_fusion::pass::Pass as _;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use super::*;

impl CudaExecutable {
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
        match read_indices {
            None => self.pending_read_indices = None,
            Some(ix) => {
                let buf = self.pending_read_indices.get_or_insert_with(Vec::new);
                buf.clear();
                buf.extend_from_slice(ix);
                normalize_read_indices(buf);
            }
        }
        let outs = self.run_inner(inputs);
        self.pending_read_indices = None;
        // NaN/Inf output-boundary scan (RLX_DEBUG_NANS). CUDA runs op-by-op on
        // the device; per-op D2H would perturb timing, so we scan the outputs
        // here (when reading all of them, where they align with graph.outputs).
        // For internal localization replay the same graph on the CPU backend.
        let scanner = rlx_ir::numeric_check::DebugScanner::from_env("cuda");
        if scanner.enabled() && read_indices.is_none() {
            for (buf, &id) in outs.iter().zip(self.graph.outputs.iter()) {
                scanner.check(&self.graph, id, buf, &[]);
            }
        }
        self.dump_cuda_outputs_if_requested(&outs);
        outs
    }

    /// Output-buffer dump (`RLX_CUDA_DUMP_NODES=1`) for Metal/CPU cross-diff.
    /// Dumps host-side graph outputs only (already read back) — cheap enough
    /// for packed Qwen35. Intermediate arena D2H is opt-in via
    /// `RLX_CUDA_DUMP_INTERMEDIATE=1` with `RLX_CUDA_DUMP_NODES_LIMIT`.
    pub(crate) fn dump_cuda_outputs_if_requested(&self, outs: &[Vec<f32>]) {
        if !rlx_ir::env::flag("RLX_CUDA_DUMP_NODES") {
            return;
        }
        let limit = rlx_ir::env::parse_or("RLX_CUDA_DUMP_NODES_LIMIT", 64usize);
        eprintln!(
            "[rlx-cuda-dump] graph outputs max|x| (limit={limit}); set RLX_CUDA_DUMP_INTERMEDIATE=1 for arena nodes"
        );
        for (i, (buf, &id)) in outs.iter().zip(self.graph.outputs.iter()).enumerate() {
            if i >= limit {
                break;
            }
            let max = buf.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let nz = buf.iter().filter(|&&v| v != 0.0).count();
            let nan = buf.iter().filter(|&&v| v.is_nan()).count();
            let op = &self.graph.node(id).op;
            eprintln!(
                "  [out {i:>3}] {:?} id={} max={max:.6} nz={nz}/{} nan={nan}",
                op,
                id.0,
                buf.len()
            );
        }
        if !rlx_ir::env::flag("RLX_CUDA_DUMP_INTERMEDIATE") {
            return;
        }
        // Expensive: D2H a capped set of non-input F32 arena slots after sync.
        let stream = self.ctx.default_stream();
        let _ = stream.synchronize();
        let mut shown = 0usize;
        for (i, node) in self.graph.nodes().iter().enumerate() {
            if shown >= limit {
                break;
            }
            if !self.arena.has(node.id) {
                continue;
            }
            // `RLX_CUDA_DUMP_IO=1` also dumps Input/Param/Constant slots — needed
            // to tell "input not landing in slot" from "param not loaded" from a
            // genuine compute-op zero. Reshape/Cast stay skipped (they alias the
            // parent slot, so their value is the parent's, already shown).
            let dump_io = rlx_ir::env::flag("RLX_CUDA_DUMP_IO");
            let is_io = matches!(
                node.op,
                rlx_ir::Op::Input { .. } | rlx_ir::Op::Param { .. } | rlx_ir::Op::Constant { .. }
            );
            if matches!(
                node.op,
                rlx_ir::Op::Reshape { .. } | rlx_ir::Op::Cast { .. }
            ) || (is_io && !dump_io)
            {
                continue;
            }
            let n = node.shape.num_elements().unwrap_or(0);
            if n == 0 || n > 1_048_576 {
                continue;
            }
            let off_f32 = self.arena.offset(node.id) / 4;
            let mut host = vec![0f32; n];
            let slice = self.arena.f32_buf().slice(off_f32..off_f32 + n);
            if stream.memcpy_dtoh(&slice, &mut host).is_err() {
                continue;
            }
            let max = host.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let nz = host.iter().filter(|&&v| v != 0.0).count();
            let nan = host.iter().filter(|&&v| v.is_nan()).count();
            eprintln!(
                "  [{i:>3}] {:?} max={max:.6} nz={nz}/{} nan={nan}",
                node.op,
                host.len()
            );
            shown += 1;
        }
    }

    pub(crate) fn run_inner(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        let default_stream = self.ctx.default_stream();
        // ANY CUDA graph capture (whole-graph `ExecMode::Graph` OR segmented)
        // routes the ENTIRE run — input H2D copy, dispatch, capture — onto a
        // dedicated non-null stream, because the legacy null default stream is
        // (a) uncapturable and (b) would make captured kernels depend on the
        // null-stream input copy → `CAPTURE_ISOLATION`. This is the fix for
        // rlx's latent whole-graph no-op: `ExecMode::Graph` previously called
        // `begin_capture` on the null stream, silently failed, and ran eager.
        // Created here (before the input copy) so the copy lands on it too;
        // reused as `seg_stream` in the dispatch loop below. Off unless graph
        // mode is requested → default behaviour byte-identical.
        let want_capture_stream = self.exec_mode == ExecMode::Graph
            && self.active_extent.is_none()
            && !self.capture_disabled
            && ((rlx_ir::env::flag("RLX_CUDA_SEGMENTED_CAPTURE")
                && rlx_ir::env::flag("RLX_CUDA_SEGMENTED_CAPTURE_ENGAGE"))
                // Whole-graph capture is its OWN opt-in: it now begins capture on
                // the non-null stream (the null-stream no-op is fixed), but a
                // cuBLAS-internal op still async-invalidates the capture on
                // multi-matmul graphs (end_capture → CAPTURE_INVALIDATED; falls
                // back to eager gracefully). Kept opt-in until that's resolved so
                // default `ExecMode::Graph` stays byte-identical (silent eager).
                || rlx_ir::env::flag("RLX_CUDA_WHOLE_GRAPH_CAPTURE"));
        if want_capture_stream && self.segment_stream.is_none() {
            self.segment_stream = self.ctx.new_stream().ok();
        }
        // Creating a stream via `new_stream()` swaps the context into
        // multi-stream mode, which turns on cudarc's per-`CudaSlice` cross-stream
        // event tracking: every `.arg(arena)` then inserts a wait on the slice's
        // last-use stream. During capture that wait targets uncaptured null-
        // stream work → `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`. The whole
        // segmented dispatch is single-stream (everything ordered on
        // `seg_stream`), so we disable that auto-sync for the run — stream order
        // alone is sufficient — and re-enable it after the loop.
        if want_capture_stream && self.segment_stream.is_some() {
            unsafe { self.ctx.disable_event_tracking() };
        }
        let stream = if want_capture_stream {
            self.segment_stream
                .clone()
                .unwrap_or_else(|| default_stream.clone())
        } else {
            default_stream.clone()
        };

        self.stage_gpu_handle_inputs(&stream, inputs);

        // Copy inputs to device. Always done outside any graph capture
        // — inputs change between runs and shouldn't be baked into the
        // captured CUDA Graph.
        let input_diag = rlx_ir::env::flag("RLX_CUDA_INPUT_DIAG");
        for &(name, data) in inputs {
            if let Some(&id) = self.input_offsets.get(name)
                && self.arena.has(id)
            {
                let off_f32 = self.arena.offset(id) / 4;
                let mut slot = self
                    .arena
                    .f32_buf_mut()
                    .slice_mut(off_f32..off_f32 + data.len());
                if let Some(host) = self.input_staging.get_mut(name) {
                    host.copy_from_host(data);
                    host.htod(&stream, &mut slot, data.len())
                        .expect("rlx-cuda: pinned input upload failed");
                } else {
                    stream
                        .memcpy_htod(data, &mut slot)
                        .expect("rlx-cuda: input upload failed");
                }
                if input_diag {
                    eprintln!("[cuda-input] {name:?} -> uploaded ({} f32)", data.len());
                }
            } else if input_diag {
                // A provided input that is NOT uploaded here means the graph will
                // read zero-initialised memory for it (the split-graph TTS silent
                // bug). Distinguish the cause. gpu-handle inputs are staged
                // separately (stage_gpu_handle_inputs) and are expected here.
                let reason = match self.input_offsets.get(name) {
                    None if self.has_gpu_handle(name) => "gpu-handle (staged separately, ok)",
                    None => "NOT in input_offsets (name mismatch) -> graph reads ZEROS",
                    Some(_) => "in input_offsets but no arena slot -> graph reads ZEROS",
                };
                eprintln!("[cuda-input] {name:?} -> SKIPPED: {reason}");
            }
        }

        // Active-extent (PLAN L1): when set + every Step safe, bypass
        // captured CUDA Graph (recorded at full extent) and dispatch
        // per-step with scaled launch dims via the normal loop.
        let active = self.active_extent.filter(|_| self.all_safe_for_active());
        if self.active_extent.is_some() && active.is_none() {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let unsafe_steps: Vec<&str> = self
                    .schedule
                    .iter()
                    .filter(|s| !s.safe_for_active_extent())
                    .map(step_name)
                    .collect();
                let mut uniq = unsafe_steps;
                uniq.sort_unstable();
                uniq.dedup();
                eprintln!(
                    "[cuda] active_extent ignored — unsafe steps: {}",
                    uniq.join(", ")
                );
            }
        }
        // Scale a count by actual/upper with ceiling-division, clamped to [0, full].
        let scale = |full: u32| -> u32 {
            match active {
                Some((a, u)) if u > 0 => {
                    let f = full as usize;
                    (f * a).div_ceil(u).min(f) as u32
                }
                _ => full,
            }
        };

        // CUDA Graph fast path: replay a previously-captured schedule.
        // The first run with `ExecMode::Graph` falls through to the
        // normal dispatch loop with stream capture turned on; the
        // resulting graph is stashed in `self.captured_graph` and
        // replayed on every subsequent run.
        // RNG steps are capture-safe only under an on-device policy (Philox /
        // Zero); Ort / Bnns cold-fill on the host. Read the live policy here —
        // `set_rng` drops the captured graph when the policy changes, so a
        // captured Philox/Zero fill stays consistent with eager semantics.
        let rng_on_device = matches!(
            self.rng.read().expect("rng lock").backend,
            rlx_ir::RngBackend::Philox | rlx_ir::RngBackend::Zero
        );
        let graph_eligible = active.is_none()
            && self.exec_mode == ExecMode::Graph
            && !self.capture_disabled
            && schedule_graph_capture_safe(&self.schedule, rng_on_device);
        let do_replay = graph_eligible && self.captured_graph.is_some();
        // Whole-graph capture, like segmented, needs an eager WARM-UP run first
        // to load every kernel module + cuBLAS workspace (both illegal to
        // allocate mid-capture). So the first graph-eligible run dispatches
        // eagerly (on the capture stream) and only the SECOND run captures.
        let do_capture = graph_eligible && self.captured_graph.is_none() && self.capture_warmed;

        // Segmented capture (opt-in `RLX_CUDA_SEGMENTED_CAPTURE=1`): for the
        // schedules the WHOLE-graph gate rejects (a host step forces eager),
        // capture each maximal capture-safe run as its own graph and dispatch
        // host steps eagerly between segment replays. Orthogonal to the
        // whole-graph path above (that fires only when the WHOLE schedule is
        // safe; this only when it is NOT). Same stream throughout, so segment
        // graph launches and eager kernels serialize without extra syncs.
        // Actually engaging capture requires the extra flag
        // `RLX_CUDA_SEGMENTED_CAPTURE_ENGAGE` (kept as an opt-in while it's
        // validated on real models). The foundation is now WORKING end-to-end
        // (msi RTX 3080 Ti, cuBLAS + cuBLAS-free, capture+replay bit-exact):
        //   • dedicated non-null capture stream (the default stream is the
        //     uncapturable legacy null stream),
        //   • the run's I/O routed onto it,
        //   • and — the key — `disable_event_tracking()` for the dispatch:
        //     creating the stream swaps the context to multi-stream mode, which
        //     makes cudarc insert a per-`.arg` cross-stream WAIT (on the null
        //     stream) that breaks capture isolation for BOTH plain kernels and
        //     cuBLAS. Single-stream ordering makes that auto-sync unnecessary.
        // `RLX_CUDA_SEGMENTED_CAPTURE=1` alone still only surfaces the
        // opportunity diagnostic + runs eagerly. See `cuda_segmented_capture`.
        let seg_plan: Vec<CaptureSegment> = if rlx_ir::env::flag("RLX_CUDA_SEGMENTED_CAPTURE")
            && rlx_ir::env::flag("RLX_CUDA_SEGMENTED_CAPTURE_ENGAGE")
            && active.is_none()
            && self.exec_mode == ExecMode::Graph
            && !graph_eligible
        {
            segment_schedule_for_capture(&self.schedule, rng_on_device, SEGMENT_MIN_RUN)
        } else {
            Vec::new()
        };
        // Capture needs a NON-null stream (the default stream is the legacy
        // null stream, which CUDA refuses to capture). Create one lazily and
        // persist it so a replay run launches the captured graphs on the same
        // stream they were captured on. If creation fails, segmentation is
        // disabled this run (graceful — the loop dispatches eagerly).
        if segments_beat_all_or_nothing(&seg_plan) && self.segment_stream.is_none() {
            self.segment_stream = self.ctx.new_stream().ok();
        }
        let segmented_active =
            segments_beat_all_or_nothing(&seg_plan) && self.segment_stream.is_some();
        let seg_stream: Arc<cudarc::driver::CudaStream> = self
            .segment_stream
            .clone()
            .unwrap_or_else(|| default_stream.clone());
        let seg_num_gpu = seg_plan
            .iter()
            .filter(|s| matches!(s, CaptureSegment::Gpu { .. }))
            .count();
        let seg_replay =
            segmented_active && seg_num_gpu > 0 && self.segment_graphs.len() == seg_num_gpu;
        // Capture only AFTER a warm-up run has loaded every kernel module +
        // cuBLAS workspace (both illegal to allocate mid-capture). The first
        // segmented run is a plain eager pass on `seg_stream`.
        let seg_capture = segmented_active && !seg_replay && self.capture_warmed;
        // Route cuBLAS / cuDNN to the capture stream for the whole graph-mode
        // dispatch (segmented AND whole-graph) — else their kernels land on the
        // null stream, not captured, and a null-stream op during capture
        // invalidates it. Reset after the loop.
        if want_capture_stream {
            if let Some(blas) = self.blas.as_ref() {
                let blas = blas.lock().unwrap();
                unsafe {
                    let _ = cudarc::cublas::result::set_stream(
                        *blas.handle(),
                        seg_stream.cu_stream() as _,
                    );
                }
                // Pin a fixed cuBLAS workspace so cuBLAS does NO internal
                // `cudaMalloc` / helper-stream sync during capture — that
                // hidden op is what async-invalidates a multi-matmul whole-graph
                // capture (`CAPTURE_INVALIDATED` with per-step status ACTIVE).
                // Reuse the cuBLASLt scratch buffer (matmuls are sequential).
                if let Some(ws) = self.blas_lt_workspace.as_ref() {
                    let mut ws = ws.lock().unwrap();
                    let (ws_ptr, _r) = ws.device_ptr_mut(&seg_stream);
                    unsafe {
                        let _ = cudarc::cublas::sys::cublasSetWorkspace_v2(
                            *blas.handle(),
                            ws_ptr as *mut std::ffi::c_void,
                            CUBLASLT_WORKSPACE_BYTES,
                        );
                    }
                }
            }
            if let Some(handle) = self.dnn {
                unsafe {
                    let _ = cudarc::cudnn::result::set_stream(
                        handle,
                        seg_stream.cu_stream() as cudnn_sys::cudaStream_t,
                    );
                }
            }
        }

        if do_replay {
            self.prepare_readback_plan();
            let plan_ok = self
                .captured_readback_plan
                .as_ref()
                .is_some_and(|p| p.as_slice() == self.readback_plan_buf.as_slice());
            if plan_ok {
                self.captured_graph
                    .as_ref()
                    .unwrap()
                    .launch()
                    .expect("rlx-cuda: graph replay failed");
                if let Some(evt) = &self.replay_event {
                    evt.record(&stream)
                        .expect("rlx-cuda: replay event record failed");
                    evt.synchronize()
                        .expect("rlx-cuda: replay event sync failed");
                } else {
                    stream.synchronize().expect("rlx-cuda: stream sync failed");
                }
                run_tail_host_audio_ops(&self.schedule, &stream, self.arena.f32_buf_mut(), false);
                let plan = self.readback_plan_buf.clone();
                let read_all = plan.len() == self.graph.outputs.len();
                // DtoH must run after every replay — inputs change each run and
                // must not rely on dtoh baked into the captured graph.
                if read_all {
                    self.fill_output_staging(&stream)
                        .expect("rlx-cuda: output dtoh failed after replay");
                } else {
                    self.fill_output_staging_indices(&stream, &plan)
                        .expect("rlx-cuda: partial output dtoh failed after replay");
                }
                self.refresh_gpu_handles_from_staging(&plan);
                // Restore event tracking disabled for this graph-mode run before
                // the early return (the end-of-`run` restore is not reached).
                if want_capture_stream {
                    unsafe { self.ctx.enable_event_tracking() };
                }
                return self.outputs_from_staging_plan(&plan);
            }
            // Readback plan changed (e.g. partial grads); drop stale capture and re-dispatch.
            self.captured_graph = None;
            self.captured_readback_plan = None;
        }
        let _ = do_replay;

        let mut capturing = false;
        if do_capture {
            capturing = stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                )
                .is_ok();
        }

        // Multi-stream scheduler state. When `exec_mode ==
        // MultiStream(n)`, each Step gets assigned to one of `n` pool
        // streams based on producer-consumer dependencies on arena
        // offsets. Independent ops (e.g. unfused Q/K/V matmuls)
        // parallelise; producer-consumer chains stay on one stream.
        let multi_stream =
            matches!(self.exec_mode, ExecMode::MultiStream(_)) && !self.streams.is_empty();
        let mut producer_of: HashMap<u32, usize> = HashMap::new();
        let mut last_event: HashMap<usize, cudarc::driver::CudaEvent> = HashMap::new();
        let mut rr_cursor: usize = 0;

        // Per-step wall-time profiler (RLX_CUDA_STEP_PROFILE=1). Syncs the
        // default stream around every step and accumulates time by step name;
        // dumped after the loop. Disabled during graph capture (the syncs would
        // corrupt the capture) and skipped in multi-stream mode.
        let step_profile = rlx_ir::env::flag("RLX_CUDA_STEP_PROFILE")
            && !capturing
            && !multi_stream
            && !seg_capture;
        let mut step_prof: HashMap<&'static str, (f64, usize)> = HashMap::new();

        // Segmented-capture cursor state (all no-ops unless `segmented_active`).
        // `seg_cursor` walks `seg_plan`; `seg_gpu_k` counts Gpu segments for
        // replay indexing; `skip_until` skips a replayed Gpu segment's steps;
        // `cur_cap_end` marks the exclusive end index of an in-progress capture;
        // `captured_segs` accumulates the graphs built this run.
        let mut seg_cursor = 0usize;
        let mut seg_gpu_k = 0usize;
        let mut skip_until = 0usize;
        let mut cur_cap_end: Option<usize> = None;
        let mut captured_segs: Vec<Option<cudarc::driver::CudaGraph>> = Vec::new();
        // Whole-graph capture debug: pin the first step that invalidates the
        // capture (`RLX_CUDA_CAPTURE_DEBUG`). Checked once per step after dispatch.
        let cap_debug = capturing && rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG");
        let mut cap_invalidated = false;
        // Close a finished segment capture, instantiate + launch (capture only
        // RECORDS — the launch actually computes the segment this run). Params
        // (not captures) so it can be called from two sites without borrow
        // conflicts. Returns false if the capture broke (a mis-classified op) —
        // the caller then disables capture + re-runs eagerly for correctness
        // (the segment's data would otherwise be silently missing/stale).
        let close_seg = |ds: &std::sync::Arc<cudarc::driver::CudaStream>,
                         segs: &mut Vec<Option<cudarc::driver::CudaGraph>>|
         -> bool {
            match ds.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags
                    ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            ) {
                // The guard performs the upload+launch attempt; on success we keep
                // the instantiated graph, otherwise fall through to `false`.
                Ok(Some(g)) if g.upload().is_ok() && g.launch().is_ok() => {
                    segs.push(Some(g));
                    true
                }
                _ => false,
            }
        };
        let mut seg_capture_failed = false;

        // Dispatch each step. Each iteration is wrapped in an NVTX
        // range so nsight-systems traces show step boundaries cleanly.
        // Gated behind the `nvtx` feature because CUDA 13 removed
        // `nvToolsExt.dll`; cudarc panics on first call when the lib
        // isn't loadable. A range `for` (not `while`) so the many inner
        // `continue`s in the match auto-advance the index.
        for step_i in 0..self.schedule.len() {
            // Segmented capture: close a finished capture FIRST (robust to an
            // inner `continue` in the previous step leaving it open), then run
            // the segment START boundary for this index.
            if segmented_active {
                if let Some(end) = cur_cap_end
                    && step_i >= end
                {
                    seg_capture_failed |= !close_seg(&seg_stream, &mut captured_segs);
                    cur_cap_end = None;
                }
                while seg_cursor < seg_plan.len() && seg_plan[seg_cursor].range().end <= step_i {
                    seg_cursor += 1;
                }
                if let CaptureSegment::Gpu { start, end } = seg_plan[seg_cursor] {
                    if step_i == start {
                        if seg_replay {
                            if let Some(Some(g)) = self.segment_graphs.get(seg_gpu_k) {
                                g.launch().expect("rlx-cuda: segment graph replay failed");
                            }
                            seg_gpu_k += 1;
                            skip_until = end;
                        } else if seg_capture {
                            let cap = seg_stream.begin_capture(
                                cudarc::driver::sys::CUstreamCaptureMode
                                    ::CU_STREAM_CAPTURE_MODE_RELAXED,
                            );
                            if let Err(e) = &cap
                                && rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG")
                            {
                                eprintln!(
                                    "rlx-cuda: segment begin_capture failed at step {step_i}: {e:?}"
                                );
                            }
                            cur_cap_end = if cap.is_ok() { Some(end) } else { None };
                        }
                    }
                }
            }
            // Replay: steps inside an already-launched Gpu segment are skipped.
            if step_i < skip_until {
                continue;
            }
            let step = &self.schedule[step_i];
            // Capture-status probe at the TOP (before the match's many `continue`s
            // could skip it): if the capture is invalidated here, step_i-1 is the
            // culprit.
            if cap_debug && !cap_invalidated && step_i > 0 {
                if let Ok(s) = stream.capture_status() {
                    if s != cudarc::driver::sys::CUstreamCaptureStatus
                        ::CU_STREAM_CAPTURE_STATUS_ACTIVE
                    {
                        eprintln!(
                            "rlx-cuda: capture invalidated BY step {} = {} (status {s:?})",
                            step_i - 1,
                            step_name(&self.schedule[step_i - 1])
                        );
                        cap_invalidated = true;
                    }
                }
            }
            let _prof_t0 = if step_profile {
                let _ = default_stream.synchronize();
                Some(std::time::Instant::now())
            } else {
                None
            };
            #[cfg(feature = "nvtx")]
            let _nvtx = cudarc::nvtx::scoped_range(step_name(step));
            // PLAN L3: cross-backend Perfetto trace; no-op when env
            // var RLX_TRACE_PERFETTO unset.
            let _perf = rlx_ir::perfetto::TraceSpan::new(step_name(step), "cuda");

            // Per-step stream selection. In single-stream mode `stream`
            // shadows to the default stream; in multi-stream mode it
            // shadows to the assigned pool stream (and we cross-stream
            // event-wait on every producer not on the chosen stream).
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
                    // Multiple producers — keep the chosen one's queue
                    // intact and event-wait on the others.
                    let chosen = *producer_streams.iter().next().unwrap();
                    for s in &producer_streams {
                        if *s != chosen
                            && let Some(evt) = last_event.get(s)
                        {
                            let _ = self.streams[chosen].wait(evt);
                        }
                    }
                    chosen
                };
                Some(chosen)
            } else {
                None
            };
            // Single-stream (incl. segmented) dispatches on `seg_stream`, which
            // is the dedicated non-null capture stream when engaged and the null
            // default stream otherwise — so input copy, dispatch, and capture
            // all share one stream (dependency-isolated captures).
            let stream: Arc<cudarc::driver::CudaStream> = match assigned_idx {
                Some(i) => self.streams[i].clone(),
                None => seg_stream.clone(),
            };
            // Re-bind cuBLAS / cuDNN handles to the active stream so
            // their internal kernel launches go to the right queue.
            if multi_stream {
                if let Some(blas) = self.blas.as_ref() {
                    let blas = blas.lock().unwrap();
                    unsafe {
                        let _ = cudarc::cublas::result::set_stream(
                            *blas.handle(),
                            stream.cu_stream() as _,
                        );
                    }
                }
                if let Some(handle) = self.dnn {
                    unsafe {
                        let _ = cudarc::cudnn::result::set_stream(
                            handle,
                            stream.cu_stream() as cudnn_sys::cudaStream_t,
                        );
                    }
                }
            }
            match step {
                Step::ScaledQuantScale {
                    x_off_f32,
                    scale_off_f32,
                    n,
                    max_finite,
                } => {
                    let kernel = crate::kernels::scaled_quant_scale_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_off_f32)
                        .arg(n)
                        .arg(max_finite);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quant_scale launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_off_f32)
                        .arg(out_byte_off)
                        .arg(n)
                        .arg(e5m2);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quantize_fp8 launch failed");
                    }
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
                    let lt_handle = self
                        .blas_lt
                        .expect("rlx-cuda ScaledMatMul: cublasLt handle required for FP8 GEMM");
                    let mut workspace = self
                        .blas_lt_workspace
                        .as_ref()
                        .expect("rlx-cuda ScaledMatMul: cublasLt workspace required")
                        .lock()
                        .unwrap();
                    let (workspace_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                    let (arena_ptr, _record) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                    let cu_stream = stream.cu_stream();
                    let r = unsafe {
                        cublaslt_matmul_fp8(
                            lt_handle,
                            workspace_ptr,
                            CUBLASLT_WORKSPACE_BYTES,
                            arena_ptr,
                            *m,
                            *k,
                            *n,
                            *lhs_byte_off,
                            *rhs_byte_off,
                            *lhs_scale_byte_off,
                            *rhs_scale_byte_off,
                            *out_byte_off,
                            *has_bias != 0,
                            *bias_byte_off,
                            *lhs_e5m2 != 0,
                            *rhs_e5m2 != 0,
                            cu_stream,
                        )
                    };
                    r.expect(
                        "rlx-cuda: cublasLt FP8 GEMM failed (needs sm_89+ and 16B-aligned operands)",
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (blk, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_byte_off)
                        .arg(rows)
                        .arg(cols)
                        .arg(fmt)
                        .arg(scale_mode)
                        .arg(block);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quant_scale_general launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (blk, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off_f32)
                        .arg(scale_byte_off)
                        .arg(out_byte_off)
                        .arg(rows)
                        .arg(cols)
                        .arg(fmt)
                        .arg(scale_mode)
                        .arg(block);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_quantize_general launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (blk, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(codes_byte_off)
                        .arg(scale_byte_off)
                        .arg(out_off_f32)
                        .arg(rows)
                        .arg(cols)
                        .arg(fmt)
                        .arg(scale_mode)
                        .arg(block);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_dequantize_general launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(16), (*m).div_ceil(16), 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(lhs_byte_off)
                        .arg(rhs_byte_off)
                        .arg(lhs_scale_byte_off)
                        .arg(rhs_scale_byte_off)
                        .arg(out_off_f32)
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(lhs_fmt)
                        .arg(rhs_fmt)
                        .arg(scale_mode)
                        .arg(block)
                        .arg(has_bias)
                        .arg(bias_off_f32);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_matmul_decode launch failed");
                    }
                }
                Step::ScaledGroupedMatMulDecode {
                    m,
                    k,
                    n,
                    num_experts,
                    input_byte_off,
                    weight_byte_off,
                    input_scale_byte_off,
                    weight_scale_byte_off,
                    idx_off_f32,
                    out_off_f32,
                    bias_off_f32,
                    lhs_fmt,
                    rhs_fmt,
                    scale_mode,
                    block,
                    has_bias,
                } => {
                    let kernel = crate::kernels::scaled_grouped_matmul_decode_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(16), (*m).div_ceil(16), 1),
                        block_dim: (16, 16, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(input_byte_off)
                        .arg(weight_byte_off)
                        .arg(input_scale_byte_off)
                        .arg(weight_scale_byte_off)
                        .arg(idx_off_f32)
                        .arg(out_off_f32)
                        .arg(bias_off_f32)
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(num_experts)
                        .arg(lhs_fmt)
                        .arg(rhs_fmt)
                        .arg(scale_mode)
                        .arg(block)
                        .arg(has_bias);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scaled_grouped_matmul_decode launch failed");
                    }
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
                    // Opt-in Hopper TMA GEMM (`RLX_CUDA_TMA` on sm_90). Only
                    // eligible shapes route here; everything else falls through
                    // to the cuBLAS cascade below. Bias/activation go through the
                    // shared epilogue kernel afterward, exactly like WMMA.
                    if tma_arch(&self.ctx).is_some() {
                        if let Some((a_map, b_map)) = build_tma_gemm_maps(
                            &self.arena,
                            &stream,
                            *m,
                            *k,
                            *n,
                            *batch,
                            *a_off_f32,
                            *b_off_f32,
                        ) {
                            let kernel = matmul_tma_kernel(&self.ctx);
                            let cfg = LaunchConfig {
                                grid_dim: ((*n).div_ceil(64), (*m).div_ceil(64), 1),
                                block_dim: (16, 16, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(&a_map)
                                .arg(&b_map)
                                .arg(self.arena.f32_buf_mut())
                                .arg(m)
                                .arg(k)
                                .arg(n)
                                .arg(c_off_f32);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_tma launch failed");
                            }
                            if *has_bias != 0 || *act_id != 0xFFFFu32 {
                                let kernel = matmul_epilogue_kernel(&self.ctx);
                                let total = m * n * batch;
                                let (grid, block) = dispatch_grid_1d(total, 256);
                                let cfg = LaunchConfig {
                                    grid_dim: (grid, 1, 1),
                                    block_dim: (block, 1, 1),
                                    shared_mem_bytes: 0,
                                };
                                let mut launcher = stream.launch_builder(&kernel.function);
                                launcher
                                    .arg(self.arena.f32_buf_mut())
                                    .arg(&total)
                                    .arg(n)
                                    .arg(c_off_f32)
                                    .arg(has_bias)
                                    .arg(bias_off_f32)
                                    .arg(act_id);
                                unsafe {
                                    launcher
                                        .launch(cfg)
                                        .expect("rlx-cuda: matmul_epilogue (post-tma) failed");
                                }
                            }
                            if let Some(idx) = assigned_idx {
                                if let Ok(evt) = stream.record_event(None) {
                                    last_event.insert(idx, evt);
                                }
                                let (_, writes) = step_offsets(step);
                                for w in &writes {
                                    producer_of.insert(*w, idx);
                                }
                            }
                            continue;
                        }
                    }

                    if matmul_parity_mode() {
                        let kernel = matmul_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(64), *batch),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride)
                            .arg(has_bias)
                            .arg(bias_off_f32)
                            .arg(act_id);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul (parity) launch failed");
                        }
                        if let Some(idx) = assigned_idx {
                            if let Ok(evt) = stream.record_event(None) {
                                last_event.insert(idx, evt);
                            }
                            let (_, writes) = step_offsets(step);
                            for w in &writes {
                                producer_of.insert(*w, idx);
                            }
                        }
                        continue;
                    }

                    // Tier 0: mixed-precision GemmEx — when B (the weight)
                    // is stored in the half-arena, cast activations to
                    // f16/bf16 in a scratch buffer and call cublasGemmEx
                    // with both inputs half + f32 accumulator. Falls
                    // through to cublasLt on any setup or runtime error.
                    let used_mixed = try_mixed_precision_gemm(
                        &self.ctx,
                        &mut self.arena,
                        &mut self.half_act_scratch,
                        self.blas.as_ref(),
                        &stream,
                        *m,
                        *k,
                        *n,
                        *batch,
                        *a_off_f32,
                        *b_off_f32,
                        *c_off_f32,
                    );
                    if used_mixed {
                        // Optional bias / activation epilogue.
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue (mixed) failed");
                            }
                        }
                        // Multi-stream tail bookkeeping still runs at end of step.
                        if let Some(idx) = assigned_idx {
                            if let Ok(evt) = stream.record_event(None) {
                                last_event.insert(idx, evt);
                            }
                            let (_, writes) = step_offsets(step);
                            for w in &writes {
                                producer_of.insert(*w, idx);
                            }
                        }
                        continue;
                    }

                    // Tier 1: cublasLt fused (matmul + bias + relu/gelu in
                    // one launch). Only used when the activation is one of
                    // the two cublasLt natively fuses; other acts (silu,
                    // sigmoid, etc.) fall through to the sgemm + epilogue
                    // kernel path.
                    let try_cublaslt = self.blas_lt.is_some()
                        && self.blas_lt_workspace.is_some()
                        && cublaslt_act_supported(*act_id);
                    let used_cublaslt = if try_cublaslt {
                        let lt_handle = self.blas_lt.unwrap();
                        let mut workspace =
                            self.blas_lt_workspace.as_ref().unwrap().lock().unwrap();
                        let (workspace_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _record) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let cu_stream = stream.cu_stream();
                        let act = cublaslt_act_for(*act_id);
                        let r = unsafe {
                            cublaslt_matmul_fused(
                                lt_handle,
                                workspace_ptr,
                                CUBLASLT_WORKSPACE_BYTES,
                                arena_ptr,
                                *m,
                                *k,
                                *n,
                                *a_off_f32,
                                *b_off_f32,
                                *c_off_f32,
                                *has_bias != 0,
                                *bias_off_f32,
                                act,
                                *batch,
                                *a_batch_stride,
                                *b_batch_stride,
                                *c_batch_stride,
                                cu_stream,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("matmul.cublasLt", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cublaslt {
                        continue;
                    }

                    // Tier 2: cuBLAS sgemm via raw pointers (bypasses
                    // the borrow checker's same-buffer aliasing).
                    let used_cublas = if let Some(blas) = self.blas.as_ref() {
                        let blas = blas.lock().unwrap();
                        let (arena_ptr_u64, _record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let a_dev = arena_ptr_u64 + (*a_off_f32 as u64) * 4;
                        let b_dev = arena_ptr_u64 + (*b_off_f32 as u64) * 4;
                        let c_dev = arena_ptr_u64 + (*c_off_f32 as u64) * 4;
                        let alpha: f32 = 1.0;
                        let beta: f32 = 0.0;
                        // cuBLAS is column-major; we have row-major. Trick:
                        // computing C = A·B (row-major) is the same as
                        // computing C^T = B^T · A^T (column-major), and
                        // viewing our row-major arrays as column-major
                        // automatically yields the transpose.
                        let result = unsafe {
                            if *batch == 1 {
                                cudarc::cublas::result::sgemm(
                                    *blas.handle(),
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
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
                                cudarc::cublas::result::sgemm_strided_batched(
                                    *blas.handle(),
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
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
                        if let Err(ref e) = result {
                            log_fallback("matmul.cublasSgemm", e);
                        }
                        result.is_ok()
                    } else {
                        false
                    };

                    if used_cublas {
                        // Optional fused epilogue (bias + activation) as
                        // a separate element-wise kernel.
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue launch failed");
                            }
                        }
                    } else if use_wmma() {
                        // WMMA Tensor Core path: 32×64 block tile, 128 threads/block,
                        // SM 70+ only. Doesn't fuse bias/activation — those go to the
                        // shared epilogue kernel.
                        let kernel = matmul_wmma_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(32), *batch),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul_wmma launch failed");
                        }
                        if *has_bias != 0 || *act_id != 0xFFFFu32 {
                            let kernel = matmul_epilogue_kernel(&self.ctx);
                            let total = m * n * batch;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(&total)
                                .arg(n)
                                .arg(c_off_f32)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: matmul_epilogue (post-wmma) failed");
                            }
                        }
                    } else {
                        // Custom scalar kernel fallback: 64×64 block tile, 4×4 register tile.
                        let kernel = matmul_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: ((*n).div_ceil(64), (*m).div_ceil(64), *batch),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(m)
                            .arg(k)
                            .arg(n)
                            .arg(a_off_f32)
                            .arg(b_off_f32)
                            .arg(c_off_f32)
                            .arg(batch)
                            .arg(a_batch_stride)
                            .arg(b_batch_stride)
                            .arg(c_batch_stride)
                            .arg(has_bias)
                            .arg(bias_off_f32)
                            .arg(act_id);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: matmul launch failed");
                        }
                    }
                }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: binary launch failed");
                    }
                }
                Step::BinaryBroadcast {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    op,
                    rank,
                    meta_idx,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = binary_broadcast_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(op)
                        .arg(rank)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: binary_broadcast launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (gx, gy, gz),
                        block_dim: (bx, by, bz),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    // input_modulus is passed by-value as a 64-byte
                    // const param (16 u32s). Could move to meta_buffer
                    // but a constant param keeps the kernel signature
                    // self-describing.
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&len_s)
                        .arg(num_inputs)
                        .arg(num_steps)
                        .arg(dst_off)
                        .arg(&self.meta_buffers[*meta_idx])
                        .arg(scalar_input_mask)
                        .arg(input_modulus);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: elementwise_region launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid_x, 1, num_batch_s),
                        block_dim: (block_x, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&slice_len_s)
                        .arg(&num_batch_s)
                        .arg(num_steps)
                        .arg(base_dst_off)
                        .arg(slice_elems)
                        .arg(&self.meta_buffers[*batch_offs_idx])
                        .arg(&self.meta_buffers[*meta_idx])
                        .arg(scalar_input_mask)
                        .arg(input_modulus);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: batch_elementwise_region launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(out_off)
                        .arg(bin_op)
                        .arg(un_op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_binary_unary launch failed");
                    }
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
                    if rlx_ir::env::flag("RLX_CUDA_INPUT_DIAG")
                        && (*op == 13 || *op == 14 || *op == 12)
                    {
                        eprintln!(
                            "[cuda-unary] op={op} n={n_s} in_off={in_off} out_off={out_off} (12=Round,13=Sin,14=Cos)"
                        );
                    }
                    let kernel = unary_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(op);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: unary launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: compare launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(cond_off)
                        .arg(x_off)
                        .arg(y_off)
                        .arg(out_off);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: where launch failed");
                    }
                }
                Step::Fma {
                    n,
                    a_off,
                    b_off,
                    c_off,
                    out_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = fma_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(out_off);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: fma launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: reduce launch failed");
                    }
                }
                Step::Softmax {
                    num_rows,
                    axis_len,
                    stride,
                    in_off,
                    out_off,
                } => {
                    // batch scales the vector count (batch is a factor of num_rows)
                    let rows_s = scale(*num_rows);
                    if rows_s == 0 {
                        continue;
                    }
                    let kernel = softmax_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (rows_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&rows_s)
                        .arg(axis_len)
                        .arg(stride)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: softmax launch failed");
                    }
                }
                Step::ReluBackward {
                    n,
                    x_off,
                    dy_off,
                    dx_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = relu_backward_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(x_off)
                        .arg(dy_off)
                        .arg(dx_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: relu_backward launch failed");
                    }
                }
                Step::ActivationBackward {
                    n,
                    x_off,
                    dy_off,
                    dx_off,
                    op,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = activation_backward_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(x_off)
                        .arg(dy_off)
                        .arg(dx_off)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: activation_backward launch failed");
                    }
                }
                Step::SoftmaxCrossEntropy {
                    outer,
                    inner,
                    logits_off,
                    targets_off,
                    out_off,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = softmax_cross_entropy_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(logits_off)
                        .arg(targets_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: softmax_cross_entropy launch failed");
                    }
                }
                Step::SoftmaxCrossEntropyWithLogits {
                    outer,
                    inner,
                    logits_off,
                    labels_off,
                    out_off,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = softmax_cross_entropy_with_logits_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(logits_off)
                        .arg(labels_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: softmax_cross_entropy_with_logits launch failed");
                    }
                }
                Step::SoftmaxCrossEntropyBackward {
                    outer,
                    inner,
                    logits_off,
                    labels_off,
                    d_loss_off,
                    out_off,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = softmax_cross_entropy_backward_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(logits_off)
                        .arg(labels_off)
                        .arg(d_loss_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: softmax_cross_entropy_backward launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(eps_bits)
                        .arg(op);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layernorm launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(residual_off)
                        .arg(bias_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(out_off)
                        .arg(eps_bits)
                        .arg(has_bias);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_residual_ln launch failed");
                    }
                }
                Step::FusedResidualRmsNorm {
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
                    let kernel = fused_residual_rms_norm_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(residual_off)
                        .arg(bias_off)
                        .arg(gamma_off)
                        .arg(beta_off)
                        .arg(out_off)
                        .arg(eps_bits)
                        .arg(has_bias);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_residual_rms_norm launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (outer_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(scale_off)
                        .arg(shift_off)
                        .arg(out_off)
                        .arg(eps_bits)
                        .arg(layer_norm)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: ada_layer_norm launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&total_s)
                        .arg(inner)
                        .arg(x_off)
                        .arg(y_off)
                        .arg(gate_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gated_residual launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (mod_rows_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&mod_rows_s)
                        .arg(seq_per_mod)
                        .arg(inner)
                        .arg(x_off)
                        .arg(scale_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(eps_bits)
                        .arg(layer_norm);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: ada_layer_norm_backward launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (mod_rows_s, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&mod_rows_s)
                        .arg(seq_per_mod)
                        .arg(inner)
                        .arg(y_off)
                        .arg(gate_off)
                        .arg(dy_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gated_residual_backward launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n_out)
                        .arg(n_idx)
                        .arg(dim)
                        .arg(vocab)
                        .arg(in_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gather launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(total)
                        .arg(outer)
                        .arg(axis_dim)
                        .arg(num_idx)
                        .arg(trailing)
                        .arg(table_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: gather_axis launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(total)
                        .arg(outer)
                        .arg(inner)
                        .arg(axis_in_size)
                        .arg(axis_out_size)
                        .arg(start)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: narrow launch failed");
                    }
                }
                Step::Argmax {
                    outer,
                    inner,
                    in_off,
                    out_off,
                } => {
                    let kernel = argmax_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*outer, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(outer)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: argmax launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(rank)
                        .arg(out_total)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: transpose launch failed");
                    }
                }
                Step::Expand {
                    rank,
                    out_total,
                    in_off,
                    out_off,
                    meta_idx,
                } => {
                    if *out_total == 0 {
                        continue;
                    }
                    let kernel = expand_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(rank)
                        .arg(out_total)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: expand launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(total)
                        .arg(outer)
                        .arg(inner)
                        .arg(axis_in_size)
                        .arg(axis_out_size)
                        .arg(start)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: concat launch failed");
                    }
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
                    // Active-extent: scale seq bounds like rlx-metal encode.rs.
                    let seq_q_full = *seq_q;
                    let seq_k_full = *seq_k;
                    let seq_q_eff = scale(seq_q_full);
                    // Bucketed decode (mask_kind == 2): new K at row `upper` in a
                    // padded past buffer — do not scale kv_seq down or we skip it.
                    let seq_k_eff = if *mask_kind == 2 && seq_k_full != seq_q_full {
                        seq_k_full
                    } else {
                        scale(seq_k_full)
                    };
                    if seq_q_eff == 0 || seq_k_eff == 0 {
                        continue;
                    }
                    // Tiled flash supports arbitrary Q/K/V strides (BSHD and BHSD).
                    // Row kernel only when head_dim exceeds the flash tile cap or forced.
                    let use_row = rlx_ir::attention_dispatch_use_row(
                        *head_dim,
                        "RLX_CUDA_FORCE_ATTENTION_ROW",
                    );
                    // Attention variant policy (`RLX_CUDA_ATTENTION`). `Auto`
                    // picks the Tensor-Core (fp16 WMMA) kernel when it's eligible
                    // (non-row head_dim<=128, mask None/Causal, no softcap, sm70+)
                    // AND the workload is big enough to amortize its per-block
                    // overhead; else scalar flash / row. The WMMA kernels are
                    // drop-ins (same signature); everything falls back to scalar.
                    // See docs/attention-variants.md.
                    use crate::config::AttentionVariant;
                    let pol = crate::runtime_config().attention;
                    let wmma_eligible = !use_row
                        && *head_dim <= 128
                        && (*mask_kind == 0 || *mask_kind == 1)
                        && f32::from_bits(*softcap_bits) <= 0.0
                        && crate::backend::helpers::device_cc(&self.ctx).0 >= 7;
                    let want_wmma = match pol {
                        // Explicit `wmma` allows the d128 kernel too (correct,
                        // for experimentation), even though it doesn't beat the
                        // scalar kernel at head_dim=128 yet.
                        AttentionVariant::Wmma => wmma_eligible,
                        // Auto only turns on the *proven-faster* d64 path
                        // (head_dim<=64) when the workload amortizes it. The
                        // d128 WMMA kernel is slower than scalar today, so auto
                        // never picks it — auto is always >= scalar.
                        AttentionVariant::Auto => {
                            let work = *batch as u64 * *heads as u64 * seq_q_eff as u64;
                            wmma_eligible
                                && *head_dim <= 64
                                && work >= crate::runtime_config().attention_wmma_min_work
                        }
                        AttentionVariant::Scalar | AttentionVariant::Row => false,
                    };
                    // 0 = scalar/row · 1 = WMMA d64 (4 warps) · 2 = WMMA d128 (2 warps)
                    let wmma_kind: u8 = if want_wmma {
                        if *head_dim <= 64 { 1 } else { 2 }
                    } else {
                        0
                    };
                    let use_row_final = wmma_kind == 0 && (use_row || pol == AttentionVariant::Row);
                    let mut launcher = stream.launch_builder(match wmma_kind {
                        1 => &attention_wmma_kernel(&self.ctx).function,
                        2 => &attention_wmma_d128_kernel(&self.ctx).function,
                        _ if use_row_final => &attention_row_kernel(&self.ctx).function,
                        _ => &attention_kernel(&self.ctx).function,
                    });
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(batch)
                        .arg(heads)
                        .arg(&seq_q_eff)
                        .arg(&seq_k_eff)
                        .arg(head_dim)
                        .arg(q_off)
                        .arg(k_off)
                        .arg(v_off)
                        .arg(out_off)
                        .arg(mask_off)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(window)
                        .arg(seq_q_stride)
                        .arg(seq_k_stride)
                        .arg(mask_batch_stride)
                        .arg(mask_head_stride)
                        .arg(q_batch_stride)
                        .arg(q_head_stride)
                        .arg(q_seq_stride)
                        .arg(k_batch_stride)
                        .arg(k_head_stride)
                        .arg(k_seq_stride)
                        .arg(v_batch_stride)
                        .arg(v_head_stride)
                        .arg(v_seq_stride)
                        .arg(o_batch_stride)
                        .arg(o_head_stride)
                        .arg(o_seq_stride)
                        .arg(softcap_bits);
                    let cfg = if use_row_final {
                        let total = batch * heads * seq_q_eff;
                        let block = 256u32;
                        LaunchConfig {
                            grid_dim: (total.div_ceil(block), 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    } else {
                        // Query-tile height (BR) + block size per variant:
                        //   WMMA d64 = 4 warps / 64-query tile
                        //   WMMA d128 = 2 warps / 32-query tile
                        //   scalar flash = 128 threads / 16-query tile
                        let (br, block) = match wmma_kind {
                            1 => (64u32, 128u32),
                            2 => (32u32, 64u32),
                            _ => (16u32, 128u32),
                        };
                        let q_blocks = seq_q_eff.div_ceil(br);
                        LaunchConfig {
                            grid_dim: (q_blocks, batch * heads, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    };
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: attention launch failed");
                    }
                }
                Step::FusedAttn {
                    qkv_off,
                    mask_off,
                    cos_off,
                    sin_off,
                    out_off,
                    batch,
                    seq,
                    heads,
                    head_dim,
                    mask_kind,
                    scale_bits,
                    has_rope,
                } => {
                    let kernel = fused_attn_kernel(&self.ctx);
                    // One block per (batch·head); score matrix [seq·seq] in
                    // dynamic shared memory. The native gate (rlx-cuda unfuse)
                    // keeps `seq` small enough to fit the 48 KB default budget.
                    let cfg = LaunchConfig {
                        grid_dim: (batch * heads, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: seq * seq * 4,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(qkv_off)
                        .arg(mask_off)
                        .arg(cos_off)
                        .arg(sin_off)
                        .arg(out_off)
                        .arg(batch)
                        .arg(seq)
                        .arg(heads)
                        .arg(head_dim)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(has_rope);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_attn launch failed");
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
                    let cfg = LaunchConfig {
                        grid_dim: (batch * heads, y_blocks, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(batch)
                        .arg(heads)
                        .arg(seq_q)
                        .arg(seq_k)
                        .arg(head_dim)
                        .arg(q_off)
                        .arg(k_off)
                        .arg(v_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(mask_off)
                        .arg(mask_kind)
                        .arg(scale_bits)
                        .arg(window)
                        .arg(wrt);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: attention_bwd launch failed");
                    }
                }
                Step::Rope {
                    n_total,
                    seq,
                    head_dim,
                    half,
                    rot_half,
                    in_off,
                    cos_off,
                    sin_off,
                    out_off,
                    last_dim,
                    interleaved,
                } => {
                    let kernel = rope_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n_total)
                        .arg(seq)
                        .arg(head_dim)
                        .arg(half)
                        .arg(rot_half)
                        .arg(in_off)
                        .arg(cos_off)
                        .arg(sin_off)
                        .arg(out_off)
                        .arg(last_dim)
                        .arg(interleaved);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: rope launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(exclusive);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: cumsum launch failed");
                    }
                }
                Step::CumScan {
                    outer,
                    inner,
                    in_off,
                    out_off,
                    exclusive,
                    is_max,
                } => {
                    let outer_s = scale(*outer);
                    if outer_s == 0 {
                        continue;
                    }
                    let kernel = cum_scan_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(outer_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&outer_s)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(exclusive)
                        .arg(is_max);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: cum_scan launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(outer)
                        .arg(inner)
                        .arg(k)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: topk launch failed");
                    }
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
                    // Tier 1: sorted-batch dispatch via cuBLAS. Reads
                    // the idx buffer back to host, finds runs of
                    // identical consecutive expert ids, and issues one
                    // cublasSgemm per run. Wins big when tokens are
                    // pre-sorted by expert (the standard MoE upstream
                    // convention) — for random idx the run count is
                    // ~m and the launch overhead would negate the win,
                    // so we fall back to the kernel in that case.
                    let used_sorted = if let Some(blas) = self.blas.as_ref() {
                        // Sync first so prior writes to idx are visible.
                        stream
                            .synchronize()
                            .expect("rlx-cuda: stream sync before idx download");
                        let idx_host = {
                            let idx_slot = self
                                .arena
                                .f32_buf()
                                .slice(*idx_off as usize..(idx_off + m) as usize);
                            stream.clone_dtoh(&idx_slot).ok()
                        };
                        match idx_host {
                            Some(idx_vec) => {
                                let mut runs: Vec<(u32, u32, u32)> = Vec::new();
                                let mut i = 0usize;
                                let mn = *m as usize;
                                while i < mn {
                                    let e = idx_vec[i] as u32;
                                    let mut j = i + 1;
                                    while j < mn && (idx_vec[j] as u32) == e {
                                        j += 1;
                                    }
                                    if e < *num_experts {
                                        runs.push((i as u32, j as u32, e));
                                    }
                                    i = j;
                                }
                                // Heuristic: bail when the run count
                                // exceeds m/4 (idx isn't usefully sorted).
                                let threshold = (mn / 4).max(2);
                                if !runs.is_empty() && runs.len() <= threshold {
                                    let blas = blas.lock().unwrap();
                                    let (arena_ptr, _record) =
                                        self.arena.f32_buf_mut().device_ptr_mut(&stream);
                                    let alpha: f32 = 1.0;
                                    let beta: f32 = 0.0;
                                    let mut all_ok = true;
                                    for (lo, hi, e) in &runs {
                                        let rows = hi - lo;
                                        let a_dev = arena_ptr + ((*in_off + lo * k) as u64) * 4;
                                        let b_dev = arena_ptr + ((*w_off + e * k * n) as u64) * 4;
                                        let c_dev = arena_ptr + ((*out_off + lo * n) as u64) * 4;
                                        let r = unsafe {
                                            cudarc::cublas::result::sgemm(
                                                *blas.handle(),
                                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                                                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
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
                                        if r.is_err() {
                                            all_ok = false;
                                            break;
                                        }
                                    }
                                    all_ok
                                } else {
                                    false
                                }
                            }
                            None => false,
                        }
                    } else {
                        false
                    };
                    if used_sorted {
                        continue;
                    }

                    // Fallback: per-token expert lookup kernel.
                    let kernel = grouped_matmul_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        block_dim: (8, 8, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(num_experts)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(idx_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: grouped_matmul launch failed");
                    }
                }
                Step::ScatterAddZero { out_off, out_total } => {
                    let kernel = scatter_add_zero_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*out_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(out_off)
                        .arg(out_total);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scatter_add_zero launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(out_off)
                        .arg(upd_off)
                        .arg(idx_off)
                        .arg(num_updates)
                        .arg(trailing)
                        .arg(out_dim);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: scatter_add_acc launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: ((*n).div_ceil(8), (*m).div_ceil(8), 1),
                        block_dim: (8, 8, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(block_size)
                        .arg(scheme_id)
                        .arg(x_off)
                        .arg(w_off)
                        .arg(scale_off)
                        .arg(zp_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: dequant_matmul launch failed");
                    }
                }
                Step::DequantMatmulMlx {
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
                    let m_s = scale(*m);
                    if m_s == 0 {
                        continue;
                    }
                    if rlx_gpu_host::mlx_dequant_gpu_disabled() {
                        crate::gguf_host::run_dequant_matmul_mlx(
                            &stream,
                            self.arena.f32_buf_mut(),
                            m_s as usize,
                            *k as usize,
                            *n as usize,
                            *scheme,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *scale_byte_off as usize,
                            *zp_byte_off as usize,
                            *out_byte_off as usize,
                        );
                        continue;
                    }
                    let (kind, bits, group_size) = scheme.mlx_gpu_launch().unwrap_or_else(|| {
                        panic!("rlx-cuda DequantMatmulMlx: unexpected {scheme:?}")
                    });
                    if m_s == 1 {
                        let kernel = dequant_matmul_mlx_gemv_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: (*n, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(k)
                            .arg(n)
                            .arg(&kind)
                            .arg(&bits)
                            .arg(&group_size)
                            .arg(x_byte_off)
                            .arg(w_byte_off)
                            .arg(scale_byte_off)
                            .arg(zp_byte_off)
                            .arg(out_byte_off);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: dequant_matmul_mlx_gemv launch failed");
                        }
                    } else {
                        let n_row_tiles = m_s.div_ceil(8);
                        let kernel = dequant_matmul_mlx_gemm_kernel(&self.ctx);
                        let cfg = LaunchConfig {
                            grid_dim: (*n, n_row_tiles, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(&m_s)
                            .arg(k)
                            .arg(n)
                            .arg(&kind)
                            .arg(&bits)
                            .arg(&group_size)
                            .arg(x_byte_off)
                            .arg(w_byte_off)
                            .arg(scale_byte_off)
                            .arg(zp_byte_off)
                            .arg(out_byte_off);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: dequant_matmul_mlx_gemm launch failed");
                        }
                    }
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
                    // Prefill active-extent: scale row count (`m`) so padded
                    // max_seq buckets skip unused prompt rows (Bonsai --fast
                    // was paying full m=max_seq GEMMs for a short prompt).
                    let m_s = scale(*m);
                    if m_s == 0 {
                        continue;
                    }
                    // Decode GEMV (m=1, Q4_K/Q6_K/Q1_0): fused on-device kernel —
                    // parity with rlx-vulkan and rlx-cpu `gguf_matmul_bt`. Prefill
                    // (m>1) uses dequant_gguf + `matmul_bt`.
                    let fused_gemv = crate::gguf_gpu::gguf_fused_gemv_m1_supported(
                        *scheme_id,
                        m_s as usize,
                        *k as usize,
                    ) && !crate::gguf_gpu::gguf_fused_m1_env_disabled();
                    if rlx_ir::env::flag("RLX_CUDA_PATH_TRACE") {
                        static GEMV_LOGGED: std::sync::atomic::AtomicU32 =
                            std::sync::atomic::AtomicU32::new(0);
                        let nlog = GEMV_LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if nlog < 3 {
                            let path = if fused_gemv {
                                "fused_gemv"
                            } else if self.dequant_scratch_off > 0
                                && rlx_ir::env::var("RLX_CUDA_GGUF_HOST").as_deref() != Some("1")
                            {
                                "dequant+matmul"
                            } else {
                                "host"
                            };
                            eprintln!(
                                "[cuda-path] gguf={path} scheme={scheme_id} m={m_s} (full={m}) k={k} n={n}"
                            );
                        }
                    }
                    if fused_gemv {
                        crate::gguf_gpu::run_dequant_matmul_gguf_gemv_m1(
                            &self.ctx,
                            &stream,
                            self.arena.f32_buf_mut(),
                            *n as usize,
                            *k as usize,
                            *scheme_id,
                            *x_byte_off as usize,
                            *w_byte_off as usize,
                            *out_byte_off as usize,
                        );
                    } else {
                        // Keep the dequant+matmul on-device by DEFAULT, including
                        // decode (m=1) for schemes without a fused GEMV (e.g. Q5_0,
                        // Q8_0). Host dequant per decode step is ~6x slower end-to-end
                        // (gemma3-270m: 3.1 → 18.6 tok/s on NVIDIA GPU). Opt out with
                        // RLX_CUDA_GGUF_HOST=1.
                        let use_gpu = self.dequant_scratch_off > 0
                            && rlx_ir::env::var("RLX_CUDA_GGUF_HOST").as_deref() != Some("1");
                        if use_gpu {
                            crate::gguf_gpu::run_dequant_matmul_gguf_gpu(
                                &self.ctx,
                                &stream,
                                self.arena.f32_buf_mut(),
                                m_s as usize,
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
                                &stream,
                                self.arena.f32_buf_mut(),
                                m_s as usize,
                                *k as usize,
                                *n as usize,
                                *scheme_id,
                                *x_byte_off as usize,
                                *w_byte_off as usize,
                                *out_byte_off as usize,
                            );
                        }
                    }
                }
                Step::DequantMatmulMxFp4x2 {
                    m,
                    k,
                    n,
                    group,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    out_byte_off,
                } => {
                    let m_s = scale(*m);
                    if m_s == 0 {
                        continue;
                    }
                    assert!(
                        self.dequant_scratch_off > 0,
                        "rlx-cuda DequantMatMul(MxFp4x2): needs dequant scratch"
                    );
                    crate::gguf_gpu::run_dequant_matmul_mxfp4x2_gpu(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        m_s as usize,
                        *k as usize,
                        *n as usize,
                        *group as usize,
                        *x_byte_off as usize,
                        *w_byte_off as usize,
                        *scale_byte_off as usize,
                        self.dequant_scratch_off,
                        *out_byte_off as usize,
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
                    let use_gpu = self.dequant_scratch_off > 0;
                    if use_gpu {
                        crate::gguf_gpu::run_dequant_grouped_matmul_gguf_gpu(
                            &self.ctx,
                            &stream,
                            self.arena.f32_buf_mut(),
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
                            &stream,
                            self.arena.f32_buf_mut(),
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
                        &stream,
                        self.arena.f32_buf_mut(),
                        *m as usize,
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
                Step::DequantGroupedMatmulMlxNative {
                    m,
                    k,
                    n,
                    num_experts,
                    group_size,
                    x_byte_off,
                    w_byte_off,
                    scale_byte_off,
                    idx_byte_off,
                    out_byte_off,
                } => {
                    // m>1 (prefill) with a register-safe row count → the amortized kernel
                    // (decode each fired expert's weight once, reuse across its rows);
                    // otherwise (decode, m==1) the per-element kernel is optimal.
                    let amort = *m > 1 && *m <= 16;
                    let kernel = if amort {
                        crate::kernels::dequant_grouped_matmul_mlx_mxfp4_amort_kernel(&self.ctx)
                    } else {
                        crate::kernels::dequant_grouped_matmul_mlx_mxfp4_kernel(&self.ctx)
                    };
                    let cfg = if amort {
                        LaunchConfig {
                            grid_dim: ((*n).div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        }
                    } else {
                        LaunchConfig {
                            grid_dim: ((*n).div_ceil(16), (*m).div_ceil(16), 1),
                            block_dim: (16, 16, 1),
                            shared_mem_bytes: 0,
                        }
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(num_experts)
                        .arg(group_size)
                        .arg(x_byte_off)
                        .arg(w_byte_off)
                        .arg(scale_byte_off)
                        .arg(idx_byte_off)
                        .arg(out_byte_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: dequant_grouped_matmul_mlx_mxfp4 launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(outer)
                        .arg(inner)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(top_k)
                        .arg(top_p_bits)
                        .arg(temp_bits)
                        .arg(seed_lo)
                        .arg(seed_hi);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: sample launch failed");
                    }
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
                    if !crate::rng_gpu::try_rng_normal(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *dst_byte_off as usize,
                        *len as usize,
                        *mean,
                        *scale,
                        *key,
                        *op_seed,
                        opts,
                    ) {
                        crate::rng_host::run_rng_normal(
                            &stream,
                            self.arena.f32_buf_mut(),
                            *dst_byte_off as usize,
                            *len as usize,
                            *mean,
                            *scale,
                            *key,
                            *op_seed,
                            opts,
                        );
                    }
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
                    if !crate::rng_gpu::try_rng_uniform(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *dst_byte_off as usize,
                        *len as usize,
                        *low,
                        *high,
                        *key,
                        *op_seed,
                        opts,
                    ) {
                        crate::rng_host::run_rng_uniform(
                            &stream,
                            self.arena.f32_buf_mut(),
                            *dst_byte_off as usize,
                            *len as usize,
                            *low,
                            *high,
                            *key,
                            *op_seed,
                            opts,
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(batch)
                        .arg(seq)
                        .arg(hidden)
                        .arg(state_size)
                        .arg(x_off)
                        .arg(delta_off)
                        .arg(a_off)
                        .arg(b_off)
                        .arg(c_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: selective_scan launch failed");
                    }
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
                    real_input,
                } => {
                    if *use_gpu {
                        let norm = rlx_ir::fft::FftNorm::from_tag(*norm_tag);
                        let scale = norm.output_scale(*n_complex as usize, *inverse) as f32;
                        // Backend precedence for the GPU FFT op: native Stockham
                        // (n≤4096) → cuFFT → native multi/single-kernel. Each is
                        // behind its own feature; with none on, only the last arm
                        // compiles. `real_input` (the fused real→complex path) is
                        // only the native kernel can read, so it forces this arm.
                        #[allow(unused_mut)]
                        let mut handled = false;
                        let _ = real_input;

                        #[cfg(feature = "native-cuda-fft")]
                        if !handled
                            && (*real_input
                                || (crate::native_fft_dispatch::stockham_enabled()
                                    && crate::native_fft_dispatch::stockham_eligible(*n_complex)))
                        {
                            crate::native_fft_dispatch::run_fft_native_stockham(
                                &self.ctx,
                                &stream,
                                self.arena.f32_buf_mut(),
                                (*src_byte_off / 4) as u32,
                                (*dst_byte_off / 4) as u32,
                                *outer,
                                *n_complex,
                                *inverse,
                                scale,
                                *real_input,
                            );
                            handled = true;
                        }

                        #[cfg(feature = "cufft")]
                        if !handled && crate::cufft_dispatch::cufft_should_use(*n_complex) {
                            crate::cufft_dispatch::run_fft_cufft(
                                &self.ctx,
                                &stream,
                                &mut self.cufft_state,
                                self.arena.f32_buf_mut(),
                                (*src_byte_off / 4) as u32,
                                (*dst_byte_off / 4) as u32,
                                *outer,
                                *n_complex,
                                *inverse,
                                scale,
                            );
                            handled = true;
                        }

                        if !handled {
                            crate::fft_dispatch::run_fft_gpu(
                                &self.ctx,
                                &stream,
                                self.arena.f32_buf_mut(),
                                (*src_byte_off / 4) as u32,
                                (*dst_byte_off / 4) as u32,
                                *outer,
                                *n_complex,
                                *inverse,
                                scale,
                            );
                        }
                    } else {
                        let (buf, arena_size) = self.arena.f32_buf_and_size();
                        crate::fft_host::run_fft1d(
                            &stream,
                            buf,
                            arena_size,
                            *src_byte_off as usize,
                            *dst_byte_off as usize,
                            *outer as usize,
                            *n_complex as usize,
                            *inverse,
                            *norm_tag,
                            fft_dtype_from_tag(*dtype_tag),
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
                        &stream,
                        self.arena.f32_buf_mut(),
                        *spec_off,
                        *dst_off,
                        *welch_batch,
                        *n_fft,
                        *n_segments,
                        *k,
                        *n_bins,
                    );
                }
                Step::FftButterflyStage {
                    state_off,
                    out_off,
                    gate_off,
                    rev_off,
                    tw_re_off,
                    tw_im_off,
                    batch,
                    n_fft,
                    stage,
                } => {
                    let kernel = fft_butterfly_stage_kernel(&self.ctx);
                    let block = 256u32;
                    let grid = (*batch).max(1);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(state_off)
                        .arg(out_off)
                        .arg(gate_off)
                        .arg(rev_off)
                        .arg(tw_re_off)
                        .arg(tw_im_off)
                        .arg(batch)
                        .arg(n_fft)
                        .arg(stage);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fft_butterfly_stage launch failed");
                    }
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
                        let m = *n * *h_out * *w_out;
                        let k = *c_in * *kh * *kw;
                        let total = m * k;
                        let (grid, block) = dispatch_grid_1d(total, 256);
                        let cfg = LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let x_off = (*x_byte_off / 4) as u32;
                        let col_off = (*col_byte_off / 4) as u32;
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(n)
                            .arg(c_in)
                            .arg(h)
                            .arg(w)
                            .arg(h_out)
                            .arg(w_out)
                            .arg(kh)
                            .arg(kw)
                            .arg(sh)
                            .arg(sw)
                            .arg(ph)
                            .arg(pw)
                            .arg(dh)
                            .arg(dw_dil)
                            .arg(&x_off)
                            .arg(&col_off);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: im2col launch failed");
                        }
                    } else {
                        crate::im2col_host::run_im2col(
                            &stream,
                            self.arena.f32_buf_mut(),
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
                        &stream,
                        self.arena.f32_buf_mut(),
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        dims,
                        rev_mask,
                        *elem_bytes as usize,
                    );
                }
                Step::PadHost {
                    src_byte_off,
                    dst_byte_off,
                    in_dims,
                    before,
                    after,
                    mode,
                    fill,
                    elem_bytes,
                } => {
                    crate::host_misc::run_pad(
                        &stream,
                        self.arena.f32_buf_mut(),
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        in_dims,
                        before,
                        after,
                        *mode,
                        fill,
                        *elem_bytes as usize,
                    );
                }
                Step::Pad {
                    n,
                    src_off,
                    dst_off,
                    mode,
                    fill,
                    rank,
                    meta_idx,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = pad_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(src_off)
                        .arg(dst_off)
                        .arg(mode)
                        .arg(fill)
                        .arg(rank)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: pad launch failed");
                    }
                }
                Step::SliceHost {
                    src_byte_off,
                    dst_byte_off,
                    in_dims,
                    axis,
                    start,
                    len,
                    step,
                    elem_bytes,
                } => {
                    crate::host_misc::run_slice(
                        &stream,
                        self.arena.f32_buf_mut(),
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        in_dims,
                        *axis as usize,
                        *start as usize,
                        *len as usize,
                        *step,
                        *elem_bytes as usize,
                    );
                }
                Step::Slice {
                    n,
                    src_off,
                    dst_off,
                    axis,
                    start,
                    step,
                    rank,
                    meta_idx,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = slice_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(src_off)
                        .arg(dst_off)
                        .arg(axis)
                        .arg(start)
                        .arg(step)
                        .arg(rank)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher.launch(cfg).expect("rlx-cuda: slice launch failed");
                    }
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
                        &stream,
                        self.arena.f32_buf_mut(),
                        *src_byte_off as usize,
                        *dst_byte_off as usize,
                        *outer as usize,
                        *reduced as usize,
                        *inner as usize,
                        *is_max,
                    );
                }
                Step::AxialRope2d {
                    in_off,
                    out_off,
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
                    let n_total = batch * seq * hidden;
                    let kernel = axial_rope2d_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(batch)
                        .arg(seq)
                        .arg(hidden)
                        .arg(end_x)
                        .arg(end_y)
                        .arg(head_dim)
                        .arg(num_heads)
                        .arg(repeat_factor)
                        .arg(theta)
                        .arg(in_off)
                        .arg(out_off)
                        .arg(&n_total);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: axial_rope2d launch failed");
                    }
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
                    use_gpu,
                    gate_per_channel,
                } => {
                    let state_bytes = if *use_carry {
                        *state_byte_off as usize
                    } else {
                        self.gdn_scratch_off
                    };
                    if *use_gpu {
                        // Opt-in FlashKDA-style chunked-parallel kernel (per-channel
                        // gate, n=128). Same arena+offset ABI; 128-thread block.
                        let use_chunk = *gate_per_channel
                            && *state_size == 128
                            && rlx_ir::env::flag("RLX_CUDA_KDA_CHUNK");
                        let kernel = if use_chunk {
                            kimi_delta_chunk_kernel(&self.ctx)
                        } else {
                            gated_delta_net_kernel(&self.ctx)
                        };
                        let cfg = LaunchConfig {
                            grid_dim: (*batch * *heads, 1, 1),
                            block_dim: (*state_size, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        // Kernel params are all `unsigned long long` (see
                        // gated_delta_net.cu). These MUST be u64 to match: cuLaunchKernel
                        // reads each arg at the kernel-declared 8-byte width, so passing a
                        // 4-byte u32 makes it read 4 bytes of adjacent stack as the high
                        // word → a multi-GB garbage offset → CUDA_ERROR_ILLEGAL_ADDRESS.
                        let q_off = *q_byte_off / 4;
                        let k_off = *k_byte_off / 4;
                        let v_off = *v_byte_off / 4;
                        let g_off = *g_byte_off / 4;
                        let beta_off = *beta_byte_off / 4;
                        let state_off = (state_bytes / 4) as u64;
                        let dst_off = *dst_byte_off / 4;
                        let use_carry_u: u32 = if *use_carry { 1 } else { 0 };
                        let gate_pc_u: u32 = if *gate_per_channel { 1 } else { 0 };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(&q_off)
                            .arg(&k_off)
                            .arg(&v_off)
                            .arg(&g_off)
                            .arg(&beta_off)
                            .arg(&state_off)
                            .arg(&dst_off)
                            .arg(batch)
                            .arg(seq)
                            .arg(heads)
                            .arg(state_size)
                            .arg(&use_carry_u)
                            .arg(&gate_pc_u);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: gated_delta_net launch failed");
                        }
                    } else {
                        let (buf, arena_size) = self.arena.f32_buf_and_size();
                        // Host CPU path allocates ephemeral state when !use_carry.
                        crate::gdn_host::run_gated_delta_net(
                            &stream,
                            buf,
                            arena_size,
                            *q_byte_off as usize,
                            *k_byte_off as usize,
                            *v_byte_off as usize,
                            *g_byte_off as usize,
                            *beta_byte_off as usize,
                            if *use_carry { state_bytes } else { 0 },
                            *dst_byte_off as usize,
                            *batch as usize,
                            *seq as usize,
                            *heads as usize,
                            *state_size as usize,
                            *use_carry,
                            *gate_per_channel,
                        );
                    }
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
                    // Native GPU LSTM; falls back to the host path only when the
                    // layer geometry overflows the shared-memory budget.
                    let handled = crate::lstm_gpu::run_lstm(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
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
                    if !handled {
                        let (buf, arena_size) = self.arena.f32_buf_and_size();
                        crate::lstm_host::run_lstm(
                            &stream,
                            buf,
                            arena_size,
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
                }
                Step::Gru {
                    x_byte_off,
                    w_ih_byte_off,
                    w_hh_byte_off,
                    b_ih_byte_off,
                    b_hh_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    h0_byte_off,
                } => {
                    crate::gru_gpu::run_gru(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *x_byte_off as usize,
                        *w_ih_byte_off as usize,
                        *w_hh_byte_off as usize,
                        *b_ih_byte_off as usize,
                        *b_hh_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *h0_byte_off as usize,
                    );
                }
                Step::GruHost {
                    x_byte_off,
                    w_ih_byte_off,
                    w_hh_byte_off,
                    b_ih_byte_off,
                    b_hh_byte_off,
                    h0_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::gru_host::run_gru(
                        &stream,
                        buf,
                        arena_size,
                        *x_byte_off as usize,
                        *w_ih_byte_off as usize,
                        *w_hh_byte_off as usize,
                        *b_ih_byte_off as usize,
                        *b_hh_byte_off as usize,
                        *h0_byte_off as usize,
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
                Step::Rnn {
                    x_byte_off,
                    w_ih_byte_off,
                    w_hh_byte_off,
                    bias_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    h0_byte_off,
                    relu,
                } => {
                    crate::rnn_gpu::run_rnn(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *x_byte_off as usize,
                        *w_ih_byte_off as usize,
                        *w_hh_byte_off as usize,
                        *bias_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *input_size as usize,
                        *hidden as usize,
                        *num_layers as usize,
                        *bidirectional,
                        *h0_byte_off as usize,
                        *relu,
                    );
                }
                Step::RnnHost {
                    x_byte_off,
                    w_ih_byte_off,
                    w_hh_byte_off,
                    bias_byte_off,
                    h0_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                    relu,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::rnn_host::run_rnn(
                        &stream,
                        buf,
                        arena_size,
                        *x_byte_off as usize,
                        *w_ih_byte_off as usize,
                        *w_hh_byte_off as usize,
                        *bias_byte_off as usize,
                        *h0_byte_off as usize,
                        *dst_byte_off as usize,
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
                Step::Mamba2 {
                    x_byte_off,
                    dt_byte_off,
                    a_byte_off,
                    b_byte_off,
                    c_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    heads,
                    head_dim,
                    state_size,
                } => {
                    crate::mamba2_gpu::run_mamba2(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *x_byte_off as usize,
                        *dt_byte_off as usize,
                        *a_byte_off as usize,
                        *b_byte_off as usize,
                        *c_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *heads as usize,
                        *head_dim as usize,
                        *state_size as usize,
                    );
                }
                Step::Mamba2Host {
                    x_byte_off,
                    dt_byte_off,
                    a_byte_off,
                    b_byte_off,
                    c_byte_off,
                    dst_byte_off,
                    batch,
                    seq,
                    heads,
                    head_dim,
                    state_size,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::mamba2_host::run_mamba2(
                        &stream,
                        buf,
                        arena_size,
                        *x_byte_off as usize,
                        *dt_byte_off as usize,
                        *a_byte_off as usize,
                        *b_byte_off as usize,
                        *c_byte_off as usize,
                        *dst_byte_off as usize,
                        *batch as usize,
                        *seq as usize,
                        *heads as usize,
                        *head_dim as usize,
                        *state_size as usize,
                    );
                }
                Step::ScanHost { desc } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::scan_host::run_scan(&stream, buf, arena_size, desc);
                }
                Step::HostOp { desc } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::scan_host::run_host_op(&stream, buf, arena_size, desc);
                }
                Step::CpuIndexing { thunk } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    if !crate::scatter_nd_gpu::try_run(&self.ctx, &stream, buf, thunk) {
                        crate::scan_host::run_indexing(&stream, buf, arena_size, thunk);
                    }
                }
                Step::SpdHost {
                    op,
                    out_off,
                    out_shape,
                    inputs,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::spd_host::run_spd(
                        &stream, buf, arena_size, op, *out_off, out_shape, inputs,
                    );
                }
                Step::EighNative {
                    in_off,
                    out_off,
                    n,
                    batch,
                } => {
                    let buf = self.arena.f32_buf_mut();
                    crate::eigh_native::run(&self.ctx, &stream, buf, *in_off, *out_off, *n, *batch);
                }
                Step::DenseSolveNative {
                    a_off,
                    b_off,
                    x_off,
                    n,
                    nrhs,
                } => {
                    let buf = self.arena.f32_buf_mut();
                    crate::dense_solve_native::run_dense(
                        &self.ctx, &stream, buf, *a_off, *b_off, *x_off, *n, *nrhs,
                    );
                }
                Step::BatchedDenseSolveNative {
                    a_off,
                    b_off,
                    x_off,
                    batch,
                    n,
                    nrhs,
                } => {
                    let blas = self
                        .blas
                        .as_ref()
                        .expect("rlx-cuda: BatchedDenseSolveNative requires cuBLAS");
                    let blas = blas.lock().unwrap();
                    let buf = self.arena.f32_buf_mut();
                    crate::dense_solve_native::run_batched(
                        &self.ctx,
                        &stream,
                        *blas.handle(),
                        buf,
                        *a_off,
                        *b_off,
                        *x_off,
                        *batch,
                        *n,
                        *nrhs,
                    );
                }
                Step::Llada2GroupLimitedGate {
                    sig_off,
                    route_off,
                    out_off,
                    n_elems,
                    attrs,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::llada2_gate_host::run_llada2_group_limited_gate(
                        &stream,
                        buf,
                        arena_size,
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
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::ms_deform_attn_host::run_ms_deform_attn(
                        &stream,
                        buf,
                        arena_size,
                        in_offs,
                        *out_off as usize,
                        *out_len as usize,
                        attrs,
                    );
                }
                Step::CustomHost {
                    name,
                    in_specs,
                    out_off,
                    out_shape,
                    attrs,
                } => {
                    let (buf, _arena_size) = self.arena.f32_buf_and_size();
                    if crate::dyn_quant_lstm_gpu::try_run(
                        &self.ctx, &stream, buf, name, in_specs, *out_off, out_shape, attrs,
                    ) {
                        // Handled on-device (Kitten DynamicQuantizeLSTM).
                    } else {
                        crate::onnx_custom_host::run_custom_host(
                            &stream, buf, name, in_specs, *out_off, out_shape, attrs,
                        );
                    }
                }
                Step::CollectiveHost {
                    name,
                    in_off,
                    in_len,
                    out_off,
                    out_len,
                    attrs,
                } => {
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::collective_host::run_collective(
                        &stream,
                        buf,
                        arena_size,
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
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::umap_knn_host::run_umap_knn(
                        &stream,
                        buf,
                        arena_size,
                        *pairwise_off as usize,
                        *out_off as usize,
                        *n as usize,
                        *k as usize,
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(g_off)
                        .arg(b_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layer_norm2d launch failed");
                    }
                }
                Step::CudaGpuKernel {
                    name,
                    out_off,
                    out_len,
                    in_offs,
                    attrs,
                    out_shape,
                } => {
                    // Raw-GPU custom op: fetch (NVRTC-compiling on first use) the
                    // kernel and launch it against the whole arena. Offsets are
                    // scalar args (copied into the launch at enqueue, so no async
                    // lifetime hazard), one 1-D grid over the output.
                    let gk = crate::cuda_gpu_kernels::lookup(name)
                        .expect("CudaGpuKernel vanished from the registry between compile and run");
                    let kernel = crate::cuda_gpu_kernels::get_or_build(&self.ctx, &*gk);
                    // Pad (off,len) to MAX_INPUTS with (0,0); `n_inputs` says how
                    // many are real. Trailing e0..e3 are runtime extras.
                    let n_inputs = in_offs.len() as u32;
                    let mut io = [0u32; crate::cuda_gpu_kernels::MAX_INPUTS * 2];
                    for (i, (o, l)) in in_offs.iter().enumerate() {
                        io[i * 2] = *o;
                        io[i * 2 + 1] = *l;
                    }
                    let extras = gk.extras(attrs, out_shape);
                    let bs = gk.block_size().max(1);
                    let launch_n = gk.launch_elems(*out_len, extras).max(1);
                    let grid = gk.grid_blocks(launch_n, bs).max(1);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (bs, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(out_off)
                        .arg(out_len)
                        .arg(&n_inputs);
                    for v in &io {
                        launcher.arg(v);
                    }
                    for e in &extras {
                        launcher.arg(e);
                    }
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: CudaGpuKernel launch failed");
                    }
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
                    // ConvTranspose2d ≡ cuDNN convolutionBackwardData with the
                    // PyTorch weight layout [C_in, C_out/g, kH, kW]. The naive
                    // gather kernel is ~10–50× slower on vocoder upsamplers
                    // (Kitten: 7× ≈ 1.5 s of a 3.4 s wave pass).
                    // Kitten / HiFi-GAN use 1×k kernels and depthwise groups —
                    // do NOT reuse the training fwd gate (`kh>1 && groups==1`).
                    // Opt out with `RLX_CUDA_CONV_T_KERNEL=1` or `RLX_CUDA_NO_CUDNN=1`.
                    let try_cudnn = self.dnn.is_some()
                        && self.dnn_workspace.is_some()
                        && !rlx_ir::env::flag("RLX_CUDA_NO_CUDNN")
                        && !rlx_ir::env::flag("RLX_CUDA_CONV_T_KERNEL");
                    let used_cudnn = if try_cudnn {
                        let handle = self.dnn.expect("dnn handle");
                        let workspace = self.dnn_workspace.as_ref().expect("dnn workspace");
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        // Map transpose → forward-bwd_data:
                        //   dy = transpose input  [N, C_in, H, W]
                        //   dx = transpose output [N, C_out, H_out, W_out]
                        //   W  = [C_in, C_out/g, kH, kW] (= forward filter [K, C/g, R, S]
                        //         with K=C_in, C=C_out).
                        let r = unsafe {
                            cudnn_conv2d_backward_data(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_out,
                                *c_in,
                                *h_out,
                                *w_out,
                                *h,
                                *w_in,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *ph,
                                *pw,
                                *dh,
                                *dw,
                                *groups,
                                *src_off,
                                *w_off,
                                *dst_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv_transpose2d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if rlx_ir::env::flag("RLX_CUDA_CONV_TRACE") {
                        eprintln!(
                            "[CONV-TRACE] ConvTranspose2d n={} c_in={} c_out={} {}x{}→{}x{} k={}x{} g={} -> {}",
                            *n,
                            *c_in,
                            *c_out,
                            *h,
                            *w_in,
                            *h_out,
                            *w_out,
                            *kh,
                            *kw,
                            *groups,
                            if used_cudnn { "cuDNN" } else { "kernel" }
                        );
                    }
                    if !used_cudnn {
                        let kernel = conv_transpose2d_kernel(&self.ctx);
                        let total = n * c_out * h_out * w_out;
                        let (grid, block) = dispatch_grid_1d(total, 256);
                        let cfg = LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(src_off)
                            .arg(w_off)
                            .arg(dst_off)
                            .arg(n)
                            .arg(c_in)
                            .arg(h)
                            .arg(w_in)
                            .arg(c_out)
                            .arg(h_out)
                            .arg(w_out)
                            .arg(kh)
                            .arg(kw)
                            .arg(sh)
                            .arg(sw)
                            .arg(ph)
                            .arg(pw)
                            .arg(dh)
                            .arg(dw)
                            .arg(groups);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: conv_transpose2d launch failed");
                        }
                    }
                }
                Step::ConvTranspose3d {
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
                    // ConvTranspose3d ≡ cuDNN convolution BackwardData (nd) with
                    // PyTorch weight [C_in, C_out/g, kD, kH, kW]. Same remap as CT2d.
                    // Opt out with `RLX_CUDA_CONV_T_KERNEL=1` or `RLX_CUDA_NO_CUDNN=1`.
                    let try_cudnn = self.dnn.is_some()
                        && self.dnn_workspace.is_some()
                        && !rlx_ir::env::flag("RLX_CUDA_NO_CUDNN")
                        && !rlx_ir::env::flag("RLX_CUDA_CONV_T_KERNEL");
                    let used_cudnn = if try_cudnn {
                        let handle = self.dnn.expect("dnn handle");
                        let workspace = self.dnn_workspace.as_ref().expect("dnn workspace");
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        // dy = CT input [N,C_in,D,H,W]; dx = CT output [N,C_out,Do,Ho,Wo]
                        let r = unsafe {
                            cudnn_conv3d_backward_data(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_out,
                                *c_in,
                                *d_out,
                                *h_out,
                                *w_out,
                                *d,
                                *h,
                                *w,
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
                            log_fallback("conv_transpose3d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    record_conv_transpose3d_path(used_cudnn);
                    if rlx_ir::env::flag("RLX_CUDA_LOG_CONV_PATH") {
                        eprintln!(
                            "rlx-cuda-convpath: {} n={} c_in={} c_out={} {}x{}x{}→{}x{}x{} k={}x{}x{} g={}",
                            if used_cudnn {
                                "CUDNN_CT3D"
                            } else {
                                "KERNEL_CT3D"
                            },
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
                            groups
                        );
                    }
                    if used_cudnn {
                        continue;
                    }
                    let kernel = conv_transpose3d_kernel(&self.ctx);
                    let total = n * c_out * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(dd)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv_transpose3d launch failed");
                    }
                }
                Step::FusedSwiGLU {
                    in_off,
                    out_off,
                    n_half,
                    total,
                    gate_first,
                } => {
                    let kernel = fused_swiglu_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n_half)
                        .arg(total)
                        .arg(gate_first)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fused_swiglu launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(g_off)
                        .arg(b_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(num_groups)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: group_norm launch failed");
                    }
                }
                Step::GroupNormBackwardInput {
                    x_off,
                    gamma_off,
                    dy_off,
                    out_off,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps_bits,
                } => {
                    let kernel = group_norm_bwd_input_kernel(&self.ctx);
                    let grid = n * num_groups;
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off)
                        .arg(gamma_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(num_groups)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: group_norm_bwd_input launch failed");
                    }
                }
                Step::GroupNormBackwardGamma {
                    x_off,
                    dy_off,
                    out_off,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps_bits,
                } => {
                    let kernel = group_norm_bwd_gamma_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(num_groups)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: group_norm_bwd_gamma launch failed");
                    }
                }
                Step::GroupNormBackwardBeta {
                    dy_off,
                    out_off,
                    n,
                    c,
                    h,
                    w,
                } => {
                    let kernel = group_norm_bwd_beta_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: group_norm_bwd_beta launch failed");
                    }
                }
                Step::BatchNormInference {
                    src_off,
                    g_off,
                    b_off,
                    mean_off,
                    var_off,
                    dst_off,
                    count,
                    channels,
                    eps_bits,
                } => {
                    let kernel = batch_norm_inference_kernel(&self.ctx);
                    let n = count * channels;
                    let (grid, block) = dispatch_grid_1d(n, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(g_off)
                        .arg(b_off)
                        .arg(mean_off)
                        .arg(var_off)
                        .arg(dst_off)
                        .arg(count)
                        .arg(channels)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: batch_norm_inference launch failed");
                    }
                }
                Step::BatchNormInferenceBackwardInput {
                    gamma_off,
                    var_off,
                    dy_off,
                    out_off,
                    count,
                    channels,
                    eps_bits,
                } => {
                    let kernel = batch_norm_inference_bwd_input_kernel(&self.ctx);
                    let n = count * channels;
                    let (grid, block) = dispatch_grid_1d(n, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(gamma_off)
                        .arg(var_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(count)
                        .arg(channels)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: batch_norm_inference_bwd_input launch failed");
                    }
                }
                Step::BatchNormInferenceBackwardGamma {
                    x_off,
                    mean_off,
                    var_off,
                    dy_off,
                    out_off,
                    count,
                    channels,
                    eps_bits,
                } => {
                    let kernel = batch_norm_inference_bwd_gamma_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*channels, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(x_off)
                        .arg(mean_off)
                        .arg(var_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(count)
                        .arg(channels)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: batch_norm_inference_bwd_gamma launch failed");
                    }
                }
                Step::BatchNormInferenceBackwardBeta {
                    dy_off,
                    out_off,
                    count,
                    channels,
                } => {
                    let kernel = batch_norm_inference_bwd_beta_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*channels, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(count)
                        .arg(channels);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: batch_norm_inference_bwd_beta launch failed");
                    }
                }
                Step::LayerNormBackwardInput {
                    x_off,
                    gamma_off,
                    dy_off,
                    out_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    let kernel = layer_norm_bwd_input_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (*rows, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(rows)
                        .arg(h)
                        .arg(x_off)
                        .arg(gamma_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layer_norm_bwd_input launch failed");
                    }
                }
                Step::LayerNormBackwardGamma {
                    x_off,
                    dy_off,
                    out_off,
                    rows,
                    h,
                    eps_bits,
                } => {
                    let kernel = layer_norm_bwd_gamma_kernel(&self.ctx);
                    let cfg = LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(rows)
                        .arg(h)
                        .arg(x_off)
                        .arg(dy_off)
                        .arg(out_off)
                        .arg(eps_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: layer_norm_bwd_gamma launch failed");
                    }
                }
                Step::FakeQuantizeFixed {
                    in_off,
                    scale_off,
                    out_off,
                    n,
                    chan_dim,
                    inner,
                    q_max_bits,
                } => {
                    let kernel = fake_quantize_fixed_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(q_max_bits)
                        .arg(in_off)
                        .arg(scale_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fake_quantize_fixed launch failed");
                    }
                }
                Step::FakeQuantizePerBatch {
                    in_off,
                    out_off,
                    n,
                    chan_dim,
                    inner,
                    q_max_bits,
                } => {
                    let kernel = fake_quantize_perbatch_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*chan_dim, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(q_max_bits)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fake_quantize_perbatch launch failed");
                    }
                }
                Step::FakeQuantizeEma {
                    in_off,
                    scale_off,
                    out_off,
                    n,
                    chan_dim,
                    inner,
                    q_max_bits,
                    decay_bits,
                } => {
                    let kernel = fake_quantize_ema_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*chan_dim, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(q_max_bits)
                        .arg(decay_bits)
                        .arg(in_off)
                        .arg(scale_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fake_quantize_ema launch failed");
                    }
                }
                Step::QuantizeI8 {
                    in_off,
                    q_byte_off,
                    n,
                    chan_dim,
                    inner,
                    meta_idx,
                } => {
                    let kernel = quantize_i8_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(in_off)
                        .arg(q_byte_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: quantize_i8 launch failed");
                    }
                }
                Step::DequantizeI8 {
                    q_byte_off,
                    out_off,
                    n,
                    chan_dim,
                    inner,
                    meta_idx,
                } => {
                    let kernel = dequantize_i8_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(q_byte_off)
                        .arg(out_off)
                        .arg(&self.meta_buffers[*meta_idx]);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: dequantize_i8 launch failed");
                    }
                }
                Step::QMatMul {
                    m,
                    k,
                    n,
                    x_byte_off,
                    w_byte_off,
                    bias_off,
                    out_byte_off,
                    x_zp,
                    w_zp,
                    out_zp,
                    mult_bits,
                } => {
                    let total = *m * *n;
                    let kernel = q_matmul_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(m)
                        .arg(k)
                        .arg(n)
                        .arg(x_byte_off)
                        .arg(w_byte_off)
                        .arg(bias_off)
                        .arg(out_byte_off)
                        .arg(x_zp)
                        .arg(w_zp)
                        .arg(out_zp)
                        .arg(mult_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: q_matmul launch failed");
                    }
                }
                Step::QConv2d {
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
                    x_byte_off,
                    w_byte_off,
                    bias_off,
                    out_byte_off,
                    x_zp,
                    w_zp,
                    out_zp,
                    mult_bits,
                } => {
                    let total = *n * *c_out * *h_out * *w_out;
                    let kernel = q_conv2d_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(x_byte_off)
                        .arg(w_byte_off)
                        .arg(bias_off)
                        .arg(out_byte_off)
                        .arg(x_zp)
                        .arg(w_zp)
                        .arg(out_zp)
                        .arg(mult_bits);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: q_conv2d launch failed");
                    }
                }
                Step::FakeQuantizeLsqBwdX {
                    x_off,
                    scale_off,
                    dy_off,
                    dx_off,
                    n,
                    chan_dim,
                    inner,
                    q_max_bits,
                } => {
                    let kernel = fake_quantize_lsq_bwd_x_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*n, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(q_max_bits)
                        .arg(x_off)
                        .arg(scale_off)
                        .arg(dy_off)
                        .arg(dx_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fake_quantize_lsq_bwd_x launch failed");
                    }
                }
                Step::FakeQuantizeLsqBwdScale {
                    x_off,
                    scale_off,
                    dy_off,
                    dscale_off,
                    n,
                    chan_dim,
                    inner,
                    q_max_bits,
                } => {
                    let kernel = fake_quantize_lsq_bwd_scale_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*chan_dim, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(q_max_bits)
                        .arg(x_off)
                        .arg(scale_off)
                        .arg(dy_off)
                        .arg(dscale_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fake_quantize_lsq_bwd_scale launch failed");
                    }
                }
                Step::FakeQuantizeBackward {
                    x_off,
                    dy_off,
                    dx_off,
                    n,
                    chan_dim,
                    inner,
                    q_max_bits,
                    ste_kind,
                } => {
                    let kernel = fake_quantize_backward_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(*chan_dim, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(chan_dim)
                        .arg(inner)
                        .arg(q_max_bits)
                        .arg(ste_kind)
                        .arg(x_off)
                        .arg(dy_off)
                        .arg(dx_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: fake_quantize_backward launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: resize_nearest_2x launch failed");
                    }
                }
                Step::Interpolate3d {
                    src_off,
                    dst_off,
                    n,
                    c,
                    d_in,
                    h_in,
                    w_in,
                    d_out,
                    h_out,
                    w_out,
                } => {
                    let kernel = interpolate3d_kernel(&self.ctx);
                    let total = n * c * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(src_off)
                        .arg(dst_off)
                        .arg(n)
                        .arg(c)
                        .arg(d_in)
                        .arg(h_in)
                        .arg(w_in)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: interpolate3d launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    // f32-element offsets as u64 — the kernel declares its offset
                    // params `unsigned long long`, so passing u32 here would leave
                    // the high word as stack garbage → CUDA_ERROR_ILLEGAL_ADDRESS.
                    let in_off: u64 = *in_byte_off / 4;
                    let out_off: u64 = *out_byte_off / 4;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(&in_off)
                        .arg(&out_off)
                        .arg(mode);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: complex_cast launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    // f32-element offsets as u64 (kernel params are u64 — see above).
                    let a_off: u64 = *a_byte_off / 4;
                    let b_off: u64 = *b_byte_off / 4;
                    let c_off: u64 = *c_byte_off / 4;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(&a_off)
                        .arg(&b_off)
                        .arg(&c_off)
                        .arg(op)
                        .arg(n_a)
                        .arg(n_b);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: binary_c64 launch failed");
                    }
                }
                Step::ComplexNormSq {
                    n,
                    src_byte_off,
                    dst_byte_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = complex_norm_sq_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let src_off: u64 = *src_byte_off / 4;
                    let dst_off: u64 = *dst_byte_off / 4;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(&src_off)
                        .arg(&dst_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: complex_norm_sq launch failed");
                    }
                }
                Step::ComplexNormSqBackward {
                    n,
                    z_byte_off,
                    g_byte_off,
                    dz_byte_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = complex_norm_sq_backward_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let z_off: u64 = *z_byte_off / 4;
                    let g_off: u64 = *g_byte_off / 4;
                    let dz_off: u64 = *dz_byte_off / 4;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(&z_off)
                        .arg(&g_off)
                        .arg(&dz_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: complex_norm_sq_backward launch failed");
                    }
                }
                Step::ConjugateC64 {
                    n,
                    src_byte_off,
                    dst_byte_off,
                } => {
                    let n_s = scale(*n);
                    if n_s == 0 {
                        continue;
                    }
                    let kernel = conjugate_c64_kernel(&self.ctx);
                    let (grid, block) = dispatch_grid_1d(n_s, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let src_off: u64 = *src_byte_off / 4;
                    let dst_off: u64 = *dst_byte_off / 4;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(&n_s)
                        .arg(&src_off)
                        .arg(&dst_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conjugate_c64 launch failed");
                    }
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
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    #[cfg(feature = "native-splat")]
                    crate::splat_native::run_gaussian_splat_render_native(
                        &stream,
                        buf,
                        arena_size,
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
                        &stream,
                        buf,
                        arena_size,
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
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::splat_host::run_gaussian_splat_prepare(
                        &stream,
                        buf,
                        arena_size,
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
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::splat_host::run_gaussian_splat_rasterize(
                        &stream,
                        buf,
                        arena_size,
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
                    let (buf, arena_size) = self.arena.f32_buf_and_size();
                    crate::splat_host::run_gaussian_splat_render_backward(
                        &stream,
                        buf,
                        arena_size,
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
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *h,
                        (*x_byte_off / 4) as u32,
                        (*gamma_byte_off / 4) as u32,
                        (*beta_byte_off / 4) as u32,
                        (*dy_byte_off / 4) as u32,
                        (*dx_byte_off / 4) as u32,
                        *eps_bits,
                        0,
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
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *h,
                        (*x_byte_off / 4) as u32,
                        (*gamma_byte_off / 4) as u32,
                        (*beta_byte_off / 4) as u32,
                        (*dy_byte_off / 4) as u32,
                        (*dgamma_byte_off / 4) as u32,
                        *eps_bits,
                        1,
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
                    launch_rms_norm_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *h,
                        (*x_byte_off / 4) as u32,
                        (*gamma_byte_off / 4) as u32,
                        (*beta_byte_off / 4) as u32,
                        (*dy_byte_off / 4) as u32,
                        (*dbeta_byte_off / 4) as u32,
                        *eps_bits,
                        2,
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
                    launch_rope_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *batch,
                        *seq,
                        *hidden,
                        *head_dim,
                        *n_rot,
                        (*dy_byte_off / 4) as u32,
                        (*cos_byte_off / 4) as u32,
                        (*sin_byte_off / 4) as u32,
                        (*dx_byte_off / 4) as u32,
                        *cos_len,
                    );
                }
                Step::CumsumBackward {
                    dy_byte_off,
                    dx_byte_off,
                    rows,
                    cols,
                    exclusive,
                } => {
                    launch_cumsum_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *rows,
                        *cols,
                        (*dy_byte_off / 4) as u32,
                        (*dx_byte_off / 4) as u32,
                        if *exclusive { 1 } else { 0 },
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
                    launch_gather_bwd(
                        &self.ctx,
                        &stream,
                        self.arena.f32_buf_mut(),
                        *outer,
                        *axis_dim,
                        *num_idx,
                        *trailing,
                        (*dy_byte_off / 4) as u32,
                        (*indices_byte_off / 4) as u32,
                        (*dst_byte_off / 4) as u32,
                    );
                }
                Step::MaxPool2dBackward {
                    x_byte_off,
                    dy_byte_off,
                    dx_byte_off,
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
                } => {
                    let kernel = maxpool2d_backward_kernel(&self.ctx);
                    let total = n * c * h * w;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let x_o = (*x_byte_off / 4) as u32;
                    let dy_o = (*dy_byte_off / 4) as u32;
                    let dx_o = (*dx_byte_off / 4) as u32;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(&x_o)
                        .arg(&dy_o)
                        .arg(&dx_o);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: maxpool2d_backward launch failed");
                    }
                }
                Step::Conv2dBackwardInput {
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
                    dw,
                    groups,
                } => {
                    // Match the backward-filter routing: only grouped/degenerate
                    // shapes go to the host path; normal 2-D convs keep fast cuDNN.
                    let cudnn_ok_shape = *groups == 1 && *kh > 1 && *kw > 1;
                    let allow_cudnn = !rlx_ir::env::flag("RLX_CUDA_CONV_FORCE_GATHER")
                        && (cudnn_ok_shape || rlx_ir::env::flag("RLX_CUDA_CONV_BWD_CUDNN"));
                    let used_cudnn = if allow_cudnn
                        && let (Some(handle), Some(workspace)) =
                            (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _wr) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_backward_data(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                *h,
                                *w_in,
                                *h_out,
                                *w_out,
                                *kh,
                                *kw,
                                *sh,
                                *sw,
                                *ph,
                                *pw,
                                *dh,
                                *dw,
                                *groups,
                                (*dy_byte_off / 4) as u32,
                                (*w_byte_off / 4) as u32,
                                (*dx_byte_off / 4) as u32,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d_bwd_data.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if rlx_ir::env::flag("RLX_CUDA_CONV_TRACE") {
                        let path = if used_cudnn {
                            "cuDNN"
                        } else if rlx_ir::env::flag("RLX_CUDA_CONV_BWD_HOST") {
                            "HOST"
                        } else {
                            "gather-kernel"
                        };
                        eprintln!(
                            "[CONV-TRACE] Conv2dBackwardInput n={} c_in={} c_out={} {}x{} k={}x{} g={} | cudnn_ok_shape={} dnn_loaded={} -> {path}",
                            *n,
                            *c_in,
                            *c_out,
                            *h,
                            *w_in,
                            *kh,
                            *kw,
                            *groups,
                            cudnn_ok_shape,
                            self.dnn.is_some()
                        );
                    }
                    if !used_cudnn {
                        // No cuDNN: default to the direct device gather kernel
                        // (one thread per dx element, no atomics) instead of the
                        // slow D2H→CPU→H2D host path. `RLX_CUDA_CONV_BWD_HOST=1`
                        // forces the host reference (parity/debug).
                        if rlx_ir::env::flag("RLX_CUDA_CONV_BWD_HOST") {
                            let buf = self.arena.f32_buf_mut();
                            crate::training_bwd_host::run_conv2d_backward_input(
                                &stream,
                                buf,
                                *dy_byte_off as usize / 4,
                                *w_byte_off as usize / 4,
                                *dx_byte_off as usize / 4,
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
                                *dw,
                                *groups,
                            );
                        } else {
                            let kernel = conv2d_backward_input_kernel(&self.ctx);
                            let total = n * c_in * h * w_in;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let (dy_e, w_e, dx_e) = (
                                (*dy_byte_off / 4) as u32,
                                (*w_byte_off / 4) as u32,
                                (*dx_byte_off / 4) as u32,
                            );
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(n)
                                .arg(c_in)
                                .arg(c_out)
                                .arg(h)
                                .arg(w_in)
                                .arg(h_out)
                                .arg(w_out)
                                .arg(kh)
                                .arg(kw)
                                .arg(sh)
                                .arg(sw)
                                .arg(ph)
                                .arg(pw)
                                .arg(dh)
                                .arg(dw)
                                .arg(groups)
                                .arg(&dy_e)
                                .arg(&w_e)
                                .arg(&dx_e);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: conv2d_backward_input launch failed");
                            }
                        }
                    }
                }
                Step::Conv2dBackwardWeight {
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
                    // cuDNN's backward-filter v7 heuristic returns algos that
                    // compute WRONG/nondeterministic gradients for grouped/depthwise
                    // convs and the degenerate 1×k / k×1 shapes EEGNet uses — EEG
                    // training collapsed ~5/8 on cuDNN but is bit-exact on the host
                    // im2col path (verified 8/8, loss→0.810). Normal 2-D convs
                    // (kh>1, kw>1, groups=1) — cuDNN's core case — stay on the fast
                    // cuDNN backward; only the broken shapes route to the (slower
                    // but correct) host path. `RLX_CUDA_CONV_BWD_CUDNN=1` forces
                    // cuDNN for all backward shapes; forward mirrors this routing
                    // unless `RLX_CUDA_CONV_FWD_CUDNN=1` opts it back in.
                    let cudnn_ok_shape = *groups == 1 && *kh > 1 && *kw > 1;
                    let allow_cudnn = !rlx_ir::env::flag("RLX_CUDA_CONV_FORCE_GATHER")
                        && (cudnn_ok_shape || rlx_ir::env::flag("RLX_CUDA_CONV_BWD_CUDNN"));
                    let used_cudnn = if allow_cudnn
                        && let (Some(handle), Some(workspace)) =
                            (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _wr) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_backward_filter(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
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
                                *dh,
                                *dw_dil,
                                *groups,
                                (*x_byte_off / 4) as u32,
                                (*dy_byte_off / 4) as u32,
                                (*dw_byte_off / 4) as u32,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d_bwd_filter.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if rlx_ir::env::flag("RLX_CUDA_CONV_TRACE") {
                        let path = if used_cudnn {
                            "cuDNN"
                        } else if rlx_ir::env::flag("RLX_CUDA_CONV_BWD_HOST") {
                            "HOST"
                        } else {
                            "gather-kernel"
                        };
                        eprintln!(
                            "[CONV-TRACE] Conv2dBackwardWeight n={} c_in={} c_out={} {}x{} k={}x{} g={} | cudnn_ok_shape={} dnn_loaded={} -> {path}",
                            *n,
                            *c_in,
                            *c_out,
                            *h,
                            *w,
                            *kh,
                            *kw,
                            *groups,
                            cudnn_ok_shape,
                            self.dnn.is_some()
                        );
                    }
                    if !used_cudnn {
                        // No cuDNN: direct device gather kernel (one thread per dw
                        // element) instead of the host D2H→CPU→H2D path.
                        // `RLX_CUDA_CONV_BWD_HOST=1` forces the host reference.
                        if rlx_ir::env::flag("RLX_CUDA_CONV_BWD_HOST") {
                            let buf = self.arena.f32_buf_mut();
                            crate::training_bwd_host::run_conv2d_backward_weight(
                                &stream,
                                buf,
                                *x_byte_off as usize / 4,
                                *dy_byte_off as usize / 4,
                                *dw_byte_off as usize / 4,
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
                        } else {
                            let kernel = conv2d_backward_weight_kernel(&self.ctx);
                            let c_in_per_g = c_in / groups;
                            let total = c_out * c_in_per_g * kh * kw;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let (x_e, dy_e, dw_e) = (
                                (*x_byte_off / 4) as u32,
                                (*dy_byte_off / 4) as u32,
                                (*dw_byte_off / 4) as u32,
                            );
                            let mut launcher = stream.launch_builder(&kernel.function);
                            launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(n)
                                .arg(c_in)
                                .arg(c_out)
                                .arg(h)
                                .arg(w)
                                .arg(h_out)
                                .arg(w_out)
                                .arg(kh)
                                .arg(kw)
                                .arg(sh)
                                .arg(sw)
                                .arg(ph)
                                .arg(pw)
                                .arg(dh)
                                .arg(dw_dil)
                                .arg(groups)
                                .arg(&x_e)
                                .arg(&dy_e)
                                .arg(&dw_e);
                            unsafe {
                                launcher
                                    .launch(cfg)
                                    .expect("rlx-cuda: conv2d_backward_weight launch failed");
                            }
                        }
                    }
                }
                Step::MaxPool3dBackward {
                    x_byte_off,
                    dy_byte_off,
                    dx_byte_off,
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
                } => {
                    let kernel = maxpool3d_backward_kernel(&self.ctx);
                    let total = n * c * d * h * w;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let x_o = (*x_byte_off / 4) as u32;
                    let dy_o = (*dy_byte_off / 4) as u32;
                    let dx_o = (*dx_byte_off / 4) as u32;
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(&x_o)
                        .arg(&dy_o)
                        .arg(&dx_o);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: maxpool3d_backward launch failed");
                    }
                }
                Step::Conv3dBackwardInput {
                    dy_byte_off,
                    w_byte_off,
                    dx_byte_off,
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
                    let cudnn_ok_shape = *groups == 1 && *kd > 1 && *kh > 1 && *kw > 1;
                    let allow_cudnn = !rlx_ir::env::flag("RLX_CUDA_CONV_FORCE_GATHER")
                        && (cudnn_ok_shape || rlx_ir::env::flag("RLX_CUDA_CONV_BWD_CUDNN"));
                    let used_cudnn = if allow_cudnn
                        && let (Some(handle), Some(workspace)) =
                            (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _wr) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv3d_backward_data(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
                                *n,
                                *c_in,
                                *c_out,
                                *d,
                                *h,
                                *w_in,
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
                                (*dy_byte_off / 4) as u32,
                                (*w_byte_off / 4) as u32,
                                (*dx_byte_off / 4) as u32,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv3d_bwd_data.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    record_conv3d_bwd_path(used_cudnn);
                    if !used_cudnn {
                        let kernel = conv3d_backward_input_kernel(&self.ctx);
                        let total = n * c_in * d * h * w_in;
                        let (grid, block) = dispatch_grid_1d(total, 256);
                        let cfg = LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let (dy_e, w_e, dx_e) = (
                            (*dy_byte_off / 4) as u32,
                            (*w_byte_off / 4) as u32,
                            (*dx_byte_off / 4) as u32,
                        );
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(n)
                            .arg(c_in)
                            .arg(c_out)
                            .arg(d)
                            .arg(h)
                            .arg(w_in)
                            .arg(d_out)
                            .arg(h_out)
                            .arg(w_out)
                            .arg(kd)
                            .arg(kh)
                            .arg(kw)
                            .arg(sd)
                            .arg(sh)
                            .arg(sw)
                            .arg(pd)
                            .arg(ph)
                            .arg(pw)
                            .arg(dd)
                            .arg(dh)
                            .arg(dw)
                            .arg(groups)
                            .arg(&dy_e)
                            .arg(&w_e)
                            .arg(&dx_e);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: conv3d_backward_input launch failed");
                        }
                    }
                }
                Step::Conv3dBackwardWeight {
                    x_byte_off,
                    dy_byte_off,
                    dw_byte_off,
                    n,
                    c_in,
                    d,
                    h,
                    w,
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
                    dw_dil,
                    groups,
                } => {
                    let cudnn_ok_shape = *groups == 1 && *kd > 1 && *kh > 1 && *kw > 1;
                    let allow_cudnn = !rlx_ir::env::flag("RLX_CUDA_CONV_FORCE_GATHER")
                        && (cudnn_ok_shape || rlx_ir::env::flag("RLX_CUDA_CONV_BWD_CUDNN"));
                    let used_cudnn = if allow_cudnn
                        && let (Some(handle), Some(workspace)) =
                            (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _wr) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _ar) = self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv3d_backward_filter(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
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
                                *dw_dil,
                                *groups,
                                (*x_byte_off / 4) as u32,
                                (*dy_byte_off / 4) as u32,
                                (*dw_byte_off / 4) as u32,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv3d_bwd_filter.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    record_conv3d_bwd_path(used_cudnn);
                    if !used_cudnn {
                        let kernel = conv3d_backward_weight_kernel(&self.ctx);
                        let c_in_per_g = c_in / groups;
                        let total = c_out * c_in_per_g * kd * kh * kw;
                        let (grid, block) = dispatch_grid_1d(total, 256);
                        let cfg = LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let (x_e, dy_e, dw_e) = (
                            (*x_byte_off / 4) as u32,
                            (*dy_byte_off / 4) as u32,
                            (*dw_byte_off / 4) as u32,
                        );
                        let mut launcher = stream.launch_builder(&kernel.function);
                        launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(n)
                            .arg(c_in)
                            .arg(c_out)
                            .arg(d)
                            .arg(h)
                            .arg(w)
                            .arg(d_out)
                            .arg(h_out)
                            .arg(w_out)
                            .arg(kd)
                            .arg(kh)
                            .arg(kw)
                            .arg(sd)
                            .arg(sh)
                            .arg(sw)
                            .arg(pd)
                            .arg(ph)
                            .arg(pw)
                            .arg(dd)
                            .arg(dh)
                            .arg(dw_dil)
                            .arg(groups)
                            .arg(&x_e)
                            .arg(&dy_e)
                            .arg(&dw_e);
                        unsafe {
                            launcher
                                .launch(cfg)
                                .expect("rlx-cuda: conv3d_backward_weight launch failed");
                        }
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(l)
                        .arg(l_out)
                        .arg(kl)
                        .arg(sl)
                        .arg(pl)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool1d launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool2d launch failed");
                    }
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
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(op)
                        .arg(in_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: pool3d launch failed");
                    }
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
                    // Tier 1: cuDNN — 1-D conv as a degenerate 2-D conv
                    // with H=1, kh=1, sh=1, ph=0, dh=1. Same descriptors
                    // as conv2d; the H axis just collapses to 1.
                    let used_cudnn = if let (Some(handle), Some(workspace)) =
                        (self.dnn, self.dnn_workspace.as_ref())
                    {
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv2d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
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
                                /*dh*/ 1,
                                *dl,
                                *groups,
                                *in_off,
                                *w_off,
                                *out_off,
                            )
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv1d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv1d_kernel(&self.ctx);
                    let total = n * c_out * l_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(l)
                        .arg(l_out)
                        .arg(kl)
                        .arg(sl)
                        .arg(pl)
                        .arg(dl)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv1d launch failed");
                    }
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
                    has_bias,
                    bias_off_f32,
                    act_id,
                    has_residual,
                    residual_off_f32,
                } => {
                    // cuDNN's v7 backward-filter heuristic is known to produce
                    // wrong/nondeterministic results for grouped/depthwise and
                    // degenerate 1×k / k×1 convs. Keep normal 2-D convs on cuDNN;
                    // for the suspect shapes prefer the device direct kernel
                    // (inference-sized feature maps OOM the host im2col path).
                    // `RLX_CUDA_CONV_FWD_HOST=1` forces the CPU reference (training
                    // parity with host backward). `RLX_CUDA_CONV_FWD_CUDNN=1`
                    // restores cuDNN forward for experimentation on those shapes.
                    let cudnn_ok_shape = *groups == 1 && *kh > 1 && *kw > 1;
                    // `cudnnConvolutionBiasActivationForward` only supports
                    // IDENTITY and RELU as the fused epilogue activation
                    // (sigmoid/tanh/etc. return NOT_SUPPORTED). For those, skip
                    // the doomed cuDNN probe and use the direct kernel +
                    // `conv_bias_act_epilogue` straight away. Plain conv
                    // (`has_bias==0`, act_id 0xFFFF) is unaffected.
                    let cudnn_epilogue_ok = *has_bias == 0 || *act_id == 0xFFFFu32 || *act_id == 0;
                    let try_cudnn = (cudnn_ok_shape
                        || rlx_ir::env::flag("RLX_CUDA_CONV_FWD_CUDNN"))
                        && cudnn_epilogue_ok
                        && self.dnn.is_some()
                        && self.dnn_workspace.is_some()
                        && !rlx_ir::env::flag("RLX_CUDA_NO_CUDNN");
                    let used_cudnn = if try_cudnn {
                        let handle = self.dnn.expect("dnn handle");
                        let workspace = self.dnn_workspace.as_ref().expect("dnn workspace");
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            if *has_bias != 0 {
                                // Fold bias + activation into the conv in one
                                // cuDNN call (`cudnnConvolutionBiasActivationForward`).
                                cudnn_conv2d_bias_act_forward(
                                    handle,
                                    ws_ptr,
                                    CUDNN_WORKSPACE_BYTES,
                                    arena_ptr,
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
                                    *dh,
                                    *dw,
                                    *groups,
                                    *in_off,
                                    *w_off,
                                    *out_off,
                                    *bias_off_f32,
                                    *act_id,
                                    *residual_off_f32,
                                    *has_residual != 0,
                                )
                            } else {
                                cudnn_conv2d_forward(
                                    handle,
                                    ws_ptr,
                                    CUDNN_WORKSPACE_BYTES,
                                    arena_ptr,
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
                                    *dh,
                                    *dw,
                                    *groups,
                                    *in_off,
                                    *w_off,
                                    *out_off,
                                )
                            }
                        };
                        if let Err(ref e) = r {
                            log_fallback("conv2d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    if used_cudnn {
                        if *has_bias != 0 && rlx_ir::env::flag("RLX_CUDA_LOG_CONV_PATH") {
                            eprintln!(
                                "rlx-cuda-convpath: CUDNN_FUSED c_out={c_out} kh={kh} kw={kw} groups={groups} act={act_id}"
                            );
                        }
                        continue;
                    }

                    // Opt-in host reference (matches training backward). Default
                    // is the device direct kernel below — required for large
                    // 1×k / grouped inference graphs (e.g. F5-TTS DiT).
                    if !cudnn_ok_shape && rlx_ir::env::flag("RLX_CUDA_CONV_FWD_HOST") {
                        let (buf, arena_size) = self.arena.f32_buf_and_size();
                        crate::training_bwd_host::run_conv2d_forward(
                            &stream, buf, arena_size, *in_off, *w_off, *out_off, *n, *c_in, *c_out,
                            *h, *w, *h_out, *w_out, *kh, *kw, *sh, *sw, *ph, *pw, *dh, *dw,
                            *groups,
                        );
                        if *has_bias != 0 {
                            let ep = conv_bias_act_epilogue_kernel(&self.ctx);
                            let total = n * c_out * h_out * w_out;
                            let hw = h_out * w_out;
                            let (grid, block) = dispatch_grid_1d(total, 256);
                            let ep_cfg = LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (block, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            let mut ep_launcher = stream.launch_builder(&ep.function);
                            ep_launcher
                                .arg(self.arena.f32_buf_mut())
                                .arg(&total)
                                .arg(&hw)
                                .arg(c_out)
                                .arg(out_off)
                                .arg(has_bias)
                                .arg(bias_off_f32)
                                .arg(act_id)
                                .arg(has_residual)
                                .arg(residual_off_f32);
                            unsafe {
                                ep_launcher.launch(ep_cfg).expect(
                                    "rlx-cuda: conv_bias_act_epilogue (host) launch failed",
                                );
                            }
                        }
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv2d_kernel(&self.ctx);
                    let total = n * c_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(h)
                        .arg(w)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kh)
                        .arg(kw)
                        .arg(sh)
                        .arg(sw)
                        .arg(ph)
                        .arg(pw)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv2d launch failed");
                    }
                    // Fold bias + activation after the direct conv (cuDNN's
                    // fused call handles these on its own path above).
                    if *has_bias != 0 {
                        if rlx_ir::env::flag("RLX_CUDA_LOG_CONV_PATH") {
                            eprintln!(
                                "rlx-cuda-convpath: EPILOGUE c_out={c_out} kh={kh} kw={kw} groups={groups} act={act_id}"
                            );
                        }
                        let ep = conv_bias_act_epilogue_kernel(&self.ctx);
                        let hw = h_out * w_out;
                        let ep_cfg = LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut ep_launcher = stream.launch_builder(&ep.function);
                        ep_launcher
                            .arg(self.arena.f32_buf_mut())
                            .arg(&total)
                            .arg(&hw)
                            .arg(c_out)
                            .arg(out_off)
                            .arg(has_bias)
                            .arg(bias_off_f32)
                            .arg(act_id)
                            .arg(has_residual)
                            .arg(residual_off_f32);
                        unsafe {
                            ep_launcher
                                .launch(ep_cfg)
                                .expect("rlx-cuda: conv_bias_act_epilogue launch failed");
                        }
                    }
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
                    // Tier 1: cuDNN nd-conv (NCDHW + 3-D pads/strides/dilations).
                    // Opt out with `RLX_CUDA_NO_CUDNN=1` (parity / kernel-only).
                    let try_cudnn = self.dnn.is_some()
                        && self.dnn_workspace.is_some()
                        && !rlx_ir::env::flag("RLX_CUDA_NO_CUDNN");
                    let used_cudnn = if try_cudnn {
                        let handle = self.dnn.expect("dnn handle");
                        let workspace = self.dnn_workspace.as_ref().expect("dnn workspace");
                        let mut workspace = workspace.lock().unwrap();
                        let (ws_ptr, _ws_record) = workspace.device_ptr_mut(&stream);
                        let (arena_ptr, _arena_record) =
                            self.arena.f32_buf_mut().device_ptr_mut(&stream);
                        let r = unsafe {
                            cudnn_conv3d_forward(
                                handle,
                                ws_ptr,
                                CUDNN_WORKSPACE_BYTES,
                                arena_ptr,
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
                            log_fallback("conv3d.cudnn", e);
                        }
                        r.is_ok()
                    } else {
                        false
                    };
                    record_conv3d_path(used_cudnn);
                    if rlx_ir::env::flag("RLX_CUDA_LOG_CONV_PATH") {
                        eprintln!(
                            "rlx-cuda-convpath: {} n={} c_in={} c_out={} {}x{}x{} k={}x{}x{} g={}",
                            if used_cudnn {
                                "CUDNN_CONV3D"
                            } else {
                                "KERNEL_CONV3D"
                            },
                            n,
                            c_in,
                            c_out,
                            d,
                            h,
                            w,
                            kd,
                            kh,
                            kw,
                            groups
                        );
                    }
                    if used_cudnn {
                        continue;
                    }

                    // Fallback: custom direct-convolution kernel.
                    let kernel = conv3d_kernel(&self.ctx);
                    let total = n * c_out * d_out * h_out * w_out;
                    let (grid, block) = dispatch_grid_1d(total, 256);
                    let cfg = LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut launcher = stream.launch_builder(&kernel.function);
                    launcher
                        .arg(self.arena.f32_buf_mut())
                        .arg(n)
                        .arg(c_in)
                        .arg(c_out)
                        .arg(d)
                        .arg(h)
                        .arg(w)
                        .arg(d_out)
                        .arg(h_out)
                        .arg(w_out)
                        .arg(kd)
                        .arg(kh)
                        .arg(kw)
                        .arg(sd)
                        .arg(sh)
                        .arg(sw)
                        .arg(pd)
                        .arg(ph)
                        .arg(pw)
                        .arg(dd)
                        .arg(dh)
                        .arg(dw)
                        .arg(groups)
                        .arg(in_off)
                        .arg(w_off)
                        .arg(out_off);
                    unsafe {
                        launcher
                            .launch(cfg)
                            .expect("rlx-cuda: conv3d launch failed");
                    }
                }
            }

            if cap_debug && !cap_invalidated {
                match stream.capture_status() {
                    Ok(
                        cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE,
                    ) => {}
                    other => {
                        eprintln!(
                            "rlx-cuda: capture invalidated AT step {step_i} = {} (status {other:?})",
                            step_name(step)
                        );
                        cap_invalidated = true;
                    }
                }
            }

            // Multi-stream tail: record an event so future steps can
            // wait on this one, then update producer_of with the
            // offsets this step wrote.
            if let Some(idx) = assigned_idx {
                if let Ok(evt) = stream.record_event(None) {
                    last_event.insert(idx, evt);
                }
                let (_, writes) = step_offsets(step);
                for w in &writes {
                    producer_of.insert(*w, idx);
                }
            }
            if let Some(t0) = _prof_t0 {
                let _ = default_stream.synchronize();
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                let key: &'static str = match step {
                    Step::CustomHost { name, .. } => {
                        // Leak short labels so the existing HashMap<&'static str, …>
                        // profiler can break CustomHost down by op name.
                        Box::leak(format!("CustomHost:{name}").into_boxed_str())
                    }
                    Step::CudaGpuKernel { name, .. } => {
                        Box::leak(format!("CudaGpu:{name}").into_boxed_str())
                    }
                    _ => step_name(step),
                };
                let e = step_prof.entry(key).or_insert((0.0, 0));
                e.0 += dt;
                e.1 += 1;
            }
        }
        // A capture open at the last step (a Gpu segment ending at `len`) has no
        // next iteration to close it — close it here.
        if cur_cap_end.is_some() {
            seg_capture_failed |= !close_seg(&seg_stream, &mut captured_segs);
        }
        // A segment capture broke (mis-classified op): its data was recorded but
        // not executed → arena stale. Disable capture + re-run eagerly (correct).
        if seg_capture && seg_capture_failed {
            self.capture_disabled = true;
            self.segment_graphs.clear();
            unsafe { self.ctx.enable_event_tracking() };
            if let Some(blas) = self.blas.as_ref() {
                let blas = blas.lock().unwrap();
                unsafe {
                    let _ = cudarc::cublas::result::set_stream(
                        *blas.handle(),
                        default_stream.cu_stream() as _,
                    );
                }
            }
            if rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG") {
                eprintln!(
                    "rlx-cuda: segment capture failed → disabling capture, re-running eagerly"
                );
            }
            return self.run_inner(inputs);
        }
        // Commit segment graphs only if EVERY Gpu segment captured cleanly (else
        // leave `segment_graphs` empty so the next run retries — this run has
        // already computed correct data via the launches above and the eager
        // spans).
        if seg_capture {
            if captured_segs.len() == seg_num_gpu {
                self.segment_graphs = captured_segs;
                if rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG") {
                    let replayable: usize = seg_plan
                        .iter()
                        .filter_map(|s| match s {
                            CaptureSegment::Gpu { start, end } => Some(end - start),
                            _ => None,
                        })
                        .sum();
                    eprintln!(
                        "rlx-cuda: segmented capture built {seg_num_gpu} graph(s) \
                         ({replayable}/{} steps replayable)",
                        self.schedule.len(),
                    );
                }
            } else if rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG") {
                eprintln!(
                    "rlx-cuda: segmented capture incomplete ({}/{seg_num_gpu} segments); \
                     retrying next run",
                    captured_segs.len(),
                );
            }
        }
        if want_capture_stream {
            // First graph-mode run is the eager warm-up — capture starts next run
            // (loads kernel modules + cuBLAS workspace, illegal mid-capture).
            self.capture_warmed = true;
        }
        // Graph-mode dispatch + graph launches ran on `seg_stream`; sync it so the
        // null-stream readback below sees all produced data, and reset cuBLAS /
        // cuDNN back to the null stream (whole-graph and segmented alike).
        // CRITICAL: do NOT sync while a whole-graph capture is still open
        // (`capturing`) — a stream sync mid-capture INVALIDATES it (this is why
        // whole-graph capture failed while segmented, which closes each segment
        // in-loop before this point, succeeded). end_capture + the final
        // stream sync below cover it.
        if want_capture_stream {
            if !capturing {
                let _ = seg_stream.synchronize();
            }
            if let Some(blas) = self.blas.as_ref() {
                let blas = blas.lock().unwrap();
                unsafe {
                    let _ = cudarc::cublas::result::set_stream(
                        *blas.handle(),
                        default_stream.cu_stream() as _,
                    );
                }
            }
            if let Some(handle) = self.dnn {
                unsafe {
                    let _ = cudarc::cudnn::result::set_stream(
                        handle,
                        default_stream.cu_stream() as cudnn_sys::cudaStream_t,
                    );
                }
            }
        }
        if step_profile {
            let mut v: Vec<_> = step_prof.iter().collect();
            v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
            let total: f64 = v.iter().map(|(_, (ms, _))| ms).sum();
            eprintln!("rlx-cuda: step profile (total {total:.1}ms):");
            for (name, (ms, n)) in v.iter().take(20) {
                eprintln!(
                    "  {name:<28} {ms:8.2}ms  ({n}×, {:.3}ms/call)",
                    ms / *n as f64
                );
            }
        }

        // Multi-stream: sync every pool stream so output reads see all
        // produced data.
        if multi_stream {
            for s in &self.streams {
                let _ = s.synchronize();
            }
        }

        self.prepare_readback_plan();
        let plan = self.readback_plan_buf.clone();
        run_tail_host_audio_ops(&self.schedule, &stream, self.arena.f32_buf_mut(), true);
        if !self.gpu_handle_feeds.is_empty() {
            self.propagate_gpu_handle_feeds_d2d(&stream);
        }
        let read_all = plan.len() == self.graph.outputs.len();

        if capturing {
            // End capture before dtoh — the graph records compute kernels only.
            let cu_graph = match stream.end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags
                    ::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
            ) {
                Ok(g) => g,
                Err(e) => {
                    if rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG") {
                        eprintln!("rlx-cuda: whole-graph end_capture failed: {e:?}");
                    }
                    // Capture invalidated mid-dispatch — leave `captured_graph`
                    // None (eager this run + retry) rather than crash.
                    None
                }
            };
            match cu_graph {
                Some(g) => {
                    g.upload().expect("rlx-cuda: graph upload failed");
                    g.launch().expect("rlx-cuda: graph first launch failed");
                    self.captured_graph = Some(g);
                    self.captured_readback_plan = Some(plan.clone());
                    if rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG") {
                        eprintln!(
                            "rlx-cuda: whole-graph capture built ({} steps)",
                            self.schedule.len()
                        );
                    }
                }
                None => {
                    // Capture failed (a step classified capture-safe broke it):
                    // the recorded kernels may NOT have executed, so the arena is
                    // stale → the outputs below would be WRONG. Permanently disable
                    // capture for this schedule, restore state, and re-run from the
                    // top EAGERLY (correct). The safety net for a mis-classified op.
                    self.capture_disabled = true;
                    unsafe { self.ctx.enable_event_tracking() };
                    if rlx_ir::env::flag("RLX_CUDA_CAPTURE_DEBUG") {
                        eprintln!(
                            "rlx-cuda: capture failed → disabling capture for this \
                             schedule, re-running eagerly"
                        );
                    }
                    return self.run_inner(inputs);
                }
            }
        }

        if read_all {
            self.fill_output_staging(&stream)
                .expect("rlx-cuda: output dtoh failed");
        } else {
            self.fill_output_staging_indices(&stream, &plan)
                .expect("rlx-cuda: partial output dtoh failed");
        }
        self.refresh_gpu_handles_from_staging(&plan);
        stream.synchronize().expect("rlx-cuda: stream sync failed");
        // Restore cudarc's cross-stream event tracking that the segmented
        // dispatch disabled — AFTER readback + final sync, so the whole
        // single-stream run (dispatch, capture, dtoh) stays consistently
        // untracked. Context-wide state; must be back on for other exec modes.
        if want_capture_stream {
            unsafe { self.ctx.enable_event_tracking() };
        }
        self.outputs_from_staging_plan(&plan)
    }
}
