// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `encode` — extracted from the `backend` module for navigability (see `mod.rs`).

#![allow(unused_imports)]

use crate::arena::Arena;
use crate::device::metal_device;
use crate::kernels::kernels;
use crate::thunk::{Thunk, ThunkSchedule};
use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory;
use std::collections::{HashMap, HashSet};

use super::*;

mod ops;
pub use ops::*;

/// Check a just-completed command buffer for a GPU-side failure instead of
/// silently proceeding to read the arena.
///
/// Root cause of the intermittent "Whisper garbage" / NaN-DiT-output bug
/// (F5-TTS Metal, task: investigate Metal DiT vs CPU divergence): every
/// `commit(); wait_until_completed();` call site in this file ignored
/// `MTLCommandBuffer.status`. On a machine with other processes also
/// driving the GPU concurrently (observed here: a second `rlx-qwen35
/// --device gpu` inference and a `wgpu` backend test running at the same
/// time), Metal can occasionally time out, fault, or otherwise fail a
/// command buffer under that contention (`MTLCommandBufferError::Timeout` /
/// `PageFault`). `wait_until_completed()` still returns normally in that
/// case — it does not panic or block forever — so the arena silently keeps
/// whatever was partially written (some rows computed, others left at
/// their pre-existing/undefined content), which downstream reads as
/// scattered NaNs (most visibly in in-place reduction kernels like
/// Softmax, which read back the same buffer they just wrote). Re-running
/// in isolation (no concurrent GPU load) made the corruption disappear
/// entirely across dozens of trials, confirming it was contention-driven
/// GPU failures, not a kernel or dispatch bug. Fail loudly here so a
/// future occurrence reads as a clear GPU error instead of mysterious
/// silent output corruption several ops downstream.
pub(crate) fn check_cmd_buf_status(cmd_buf: &metal::CommandBufferRef, where_: &str) {
    let status = cmd_buf.status();
    if status == metal::MTLCommandBufferStatus::Error {
        panic!(
            "rlx-metal: command buffer failed ({where_}): status=Error. This is a GPU-side \
             failure (timeout / page fault / device error), not a data bug — it usually means \
             the GPU was heavily contended by another process (concurrent Metal/MLX/wgpu \
             workload) or the system was under thermal/power pressure. Retry once the GPU is \
             free; set RLX_METAL_CMDBUF_TRACE=1 to log non-fatal near-misses too."
        );
    } else if rlx_ir::env::flag("RLX_METAL_CMDBUF_TRACE") {
        eprintln!("[rlx-metal] {where_}: command buffer status={status:?}");
    }
}

impl MetalExecutable {
    pub(crate) fn encode_and_run(&mut self) {
        if crate::thunk_profile::enabled() {
            self.run_thunk_profile();
            return;
        }
        use std::time::Instant;
        // First-run freeze: re-lower with params baked in as MPSGraph
        // constants so the optimizer can specialize matmul kernels
        // around the actual weight shapes / fold reshapes through
        // them, and the per-call feed list shrinks to just the
        // model's Input tensors.
        //
        // Opt-in via RLX_MPSGRAPH_PARAM_CONST=1 because every baked
        // constant ends up retained inside the MPSGraphExecutable
        // (separate from our arena), and 280 params × hundreds of MB
        // can quickly outweigh the kernel-specialization gain — and
        // OOM the host when the caller compiles many shapes back to
        // back (the prefill matrix harness, for example, hits 12 ×
        // ~600 MB without a tight cap). The
        // `RLX_MPSGRAPH_PARAM_CONST_CAP` knob lets callers tune the
        // per-param byte ceiling once they've opted in.
        // Hybrid schedule-split: attention stays on thunks; matmul/LN run as
        // MPSGraph segments. On big arenas (≥4 GiB) staging high-offset feeds
        // can OOM; keep all-thunks unless RLX_METAL_HYBRID_BIG_ARENA=1.
        // F16 Linear weights + unpin replan shrink Zonos toward the cliff.
        // Fused Q1_0 matmul kernels are only wired on the all-thunks path; the
        // MPSGraph/hybrid segmenter has no Q1_0 lowering and silently produces
        // garbage. A ≥4 GiB arena normally forces all-thunks, but dropping the
        // (unused) Q1_0 dequant scratch can shrink the arena below the cliff —
        // so pin all-thunks whenever the graph fuses Q1_0.
        let has_direct_fused = self.schedule.thunks.iter().any(|t| match t {
            Thunk::DequantMatMulGguf {
                scheme: rlx_ir::QuantScheme::GgufQ1_0,
                ..
            } => !rlx_ir::env::flag("RLX_METAL_Q1_0_FUSED_DISABLE"),
            Thunk::DequantMatMulGguf {
                scheme: rlx_ir::QuantScheme::GgufQ2_0,
                ..
            } => !rlx_ir::env::flag("RLX_METAL_Q2_0_FUSED_DISABLE"),
            _ => false,
        });
        let big_arena = (self.arena.buffer.length() >= (1u64 << 32) || has_direct_fused)
            && !rlx_ir::env::flag("RLX_METAL_MPSGRAPH_BIG_ARENA");
        let hybrid_on_big = matches!(
            rlx_ir::env::var("RLX_METAL_HYBRID_BIG_ARENA").as_deref(),
            Some("1") | Some("true") | Some("yes")
        );

        // First-run freeze: bake params as MPSGraph constants (opt-in).
        // Do NOT auto-freeze on big-arena hybrid — duplicating large FC
        // weights into MPSGraphExecutable is slower than arena feeds + staging.
        if !self.mps_params_frozen
            && (self.mps_plan.is_some() || self.mps_hybrid.is_some())
            && rlx_ir::env::flag("RLX_MPSGRAPH_PARAM_CONST")
        {
            self.freeze_params_to_mps_constants();
            self.mps_params_frozen = true;
        }

        // Recompute after freeze (may replace hybrid ↔ full plan).
        let hybrid_ok = self
            .mps_hybrid
            .as_ref()
            .is_some_and(|steps| crate::mps_graph_hybrid::hybrid_has_mps(steps));

        if big_arena && !(hybrid_ok && hybrid_on_big) {
            let t0 = Instant::now();
            let _ = self.encode_commit(true, None, None);
            crate::mps_profile::record("encode_path:thunks_only_big_arena", t0.elapsed());
            return;
        }

        // Active-extent (PLAN L1): when set + every thunk safe, bypass
        // MPSGraph + ICB (both pre-encode at full extent) and dispatch
        // per-op with scaled launch dims via encode_commit.
        let active_safe = self.active_extent.is_some() && self.all_safe_for_active();
        // Full-graph MPSGraph: skip on big arenas (hybrid handled below).
        if !active_safe && self.mps_plan.is_some() && !big_arena {
            // Adaptive dispatch: with RmsNorm + SDPA wired into the
            // bridge, MPSGraph's fused kernels beat per-op encoding
            // across the full qwen3 prefill range. The remaining
            // per-call ObjC overhead only matters for trivial
            // single-matmul graphs (~<1 MFLOP). Default-on whenever
            // the plan exists; override via RLX_MPSGRAPH_MIN_FLOPS or
            // RLX_MPSGRAPH_FORCE=1.
            let force = rlx_ir::env::flag("RLX_MPSGRAPH_FORCE");
            let threshold = rlx_ir::env::var("RLX_MPSGRAPH_MIN_FLOPS")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1_000_000);
            // BF16-weight matmul is only correct on MPS (the thunk Sgemm has no
            // bf16 kernel); force it there even below the FLOP threshold.
            if force || self.has_bf16_matmul || self.estimated_max_flops() >= threshold {
                let t0 = Instant::now();
                self.run_via_mps_graph();
                crate::mps_profile::record("encode_path:mps_graph_full", t0.elapsed());
                return;
            }
        }
        if !active_safe && (self.mps_plan.is_none() || big_arena) && hybrid_ok {
            let force = rlx_ir::env::flag("RLX_MPSGRAPH_FORCE");
            let threshold = rlx_ir::env::var("RLX_MPSGRAPH_MIN_FLOPS")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1_000_000);
            // See full-plan gate above: bf16-weight matmul must go through MPS.
            if force || self.has_bf16_matmul || self.estimated_max_flops() >= threshold {
                let t0 = Instant::now();
                self.run_via_mps_hybrid();
                crate::mps_profile::record("encode_path:mps_hybrid", t0.elapsed());
                return;
            }
        }
        // wait=true: synchronous, drop the buffer immediately after wait.
        // ICB segments (if any) are dispatched inline by encode_commit.
        let t0 = Instant::now();
        let _ = self.encode_commit(true, None, None);
        crate::mps_profile::record("encode_path:thunks_only", t0.elapsed());
    }

    /// Encode + commit. When `wait=true`, also waits for completion and
    /// returns None. When `wait=false`, returns the command buffer so the
    /// caller can defer the wait (pipelining N commits + one sync at the
    /// end — see `commit_no_wait`/`sync_pending`/`run_pipelined`).
    ///
    /// `blit_outputs`: if `Some`, after compute encoding ends, opens a blit
    /// encoder and copies each `output_slots[i]` arena region into
    /// `blit_outputs[i]`. Used by `run_pipelined` so each in-flight commit
    /// has its own output snapshot — without this, subsequent commits
    /// stomp the arena's output region before the caller can read it.
    ///
    /// Single function rather than separate encode/commit helpers because
    /// returning a `CommandBuffer` whose internal encoder borrow has just
    /// ended trips an obscure debug-mode use-after-free in metal-rs's
    /// reference-counting wrappers; keeping commit inline avoids it.
    /// MPSGraph and ICB fast paths are not routed through here.
    pub(crate) fn encode_commit(
        &mut self,
        wait: bool,
        blit_outputs: Option<&[metal::Buffer]>,
        thunk_range: Option<std::ops::Range<usize>>,
    ) -> Option<metal::CommandBuffer> {
        /// Host-side thunk queued between GPU segments (unified-memory arena).
        enum DeferredHostOp {
            GatedDeltaNet {
                q: usize,
                k_off: usize,
                v: usize,
                g: usize,
                beta: usize,
                state_byte: usize,
                dst: usize,
                batch: u32,
                seq: u32,
                heads: u32,
                state_size: u32,
                f16: bool,
                gate_per_channel: bool,
                carry_state: bool,
            },
            SelectiveScan {
                x: usize,
                delta: usize,
                a: usize,
                b: usize,
                c: usize,
                dst: usize,
                batch: u32,
                seq: u32,
                hidden: u32,
                state_size: u32,
            },
            Sample {
                logits: usize,
                dst: usize,
                batch: u32,
                vocab: u32,
                top_k: u32,
                top_p: f32,
                temperature: f32,
                seed: u64,
            },
            Reverse {
                src: usize,
                dst: usize,
                dims: Vec<u32>,
                rev_mask: Vec<bool>,
                elem_bytes: u8,
            },
            Pad {
                src: usize,
                dst: usize,
                in_dims: Vec<u32>,
                before: Vec<u32>,
                after: Vec<u32>,
                mode: rlx_ir::PadMode,
                fill: Vec<u8>,
                elem_bytes: u8,
            },
            Slice {
                src: usize,
                dst: usize,
                in_dims: Vec<u32>,
                axis: u32,
                start: u32,
                len: u32,
                step: i64,
                elem_bytes: u8,
            },
            ArgReduce {
                src: usize,
                dst: usize,
                outer: u32,
                reduced: u32,
                inner: u32,
                is_max: bool,
            },
            Lstm {
                x: usize,
                w_ih: usize,
                w_hh: usize,
                bias: usize,
                h0: usize,
                c0: usize,
                dst: usize,
                batch: u32,
                seq: u32,
                input_size: u32,
                hidden: u32,
                num_layers: u32,
                bidirectional: bool,
                carry: bool,
            },
            Gru {
                x: usize,
                w_ih: usize,
                w_hh: usize,
                b_ih: usize,
                b_hh: usize,
                h0: usize,
                dst: usize,
                batch: u32,
                seq: u32,
                input_size: u32,
                hidden: u32,
                num_layers: u32,
                bidirectional: bool,
                carry: bool,
            },
            Rnn {
                x: usize,
                w_ih: usize,
                w_hh: usize,
                bias: usize,
                h0: usize,
                dst: usize,
                batch: u32,
                seq: u32,
                input_size: u32,
                hidden: u32,
                num_layers: u32,
                bidirectional: bool,
                carry: bool,
                relu: bool,
            },
            Mamba2 {
                x: usize,
                dt: usize,
                a: usize,
                b: usize,
                c: usize,
                dst: usize,
                batch: u32,
                seq: u32,
                heads: u32,
                head_dim: u32,
                state_size: u32,
            },
            DequantMatMulGguf {
                x: usize,
                w_q: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                scheme: rlx_ir::quant::QuantScheme,
            },
            DequantGroupedMatMulGguf {
                input: usize,
                w_q: usize,
                expert_idx: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                num_experts: usize,
                scheme: rlx_ir::quant::QuantScheme,
            },
            DequantGroupedMatMulMlx {
                input: usize,
                w_q: usize,
                scale: usize,
                zp: usize,
                expert_idx: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                num_experts: usize,
                slab_bytes: usize,
                scheme: rlx_ir::quant::QuantScheme,
                scale_bf16: bool,
            },
            DequantMatMulInt8 {
                x: usize,
                w_q: usize,
                scale: usize,
                zp: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                block_size: u32,
                is_asymmetric: bool,
            },
            DequantMatMulInt4 {
                x: usize,
                w_q: usize,
                scale: usize,
                zp: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                block_size: u32,
                is_asymmetric: bool,
            },
            DequantMatMulFp8 {
                x: usize,
                w_q: usize,
                scale: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                e5m2: bool,
            },
            DequantMatMulNvfp4 {
                x: usize,
                w_q: usize,
                scale: usize,
                global_scale: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
            },
            DequantMatMulMxFp4x2 {
                x: usize,
                w_q: usize,
                scale: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                group: usize,
            },
            DequantMatMulMlx {
                x: usize,
                w_q: usize,
                scale: usize,
                zp: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                scheme: rlx_ir::quant::QuantScheme,
            },
            ScaledMatMul {
                lhs: usize,
                rhs: usize,
                lhs_scale: usize,
                rhs_scale: usize,
                bias: usize,
                dst: usize,
                m: usize,
                k: usize,
                n: usize,
                has_bias: bool,
                lhs_fmt: rlx_ir::ScaledFormat,
                rhs_fmt: rlx_ir::ScaledFormat,
                layout: rlx_ir::ScaleLayout,
            },
            ScaledQuantize {
                x: usize,
                scale: usize,
                dst: usize,
                rows: usize,
                cols: usize,
                fmt: rlx_ir::ScaledFormat,
                layout: rlx_ir::ScaleLayout,
            },
            ScaledDequantize {
                codes: usize,
                scale: usize,
                dst: usize,
                rows: usize,
                cols: usize,
                fmt: rlx_ir::ScaledFormat,
                layout: rlx_ir::ScaleLayout,
            },
            ScaledQuantScale {
                x: usize,
                dst: usize,
                rows: usize,
                cols: usize,
                fmt: rlx_ir::ScaledFormat,
                layout: rlx_ir::ScaleLayout,
            },
            /// Unified-memory memcpy — batched to avoid GPU dispatch on slices.
            Memcpy {
                src: usize,
                dst: usize,
                bytes: usize,
            },
            /// CPU activation on unified-memory arena (large-offset fallback).
            ActivationHost {
                data: usize,
                len: u32,
                act: rlx_ir::op::Activation,
            },
            /// CPU gelu_approx(src→dst) on unified memory (>4 GiB fused copy+act).
            GeluApproxHost { src: usize, dst: usize, len: u32 },
            /// CPU elementwise binary on unified-memory arena (large-offset fallback).
            BinaryHost {
                lhs: usize,
                rhs: usize,
                dst: usize,
                len: u32,
                op: rlx_ir::op::BinaryOp,
            },
            ConcatLastax {
                dst: usize,
                outer: u32,
                dst_axis: u32,
                segments: Vec<(usize, u32)>,
            },
            ConcatMidAxis {
                dst: usize,
                outer: u32,
                dst_axis: u32,
                inner: u32,
                segments: Vec<(usize, u32)>,
            },
            /// Row-major [rows, cols] → [cols, rows] on unified memory.
            Transpose2d {
                src: usize,
                dst: usize,
                rows: u32,
                cols: u32,
            },
            /// Narrow on the last axis (unified memory host copy).
            NarrowLastAxis {
                src: usize,
                dst: usize,
                outer: u32,
                src_axis: u32,
                start: u32,
                len: u32,
            },
        }
        /// Host ops deferred until after the final GPU wait (one sync, no extra cmd_buf).
        enum TailHostOp {
            WelchPeaks {
                spec: usize,
                dst: usize,
                welch_batch: u32,
                n_fft: u32,
                n_segments: u32,
                k: u32,
            },
        }

        let trace = rlx_ir::env::flag("RLX_METAL_TRACE");
        let t_run_start = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let dev = metal_device().expect("Metal device required");
        let mut cmd_buf = dev.queue.new_command_buffer().to_owned();
        let k = kernels();

        // Lazy compute encoder — created on first MSL thunk, ended right
        // before any MPS call. Two consecutive MPS calls don't pay an
        // encoder create/end cost. Apple's per-encoder cost (~10–50µs) used
        // to dominate small-batch text — eager creation made every MPS↔MSL
        // boundary cost a fresh encoder pair.
        //
        // **Owned, not borrowed.** `enc` was previously
        // `Option<&ComputeCommandEncoderRef>` borrowing from `cmd_buf`,
        // which fixed `cmd_buf`'s lifetime to the whole function and
        // blocked mid-function `cmd_buf` swaps for `Op::Custom` sync
        // points. Holding the owned `ComputeCommandEncoder` (a refcount
        // bump on `to_owned()`) decouples the lifetime: `enc.take()`
        // releases the encoder fully, after which `cmd_buf` is freely
        // reassignable.
        let mut enc: Option<metal::ComputeCommandEncoder> = None;
        let mut deferred_host: Vec<DeferredHostOp> = Vec::new();
        let mut tail_host: Vec<TailHostOp> = Vec::new();
        let mut narrow_batch: Option<PendingNarrowBatch> = None;

        // Concurrent dispatch (RLX_METAL_CONCURRENT): open compute encoders in
        // `Concurrent` dispatch mode so independent decode dispatches (q/k/v,
        // gate/up …) overlap on the GPU, and insert a `memoryBarrier` only
        // before a dispatch that data-depends on the current wave. The hazard
        // set is precomputed below from `mlp_io`. Off ⇒ classic Serial encoder.
        let concurrent = rlx_ir::env::flag("RLX_METAL_CONCURRENT");
        // Diagnostics (RLX_METAL_CONCURRENT_STATS): how fragmented is the
        // encoder stream? If `enc_opened` ≈ thunks, ops each get their own
        // encoder and Concurrent can't overlap anything; if `enc_opened` ≪
        // thunks, dispatches share encoders and overlap is actually in play.
        let concurrent_stats = rlx_ir::env::flag("RLX_METAL_CONCURRENT_STATS");
        // Diagnostic-only: run concurrent with NO hazard barriers. Output is
        // INCORRECT (races), but it measures the wall-clock ceiling of full
        // dispatch overlap — if this is no faster than the barriered/serial
        // path, overlap fundamentally can't help this workload.
        let concurrent_no_barrier = rlx_ir::env::flag("RLX_METAL_CONCURRENT_NOBARRIER");
        #[allow(unused_assignments)]
        let mut enc_opened = 0usize;
        #[allow(unused_assignments)]
        let mut barriers_emitted = 0usize;
        #[allow(unused_assignments)]
        let mut thunks_dispatched = 0usize;

        let flush_deferred_host =
            |cmd_buf: &mut metal::CommandBuffer,
             enc: &mut Option<metal::ComputeCommandEncoder>,
             deferred: &mut Vec<DeferredHostOp>| {
                if deferred.is_empty() {
                    return;
                }
                if let Some(active) = enc.take() {
                    active.end_encoding();
                }
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
                let arena_ptr = self.arena.buffer.contents() as *mut u8;
                for op in deferred.drain(..) {
                    match op {
                        DeferredHostOp::GatedDeltaNet {
                            q,
                            k_off,
                            v,
                            g,
                            beta,
                            state_byte,
                            dst,
                            batch,
                            seq,
                            heads,
                            state_size,
                            f16,
                            gate_per_channel,
                            carry_state,
                        } => unsafe {
                            if f16 {
                                rlx_cpu::thunk::execute_gated_delta_net_f16(
                                    q,
                                    k_off,
                                    v,
                                    g,
                                    beta,
                                    state_byte,
                                    dst,
                                    batch as usize,
                                    seq as usize,
                                    heads as usize,
                                    state_size as usize,
                                    arena_ptr,
                                );
                            } else {
                                rlx_cpu::thunk::execute_gated_delta_net_f32(
                                    q,
                                    k_off,
                                    v,
                                    g,
                                    beta,
                                    state_byte,
                                    dst,
                                    batch as usize,
                                    seq as usize,
                                    heads as usize,
                                    state_size as usize,
                                    gate_per_channel,
                                    carry_state,
                                    arena_ptr,
                                );
                            }
                        },
                        DeferredHostOp::SelectiveScan {
                            x,
                            delta,
                            a,
                            b,
                            c,
                            dst,
                            batch,
                            seq,
                            hidden,
                            state_size,
                        } => unsafe {
                            rlx_cpu::thunk::execute_selective_scan_f32(
                                x,
                                delta,
                                a,
                                b,
                                c,
                                dst,
                                batch as usize,
                                seq as usize,
                                hidden as usize,
                                state_size as usize,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Sample {
                            logits,
                            dst,
                            batch,
                            vocab,
                            top_k,
                            top_p,
                            temperature,
                            seed,
                        } => unsafe {
                            rlx_cpu::thunk::execute_sample_f32(
                                logits,
                                dst,
                                batch as usize,
                                vocab as usize,
                                top_k as usize,
                                top_p,
                                temperature,
                                seed,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Reverse {
                            src,
                            dst,
                            dims,
                            rev_mask,
                            elem_bytes,
                        } => unsafe {
                            rlx_cpu::thunk::execute_reverse(
                                src,
                                dst,
                                &dims,
                                &rev_mask,
                                elem_bytes as usize,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Pad {
                            src,
                            dst,
                            in_dims,
                            before,
                            after,
                            mode,
                            fill,
                            elem_bytes,
                        } => unsafe {
                            rlx_cpu::thunk::execute_pad(
                                src,
                                dst,
                                &in_dims,
                                &before,
                                &after,
                                mode,
                                &fill,
                                elem_bytes as usize,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Slice {
                            src,
                            dst,
                            in_dims,
                            axis,
                            start,
                            len,
                            step,
                            elem_bytes,
                        } => unsafe {
                            rlx_cpu::thunk::execute_slice(
                                src,
                                dst,
                                &in_dims,
                                axis as usize,
                                start as usize,
                                len as usize,
                                step,
                                elem_bytes as usize,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::ArgReduce {
                            src,
                            dst,
                            outer,
                            reduced,
                            inner,
                            is_max,
                        } => unsafe {
                            rlx_cpu::thunk::execute_argreduce_f32(
                                src,
                                dst,
                                outer as usize,
                                reduced as usize,
                                inner as usize,
                                is_max,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Lstm {
                            x,
                            w_ih,
                            w_hh,
                            bias,
                            h0,
                            c0,
                            dst,
                            batch,
                            seq,
                            input_size,
                            hidden,
                            num_layers,
                            bidirectional,
                            carry,
                        } => unsafe {
                            rlx_cpu::thunk::execute_lstm_f32(
                                x,
                                w_ih,
                                w_hh,
                                bias,
                                h0,
                                c0,
                                dst,
                                batch as usize,
                                seq as usize,
                                input_size as usize,
                                hidden as usize,
                                num_layers as usize,
                                bidirectional,
                                carry,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Gru {
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
                        } => unsafe {
                            rlx_cpu::thunk::execute_gru_f32(
                                x,
                                w_ih,
                                w_hh,
                                b_ih,
                                b_hh,
                                h0,
                                dst,
                                batch as usize,
                                seq as usize,
                                input_size as usize,
                                hidden as usize,
                                num_layers as usize,
                                bidirectional,
                                carry,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Rnn {
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
                        } => unsafe {
                            rlx_cpu::thunk::execute_rnn_f32(
                                x,
                                w_ih,
                                w_hh,
                                bias,
                                h0,
                                dst,
                                batch as usize,
                                seq as usize,
                                input_size as usize,
                                hidden as usize,
                                num_layers as usize,
                                bidirectional,
                                carry,
                                relu,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::Mamba2 {
                            x,
                            dt,
                            a,
                            b,
                            c,
                            dst,
                            batch,
                            seq,
                            heads,
                            head_dim,
                            state_size,
                        } => unsafe {
                            rlx_cpu::thunk::execute_mamba2_f32(
                                x,
                                dt,
                                a,
                                b,
                                c,
                                dst,
                                batch as usize,
                                seq as usize,
                                heads as usize,
                                head_dim as usize,
                                state_size as usize,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantMatMulGguf {
                            x,
                            w_q,
                            dst,
                            m,
                            k,
                            n,
                            scheme,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_matmul_gguf_f32(
                                x, w_q, dst, m, k, n, scheme, arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantGroupedMatMulGguf {
                            input,
                            w_q,
                            expert_idx,
                            dst,
                            m,
                            k,
                            n,
                            num_experts,
                            scheme,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_grouped_matmul_gguf_f32(
                                input,
                                w_q,
                                expert_idx,
                                dst,
                                m,
                                k,
                                n,
                                num_experts,
                                scheme,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantGroupedMatMulMlx {
                            input,
                            w_q,
                            scale,
                            zp,
                            expert_idx,
                            dst,
                            m,
                            k,
                            n,
                            num_experts,
                            slab_bytes,
                            scheme,
                            scale_bf16,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_grouped_matmul_mlx_f32(
                                input,
                                w_q,
                                scale,
                                zp,
                                expert_idx,
                                dst,
                                m,
                                k,
                                n,
                                num_experts,
                                slab_bytes,
                                scheme,
                                scale_bf16,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantMatMulInt8 {
                            x,
                            w_q,
                            scale,
                            zp,
                            dst,
                            m,
                            k,
                            n,
                            block_size,
                            is_asymmetric,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_matmul_int8_f32(
                                x,
                                w_q,
                                scale,
                                zp,
                                dst,
                                m,
                                k,
                                n,
                                block_size,
                                is_asymmetric,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantMatMulInt4 {
                            x,
                            w_q,
                            scale,
                            zp,
                            dst,
                            m,
                            k,
                            n,
                            block_size,
                            is_asymmetric,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_matmul_int4_f32(
                                x,
                                w_q,
                                scale,
                                zp,
                                dst,
                                m,
                                k,
                                n,
                                block_size,
                                is_asymmetric,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantMatMulFp8 {
                            x,
                            w_q,
                            scale,
                            dst,
                            m,
                            k,
                            n,
                            e5m2,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_matmul_fp8_f32(
                                x, w_q, scale, dst, m, k, n, e5m2, arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantMatMulNvfp4 {
                            x,
                            w_q,
                            scale,
                            global_scale,
                            dst,
                            m,
                            k,
                            n,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_matmul_nvfp4_f32(
                                x,
                                w_q,
                                scale,
                                global_scale,
                                dst,
                                m,
                                k,
                                n,
                                arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantMatMulMxFp4x2 {
                            x,
                            w_q,
                            scale,
                            dst,
                            m,
                            k,
                            n,
                            group,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_matmul_mxfp4x2_f32(
                                x, w_q, scale, dst, m, k, n, group, arena_ptr,
                            );
                        },
                        DeferredHostOp::DequantMatMulMlx {
                            x,
                            w_q,
                            scale,
                            zp,
                            dst,
                            m,
                            k,
                            n,
                            scheme,
                        } => unsafe {
                            rlx_cpu::thunk::execute_dequant_matmul_mlx_f32(
                                x, w_q, scale, zp, dst, m, k, n, scheme, arena_ptr,
                            );
                        },
                        DeferredHostOp::ScaledMatMul {
                            lhs,
                            rhs,
                            lhs_scale,
                            rhs_scale,
                            bias,
                            dst,
                            m,
                            k,
                            n,
                            has_bias,
                            lhs_fmt,
                            rhs_fmt,
                            layout,
                        } => unsafe {
                            rlx_cpu::thunk::execute_scaled_matmul_f32(
                                lhs, rhs, lhs_scale, rhs_scale, bias, dst, m, k, n, has_bias,
                                lhs_fmt, rhs_fmt, layout, arena_ptr,
                            );
                        },
                        DeferredHostOp::ScaledQuantize {
                            x,
                            scale,
                            dst,
                            rows,
                            cols,
                            fmt,
                            layout,
                        } => unsafe {
                            rlx_cpu::thunk::execute_scaled_quantize_f32(
                                x, scale, dst, rows, cols, fmt, layout, arena_ptr,
                            );
                        },
                        DeferredHostOp::ScaledDequantize {
                            codes,
                            scale,
                            dst,
                            rows,
                            cols,
                            fmt,
                            layout,
                        } => unsafe {
                            rlx_cpu::thunk::execute_scaled_dequantize_f32(
                                codes, scale, dst, rows, cols, fmt, layout, arena_ptr,
                            );
                        },
                        DeferredHostOp::ScaledQuantScale {
                            x,
                            dst,
                            rows,
                            cols,
                            fmt,
                            layout,
                        } => unsafe {
                            rlx_cpu::thunk::execute_scaled_quant_scale_f32(
                                x, dst, rows, cols, fmt, layout, arena_ptr,
                            );
                        },
                        DeferredHostOp::Memcpy { src, dst, bytes } => unsafe {
                            if bytes > 0 {
                                std::ptr::copy_nonoverlapping(
                                    arena_ptr.add(src),
                                    arena_ptr.add(dst),
                                    bytes,
                                );
                            }
                        },
                        DeferredHostOp::ActivationHost { data, len, act } => unsafe {
                            let len = len as usize;
                            let d = std::slice::from_raw_parts_mut(
                                arena_ptr.add(data) as *mut f32,
                                len,
                            );
                            use rlx_ir::op::Activation;
                            match act {
                                Activation::Gelu => rlx_cpu::kernels::par_gelu_inplace(d),
                                Activation::GeluApprox => {
                                    rlx_cpu::kernels::par_gelu_approx_inplace(d)
                                }
                                Activation::Silu => rlx_cpu::kernels::par_silu_inplace(d),
                                _ => {}
                            }
                        },
                        DeferredHostOp::GeluApproxHost { src, dst, len } => unsafe {
                            let len = len as usize;
                            let s =
                                std::slice::from_raw_parts(arena_ptr.add(src) as *const f32, len);
                            let d =
                                std::slice::from_raw_parts_mut(arena_ptr.add(dst) as *mut f32, len);
                            rlx_cpu::kernels::par_gelu_approx_out(s, d);
                        },
                        DeferredHostOp::BinaryHost {
                            lhs,
                            rhs,
                            dst,
                            len,
                            op,
                        } => unsafe {
                            let len = len as usize;
                            let a =
                                std::slice::from_raw_parts(arena_ptr.add(lhs) as *const f32, len);
                            let b =
                                std::slice::from_raw_parts(arena_ptr.add(rhs) as *const f32, len);
                            let c =
                                std::slice::from_raw_parts_mut(arena_ptr.add(dst) as *mut f32, len);
                            use rlx_ir::op::BinaryOp;
                            for i in 0..len {
                                c[i] = match op {
                                    BinaryOp::Add => a[i] + b[i],
                                    BinaryOp::Mul => a[i] * b[i],
                                    BinaryOp::Sub => a[i] - b[i],
                                    BinaryOp::Div => a[i] / b[i],
                                    _ => a[i],
                                };
                            }
                        },
                        DeferredHostOp::ConcatLastax {
                            dst,
                            outer,
                            dst_axis,
                            segments,
                        } => unsafe {
                            let dst_base = arena_ptr.add(dst);
                            let dst_stride = dst_axis as usize * std::mem::size_of::<f32>();
                            for o in 0..outer as usize {
                                let mut col = 0usize;
                                for &(src_off, src_axis) in &segments {
                                    let src_axis = src_axis as usize;
                                    let row_bytes = src_axis * std::mem::size_of::<f32>();
                                    let src_row = arena_ptr.add(src_off).add(o * row_bytes);
                                    let dst_row = dst_base
                                        .add(o * dst_stride + col * std::mem::size_of::<f32>());
                                    std::ptr::copy_nonoverlapping(src_row, dst_row, row_bytes);
                                    col += src_axis;
                                }
                            }
                        },
                        DeferredHostOp::ConcatMidAxis {
                            dst,
                            outer,
                            dst_axis,
                            inner,
                            segments,
                        } => unsafe {
                            let inner_b = inner as usize * std::mem::size_of::<f32>();
                            let dst_base = arena_ptr.add(dst);
                            let dst_stride = dst_axis as usize * inner_b;
                            for o in 0..outer as usize {
                                let mut axis_off = 0usize;
                                for &(src_off, src_axis) in &segments {
                                    let src_per_outer = src_axis as usize * inner_b;
                                    let src_row = arena_ptr.add(src_off).add(o * src_per_outer);
                                    let dst_row = dst_base.add(o * dst_stride + axis_off * inner_b);
                                    std::ptr::copy_nonoverlapping(src_row, dst_row, src_per_outer);
                                    axis_off += src_axis as usize;
                                }
                            }
                        },
                        DeferredHostOp::Transpose2d {
                            src,
                            dst,
                            rows,
                            cols,
                        } => unsafe {
                            let src_base = arena_ptr.add(src) as *const f32;
                            let dst_base = arena_ptr.add(dst) as *mut f32;
                            let rows = rows as usize;
                            let cols = cols as usize;
                            for r in 0..rows {
                                for c in 0..cols {
                                    *dst_base.add(c * rows + r) = *src_base.add(r * cols + c);
                                }
                            }
                        },
                        DeferredHostOp::NarrowLastAxis {
                            src,
                            dst,
                            outer,
                            src_axis,
                            start,
                            len,
                        } => unsafe {
                            let row_bytes = len as usize * std::mem::size_of::<f32>();
                            let src_stride = src_axis as usize * std::mem::size_of::<f32>();
                            let src_start = start as usize * std::mem::size_of::<f32>();
                            for o in 0..outer as usize {
                                std::ptr::copy_nonoverlapping(
                                    arena_ptr.add(src + o * src_stride + src_start),
                                    arena_ptr.add(dst + o * row_bytes),
                                    row_bytes,
                                );
                            }
                        },
                    }
                }
                *cmd_buf = dev.queue.new_command_buffer().to_owned();
            };

        macro_rules! e {
            () => {{
                flush_deferred_host(&mut cmd_buf, &mut enc, &mut deferred_host);
                if enc.is_none() {
                    let dispatch_ty = if concurrent {
                        metal::MTLDispatchType::Concurrent
                    } else {
                        metal::MTLDispatchType::Serial
                    };
                    enc = Some(
                        cmd_buf
                            .compute_command_encoder_with_dispatch_type(dispatch_ty)
                            .to_owned(),
                    );
                    enc_opened += 1;
                    let _ = enc_opened;
                }
                enc.as_deref().unwrap()
            }};
        }
        macro_rules! end_msl {
            () => {{
                if narrow_batch.is_some() {
                    flush_pending_narrow_batch(e!(), k, &self.arena.buffer, &mut narrow_batch);
                }
                flush_deferred_host(&mut cmd_buf, &mut enc, &mut deferred_host);
                if let Some(active) = enc.take() {
                    active.end_encoding();
                }
            }};
        }
        /// Commit + wait so MPS matmul / compute producers finish before a
        /// compute consumer reads the same arena slot (Gemma E2B gate→gelu).
        macro_rules! sync_gpu {
            () => {{
                flush_deferred_host(&mut cmd_buf, &mut enc, &mut deferred_host);
                if let Some(active) = enc.take() {
                    active.end_encoding();
                }
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
                check_cmd_buf_status(&cmd_buf, "sync_gpu");
                cmd_buf = dev.queue.new_command_buffer().to_owned();
            }};
        }

        // Active-extent (PLAN L1): if hint is set + every thunk is in
        // the safe set, scale launch dims per-op. ICB segments (pre-
        // encoded at full extent) are bypassed in this mode — fall
        // through to per-op encoding instead.
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

        // Indexed thunk loop: when an ICB segment covers the next range
        // of thunks, dispatch it via executeCommandsInBuffer in one shot
        // and skip past those indices instead of encoding them per-op.
        let segments = &self.icb_segments;
        let thunks = &self.schedule.thunks;
        if rlx_ir::env::flag("RLX_METAL_DUMP_BYTES") {
            use std::sync::atomic::{AtomicBool, Ordering};
            static DONE_B: AtomicBool = AtomicBool::new(false);
            // Fire on a DECODE step: many m==1 sgemms (prefill has only the
            // last-token lm_head at m==1, decode has all projections).
            let m1 = thunks
                .iter()
                .filter(|t| matches!(t, crate::thunk::Thunk::Sgemm { m: 1, .. }))
                .count();
            if m1 > 10 && !DONE_B.swap(true, Ordering::Relaxed) {
                crate::thunk::dump_thunk_bytes(thunks);
            }
        }
        if rlx_ir::env::flag("RLX_DUMP_SCHED") {
            use std::sync::atomic::{AtomicBool, Ordering};
            static DONE: AtomicBool = AtomicBool::new(false);
            if !DONE.swap(true, Ordering::Relaxed) {
                for (i, t) in thunks.iter().enumerate() {
                    let nm = crate::thunk::thunk_name(t);
                    if nm != "nop" {
                        eprintln!("[sched {i}] {nm}");
                    }
                }
            }
        }
        // Same dump but gated to a DECODE step (>10 m==1 sgemms), so it prints
        // the hot m=1 schedule rather than the one-shot prefill schedule.
        if rlx_ir::env::flag("RLX_DUMP_SCHED_DECODE") {
            use std::sync::atomic::{AtomicBool, Ordering};
            static DONE_D: AtomicBool = AtomicBool::new(false);
            let m1 = thunks
                .iter()
                .filter(|t| matches!(t, crate::thunk::Thunk::Sgemm { m: 1, .. }))
                .count();
            if m1 > 10 && !DONE_D.swap(true, Ordering::Relaxed) {
                eprintln!("[sched-decode] {} thunks, {m1} m=1 sgemms", thunks.len());
                for (i, t) in thunks.iter().enumerate() {
                    let nm = crate::thunk::thunk_name(t);
                    if nm != "nop" {
                        eprintln!("[dsched {i}] {nm}");
                    }
                }
            }
        }
        let mut seg_iter = segments.iter().peekable();
        let loop_end = thunk_range.as_ref().map(|r| r.end).unwrap_or(thunks.len());
        let mut i = thunk_range.as_ref().map(|r| r.start).unwrap_or(0);
        // Encode-time skip set for shared-x dual Q1 GEMV partners.
        let mut skip_thunks: HashSet<usize> = HashSet::new();
        let dual_q1_log = rlx_ir::env::flag("RLX_METAL_FUSE_DECODE_LOG");
        let mut dual_q1_fused = 0usize;
        // Concurrent-dispatch hazard set: thunk indices that must be preceded by
        // a memoryBarrier (RAW/WAR/WAW vs the current wave). Empty in Serial
        // mode, so the per-op check below is a cheap HashSet miss.
        let barrier_set = if concurrent {
            let start = thunk_range.as_ref().map(|r| r.start).unwrap_or(0);
            crate::thunk::concurrent_barrier_set(thunks, start, loop_end)
        } else {
            HashSet::new()
        };
        while i < loop_end {
            if skip_thunks.contains(&i) {
                i += 1;
                continue;
            }
            if thunk_range.is_none()
                && active.is_none()
                && !concurrent
                && let Some(range) = seg_iter.peek()
                && range.start == i
            {
                range.segment.execute_on(e!(), &self.arena.buffer);
                i = range.end;
                seg_iter.next();
                continue;
            }
            let idx = i;
            let thunk = thunks[i].clone();
            i += 1;
            if !matches!(thunk, Thunk::Narrow { .. } | Thunk::SplitLastAxis { .. })
                && narrow_batch.is_some()
            {
                flush_pending_narrow_batch(e!(), k, &self.arena.buffer, &mut narrow_batch);
            }
            // Concurrent dispatch: order this op after the current wave iff it
            // data-depends on it. Only meaningful while an encoder is open —
            // end_msl!/sync_gpu! start a fresh encoder, which Metal already
            // orders after the previous one (a stronger barrier), so a redundant
            // memoryBarrier is skipped here when `enc` is None.
            if concurrent && !concurrent_no_barrier && barrier_set.contains(&idx) {
                if let Some(active_enc) = enc.as_deref() {
                    // memoryBarrierWithScope: MTLBarrierScopeBuffers (=1) — covers
                    // the arena (the only writable buffer; weights are read-only).
                    // metal-rs 0.30 exposes no scope-barrier wrapper, so message
                    // it directly (same objc path as the GPU-timestamp probe).
                    unsafe {
                        use objc::{msg_send, runtime::Object, sel, sel_impl};
                        let obj =
                            active_enc as *const metal::ComputeCommandEncoderRef as *mut Object;
                        let _: () = msg_send![obj, memoryBarrierWithScope: 1u64];
                    }
                    barriers_emitted += 1;
                }
            }
            if concurrent_stats {
                thunks_dispatched += 1;
            }
            // PLAN L3: per-thunk Perfetto span. No-op when env var
            // RLX_TRACE_PERFETTO unset.
            let _span = rlx_ir::perfetto::TraceSpan::new(crate::thunk::thunk_name(&thunk), "metal");
            match &thunk {
                Thunk::Nop => {}
                Thunk::Cast {
                    src,
                    dst,
                    len,
                    src_dt,
                    dst_dt,
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_cast(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        len,
                        *src_dt,
                        *dst_dt,
                    );
                }
                Thunk::CastHost {
                    src,
                    dst,
                    len,
                    src_dt,
                    dst_dt,
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    // Integer / bool / mixed-width / exotic (F16/BF16/F64/C64)
                    // cast: flush GPU, convert on host against the
                    // unified-memory arena (same sync pattern as CustomOp /
                    // SpdHost). Reuse rlx-cpu's generic scalar-cast kernel —
                    // it handles ALL 12 dtypes with correct numeric semantics
                    // (float→int saturates, int→int wraps, f16/bf16
                    // round-nearest, C64 real↔complex), so no pair panics.
                    // The arena is a flat byte buffer whose per-node slots are
                    // sized by real dtype width, so exotic dtypes (which lack
                    // Metal *device* storage) are still host-representable and
                    // convert correctly here.
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    rlx_cpu::thunk::exec_cast_generic(
                        *src,
                        *dst,
                        len as usize,
                        *src_dt,
                        *dst_dt,
                        arena_ptr,
                    );
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }
                Thunk::CastTruncF32 { src, dst, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    let s = unsafe {
                        std::slice::from_raw_parts(arena_ptr.add(*src) as *const f32, len as usize)
                    };
                    let d = unsafe {
                        std::slice::from_raw_parts_mut(
                            arena_ptr.add(*dst) as *mut f32,
                            len as usize,
                        )
                    };
                    for i in 0..len as usize {
                        d[i] = s[i].trunc();
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }
                Thunk::Sgemm {
                    a,
                    b,
                    c,
                    m,
                    k: kk,
                    n,
                    dt,
                    b_f16,
                    a_f16,
                    ta,
                    tb,
                } => {
                    use crate::thunk::HalfFlag;
                    let (a, b, c, m, kk, n, dt, b_f16, a_f16, ta, tb) =
                        (*a, *b, *c, *m, *kk, *n, *dt, *b_f16, *a_f16, *ta, *tb);
                    let m_scaled = scale(m);
                    if m_scaled == 0 {
                        continue;
                    }
                    let (mu, ku, nu) = (m_scaled as usize, kk as usize, n as usize);
                    // Transpose-folded GEMM: read the pre-transpose source(s) with
                    // MPS transposeLeft/Right — no materialized transpose. Each
                    // operand is resolved to its own buffer (arena, or the weight
                    // buffer when `RLX_METAL_MATMUL_TRANSPOSE_FOLD_WEIGHTS` folded a
                    // `matmul_t(weight)`); for the arena-only fold this is identical
                    // to the single-buffer path.
                    if ta || tb {
                        end_msl!();
                        let (a_buf, a_off) = self.resolve_off(a);
                        let (b_buf, b_off) = self.resolve_off(b);
                        let (c_buf, c_off) = self.resolve_off(c);
                        crate::mps_blas::encode_mps_sgemm_t_bufs(
                            &cmd_buf, a_buf, a_off, b_buf, b_off, c_buf, c_off, mu, ku, nu, ta, tb,
                        );
                        continue;
                    }
                    let use_mps = crate::cost::hw_model().pick_sgemm(mu, ku, nu)
                        == crate::cost::SgemmVariant::Mps;
                    if a_f16 && !b_f16 && matches!(dt, HalfFlag::F32) {
                        // Mixed-precision backward: A is f16, B (upstream grad) is
                        // f32, C is f32. Read A as half in-kernel with f32 accumulate
                        // — the plain f32 sgemm would misread A's 2-byte elements.
                        let (a_buf, a_off) = self.resolve_off(a);
                        let (b_buf, b_off) = self.resolve_off(b);
                        let (c_buf, c_off) = self.resolve_off(c);
                        crate::blas::metal_sgemm_f16a_bufs(
                            e!(),
                            a_buf,
                            a_off,
                            b_buf,
                            b_off,
                            c_buf,
                            c_off,
                            mu,
                            ku,
                            nu,
                        );
                    } else if b_f16 && matches!(dt, HalfFlag::F32) {
                        // Native F16-weight sgemm — no per-step full-matrix cast.
                        let (a_buf, a_off) = self.resolve_off(a);
                        let (b_buf, b_off) = self.resolve_off(b);
                        let (c_buf, c_off) = self.resolve_off(c);
                        crate::blas::metal_sgemm_f16w_bufs(
                            e!(),
                            a_buf,
                            a_off,
                            b_buf,
                            b_off,
                            c_buf,
                            c_off,
                            mu,
                            ku,
                            nu,
                        );
                    } else if use_mps && matches!(dt, HalfFlag::F16) {
                        end_msl!();
                        // MPS GEMM helpers are single-buffer today; weight-split
                        // graphs keep large B in the weight buffer — fall back
                        // to MSL when any operand is weight-tagged.
                        if crate::thunk::is_weight_off(a)
                            || crate::thunk::is_weight_off(b)
                            || crate::thunk::is_weight_off(c)
                        {
                            let (a_buf, a_off) = self.resolve_off(a);
                            let (b_buf, b_off) = self.resolve_off(b);
                            let (c_buf, c_off) = self.resolve_off(c);
                            crate::blas::metal_hgemm_bufs(
                                e!(),
                                a_buf,
                                a_off,
                                b_buf,
                                b_off,
                                c_buf,
                                c_off,
                                mu,
                                ku,
                                nu,
                            );
                        } else {
                            crate::mps_blas::encode_mps_hgemm(
                                &cmd_buf,
                                &self.arena.buffer,
                                a,
                                b,
                                c,
                                mu,
                                ku,
                                nu,
                            );
                        }
                    } else if use_mps {
                        end_msl!();
                        if crate::thunk::is_weight_off(a)
                            || crate::thunk::is_weight_off(b)
                            || crate::thunk::is_weight_off(c)
                        {
                            let (a_buf, a_off) = self.resolve_off(a);
                            let (b_buf, b_off) = self.resolve_off(b);
                            let (c_buf, c_off) = self.resolve_off(c);
                            crate::blas::metal_sgemm_bufs(
                                e!(),
                                a_buf,
                                a_off,
                                b_buf,
                                b_off,
                                c_buf,
                                c_off,
                                mu,
                                ku,
                                nu,
                            );
                        } else {
                            crate::mps_blas::encode_mps_sgemm(
                                &cmd_buf,
                                &self.arena.buffer,
                                a,
                                b,
                                c,
                                mu,
                                ku,
                                nu,
                            );
                        }
                    } else if matches!(dt, HalfFlag::F16) {
                        let (a_buf, a_off) = self.resolve_off(a);
                        let (b_buf, b_off) = self.resolve_off(b);
                        let (c_buf, c_off) = self.resolve_off(c);
                        crate::blas::metal_hgemm_bufs(
                            e!(),
                            a_buf,
                            a_off,
                            b_buf,
                            b_off,
                            c_buf,
                            c_off,
                            mu,
                            ku,
                            nu,
                        );
                    } else {
                        let (a_buf, a_off) = self.resolve_off(a);
                        let (b_buf, b_off) = self.resolve_off(b);
                        let (c_buf, c_off) = self.resolve_off(c);
                        crate::blas::metal_sgemm_bufs(
                            e!(),
                            a_buf,
                            a_off,
                            b_buf,
                            b_off,
                            c_buf,
                            c_off,
                            mu,
                            ku,
                            nu,
                        );
                    }
                }
                Thunk::SgemmResidual {
                    a,
                    b,
                    c,
                    r,
                    m,
                    k: kk,
                    n,
                    dt: _,
                } => {
                    use crate::thunk::HalfFlag;
                    let (a, b, c, r, m, kk, n) = (*a, *b, *c, *r, *m, *kk, *n);
                    let m_scaled = scale(m);
                    if m_scaled == 0 {
                        continue;
                    }
                    let (mu, ku, nu) = (m_scaled as usize, kk as usize, n as usize);
                    let (a_buf, a_off) = self.resolve_off(a);
                    let (b_buf, b_off) = self.resolve_off(b);
                    let (c_buf, c_off) = self.resolve_off(c);
                    let (r_buf, r_off) = self.resolve_off(r);
                    let fused = crate::blas::metal_sgemm_residual_bufs(
                        e!(),
                        a_buf,
                        a_off,
                        b_buf,
                        b_off,
                        c_buf,
                        c_off,
                        r_buf,
                        r_off,
                        mu,
                        ku,
                        nu,
                    );
                    if !fused {
                        // Picked variant has no residual epilogue: `C = A·B` is
                        // written; add the residual as a separate pass. `c` and
                        // `r` are activation arena offsets (never weight-split).
                        let len = (mu * nu) as u32;
                        encode_binary(
                            e!(),
                            k,
                            &self.arena.buffer,
                            c,
                            r,
                            c,
                            len,
                            rlx_ir::op::BinaryOp::Add,
                            HalfFlag::F32,
                        );
                    }
                }
                Thunk::FusedMmBiasAct {
                    a,
                    w,
                    bias,
                    c,
                    m,
                    k: kk,
                    n,
                    act,
                    dt,
                } => {
                    use crate::thunk::HalfFlag;
                    use rlx_ir::op::Activation;
                    let fa = match act {
                        Some(Activation::Gelu) => crate::blas::FusedAct::Gelu,
                        Some(Activation::Silu) => crate::blas::FusedAct::Silu,
                        _ => crate::blas::FusedAct::None,
                    };
                    let kernel_applies_act =
                        matches!(act, Some(Activation::Gelu) | Some(Activation::Silu));
                    let m_scaled = scale(*m);
                    if m_scaled == 0 {
                        continue;
                    }
                    // Opt-in register-blocked fused GEMM (measured ~2.14× vs
                    // MPS+separate-epilogue on aligned shapes). Safe subset: f32,
                    // 64-aligned, bias + {none|ReLU}. Buffer-aware, so it handles
                    // weight-buffer operands. Everything else falls through.
                    if rlx_ir::env::flag("RLX_METAL_RB_FUSED_GEMM")
                        && matches!(dt, HalfFlag::F32)
                        && matches!(act, None | Some(Activation::Relu))
                        && (m_scaled as usize).is_multiple_of(64)
                        && (*n as usize).is_multiple_of(64)
                        && (*kk as usize).is_multiple_of(16)
                    {
                        if rlx_ir::env::flag("RLX_METAL_RB_DEBUG") {
                            eprintln!("[rb] fused gemm {}x{}x{}", m_scaled, kk, n);
                        }
                        let (a_buf, a_off) = self.resolve_off(*a);
                        let (w_buf, w_off) = self.resolve_off(*w);
                        let (bias_buf, bias_off) = self.resolve_off(*bias);
                        let (c_buf, c_off) = self.resolve_off(*c);
                        let enc = e!();
                        enc.set_compute_pipeline_state(&k.gemm_rb_bias);
                        enc.set_buffer(0, Some(a_buf), a_off as u64);
                        enc.set_buffer(1, Some(w_buf), w_off as u64);
                        enc.set_buffer(2, Some(bias_buf), bias_off as u64);
                        enc.set_buffer(3, Some(c_buf), c_off as u64);
                        for (i, v) in [m_scaled, *kk, *n].iter().enumerate() {
                            enc.set_bytes((i + 4) as u64, 4, v as *const u32 as *const _);
                        }
                        let act_id: u32 = u32::from(matches!(act, Some(Activation::Relu)));
                        enc.set_bytes(7, 4, &act_id as *const u32 as *const _);
                        enc.dispatch_thread_groups(
                            metal::MTLSize {
                                width: (*n as u64) / 64,
                                height: (m_scaled as u64) / 64,
                                depth: 1,
                            },
                            metal::MTLSize {
                                width: 512,
                                height: 1,
                                depth: 1,
                            },
                        );
                        continue;
                    }
                    let (mu, ku, nu) = (m_scaled as usize, *kk as usize, *n as usize);
                    let use_mps = crate::cost::hw_model().pick_sgemm(mu, ku, nu)
                        == crate::cost::SgemmVariant::Mps;
                    let weight_split = crate::thunk::is_weight_off(*w)
                        || crate::thunk::is_weight_off(*bias)
                        || crate::thunk::is_weight_off(*a)
                        || crate::thunk::is_weight_off(*c);
                    if weight_split {
                        let (a_buf, a_off) = self.resolve_off(*a);
                        let (w_buf, w_off) = self.resolve_off(*w);
                        let (c_buf, c_off) = self.resolve_off(*c);
                        let (bias_buf, bias_off) = self.resolve_off(*bias);
                        if matches!(dt, HalfFlag::F16) {
                            crate::blas::metal_hgemm_bufs(
                                e!(),
                                a_buf,
                                a_off,
                                w_buf,
                                w_off,
                                c_buf,
                                c_off,
                                mu,
                                ku,
                                nu,
                            );
                        } else {
                            crate::blas::metal_sgemm_bufs(
                                e!(),
                                a_buf,
                                a_off,
                                w_buf,
                                w_off,
                                c_buf,
                                c_off,
                                mu,
                                ku,
                                nu,
                            );
                        }
                        encode_bias_add(
                            e!(),
                            k,
                            c_buf,
                            c_off,
                            bias_buf,
                            bias_off,
                            m_scaled,
                            *n,
                            *dt,
                        );
                        if let Some(activation) = act {
                            encode_activation(
                                e!(),
                                k,
                                c_buf,
                                c_off,
                                m_scaled * *n,
                                *activation,
                                *dt,
                            );
                        }
                    } else if use_mps {
                        end_msl!();
                        if matches!(dt, HalfFlag::F16) {
                            crate::mps_blas::encode_mps_hgemm(
                                &cmd_buf,
                                &self.arena.buffer,
                                *a,
                                *w,
                                *c,
                                mu,
                                ku,
                                nu,
                            );
                        } else {
                            crate::mps_blas::encode_mps_sgemm(
                                &cmd_buf,
                                &self.arena.buffer,
                                *a,
                                *w,
                                *c,
                                mu,
                                ku,
                                nu,
                            );
                        }
                        {
                            let (c_buf, c_off) = self.resolve_off(*c);
                            let (bias_buf, bias_off) = self.resolve_off(*bias);
                            encode_bias_add(
                                e!(),
                                k,
                                c_buf,
                                c_off,
                                bias_buf,
                                bias_off,
                                m_scaled,
                                *n,
                                *dt,
                            );
                        }
                        if let Some(activation) = act {
                            encode_activation(
                                e!(),
                                k,
                                &self.arena.buffer,
                                *c,
                                m_scaled * *n,
                                *activation,
                                *dt,
                            );
                        }
                    } else if matches!(dt, HalfFlag::F16) {
                        crate::blas::metal_hgemm_bias(
                            e!(),
                            &self.arena.buffer,
                            *a,
                            *w,
                            *bias,
                            *c,
                            mu,
                            ku,
                            nu,
                            fa,
                        );
                        if let Some(activation) = act.filter(|_| !kernel_applies_act) {
                            encode_activation(
                                e!(),
                                k,
                                &self.arena.buffer,
                                *c,
                                m_scaled * *n,
                                activation,
                                *dt,
                            );
                        }
                    } else {
                        crate::blas::metal_sgemm_bias(
                            e!(),
                            &self.arena.buffer,
                            *a,
                            *w,
                            *bias,
                            *c,
                            mu,
                            ku,
                            nu,
                            fa,
                        );
                        if let Some(activation) = act.filter(|_| !kernel_applies_act) {
                            encode_activation(
                                e!(),
                                k,
                                &self.arena.buffer,
                                *c,
                                m_scaled * *n,
                                activation,
                                *dt,
                            );
                        }
                    }
                }
                Thunk::GeluApproxHost { src, dst, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    deferred_host.push(DeferredHostOp::GeluApproxHost {
                        src: *src,
                        dst: *dst,
                        len,
                    });
                }
                Thunk::GeluApproxOut { src, dst, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    if arena_off_large(*src) || arena_off_large(*dst) {
                        flush_deferred_host(&mut cmd_buf, &mut enc, &mut deferred_host);
                        if let Some(active) = enc.take() {
                            active.end_encoding();
                        }
                        cmd_buf.commit();
                        cmd_buf.wait_until_completed();
                        cmd_buf = dev.queue.new_command_buffer().to_owned();
                        crate::mps_gelu::encode_gelu_approx_out(
                            &dev.queue,
                            &self.arena.buffer,
                            *src,
                            *dst,
                            len as usize,
                        );
                    } else {
                        sync_gpu!();
                        encode_gelu_approx_out(e!(), k, &self.arena.buffer, *src, *dst, len);
                    }
                }
                Thunk::ActivationInPlace { data, len, act, dt } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    let host_act = metal_host_fallback_enabled()
                        && arena_off_large(*data)
                        && matches!(
                            act,
                            rlx_ir::op::Activation::Gelu
                                | rlx_ir::op::Activation::GeluApprox
                                | rlx_ir::op::Activation::Silu
                        )
                        && matches!(*dt, crate::thunk::HalfFlag::F32);
                    if host_act {
                        deferred_host.push(DeferredHostOp::ActivationHost {
                            data: *data,
                            len,
                            act: *act,
                        });
                    } else if matches!(act, rlx_ir::op::Activation::GeluApprox)
                        && arena_off_large(*data)
                        && matches!(*dt, crate::thunk::HalfFlag::F32)
                    {
                        flush_deferred_host(&mut cmd_buf, &mut enc, &mut deferred_host);
                        if let Some(active) = enc.take() {
                            active.end_encoding();
                        }
                        cmd_buf.commit();
                        cmd_buf.wait_until_completed();
                        cmd_buf = dev.queue.new_command_buffer().to_owned();
                        crate::mps_gelu::encode_gelu_approx_out(
                            &dev.queue,
                            &self.arena.buffer,
                            *data,
                            *data,
                            len as usize,
                        );
                    } else {
                        encode_activation(e!(), k, &self.arena.buffer, *data, len, *act, *dt);
                    }
                }
                Thunk::ActivationOut {
                    src,
                    dst,
                    len,
                    act,
                    dt,
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_activation_out(e!(), k, &self.arena.buffer, *src, *dst, len, *act, *dt);
                }
                Thunk::FusedBinaryActivation {
                    lhs,
                    rhs,
                    dst,
                    len,
                    op,
                    act,
                    dt,
                } => {
                    use crate::thunk::HalfFlag;
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    if matches!(dt, HalfFlag::F32) {
                        encode_fused_binary_activation(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *lhs,
                            *rhs,
                            *dst,
                            len,
                            *op,
                            *act,
                        );
                    } else {
                        // No fused f16 kernel yet — binary then activation.
                        encode_binary(e!(), k, &self.arena.buffer, *lhs, *rhs, *dst, len, *op, *dt);
                        encode_activation(e!(), k, &self.arena.buffer, *dst, len, *act, *dt);
                    }
                }
                Thunk::FusedTernaryActivation {
                    lhs,
                    rhs0,
                    rhs1,
                    dst,
                    len,
                    op0,
                    op1,
                    act,
                    dt,
                } => {
                    use crate::thunk::HalfFlag;
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    if matches!(dt, HalfFlag::F32) {
                        encode_fused_ternary_activation(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *lhs,
                            *rhs0,
                            *rhs1,
                            *dst,
                            len,
                            *op0,
                            *op1,
                            *act,
                        );
                    } else {
                        // Decompose: (lhs op0 rhs0) → dst, then (dst op1 rhs1) → dst,
                        // then activation. Avoids the previous silent F16 skip.
                        encode_binary(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *lhs,
                            *rhs0,
                            *dst,
                            len,
                            *op0,
                            *dt,
                        );
                        encode_binary(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *dst,
                            *rhs1,
                            *dst,
                            len,
                            *op1,
                            *dt,
                        );
                        encode_activation(e!(), k, &self.arena.buffer, *dst, len, *act, *dt);
                    }
                }
                Thunk::LayerNorm {
                    src,
                    g,
                    b,
                    dst,
                    rows,
                    h,
                    eps,
                    dt,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_layer_norm(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *g,
                        *b,
                        *dst,
                        rows,
                        *h,
                        *eps,
                        *dt,
                    );
                }
                Thunk::GroupNorm {
                    src,
                    g,
                    b,
                    dst,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps,
                    dt: _,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_group_norm(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *g,
                        *b,
                        *dst,
                        n,
                        *c,
                        *h,
                        *w,
                        *num_groups,
                        *eps,
                    );
                }
                Thunk::LayerNorm2d {
                    src,
                    g,
                    b,
                    dst,
                    n,
                    c,
                    h,
                    w,
                    eps,
                    dt: _,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_layer_norm2d(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *g,
                        *b,
                        *dst,
                        n,
                        *c,
                        *h,
                        *w,
                        *eps,
                    );
                }
                Thunk::ConvTranspose2d {
                    src,
                    weight,
                    dst,
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
                    dt: _,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_conv_transpose2d(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *weight,
                        *dst,
                        n,
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
                }
                Thunk::Conv3d {
                    src,
                    weight,
                    dst,
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
                    dt: _,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_conv3d(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *weight,
                        *dst,
                        n,
                        *c_in,
                        *d,
                        *h,
                        *w_in,
                        *c_out,
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
                    );
                }
                Thunk::ConvTranspose3d {
                    src,
                    weight,
                    dst,
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
                    dt: _,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_conv_transpose3d(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *weight,
                        *dst,
                        n,
                        *c_in,
                        *d,
                        *h,
                        *w_in,
                        *c_out,
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
                    );
                }
                Thunk::ResizeNearest2x {
                    src,
                    dst,
                    n,
                    c,
                    h,
                    w,
                    dt: _,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_resize_nearest_2x(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        n,
                        *c,
                        *h,
                        *w,
                    );
                }
                Thunk::RmsNorm {
                    src,
                    g,
                    b,
                    dst,
                    rows,
                    h,
                    eps,
                    dt,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_rms_norm(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *g,
                        *b,
                        *dst,
                        rows,
                        *h,
                        *eps,
                        *dt,
                    );
                }
                Thunk::BiasAdd {
                    src,
                    bias,
                    dst,
                    m,
                    n,
                    dt,
                } => {
                    let m_scaled = scale(*m);
                    if m_scaled == 0 {
                        continue;
                    }
                    if *src != *dst {
                        encode_copy(e!(), k, &self.arena.buffer, *src, *dst, m_scaled * n, *dt);
                    }
                    {
                        let (dst_buf, dst_off) = self.resolve_off(*dst);
                        let (bias_buf, bias_off) = self.resolve_off(*bias);
                        encode_bias_add(
                            e!(),
                            k,
                            dst_buf,
                            dst_off,
                            bias_buf,
                            bias_off,
                            m_scaled,
                            *n,
                            *dt,
                        );
                    }
                }
                Thunk::BinaryFull {
                    lhs,
                    rhs,
                    dst,
                    len,
                    op,
                    dt,
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    use crate::thunk::HalfFlag;
                    use rlx_ir::op::BinaryOp;
                    let big =
                        arena_off_large(*lhs) || arena_off_large(*rhs) || arena_off_large(*dst);
                    if metal_host_fallback_enabled()
                        && big
                        && matches!(*dt, HalfFlag::F32)
                        && matches!(
                            op,
                            BinaryOp::Add | BinaryOp::Mul | BinaryOp::Sub | BinaryOp::Div
                        )
                    {
                        deferred_host.push(DeferredHostOp::BinaryHost {
                            lhs: *lhs,
                            rhs: *rhs,
                            dst: *dst,
                            len,
                            op: *op,
                        });
                    } else {
                        encode_binary(e!(), k, &self.arena.buffer, *lhs, *rhs, *dst, len, *op, *dt);
                    }
                }
                Thunk::BatchedSgemm {
                    a,
                    b,
                    c,
                    batch,
                    m,
                    k: kk,
                    n,
                    dt,
                    a_bcast,
                    b_bcast,
                } => {
                    use crate::thunk::HalfFlag;
                    let m_scaled = scale(*m);
                    if m_scaled == 0 {
                        continue;
                    }
                    let (mu, ku, nu, b_) = (
                        m_scaled as usize,
                        *kk as usize,
                        *n as usize,
                        *batch as usize,
                    );
                    let elem = if matches!(dt, HalfFlag::F16) { 2 } else { 4 };
                    // A broadcast operand (batch dim 1) reuses matrix 0 for every
                    // output batch → per-matrix stride 0.
                    let a_stride = if *a_bcast { 0 } else { mu * ku * elem };
                    let b_stride = if *b_bcast { 0 } else { ku * nu * elem };
                    let c_stride = mu * nu * elem;
                    // `BatchedSgemm` defaults to MPS for throughput, but
                    // MPS uses `simdgroup_float8x8` reduced-precision
                    // accumulators (~fp16 class) that drift ~1e-1 vs
                    // CPU on tight-tolerance work. Honor
                    // `RLX_METAL_PRECISE=1` by routing through the
                    // scalar `metal_sgemm` instead — same path
                    // `Thunk::Sgemm` takes when `pick_sgemm` returns
                    // `SgemmVariant::Naive`. F16 has no scalar
                    // fallback right now, so we keep MPS for it.
                    let force_precise = matches!(
                        crate::cost::sgemm_variant_override(),
                        Some(crate::cost::SgemmVariant::Naive)
                    );
                    if force_precise && !matches!(dt, HalfFlag::F16) {
                        for bi in 0..b_ {
                            let a_off = *a + bi * a_stride;
                            let b_off = *b + bi * b_stride;
                            let c_off = *c + bi * c_stride;
                            crate::blas::metal_sgemm(
                                e!(),
                                &self.arena.buffer,
                                a_off,
                                b_off,
                                c_off,
                                mu,
                                ku,
                                nu,
                            );
                        }
                    } else {
                        // End any open compute encoder; MPS opens its own.
                        end_msl!();
                        for bi in 0..b_ {
                            let a_off = *a + bi * a_stride;
                            let b_off = *b + bi * b_stride;
                            let c_off = *c + bi * c_stride;
                            if matches!(dt, HalfFlag::F16) {
                                crate::mps_blas::encode_mps_hgemm(
                                    &cmd_buf,
                                    &self.arena.buffer,
                                    a_off,
                                    b_off,
                                    c_off,
                                    mu,
                                    ku,
                                    nu,
                                );
                            } else {
                                crate::mps_blas::encode_mps_sgemm(
                                    &cmd_buf,
                                    &self.arena.buffer,
                                    a_off,
                                    b_off,
                                    c_off,
                                    mu,
                                    ku,
                                    nu,
                                );
                            }
                        }
                    }
                }
                Thunk::BinaryBroadcast {
                    lhs,
                    rhs,
                    dst,
                    len,
                    op,
                    dt,
                    rank,
                    out_dims,
                    lhs_strides,
                    rhs_strides,
                } => {
                    use crate::thunk::HalfFlag;
                    let total_out = scale(*len) as usize;
                    if total_out == 0 {
                        continue;
                    }
                    // F16 path still falls back to the host (no f16 MSL
                    // kernel yet); f32 uses the dedicated GPU kernel.
                    if matches!(dt, HalfFlag::F32) {
                        let op_id: u32 = op.opcode();
                        // Sub/Div/Pow are non-commutative. The scalar/col/row/
                        // 1-axis fast paths swap operands when the *lhs* is the
                        // broadcast side (feeding the kernel `a OP b` with
                        // a=rhs, b=lhs), which silently computes `rhs OP lhs`.
                        // Only take a swapping fast path when no swap is needed
                        // (rhs is the broadcast side) or the op is commutative;
                        // otherwise fall through to the order-preserving general
                        // kernels (`binary_broadcast_rank2` / `binary_broadcast_f32`).
                        let commutative = matches!(
                            op,
                            rlx_ir::op::BinaryOp::Add
                                | rlx_ir::op::BinaryOp::Mul
                                | rlx_ir::op::BinaryOp::Max
                                | rlx_ir::op::BinaryOp::Min
                        );
                        let enc = e!();
                        if let Some(rhs_scalar) =
                            detect_scalar_broadcast(*rank, out_dims, lhs_strides, rhs_strides)
                                .filter(|&rhs_scalar| rhs_scalar || commutative)
                        {
                            encode_binary_broadcast_rhs_scalar(
                                enc,
                                k,
                                &self.arena.buffer,
                                *lhs,
                                *rhs,
                                *dst,
                                total_out as u32,
                                op_id,
                                rhs_scalar,
                            );
                            continue;
                        }
                        if let Some((rows, cols, rhs_col)) = detect_last_axis_col_broadcast(
                            *rank,
                            out_dims,
                            lhs_strides,
                            rhs_strides,
                        )
                        .filter(|&(_, _, rhs_col)| rhs_col || commutative)
                        {
                            encode_binary_broadcast_rhs_col(
                                enc,
                                k,
                                &self.arena.buffer,
                                *lhs,
                                *rhs,
                                *dst,
                                rows,
                                cols,
                                op_id,
                                rhs_col,
                            );
                            continue;
                        }
                        if let Some((rows, cols, rhs_row)) = detect_last_axis_row_broadcast(
                            *rank,
                            out_dims,
                            lhs_strides,
                            rhs_strides,
                        )
                        .filter(|&(_, _, rhs_row)| rhs_row || commutative)
                        {
                            encode_binary_broadcast_rhs_row(
                                enc,
                                k,
                                &self.arena.buffer,
                                *lhs,
                                *rhs,
                                *dst,
                                rows,
                                cols,
                                op_id,
                                rhs_row,
                            );
                            continue;
                        }
                        if let Some((rows, cols, mid, rhs_1ax)) =
                            detect_single_axis_broadcast(*rank, out_dims, lhs_strides, rhs_strides)
                                .filter(|&(_, _, _, rhs_1ax)| rhs_1ax || commutative)
                        {
                            encode_binary_broadcast_1ax(
                                enc,
                                k,
                                &self.arena.buffer,
                                *lhs,
                                *rhs,
                                *dst,
                                rows,
                                cols,
                                mid,
                                op_id,
                                rhs_1ax,
                            );
                            continue;
                        }
                        if *rank == 2
                            && out_dims.len() >= 2
                            && lhs_strides.len() >= 2
                            && rhs_strides.len() >= 2
                        {
                            encode_binary_broadcast_rank2(
                                enc,
                                k,
                                &self.arena.buffer,
                                *lhs,
                                *rhs,
                                *dst,
                                total_out as u32,
                                out_dims[0],
                                out_dims[1],
                                lhs_strides[0],
                                lhs_strides[1],
                                rhs_strides[0],
                                rhs_strides[1],
                                op_id,
                            );
                            continue;
                        }
                        enc.set_compute_pipeline_state(&k.binary_broadcast_f32);
                        enc.set_buffer(0, Some(&self.arena.buffer), *lhs as u64);
                        enc.set_buffer(1, Some(&self.arena.buffer), *rhs as u64);
                        enc.set_buffer(2, Some(&self.arena.buffer), *dst as u64);
                        let len_u32 = total_out as u32;
                        let rank_u32 = *rank;
                        enc.set_bytes(3, 4, &len_u32 as *const u32 as *const _);
                        enc.set_bytes(4, 4, &rank_u32 as *const u32 as *const _);
                        let dims_bytes = (out_dims.len() * 4) as u64;
                        enc.set_bytes(5, dims_bytes, out_dims.as_ptr() as *const _);
                        enc.set_bytes(
                            6,
                            (lhs_strides.len() * 4) as u64,
                            lhs_strides.as_ptr() as *const _,
                        );
                        enc.set_bytes(
                            7,
                            (rhs_strides.len() * 4) as u64,
                            rhs_strides.as_ptr() as *const _,
                        );
                        enc.set_bytes(8, 4, &op_id as *const u32 as *const _);
                        let grid = metal::MTLSize {
                            width: total_out as u64,
                            height: 1,
                            depth: 1,
                        };
                        let tg_w = k
                            .binary_broadcast_f32
                            .thread_execution_width()
                            .min(total_out as u64);
                        let tg = metal::MTLSize {
                            width: tg_w,
                            height: 1,
                            depth: 1,
                        };
                        enc.dispatch_threads(grid, tg);
                    } else {
                        // f16: unified-memory host fallback (rare path
                        // until we get a half-precision kernel).
                        end_msl!();
                        cmd_buf.commit();
                        cmd_buf.wait_until_completed();
                        let arena_ptr = self.arena.buffer.contents() as *mut u8;
                        let lhs_len_in = inferred_input_len(lhs_strides, out_dims);
                        let rhs_len_in = inferred_input_len(rhs_strides, out_dims);
                        unsafe {
                            binary_broadcast_host::<half::f16>(
                                arena_ptr.add(*lhs) as *const half::f16,
                                lhs_len_in,
                                arena_ptr.add(*rhs) as *const half::f16,
                                rhs_len_in,
                                arena_ptr.add(*dst) as *mut half::f16,
                                total_out,
                                *rank as usize,
                                out_dims,
                                lhs_strides,
                                rhs_strides,
                                *op,
                            );
                        }
                        cmd_buf = dev.queue.new_command_buffer().to_owned();
                    }
                }
                Thunk::FusedResidualLN {
                    x,
                    res,
                    bias,
                    g,
                    b,
                    out,
                    rows,
                    h,
                    eps,
                    has_bias,
                    dt,
                } => {
                    let _ = (bias, has_bias);
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_fused_residual_ln(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *res,
                        *g,
                        *b,
                        *out,
                        rows,
                        *h,
                        *eps,
                        *dt,
                    );
                }
                Thunk::FusedResidualRmsNorm {
                    x,
                    res,
                    bias,
                    g,
                    b,
                    out,
                    rows,
                    h,
                    eps,
                    has_bias,
                    dt,
                    sum_out,
                } => {
                    let _ = (bias, has_bias);
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_fused_residual_rms_norm(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *res,
                        *g,
                        *b,
                        *out,
                        rows,
                        *h,
                        *eps,
                        *dt,
                        *sum_out,
                    );
                }
                Thunk::AdaLayerNorm {
                    x,
                    scale: scale_off,
                    shift,
                    out,
                    rows,
                    h,
                    eps,
                    layer_norm,
                    lead_pack,
                    dt,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_ada_layer_norm(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *scale_off,
                        *shift,
                        *out,
                        rows,
                        *h,
                        *eps,
                        *layer_norm,
                        lead_pack,
                        *dt,
                    );
                }
                Thunk::GatedResidual {
                    x,
                    y,
                    gate,
                    out,
                    rows,
                    h,
                    lead_pack,
                    dt,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_gated_residual(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *y,
                        *gate,
                        *out,
                        rows,
                        *h,
                        lead_pack,
                        *dt,
                    );
                }
                Thunk::AdaLayerNormBackward {
                    x,
                    scale: scale_off,
                    dy,
                    out,
                    h,
                    eps,
                    layer_norm,
                    seq_per_mod,
                    mod_rows,
                    dt,
                } => {
                    if *mod_rows == 0 {
                        continue;
                    }
                    encode_ada_layer_norm_backward(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *scale_off,
                        *dy,
                        *out,
                        *h,
                        *eps,
                        *layer_norm,
                        *seq_per_mod,
                        *mod_rows,
                        *dt,
                    );
                }
                Thunk::GatedResidualBackward {
                    y,
                    gate,
                    dy,
                    out,
                    h,
                    seq_per_mod,
                    mod_rows,
                    dt,
                } => {
                    if *mod_rows == 0 {
                        continue;
                    }
                    encode_gated_residual_backward(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *y,
                        *gate,
                        *dy,
                        *out,
                        *h,
                        *seq_per_mod,
                        *mod_rows,
                        *dt,
                    );
                }
                Thunk::FusedRmsNormMulSilu {
                    x,
                    g,
                    b,
                    z,
                    out,
                    rows,
                    h,
                    eps,
                    dt,
                } => {
                    let _ = dt;
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_rms_norm_mul_silu(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *g,
                        *b,
                        *z,
                        *out,
                        rows,
                        *h,
                        *eps,
                    );
                }
                Thunk::FusedDepthwiseConv1dBsc {
                    src,
                    weight,
                    dst,
                    batch,
                    width,
                    out_seq,
                    channels,
                    k: kw,
                    silu,
                } => {
                    let batch = scale(*batch);
                    if batch == 0 {
                        continue;
                    }
                    let (w_buf, w_raw) = self.resolve_off(*weight);
                    encode_depthwise_conv1d_bsc(
                        e!(),
                        k,
                        &self.arena.buffer,
                        w_buf,
                        *src,
                        w_raw,
                        *dst,
                        batch,
                        *width,
                        *out_seq,
                        *channels,
                        *kw,
                        *silu,
                    );
                }
                Thunk::Gather {
                    table,
                    idx,
                    dst,
                    num_idx,
                    trailing,
                    dt,
                } => {
                    let num_idx = scale(*num_idx);
                    if num_idx == 0 {
                        continue;
                    }
                    {
                        let (table_buf, table_off) = self.resolve_off(*table);
                        let (idx_buf, idx_off) = self.resolve_off(*idx);
                        let (dst_buf, dst_off) = self.resolve_off(*dst);
                        encode_gather(
                            e!(),
                            k,
                            table_buf,
                            table_off,
                            idx_buf,
                            idx_off,
                            dst_buf,
                            dst_off,
                            num_idx,
                            *trailing,
                            *dt,
                        );
                    }
                }
                Thunk::SplitLastAxis {
                    src,
                    outer,
                    src_axis,
                    dt,
                    segments,
                } => {
                    let outer = scale(*outer);
                    if outer == 0 || segments.is_empty() {
                        continue;
                    }
                    if narrow_batch.is_some() {
                        flush_pending_narrow_batch(e!(), k, &self.arena.buffer, &mut narrow_batch);
                    }
                    let batch = PendingNarrowBatch {
                        src: *src,
                        outer,
                        src_axis: *src_axis,
                        dt: *dt,
                        segments: segments.clone(),
                    };
                    encode_split_lastax(e!(), k, &self.arena.buffer, &batch);
                }
                Thunk::Narrow {
                    src,
                    dst,
                    outer,
                    src_axis,
                    start,
                    len,
                    dt,
                } => {
                    let outer = scale(*outer);
                    if outer == 0 {
                        continue;
                    }
                    if metal_host_slices_enabled() && matches!(*dt, crate::thunk::HalfFlag::F32) {
                        if narrow_batch.is_some() {
                            flush_pending_narrow_batch(
                                e!(),
                                k,
                                &self.arena.buffer,
                                &mut narrow_batch,
                            );
                        }
                        deferred_host.push(DeferredHostOp::NarrowLastAxis {
                            src: *src,
                            dst: *dst,
                            outer,
                            src_axis: *src_axis,
                            start: *start,
                            len: *len,
                        });
                    } else if outer == 1 {
                        if narrow_batch.is_some() {
                            flush_pending_narrow_batch(
                                e!(),
                                k,
                                &self.arena.buffer,
                                &mut narrow_batch,
                            );
                        }
                        let elem = match *dt {
                            crate::thunk::HalfFlag::F16 => 2usize,
                            crate::thunk::HalfFlag::F32 => 4usize,
                        };
                        let src_off = *src + (*start as usize) * elem;
                        encode_copy(e!(), k, &self.arena.buffer, src_off, *dst, *len, *dt);
                    } else if *start == 0 && *src_axis == *len {
                        if narrow_batch.is_some() {
                            flush_pending_narrow_batch(
                                e!(),
                                k,
                                &self.arena.buffer,
                                &mut narrow_batch,
                            );
                        }
                        encode_copy(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *src,
                            *dst,
                            outer.saturating_mul(*len),
                            *dt,
                        );
                    } else if !try_queue_narrow_batch(
                        &mut narrow_batch,
                        *src,
                        *dst,
                        outer,
                        *src_axis,
                        *start,
                        *len,
                        *dt,
                    ) {
                        flush_pending_narrow_batch(e!(), k, &self.arena.buffer, &mut narrow_batch);
                        if !try_queue_narrow_batch(
                            &mut narrow_batch,
                            *src,
                            *dst,
                            outer,
                            *src_axis,
                            *start,
                            *len,
                            *dt,
                        ) {
                            encode_narrow(
                                e!(),
                                k,
                                &self.arena.buffer,
                                *src,
                                *dst,
                                outer,
                                *src_axis,
                                *start,
                                *len,
                                *dt,
                            );
                        }
                    }
                }
                Thunk::Copy { src, dst, len, dt } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    let bytes = len as usize * std::mem::size_of::<f32>();
                    if metal_host_fallback_enabled() && matches!(*dt, crate::thunk::HalfFlag::F32) {
                        deferred_host.push(DeferredHostOp::Memcpy {
                            src: *src,
                            dst: *dst,
                            bytes,
                        });
                    } else {
                        encode_copy(e!(), k, &self.arena.buffer, *src, *dst, len, *dt);
                    }
                }
                Thunk::KvAppend {
                    src,
                    dst,
                    outer,
                    seq_cap,
                    pos,
                    inner,
                    dt,
                } => {
                    // Fixed single-row write into the aliased cache buffer — NOT
                    // active-extent-scaled (the new K always lands at row `pos`,
                    // the bucket end; the mask covers the padded gap). Per batch:
                    // copy `inner` elems to dst[(o*seq_cap + pos)*inner ..].
                    let dt_bytes = match dt {
                        crate::thunk::HalfFlag::F16 => 2usize,
                        _ => 4usize,
                    };
                    let inner = *inner as usize;
                    let pos = *pos as usize;
                    let seq_cap = *seq_cap as usize;
                    for o in 0..*outer as usize {
                        let s = *src + o * inner * dt_bytes;
                        let d = *dst + (o * seq_cap + pos) * inner * dt_bytes;
                        encode_copy(e!(), k, &self.arena.buffer, s, d, inner as u32, *dt);
                    }
                }
                Thunk::AttentionBackwardAll {
                    q,
                    k: kk,
                    v,
                    dy,
                    out_dq,
                    out_dk,
                    out_dv,
                    batch,
                    seq,
                    kv_seq,
                    heads,
                    head_dim,
                    mask_kind,
                    window,
                } => {
                    let sq = scale(*seq) as usize;
                    let sk = scale(*kv_seq) as usize;
                    if sq == 0 || sk == 0 {
                        continue;
                    }
                    // Emitted only when GPU-eligible (compile-time: no custom/bias
                    // mask, not [B,H,S,D]); the presence of the source
                    // `AttentionBackward` nodes guarantees the scratch is sized.
                    debug_assert!(
                        self.attn_bwd_scratch_off != 0,
                        "AttentionBackwardAll requires attn-bwd scratch"
                    );
                    crate::attention_bwd_gpu::encode_attention_bwd_all(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *q,
                        *kk,
                        *v,
                        *dy,
                        *out_dq,
                        *out_dk,
                        *out_dv,
                        self.attn_bwd_scratch_off,
                        *batch,
                        sq as u32,
                        sk as u32,
                        *heads,
                        *head_dim,
                        *mask_kind,
                        *window,
                    );
                }
                Thunk::AttentionBackward {
                    q,
                    k: kk,
                    v,
                    dy,
                    mask,
                    out,
                    batch,
                    seq,
                    kv_seq,
                    heads,
                    head_dim,
                    mask_kind,
                    window,
                    wrt,
                    bhsd,
                } => {
                    use rlx_ir::op::{AttentionBwdWrt, MaskKind};
                    let b = *batch as usize;
                    let nh = *heads as usize;
                    let sq = scale(*seq) as usize;
                    let sk = scale(*kv_seq) as usize;
                    let dh = *head_dim as usize;
                    if sq == 0 || sk == 0 {
                        continue;
                    }
                    if crate::attention_bwd_gpu::use_gpu(
                        *mask_kind,
                        *bhsd,
                        sq,
                        sk,
                        self.attn_bwd_scratch_off,
                    ) {
                        crate::attention_bwd_gpu::encode_attention_bwd(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *q,
                            *kk,
                            *v,
                            *dy,
                            *out,
                            self.attn_bwd_scratch_off,
                            *batch,
                            sq as u32,
                            sk as u32,
                            *heads,
                            *head_dim,
                            *mask_kind,
                            *window,
                            *wrt,
                        );
                    } else {
                        let bhsd = *bhsd != 0;
                        let q_len = if bhsd {
                            b * nh * sq * dh
                        } else {
                            b * sq * nh * dh
                        };
                        let k_len = if bhsd {
                            b * nh * sk * dh
                        } else {
                            b * sk * nh * dh
                        };
                        let mask_kind_ir = match *mask_kind {
                            0 => MaskKind::None,
                            1 => MaskKind::Causal,
                            2 => MaskKind::Custom,
                            3 => MaskKind::SlidingWindow(*window as usize),
                            4 => MaskKind::Bias,
                            _ => MaskKind::None,
                        };
                        let wrt_ir = match *wrt {
                            0 => AttentionBwdWrt::Query,
                            1 => AttentionBwdWrt::Key,
                            _ => AttentionBwdWrt::Value,
                        };
                        unsafe {
                            let base = self.arena.buffer.contents() as *mut u8;
                            let f32_at = |byte_off: usize, len: usize| -> &[f32] {
                                std::slice::from_raw_parts(base.add(byte_off) as *const f32, len)
                            };
                            let f32_at_mut = |byte_off: usize, len: usize| -> &mut [f32] {
                                std::slice::from_raw_parts_mut(base.add(byte_off) as *mut f32, len)
                            };
                            let q_data = f32_at(*q, q_len);
                            let k_data = f32_at(*kk, k_len);
                            let v_data = f32_at(*v, k_len);
                            let dy_data = f32_at(*dy, q_len);
                            let out_len = if *wrt == 0 { q_len } else { k_len };
                            let out_data = f32_at_mut(*out, out_len);
                            let mask_data: &[f32] = if *mask_kind == 2 || *mask_kind == 4 {
                                let ml = if *mask_kind == 2 {
                                    b * sk
                                } else {
                                    b * nh * sq * sk
                                };
                                f32_at(*mask, ml)
                            } else {
                                &[]
                            };
                            rlx_cpu::attention_bwd::attention_backward(
                                wrt_ir,
                                q_data,
                                k_data,
                                v_data,
                                dy_data,
                                out_data,
                                b,
                                nh,
                                sq,
                                sk,
                                dh,
                                mask_kind_ir,
                                mask_data,
                                bhsd,
                            );
                        }
                    }
                }
                Thunk::RmsNormBackwardInput {
                    x,
                    gamma,
                    beta,
                    dy,
                    dx,
                    rows,
                    h,
                    eps,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_rms_norm_bwd_input(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *gamma,
                        *beta,
                        *dy,
                        *dx,
                        rows,
                        *h,
                        *eps,
                    );
                }
                Thunk::RmsNormBackwardGamma {
                    x,
                    gamma,
                    beta,
                    dy,
                    dgamma,
                    rows,
                    h,
                    eps,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_rms_norm_bwd_param(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *gamma,
                        *beta,
                        *dy,
                        *dgamma,
                        rows,
                        *h,
                        *eps,
                        1,
                        self.rms_norm_bwd_scratch_off,
                    );
                }
                Thunk::RmsNormBackwardBeta {
                    x,
                    gamma,
                    beta,
                    dy,
                    dbeta,
                    rows,
                    h,
                    eps,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_rms_norm_bwd_param(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *gamma,
                        *beta,
                        *dy,
                        *dbeta,
                        rows,
                        *h,
                        *eps,
                        2,
                        self.rms_norm_bwd_scratch_off,
                    );
                }
                Thunk::LayerNormBackwardInput {
                    x,
                    gamma,
                    dy,
                    dx,
                    rows,
                    h,
                    eps,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_layer_norm_bwd_input(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *gamma,
                        *dy,
                        *dx,
                        rows,
                        *h,
                        *eps,
                    );
                }
                Thunk::LayerNormBackwardGamma {
                    x,
                    dy,
                    dgamma,
                    rows,
                    h,
                    eps,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_layer_norm_bwd_gamma(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *dy,
                        *dgamma,
                        rows,
                        *h,
                        *eps,
                        self.rms_norm_bwd_scratch_off,
                    );
                }
                Thunk::GroupNormBackwardInput {
                    x,
                    gamma,
                    beta: _,
                    dy,
                    dx,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_group_norm_bwd_input(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *gamma,
                        *dy,
                        *dx,
                        n,
                        *c,
                        *h,
                        *w,
                        *num_groups,
                        *eps,
                    );
                }
                Thunk::GroupNormBackwardGamma {
                    x,
                    dy,
                    dgamma,
                    n,
                    c,
                    h,
                    w,
                    num_groups,
                    eps,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_group_norm_bwd_gamma(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *dy,
                        *dgamma,
                        n,
                        *c,
                        *h,
                        *w,
                        *num_groups,
                        *eps,
                    );
                }
                Thunk::GroupNormBackwardBeta {
                    dy,
                    dbeta,
                    n,
                    c,
                    h,
                    w,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_group_norm_bwd_beta(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *dy,
                        *dbeta,
                        n,
                        *c,
                        *h,
                        *w,
                    );
                }
                Thunk::RopeBackward {
                    dy,
                    cos,
                    sin,
                    dx,
                    batch,
                    seq,
                    hidden,
                    head_dim,
                    n_rot,
                    cos_len,
                } => {
                    let seq = scale(*seq);
                    if seq == 0 {
                        continue;
                    }
                    encode_rope_bwd(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *dy,
                        *cos,
                        *sin,
                        *dx,
                        *batch,
                        seq,
                        *hidden,
                        *head_dim,
                        *n_rot,
                        *cos_len,
                    );
                }
                Thunk::CumsumBackward {
                    dy,
                    dx,
                    rows,
                    cols,
                    exclusive,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_cumsum_bwd(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *dy,
                        *dx,
                        rows,
                        *cols,
                        *exclusive,
                    );
                }
                Thunk::GatherBackward {
                    dy,
                    indices,
                    dst,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                } => {
                    let outer = scale(*outer);
                    if outer == 0 {
                        continue;
                    }
                    encode_gather_bwd(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *dy,
                        *indices,
                        *dst,
                        outer,
                        *axis_dim,
                        *num_idx,
                        *trailing,
                    );
                }
                Thunk::MaxPool2dBackward {
                    x,
                    dy,
                    dx,
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
                    let n_eff = scale(*n);
                    if n_eff == 0 {
                        continue;
                    }
                    // Native GPU max-pool backward (output-parallel, no sync).
                    encode_maxpool2d_backward(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *dy,
                        *dx,
                        n_eff,
                        *c,
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
                    );
                }
                Thunk::Conv2dBackwardInput {
                    dy,
                    w,
                    dx,
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
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    // The input gradient is a *transposed* convolution (gather of
                    // `dy` through the weight). Native GPU kernel — one thread per
                    // dx element, no atomics, no sync (replaces the old CPU
                    // fallback that stalled the command-buffer pipeline).
                    encode_conv2d_backward_input(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *dy,
                        *w,
                        *dx,
                        n,
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
                }
                Thunk::Conv2dBackwardWeight {
                    x,
                    dy,
                    dw,
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
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    if n > 1 && self.conv_bwd_scratch_off != 0 {
                        // Native GPU weight-grad, two-pass (batch-parallel):
                        //   pass 1 — one thread per (n, co, ci, ki, kj) writes a
                        //            per-sample partial into scratch (threads scale
                        //            with N, fixing conv1's 288-thread starvation);
                        //   pass 2 — one thread per dw element reduces over N.
                        // Deterministic (no atomics), no GPU→CPU sync.
                        encode_conv2d_backward_weight_2pass(
                            e!(),
                            k,
                            &self.arena.buffer,
                            self.conv_bwd_scratch_off,
                            *x,
                            *dy,
                            *dw,
                            n,
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
                    } else if self.conv_bwd_scratch_off == 0 {
                        // Scratch unavailable (rare): single-pass direct kernel —
                        // one thread per dw element sums dy*x over the batch.
                        encode_conv2d_backward_weight(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *x,
                            *dy,
                            *dw,
                            n,
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
                        let c_in_per_g = *c_in / *groups;
                        let c_out_per_g = *c_out / *groups;
                        let n_dim = c_in_per_g * *kh * *kw;
                        let k_dim = *h_out * *w_out;
                        let x_stride_g = c_in_per_g * *h * *w;
                        let dy_stride_g = c_out_per_g * *h_out * *w_out;
                        let dw_stride_g = c_out_per_g * n_dim;
                        let nchw_slice = [c_in_per_g, *h, *w, 0u32]; // MSL: .x=C, .y=H, .z=W
                        let out_dims = [0u32, *h_out, *w_out, 0u32];
                        let kshape = [*kh, *kw, *sh, *sw];
                        let padd = [*ph, *pw, *dh, *dw_dil];
                        let m = c_out_per_g as usize;
                        let k_dim_usize = k_dim as usize;
                        let n = n_dim as usize;
                        let use_implicit = conv_bwd_weight_use_implicit_gemm(m, k_dim_usize, n);
                        for g in 0..*groups {
                            let x_g = *x + (g * x_stride_g * 4) as usize;
                            let dy_g = *dy + (g * dy_stride_g * 4) as usize;
                            let dw_g = *dw + (g * dw_stride_g * 4) as usize;
                            if use_implicit {
                                encode_conv2d_bwd_weight_gemm(
                                    e!(),
                                    k,
                                    &self.arena.buffer,
                                    dy_g,
                                    x_g,
                                    dw_g,
                                    m,
                                    k_dim_usize,
                                    n,
                                    &nchw_slice,
                                    &out_dims,
                                    &kshape,
                                    &padd,
                                );
                            } else {
                                let im2col_elems = (n_dim * k_dim) as u64;
                                encode_im2col_group(
                                    e!(),
                                    k,
                                    &self.arena.buffer,
                                    x_g,
                                    self.conv_bwd_scratch_off,
                                    &nchw_slice,
                                    &out_dims,
                                    &kshape,
                                    &padd,
                                    im2col_elems,
                                );
                                crate::blas::metal_sgemm(
                                    e!(),
                                    &self.arena.buffer,
                                    dy_g,
                                    self.conv_bwd_scratch_off,
                                    dw_g,
                                    m,
                                    k_dim_usize,
                                    n,
                                );
                            }
                        }
                    }
                }
                Thunk::MaxPool3dBackward {
                    x,
                    dy,
                    dx,
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
                    let n_eff = scale(*n);
                    if n_eff == 0 {
                        continue;
                    }
                    encode_maxpool3d_backward(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *dy,
                        *dx,
                        n_eff,
                        *c,
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
                    );
                }
                Thunk::Conv3dBackwardInput {
                    dy,
                    w,
                    dx,
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
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_conv3d_backward_input(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *dy,
                        *w,
                        *dx,
                        n,
                        *c_in,
                        *d,
                        *h,
                        *w_in,
                        *c_out,
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
                    );
                }
                Thunk::Conv3dBackwardWeight {
                    x,
                    dy,
                    dw,
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
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_conv3d_backward_weight(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *x,
                        *dy,
                        *dw,
                        n,
                        *c_in,
                        *d,
                        *h,
                        *w,
                        *c_out,
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
                    );
                }
                Thunk::Attention {
                    q,
                    k: kk,
                    v,
                    mask,
                    out,
                    batch,
                    seq,
                    kv_seq,
                    heads,
                    kv_heads,
                    head_dim,
                    v_head_dim,
                    mask_kind,
                    window,
                    dt,
                    kv_f16,
                    bhsd,
                    score_scale,
                    attn_logit_softcap,
                } => {
                    // PLAN L1: split seq into runtime-scaled bound +
                    // compile-time full-extent stride; safe at any batch.
                    let seq_stride = *seq;
                    let kv_stride = *kv_seq;
                    let seq_full = *seq;
                    let kv_seq_full = *kv_seq;
                    let seq = scale(seq_full);
                    // Bucketed decode (`mask_kind == 2`): new K lives at row `upper`
                    // inside a padded `[0..upper]` past buffer. Scaling `kv_seq` down to
                    // `past_seq+1` skips that row; the binary mask handles padding instead.
                    let kv_seq_eff = if *mask_kind == 2 && kv_seq_full != seq_full {
                        kv_seq_full
                    } else {
                        scale(kv_seq_full)
                    };
                    if seq == 0 || kv_seq_eff == 0 {
                        continue;
                    }
                    if rlx_ir::env::flag("RLX_METAL_ATTN_TRACE") {
                        eprintln!(
                            "[metal-attn] dispatch batch={batch} seq={seq} kv_seq={kv_seq_eff} heads={heads} kv_heads={kv_heads} head_dim={head_dim} mask={mask_kind} bhsd={bhsd} scale={score_scale} out_off={out:#x}",
                            batch = batch,
                            heads = heads,
                            kv_heads = kv_heads,
                            head_dim = head_dim,
                            mask_kind = mask_kind,
                            bhsd = bhsd,
                            out = out,
                        );
                    }
                    // NOTE: tap-based bisection (`RLX_TAP_L0=1` on bench_q4_decode) shows
                    // that Op::Attention output reads as exact 0.0 from `graph.outputs`
                    // on Metal even when (1) the kernel IS being dispatched (above trace
                    // confirms) and (2) Q/K/V inputs are clean f32 matching MLX byte-for-
                    // byte. A sentinel `OUT[0] = 12345.5` in the MSL kernel did not show
                    // up at the tap either, ruling out the kernel math and pointing at a
                    // memory-planner / arena-aliasing issue between the Op::Attention GPU
                    // dispatch site and the consumer Op::DequantMatMul (o_proj) when the
                    // latter falls back to deferred-host execution (`RLX_METAL_DEQUANT_
                    // GPU_DISABLE=1`). Task #50 follow-up.
                    let token_bucket = self.sdpa_kernel_plan.token_buckets.bucket_of(kv_seq_eff);
                    let mut launch_builder =
                        crate::kernel_plan::LaunchBuilder::from_plan(&self.sdpa_kernel_plan)
                            .m(seq)
                            .tokens(kv_seq_eff)
                            .batch(*batch)
                            .num_heads(*heads)
                            .head_dim(*head_dim)
                            .seq_q(seq)
                            .seq_kv(kv_seq_eff);
                    if let Some(tuned) = self.sdpa_tuned_candidate(token_bucket) {
                        launch_builder = launch_builder.candidate(tuned);
                    }
                    let launch = launch_builder.build().ok();

                    // Split-KV path guard: must satisfy kernel shape constraints.
                    let flash_ok = matches!(*dt, crate::thunk::HalfFlag::F32)
                        && seq == 1
                        && *head_dim <= 128
                        && (*v_head_dim == 0 || *v_head_dim <= 128);

                    let mut want_flash =
                        launch.as_ref().is_some_and(|l| l.candidate.partial_reduce) && flash_ok;

                    // Config override keeps existing operational behavior.
                    if let Some(v) = crate::runtime_config().sdpa_flash_decode {
                        want_flash = v && flash_ok && kv_seq_eff > 64;
                    }

                    let kernel_name = launch
                        .as_ref()
                        .map(|l| l.kernel_name.as_str())
                        .unwrap_or("sdpa_decode_default");
                    if rlx_ir::env::flag("RLX_METAL_ATTN_TRACE") {
                        eprintln!("[metal-attn] kernel={kernel_name}");
                    }

                    if let Some(l) = &launch {
                        self.sdpa_record_candidate(token_bucket, l.candidate);
                    }

                    if want_flash {
                        let hinted_parts =
                            launch.as_ref().map(|l| l.candidate.split_k).unwrap_or(1);
                        let (tile_n, tile_k, pad_kv) = launch
                            .as_ref()
                            .map(|l| (l.candidate.tile_n, l.candidate.tile_k, l.candidate.pad_kv))
                            .unwrap_or((128, 32, 1));
                        let n_part = sdpa_flash_partitions_tuned(
                            *batch, *heads, kv_seq_eff, tile_n, tile_k, pad_kv,
                        )
                        .max(hinted_parts);
                        const SLOT: u64 = 2 + 128;
                        let need_bytes = (*batch as u64)
                            * (*heads as u64)
                            * (n_part as u64)
                            * SLOT
                            * std::mem::size_of::<f32>() as u64;
                        {
                            let mut sc = self.sdpa_flash_scratch.borrow_mut();
                            let grow = sc.as_ref().map(|b| b.length() < need_bytes).unwrap_or(true);
                            if grow {
                                *sc = Some(dev.device.new_buffer(
                                    need_bytes,
                                    metal::MTLResourceOptions::StorageModeShared,
                                ));
                            }
                        }
                        let sc = self.sdpa_flash_scratch.borrow();
                        let scratch = sc.as_ref().unwrap();
                        // W8A8 decode attention (int8 Q·K integer dot + int8 V):
                        // opt-in via RLX_METAL_W8A8_ATTN, decode-only (seq==1),
                        // head_dim ≤ 128 (guaranteed by flash_ok). ~1.5–1.8× at
                        // long ctx; ~1e-4 attention-output drift (approximation).
                        let use_w8a8 = seq == 1
                            && *head_dim <= 128
                            && (*v_head_dim == 0 || *v_head_dim <= 128)
                            && rlx_ir::env::flag("RLX_METAL_W8A8_ATTN");
                        if use_w8a8 && rlx_ir::env::flag("RLX_METAL_ATTN_TRACE") {
                            eprintln!(
                                "[metal-attn] W8A8 ACTIVE kv_seq={kv_seq_eff} n_part={n_part} kv_f16={kv_f16}"
                            );
                        }
                        if use_w8a8 {
                            let kvh = if *kv_heads == 0 || !heads.is_multiple_of(*kv_heads) {
                                *heads
                            } else {
                                *kv_heads
                            };
                            let vhd = if *v_head_dim == 0 {
                                *head_dim
                            } else {
                                *v_head_dim
                            };
                            let blk = rlx_ir::env::flag("RLX_METAL_W8A8_BLOCK");
                            let nbk = if blk { *head_dim as u64 / 32 } else { 1 };
                            let nbv = if blk { vhd as u64 / 32 } else { 1 };
                            let align256 = |x: u64| (x + 255) & !255u64;
                            let nrows = (*batch as u64) * (kvh as u64) * (kv_seq_eff as u64);
                            let i8v_off = align256(nrows * *head_dim as u64);
                            let ksc_off = align256(i8v_off + nrows * vhd as u64);
                            let vsc_off = align256(ksc_off + nrows * nbk * 4);
                            let need_i8 = align256(vsc_off + nrows * nbv * 4).max(256);
                            {
                                let mut b = self.sdpa_w8a8_scratch.borrow_mut();
                                let grow = b.as_ref().map(|x| x.length() < need_i8).unwrap_or(true);
                                if grow {
                                    *b = Some(dev.device.new_buffer(
                                        need_i8,
                                        metal::MTLResourceOptions::StorageModeShared,
                                    ));
                                }
                            }
                            let i8b = self.sdpa_w8a8_scratch.borrow();
                            let i8scratch = i8b.as_ref().unwrap();
                            encode_sdpa_flash_decode_w8a8(
                                e!(),
                                k,
                                &self.arena.buffer,
                                scratch,
                                i8scratch,
                                n_part,
                                *q,
                                *kk,
                                *v,
                                *mask,
                                *out,
                                *batch,
                                *heads,
                                *kv_heads,
                                *head_dim,
                                *v_head_dim,
                                seq_stride,
                                *mask_kind,
                                *window,
                                kv_seq_eff,
                                kv_stride,
                                *bhsd,
                                *score_scale,
                                *attn_logit_softcap,
                                *kv_f16,
                            );
                        } else {
                            encode_sdpa_flash_decode(
                                e!(),
                                k,
                                &self.arena.buffer,
                                scratch,
                                n_part,
                                *q,
                                *kk,
                                *v,
                                *mask,
                                *out,
                                *batch,
                                *heads,
                                *kv_heads,
                                *head_dim,
                                *v_head_dim,
                                seq_stride,
                                *mask_kind,
                                *window,
                                kv_seq_eff,
                                kv_stride,
                                *bhsd,
                                *score_scale,
                                *attn_logit_softcap,
                                *kv_f16,
                            );
                        }
                    } else {
                        encode_sdpa(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *q,
                            *kk,
                            *v,
                            *mask,
                            *out,
                            *batch,
                            seq,
                            *heads,
                            *kv_heads,
                            *head_dim,
                            *v_head_dim,
                            *dt,
                            seq_stride,
                            *mask_kind,
                            *window,
                            kv_seq_eff,
                            kv_stride,
                            *bhsd,
                            *score_scale,
                            *attn_logit_softcap,
                            *kv_f16,
                        );
                    }
                    // Keep Serial encoder open for following MSL ops (Q1 GEMV /
                    // GDN). Host dequant still flushes via `e!()` / deferred_host.
                }
                Thunk::FusedAttn {
                    qkv,
                    mask,
                    cos,
                    sin,
                    out,
                    batch,
                    seq,
                    heads,
                    head_dim,
                    mask_kind,
                    scale_bits,
                    has_rope,
                } => {
                    let enc = e!();
                    let buffer = &self.arena.buffer;
                    enc.set_compute_pipeline_state(&k.fused_attn_block);
                    // Small FAB scratch (gated to seq ≤ 64): bind each buffer at
                    // its byte offset directly. The >4 GB `set_buffer`-offset
                    // write-drop (task #50) doesn't bite here — these are small,
                    // low arena offsets.
                    enc.set_buffer(0, Some(buffer), *qkv as u64);
                    enc.set_buffer(1, Some(buffer), *mask as u64);
                    enc.set_buffer(2, Some(buffer), *cos as u64);
                    enc.set_buffer(3, Some(buffer), *sin as u64);
                    enc.set_buffer(4, Some(buffer), *out as u64);
                    let u4 = std::mem::size_of::<u32>() as u64;
                    enc.set_bytes(5, u4, batch as *const u32 as *const _);
                    enc.set_bytes(6, u4, seq as *const u32 as *const _);
                    enc.set_bytes(7, u4, heads as *const u32 as *const _);
                    enc.set_bytes(8, u4, head_dim as *const u32 as *const _);
                    enc.set_bytes(9, u4, mask_kind as *const u32 as *const _);
                    enc.set_bytes(10, u4, scale_bits as *const u32 as *const _);
                    enc.set_bytes(11, u4, has_rope as *const u32 as *const _);
                    let groups = (*batch * *heads).max(1) as u64;
                    enc.dispatch_thread_groups(
                        metal::MTLSize {
                            width: groups,
                            height: 1,
                            depth: 1,
                        },
                        metal::MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                Thunk::Rope {
                    src,
                    cos,
                    sin,
                    dst,
                    batch,
                    seq,
                    hidden,
                    head_dim,
                    n_rot,
                    dt,
                    src_row_stride,
                    cos_per_token,
                    interleaved,
                } => {
                    // Active-extent: seq is the runtime-scaled loop bound.
                    // seq_stride stays at compile-time full extent so per-
                    // batch buffer offsets stay correct at any batch.
                    let seq_stride = *seq;
                    let seq = scale(*seq);
                    if seq == 0 {
                        continue;
                    }
                    encode_rope(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *cos,
                        *sin,
                        *dst,
                        *batch,
                        seq,
                        *hidden,
                        *head_dim,
                        *n_rot,
                        *dt,
                        *src_row_stride,
                        seq_stride,
                        *cos_per_token,
                        *interleaved,
                    );
                }
                Thunk::Softmax {
                    data,
                    rows,
                    cols,
                    dt,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_softmax(e!(), k, &self.arena.buffer, *data, rows, *cols, *dt);
                }
                Thunk::SoftmaxCrossEntropyDense {
                    logits,
                    targets,
                    dst,
                    n,
                    c,
                } => {
                    let rows = scale(*n);
                    if rows == 0 {
                        continue;
                    }
                    encode_softmax_cross_entropy_dense(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *logits,
                        *targets,
                        *dst,
                        rows,
                        *c,
                    );
                }
                Thunk::SoftmaxCrossEntropyWithLogits {
                    logits,
                    labels,
                    dst,
                    n,
                    c,
                } => {
                    let rows = scale(*n);
                    if rows == 0 {
                        continue;
                    }
                    encode_softmax_cross_entropy_with_logits(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *logits,
                        *labels,
                        *dst,
                        rows,
                        *c,
                    );
                }
                Thunk::SoftmaxCrossEntropyBackward {
                    logits,
                    labels,
                    d_loss,
                    dlogits,
                    n,
                    c,
                } => {
                    let rows = scale(*n);
                    if rows == 0 {
                        continue;
                    }
                    encode_softmax_cross_entropy_backward(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *logits,
                        *labels,
                        *d_loss,
                        *dlogits,
                        rows,
                        *c,
                    );
                }
                Thunk::Cumsum {
                    src,
                    dst,
                    rows,
                    cols,
                    exclusive,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_cumsum(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        rows,
                        *cols,
                        *exclusive,
                    );
                }
                Thunk::CumScan {
                    src,
                    dst,
                    rows,
                    cols,
                    exclusive,
                    is_max,
                } => {
                    let rows = scale(*rows);
                    if rows == 0 {
                        continue;
                    }
                    encode_cum_scan(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        rows,
                        *cols,
                        *exclusive,
                        *is_max,
                    );
                }
                Thunk::FusedSwiGLU {
                    src,
                    dst,
                    n_half,
                    total,
                    src_dt,
                    dst_dt,
                    gate_first,
                } => {
                    let total = scale(*total);
                    if total == 0 {
                        continue;
                    }
                    encode_fused_swiglu(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        *n_half,
                        total,
                        *src_dt,
                        *dst_dt,
                        *gate_first,
                    );
                }
                Thunk::Concat {
                    dst,
                    outer,
                    dst_axis,
                    inner,
                    dt,
                    inputs,
                    input_dts,
                    weight_const,
                } => {
                    let outer = scale(*outer);
                    if outer == 0 {
                        continue;
                    }
                    // Option A: a concat of constant weights is invariant across
                    // steps — compute once, then skip (fused weight stays put).
                    if *weight_const && rlx_ir::env::flag("RLX_QWEN3_BAKE_WEIGHTS") {
                        if self.baked_weight_concats.borrow().contains(dst) {
                            continue;
                        }
                        self.baked_weight_concats.borrow_mut().insert(*dst);
                    }
                    if metal_host_slices_enabled() && matches!(*dt, crate::thunk::HalfFlag::F32) {
                        if *inner == 1 {
                            deferred_host.push(DeferredHostOp::ConcatLastax {
                                dst: *dst,
                                outer,
                                dst_axis: *dst_axis,
                                segments: inputs.to_vec(),
                            });
                        } else {
                            deferred_host.push(DeferredHostOp::ConcatMidAxis {
                                dst: *dst,
                                outer,
                                dst_axis: *dst_axis,
                                inner: *inner,
                                segments: inputs.to_vec(),
                            });
                        }
                    } else if *inner == 1 {
                        // Last-axis concat — use the existing kernel.
                        encode_concat_lastax(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *dst,
                            outer,
                            *dst_axis,
                            *dt,
                            inputs,
                        );
                    } else if !rlx_ir::env::flag("RLX_METAL_CONCAT_HOST") {
                        // GPU mid-axis concat (default) — encodes into the live
                        // command buffer, no per-concat commit/wait. The host
                        // fallback (opt-in `RLX_METAL_CONCAT_HOST=1`) syncs the
                        // command buffer per concat (~112/step on KV decode);
                        // once the MLP-fusion fix shrank per-step GPU work, that
                        // sync dominates, so keeping the whole step in one
                        // command buffer is the win (decode RTF 21.2 → 19.9).
                        encode_concat_midaxis(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *dst,
                            outer,
                            *dst_axis,
                            *inner,
                            *dt,
                            inputs,
                            input_dts,
                        );
                    } else {
                        // Mid-shape concat — sync + host copy fallback.
                        end_msl!();
                        cmd_buf.commit();
                        cmd_buf.wait_until_completed();
                        let arena_ptr = self.arena.buffer.contents() as *mut u8;
                        let elem = match dt {
                            crate::thunk::HalfFlag::F32 => 4usize,
                            crate::thunk::HalfFlag::F16 => 2usize,
                        };
                        let inner_b = *inner as usize * elem;
                        let dst_axis_total = *dst_axis as usize;
                        unsafe {
                            let dst_base = arena_ptr.add(*dst);
                            for o in 0..outer as usize {
                                let mut axis_off = 0usize;
                                for &(src_off, src_axis) in inputs {
                                    let src_base = arena_ptr.add(src_off);
                                    let src_per_outer = src_axis as usize * inner_b;
                                    let src_row = src_base.add(o * src_per_outer);
                                    let dst_per_outer = dst_axis_total * inner_b;
                                    let dst_row =
                                        dst_base.add(o * dst_per_outer + axis_off * inner_b);
                                    std::ptr::copy_nonoverlapping(src_row, dst_row, src_per_outer);
                                    axis_off += src_axis as usize;
                                }
                            }
                        }
                        cmd_buf = dev.queue.new_command_buffer().to_owned();
                    }
                }
                Thunk::Compare {
                    lhs,
                    rhs,
                    dst,
                    len,
                    op,
                    lhs_scalar,
                    rhs_scalar,
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_compare(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *lhs,
                        *rhs,
                        *dst,
                        len,
                        *op,
                        *lhs_scalar,
                        *rhs_scalar,
                    );
                }
                Thunk::Where {
                    cond,
                    on_true,
                    on_false,
                    dst,
                    len,
                    cond_scalar,
                    true_scalar,
                    false_scalar,
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_where(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *cond,
                        *on_true,
                        *on_false,
                        *dst,
                        len,
                        *cond_scalar,
                        *true_scalar,
                        *false_scalar,
                    );
                }
                Thunk::Fma { a, b, c, dst, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_fma(e!(), k, &self.arena.buffer, *a, *b, *c, *dst, len);
                }
                Thunk::ReluBackward { x, dy, dx, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_relu_backward(e!(), k, &self.arena.buffer, *x, *dy, *dx, len);
                }
                Thunk::ActivationBackward { x, dy, dx, len, op } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_activation_backward(e!(), k, &self.arena.buffer, *x, *dy, *dx, len, *op);
                }
                Thunk::ComplexNormSq { src, dst, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_complex_norm_sq(e!(), k, &self.arena.buffer, *src, *dst, len);
                }
                Thunk::ComplexNormSqBackward { z, g, dz, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_complex_norm_sq_backward(e!(), k, &self.arena.buffer, *z, *g, *dz, len);
                }
                Thunk::ConjugateC64 { src, dst, len } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_conjugate_c64(e!(), k, &self.arena.buffer, *src, *dst, len);
                }
                Thunk::FftButterflyStage {
                    state,
                    out,
                    gate,
                    rev,
                    tw_re,
                    tw_im,
                    batch,
                    n_fft,
                    stage,
                } => {
                    let batch = scale(*batch);
                    if batch == 0 || *n_fft == 0 {
                        continue;
                    }
                    encode_fft_butterfly_stage(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *state,
                        *out,
                        *gate,
                        *rev,
                        *tw_re,
                        *tw_im,
                        batch,
                        *n_fft,
                        *stage,
                    );
                }
                Thunk::FakeQuantizeFixed {
                    src,
                    scale: scale_off,
                    dst,
                    n,
                    chan_dim,
                    inner,
                    q_max,
                } => {
                    let n = scale(*n);
                    let inner = if *chan_dim <= 1 { n.max(1) } else { *inner };
                    if n == 0 {
                        continue;
                    }
                    encode_fake_quantize_fixed(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *scale_off,
                        *dst,
                        n,
                        *chan_dim,
                        inner,
                        *q_max,
                    );
                }
                Thunk::FakeQuantizePerBatch {
                    src,
                    dst,
                    n,
                    chan_dim,
                    inner,
                    q_max,
                } => {
                    let n = scale(*n);
                    let inner = if *chan_dim <= 1 { n.max(1) } else { *inner };
                    if n == 0 {
                        continue;
                    }
                    encode_fake_quantize_perbatch(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        n,
                        *chan_dim,
                        inner,
                        *q_max,
                    );
                }
                Thunk::Reduce {
                    src,
                    dst,
                    outer,
                    reduced,
                    inner,
                    op,
                    dt,
                } => {
                    let outer = scale(*outer);
                    if outer == 0 {
                        continue;
                    }
                    // High-precision path: a full f32 sum-to-scalar reduction
                    // accumulated in double-single (2× f32 ≈ f64), opt-in via
                    // RLX_METAL_DW_SUM. Encoded into the same command buffer —
                    // the pipeline's precise-math compilation is independent of
                    // the (fast-math) main library. Buffers bound at the src/dst
                    // byte offsets, exactly like `encode_reduce_axes`.
                    if crate::double_single::dw_sum_reduce_enabled()
                        && *op == rlx_ir::op::ReduceOp::Sum
                        && outer == 1
                        && *inner == 1
                        && matches!(dt, crate::thunk::HalfFlag::F32)
                    {
                        let e = e!();
                        e.set_compute_pipeline_state(&k.dw_sum_arena);
                        e.set_buffer(0, Some(&self.arena.buffer), *src as u64);
                        e.set_buffer(1, Some(&self.arena.buffer), *dst as u64);
                        let n = *reduced;
                        e.set_bytes(2, 4, &n as *const u32 as *const _);
                        e.dispatch_thread_groups(
                            metal::MTLSize::new(1, 1, 1),
                            metal::MTLSize::new(256, 1, 1),
                        );
                    } else {
                        encode_reduce_axes(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *src,
                            *dst,
                            outer,
                            *reduced,
                            *inner,
                            *op,
                            *dt,
                        );
                    }
                }
                Thunk::TopK {
                    src,
                    dst,
                    outer,
                    axis_dim,
                    k: kk,
                } => {
                    let outer = scale(*outer);
                    if outer == 0 {
                        continue;
                    }
                    encode_topk(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        outer,
                        *axis_dim,
                        *kk,
                    );
                }
                Thunk::GroupedMatMul {
                    input,
                    weight,
                    expert_idx,
                    dst,
                    m,
                    k_dim,
                    n,
                    num_experts,
                } => {
                    let m_scaled = scale(*m);
                    if m_scaled == 0 {
                        continue;
                    }
                    encode_grouped_matmul(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *input,
                        *weight,
                        *expert_idx,
                        *dst,
                        m_scaled,
                        *k_dim,
                        *n,
                        *num_experts,
                    );
                }
                Thunk::ElementwiseRegion {
                    len,
                    num_inputs,
                    num_steps,
                    dst,
                    input_offs,
                    chain,
                    scalar_input_mask,
                    input_modulus,
                    prologue,
                    out_n,
                    out_c,
                    out_h,
                    out_w,
                    prologue_input,
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_elementwise_region(
                        e!(),
                        k,
                        &self.arena.buffer,
                        len,
                        *num_inputs,
                        *num_steps,
                        *dst,
                        input_offs,
                        chain,
                        *scalar_input_mask,
                        input_modulus,
                        *prologue,
                        *out_n,
                        *out_c,
                        *out_h,
                        *out_w,
                        *prologue_input,
                    );
                }
                Thunk::BatchElementwiseRegion {
                    slice_len,
                    num_batch,
                    num_steps,
                    base_dst,
                    slice_elems,
                    batch_input_offs,
                    chain,
                    scalar_input_mask,
                    input_modulus,
                } => {
                    let slice_len = scale(*slice_len);
                    let num_batch = scale(*num_batch);
                    if slice_len == 0 || num_batch == 0 {
                        continue;
                    }
                    encode_batch_elementwise_region(
                        e!(),
                        k,
                        &self.arena.buffer,
                        slice_len,
                        num_batch,
                        *num_steps,
                        *base_dst,
                        *slice_elems,
                        batch_input_offs,
                        chain,
                        *scalar_input_mask,
                        input_modulus,
                    );
                }
                Thunk::ScatterAdd {
                    updates,
                    indices,
                    dst,
                    num_updates,
                    out_dim,
                    trailing,
                } => {
                    // Active-extent on ScatterAdd (CPU-style):
                    //   - Phase 0 zeros FULL output (preserves accumulator semantics)
                    //   - Phase 1 scatters first num_updates_active updates only
                    let num_updates = scale(*num_updates);
                    encode_scatter_add(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *updates,
                        *indices,
                        *dst,
                        num_updates,
                        *out_dim,
                        *trailing,
                    );
                }
                Thunk::Transpose {
                    src,
                    dst,
                    total,
                    out_dims,
                    in_strides,
                    half,
                } => {
                    // Active-extent on Transpose (predicate-vetted
                    // perm[0]==0 via in_strides[0] == product(out_dims[1..])):
                    // scale total by `s_active * inner_product`. Other
                    // transposes fall back to full extent.
                    let inner: u32 = out_dims[1..].iter().product();
                    let total_scaled =
                        if !out_dims.is_empty() && !in_strides.is_empty() && in_strides[0] == inner
                        {
                            scale(out_dims[0]) * inner
                        } else {
                            *total
                        };
                    if total_scaled == 0 {
                        continue;
                    }
                    let is_2d_swap = out_dims.len() == 2
                        && in_strides.len() == 2
                        && in_strides[0] == 1
                        && in_strides[1] == out_dims[0];
                    let last2_batched = detect_last2_batched_swap(out_dims, in_strides);
                    // The specialized swap kernels are f32-only; F16 reindex
                    // (e.g. repeat_kv Expand over an F16 KV cache) always takes
                    // the generic `transpose_nd_h` path.
                    if *half {
                        encode_transpose(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *src,
                            *dst,
                            total_scaled,
                            out_dims,
                            in_strides,
                            true,
                        );
                    } else if is_2d_swap {
                        let rows = out_dims[1];
                        let cols = out_dims[0];
                        if metal_host_slices_enabled() {
                            deferred_host.push(DeferredHostOp::Transpose2d {
                                src: *src,
                                dst: *dst,
                                rows,
                                cols,
                            });
                        } else {
                            encode_transpose_2d(
                                e!(),
                                k,
                                &self.arena.buffer,
                                *src,
                                *dst,
                                rows,
                                cols,
                            );
                        }
                    } else if let Some((batch, rows, cols)) = last2_batched {
                        encode_transpose_last2_batched(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *src,
                            *dst,
                            batch,
                            rows,
                            cols,
                        );
                    } else if let Some((batch, rows, cols, trail)) =
                        detect_swap12_batched_trailing(out_dims, in_strides)
                    {
                        encode_transpose_swap12_batched_trailing(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *src,
                            *dst,
                            batch,
                            rows,
                            cols,
                            trail,
                        );
                    } else {
                        encode_transpose(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *src,
                            *dst,
                            total_scaled,
                            out_dims,
                            in_strides,
                            false,
                        );
                    }
                }
                Thunk::GatherAxis {
                    table,
                    idx,
                    dst,
                    outer,
                    axis_dim,
                    num_idx,
                    trailing,
                } => {
                    let outer = scale(*outer);
                    if outer == 0 {
                        continue;
                    }
                    encode_gather_axis(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *table,
                        *idx,
                        *dst,
                        outer,
                        *axis_dim,
                        *num_idx,
                        *trailing,
                    );
                }
                Thunk::Pool2D {
                    src,
                    dst,
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
                    kind,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_pool2d(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *dst,
                        n,
                        *c,
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
                        *kind,
                    );
                }
                Thunk::Conv2D {
                    src,
                    weight,
                    dst,
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
                    dw,
                    groups,
                } => {
                    let n = scale(*n);
                    if n == 0 {
                        continue;
                    }
                    encode_conv2d(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *src,
                        *weight,
                        *dst,
                        n,
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
                        *dw,
                        *groups,
                    );
                }
                Thunk::CustomOp {
                    kernel,
                    inputs,
                    output,
                    attrs,
                } => {
                    let ingraph_qmatmul =
                        crate::onnx_qmatmul::is_ingraph_qmatmul_kernel(kernel.name())
                            && crate::onnx_qmatmul::ingraph_enabled();
                    if ingraph_qmatmul {
                        let flops = crate::onnx_qmatmul::matmul_flops(inputs, output);
                        let use_gpu = crate::onnx_qmatmul::ingraph_gpu_enabled()
                            && flops >= crate::onnx_qmatmul::gpu_min_flops()
                            && self.onnx_qmatmul_act_scratch_off > 0;
                        if use_gpu {
                            let enc = e!();
                            if kernel.name() == crate::onnx_qmatmul::BAKED_KERNEL_NAME {
                                crate::onnx_qmatmul::encode_onnx_qmatmul_baked_f32_gpu(
                                    enc,
                                    &self.arena.buffer,
                                    self.onnx_qmatmul_act_scratch_off,
                                    inputs,
                                    output,
                                );
                            } else {
                                crate::onnx_qmatmul::encode_onnx_qmatmul_f32_gpu(
                                    enc,
                                    &self.arena.buffer,
                                    self.onnx_qmatmul_act_scratch_off,
                                    &mut self.qmatmul_weight_cache.borrow_mut(),
                                    inputs,
                                    output,
                                );
                            }
                        } else {
                            let arena_ptr = self.arena.buffer.contents() as *mut u8;
                            let in_views: Vec<(&[u8], &rlx_ir::Shape)> = inputs
                                .iter()
                                .map(|(off, len, shape)| {
                                    let n_bytes = (*len as usize) * shape.dtype().size_bytes();
                                    let data: &[u8] = unsafe {
                                        std::slice::from_raw_parts(arena_ptr.add(*off), n_bytes)
                                    };
                                    (data, shape)
                                })
                                .collect();
                            let (out_off, out_len, out_shape) = output;
                            let out_bytes = (*out_len as usize) * out_shape.dtype().size_bytes();
                            let out_data: &mut [u8] = unsafe {
                                std::slice::from_raw_parts_mut(arena_ptr.add(*out_off), out_bytes)
                            };
                            if let Err(e) = kernel.execute(&in_views, (out_data, out_shape), attrs)
                            {
                                panic!(
                                    "rlx-metal: Op::Custom('{}') kernel failed: {e}",
                                    kernel.name()
                                );
                            }
                        }
                    } else {
                        // Op::Custom is a sync point. Encoder is now
                        // owned (refcounted) rather than borrowed from
                        // cmd_buf, so we can flush the current cmd_buf
                        // and rebind it to a fresh one without borrow
                        // conflicts. Sync cost is one queue trip
                        // (wait_until_completed); the host kernel runs
                        // against the unified-memory arena directly —
                        // `Buffer::contents()` is host-accessible for
                        // shared-storage buffers on Apple Silicon, so
                        // there's no copy.
                        end_msl!();
                        cmd_buf.commit();
                        cmd_buf.wait_until_completed();

                        let arena_ptr = self.arena.buffer.contents() as *mut u8;
                        let in_views: Vec<(&[u8], &rlx_ir::Shape)> = inputs
                            .iter()
                            .map(|(off, len, shape)| {
                                let n_bytes = (*len as usize) * shape.dtype().size_bytes();
                                let data: &[u8] = unsafe {
                                    std::slice::from_raw_parts(arena_ptr.add(*off), n_bytes)
                                };
                                (data, shape)
                            })
                            .collect();
                        let (out_off, out_len, out_shape) = output;
                        let out_bytes = (*out_len as usize) * out_shape.dtype().size_bytes();
                        let out_data: &mut [u8] = unsafe {
                            std::slice::from_raw_parts_mut(arena_ptr.add(*out_off), out_bytes)
                        };
                        if let Err(e) = kernel.execute(&in_views, (out_data, out_shape), attrs) {
                            panic!(
                                "rlx-metal: Op::Custom('{}') kernel failed: {e}",
                                kernel.name()
                            );
                        }

                        // Fresh cmd_buf for subsequent thunks. The outer
                        // function's final `cmd_buf.commit()` will commit
                        // this one (containing whatever ops follow, or
                        // empty if Op::Custom was the trailing thunk).
                        cmd_buf = dev.queue.new_command_buffer().to_owned();
                    }
                }

                Thunk::CustomGpuOp {
                    kernel,
                    inputs,
                    output,
                    attrs,
                } => {
                    // Raw-GPU custom op: encode straight onto the active compute
                    // encoder — no end_msl!/commit/wait and no cmd_buf swap, so
                    // subsequent thunks keep encoding on the same command buffer.
                    // Metal's automatic hazard tracking on the shared arena buffer
                    // orders this dispatch against neighbours in the serial
                    // encoder.
                    let enc = e!();
                    let d = crate::op_registry::MetalGpuDispatch {
                        encoder: enc,
                        arena: &self.arena.buffer,
                        inputs,
                        output,
                        attrs,
                    };
                    if let Err(err) = kernel.encode(&d) {
                        panic!(
                            "rlx-metal: Op::Custom('{}') GPU kernel failed: {err}",
                            kernel.name()
                        );
                    }
                }

                Thunk::SpdHost { op, inputs, output } => {
                    // Same sync pattern as Thunk::Fft1d / Thunk::CustomOp: flush
                    // the GPU, run the CPU SPD reference against the
                    // unified-memory arena, restart cmd_buf. The arena is a
                    // shared-storage MTLBuffer on Apple Silicon, so
                    // `contents()` is host-addressable — no copies. The SPD
                    // subgraph's tensors are stored as f32 in the arena (widened
                    // at planning time); `crate::spd::eval` widens f32→f64,
                    // runs `rlx_cpu::spd`, and narrows back.
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();

                    let arena_ptr = self.arena.buffer.contents() as *const u8;
                    // Read each operand's f32 span at its byte offset. Shapes
                    // carry the REAL declared F64 dtype for the CPU thunk graph.
                    let in_specs: Vec<(rlx_ir::Shape, Vec<f32>)> = inputs
                        .iter()
                        .map(|(off, len, shape)| {
                            let n = *len as usize;
                            let vals = unsafe {
                                std::slice::from_raw_parts(arena_ptr.add(*off) as *const f32, n)
                                    .to_vec()
                            };
                            (shape.clone(), vals)
                        })
                        .collect();
                    let (out_off, out_len, out_shape) = output;
                    let y = crate::spd::eval(op, out_shape, &in_specs);
                    let m = (*out_len as usize).min(y.len());
                    unsafe {
                        let dst =
                            (self.arena.buffer.contents() as *mut u8).add(*out_off) as *mut f32;
                        std::ptr::copy_nonoverlapping(y.as_ptr(), dst, m);
                    }

                    // Fresh cmd_buf for subsequent thunks (mirrors CustomOp).
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::GaussianSplatRender {
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
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        #[cfg(all(feature = "native-splat", target_os = "macos"))]
                        {
                            crate::splat_native::execute_gaussian_splat_render_native(
                                *positions_off,
                                *positions_len,
                                *scales_off,
                                *scales_len,
                                *rotations_off,
                                *rotations_len,
                                *opacities_off,
                                *opacities_len,
                                *colors_off,
                                *colors_len,
                                *sh_coeffs_off,
                                *sh_coeffs_len,
                                *meta_off,
                                *dst_off,
                                *dst_len,
                                *width,
                                *height,
                                *tile_size,
                                *radius_scale,
                                *alpha_cutoff,
                                *max_splat_steps,
                                *transmittance_threshold,
                                *max_list_entries,
                                arena_ptr,
                                &self.arena.buffer,
                            );
                        }
                        #[cfg(not(all(feature = "native-splat", target_os = "macos")))]
                        rlx_cpu::splat::execute_gaussian_splat_render(
                            *positions_off,
                            *positions_len,
                            *scales_off,
                            *scales_len,
                            *rotations_off,
                            *rotations_len,
                            *opacities_off,
                            *opacities_len,
                            *colors_off,
                            *colors_len,
                            *sh_coeffs_off,
                            *sh_coeffs_len,
                            *meta_off,
                            *dst_off,
                            *dst_len,
                            *width,
                            *height,
                            *tile_size,
                            *radius_scale,
                            *alpha_cutoff,
                            *max_splat_steps,
                            *transmittance_threshold,
                            *max_list_entries,
                            arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::GaussianSplatRenderBackward {
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
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::splat::execute_gaussian_splat_render_backward(
                            *positions_off,
                            *positions_len,
                            *scales_off,
                            *scales_len,
                            *rotations_off,
                            *rotations_len,
                            *opacities_off,
                            *opacities_len,
                            *colors_off,
                            *colors_len,
                            *sh_coeffs_off,
                            *sh_coeffs_len,
                            *meta_off,
                            *d_loss_off,
                            *d_loss_len,
                            *packed_off,
                            *packed_len,
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
                            arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::GaussianSplatPrepare {
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
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::splat::execute_gaussian_splat_prepare(
                            *positions_off,
                            *positions_len,
                            *scales_off,
                            *scales_len,
                            *rotations_off,
                            *rotations_len,
                            *opacities_off,
                            *opacities_len,
                            *colors_off,
                            *colors_len,
                            *sh_coeffs_off,
                            *sh_coeffs_len,
                            *meta_off,
                            *meta_len,
                            *prep_off,
                            *prep_len,
                            *width,
                            *height,
                            *tile_size,
                            *radius_scale,
                            *alpha_cutoff,
                            *max_splat_steps,
                            *transmittance_threshold,
                            *max_list_entries,
                            arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::GaussianSplatRasterize {
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
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        #[cfg(all(feature = "native-splat", target_os = "macos"))]
                        {
                            crate::splat_native::execute_gaussian_splat_rasterize_native(
                                *prep_off,
                                *prep_len,
                                *meta_off,
                                *meta_len,
                                *dst_off,
                                *dst_len,
                                *count,
                                *width,
                                *height,
                                *tile_size,
                                *alpha_cutoff,
                                *max_splat_steps,
                                *transmittance_threshold,
                                *max_list_entries,
                                arena_ptr,
                                &self.arena.buffer,
                            );
                        }
                        #[cfg(not(all(feature = "native-splat", target_os = "macos")))]
                        rlx_cpu::splat::execute_gaussian_splat_rasterize(
                            *prep_off,
                            *prep_len,
                            *meta_off,
                            *meta_len,
                            *dst_off,
                            *dst_len,
                            *count,
                            *width,
                            *height,
                            *tile_size,
                            *alpha_cutoff,
                            *max_splat_steps,
                            *transmittance_threshold,
                            *max_list_entries,
                            arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::AxialRope2dHost {
                    src,
                    dst,
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
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::thunk::execute_axial_rope2d_f32(
                            *src,
                            *dst,
                            *batch as usize,
                            *seq as usize,
                            *hidden as usize,
                            *end_x as usize,
                            *end_y as usize,
                            *head_dim as usize,
                            *num_heads as usize,
                            *theta,
                            *repeat_factor as usize,
                            arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::Fft1d {
                    src,
                    dst,
                    outer,
                    n_complex,
                    inverse,
                    norm_tag,
                    dtype,
                    real_input,
                } => {
                    // Native multi-kernel MSL path: f32 + power-of-2 N≥2.
                    // f64/C64 and non-pow2 fall through to host CPU FFT.
                    // Set RLX_METAL_FFT_HOST_FALLBACK=1 to force host path.
                    let force_host = rlx_ir::env::flag("RLX_METAL_FFT_HOST_FALLBACK");
                    let n = *n_complex as usize;
                    // `real_input` (fused real→complex) requires the native path —
                    // `src` is the n-wide signal, which the host FFT can't read —
                    // so it overrides the debug host-fallback flag.
                    let can_native = (*real_input || !force_host)
                        && matches!(dtype, rlx_ir::DType::F32)
                        && n.is_power_of_two()
                        && n >= 2;
                    if can_native {
                        let enc = e!();
                        let norm = rlx_ir::fft::FftNorm::from_tag(*norm_tag);
                        let norm_scale = norm.output_scale(n, *inverse) as f32;
                        crate::fft_dispatch::run_fft_gpu(
                            k,
                            enc,
                            &self.arena.buffer,
                            (*src as u64 / 4) as u32,
                            (*dst as u64 / 4) as u32,
                            *outer,
                            n as u32,
                            *inverse,
                            norm_scale,
                            *real_input,
                        );
                    } else {
                        // Host fallback — same sync pattern as
                        // Thunk::CustomOp: flush the GPU, run the
                        // kernel against the unified-memory arena,
                        // restart cmd_buf. No copies on Apple Silicon
                        // (shared-storage buffer is host-addressable).
                        end_msl!();
                        cmd_buf.commit();
                        cmd_buf.wait_until_completed();
                        let arena_ptr = self.arena.buffer.contents() as *mut u8;
                        unsafe {
                            match dtype {
                                rlx_ir::DType::F32 => rlx_cpu::thunk::execute_fft1d_f32(
                                    *src,
                                    *dst,
                                    *outer as usize,
                                    n,
                                    *inverse,
                                    *norm_tag,
                                    arena_ptr,
                                ),
                                rlx_ir::DType::F64 => rlx_cpu::thunk::execute_fft1d_f64(
                                    *src,
                                    *dst,
                                    *outer as usize,
                                    n,
                                    *inverse,
                                    *norm_tag,
                                    arena_ptr,
                                ),
                                rlx_ir::DType::C64 => rlx_cpu::thunk::execute_fft1d_c64(
                                    *src,
                                    *dst,
                                    *outer as usize,
                                    n,
                                    *inverse,
                                    *norm_tag,
                                    arena_ptr,
                                ),
                                other => panic!(
                                    "rlx-metal Op::Fft host fallback: unsupported dtype {other:?}"
                                ),
                            }
                        }
                        cmd_buf = dev.queue.new_command_buffer().to_owned();
                    }
                }

                Thunk::VqAssign {
                    x,
                    cb,
                    out,
                    n,
                    d,
                    k: k_codes,
                    metric,
                } => {
                    // On-GPU fused nearest-code: one threadgroup per row.
                    let (nn, dd, kk, mm) = (*n, *d, *k_codes, *metric);
                    let enc = e!();
                    enc.set_compute_pipeline_state(&k.vq_assign);
                    enc.set_buffer(0, Some(&self.arena.buffer), *x as u64);
                    enc.set_buffer(1, Some(&self.arena.buffer), *cb as u64);
                    enc.set_buffer(2, Some(&self.arena.buffer), *out as u64);
                    enc.set_bytes(3, 4, &nn as *const u32 as *const _);
                    enc.set_bytes(4, 4, &dd as *const u32 as *const _);
                    enc.set_bytes(5, 4, &kk as *const u32 as *const _);
                    enc.set_bytes(6, 4, &mm as *const u32 as *const _);
                    let grid = metal::MTLSize {
                        width: nn as u64,
                        height: 1,
                        depth: 1,
                    };
                    let tg = metal::MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    };
                    enc.dispatch_thread_groups(grid, tg);
                }

                Thunk::ScanHost { desc } => {
                    // Same host-fallback sync pattern as Fft1d: flush the GPU,
                    // run the compiled scan body loop on the CPU against the
                    // unified-memory arena, restart cmd_buf.
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::rlx_execute_scan_on_bytes!(arena_ptr, desc);
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }
                Thunk::HostOp { desc } => {
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::rlx_execute_host_op_on_bytes!(arena_ptr, desc);
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }
                Thunk::CpuIndexing { thunk } => {
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    check_cmd_buf_status(&cmd_buf, "CpuIndexing (scatter/gather host read)");
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::rlx_execute_indexing_on_bytes!(arena_ptr, thunk.inner());
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::LogMel {
                    spec,
                    filters,
                    dst,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::thunk::execute_log_mel_f32(
                            *spec,
                            *filters,
                            *dst,
                            *outer as usize,
                            *n_fft as usize,
                            *n_bins as usize,
                            *n_mels as usize,
                            arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::LogMelBackward {
                    spec,
                    filters,
                    dy,
                    dst,
                    outer,
                    n_fft,
                    n_bins,
                    n_mels,
                } => {
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::thunk::execute_log_mel_backward_f32(
                            *spec,
                            *filters,
                            *dy,
                            *dst,
                            *outer as usize,
                            *n_fft as usize,
                            *n_bins as usize,
                            *n_mels as usize,
                            arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::WelchPeaks {
                    spec,
                    dst,
                    welch_batch,
                    n_fft,
                    n_segments,
                    k,
                } => {
                    tail_host.push(TailHostOp::WelchPeaks {
                        spec: *spec,
                        dst: *dst,
                        welch_batch: *welch_batch,
                        n_fft: *n_fft,
                        n_segments: *n_segments,
                        k: *k,
                    });
                }

                Thunk::RngNormal {
                    dst,
                    len,
                    mean,
                    scale,
                    key,
                    op_seed,
                } => {
                    let opts = *self.schedule.rng.read().unwrap();
                    match opts.backend {
                        // On-device Philox / Zero — encoded inline in the MSL
                        // batch (no commit / sync / CPU-fill bubble).
                        rlx_ir::RngBackend::Philox | rlx_ir::RngBackend::Zero => {
                            let enc = e!();
                            let len_u = *len;
                            let zero = matches!(opts.backend, rlx_ir::RngBackend::Zero);
                            let pipe = if zero {
                                &k.rng_fill_zero
                            } else {
                                &k.rng_normal_philox
                            };
                            enc.set_compute_pipeline_state(pipe);
                            enc.set_buffer(0, Some(&self.arena.buffer), *dst as u64);
                            enc.set_bytes(1, 4, &len_u as *const u32 as *const _);
                            if !zero {
                                let seed = rlx_ir::combine_seed(opts.seed, *key);
                                let seed_lo = (seed & 0xFFFF_FFFF) as u32;
                                let seed_hi = (seed >> 32) as u32;
                                enc.set_bytes(2, 4, mean as *const f32 as *const _);
                                enc.set_bytes(3, 4, scale as *const f32 as *const _);
                                enc.set_bytes(4, 4, &seed_lo as *const u32 as *const _);
                                enc.set_bytes(5, 4, &seed_hi as *const u32 as *const _);
                            }
                            let grid = metal::MTLSize {
                                width: len_u.max(1) as u64,
                                height: 1,
                                depth: 1,
                            };
                            let tg_w = pipe.thread_execution_width().min(len_u.max(1) as u64);
                            let tg = metal::MTLSize {
                                width: tg_w,
                                height: 1,
                                depth: 1,
                            };
                            enc.dispatch_threads(grid, tg);
                        }
                        // Ort / Bnns parity streams: unified-memory host fill.
                        rlx_ir::RngBackend::Ort | rlx_ir::RngBackend::Bnns => {
                            end_msl!();
                            cmd_buf.commit();
                            cmd_buf.wait_until_completed();
                            let arena_ptr = self.arena.buffer.contents() as *mut u8;
                            unsafe {
                                rlx_cpu::thunk::fill_rng_normal_arena(
                                    *dst,
                                    *len as usize,
                                    *mean,
                                    *scale,
                                    *key,
                                    *op_seed,
                                    opts,
                                    arena_ptr,
                                );
                            }
                            cmd_buf = dev.queue.new_command_buffer().to_owned();
                        }
                    }
                }

                Thunk::RngUniform {
                    dst,
                    len,
                    low,
                    high,
                    key,
                    op_seed,
                } => {
                    let opts = *self.schedule.rng.read().unwrap();
                    match opts.backend {
                        rlx_ir::RngBackend::Philox | rlx_ir::RngBackend::Zero => {
                            let enc = e!();
                            let len_u = *len;
                            let zero = matches!(opts.backend, rlx_ir::RngBackend::Zero);
                            let pipe = if zero {
                                &k.rng_fill_zero
                            } else {
                                &k.rng_uniform_philox
                            };
                            enc.set_compute_pipeline_state(pipe);
                            enc.set_buffer(0, Some(&self.arena.buffer), *dst as u64);
                            enc.set_bytes(1, 4, &len_u as *const u32 as *const _);
                            if !zero {
                                let seed = rlx_ir::combine_seed(opts.seed, *key);
                                let seed_lo = (seed & 0xFFFF_FFFF) as u32;
                                let seed_hi = (seed >> 32) as u32;
                                enc.set_bytes(2, 4, low as *const f32 as *const _);
                                enc.set_bytes(3, 4, high as *const f32 as *const _);
                                enc.set_bytes(4, 4, &seed_lo as *const u32 as *const _);
                                enc.set_bytes(5, 4, &seed_hi as *const u32 as *const _);
                            }
                            let grid = metal::MTLSize {
                                width: len_u.max(1) as u64,
                                height: 1,
                                depth: 1,
                            };
                            let tg_w = pipe.thread_execution_width().min(len_u.max(1) as u64);
                            let tg = metal::MTLSize {
                                width: tg_w,
                                height: 1,
                                depth: 1,
                            };
                            enc.dispatch_threads(grid, tg);
                        }
                        rlx_ir::RngBackend::Ort | rlx_ir::RngBackend::Bnns => {
                            end_msl!();
                            cmd_buf.commit();
                            cmd_buf.wait_until_completed();
                            let arena_ptr = self.arena.buffer.contents() as *mut u8;
                            unsafe {
                                rlx_cpu::thunk::fill_rng_uniform_arena(
                                    *dst,
                                    *len as usize,
                                    *low,
                                    *high,
                                    *key,
                                    *op_seed,
                                    opts,
                                    arena_ptr,
                                );
                            }
                            cmd_buf = dev.queue.new_command_buffer().to_owned();
                        }
                    }
                }

                Thunk::Im2Col {
                    x,
                    col,
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
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::im2col::execute_im2col_rows_layout(
                            *x, *col, *n, *c_in, *h, *w, *h_out, *w_out, *kh, *kw, *sh, *sw, *ph,
                            *pw, *dh, *dw_dil, arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
                }

                Thunk::GatedDeltaNet {
                    q,
                    k: k_off,
                    v,
                    g,
                    beta,
                    state,
                    dst,
                    batch,
                    seq,
                    heads,
                    state_size,
                    f16,
                    gate_per_channel,
                    carry_state,
                } => {
                    // Native MSL GDN (one thread per head). Opt out with
                    // RLX_METAL_GDN_HOST_FALLBACK=1 / RLX_METAL_GDN_CPU=1.
                    let force_host = rlx_ir::env::flag("RLX_METAL_GDN_HOST_FALLBACK")
                        || rlx_ir::env::flag("RLX_METAL_GDN_CPU");
                    let prefer_cpu_blas = false;
                    let use_carry = *carry_state;
                    // Prefill (`!use_carry`): native MSL needs an ephemeral scratch
                    // slot (zeroed inside the kernel). Host CPU GDN treats any
                    // nonzero `state` as a live carry — reusing the shared scratch
                    // would leak layer N's final SSM into layer N+1 prefill.
                    let state_byte = if use_carry {
                        *state
                    } else if force_host || prefer_cpu_blas || *f16 || *state_size > 128 {
                        0
                    } else {
                        self.gdn_scratch_off
                    };
                    let can_native = !force_host
                        && !prefer_cpu_blas
                        && !*f16
                        && *state_size <= 128
                        && (!use_carry || state_byte != 0);
                    if can_native {
                        let enc = e!();
                        encode_gated_delta_net(
                            enc,
                            k,
                            &self.arena.buffer,
                            *q,
                            *k_off,
                            *v,
                            *g,
                            *beta,
                            state_byte,
                            *dst,
                            *batch,
                            *seq,
                            *heads,
                            *state_size,
                            use_carry,
                            *gate_per_channel,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::GatedDeltaNet {
                            q: *q,
                            k_off: *k_off,
                            v: *v,
                            g: *g,
                            beta: *beta,
                            state_byte,
                            dst: *dst,
                            batch: *batch,
                            seq: *seq,
                            heads: *heads,
                            state_size: *state_size,
                            f16: *f16,
                            gate_per_channel: *gate_per_channel,
                            carry_state: *carry_state,
                        });
                    }
                }

                Thunk::SelectiveScan {
                    x,
                    delta,
                    a,
                    b,
                    c,
                    dst,
                    batch,
                    seq,
                    hidden,
                    state_size,
                } => {
                    // Native MSL kernel covers f32 with state_size ≤ 128
                    // (SSM_MAX_N in kernels.rs). Larger state or the
                    // RLX_METAL_SSM_HOST_FALLBACK / RLX_METAL_SSM_CPU opt-out
                    // take the verified CPU kernel on the unified-memory arena.
                    let force_host = rlx_ir::env::flag("RLX_METAL_SSM_HOST_FALLBACK")
                        || rlx_ir::env::flag("RLX_METAL_SSM_CPU");
                    if !force_host && *state_size <= 128 {
                        let enc = e!();
                        encode_selective_scan(
                            enc,
                            k,
                            &self.arena.buffer,
                            *x,
                            *delta,
                            *a,
                            *b,
                            *c,
                            *dst,
                            *batch,
                            *seq,
                            *hidden,
                            *state_size,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::SelectiveScan {
                            x: *x,
                            delta: *delta,
                            a: *a,
                            b: *b,
                            c: *c,
                            dst: *dst,
                            batch: *batch,
                            seq: *seq,
                            hidden: *hidden,
                            state_size: *state_size,
                        });
                    }
                }

                Thunk::Sample {
                    logits,
                    dst,
                    batch,
                    vocab,
                    top_k,
                    top_p,
                    temperature,
                    seed,
                } => {
                    // Native on-GPU sample: temperature -> top-k -> softmax ->
                    // top-p -> Philox inverse-CDF, one threadgroup per batch
                    // row (`sample_logits`). Matches the CPU `sample_row`
                    // algorithm and Philox stream bit-for-bit. Falls back to the
                    // host kernel only when a deferred host op precedes it in
                    // this segment (ordering) or via RLX_METAL_SAMPLE_HOST=1.
                    if deferred_host.is_empty() && !rlx_ir::env::flag("RLX_METAL_SAMPLE_HOST") {
                        let enc = e!();
                        enc.set_compute_pipeline_state(&k.sample_logits);
                        enc.set_buffer(0, Some(&self.arena.buffer), 0);
                        let lg = *logits as u64;
                        enc.set_bytes(1, 8, &lg as *const u64 as *const _);
                        let ds = *dst as u64;
                        enc.set_bytes(2, 8, &ds as *const u64 as *const _);
                        let (bt, vc, tk) = (*batch, *vocab, *top_k);
                        enc.set_bytes(3, 4, &bt as *const u32 as *const _);
                        enc.set_bytes(4, 4, &vc as *const u32 as *const _);
                        enc.set_bytes(5, 4, &tk as *const u32 as *const _);
                        let tp = *top_p;
                        enc.set_bytes(6, 4, &tp as *const f32 as *const _);
                        let tm = *temperature;
                        enc.set_bytes(7, 4, &tm as *const f32 as *const _);
                        let sd = *seed;
                        enc.set_bytes(8, 8, &sd as *const u64 as *const _);
                        let grid = metal::MTLSize {
                            width: bt as u64,
                            height: 1,
                            depth: 1,
                        };
                        let tg = metal::MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        };
                        enc.dispatch_thread_groups(grid, tg);
                    } else {
                        deferred_host.push(DeferredHostOp::Sample {
                            logits: *logits,
                            dst: *dst,
                            batch: *batch,
                            vocab: *vocab,
                            top_k: *top_k,
                            top_p: *top_p,
                            temperature: *temperature,
                            seed: *seed,
                        });
                    }
                }

                Thunk::Reverse {
                    src,
                    dst,
                    dims,
                    rev_mask,
                    elem_bytes,
                } => {
                    deferred_host.push(DeferredHostOp::Reverse {
                        src: *src,
                        dst: *dst,
                        dims: dims.clone(),
                        rev_mask: rev_mask.clone(),
                        elem_bytes: *elem_bytes,
                    });
                }

                Thunk::Pad {
                    src,
                    dst,
                    in_dims,
                    before,
                    after,
                    mode,
                    fill,
                    elem_bytes,
                } => {
                    deferred_host.push(DeferredHostOp::Pad {
                        src: *src,
                        dst: *dst,
                        in_dims: in_dims.clone(),
                        before: before.clone(),
                        after: after.clone(),
                        mode: *mode,
                        fill: fill.clone(),
                        elem_bytes: *elem_bytes,
                    });
                }

                Thunk::Slice {
                    src,
                    dst,
                    in_dims,
                    axis,
                    start,
                    len,
                    step,
                    elem_bytes,
                } => {
                    deferred_host.push(DeferredHostOp::Slice {
                        src: *src,
                        dst: *dst,
                        in_dims: in_dims.clone(),
                        axis: *axis,
                        start: *start,
                        len: *len,
                        step: *step,
                        elem_bytes: *elem_bytes,
                    });
                }

                Thunk::ArgReduce {
                    src,
                    dst,
                    outer,
                    reduced,
                    inner,
                    is_max,
                } => {
                    // Native GPU dispatch when this op's inputs are all
                    // GPU-produced (no host op queued before it in this
                    // segment). If a deferred host op precedes it, fall back to
                    // host so the end-of-segment ordering stays correct.
                    if deferred_host.is_empty() {
                        let (o, r, inn, im) = (*outer, *reduced, *inner, *is_max as u32);
                        if inn == 1 {
                            // Cooperative last-axis reduction — one threadgroup
                            // folds the whole row (decode: vocab ~128k on a
                            // single lane via the serial kernel was the stall).
                            let enc = e!();
                            enc.set_compute_pipeline_state(&k.argreduce_lastaxis);
                            enc.set_buffer(0, Some(&self.arena.buffer), *src as u64);
                            enc.set_buffer(1, Some(&self.arena.buffer), *dst as u64);
                            enc.set_bytes(2, 4, &o as *const u32 as *const _);
                            enc.set_bytes(3, 4, &r as *const u32 as *const _);
                            enc.set_bytes(4, 4, &im as *const u32 as *const _);
                            let grid = metal::MTLSize {
                                width: o as u64,
                                height: 1,
                                depth: 1,
                            };
                            let tg = metal::MTLSize {
                                width: 256,
                                height: 1,
                                depth: 1,
                            };
                            enc.dispatch_thread_groups(grid, tg);
                        } else {
                            let enc = e!();
                            enc.set_compute_pipeline_state(&k.argreduce);
                            enc.set_buffer(0, Some(&self.arena.buffer), *src as u64);
                            enc.set_buffer(1, Some(&self.arena.buffer), *dst as u64);
                            enc.set_bytes(2, 4, &o as *const u32 as *const _);
                            enc.set_bytes(3, 4, &r as *const u32 as *const _);
                            enc.set_bytes(4, 4, &inn as *const u32 as *const _);
                            enc.set_bytes(5, 4, &im as *const u32 as *const _);
                            let total = (o * inn) as u64;
                            let grid = metal::MTLSize {
                                width: total,
                                height: 1,
                                depth: 1,
                            };
                            let tg = metal::MTLSize {
                                width: total.min(256),
                                height: 1,
                                depth: 1,
                            };
                            enc.dispatch_threads(grid, tg);
                        }
                    } else {
                        deferred_host.push(DeferredHostOp::ArgReduce {
                            src: *src,
                            dst: *dst,
                            outer: *outer,
                            reduced: *reduced,
                            inner: *inner,
                            is_max: *is_max,
                        });
                    }
                }

                Thunk::Lstm {
                    x,
                    w_ih,
                    w_hh,
                    bias,
                    h0,
                    c0,
                    dst,
                    batch,
                    seq,
                    input_size,
                    hidden,
                    num_layers,
                    bidirectional,
                    carry,
                } => {
                    // Native MSL for any layers / dirs / carry, hidden ≤ 1024 (=
                    // max threads/threadgroup; multi-layer ping-pongs through the
                    // in-arena scratch pair). Opt out via RLX_METAL_LSTM_CPU=1 /
                    // RLX_METAL_LSTM_HOST_FALLBACK=1.
                    let force_host = rlx_ir::env::flag("RLX_METAL_LSTM_HOST_FALLBACK")
                        || rlx_ir::env::flag("RLX_METAL_LSTM_CPU");
                    if !force_host && *hidden <= 1024 {
                        let enc = e!();
                        encode_lstm(
                            enc,
                            k,
                            &self.arena.buffer,
                            self.rnn_gru_scratch_off,
                            *x,
                            *w_ih,
                            *w_hh,
                            *bias,
                            *h0,
                            *c0,
                            *dst,
                            *batch,
                            *seq,
                            *input_size,
                            *hidden,
                            *num_layers,
                            *bidirectional,
                            *carry,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::Lstm {
                            x: *x,
                            w_ih: *w_ih,
                            w_hh: *w_hh,
                            bias: *bias,
                            h0: *h0,
                            c0: *c0,
                            dst: *dst,
                            batch: *batch,
                            seq: *seq,
                            input_size: *input_size,
                            hidden: *hidden,
                            num_layers: *num_layers,
                            bidirectional: *bidirectional,
                            carry: *carry,
                        });
                    }
                }

                Thunk::Gru {
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
                    // Native MSL for any layers / dirs / carry, hidden ≤ 1024
                    // (multi-layer ping-pongs through the in-arena scratch pair).
                    let force_host = rlx_ir::env::flag("RLX_METAL_RNN_HOST_FALLBACK");
                    if !force_host && *hidden <= 1024 {
                        encode_gru(
                            e!(),
                            k,
                            &self.arena.buffer,
                            self.rnn_gru_scratch_off,
                            *x,
                            *w_ih,
                            *w_hh,
                            *b_ih,
                            *b_hh,
                            *h0,
                            *dst,
                            *batch,
                            *seq,
                            *input_size,
                            *hidden,
                            *num_layers,
                            *bidirectional,
                            *carry,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::Gru {
                            x: *x,
                            w_ih: *w_ih,
                            w_hh: *w_hh,
                            b_ih: *b_ih,
                            b_hh: *b_hh,
                            h0: *h0,
                            dst: *dst,
                            batch: *batch,
                            seq: *seq,
                            input_size: *input_size,
                            hidden: *hidden,
                            num_layers: *num_layers,
                            bidirectional: *bidirectional,
                            carry: *carry,
                        });
                    }
                }

                Thunk::Rnn {
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
                    let force_host = rlx_ir::env::flag("RLX_METAL_RNN_HOST_FALLBACK");
                    if !force_host && *hidden <= 1024 {
                        encode_rnn(
                            e!(),
                            k,
                            &self.arena.buffer,
                            self.rnn_gru_scratch_off,
                            *x,
                            *w_ih,
                            *w_hh,
                            *bias,
                            *h0,
                            *dst,
                            *batch,
                            *seq,
                            *input_size,
                            *hidden,
                            *num_layers,
                            *bidirectional,
                            *carry,
                            *relu,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::Rnn {
                            x: *x,
                            w_ih: *w_ih,
                            w_hh: *w_hh,
                            bias: *bias,
                            h0: *h0,
                            dst: *dst,
                            batch: *batch,
                            seq: *seq,
                            input_size: *input_size,
                            hidden: *hidden,
                            num_layers: *num_layers,
                            bidirectional: *bidirectional,
                            carry: *carry,
                            relu: *relu,
                        });
                    }
                }

                Thunk::Mamba2 {
                    x,
                    dt,
                    a,
                    b,
                    c,
                    dst,
                    batch,
                    seq,
                    heads,
                    head_dim,
                    state_size,
                } => {
                    // Native MSL for state_size ≤ 128 (MAMBA2_MAX_N).
                    let force_host = rlx_ir::env::flag("RLX_METAL_SSM_HOST_FALLBACK")
                        || rlx_ir::env::flag("RLX_METAL_SSM_CPU");
                    if !force_host && *state_size <= 128 {
                        encode_mamba2(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *x,
                            *dt,
                            *a,
                            *b,
                            *c,
                            *dst,
                            *batch,
                            *seq,
                            *heads,
                            *head_dim,
                            *state_size,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::Mamba2 {
                            x: *x,
                            dt: *dt,
                            a: *a,
                            b: *b,
                            c: *c,
                            dst: *dst,
                            batch: *batch,
                            seq: *seq,
                            heads: *heads,
                            head_dim: *head_dim,
                            state_size: *state_size,
                        });
                    }
                }

                Thunk::SynthMatMulBackward {
                    x,
                    indices,
                    codebook,
                    upstream,
                    dst,
                    m,
                    n,
                    k: k_dim,
                    entry_dim,
                    num_entries,
                    dx,
                } => {
                    let enc = e!();
                    encode_synth_bwd(
                        enc,
                        k,
                        &self.arena.buffer,
                        *x,
                        *indices,
                        *codebook,
                        *upstream,
                        *dst,
                        *m,
                        *n,
                        *k_dim,
                        *entry_dim,
                        *num_entries,
                        *dx,
                    );
                }

                Thunk::SplineActivation {
                    x,
                    coeff,
                    dst,
                    rows,
                    channels,
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let enc = e!();
                    let total = (*rows) * (*channels);
                    encode_spline_activation(
                        enc,
                        k,
                        &self.arena.buffer,
                        *x,
                        *coeff,
                        *dst,
                        total,
                        *channels,
                        *num_basis,
                        *grid_min,
                        *grid_max,
                    );
                }

                Thunk::SplineActivationBackwardX {
                    x,
                    coeff,
                    upstream,
                    dst,
                    rows,
                    channels,
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let enc = e!();
                    let total = (*rows) * (*channels);
                    enc.set_compute_pipeline_state(&k.spline_activation_backward_x);
                    enc.set_buffer(0, Some(&self.arena.buffer), 0);
                    for (i, v) in [*x as u64, *coeff as u64, *upstream as u64, *dst as u64]
                        .iter()
                        .enumerate()
                    {
                        enc.set_bytes((i + 1) as u64, 8, v as *const u64 as *const _);
                    }
                    enc.set_bytes(5, 4, &total as *const u32 as *const _);
                    enc.set_bytes(6, 4, channels as *const u32 as *const _);
                    enc.set_bytes(7, 4, num_basis as *const u32 as *const _);
                    enc.set_bytes(8, 4, grid_min as *const f32 as *const _);
                    enc.set_bytes(9, 4, grid_max as *const f32 as *const _);
                    enc.dispatch_threads(
                        metal::MTLSize {
                            width: total as u64,
                            height: 1,
                            depth: 1,
                        },
                        metal::MTLSize {
                            width: 256.min(total) as u64,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                Thunk::SplineActivationBackwardCoeff {
                    x,
                    upstream,
                    dst,
                    rows,
                    channels,
                    num_basis,
                    grid_min,
                    grid_max,
                } => {
                    let enc = e!();
                    let total = (*rows) * (*channels);
                    let dcoeff_n = (*channels) * (*num_basis);
                    // zero dcoeff (atomic accumulate target)
                    enc.set_compute_pipeline_state(&k.zero_f32);
                    enc.set_buffer(0, Some(&self.arena.buffer), *dst as u64);
                    enc.set_bytes(1, 4, &dcoeff_n as *const u32 as *const _);
                    enc.dispatch_threads(
                        metal::MTLSize {
                            width: dcoeff_n as u64,
                            height: 1,
                            depth: 1,
                        },
                        metal::MTLSize {
                            width: 256.min(dcoeff_n) as u64,
                            height: 1,
                            depth: 1,
                        },
                    );
                    // accumulate
                    enc.set_compute_pipeline_state(&k.spline_activation_backward_coeff);
                    enc.set_buffer(0, Some(&self.arena.buffer), 0);
                    for (i, v) in [*x as u64, *upstream as u64, *dst as u64]
                        .iter()
                        .enumerate()
                    {
                        enc.set_bytes((i + 1) as u64, 8, v as *const u64 as *const _);
                    }
                    enc.set_bytes(4, 4, &total as *const u32 as *const _);
                    enc.set_bytes(5, 4, channels as *const u32 as *const _);
                    enc.set_bytes(6, 4, num_basis as *const u32 as *const _);
                    enc.set_bytes(7, 4, grid_min as *const f32 as *const _);
                    enc.set_bytes(8, 4, grid_max as *const f32 as *const _);
                    enc.dispatch_threads(
                        metal::MTLSize {
                            width: total as u64,
                            height: 1,
                            depth: 1,
                        },
                        metal::MTLSize {
                            width: 256.min(total) as u64,
                            height: 1,
                            depth: 1,
                        },
                    );
                }

                Thunk::SynthReconstruct {
                    indices,
                    codebook,
                    dst,
                    k: k_dim,
                    n,
                    entry_dim,
                } => {
                    let enc = e!();
                    enc.set_compute_pipeline_state(&k.synth_reconstruct_nk);
                    enc.set_buffer(0, Some(&self.arena.buffer), 0);
                    for (i, v) in [*indices as u64, *codebook as u64, *dst as u64]
                        .iter()
                        .enumerate()
                    {
                        enc.set_bytes((i + 1) as u64, 8, v as *const u64 as *const _);
                    }
                    // Kernel buffers 4,5,6 = n, k, d (writes w_bt[n,k], coalesced over k).
                    for (i, v) in [*n, *k_dim, *entry_dim].iter().enumerate() {
                        enc.set_bytes((i + 4) as u64, 4, v as *const u32 as *const _);
                    }
                    enc.dispatch_threads(
                        metal::MTLSize {
                            width: *k_dim as u64,
                            height: *n as u64,
                            depth: 1,
                        },
                        metal::MTLSize {
                            width: 32,
                            height: 8,
                            depth: 1,
                        },
                    );
                }

                Thunk::SynthMatMul {
                    x,
                    indices,
                    codebook,
                    dst,
                    m,
                    k: kk,
                    n,
                    entry_dim,
                    num_entries,
                    half,
                } => {
                    // Opt-in (RLX_METAL_SYNTH_TILED): threadgroup-tiled fused kernel
                    // — reconstructs the weight tile on-chip (no dense-weight DRAM
                    // scratch, no MPS launch, single capture-friendly dispatch).
                    // MEASURED slower than recon→MPS (hand GEMM ≈40% of MPS), so it is
                    // NOT the default; opt-in for zero-scratch / capturable cases only.
                    if *m > 8 && !*half && rlx_ir::env::flag("RLX_METAL_SYNTH_TILED") {
                        let enc = e!();
                        encode_synth_matmul_tiled(
                            enc,
                            k,
                            &self.arena.buffer,
                            *x,
                            *indices,
                            *codebook,
                            *dst,
                            *m,
                            *kk,
                            *n,
                            *entry_dim,
                            *num_entries,
                            rlx_ir::env::flag("RLX_METAL_SYNTH_TILED_F16"),
                        );
                    }
                    // Prefill (m>8, f32): reconstruct Wᵀ[n,k] into arena scratch,
                    // then MPS sgemm(x·Wᵀ) — ~f32 parity, ~7× the fused kernel (a
                    // fused GPU kernel can't beat MPS; reconstruction is cache-
                    // cheap). Decode / small-m / f16 keep the fused split-K / mm
                    // kernel. Off-switch: RLX_METAL_SYNTH_MPS_DISABLE.
                    else if *m > 8
                        && !*half
                        && self.synth_matmul_scratch_off != 0
                        && !rlx_ir::env::flag("RLX_METAL_SYNTH_MPS_DISABLE")
                    {
                        let scratch = self.synth_matmul_scratch_off;
                        let (m_u, kk_u, n_u) = (*m as usize, *kk as usize, *n as usize);
                        if rlx_ir::env::flag("RLX_METAL_SYNTH_RECON_F16") {
                            // Opt-in f16 prefill: cast x→f16, reconstruct W[k,n]→f16
                            // (half the scratch roundtrip), MPS hgemm, cast dst→f32.
                            // ~1.3× + 2× smaller scratch + steadier latency under load;
                            // relaxed precision (f16 weight/activation, f32 accumulate).
                            let a256 = |b: usize| (b + 255) & !255;
                            let w16 = scratch;
                            let x16 = a256(scratch + kk_u * n_u * 2);
                            let dst16 = a256(x16 + m_u * kk_u * 2);
                            let enc = e!();
                            encode_arena_cast(
                                enc,
                                &k.cast_f32_to_f16,
                                &self.arena.buffer,
                                *x,
                                x16,
                                (m_u * kk_u) as u32,
                            );
                            encode_synth_reconstruct_h(
                                enc,
                                k,
                                &self.arena.buffer,
                                *indices,
                                *codebook,
                                w16,
                                *kk,
                                *n,
                                *entry_dim,
                            );
                            end_msl!();
                            crate::mps_blas::encode_mps_hgemm(
                                &cmd_buf,
                                &self.arena.buffer,
                                x16,
                                w16,
                                dst16,
                                m_u,
                                kk_u,
                                n_u,
                            );
                            let enc = e!();
                            encode_arena_cast(
                                enc,
                                &k.cast_f16_to_f32,
                                &self.arena.buffer,
                                dst16,
                                *dst,
                                (m_u * n_u) as u32,
                            );
                        } else {
                            let enc = e!();
                            encode_synth_reconstruct(
                                enc,
                                k,
                                &self.arena.buffer,
                                *indices,
                                *codebook,
                                scratch,
                                *kk,
                                *n,
                                *entry_dim,
                            );
                            end_msl!();
                            crate::mps_blas::encode_mps_sgemm_bt(
                                &cmd_buf,
                                &self.arena.buffer,
                                *x,
                                scratch,
                                *dst,
                                m_u,
                                kk_u,
                                n_u,
                            );
                        }
                    } else {
                        let enc = e!();
                        encode_synth_matmul(
                            enc,
                            k,
                            &self.arena.buffer,
                            *x,
                            *indices,
                            *codebook,
                            *dst,
                            *m,
                            *kk,
                            *n,
                            *entry_dim,
                            *num_entries,
                            *half,
                        );
                    }
                }

                Thunk::DequantMatMulGguf {
                    x,
                    w_q,
                    dst,
                    m,
                    k: kk,
                    n,
                    scheme,
                    x_f16,
                    dst_f16,
                } => {
                    let m_u = *m as usize;
                    let k_u = *kk as usize;
                    let n_u = *n as usize;
                    // GPU dequant + MPS sgemm path is now default. On Gemma 4
                    // 12B Q4_K_M this is ~127× faster on prefill and ~10×
                    // faster on cached decode vs the host fallback that
                    // dequantizes and runs sgemm on rlx-cpu. Disable with
                    // RLX_METAL_DEQUANT_GPU_DISABLE=1 to revert.
                    let use_gpu_dequant = !crate::runtime_config().dequant_gpu_disable;
                    // Fused single-pass Q4_K GEMV — skips the f32 scratch
                    // entirely. Decode-only and k%256==0 (always true for
                    // GGUF Q4K). Off-switch: RLX_METAL_Q4K_FUSED_DISABLE=1.
                    // Q4_0 / Q8_0 fused GEMV: m==1, k%32==0; disable via
                    // RLX_METAL_Q40_FUSED_DISABLE / RLX_METAL_Q80_FUSED_DISABLE.
                    let use_fused_q4k_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ4K)
                        && !rlx_ir::env::flag("RLX_METAL_Q4K_FUSED_DISABLE");
                    // Fused single-pass Q3_K GEMV — the Q3_K_S bulk-weight decode
                    // path. Skips the dequant-to-f32-scratch + MPS sgemm.
                    // Off-switch: RLX_METAL_Q3K_FUSED_DISABLE=1.
                    let use_fused_q3k_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ3K)
                        && !rlx_ir::env::flag("RLX_METAL_Q3K_FUSED_DISABLE");
                    // Fused single-pass Q6_K GEMV — the Q6_K LM head (avoids a
                    // ~5 GB dequant-to-scratch per token). RLX_METAL_Q6K_FUSED_DISABLE=1.
                    let use_fused_q6k_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ6K)
                        && !rlx_ir::env::flag("RLX_METAL_Q6K_FUSED_DISABLE");
                    let use_fused_q4_0_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(32)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ4_0)
                        && !rlx_ir::env::flag("RLX_METAL_Q40_FUSED_DISABLE");
                    let use_fused_q4_1_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(32)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ4_1)
                        && !rlx_ir::env::flag("RLX_METAL_Q41_FUSED_DISABLE");
                    let use_fused_q8_0_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(32)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ8_0)
                        && !rlx_ir::env::flag("RLX_METAL_Q80_FUSED_DISABLE");
                    let use_fused_iq4_nl_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(32)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ4NL)
                        && !rlx_ir::env::flag("RLX_METAL_IQ4NL_FUSED_DISABLE");
                    let use_fused_iq2_xxs_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ2XXS)
                        && !rlx_ir::env::flag("RLX_METAL_IQ2XXS_FUSED_DISABLE");
                    let use_fused_iq2_xs_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ2XS)
                        && !rlx_ir::env::flag("RLX_METAL_IQ2XS_FUSED_DISABLE");
                    let use_fused_iq3_xxs_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ3XXS)
                        && !rlx_ir::env::flag("RLX_METAL_IQ3XXS_FUSED_DISABLE");
                    let use_fused_iq2_s_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ2S)
                        && !rlx_ir::env::flag("RLX_METAL_IQ2S_FUSED_DISABLE");
                    let use_fused_iq3_s_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ3S)
                        && !rlx_ir::env::flag("RLX_METAL_IQ3S_FUSED_DISABLE");
                    let use_fused_iq1_s_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ1S)
                        && !rlx_ir::env::flag("RLX_METAL_IQ1S_FUSED_DISABLE");
                    let use_fused_iq1_m_mv = use_gpu_dequant
                        && m_u == 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufIQ1M)
                        && !rlx_ir::env::flag("RLX_METAL_IQ1M_FUSED_DISABLE");
                    // Simdgroup-cooperative variant: 8 outputs per simdgroup
                    // with simd_sum. Requires n%8==0. Off-switch:
                    // RLX_METAL_Q4K_SG_DISABLE=1.
                    let use_q4k_mv_sg = use_fused_q4k_mv
                        && n_u.is_multiple_of(8)
                        && !rlx_ir::env::flag("RLX_METAL_Q4K_SG_DISABLE");
                    // Simdgroup-cooperative Q6_K / Q8_0 decode GEMV: 32 threads
                    // reduce each output row via simd_sum instead of one thread
                    // per row (occupancy-starved at small n). Row-guarded, so no
                    // n-divisibility constraint. Off-switches below.
                    let use_q6k_mv_sg =
                        use_fused_q6k_mv && !rlx_ir::env::flag("RLX_METAL_Q6K_SG_DISABLE");
                    let use_q8_0_mv_sg =
                        use_fused_q8_0_mv && !rlx_ir::env::flag("RLX_METAL_Q8_0_SG_DISABLE");
                    let use_q4_0_mv_sg =
                        use_fused_q4_0_mv && !rlx_ir::env::flag("RLX_METAL_Q40_SG_DISABLE");
                    let use_q4_1_mv_sg =
                        use_fused_q4_1_mv && !rlx_ir::env::flag("RLX_METAL_Q41_SG_DISABLE");
                    let use_q3k_mv_sg =
                        use_fused_q3k_mv && !rlx_ir::env::flag("RLX_METAL_Q3K_SG_DISABLE");
                    // Fused Q4_K / Q6_K prefill GEMM (m > 1): reads packed
                    // weight directly, dequants in-register, accumulates a row
                    // tile — replaces the dequant-to-f32-scratch + MPS sgemm
                    // path. Off-switch: RLX_METAL_Q4K_GEMM_DISABLE /
                    // RLX_METAL_Q6K_GEMM_DISABLE (→ legacy MPS path).
                    let use_fused_q4k_mm = use_gpu_dequant
                        && m_u > 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ4K)
                        && !rlx_ir::env::flag("RLX_METAL_Q4K_GEMM_DISABLE");
                    let use_fused_q6k_mm = use_gpu_dequant
                        && m_u > 1
                        && k_u.is_multiple_of(256)
                        && matches!(scheme, rlx_ir::QuantScheme::GgufQ6K)
                        && !rlx_ir::env::flag("RLX_METAL_Q6K_GEMM_DISABLE");
                    // Fused Q1_0 (Bonsai-27B 1-bit) GEMV (m==1) / GEMM (m>1):
                    // read packed weight directly, no f32 dequant scratch. The
                    // scratch path shares one buffer across every DequantMatMul
                    // and races at large n·k, zeroing the big projections in the
                    // real 27B graph (isolated single-op tests miss it). Also
                    // avoids the ~28× f32 blow-up. Off-switch:
                    // RLX_METAL_Q1_0_FUSED_DISABLE=1 (→ legacy scratch+sgemm).
                    let use_fused_q1_0 = use_gpu_dequant
                        && k_u.is_multiple_of(128)
                        && match scheme {
                            rlx_ir::QuantScheme::GgufQ1_0 => {
                                !rlx_ir::env::flag("RLX_METAL_Q1_0_FUSED_DISABLE")
                            }
                            rlx_ir::QuantScheme::GgufQ2_0 => {
                                !rlx_ir::env::flag("RLX_METAL_Q2_0_FUSED_DISABLE")
                            }
                            _ => false,
                        };
                    let use_fused_q1_0_mv = use_fused_q1_0 && m_u == 1;
                    // Simdgroup Q1_0 GEMV (llama.cpp path). Off-switch:
                    // RLX_METAL_Q1_0_SG_DISABLE=1 → naive one-thread-per-col.
                    let use_q1_0_mv_sg = use_fused_q1_0_mv
                        && n_u.is_multiple_of(8)
                        && match scheme {
                            rlx_ir::QuantScheme::GgufQ1_0 => {
                                !rlx_ir::env::flag("RLX_METAL_Q1_0_SG_DISABLE")
                            }
                            rlx_ir::QuantScheme::GgufQ2_0 => {
                                !rlx_ir::env::flag("RLX_METAL_Q2_0_SG_DISABLE")
                            }
                            _ => false,
                        };
                    let use_fused_q1_0_mm = use_fused_q1_0 && m_u > 1;
                    // The fused Q1_0 kernels read packed weights directly and need
                    // no dequant scratch, so a zero `dequant_scratch_off` (left
                    // unallocated for Q1_0-only graphs to save ~5 GiB) must NOT
                    // divert them to the host path.
                    let needs_scratch = !(use_fused_q1_0_mv || use_fused_q1_0_mm);
                    if !use_gpu_dequant
                        || (needs_scratch && self.dequant_scratch_off == 0)
                        || !has_metal_dequant_kernel(*scheme)
                    {
                        deferred_host.push(DeferredHostOp::DequantMatMulGguf {
                            x: *x,
                            w_q: *w_q,
                            dst: *dst,
                            m: m_u,
                            k: k_u,
                            n: n_u,
                            scheme: *scheme,
                        });
                    } else if use_q4k_mv_sg {
                        let enc = e!();
                        encode_q4k_mv_f32_sg(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_q4k_mv {
                        let enc = e!();
                        encode_q4k_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_q3k_mv_sg {
                        let enc = e!();
                        encode_q3k_mv_f32_sg(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_q3k_mv {
                        let enc = e!();
                        encode_q3k_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_q6k_mv_sg {
                        let enc = e!();
                        encode_q6k_mv_f32_sg(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_q6k_mv {
                        let enc = e!();
                        encode_q6k_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_q8_0_mv_sg {
                        let enc = e!();
                        encode_q8_0_mv_f32_sg(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_q4_0_mv_sg {
                        let enc = e!();
                        encode_q4_0_mv_f32_sg(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_q4_0_mv {
                        let enc = e!();
                        encode_q4_0_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_q4_1_mv_sg {
                        let enc = e!();
                        encode_q4_1_mv_f32_sg(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_q4_1_mv {
                        let enc = e!();
                        encode_q4_1_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_q8_0_mv {
                        let enc = e!();
                        encode_q8_0_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq4_nl_mv {
                        let enc = e!();
                        encode_iq4_nl_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq2_xxs_mv {
                        let enc = e!();
                        encode_iq2_xxs_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq2_xs_mv {
                        let enc = e!();
                        encode_iq2_xs_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq3_xxs_mv {
                        let enc = e!();
                        encode_iq3_xxs_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq2_s_mv {
                        let enc = e!();
                        encode_iq2_s_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq3_s_mv {
                        let enc = e!();
                        encode_iq3_s_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq1_s_mv {
                        let enc = e!();
                        encode_iq1_s_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_iq1_m_mv {
                        let enc = e!();
                        encode_iq1_m_mv_f32(enc, k, &self.arena.buffer, *x, *w_q, *dst, k_u, n_u);
                    } else if use_fused_q4k_mm {
                        let enc = e!();
                        encode_qk_mm_f32(
                            enc,
                            &k.q4k_mm_f32,
                            &self.arena.buffer,
                            *x,
                            *w_q,
                            *dst,
                            m_u,
                            k_u,
                            n_u,
                        );
                    } else if use_fused_q6k_mm {
                        let enc = e!();
                        encode_qk_mm_f32(
                            enc,
                            &k.q6k_mm_f32,
                            &self.arena.buffer,
                            *x,
                            *w_q,
                            *dst,
                            m_u,
                            k_u,
                            n_u,
                        );
                    } else if use_q1_0_mv_sg {
                        let xf16 = *x_f16;
                        let dstf16 = *dst_f16;
                        let cur = i - 1;
                        let partner = find_shared_x_q1_partner(
                            thunks,
                            cur,
                            loop_end,
                            &skip_thunks,
                            *scheme,
                            *x,
                            *dst,
                            k_u,
                        );
                        let (w_buf, w_raw) = self.resolve_off(*w_q);
                        if let Some((j, w1, dst1, n1)) = partner {
                            let (w_buf1, w1_raw) = self.resolve_off(w1);
                            // Dual kernel binds one weight buffer; both must live there.
                            // Require matching dst dtype flags (AMP residual stream).
                            let partner_dst_f16 = match &thunks[j] {
                                Thunk::DequantMatMulGguf { dst_f16, .. } => *dst_f16,
                                _ => *dst_f16,
                            };
                            if std::ptr::eq(w_buf as *const _, w_buf1 as *const _)
                                && n1.is_multiple_of(8)
                                && partner_dst_f16 == dstf16
                                && match scheme {
                                    rlx_ir::QuantScheme::GgufQ1_0 => {
                                        !rlx_ir::env::flag("RLX_METAL_Q1_DUAL_DISABLE")
                                    }
                                    rlx_ir::QuantScheme::GgufQ2_0 => {
                                        !rlx_ir::env::flag("RLX_METAL_Q2_DUAL_DISABLE")
                                    }
                                    _ => false,
                                }
                            {
                                skip_thunks.insert(j);
                                dual_q1_fused += 1;
                                let enc = e!();
                                encode_q1_0_dual_mv_f32_sg_flags(
                                    enc,
                                    k,
                                    *scheme,
                                    &self.arena.buffer,
                                    w_buf,
                                    *x,
                                    w_raw,
                                    w1_raw,
                                    *dst,
                                    dst1,
                                    k_u,
                                    n_u,
                                    n1,
                                    xf16,
                                    dstf16,
                                );
                            } else {
                                let enc = e!();
                                encode_q1_0_mv_f32_sg_flags(
                                    enc,
                                    k,
                                    *scheme,
                                    &self.arena.buffer,
                                    w_buf,
                                    *x,
                                    w_raw,
                                    *dst,
                                    k_u,
                                    n_u,
                                    xf16,
                                    dstf16,
                                );
                            }
                        } else {
                            let enc = e!();
                            encode_q1_0_mv_f32_sg_flags(
                                enc,
                                k,
                                *scheme,
                                &self.arena.buffer,
                                w_buf,
                                *x,
                                w_raw,
                                *dst,
                                k_u,
                                n_u,
                                xf16,
                                dstf16,
                            );
                        }
                        // Keep the compute encoder open across consecutive Q1_0
                        // MSL matmuls (Serial dispatch orders them). Ending here
                        // was ~500 encoder restarts/token on Bonsai-27B decode.
                    } else if use_fused_q1_0_mv {
                        // Weight may be tagged for the external weight buffer
                        // (large params); resolve to (buffer, raw offset).
                        let (w_buf, w_raw) = self.resolve_off(*w_q);
                        let enc = e!();
                        encode_q1_0_mv_f32(
                            enc,
                            k,
                            *scheme,
                            &self.arena.buffer,
                            w_buf,
                            *x,
                            w_raw,
                            *dst,
                            k_u,
                            n_u,
                        );
                    } else if use_fused_q1_0_mm {
                        let (w_buf, w_raw) = self.resolve_off(*w_q);
                        let enc = e!();
                        encode_q1_0_mm_f32(
                            enc,
                            k,
                            *scheme,
                            &self.arena.buffer,
                            w_buf,
                            *x,
                            w_raw,
                            *dst,
                            m_u,
                            k_u,
                            n_u,
                        );
                    } else {
                        let enc = e!();
                        encode_dequant_gguf(
                            enc,
                            k,
                            &self.arena.buffer,
                            *w_q,
                            self.dequant_scratch_off,
                            *scheme,
                            k_u,
                            n_u,
                        );
                        end_msl!();
                        // B is [n,k] row-major in scratch; use MPS with B^T.
                        crate::mps_blas::encode_mps_sgemm_bt(
                            &cmd_buf,
                            &self.arena.buffer,
                            *x,
                            self.dequant_scratch_off,
                            *dst,
                            m_u,
                            k_u,
                            n_u,
                        );
                    }
                }

                Thunk::FusedMlpGateUpSwiGLU {
                    x,
                    gate_w,
                    up_w,
                    dst,
                    k: kk,
                    n,
                    scheme,
                    x_f16,
                    dst_f16,
                } => {
                    if matches!(
                        scheme,
                        rlx_ir::QuantScheme::GgufQ1_0 | rlx_ir::QuantScheme::GgufQ2_0
                    ) {
                        let (g_buf, g_raw) = self.resolve_off(*gate_w);
                        let (_u_buf, u_raw) = self.resolve_off(*up_w);
                        let enc = e!();
                        encode_q1_0_swiglu_mv_f32(
                            enc,
                            k,
                            *scheme,
                            &self.arena.buffer,
                            g_buf,
                            *x,
                            g_raw,
                            u_raw,
                            *dst,
                            *kk as usize,
                            *n as usize,
                            *x_f16,
                            *dst_f16,
                        );
                    } else {
                        let enc = e!();
                        encode_fused_mlp_gate_up_swiglu(
                            enc,
                            k,
                            &self.arena.buffer,
                            *scheme,
                            *x,
                            *gate_w,
                            *up_w,
                            *dst,
                            *kk as usize,
                            *n as usize,
                        );
                    }
                }

                Thunk::FusedMlpGateUpGelu {
                    x,
                    gate_w,
                    up_w,
                    dst,
                    k: kk,
                    n,
                    scheme,
                } => {
                    let enc = e!();
                    encode_fused_mlp_gate_up_gelu(
                        enc,
                        k,
                        &self.arena.buffer,
                        *scheme,
                        *x,
                        *gate_w,
                        *up_w,
                        *dst,
                        *kk as usize,
                        *n as usize,
                    );
                }

                Thunk::FusedMlpDownResidual {
                    x,
                    w,
                    res,
                    dst,
                    k: kk,
                    n,
                    scheme,
                    x_f16,
                    dst_f16,
                    res_f16,
                } => {
                    if matches!(
                        scheme,
                        rlx_ir::QuantScheme::GgufQ1_0 | rlx_ir::QuantScheme::GgufQ2_0
                    ) {
                        let (w_buf, w_raw) = self.resolve_off(*w);
                        let enc = e!();
                        encode_q1_0_mv_residual_f32(
                            enc,
                            k,
                            *scheme,
                            &self.arena.buffer,
                            w_buf,
                            *x,
                            w_raw,
                            *res,
                            *dst,
                            *kk as usize,
                            *n as usize,
                            *x_f16,
                            *dst_f16,
                            *res_f16,
                        );
                    } else {
                        let pipeline = match scheme {
                            rlx_ir::QuantScheme::GgufQ4K => &k.q4k_mv_residual_f32,
                            rlx_ir::QuantScheme::GgufQ5_0 => &k.q5_0_mv_residual_f32,
                            rlx_ir::QuantScheme::GgufQ6K => &k.q6k_mv_residual_f32,
                            other => panic!(
                                "FusedMlpDownResidual: unsupported scheme {other:?} \
                                 (fuse_decode_mlp only emits Q4_K / Q5_0 / Q6_K / Q1_0)"
                            ),
                        };
                        let enc = e!();
                        encode_q4k_mv_residual_f32(
                            enc,
                            pipeline,
                            &self.arena.buffer,
                            *x,
                            *w,
                            *res,
                            *dst,
                            *kk as usize,
                            *n as usize,
                        );
                        end_msl!();
                    }
                }

                Thunk::DequantGroupedMatMulGguf {
                    input,
                    w_q,
                    expert_idx,
                    dst,
                    m,
                    k_dim: kk,
                    n,
                    num_experts,
                    scheme,
                } => {
                    let m_u = *m as usize;
                    let k_u = *kk as usize;
                    let n_u = *n as usize;
                    let ne = *num_experts as usize;
                    // Matches Thunk::DequantMatMulGguf above — GPU default,
                    // RLX_METAL_DEQUANT_GPU_DISABLE=1 to revert.
                    let use_gpu_dequant = !crate::runtime_config().dequant_gpu_disable;
                    if !use_gpu_dequant
                        || self.dequant_scratch_off == 0
                        || !has_metal_dequant_kernel(*scheme)
                    {
                        deferred_host.push(DeferredHostOp::DequantGroupedMatMulGguf {
                            input: *input,
                            w_q: *w_q,
                            expert_idx: *expert_idx,
                            dst: *dst,
                            m: m_u,
                            k: k_u,
                            n: n_u,
                            num_experts: ne,
                            scheme: *scheme,
                        });
                    } else if dequant_grouped_can_encode_per_row(*scheme, k_u) {
                        // Decode / K-quant fast path: per-token fused GEMV on the
                        // parent encoder — no host sort, no private cmd_buf wait.
                        encode_dequant_grouped_matmul_gguf_per_row(
                            e!(),
                            k,
                            &self.arena.buffer,
                            *input,
                            *w_q,
                            *expert_idx,
                            *dst,
                            m_u,
                            k_u,
                            n_u,
                            ne,
                            *scheme,
                        );
                    } else {
                        // Grouped path interleaves MSL dequant with MPS sgemm
                        // and needs the routed `input`/`expert_idx` on the host
                        // for the sort + unpermute — flush all prior GPU work
                        // first, then let the helper drive its own command
                        // buffer. `sync_gpu!` leaves `cmd_buf` fresh for the
                        // thunks that follow.
                        sync_gpu!();
                        encode_dequant_grouped_matmul_gguf(
                            &dev.queue,
                            k,
                            &self.arena.buffer,
                            self.dequant_scratch_off,
                            *input,
                            *w_q,
                            *expert_idx,
                            *dst,
                            m_u,
                            k_u,
                            n_u,
                            ne,
                            *scheme,
                        );
                    }
                }

                Thunk::DequantMatMulInt8 {
                    x,
                    w_q,
                    scale,
                    zp,
                    dst,
                    m,
                    k: kk,
                    n,
                    block_size,
                    is_asymmetric,
                } => {
                    // Native GPU dequant-matmul when inputs are GPU-produced;
                    // defer to host otherwise (ordering with prior host ops).
                    if deferred_host.is_empty() {
                        encode_dequant_matmul(
                            e!(),
                            &k.dequant_matmul_int8,
                            &self.arena.buffer,
                            *x,
                            *w_q,
                            *scale,
                            *zp,
                            *dst,
                            *m,
                            *kk,
                            *n,
                            *block_size,
                            *is_asymmetric as u32,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::DequantMatMulInt8 {
                            x: *x,
                            w_q: *w_q,
                            scale: *scale,
                            zp: *zp,
                            dst: *dst,
                            m: *m as usize,
                            k: *kk as usize,
                            n: *n as usize,
                            block_size: *block_size,
                            is_asymmetric: *is_asymmetric,
                        });
                    }
                }

                Thunk::DequantMatMulInt4 {
                    x,
                    w_q,
                    scale,
                    zp,
                    dst,
                    m,
                    k: kk,
                    n,
                    block_size,
                    is_asymmetric,
                } => {
                    if deferred_host.is_empty() {
                        encode_dequant_matmul(
                            e!(),
                            &k.dequant_matmul_int4,
                            &self.arena.buffer,
                            *x,
                            *w_q,
                            *scale,
                            *zp,
                            *dst,
                            *m,
                            *kk,
                            *n,
                            *block_size,
                            *is_asymmetric as u32,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::DequantMatMulInt4 {
                            x: *x,
                            w_q: *w_q,
                            scale: *scale,
                            zp: *zp,
                            dst: *dst,
                            m: *m as usize,
                            k: *kk as usize,
                            n: *n as usize,
                            block_size: *block_size,
                            is_asymmetric: *is_asymmetric,
                        });
                    }
                }

                Thunk::DequantMatMulFp8 {
                    x,
                    w_q,
                    scale,
                    dst,
                    m,
                    k: kk,
                    n,
                    e5m2,
                } => {
                    if deferred_host.is_empty() {
                        encode_dequant_matmul_fp8(
                            e!(),
                            &k.dequant_matmul_fp8,
                            &self.arena.buffer,
                            *x,
                            *w_q,
                            *scale,
                            *dst,
                            *m,
                            *kk,
                            *n,
                            *e5m2 as u32,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::DequantMatMulFp8 {
                            x: *x,
                            w_q: *w_q,
                            scale: *scale,
                            dst: *dst,
                            m: *m as usize,
                            k: *kk as usize,
                            n: *n as usize,
                            e5m2: *e5m2,
                        });
                    }
                }

                Thunk::DequantMatMulNvfp4 {
                    x,
                    w_q,
                    scale,
                    global_scale,
                    dst,
                    m,
                    k: kk,
                    n,
                } => {
                    if deferred_host.is_empty() {
                        encode_dequant_matmul_nvfp4(
                            e!(),
                            &k.dequant_matmul_nvfp4,
                            &self.arena.buffer,
                            *x,
                            *w_q,
                            *scale,
                            *global_scale,
                            *dst,
                            *m,
                            *kk,
                            *n,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::DequantMatMulNvfp4 {
                            x: *x,
                            w_q: *w_q,
                            scale: *scale,
                            global_scale: *global_scale,
                            dst: *dst,
                            m: *m as usize,
                            k: *kk as usize,
                            n: *n as usize,
                        });
                    }
                }

                Thunk::DequantMatMulMxFp4x2 {
                    x,
                    w_q,
                    scale,
                    dst,
                    m,
                    k: kk,
                    n,
                    group,
                } => {
                    if deferred_host.is_empty() {
                        encode_dequant_matmul_mxfp4x2(
                            e!(),
                            &k.dequant_matmul_mxfp4x2,
                            &self.arena.buffer,
                            *x,
                            *w_q,
                            *scale,
                            *dst,
                            *m,
                            *kk,
                            *n,
                            *group,
                        );
                    } else {
                        deferred_host.push(DeferredHostOp::DequantMatMulMxFp4x2 {
                            x: *x,
                            w_q: *w_q,
                            scale: *scale,
                            dst: *dst,
                            m: *m as usize,
                            k: *kk as usize,
                            n: *n as usize,
                            group: *group as usize,
                        });
                    }
                }

                Thunk::DequantMatMulMlx {
                    x,
                    w_q,
                    scale,
                    zp,
                    dst,
                    m,
                    k: kk,
                    n,
                    scheme,
                } => {
                    let (kind, bits, group_size) = scheme.mlx_gpu_launch().unwrap_or_else(|| {
                        panic!("rlx-metal DequantMatMulMlx: unexpected {scheme:?}")
                    });
                    let use_gpu = deferred_host.is_empty()
                        && !crate::runtime_config().dequant_gpu_disable
                        && !rlx_gpu_host::mlx_dequant_gpu_disabled();
                    if use_gpu {
                        if *m == 1 {
                            crate::prefill_stats::record_dequant_gemv();
                            if rlx_ir::env::flag("RLX_METAL_PREFILL_TRACE") {
                                eprintln!(
                                    "[prefill-trace] dequant_mlx m=1 k={} n={} kind={} bits={} group={} -> gemv",
                                    *kk, *n, kind, bits, group_size
                                );
                            }
                            encode_dequant_matmul_mlx_gemv(
                                e!(),
                                &k.dequant_matmul_mlx_gemv,
                                &self.arena.buffer,
                                *x,
                                *w_q,
                                *scale,
                                *zp,
                                *dst,
                                *kk,
                                *n,
                                kind,
                                bits,
                                group_size,
                            );
                        } else {
                            crate::prefill_stats::record_dequant_gemm();
                            if rlx_ir::env::flag("RLX_METAL_PREFILL_TRACE") {
                                eprintln!(
                                    "[prefill-trace] dequant_mlx m={} k={} n={} kind={} bits={} group={} -> gemm",
                                    *m, *kk, *n, kind, bits, group_size
                                );
                            }
                            encode_dequant_matmul_mlx_gemm(
                                e!(),
                                &k.dequant_matmul_mlx_gemm,
                                &self.arena.buffer,
                                *x,
                                *w_q,
                                *scale,
                                *zp,
                                *dst,
                                *m,
                                *kk,
                                *n,
                                kind,
                                bits,
                                group_size,
                            );
                        }
                    } else {
                        deferred_host.push(DeferredHostOp::DequantMatMulMlx {
                            x: *x,
                            w_q: *w_q,
                            scale: *scale,
                            zp: *zp,
                            dst: *dst,
                            m: *m as usize,
                            k: *kk as usize,
                            n: *n as usize,
                            scheme: *scheme,
                        });
                    }
                }

                // MLX-affine/MXFP4 MoE grouped matmul. Native GPU expert kernel for
                // the decode GEMV (m==1) — dequant+matmul the selected expert's slab
                // straight from unified memory; host-delegate the prefill GEMM and
                // BF16-scale paths (not yet native), matching the CPU per-row dequant.
                Thunk::DequantGroupedMatMulMlx {
                    input,
                    w_q,
                    scale,
                    zp,
                    expert_idx,
                    dst,
                    m,
                    k_dim: kk,
                    n,
                    num_experts,
                    slab_bytes,
                    scheme,
                    scale_bf16,
                } => {
                    let launch = scheme.mlx_gpu_launch();
                    let use_gpu = launch.is_some()
                        && deferred_host.is_empty()
                        && !crate::runtime_config().dequant_gpu_disable
                        && !rlx_gpu_host::mlx_dequant_gpu_disabled();
                    if use_gpu {
                        let (kind, bits, group_size) = launch.unwrap();
                        if *m == 1 {
                            encode_grouped_dequant_matmul_mlx_gemv(
                                e!(),
                                &k.grouped_dequant_matmul_mlx_gemv,
                                &self.arena.buffer,
                                *input,
                                *w_q,
                                *scale,
                                *zp,
                                *dst,
                                *expert_idx,
                                *kk,
                                *n,
                                kind,
                                bits,
                                group_size,
                                *slab_bytes,
                                u32::from(*scale_bf16),
                            );
                        } else {
                            encode_grouped_dequant_matmul_mlx_gemm(
                                e!(),
                                &k.grouped_dequant_matmul_mlx_gemm,
                                &self.arena.buffer,
                                *input,
                                *w_q,
                                *scale,
                                *zp,
                                *dst,
                                *expert_idx,
                                *m,
                                *kk,
                                *n,
                                kind,
                                bits,
                                group_size,
                                *slab_bytes,
                                u32::from(*scale_bf16),
                            );
                        }
                    } else {
                        deferred_host.push(DeferredHostOp::DequantGroupedMatMulMlx {
                            input: *input,
                            w_q: *w_q,
                            scale: *scale,
                            zp: *zp,
                            expert_idx: *expert_idx,
                            dst: *dst,
                            m: *m as usize,
                            k: *kk as usize,
                            n: *n as usize,
                            num_experts: *num_experts as usize,
                            slab_bytes: *slab_bytes as usize,
                            scheme: *scheme,
                            scale_bf16: *scale_bf16,
                        });
                    }
                }

                // Native low-precision GEMM + quantize: Apple GPUs have no FP8
                // matrix HW, so these always run as the host decode-and-accumulate
                // reference on unified memory (ordered after GPU compute).
                Thunk::ScaledMatMul {
                    lhs,
                    rhs,
                    lhs_scale,
                    rhs_scale,
                    bias,
                    dst,
                    m,
                    k: kk,
                    n,
                    lhs_fmt,
                    rhs_fmt,
                    layout,
                    has_bias,
                } => {
                    deferred_host.push(DeferredHostOp::ScaledMatMul {
                        lhs: *lhs,
                        rhs: *rhs,
                        lhs_scale: *lhs_scale,
                        rhs_scale: *rhs_scale,
                        bias: *bias,
                        dst: *dst,
                        m: *m as usize,
                        k: *kk as usize,
                        n: *n as usize,
                        has_bias: *has_bias,
                        lhs_fmt: *lhs_fmt,
                        rhs_fmt: *rhs_fmt,
                        layout: *layout,
                    });
                }
                Thunk::ScaledQuantize {
                    x,
                    scale,
                    dst,
                    rows,
                    cols,
                    fmt,
                    layout,
                } => {
                    deferred_host.push(DeferredHostOp::ScaledQuantize {
                        x: *x,
                        scale: *scale,
                        dst: *dst,
                        rows: *rows as usize,
                        cols: *cols as usize,
                        fmt: *fmt,
                        layout: *layout,
                    });
                }
                Thunk::ScaledDequantize {
                    codes,
                    scale,
                    dst,
                    rows,
                    cols,
                    fmt,
                    layout,
                } => {
                    deferred_host.push(DeferredHostOp::ScaledDequantize {
                        codes: *codes,
                        scale: *scale,
                        dst: *dst,
                        rows: *rows as usize,
                        cols: *cols as usize,
                        fmt: *fmt,
                        layout: *layout,
                    });
                }
                Thunk::ScaledQuantScale {
                    x,
                    dst,
                    rows,
                    cols,
                    fmt,
                    layout,
                } => {
                    deferred_host.push(DeferredHostOp::ScaledQuantScale {
                        x: *x,
                        dst: *dst,
                        rows: *rows as usize,
                        cols: *cols as usize,
                        fmt: *fmt,
                        layout: *layout,
                    });
                }
            }
        }

        if dual_q1_log && dual_q1_fused > 0 {
            eprintln!("[rlx-metal] encode dual Q1 shared-x: {dual_q1_fused} pairs fused this run");
        }

        let _ = (enc_opened, barriers_emitted, thunks_dispatched);
        if concurrent_stats {
            let shared = thunks_dispatched.saturating_sub(enc_opened);
            eprintln!(
                "[rlx-metal] concurrent: {thunks_dispatched} thunks, {enc_opened} encoders opened \
                 ({shared} dispatches shared an encoder), {barriers_emitted} barriers emitted"
            );
        }

        end_msl!();
        // Per-commit output snapshot for pipelined runs. Encoded as a blit
        // *after* the compute work — Metal serialises encoders within a
        // single command buffer, so the blit reads the arena once compute
        // has finished writing to it.
        if let Some(dests) = blit_outputs {
            assert_eq!(
                dests.len(),
                self.output_slots.len(),
                "blit_outputs len must match graph output count"
            );
            let blit = cmd_buf.new_blit_command_encoder();
            for (i, (off, len)) in self.output_slots.iter().enumerate() {
                // F16 outputs occupy 2 bytes/elem; every other dtype's f32-lane
                // count already maps 1:1 to 4 bytes (F64/C64/C128 pre-expanded).
                let elem_bytes =
                    if self.graph.node(self.graph.outputs[i]).shape.dtype() == rlx_ir::DType::F16 {
                        2
                    } else {
                        4
                    };
                let bytes = (*len as u64) * elem_bytes;
                if bytes == 0 {
                    continue;
                }
                blit.copy_from_buffer(&self.arena.buffer, *off as u64, &dests[i], 0, bytes);
            }
            blit.end_encoding();
        }
        // Optional micro-instrumentation: RLX_METAL_TRACE=1 prints
        // encode/commit/wait µs split.
        let t_enc_done = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };
        cmd_buf.commit();
        let t_commit_done = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };
        if wait {
            cmd_buf.wait_until_completed();
            check_cmd_buf_status(&cmd_buf, "encode_commit (final wait)");
            if !tail_host.is_empty() {
                let arena_ptr = self.arena.buffer.contents() as *mut u8;
                for op in tail_host.drain(..) {
                    match op {
                        TailHostOp::WelchPeaks {
                            spec,
                            dst,
                            welch_batch,
                            n_fft,
                            n_segments,
                            k,
                        } => unsafe {
                            rlx_cpu::thunk::execute_welch_peaks_f32(
                                spec,
                                dst,
                                welch_batch as usize,
                                n_fft as usize,
                                n_segments as usize,
                                k as usize,
                                arena_ptr,
                            );
                        },
                    }
                }
            }
            if trace {
                let t_wait_done = std::time::Instant::now();
                let t_start = t_run_start.unwrap();
                let enc_us = t_enc_done.unwrap().duration_since(t_start).as_secs_f64() * 1e6;
                let commit_us = t_commit_done
                    .unwrap()
                    .duration_since(t_enc_done.unwrap())
                    .as_secs_f64()
                    * 1e6;
                let wait_us = t_wait_done
                    .duration_since(t_commit_done.unwrap())
                    .as_secs_f64()
                    * 1e6;
                eprintln!(
                    "[metal-trace] encode={enc_us:.1}µs commit={commit_us:.1}µs wait={wait_us:.1}µs"
                );
            }
            None
        } else {
            if trace {
                let enc_us = t_enc_done
                    .unwrap()
                    .duration_since(t_run_start.unwrap())
                    .as_secs_f64()
                    * 1e6;
                let commit_us = t_commit_done
                    .unwrap()
                    .duration_since(t_enc_done.unwrap())
                    .as_secs_f64()
                    * 1e6;
                eprintln!(
                    "[metal-trace] encode={enc_us:.1}µs commit={commit_us:.1}µs (wait deferred)"
                );
            }
            Some(cmd_buf)
        }
    }
}

/// Look ahead for a second Q1_0 GEMV that shares `x` (qkv+gate pattern).
/// Stops at the first non-Nop/Copy thunk that isn't a matching partner.
fn find_shared_x_q1_partner(
    thunks: &[Thunk],
    cur: usize,
    loop_end: usize,
    skip: &HashSet<usize>,
    scheme: rlx_ir::QuantScheme,
    x: usize,
    dst0: usize,
    k: usize,
) -> Option<(usize, usize, usize, usize)> {
    use rlx_ir::QuantScheme;
    let end = (cur + 1 + 24).min(loop_end);
    for j in (cur + 1)..end {
        if skip.contains(&j) {
            continue;
        }
        match &thunks[j] {
            Thunk::Nop => {}
            Thunk::Copy { src, dst, .. } => {
                if *src == dst0 || *dst == dst0 || *dst == x {
                    return None;
                }
            }
            Thunk::DequantMatMulGguf {
                x: x2,
                w_q,
                dst,
                m: 1,
                k: k2,
                n,
                scheme: scheme2,
                ..
            } if *scheme2 == scheme && *x2 == x && *k2 as usize == k && *dst != dst0 => {
                return Some((j, *w_q, *dst, *n as usize));
            }
            _ => return None,
        }
    }
    None
}
