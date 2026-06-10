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

//! Metal backend — implements rlx-runtime's Backend trait.
//!
//! Pipeline:
//!   1. Run rlx-opt fusion passes on the graph
//!   2. Plan memory (single arena, GPU buffer)
//!   3. Compile thunk schedule
//!   4. On each run: encode thunks into a command buffer, commit, wait

use rlx_ir::{Graph, NodeId, Op};
use rlx_opt::memory;
use std::collections::HashMap;

use crate::arena::Arena;
use crate::device::metal_device;
use crate::kernels::kernels;
use crate::thunk::{Thunk, ThunkSchedule};

/// Numeric precision for Metal graph compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalPrecision {
    /// Full f32 throughout. Always supported.
    F32,
    /// Half precision (f16). Requires f16 kernel variants for every op
    /// in the graph — currently only matmul has f16 kernels (`hgemm_*`).
    /// Until other ops are ported, F16 compile falls back to F32.
    F16,
}

/// Metal-compiled executable graph.
pub struct MetalExecutable {
    graph: Graph,
    arena: Arena,
    schedule: ThunkSchedule,
    input_ids: HashMap<String, NodeId>,
    param_ids: HashMap<String, NodeId>,
    /// Pre-resolved (name, byte_offset, max_f32_len) per input — for run_slots.
    input_slots: Vec<(String, usize, usize)>,
    output_slots: Vec<(usize, usize)>, // (byte_offset, f32_len)
    /// Precision this graph was compiled at.
    precision: MetalPrecision,
    /// Optional MPSGraph plan — populated when `RLX_USE_MPSGRAPH=1` and
    /// every op in the graph is supported by the lowerer. Replaces the
    /// per-op thunk path with one compiled MPSGraph for the whole forward.
    mps_plan: Option<crate::mps_graph_lower::MpsGraphPlan>,
    /// Hybrid MPSGraph + thunk schedule when whole-graph lowering fails
    /// (Qwen3.5 decode: matmul/norm/attn via MPS, GDN via thunks).
    mps_hybrid: Option<Vec<crate::mps_graph_hybrid::HybridStep>>,
    /// ICB segments — populated when `RLX_USE_ICB=1`. One segment per
    /// maximal run of ICB-compatible thunks in the schedule. Each segment
    /// pre-encodes its run into an `MTLIndirectCommandBuffer` at compile
    /// time; runtime calls `executeCommandsInBuffer` once per segment.
    /// Empty when ICB is disabled or no run exceeds the minimum length.
    icb_segments: Vec<crate::icb::IcbRange>,
    /// In-flight command buffers from `commit_no_wait`. Drained by
    /// `sync_pending`. Used by callers that pipeline multiple commits
    /// to amortize the GPU sync latency (~150µs/commit on Apple Silicon).
    pending_cmd_bufs: Vec<metal::CommandBuffer>,
    /// Active-extent hint (`Some((actual, upper))`) for L1 bucketed
    /// dispatch. When set AND every thunk in `schedule` is in the
    /// safe set, `encode_commit` bypasses MPSGraph + ICB segments
    /// (both pre-encode at full extent) and dispatches per-op with
    /// scaled launch dimensions. Otherwise full-extent fallback.
    pub(crate) active_extent: Option<(usize, usize)>,
    /// Largest matmul FLOP count seen at compile time. Drives the
    /// MPSGraph-vs-per-op adaptive dispatch (see `encode_and_run`).
    /// Computed once because graph shape is static after compile.
    max_matmul_flops: u64,
    /// Set after the first `encode_and_run` triggers
    /// `freeze_params_to_mps_constants`. Subsequent runs skip the
    /// (idempotent but not free) re-lower.
    mps_params_frozen: bool,
    /// Arena tail reserved for ephemeral GatedDeltaNet state when
    /// `Op::GatedDeltaNet` runs without carry (state input absent).
    gdn_scratch_off: usize,
    /// Arena tail scratch for GPU GGUF dequant before matmul (reused per op).
    dequant_scratch_off: usize,
    /// Arena tail scratch for GPU im2col before conv weight backward GEMM.
    conv_bwd_scratch_off: usize,
    /// Arena tail scratch for GPU attention backward (scores, dp, ds).
    attn_bwd_scratch_off: usize,
    /// Arena tail scratch for parallel RMSNorm param backward.
    rms_norm_bwd_scratch_off: usize,
    /// Persistent KV / state inputs (unified-memory `Vec`, fed into arena each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    /// After each run, copy `graph.outputs[idx]` into the named handle.
    gpu_handle_feeds: HashMap<String, usize>,
    /// Handles whose arena input slots are authoritative (skip host mirror ping-pong).
    gpu_handle_resident: std::collections::HashSet<String>,
}

unsafe impl Send for MetalExecutable {}

impl Drop for MetalExecutable {
    fn drop(&mut self) {
        // Drain deferred commits before releasing MTL buffers / MPSGraph
        // executables — otherwise Metal logs "operations may not have completed".
        self.sync_pending();
        crate::device::drain_command_queue();
        crate::mps_blas::invalidate_caches();
    }
}

impl MetalExecutable {
    /// Compile at the requested precision.
    pub fn compile_with_precision(graph: Graph, precision: MetalPrecision) -> Self {
        // F16 compilation requires every kernel in the graph to have an f16
        // variant. Until they do, transparently fall back to F32 with a note.
        let effective = if precision == MetalPrecision::F16 {
            let verbose = rlx_ir::env::var("RLX_VERBOSE")
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(0)
                >= 1;
            if verbose {
                eprintln!(
                    "[rlx-metal] F16 requested but full-graph f16 kernels are WIP; using F32"
                );
            }
            MetalPrecision::F32
        } else {
            precision
        };
        let mut exe = Self::compile(graph);
        exe.precision = effective;
        exe
    }

    pub fn compile(graph: Graph) -> Self {
        Self::compile_inner(graph, None, None, false)
    }

    /// Compile with an optional `PrecisionPolicy`. The pass runs *after*
    /// fusion to avoid breaking pattern-match-based fusion via interleaved
    /// Cast nodes.
    pub fn compile_with_policy(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
    ) -> Self {
        Self::compile_inner(graph, policy, supported_ops, false)
    }

    /// Compile a graph that already went through the fusion pipeline
    /// (e.g. from [`rlx_ir::LirModule`]). Skips re-fusion so backends
    /// invoked via `Backend::compile_lir` do not undo fused ops.
    pub fn compile_from_fused(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
    ) -> Self {
        Self::compile_inner(graph, policy, supported_ops, true)
    }

    fn compile_inner(
        graph: Graph,
        policy: Option<rlx_opt::PrecisionPolicy>,
        supported_ops: Option<&'static [rlx_ir::OpKind]>,
        skip_fusion: bool,
    ) -> Self {
        let verbose = rlx_ir::env::var("RLX_VERBOSE")
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(0)
            >= 1;

        if verbose {
            eprintln!("[rlx-metal] compiling graph: {} nodes", graph.len());
        }

        // Drop the global MPSMatrix / MPSMatrixDescriptor / MPSMatrixMul
        // caches before building this compile's arena. The cache keys
        // include the Buffer-wrapper address, which CAN recycle when
        // the prior `MetalExecutable` is dropped — without this reset
        // a fresh Sam (e.g. CPU → Metal in the same process) gets
        // back stale `MPSMatrix` wrappers pointing at freed memory and
        // produces NaN outputs.
        crate::mps_blas::invalidate_caches();

        // Backend-aware fusion: only emit fused ops Metal can lower.
        let fused = if skip_fusion {
            graph
        } else {
            let mut pipe = rlx_opt::CompilePipeline::new(rlx_opt::FusionTarget::Metal)
                .with_assert_fusion_clean(false);
            if let Some(ops) = supported_ops {
                pipe = pipe.with_supported_ops(ops);
            }
            let compile_result = pipe.compile_graph(graph);
            if verbose {
                eprintln!(
                    "[rlx-metal] fusion: {} → {} nodes",
                    compile_result.fusion.nodes_before, compile_result.fusion.nodes_after
                );
            }
            compile_result.lir.into_graph()
        };

        // AutoMixedPrecision runs AFTER fusion: Cast nodes interleave between
        // the (now flattened) ops without breaking earlier pattern matchers.
        let fused = match policy {
            Some(p) => {
                use rlx_opt::pass::Pass;
                let g = rlx_opt::AutoMixedPrecision::new(p).run(fused);
                if verbose {
                    eprintln!("[rlx-metal] after AutoMixedPrecision: {} nodes", g.len());
                }
                g
            }
            None => fused,
        };

        if verbose {
            eprintln!("[rlx-metal] after fusion: {} nodes", fused.len());
        }

        // Memory plan with GPU-aligned cache lines (128B on Apple Silicon)
        let gdn_scratch = gdn_ephemeral_state_bytes(&fused);
        let dequant_scratch = dequant_gguf_scratch_bytes(&fused);
        let conv_bwd_scratch = conv_bwd_scratch_bytes(&fused);
        let attn_bwd_scratch = crate::attention_bwd_gpu::scratch_bytes(&fused);
        let rms_norm_bwd_scratch = rms_norm_bwd_scratch_bytes(&fused);
        let mut plan = memory::plan_memory_aligned(&fused, 128);
        let mut tail = plan.arena_size;
        let gdn_scratch_off = if gdn_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + gdn_scratch;
            off
        } else {
            0
        };
        let dequant_scratch_off = if dequant_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + dequant_scratch;
            off
        } else {
            0
        };
        let conv_bwd_scratch_off = if conv_bwd_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + conv_bwd_scratch;
            off
        } else {
            0
        };
        let attn_bwd_scratch_off = if attn_bwd_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + attn_bwd_scratch;
            off
        } else {
            0
        };
        let rms_norm_bwd_scratch_off = if rms_norm_bwd_scratch > 0 {
            tail = (tail + 127) & !127;
            let off = tail;
            tail = off + rms_norm_bwd_scratch;
            off
        } else {
            0
        };
        plan.arena_size = tail;
        if verbose && gdn_scratch > 0 {
            eprintln!(
                "[rlx-metal] GatedDeltaNet scratch: {} bytes @ offset {}",
                gdn_scratch, gdn_scratch_off
            );
        }
        if verbose && dequant_scratch > 0 {
            eprintln!(
                "[rlx-metal] DequantMatMul scratch: {} bytes @ offset {}",
                dequant_scratch, dequant_scratch_off
            );
        }
        if verbose && conv_bwd_scratch > 0 {
            eprintln!(
                "[rlx-metal] Conv2dBackwardWeight scratch: {} bytes @ offset {}",
                conv_bwd_scratch, conv_bwd_scratch_off
            );
        }
        if verbose && attn_bwd_scratch > 0 {
            eprintln!(
                "[rlx-metal] AttentionBackward scratch: {} bytes @ offset {}",
                attn_bwd_scratch, attn_bwd_scratch_off
            );
        }
        if verbose && rms_norm_bwd_scratch > 0 {
            eprintln!(
                "[rlx-metal] RmsNormBackward param scratch: {} bytes @ offset {}",
                rms_norm_bwd_scratch, rms_norm_bwd_scratch_off
            );
        }
        if verbose {
            eprintln!(
                "[rlx-metal] arena: {} bytes, {} buffers",
                plan.arena_size,
                plan.assignments.len()
            );
        }
        // Build precision-aware arena: per-node DType drives buffer sizing
        // and downstream kernel dispatch.
        let arena = Arena::from_plan_with_graph(plan, Some(&fused));

        // Initialize `Op::Constant` slots with their literal data. The
        // arena is shared-storage MTLBuffer (unified memory on Apple
        // Silicon) so we can write directly via `contents()`. F64 + I32 +
        // similar non-F32 dtypes go in as raw bytes; F32 also as raw
        // bytes (a constant's `data` field is little-endian dtype-native
        // already). Without this step, custom-op kernels reading from
        // a Constant input slot see zeros.
        for node in fused.nodes() {
            if let Op::Constant { data } = &node.op
                && !data.is_empty()
                && arena.has_buffer(node.id)
            {
                let off = arena.byte_offset(node.id);
                unsafe {
                    let dst = (arena.buffer.contents() as *mut u8).add(off);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                }
            }
        }

        let schedule = ThunkSchedule::compile(&fused, &arena);

        if verbose {
            let nop_count = schedule
                .thunks
                .iter()
                .filter(|t| matches!(t, crate::thunk::Thunk::Nop))
                .count();
            eprintln!(
                "[rlx-metal] schedule: {} thunks ({} compute, {} nop)",
                schedule.thunks.len(),
                schedule.thunks.len() - nop_count,
                nop_count
            );
        }

        let mut input_ids = HashMap::new();
        let mut param_ids = HashMap::new();
        for node in fused.nodes() {
            match &node.op {
                Op::Input { name } => {
                    input_ids.insert(name.clone(), node.id);
                }
                Op::Param { name } => {
                    param_ids.insert(name.clone(), node.id);
                }
                _ => {}
            }
        }

        let output_slots: Vec<(usize, usize)> = fused
            .outputs
            .iter()
            .map(|&id| {
                let off = if arena.has_buffer(id) {
                    arena.byte_offset(id)
                } else {
                    0
                };
                let logical = fused.node(id).shape.num_elements().unwrap_or(0);
                (off, logical)
            })
            .collect();

        // Pre-resolve input slots in graph-input order
        let mut input_slots = Vec::new();
        for node in fused.nodes() {
            if let Op::Input { name } = &node.op {
                let off = if arena.has_buffer(node.id) {
                    arena.byte_offset(node.id)
                } else {
                    0
                };
                let len = node.shape.num_elements().unwrap_or(0);
                input_slots.push((name.clone(), off, len));
            }
        }

        // MPSGraph lowering: on by default whenever every op is
        // supported by the bridge. Apple's fused MPSGraph kernels
        // outperform our per-op MSL encoder across the qwen3 prefill
        // range once RmsNorm + SDPA are wired (see mps_graph.rs).
        // Opt out with RLX_DISABLE_MPSGRAPH=1.
        let mps_plan = if rlx_ir::env::flag("RLX_DISABLE_MPSGRAPH") {
            None
        } else {
            let plan = crate::mps_graph_lower::try_lower(&fused);
            if verbose {
                match &plan {
                    Some(_) => eprintln!("[rlx-metal] MPSGraph lowering: success"),
                    None => eprintln!(
                        "[rlx-metal] MPSGraph lowering: unsupported op or dynamic shape; falling back to thunks"
                    ),
                }
            }
            plan
        };
        let mps_hybrid = if mps_plan.is_none()
            && rlx_ir::env::is_unset("RLX_DISABLE_MPSGRAPH")
            && rlx_ir::env::is_unset("RLX_DISABLE_MPSGRAPH_HYBRID")
        {
            crate::mps_graph_hybrid::build_hybrid_plan(&fused, None)
                .filter(|steps| crate::mps_graph_hybrid::hybrid_has_mps(steps))
        } else {
            None
        };
        if verbose && mps_hybrid.is_some() {
            eprintln!("[rlx-metal] MPSGraph hybrid lowering: enabled");
        }

        // Optional ICB pre-encoding: opt-in via env var. Pre-encodes the
        // ICB-compatible thunks (small element-wise / norm / copy ops) into
        // an IndirectCommandBuffer at compile time so encode_and_run can
        // issue them as one `executeCommandsInBuffer` call instead of N
        // individual `set_pipeline + set_buffer + dispatch` round-trips.
        let icb_segments = if rlx_ir::env::flag("RLX_USE_ICB") {
            let dev_ref = metal_device().expect("Metal device required");
            let segs =
                crate::icb::compile_segments(&schedule.thunks, &arena.buffer, &dev_ref.device);
            if verbose {
                let total_cmds: u64 = segs.iter().map(|r| r.segment.command_count).sum();
                eprintln!(
                    "[rlx-metal] ICB pre-encoded {} segments / {} commands",
                    segs.len(),
                    total_cmds
                );
            }
            segs
        } else {
            Vec::new()
        };

        let max_matmul_flops = max_matmul_flops_in(&fused);

        let mut me = Self {
            graph: fused,
            arena,
            schedule,
            input_ids,
            param_ids,
            input_slots,
            output_slots,
            precision: MetalPrecision::F32,
            mps_plan,
            mps_hybrid,
            icb_segments,
            pending_cmd_bufs: Vec::new(),
            active_extent: None,
            max_matmul_flops,
            mps_params_frozen: false,
            gdn_scratch_off,
            dequant_scratch_off,
            conv_bwd_scratch_off,
            attn_bwd_scratch_off,
            rms_norm_bwd_scratch_off,
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: std::collections::HashSet::new(),
        };
        // Bind the MPSGraph executable's input/output arrays to the
        // arena once. After this, run_cached() avoids all per-call
        // ObjC allocation. Arena buffer + per-node byte offsets are
        // fixed across runs, so the cached arrays stay valid for the
        // lifetime of `me`.
        me.bind_mps_executable_to_arena();
        me
    }

    /// Re-lower the MPSGraph plan, baking every param's current arena
    /// bytes in as a graph constant. After this call, the executable's
    /// feed list contains only the model's `Input`s — params are
    /// frozen into the compiled binary.
    ///
    /// Idempotent: a second call rebuilds against whatever bytes are
    /// in the arena now. Callers run this AFTER `set_param` has
    /// uploaded every weight (typical sequence: compile → set_param ×
    /// N → freeze → run × M). Triggered automatically on the first
    /// `run()` unless disabled with `RLX_DISABLE_MPSGRAPH_PARAM_CONST=1`.
    pub fn freeze_params_to_mps_constants(&mut self) {
        if self.mps_plan.is_none() && self.mps_hybrid.is_none() {
            return;
        }

        // Snapshot each param's current bytes from the arena. We only
        // freeze F32 params for now — typed-param plumbing (F16/BF16)
        // is a separate workstream; mixed-dtype paths stay on
        // placeholders for those.
        //
        // Size cap: `constantWithData:` ends up retained inside the
        // MPSGraphExecutable and never aliases the arena buffer, so
        // every baked constant is a fresh allocation outside our
        // arena. The qwen3 LM head weight alone is ~600 MB, and
        // compiling for multiple (B, L, mode) cells multiplies that.
        // Cap at 32 MB per param — large enough to bake all per-layer
        // projections, small enough to skip the LM head & token
        // embedding tables. Override with RLX_MPSGRAPH_PARAM_CONST_CAP=N
        // (bytes; 0 disables the cap).
        let cap_bytes = rlx_ir::env::var("RLX_MPSGRAPH_PARAM_CONST_CAP")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4 * 1024 * 1024);
        let arena_ptr = self.arena.buffer.contents() as *const u8;
        let mut param_bytes: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, id) in &self.param_ids {
            let node = self.graph.node(*id);
            if matches!(node.shape.dtype(), rlx_ir::DType::F32) {
                let n_elem = match node.shape.num_elements() {
                    Some(n) => n,
                    None => continue,
                };
                let len_bytes = n_elem * 4;
                if cap_bytes != 0 && len_bytes > cap_bytes {
                    continue;
                }
                let off = self.arena.byte_offset(*id);
                let bytes: Vec<u8> =
                    unsafe { std::slice::from_raw_parts(arena_ptr.add(off), len_bytes).to_vec() };
                param_bytes.insert(name.clone(), bytes);
                continue;
            }
            if !matches!(node.shape.dtype(), rlx_ir::DType::U8) {
                continue;
            }
            let Some((k, n, scheme)) = gguf_dequant_dims_for_param(&self.graph, *id) else {
                continue;
            };
            let u8_len = match node.shape.num_elements() {
                Some(n) => n,
                None => continue,
            };
            let f32_len = k * n * 4;
            if cap_bytes != 0 && f32_len > cap_bytes {
                continue;
            }
            let off = self.arena.byte_offset(*id);
            let u8_slice: &[u8] = unsafe { std::slice::from_raw_parts(arena_ptr.add(off), u8_len) };
            let dequant = rlx_cpu::dequant_cache::gguf_weight_f32(off, u8_slice, k, n, scheme);
            let kn_bytes = transpose_nk_to_kn_bytes(&dequant, n, k);
            param_bytes.insert(name.clone(), kn_bytes);
        }

        // Re-run lowering with the params marked as constants. Old
        // plan is dropped, which releases the old executable and
        // cached arrays.
        let new_plan =
            crate::mps_graph_lower::try_lower_with_constants(&self.graph, Some(&param_bytes));
        if let Some(plan) = new_plan {
            self.mps_plan = Some(plan);
            self.mps_hybrid = None;
            // Re-bind the (now much smaller) feed list to the arena.
            self.bind_mps_executable_to_arena();
        } else if self.mps_plan.is_none() {
            self.mps_hybrid =
                crate::mps_graph_hybrid::build_hybrid_plan(&self.graph, Some(&param_bytes))
                    .filter(|steps| crate::mps_graph_hybrid::hybrid_has_mps(steps));
        }
    }

    fn bind_mps_executable_to_arena(&mut self) {
        let Some(plan) = self.mps_plan.as_mut() else {
            return;
        };
        let Some(exec) = plan.executable.as_mut() else {
            return;
        };
        let arena_buf = &self.arena.buffer;

        let mut feed_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut feed_offsets: Vec<usize> = Vec::new();
        let mut feed_shapes: Vec<Vec<usize>> = Vec::new();
        let mut feed_dtypes: Vec<u32> = Vec::new();
        for (name, _t, shape, dt) in &plan.inputs {
            let id = self.input_ids.get(name).expect("input id");
            feed_buffers.push(arena_buf);
            feed_offsets.push(self.arena.byte_offset(*id));
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }
        for (name, _t, shape, dt) in &plan.params {
            let id = self.param_ids.get(name).expect("param id");
            feed_buffers.push(arena_buf);
            feed_offsets.push(self.arena.byte_offset(*id));
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }

        let mut out_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut out_offsets: Vec<usize> = Vec::new();
        let mut out_shapes: Vec<Vec<usize>> = Vec::new();
        let mut out_dtypes: Vec<u32> = Vec::new();
        for (id, _t, shape, dt) in &plan.outputs {
            out_buffers.push(arena_buf);
            out_offsets.push(self.arena.byte_offset(*id));
            out_shapes.push(shape.clone());
            out_dtypes.push(*dt);
        }

        exec.bind_arena(
            &feed_buffers,
            &feed_offsets,
            &feed_shapes,
            &feed_dtypes,
            &out_buffers,
            &out_offsets,
            &out_shapes,
            &out_dtypes,
        );
    }

    fn estimated_max_flops(&self) -> u64 {
        self.max_matmul_flops
    }

    /// Fastest path: inputs by slot index. Outputs are read directly from
    /// the shared arena buffer (zero-copy on Apple Silicon unified memory).
    pub fn run_slots(&mut self, inputs: &[&[f32]]) -> &[(usize, usize)] {
        if crate::mps_profile::enabled() {
            crate::mps_profile::reset();
        }
        unsafe {
            let buf_ptr = self.arena.buffer.contents() as *mut u8;
            for (i, &data) in inputs.iter().enumerate() {
                if i < self.input_slots.len() {
                    let (_, off, max_len) = &self.input_slots[i];
                    let len = data.len().min(*max_len);
                    let dst = buf_ptr.add(*off) as *mut f32;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, len);
                }
            }
        }
        self.encode_and_run();
        if crate::mps_profile::enabled() {
            crate::mps_profile::print_summary();
        }
        &self.output_slots
    }

    pub fn arena_ptr(&self) -> *const u8 {
        self.arena.buffer.contents() as *const u8
    }

    /// Read each graph-output's arena region as raw bytes in its
    /// declared dtype. Caller is responsible for ensuring the latest
    /// `run()` / `encode_and_run()` has completed (the caller-facing
    /// methods all wait_until_completed before returning, so this
    /// is true after any of them).
    ///
    /// Used by `MetalExecutableWrapper::run_typed` to avoid the
    /// f32 round-trip on F64 outputs — the f32 path narrows F64
    /// arena bytes to f32 (lossy) before widening them back to F64
    /// bytes for the typed-output contract.
    pub fn output_bytes_per_node(&self) -> Vec<Vec<u8>> {
        let base = self.arena.buffer.contents() as *const u8;
        self.graph
            .outputs
            .iter()
            .map(|&id| {
                let off = if self.arena.has_buffer(id) {
                    self.arena.byte_offset(id)
                } else {
                    0
                };
                let n_elems = self.graph.node(id).shape.num_elements().unwrap_or(0);
                let dt = self.graph.node(id).shape.dtype();
                let n_bytes = n_elems * dt.size_bytes();
                unsafe { std::slice::from_raw_parts(base.add(off), n_bytes).to_vec() }
            })
            .collect()
    }

    /// Encode + commit a forward pass without waiting for GPU completion.
    ///
    /// Use this to pipeline N runs and amortize the per-commit GPU sync
    /// latency (~150 µs on Apple Silicon). Caller MUST drain via
    /// `sync_pending` before reading any output (the arena is shared
    /// across pending commits, so output values are undefined until
    /// the GPU has caught up).
    ///
    /// Typical use: throughput benchmarks. Real-inference callers usually
    /// want `run` instead — pipelining requires per-commit output buffers
    /// or accepting that intermediate runs' outputs are stomped.
    pub fn commit_no_wait(&mut self, inputs: &[(&str, &[f32])]) {
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        // Outputs go to the shared arena — caller is responsible for not
        // reading until sync_pending() AND for tolerating intermediate
        // commits stomping the output region. Use run_pipelined() if you
        // need outputs from each individual commit.
        if let Some(cmd_buf) = self.encode_commit(false, None, None) {
            self.pending_cmd_bufs.push(cmd_buf);
        }
    }

    /// High-throughput batch inference with per-run output snapshots.
    ///
    /// Issues one commit per input set, deferring all waits, then waits
    /// once at the end. Unlike `commit_no_wait`, this allocates a
    /// per-commit output buffer and encodes a blit so each in-flight run's
    /// outputs survive subsequent commits stomping the shared arena.
    ///
    /// Returns outputs in commit order: `out[run_idx][output_idx][element_idx]`.
    pub fn run_pipelined(&mut self, input_sets: &[Vec<(&str, &[f32])>]) -> Vec<Vec<Vec<f32>>> {
        if input_sets.is_empty() {
            return Vec::new();
        }
        let dev = metal_device().expect("Metal device required");

        // Snapshot output sizes once so per-commit allocation doesn't
        // conflict with the &mut self that encode_commit needs.
        let out_sizes: Vec<usize> = self
            .output_slots
            .iter()
            .map(|(_, len)| (*len).max(1) * 4)
            .collect();

        let mut pending: Vec<(metal::CommandBuffer, Vec<metal::Buffer>)> =
            Vec::with_capacity(input_sets.len());

        for inputs in input_sets {
            // Write inputs into the shared arena. Subsequent commits will
            // overwrite these — fine since each run's compute consumes
            // its inputs before the next commit's writes.
            for &(name, data) in inputs {
                if let Some(&id) = self.input_ids.get(name)
                    && self.arena.has_buffer(id)
                {
                    self.arena.write_from_f32(id, data);
                }
            }
            // Allocate per-commit output buffers. Shared storage so the
            // read-back at the end is just a pointer cast on Apple
            // unified memory (no GPU→CPU copy).
            let dests: Vec<metal::Buffer> =
                out_sizes.iter().map(|&b| dev.alloc_shared(b)).collect();
            if let Some(cmd_buf) = self.encode_commit(false, Some(&dests), None) {
                pending.push((cmd_buf, dests));
            }
        }

        // Single sync at the end. Metal queues are FIFO so waiting on the
        // last buffer guarantees all prior commits have completed.
        if let Some((last, _)) = pending.last() {
            last.wait_until_completed();
        }

        // Read back. Apple unified memory → contents() points at the same
        // bytes the GPU wrote.
        pending
            .into_iter()
            .map(|(_cb, bufs)| {
                bufs.into_iter()
                    .enumerate()
                    .map(|(i, buf)| {
                        let len = self.output_slots[i].1;
                        unsafe {
                            std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec()
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Wait for every command buffer queued by `commit_no_wait`.
    pub fn sync_pending(&mut self) {
        for cb in self.pending_cmd_bufs.drain(..) {
            cb.wait_until_completed();
        }
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_ids.get(name)
            && self.arena.has_buffer(id)
        {
            // Converts to f16 if the param node's dtype is F16.
            self.arena.write_from_f32(id, data);
        }
    }

    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_ids.get(name)
            && self.arena.has_buffer(id)
        {
            self.arena.write_bytes(id, data);
        }
    }

    /// Hint the next `run` to process only the first `actual` rows
    /// along the bucket axis (out of `upper`, the compile extent).
    /// Honored when every thunk in the schedule passes
    /// `Thunk::safe_for_active_extent`; otherwise falls back to
    /// full-extent. See PLAN L1.
    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    /// Declared graph-output dtypes, in `graph.outputs` order. Used by
    /// the runtime wrapper's `run_typed` to narrow the f32 outputs back
    /// to F16/BF16/etc. on the way out, mirroring what backends with
    /// native-dtype storage emit.
    pub fn output_dtypes(&self) -> Vec<rlx_ir::DType> {
        self.graph
            .outputs
            .iter()
            .map(|&id| self.graph.node(id).shape.dtype())
            .collect()
    }

    /// True when every thunk in the schedule is safe for active-extent
    /// dispatch — guards `encode_commit`'s bypass of MPSGraph + ICB.
    fn all_safe_for_active(&self) -> bool {
        self.schedule
            .thunks
            .iter()
            .all(|t| t.safe_for_active_extent())
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    /// Read one graph output at IR logical length (matches CPU `read_output`).
    ///
    /// Bucket-padded arena slots may be larger than the declared output
    /// shape; embeds decode graphs write the active row at index 0.
    fn read_graph_output_f32(&self, out_idx: usize) -> Vec<f32> {
        let id = self.graph.outputs[out_idx];
        let logical_len = self.output_slots[out_idx].1;
        let full = self.arena.read_as_f32(id);
        if logical_len == 0 || full.len() <= logical_len {
            return full;
        }
        full[..logical_len].to_vec()
    }

    /// Read one row from a row-major graph output (bucketed decode K/V).
    pub fn read_graph_output_row(&self, out_idx: usize, row: usize, row_inner: usize) -> Vec<f32> {
        let id = self.graph.outputs[out_idx];
        let start = row * row_inner;
        let end = start + row_inner;
        if self.arena.dtype(id) == rlx_ir::DType::F32 {
            let slice = self.arena.slice(id);
            assert!(
                end <= slice.len(),
                "read_graph_output_row: out={out_idx} row={row} inner={row_inner} need {end} f32, have {}",
                slice.len()
            );
            slice[start..end].to_vec()
        } else {
            let full = self.arena.read_as_f32(id);
            assert!(
                end <= full.len(),
                "read_graph_output_row: out={out_idx} truncated"
            );
            full[start..end].to_vec()
        }
    }

    /// Run and read back only selected graph outputs (e.g. logits-only decode).
    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        if crate::mps_profile::enabled() {
            crate::mps_profile::reset();
        }
        for (name, data) in &self.gpu_handles {
            if self.gpu_handle_resident.contains(name) || inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        self.encode_and_run();
        if !self.gpu_handle_feeds.is_empty() {
            self.propagate_gpu_handle_feeds_in_arena();
            if read_indices.is_none() || rlx_ir::env::flag("RLX_GPU_HANDLE_HOST_MIRROR") {
                self.refresh_gpu_handles_from_outputs();
            }
        }
        if crate::mps_profile::enabled() {
            crate::mps_profile::print_summary();
        }
        let n_out = self.graph.outputs.len();
        let indices: Vec<usize> = match read_indices {
            None => (0..n_out).collect(),
            Some(ix) => ix.to_vec(),
        };
        indices
            .iter()
            .map(|&i| self.read_graph_output_f32(i))
            .collect()
    }

    /// Persistent input buffer for KV-cache style graphs (unified memory).
    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        if !self.input_ids.contains_key(name) {
            return false;
        }
        self.gpu_handle_resident.remove(name);
        // Reuse arena slot when capacity matches (megakernel bucket reinstall).
        if let Some(&id) = self.input_ids.get(name) {
            let cap = *self.arena.element_counts.get(&id).unwrap_or(&0);
            if self.arena.has_buffer(id) && cap == data.len() {
                self.arena.write_from_f32(id, data);
                self.gpu_handles.insert(name.to_string(), Vec::new());
                return true;
            }
        }
        self.gpu_handles.insert(name.to_string(), data.to_vec());
        true
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    pub fn set_gpu_handle_feed(&mut self, handle_name: &str, output_index: usize) {
        self.gpu_handle_feeds
            .insert(handle_name.to_string(), output_index);
    }

    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(&out_idx) = self.gpu_handle_feeds.get(name) {
            if out_idx < self.graph.outputs.len() {
                return Some(self.read_graph_output_f32(out_idx));
            }
        }
        if self.gpu_handle_resident.contains(name) {
            if let Some(&id) = self.input_ids.get(name) {
                return Some(self.arena.read_as_f32(id));
            }
        }
        self.gpu_handles.get(name).cloned()
    }

    fn propagate_gpu_handle_feeds_in_arena(&mut self) {
        let extent = self.active_extent;
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                let out_elems = *self.arena.element_counts.get(&out_id).unwrap_or(&0);
                let copy_elems = match extent {
                    Some((actual, upper)) if upper > 0 => actual * (out_elems / (upper + 1)).max(1),
                    _ => out_elems,
                };
                self.arena
                    .copy_node_f32_prefix(in_id, out_id, copy_elems.min(out_elems));
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    fn refresh_gpu_handles_from_outputs(&mut self) {
        for (name, &out_idx) in &self.gpu_handle_feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let id = self.graph.outputs[out_idx];
            let src = self.arena.slice(id);
            let entry = self
                .gpu_handles
                .entry(name.clone())
                .or_insert_with(|| vec![0.0; src.len()]);
            if entry.len() != src.len() {
                entry.resize(src.len(), 0.0);
            }
            entry.copy_from_slice(src);
        }
    }

    fn encode_and_run(&mut self) {
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
        if !self.mps_params_frozen
            && (self.mps_plan.is_some() || self.mps_hybrid.is_some())
            && rlx_ir::env::flag("RLX_MPSGRAPH_PARAM_CONST")
        {
            self.freeze_params_to_mps_constants();
            self.mps_params_frozen = true;
        }

        // Active-extent (PLAN L1): when set + every thunk safe, bypass
        // MPSGraph + ICB (both pre-encode at full extent) and dispatch
        // per-op with scaled launch dims via encode_commit.
        let active_safe = self.active_extent.is_some() && self.all_safe_for_active();
        if !active_safe && self.mps_plan.is_some() {
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
            if force || self.estimated_max_flops() >= threshold {
                let t0 = Instant::now();
                self.run_via_mps_graph();
                crate::mps_profile::record("encode_path:mps_graph_full", t0.elapsed());
                return;
            }
        }
        if !active_safe
            && self.mps_plan.is_none()
            && self
                .mps_hybrid
                .as_ref()
                .is_some_and(|steps| crate::mps_graph_hybrid::hybrid_has_mps(steps))
        {
            let force = rlx_ir::env::flag("RLX_MPSGRAPH_FORCE");
            let threshold = rlx_ir::env::var("RLX_MPSGRAPH_MIN_FLOPS")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1_000_000);
            if force || self.estimated_max_flops() >= threshold {
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

    /// Sequential per-thunk GPU timing (`RLX_METAL_THUNK_PROFILE=1`).
    fn run_thunk_profile(&mut self) {
        use std::time::Instant;
        crate::thunk_profile::reset();
        let n = self.schedule.thunks.len();
        for i in 0..n {
            let name = crate::thunk::thunk_name(&self.schedule.thunks[i]);
            if name == "nop" {
                continue;
            }
            let t0 = Instant::now();
            let _ = self.encode_commit(true, None, Some(i..i + 1));
            crate::thunk_profile::record(name, t0.elapsed());
        }
        crate::thunk_profile::print_summary();
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
    fn encode_commit(
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
            /// Unified-memory memcpy — batched to avoid GPU dispatch on slices.
            Memcpy {
                src: usize,
                dst: usize,
                bytes: usize,
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
                                    arena_ptr,
                                );
                            }
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
                        DeferredHostOp::Memcpy { src, dst, bytes } => unsafe {
                            if bytes > 0 {
                                std::ptr::copy_nonoverlapping(
                                    arena_ptr.add(src),
                                    arena_ptr.add(dst),
                                    bytes,
                                );
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
                    enc = Some(
                        cmd_buf
                            .compute_command_encoder_with_dispatch_type(
                                metal::MTLDispatchType::Serial,
                            )
                            .to_owned(),
                    );
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
        let mut seg_iter = segments.iter().peekable();
        let loop_end = thunk_range.as_ref().map(|r| r.end).unwrap_or(thunks.len());
        let mut i = thunk_range.as_ref().map(|r| r.start).unwrap_or(0);
        while i < loop_end {
            if thunk_range.is_none()
                && active.is_none()
                && let Some(range) = seg_iter.peek()
                && range.start == i
            {
                range.segment.execute_on(e!(), &self.arena.buffer);
                i = range.end;
                seg_iter.next();
                continue;
            }
            let thunk = &thunks[i];
            i += 1;
            if !matches!(thunk, Thunk::Narrow { .. } | Thunk::SplitLastAxis { .. })
                && narrow_batch.is_some()
            {
                flush_pending_narrow_batch(e!(), k, &self.arena.buffer, &mut narrow_batch);
            }
            // PLAN L3: per-thunk Perfetto span. No-op when env var
            // RLX_TRACE_PERFETTO unset.
            let _span = rlx_ir::perfetto::TraceSpan::new(crate::thunk::thunk_name(thunk), "metal");
            match thunk {
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
                Thunk::Sgemm {
                    a,
                    b,
                    c,
                    m,
                    k: kk,
                    n,
                    dt,
                } => {
                    use crate::thunk::HalfFlag;
                    let m_scaled = scale(*m);
                    if m_scaled == 0 {
                        continue;
                    }
                    let (mu, ku, nu) = (m_scaled as usize, *kk as usize, *n as usize);
                    let use_mps = crate::cost::hw_model().pick_sgemm(mu, ku, nu)
                        == crate::cost::SgemmVariant::Mps;
                    if use_mps && matches!(dt, HalfFlag::F16) {
                        end_msl!();
                        crate::mps_blas::encode_mps_hgemm(
                            &cmd_buf,
                            &self.arena.buffer,
                            *a,
                            *b,
                            *c,
                            mu,
                            ku,
                            nu,
                        );
                    } else if use_mps {
                        end_msl!();
                        crate::mps_blas::encode_mps_sgemm(
                            &cmd_buf,
                            &self.arena.buffer,
                            *a,
                            *b,
                            *c,
                            mu,
                            ku,
                            nu,
                        );
                    } else if matches!(dt, HalfFlag::F16) {
                        crate::blas::metal_hgemm(e!(), &self.arena.buffer, *a, *b, *c, mu, ku, nu);
                    } else {
                        crate::blas::metal_sgemm(e!(), &self.arena.buffer, *a, *b, *c, mu, ku, nu);
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
                    let (mu, ku, nu) = (m_scaled as usize, *kk as usize, *n as usize);
                    let use_mps = crate::cost::hw_model().pick_sgemm(mu, ku, nu)
                        == crate::cost::SgemmVariant::Mps;
                    if use_mps {
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
                        encode_bias_add(e!(), k, &self.arena.buffer, *c, *bias, m_scaled, *n, *dt);
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
                Thunk::ActivationInPlace { data, len, act, dt } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_activation(e!(), k, &self.arena.buffer, *data, len, *act, *dt);
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
                    if !matches!(dt, HalfFlag::F32) {
                        continue;
                    }
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
                    if !matches!(dt, HalfFlag::F32) {
                        continue;
                    }
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
                    encode_bias_add(e!(), k, &self.arena.buffer, *dst, *bias, m_scaled, *n, *dt);
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
                    encode_binary(e!(), k, &self.arena.buffer, *lhs, *rhs, *dst, len, *op, *dt);
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
                    let a_stride = mu * ku * elem;
                    let b_stride = ku * nu * elem;
                    let c_stride = mu * nu * elem;
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
                        let op_id: u32 = match op {
                            rlx_ir::op::BinaryOp::Add => 0,
                            rlx_ir::op::BinaryOp::Sub => 1,
                            rlx_ir::op::BinaryOp::Mul => 2,
                            rlx_ir::op::BinaryOp::Div => 3,
                            rlx_ir::op::BinaryOp::Max => 4,
                            rlx_ir::op::BinaryOp::Min => 5,
                            rlx_ir::op::BinaryOp::Pow => 6,
                        };
                        let enc = e!();
                        if let Some(rhs_scalar) =
                            detect_scalar_broadcast(*rank, out_dims, lhs_strides, rhs_strides)
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
                        ) {
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
                        ) {
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
                    encode_gather(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *table,
                        *idx,
                        *dst,
                        num_idx,
                        *trailing,
                        *dt,
                    );
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
                        let src_off = *src + (*start as usize) * std::mem::size_of::<f32>();
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
                    if metal_host_slices_enabled() && matches!(*dt, crate::thunk::HalfFlag::F32) {
                        deferred_host.push(DeferredHostOp::Memcpy {
                            src: *src,
                            dst: *dst,
                            bytes: len as usize * std::mem::size_of::<f32>(),
                        });
                    } else {
                        encode_copy(e!(), k, &self.arena.buffer, *src, *dst, len, *dt);
                    }
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
                    end_msl!();
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                    let arena_ptr = self.arena.buffer.contents() as *mut u8;
                    unsafe {
                        rlx_cpu::thunk::execute_maxpool2d_backward_f32(
                            *x, *dy, *dx, n_eff, *c, *h, *w, *h_out, *w_out, *kh, *kw, *sh, *sw,
                            *ph, *pw, arena_ptr,
                        );
                    }
                    cmd_buf = dev.queue.new_command_buffer().to_owned();
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
                    // Same lowering as decomposed `Op::Conv(dy, w)` → `Conv2D`.
                    encode_conv2d(
                        e!(),
                        k,
                        &self.arena.buffer,
                        *dy,
                        *w,
                        *dx,
                        n,
                        *c_out,
                        *h_out,
                        *w_out,
                        *c_in,
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
                    if n > 1 || self.conv_bwd_scratch_off == 0 {
                        unsafe {
                            let base = self.arena.buffer.contents() as *mut u8;
                            rlx_cpu::conv_bwd::execute_conv2d_backward_weight_f32(
                                base, *x, *dy, *dw, n, *c_in, *h, *w, *c_out, *h_out, *w_out, *kh,
                                *kw, *sh, *sw, *ph, *pw, *dh, *dw_dil, *groups,
                            );
                        }
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
                    head_dim,
                    mask_kind,
                    window,
                    dt,
                    bhsd,
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
                        *head_dim,
                        *dt,
                        seq_stride,
                        *mask_kind,
                        *window,
                        kv_seq_eff,
                        kv_stride,
                        *bhsd,
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
                } => {
                    let outer = scale(*outer);
                    if outer == 0 {
                        continue;
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
                } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_compare(e!(), k, &self.arena.buffer, *lhs, *rhs, *dst, len, *op);
                }
                Thunk::Where {
                    cond,
                    on_true,
                    on_false,
                    dst,
                    len,
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
                    if is_2d_swap {
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
                            let data: &[u8] =
                                unsafe { std::slice::from_raw_parts(arena_ptr.add(*off), n_bytes) };
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
                } => {
                    // Native multi-kernel MSL path: f32 + power-of-2 N≥2.
                    // f64/C64 and non-pow2 fall through to host CPU FFT.
                    // Set RLX_METAL_FFT_HOST_FALLBACK=1 to force host path.
                    let force_host = rlx_ir::env::flag("RLX_METAL_FFT_HOST_FALLBACK");
                    let n = *n_complex as usize;
                    let can_native = !force_host
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
                } => {
                    // Native MSL kernel supports f32 with n ≤ 128 (qwen35 uses 128).
                    // GPU path is the default; opt out with RLX_METAL_GDN_CPU=1 or
                    // RLX_METAL_GDN_HOST_FALLBACK=1. f16 tensors and state_size > 128
                    // still take the CPU thunk regardless of the flag.
                    let force_host = rlx_ir::env::flag("RLX_METAL_GDN_HOST_FALLBACK")
                        || rlx_ir::env::flag("RLX_METAL_GDN_CPU");
                    let prefer_cpu_blas = false;
                    let use_carry = *state != 0;
                    let state_byte = if use_carry {
                        *state
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
                        });
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
                } => {
                    let m_u = *m as usize;
                    let k_u = *kk as usize;
                    let n_u = *n as usize;
                    let use_gpu_dequant = rlx_ir::env::flag("RLX_METAL_DEQUANT_GPU");
                    if !use_gpu_dequant || self.dequant_scratch_off == 0 {
                        deferred_host.push(DeferredHostOp::DequantMatMulGguf {
                            x: *x,
                            w_q: *w_q,
                            dst: *dst,
                            m: m_u,
                            k: k_u,
                            n: n_u,
                            scheme: *scheme,
                        });
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
                    let use_gpu_dequant = rlx_ir::env::flag("RLX_METAL_DEQUANT_GPU");
                    if !use_gpu_dequant || self.dequant_scratch_off == 0 {
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
                    } else {
                        let enc = e!();
                        encode_dequant_grouped_matmul_gguf(
                            &cmd_buf,
                            enc,
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
                        end_msl!();
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
                let bytes = (*len as u64) * 4;
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

    pub fn output_slots(&self) -> &[(usize, usize)] {
        &self.output_slots
    }

    /// Execute the graph via MPSGraph (set up by lowering at compile time).
    /// All inputs/params are bound to their respective arena offsets; outputs
    /// are written into the arena slots so downstream consumers (run_slots
    /// callers) see them as if a thunk schedule had run.
    fn run_via_mps_graph(&mut self) {
        use std::time::Instant;
        let plan = self.mps_plan.as_ref().expect("plan present");
        let t0 = Instant::now();
        self.dispatch_mps_plan(plan, None, None);
        crate::mps_profile::record("mps_graph:dispatch_full", t0.elapsed());
    }

    /// Interleaved MPS sub-graph + thunk dispatch for Qwen3.5 decode.
    fn run_via_mps_hybrid(&mut self) {
        use std::time::Instant;
        let n = self.mps_hybrid.as_ref().expect("hybrid plan present").len();
        for i in 0..n {
            if let crate::mps_graph_hybrid::HybridStep::Thunks(range) =
                &self.mps_hybrid.as_ref().unwrap()[i]
            {
                let r = range.clone();
                let t0 = Instant::now();
                let _ = self.encode_commit(true, None, Some(r));
                crate::mps_profile::record(format!("hybrid:thunks[{i}]"), t0.elapsed());
                continue;
            }
            if let crate::mps_graph_hybrid::HybridStep::SubGraph {
                plan,
                boundary_parent_ids,
                output_parent_ids,
                ..
            } = &self.mps_hybrid.as_ref().unwrap()[i]
            {
                let t0 = Instant::now();
                self.dispatch_mps_plan(plan, Some(boundary_parent_ids), Some(output_parent_ids));
                crate::mps_profile::record(format!("hybrid:mps_subgraph[{i}]"), t0.elapsed());
            }
        }
    }

    fn dispatch_mps_plan(
        &self,
        plan: &crate::mps_graph_lower::MpsGraphPlan,
        boundary_parent_ids: Option<&HashMap<String, NodeId>>,
        output_parent_ids: Option<&[(NodeId, NodeId)]>,
    ) {
        let dev = metal_device().expect("Metal device");
        let arena_buf = &self.arena.buffer;

        let mut feed_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut feed_offsets: Vec<usize> = Vec::new();
        let mut feed_shapes: Vec<Vec<usize>> = Vec::new();
        let mut feed_dtypes: Vec<u32> = Vec::new();

        for (name, _t, shape, dt) in &plan.inputs {
            let off = if name.starts_with("__boundary_") {
                let parent = boundary_parent_ids
                    .and_then(|m| m.get(name))
                    .expect("hybrid boundary input");
                self.arena.byte_offset(*parent)
            } else {
                let id = self.input_ids.get(name).expect("input id");
                self.arena.byte_offset(*id)
            };
            feed_buffers.push(arena_buf);
            feed_offsets.push(off);
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }
        for (name, _t, shape, dt) in &plan.params {
            let id = self.param_ids.get(name).expect("param id");
            feed_buffers.push(arena_buf);
            feed_offsets.push(self.arena.byte_offset(*id));
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }

        let mut out_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut out_offsets: Vec<usize> = Vec::new();
        let mut out_shapes: Vec<Vec<usize>> = Vec::new();
        let mut out_dtypes: Vec<u32> = Vec::new();
        if let Some(out_map) = output_parent_ids {
            for (sub_id, parent_id) in out_map {
                let off = self.arena.byte_offset(*parent_id);
                let (_, _t, shape, dt) = plan
                    .outputs
                    .iter()
                    .find(|(id, _, _, _)| id == sub_id)
                    .expect("hybrid output id");
                out_buffers.push(arena_buf);
                out_offsets.push(off);
                out_shapes.push(shape.clone());
                out_dtypes.push(*dt);
            }
        } else {
            for (id, _t, shape, dt) in &plan.outputs {
                out_buffers.push(arena_buf);
                out_offsets.push(self.arena.byte_offset(*id));
                out_shapes.push(shape.clone());
                out_dtypes.push(*dt);
            }
        }

        if let Some(exec) = plan.executable.as_ref() {
            if exec.has_cached_binding() {
                exec.run_cached(&dev.queue);
                return;
            }
            exec.run(
                &dev.queue,
                &feed_buffers,
                &feed_offsets,
                &feed_shapes,
                &feed_dtypes,
                &out_buffers,
                &out_offsets,
                &out_shapes,
                &out_dtypes,
            );
            return;
        }

        let feed_tensors: Vec<&crate::mps_graph::MpsTensor> = plan
            .inputs
            .iter()
            .map(|(_, t, _, _)| t)
            .chain(plan.params.iter().map(|(_, t, _, _)| t))
            .collect();
        let out_tensors: Vec<&crate::mps_graph::MpsTensor> =
            plan.outputs.iter().map(|(_, t, _, _)| t).collect();
        plan.graph.run_jit(
            &dev.queue,
            &feed_tensors,
            &feed_buffers,
            &feed_offsets,
            &feed_shapes,
            &feed_dtypes,
            &out_tensors,
            &out_buffers,
            &out_offsets,
            &out_shapes,
            &out_dtypes,
        );
    }
}

/// Largest `m·k·n` across every `Op::MatMul` and `Op::FusedMatMulBiasAct`
/// in the graph. Used by the MPSGraph adaptive-dispatch heuristic to
/// decide whether the per-call overhead is worth eating for this
/// workload.
fn max_matmul_flops_in(graph: &Graph) -> u64 {
    let mut best: u64 = 0;
    for node in graph.nodes() {
        if !matches!(node.op, Op::MatMul | Op::FusedMatMulBiasAct { .. }) {
            continue;
        }
        let out_shape = &node.shape;
        let n_dim = match out_shape.dim(out_shape.rank().saturating_sub(1)) {
            d if d.is_static() => d.unwrap_static(),
            _ => continue,
        };
        let out_total: usize = match out_shape.num_elements() {
            Some(v) => v,
            None => continue,
        };
        let m_dim = out_total / n_dim.max(1);
        let a_shape = &graph.node(node.inputs[0]).shape;
        let a_total: usize = match a_shape.num_elements() {
            Some(v) => v,
            None => continue,
        };
        let k_dim = a_total / m_dim.max(1);
        let flops = (m_dim as u64) * (k_dim as u64) * (n_dim as u64);
        if flops > best {
            best = flops;
        }
    }
    best
}

fn gguf_dequant_dims_for_param(
    graph: &Graph,
    param_id: NodeId,
) -> Option<(usize, usize, rlx_ir::quant::QuantScheme)> {
    for node in graph.nodes() {
        if let Op::DequantMatMul { scheme } = &node.op
            && node.inputs.get(1) == Some(&param_id)
        {
            let n = node
                .shape
                .dim(node.shape.rank().saturating_sub(1))
                .unwrap_static();
            let out_total = node.shape.num_elements()?;
            let m = out_total / n.max(1);
            let a_total = graph.node(node.inputs[0]).shape.num_elements()?;
            let k = a_total / m.max(1);
            return Some((k, n, *scheme));
        }
    }
    None
}

fn transpose_nk_to_kn_bytes(dequant: &[f32], n: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(k * n * 4);
    for p in 0..k {
        for j in 0..n {
            out.extend_from_slice(&dequant[j * k + p].to_le_bytes());
        }
    }
    out
}

// ── Host-side shape-aware broadcast (Apple Silicon unified memory) ──

/// Compute the in-buffer element count implied by a broadcast-stride
/// vector. A stride of 0 means "size 1" along that output axis (we
/// don't read past element 0 of that axis); a non-zero stride means
/// the axis size matches `out_dims[axis]`.
fn inferred_input_len(strides: &[u32], out_dims: &[u32]) -> usize {
    let mut acc: usize = 1;
    for d in 0..out_dims.len() {
        if strides[d] != 0 {
            acc *= out_dims[d] as usize;
        }
    }
    acc
}

/// Generic host-side binary broadcast. Walks the output index space,
/// decomposes into per-axis coords, and reads via the provided
/// broadcast strides (0 ⇒ replicate along that axis). Correctness-first
/// implementation — a proper MSL kernel would be a follow-on.
#[allow(clippy::too_many_arguments)]
unsafe fn binary_broadcast_host<T>(
    lhs: *const T,
    lhs_len: usize,
    rhs: *const T,
    rhs_len: usize,
    dst: *mut T,
    out_len: usize,
    rank: usize,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
    op: rlx_ir::op::BinaryOp,
) where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + PartialOrd,
{
    use rlx_ir::op::BinaryOp;
    let l = unsafe { std::slice::from_raw_parts(lhs, lhs_len) };
    let r = unsafe { std::slice::from_raw_parts(rhs, rhs_len) };
    let o = unsafe { std::slice::from_raw_parts_mut(dst, out_len) };
    for i in 0..out_len {
        // Decompose flat output index into per-axis coords.
        let mut rem = i;
        let mut li: usize = 0;
        let mut ri: usize = 0;
        for ax in (0..rank).rev() {
            let sz = out_dims[ax] as usize;
            let coord = rem % sz;
            rem /= sz;
            li += coord * lhs_strides[ax] as usize;
            ri += coord * rhs_strides[ax] as usize;
        }
        let lv = l[li];
        let rv = r[ri];
        o[i] = match op {
            BinaryOp::Add => lv + rv,
            BinaryOp::Sub => lv - rv,
            BinaryOp::Mul => lv * rv,
            BinaryOp::Div => lv / rv,
            BinaryOp::Max => {
                if lv >= rv {
                    lv
                } else {
                    rv
                }
            }
            BinaryOp::Min => {
                if lv <= rv {
                    lv
                } else {
                    rv
                }
            }
            BinaryOp::Pow => {
                // Generic Pow isn't expressible at the T trait level;
                // SAM doesn't need it on this code path. Fall back to
                // a panic to avoid silent wrong results.
                panic!("BinaryBroadcast Pow not implemented in host path");
            }
        };
    }
}

fn encode_cast(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
    src_dt: crate::thunk::HalfFlag,
    dst_dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match (src_dt, dst_dt) {
        (HalfFlag::F32, HalfFlag::F16) => &k.cast_f32_to_f16,
        (HalfFlag::F16, HalfFlag::F32) => &k.cast_f16_to_f32,
        // Same precision → plain copy (lets us stay on this compute encoder).
        // For F16→F16 we copy half the bytes by treating the buffer as f32
        // pairs (len f16s = len/2 f32s rounded up): use 2 elements per i.
        (a, b) if a == b => {
            let n = match a {
                HalfFlag::F32 => len,
                HalfFlag::F16 => len.div_ceil(2),
            };
            let p = &k.copy_f32;
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(buffer), src as u64);
            enc.set_buffer(1, Some(buffer), dst as u64);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
            let tg_w = p.thread_execution_width().min(n as u64);
            enc.dispatch_threads(
                metal::MTLSize {
                    width: n as u64,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: tg_w,
                    height: 1,
                    depth: 1,
                },
            );
            return;
        }
        _ => return,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &len as *const u32 as *const _);
    let tg_w = pipeline.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_bias_add(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    data_off: usize,
    bias_off: usize,
    m: u32,
    n: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.bias_add,
        HalfFlag::F16 => &k.bias_add_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), data_off as u64);
    enc.set_buffer(1, Some(buffer), bias_off as u64);
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &m as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &n as *const u32 as *const _,
    );
    let grid = metal::MTLSize {
        width: n as u64,
        height: m as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 16.min(n as u64),
        height: 16.min(m as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_fused_binary_activation(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: rlx_ir::op::BinaryOp,
    act: rlx_ir::op::Activation,
) {
    use rlx_ir::op::{Activation, BinaryOp};
    let bin_op: u32 = match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
    };
    let act_op: u32 = match act {
        Activation::Gelu | Activation::GeluApprox => 0,
        Activation::Silu => 1,
        Activation::Relu => 2,
        Activation::Sigmoid => 3,
        Activation::Tanh => 4,
        _ => 255,
    };
    let use_vec4 = len.is_multiple_of(4) && len >= 4;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.fused_binary_activation4);
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &bin_op as *const u32 as *const _);
        enc.set_bytes(5, 4, &act_op as *const u32 as *const _);
        let tg_w = k
            .fused_binary_activation4
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.fused_binary_activation_f32);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &bin_op as *const u32 as *const _);
    enc.set_bytes(5, 4, &act_op as *const u32 as *const _);
    let tg_w = k
        .fused_binary_activation_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_fused_ternary_activation(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs0: usize,
    rhs1: usize,
    dst: usize,
    len: u32,
    op0: rlx_ir::op::BinaryOp,
    op1: rlx_ir::op::BinaryOp,
    act: rlx_ir::op::Activation,
) {
    use rlx_ir::op::{Activation, BinaryOp};
    let bin_op = |op: BinaryOp| -> u32 {
        match op {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 1,
            BinaryOp::Mul => 2,
            BinaryOp::Div => 3,
            BinaryOp::Max => 4,
            BinaryOp::Min => 5,
            BinaryOp::Pow => 6,
        }
    };
    let bin_op0 = bin_op(op0);
    let bin_op1 = bin_op(op1);
    let act_op: u32 = match act {
        Activation::Gelu | Activation::GeluApprox => 0,
        Activation::Silu => 1,
        Activation::Relu => 2,
        Activation::Sigmoid => 3,
        Activation::Tanh => 4,
        _ => 255,
    };
    let use_vec4 = len.is_multiple_of(4) && len >= 4;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.fused_ternary_activation4);
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs0 as u64);
        enc.set_buffer(2, Some(buffer), rhs1 as u64);
        enc.set_buffer(3, Some(buffer), dst as u64);
        enc.set_bytes(4, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &bin_op0 as *const u32 as *const _);
        enc.set_bytes(6, 4, &bin_op1 as *const u32 as *const _);
        enc.set_bytes(7, 4, &act_op as *const u32 as *const _);
        let tg_w = k
            .fused_ternary_activation4
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.fused_ternary_activation_f32);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs0 as u64);
    enc.set_buffer(2, Some(buffer), rhs1 as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    enc.set_bytes(5, 4, &bin_op0 as *const u32 as *const _);
    enc.set_bytes(6, 4, &bin_op1 as *const u32 as *const _);
    enc.set_bytes(7, 4, &act_op as *const u32 as *const _);
    let tg_w = k
        .fused_ternary_activation_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_activation(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    data_off: usize,
    len: u32,
    act: rlx_ir::op::Activation,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    use rlx_ir::op::Activation;
    if matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(act, Activation::Gelu | Activation::GeluApprox)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.gelu_inplace4);
        enc.set_buffer(0, Some(buffer), data_off as u64);
        enc.set_bytes(1, 4, &len4 as *const u32 as *const _);
        let tg_w = k.gelu_inplace4.thread_execution_width().min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    if matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(act, Activation::Silu)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.silu_inplace4);
        enc.set_buffer(0, Some(buffer), data_off as u64);
        enc.set_bytes(1, 4, &len4 as *const u32 as *const _);
        let tg_w = k.silu_inplace4.thread_execution_width().min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    // f16 has h variants only for the activations Nomic actually uses
    // (Gelu, Silu). Other variants fall back to the f32 kernel — that's
    // a real correctness hole if a model uses them in mixed precision,
    // but no current burnembed model does.
    let pipeline = match (dt, act) {
        (HalfFlag::F16, Activation::Gelu) | (HalfFlag::F16, Activation::GeluApprox) => {
            &k.gelu_inplace_h
        }
        (HalfFlag::F16, Activation::Silu) => &k.silu_inplace_h,
        (_, Activation::Gelu) | (_, Activation::GeluApprox) => &k.gelu_inplace,
        (_, Activation::Silu) => &k.silu_inplace,
        (_, Activation::Relu) => &k.relu_inplace,
        (_, Activation::Sigmoid) => &k.sigmoid_inplace,
        (_, Activation::Tanh) => &k.tanh_inplace,
        (_, Activation::Exp) => &k.exp_inplace,
        (_, Activation::Log) => &k.log_inplace,
        (_, Activation::Sqrt) => &k.sqrt_inplace,
        (_, Activation::Rsqrt) => &k.rsqrt_inplace,
        (_, Activation::Neg) => &k.neg_inplace,
        (_, Activation::Abs) => &k.abs_inplace,
        (_, Activation::Sin) => &k.sin_inplace,
        (_, Activation::Cos) => &k.cos_inplace,
        (_, Activation::Tan) => &k.tan_inplace,
        (_, Activation::Atan) => &k.atan_inplace,
        (_, Activation::Round) => panic!("rlx-metal: Activation::Round is training-only (rlx-cpu)"),
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), data_off as u64);
    enc.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &len as *const u32 as *const _,
    );
    let tg_size = pipeline.thread_execution_width().min(len as u64);
    let grid = metal::MTLSize {
        width: len as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_layer_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.layer_norm,
        HalfFlag::F16 => &k.layer_norm_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), b as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    // One threadgroup per row; reduction requires power-of-2 threadgroup size.
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let grid = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Row-major dense strides for `out_dims`.
fn dense_row_major_strides(out_dims: &[u32], rank: usize) -> Vec<u32> {
    let mut dense = vec![1u32; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        dense[i] = dense[i + 1].saturating_mul(out_dims[i + 1].max(1));
    }
    dense
}

fn broadcast_strides_u32(in_dims: &[u32], out_dims: &[u32]) -> Vec<u32> {
    let r_out = out_dims.len();
    let r_in = in_dims.len();
    let pad = r_out.saturating_sub(r_in);
    let mut strides = vec![0u32; r_out];
    let mut acc: u32 = 1;
    for d in (0..r_out).rev() {
        let in_size = if d < pad { 1 } else { in_dims[d - pad].max(1) };
        if in_size == 1 {
            strides[d] = 0;
        } else {
            strides[d] = acc;
            acc = acc.saturating_mul(in_size);
        }
    }
    strides
}

fn is_scalar_broadcast(strides: &[u32], rank: usize) -> bool {
    rank == 0 || (strides.len() >= rank && strides[..rank].iter().all(|&s| s == 0))
}

fn is_row_vector_broadcast(strides: &[u32], rank: usize, out_dims: &[u32]) -> bool {
    if rank < 2 || strides.len() < rank || strides[rank - 1] != 0 {
        return false;
    }
    let mut in_dims = Vec::with_capacity(rank);
    for i in 0..rank - 1 {
        in_dims.push(out_dims[i]);
    }
    in_dims.push(1);
    let expected = broadcast_strides_u32(&in_dims, out_dims);
    strides[..rank] == expected[..rank]
}

/// `Some(rhs_is_scalar)` when one side is dense and the other is a scalar broadcast.
fn detect_scalar_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<bool> {
    let rank = rank as usize;
    if out_dims.len() < rank {
        return None;
    }
    let dense = dense_row_major_strides(out_dims, rank);
    if lhs_strides.len() >= rank
        && rhs_strides.len() >= rank
        && lhs_strides[..rank] == dense[..]
        && is_scalar_broadcast(rhs_strides, rank)
    {
        return Some(true);
    }
    if rhs_strides.len() >= rank
        && lhs_strides.len() >= rank
        && rhs_strides[..rank] == dense[..]
        && is_scalar_broadcast(lhs_strides, rank)
    {
        return Some(false);
    }
    None
}

/// `Some((rows, cols, rhs_is_broadcast))` when one operand is dense row-major and the other
/// is a last-axis vector broadcast over all leading dimensions (`stride 0` on outer axes).
fn detect_last_axis_col_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<(u32, u32, bool)> {
    let rank = rank as usize;
    if rank < 2 || out_dims.len() < rank {
        return None;
    }
    let cols = out_dims[rank - 1];
    if cols == 0 {
        return None;
    }
    let mut rows_u64 = 1u64;
    for &d in &out_dims[..rank - 1] {
        rows_u64 = rows_u64.saturating_mul(d.max(1) as u64);
    }
    if rows_u64 == 0 || rows_u64 > u32::MAX as u64 {
        return None;
    }
    let rows = rows_u64 as u32;

    let dense = dense_row_major_strides(out_dims, rank);

    let rhs_is_vec = |strides: &[u32]| -> bool {
        strides.len() >= rank
            && strides[rank - 1] == 1
            && strides[..rank - 1].iter().all(|&s| s == 0)
    };

    if lhs_strides.len() >= rank
        && rhs_strides.len() >= rank
        && lhs_strides[..rank] == dense[..]
        && rhs_is_vec(rhs_strides)
    {
        return Some((rows, cols, true));
    }
    if rhs_strides.len() >= rank
        && lhs_strides.len() >= rank
        && rhs_strides[..rank] == dense[..]
        && rhs_is_vec(lhs_strides)
    {
        return Some((rows, cols, false));
    }
    None
}

/// `Some((rows, cols, rhs_is_broadcast))` for `[…, cols] op […, 1]` row-vector broadcast.
fn detect_last_axis_row_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<(u32, u32, bool)> {
    let rank = rank as usize;
    if rank < 2 || out_dims.len() < rank {
        return None;
    }
    let cols = out_dims[rank - 1];
    if cols == 0 {
        return None;
    }
    let mut rows_u64 = 1u64;
    for &d in &out_dims[..rank - 1] {
        rows_u64 = rows_u64.saturating_mul(d.max(1) as u64);
    }
    if rows_u64 == 0 || rows_u64 > u32::MAX as u64 {
        return None;
    }
    let rows = rows_u64 as u32;
    let dense = dense_row_major_strides(out_dims, rank);

    if lhs_strides.len() >= rank
        && rhs_strides.len() >= rank
        && lhs_strides[..rank] == dense[..]
        && is_row_vector_broadcast(rhs_strides, rank, out_dims)
    {
        return Some((rows, cols, true));
    }
    if rhs_strides.len() >= rank
        && lhs_strides.len() >= rank
        && rhs_strides[..rank] == dense[..]
        && is_row_vector_broadcast(lhs_strides, rank, out_dims)
    {
        return Some((rows, cols, false));
    }
    None
}

/// Exactly one broadcast axis (e.g. `[B, T, H] op [B, 1, H]`).
fn detect_single_axis_broadcast(
    rank: u32,
    out_dims: &[u32],
    lhs_strides: &[u32],
    rhs_strides: &[u32],
) -> Option<(u32, u32, u32, bool)> {
    let rank = rank as usize;
    if rank < 2 || out_dims.len() < rank {
        return None;
    }
    let dense = dense_row_major_strides(out_dims, rank);

    let try_side = |strides: &[u32], other: &[u32]| -> Option<(u32, u32, u32)> {
        if strides.len() < rank || other.len() < rank || other[..rank] != dense[..rank] {
            return None;
        }
        let zero_axes: Vec<usize> = (0..rank).filter(|&i| strides[i] == 0).collect();
        if zero_axes.len() != 1 {
            return None;
        }
        let ba = zero_axes[0];
        let mut in_dims = out_dims[..rank].to_vec();
        in_dims[ba] = 1;
        let expected = broadcast_strides_u32(&in_dims, out_dims);
        if strides[..rank] != expected[..rank] {
            return None;
        }
        let mut pre_u64 = 1u64;
        for &d in &out_dims[..ba] {
            pre_u64 = pre_u64.saturating_mul(d.max(1) as u64);
        }
        let mid = out_dims[ba];
        let mut post_u64 = 1u64;
        for &d in &out_dims[ba + 1..] {
            post_u64 = post_u64.saturating_mul(d.max(1) as u64);
        }
        let rows_u64 = pre_u64.saturating_mul(mid.max(1) as u64);
        if rows_u64 == 0
            || post_u64 == 0
            || rows_u64 > u32::MAX as u64
            || post_u64 > u32::MAX as u64
        {
            return None;
        }
        Some((rows_u64 as u32, post_u64 as u32, mid))
    };

    if let Some((rows, cols, mid)) = try_side(rhs_strides, lhs_strides) {
        return Some((rows, cols, mid, true));
    }
    if let Some((rows, cols, mid)) = try_side(lhs_strides, rhs_strides) {
        return Some((rows, cols, mid, false));
    }
    None
}

fn encode_binary_broadcast_1ax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    mid: u32,
    op: u32,
    rhs_is_broadcast: bool,
) {
    let (a, b) = if rhs_is_broadcast {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = cols.is_multiple_of(4) && cols >= 4;
    if use_vec4 {
        let cols4 = cols / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_1ax4);
        enc.set_buffer(0, Some(buffer), a as u64);
        enc.set_buffer(1, Some(buffer), b as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &cols4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &mid as *const u32 as *const _);
        enc.set_bytes(6, 4, &op as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: cols4 as u64,
            height: rows as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 64.min(cols4 as u64),
            height: 4.min(rows as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_1ax_f32);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &mid as *const u32 as *const _);
    enc.set_bytes(6, 4, &op as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: cols as u64,
        height: rows as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32.min(cols as u64),
        height: 8.min(rows as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_binary_broadcast_rhs_scalar(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: u32,
    rhs_is_scalar: bool,
) {
    let (a, b) = if rhs_is_scalar {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = len.is_multiple_of(4) && len >= 4;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_scalar4);
        enc.set_buffer(0, Some(buffer), a as u64);
        enc.set_buffer(1, Some(buffer), b as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &op as *const u32 as *const _);
        let tg_w = k
            .binary_broadcast_rhs_scalar4
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_scalar_f32);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &op as *const u32 as *const _);
    let tg_w = k
        .binary_broadcast_rhs_scalar_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_binary_broadcast_rhs_row(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    op: u32,
    rhs_is_broadcast: bool,
) {
    let (a, b) = if rhs_is_broadcast {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = cols.is_multiple_of(4) && cols >= 4;
    if use_vec4 {
        let cols4 = cols / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_row4);
        enc.set_buffer(0, Some(buffer), a as u64);
        enc.set_buffer(1, Some(buffer), b as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &cols4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &op as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: cols4 as u64,
            height: rows as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 64.min(cols4 as u64),
            height: 4.min(rows as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_row_f32);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &op as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: cols as u64,
        height: rows as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32.min(cols as u64),
        height: 8.min(rows as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_binary_broadcast_rhs_col(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    op: u32,
    rhs_is_broadcast: bool,
) {
    let (a, b) = if rhs_is_broadcast {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let use_vec4 = cols.is_multiple_of(4) && cols >= 4;
    if use_vec4 {
        let cols4 = cols / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_col4);
        enc.set_buffer(0, Some(buffer), a as u64);
        enc.set_buffer(1, Some(buffer), b as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &rows as *const u32 as *const _);
        enc.set_bytes(4, 4, &cols4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &op as *const u32 as *const _);
        let grid = metal::MTLSize {
            width: cols4 as u64,
            height: rows as u64,
            depth: 1,
        };
        let tg = metal::MTLSize {
            width: 64.min(cols4 as u64),
            height: 4.min(rows as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rhs_col_f32);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &op as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: cols as u64,
        height: rows as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32.min(cols as u64),
        height: 8.min(rows as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_binary_broadcast_rank2(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    dim0: u32,
    dim1: u32,
    lhs_stride0: u32,
    lhs_stride1: u32,
    rhs_stride0: u32,
    rhs_stride1: u32,
    op: u32,
) {
    let use_vec4 = len.is_multiple_of(4)
        && dim1.is_multiple_of(4)
        && len >= 4
        && lhs_stride1 == 1
        && rhs_stride1 == 1;
    if use_vec4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.binary_broadcast_rank24);
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &dim0 as *const u32 as *const _);
        enc.set_bytes(5, 4, &dim1 as *const u32 as *const _);
        enc.set_bytes(6, 4, &lhs_stride0 as *const u32 as *const _);
        enc.set_bytes(7, 4, &lhs_stride1 as *const u32 as *const _);
        enc.set_bytes(8, 4, &rhs_stride0 as *const u32 as *const _);
        enc.set_bytes(9, 4, &rhs_stride1 as *const u32 as *const _);
        enc.set_bytes(10, 4, &op as *const u32 as *const _);
        let tg_w = k
            .binary_broadcast_rank24
            .thread_execution_width()
            .min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    enc.set_compute_pipeline_state(&k.binary_broadcast_rank2_f32);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &dim0 as *const u32 as *const _);
    enc.set_bytes(5, 4, &dim1 as *const u32 as *const _);
    enc.set_bytes(6, 4, &lhs_stride0 as *const u32 as *const _);
    enc.set_bytes(7, 4, &lhs_stride1 as *const u32 as *const _);
    enc.set_bytes(8, 4, &rhs_stride0 as *const u32 as *const _);
    enc.set_bytes(9, 4, &rhs_stride1 as *const u32 as *const _);
    enc.set_bytes(10, 4, &op as *const u32 as *const _);
    let tg_w = k
        .binary_broadcast_rank2_f32
        .thread_execution_width()
        .min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_binary(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: rlx_ir::op::BinaryOp,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    use rlx_ir::op::BinaryOp;
    let use_vec4 = matches!(dt, HalfFlag::F32) && len.is_multiple_of(4) && len >= 4;
    // f16 covers Add and Mul (the Nomic residual + SwiGLU patterns).
    // Other binaries silently fall back to f32 kernels in mixed
    // precision — same caveat as encode_activation.
    let pipeline = match (dt, op, use_vec4) {
        (HalfFlag::F16, BinaryOp::Add, _) => &k.elem_add_h,
        (HalfFlag::F16, BinaryOp::Mul, _) => &k.elem_mul_h,
        (_, BinaryOp::Add, true) => &k.elem_add4,
        (_, BinaryOp::Mul, true) => &k.elem_mul4,
        (_, BinaryOp::Sub, true) => &k.elem_sub4,
        (_, BinaryOp::Div, true) => &k.elem_div4,
        (_, BinaryOp::Add, false) => &k.elem_add,
        (_, BinaryOp::Mul, false) => &k.elem_mul,
        (_, BinaryOp::Sub, false) => &k.elem_sub,
        (_, BinaryOp::Div, false) => &k.elem_div,
        (_, BinaryOp::Max, _) => &k.elem_max,
        (_, BinaryOp::Min, _) => &k.elem_min,
        (_, BinaryOp::Pow, _) => &k.elem_pow,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    let dispatch_len = if use_vec4 { len / 4 } else { len };
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &dispatch_len as *const u32 as *const _,
    );
    let tg_w = pipeline.thread_execution_width().min(dispatch_len as u64);
    let grid = metal::MTLSize {
        width: dispatch_len as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_copy(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    if matches!(dt, HalfFlag::F32) && len.is_multiple_of(4) && len >= 4 {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.copy4);
        enc.set_buffer(0, Some(buffer), src as u64);
        enc.set_buffer(1, Some(buffer), dst as u64);
        enc.set_bytes(2, 4, &len4 as *const u32 as *const _);
        let tg_w = k.copy4.thread_execution_width().min(len4 as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len4 as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
        return;
    }
    // copy_f32 moves 4 bytes per dispatch slot. For f16, two f16 values
    // pack into one f32 slot, so we halve the dispatch count and reuse
    // the same kernel. Assumes even len (Nomic shapes always are).
    let dispatch_len = match dt {
        HalfFlag::F32 => len,
        HalfFlag::F16 => len.div_ceil(2),
    };
    enc.set_compute_pipeline_state(&k.copy_f32);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &dispatch_len as *const u32 as *const _);
    let tg_w = k.copy_f32.thread_execution_width().min(dispatch_len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: dispatch_len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_gather(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    table: usize,
    idx: usize,
    dst: usize,
    num_idx: u32,
    trailing: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.gather_axis0,
        HalfFlag::F16 => &k.gather_axis0_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), table as u64);
    enc.set_buffer(1, Some(buffer), idx as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &num_idx as *const u32 as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &trailing as *const u32 as *const _,
    );
    let grid = metal::MTLSize {
        width: trailing as u64,
        height: num_idx as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 16.min(trailing as u64),
        height: 16.min(num_idx as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NarrowSegGpu {
    dst: u32,
    start: u32,
    len: u32,
}

struct PendingNarrowBatch {
    src: usize,
    outer: u32,
    src_axis: u32,
    dt: crate::thunk::HalfFlag,
    segments: Vec<(usize, u32, u32)>,
}

const NARROW_BATCH_MAX: usize = 64;

fn metal_narrow_batch_enabled() -> bool {
    !rlx_ir::env::flag("RLX_METAL_NARROW_BATCH")
}

fn narrow_segments_partition(src_axis: u32, segments: &[(u32, u32)]) -> bool {
    let mut sorted = segments.to_vec();
    sorted.sort_by_key(|(s, _)| *s);
    let mut end = 0u32;
    for (start, len) in sorted {
        if start != end {
            return false;
        }
        end = end.saturating_add(len);
    }
    end == src_axis
}

fn flush_pending_narrow_batch(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    batch: &mut Option<PendingNarrowBatch>,
) {
    let Some(b) = batch.take() else {
        return;
    };
    if b.segments.is_empty() {
        return;
    }
    if b.segments.len() == 1 {
        let (dst, start, len) = b.segments[0];
        encode_narrow(
            enc, k, buffer, b.src, dst, b.outer, b.src_axis, start, len, b.dt,
        );
        return;
    }
    let meta: Vec<(u32, u32)> = b
        .segments
        .iter()
        .map(|(_, start, len)| (*start, *len))
        .collect();
    if narrow_segments_partition(b.src_axis, &meta) {
        encode_split_lastax(enc, k, buffer, &b);
    } else {
        for (dst, start, len) in b.segments {
            encode_narrow(
                enc, k, buffer, b.src, dst, b.outer, b.src_axis, start, len, b.dt,
            );
        }
    }
}

fn try_queue_narrow_batch(
    batch: &mut Option<PendingNarrowBatch>,
    src: usize,
    dst: usize,
    outer: u32,
    src_axis: u32,
    start: u32,
    len: u32,
    dt: crate::thunk::HalfFlag,
) -> bool {
    if !metal_narrow_batch_enabled() || outer == 0 {
        return false;
    }
    if !matches!(dt, crate::thunk::HalfFlag::F32) {
        return false;
    }
    match batch {
        None => {
            *batch = Some(PendingNarrowBatch {
                src,
                outer,
                src_axis,
                dt,
                segments: vec![(dst, start, len)],
            });
            true
        }
        Some(b) if b.src == src && b.outer == outer && b.src_axis == src_axis && b.dt == dt => {
            if b.segments.len() >= NARROW_BATCH_MAX {
                return false;
            }
            let mut meta: Vec<(u32, u32)> = b.segments.iter().map(|(_, s, l)| (*s, *l)).collect();
            meta.push((start, len));
            if !narrow_segments_partition(b.src_axis, &meta) {
                return false;
            }
            b.segments.push((dst, start, len));
            true
        }
        Some(_) => false,
    }
}

fn encode_split_lastax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    batch: &PendingNarrowBatch,
) {
    use crate::thunk::HalfFlag;
    debug_assert!(batch.segments.len() >= 2);
    let segs: Vec<NarrowSegGpu> = batch
        .segments
        .iter()
        .map(|(dst, start, len)| NarrowSegGpu {
            dst: *dst as u32,
            start: *start,
            len: *len,
        })
        .collect();
    let num_seg = segs.len() as u32;
    let max_len = segs.iter().map(|s| s.len).max().unwrap_or(0);
    let use_vec4 = batch.src_axis.is_multiple_of(4)
        && segs
            .iter()
            .all(|s| (s.start % 4) == 0 && (s.len % 4) == 0 && s.len >= 4);
    if use_vec4 {
        let src_axis4 = batch.src_axis / 4;
        let max_len4 = max_len / 4;
        enc.set_compute_pipeline_state(&k.split_lastax4);
        enc.set_buffer(0, Some(buffer), batch.src as u64);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_bytes(2, 4, &batch.outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<NarrowSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let grid = metal::MTLSize {
            width: max_len4 as u64,
            height: batch.outer as u64,
            depth: num_seg as u64,
        };
        let tg = metal::MTLSize {
            width: 64.min(max_len4 as u64),
            height: 4.min(batch.outer as u64),
            depth: num_seg as u64,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        enc.set_compute_pipeline_state(&k.split_lastax);
        enc.set_buffer(0, Some(buffer), batch.src as u64);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_bytes(2, 4, &batch.outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &batch.src_axis as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<NarrowSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let grid = metal::MTLSize {
            width: max_len as u64,
            height: batch.outer as u64,
            depth: num_seg as u64,
        };
        let tg = metal::MTLSize {
            width: 32.min(max_len as u64),
            height: 8.min(batch.outer as u64),
            depth: num_seg as u64,
        };
        enc.dispatch_threads(grid, tg);
    }
    let _ = HalfFlag::F32;
}

fn encode_narrow(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    outer: u32,
    src_axis: u32,
    start: u32,
    len: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    match dt {
        HalfFlag::F32
            if start.is_multiple_of(4)
                && src_axis.is_multiple_of(4)
                && len.is_multiple_of(4)
                && len >= 4 =>
        {
            let src_axis4 = src_axis / 4;
            let start4 = start / 4;
            let len4 = len / 4;
            enc.set_compute_pipeline_state(&k.narrow_lastax4);
            enc.set_buffer(0, Some(buffer), src as u64);
            enc.set_buffer(1, Some(buffer), dst as u64);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &start4 as *const u32 as *const _);
            enc.set_bytes(5, 4, &len4 as *const u32 as *const _);
            let grid = metal::MTLSize {
                width: len4 as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64.min(len4 as u64),
                height: 4.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        HalfFlag::F32 => {
            enc.set_compute_pipeline_state(&k.narrow_lastax);
            enc.set_buffer(0, Some(buffer), src as u64);
            enc.set_buffer(1, Some(buffer), dst as u64);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &start as *const u32 as *const _);
            enc.set_bytes(5, 4, &len as *const u32 as *const _);
            let grid = metal::MTLSize {
                width: len as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64.min(len as u64),
                height: 8.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        HalfFlag::F16 => {
            enc.set_compute_pipeline_state(&k.narrow_lastax_h);
            enc.set_buffer(0, Some(buffer), src as u64);
            enc.set_buffer(1, Some(buffer), dst as u64);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &start as *const u32 as *const _);
            enc.set_bytes(5, 4, &len as *const u32 as *const _);
            let grid = metal::MTLSize {
                width: len as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 32.min(len as u64),
                height: 8.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
    }
}

fn encode_fused_residual_ln(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    res: usize,
    g: usize,
    b: usize,
    out: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.fused_residual_ln,
        HalfFlag::F16 => &k.fused_residual_ln_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), res as u64);
    enc.set_buffer(2, Some(buffer), g as u64);
    enc.set_buffer(3, Some(buffer), b as u64);
    enc.set_buffer(4, Some(buffer), out as u64);
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    let tg_count = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg);
}

fn encode_fused_residual_rms_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    res: usize,
    g: usize,
    b: usize,
    out: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.fused_residual_rms_norm,
        HalfFlag::F16 => &k.fused_residual_rms_norm_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), res as u64);
    enc.set_buffer(2, Some(buffer), g as u64);
    enc.set_buffer(3, Some(buffer), b as u64);
    enc.set_buffer(4, Some(buffer), out as u64);
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    let tg_count = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg);
}

#[allow(clippy::too_many_arguments)]
fn encode_sdpa(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    q: usize,
    k_off: usize,
    v: usize,
    mask: usize,
    out: usize,
    batch: u32,
    seq: u32,
    heads: u32,
    head_dim: u32,
    dt: crate::thunk::HalfFlag,
    seq_stride: u32,
    mask_kind: u32,
    window: u32,
    kv_seq: u32,
    kv_stride: u32,
    bhsd: u32,
) {
    use crate::thunk::HalfFlag;
    // The two-pass `sdpa` / `sdpa_h` kernels store an [seq, seq] scores
    // matrix in threadgroup memory (`scores[64*64]`); they're correct
    // only for self-attention prefill where Lq == Lk and seq ≤ 64.
    // For longer sequences (e.g. NomicVision's seq=257
    // = 256 patches + 1 CLS) we route to `sdpa_long`, an online-softmax
    // FA-v1 variant that's O(D) memory per query row and scales to any
    // seq length. Also route decode steps (Lq=1, Lk=past+1) through
    // `sdpa_long` — the rectangular `sdpa` TG scores buffer is sized for
    // self-attention prefill; bucketed decode must use the online kernel.
    // F16 input/output isn't supported by sdpa_long yet —
    // that path falls through and would hit the seq-64 ceiling; today
    // no f16-tagged graph hits seq>64 in production.
    if matches!(dt, HalfFlag::F32) && (seq > 64 || kv_seq > 64 || kv_seq != seq) {
        // Pick between the scalar online-softmax (`sdpa_long`) and the
        // tile-based flash-attention (`sdpa_fa_f32`). FA amortizes K/V
        // reads across an 8-query tile via threadgroup memory, so it
        // wins over `sdpa_long` (~35% faster) when Lk dominates. It
        // still lags MPSGraph's batched matmul decomp for SAM3 image
        // CA (Lq=201, Lk=5184, dh=16) because MPSGraph uses
        // simdgroup_float8x8 internally; opt-in via `RLX_METAL_FA=1`
        // for benchmarking until the kernel is upgraded to use
        // simdgroup matrix primitives.
        let use_fa = kv_seq >= 256 && head_dim <= 32 && rlx_ir::env::flag("RLX_METAL_FA");
        let pipeline = if use_fa { &k.sdpa_fa_f32 } else { &k.sdpa_long };
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(buffer), q as u64);
        enc.set_buffer(1, Some(buffer), k_off as u64);
        enc.set_buffer(2, Some(buffer), v as u64);
        enc.set_buffer(3, Some(buffer), mask as u64);
        enc.set_buffer(4, Some(buffer), out as u64);
        enc.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &batch as *const u32 as *const _,
        );
        enc.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &seq as *const u32 as *const _,
        );
        enc.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &heads as *const u32 as *const _,
        );
        enc.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        enc.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            &seq_stride as *const u32 as *const _,
        );
        enc.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            &mask_kind as *const u32 as *const _,
        );
        enc.set_bytes(
            11,
            std::mem::size_of::<u32>() as u64,
            &kv_seq as *const u32 as *const _,
        );
        enc.set_bytes(
            12,
            std::mem::size_of::<u32>() as u64,
            &kv_stride as *const u32 as *const _,
        );
        enc.set_bytes(
            13,
            std::mem::size_of::<u32>() as u64,
            &bhsd as *const u32 as *const _,
        );
        enc.set_bytes(
            14,
            std::mem::size_of::<u32>() as u64,
            &window as *const u32 as *const _,
        );
        if use_fa {
            // FA kernel: 1 TG per (q_tile, head, batch), 64 threads, Br=8.
            const BR: u32 = 8;
            let q_tiles = seq.div_ceil(BR);
            let grid = metal::MTLSize {
                width: q_tiles as u64,
                height: heads as u64,
                depth: batch as u64,
            };
            let tg = metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(grid, tg);
        } else {
            let total = (batch as u64) * (heads as u64) * (seq as u64);
            let grid = metal::MTLSize {
                width: total,
                height: 1,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        return;
    }
    let pipeline = match dt {
        HalfFlag::F32 => &k.sdpa,
        HalfFlag::F16 => &k.sdpa_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), q as u64);
    enc.set_buffer(1, Some(buffer), k_off as u64);
    enc.set_buffer(2, Some(buffer), v as u64);
    enc.set_buffer(3, Some(buffer), mask as u64);
    enc.set_buffer(4, Some(buffer), out as u64);
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &batch as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &seq as *const u32 as *const _,
    );
    enc.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &heads as *const u32 as *const _,
    );
    enc.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &head_dim as *const u32 as *const _,
    );
    enc.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &seq_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &mask_kind as *const u32 as *const _,
    );
    enc.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &kv_seq as *const u32 as *const _,
    );
    enc.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        &kv_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        13,
        std::mem::size_of::<u32>() as u64,
        &bhsd as *const u32 as *const _,
    );
    enc.set_bytes(
        14,
        std::mem::size_of::<u32>() as u64,
        &window as *const u32 as *const _,
    );
    let tg_count = metal::MTLSize {
        width: (batch * heads) as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(tg_count, tg);
}

fn encode_rope(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    cos: usize,
    sin: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    dt: crate::thunk::HalfFlag,
    src_row_stride: u32,
    seq_stride: u32,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.rope,
        HalfFlag::F16 => &k.rope_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), cos as u64);
    enc.set_buffer(2, Some(buffer), sin as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &batch as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &seq as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        std::mem::size_of::<u32>() as u64,
        &hidden as *const u32 as *const _,
    );
    enc.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &head_dim as *const u32 as *const _,
    );
    enc.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &src_row_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &seq_stride as *const u32 as *const _,
    );
    enc.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &n_rot as *const u32 as *const _,
    );
    let nh = hidden / head_dim;
    let grid = metal::MTLSize {
        width: head_dim as u64,
        height: nh as u64,
        depth: (batch * seq) as u64,
    };
    let tg = metal::MTLSize {
        width: head_dim.min(16) as u64,
        height: nh.min(8) as u64,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_rms_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    rows: u32,
    h: u32,
    eps: f32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.rms_norm,
        HalfFlag::F16 => &k.rms_norm_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), b as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &h as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<f32>() as u64,
        &eps as *const f32 as *const _,
    );
    // One threadgroup per row; power-of-2 tg size for reduction (see encode_layer_norm).
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= h as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    let grid = metal::MTLSize {
        width: rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

fn encode_rms_norm_bwd_input(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    gamma: usize,
    beta: usize,
    dy: usize,
    dx: usize,
    rows: u32,
    h: u32,
    eps: f32,
) {
    enc.set_compute_pipeline_state(&k.rms_norm_bwd);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), gamma as u64);
    enc.set_buffer(2, Some(buffer), beta as u64);
    enc.set_buffer(3, Some(buffer), dy as u64);
    enc.set_buffer(4, Some(buffer), dx as u64);
    enc.set_bytes(5, 4, &h as *const u32 as *const _);
    enc.set_bytes(6, 4, &eps as *const f32 as *const _);
    let wrt: u32 = 0;
    enc.set_bytes(7, 4, &wrt as *const u32 as *const _);
    let tg_w = 256u64.min(h as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: tg_w * rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_rms_norm_bwd_param(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    _gamma: usize,
    _beta: usize,
    dy: usize,
    out: usize,
    rows: u32,
    h: u32,
    eps: f32,
    wrt: u32,
    inv_r_scratch: usize,
) {
    let use_parallel = inv_r_scratch != 0 && rows > 1;
    if !use_parallel {
        enc.set_compute_pipeline_state(&k.rms_norm_bwd_param);
        enc.set_buffer(0, Some(buffer), x as u64);
        enc.set_buffer(1, Some(buffer), _gamma as u64);
        enc.set_buffer(2, Some(buffer), _beta as u64);
        enc.set_buffer(3, Some(buffer), dy as u64);
        enc.set_buffer(4, Some(buffer), out as u64);
        enc.set_bytes(5, 4, &rows as *const u32 as *const _);
        enc.set_bytes(6, 4, &h as *const u32 as *const _);
        enc.set_bytes(7, 4, &eps as *const f32 as *const _);
        enc.set_bytes(8, 4, &wrt as *const u32 as *const _);
        enc.dispatch_threads(
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        return;
    }

    if wrt == 1 {
        enc.set_compute_pipeline_state(&k.rms_norm_bwd_inv_r_f32);
        enc.set_buffer(0, Some(buffer), x as u64);
        enc.set_buffer(1, Some(buffer), inv_r_scratch as u64);
        enc.set_bytes(2, 4, &h as *const u32 as *const _);
        enc.set_bytes(3, 4, &eps as *const f32 as *const _);
        enc.dispatch_threads(
            metal::MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 256.min(rows as u64).max(1),
                height: 1,
                depth: 1,
            },
        );
    }

    enc.set_compute_pipeline_state(&k.rms_norm_bwd_param_reduce_f32);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), inv_r_scratch as u64);
    enc.set_buffer(3, Some(buffer), out as u64);
    enc.set_bytes(4, 4, &rows as *const u32 as *const _);
    enc.set_bytes(5, 4, &h as *const u32 as *const _);
    enc.set_bytes(6, 4, &wrt as *const u32 as *const _);
    let tg_w = 256u64.min(h as u64).max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: h as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn rms_norm_bwd_scratch_bytes(graph: &Graph) -> usize {
    let mut max_rows = 0usize;
    for node in graph.nodes() {
        if matches!(
            node.op,
            Op::RmsNormBackwardGamma { .. } | Op::RmsNormBackwardBeta { .. }
        ) {
            let x_shape = &graph.node(node.inputs[0]).shape;
            let h = x_shape.dim(x_shape.rank() - 1).unwrap_static();
            let rows = x_shape.num_elements().unwrap() / h;
            max_rows = max_rows.max(rows);
        }
    }
    max_rows * std::mem::size_of::<f32>()
}

fn encode_rope_bwd(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    cos: usize,
    sin: usize,
    dx: usize,
    batch: u32,
    seq: u32,
    hidden: u32,
    head_dim: u32,
    n_rot: u32,
    cos_len: u32,
) {
    enc.set_compute_pipeline_state(&k.rope_bwd);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), cos as u64);
    enc.set_buffer(2, Some(buffer), sin as u64);
    enc.set_buffer(3, Some(buffer), dx as u64);
    enc.set_bytes(4, 4, &batch as *const u32 as *const _);
    enc.set_bytes(5, 4, &seq as *const u32 as *const _);
    enc.set_bytes(6, 4, &hidden as *const u32 as *const _);
    enc.set_bytes(7, 4, &head_dim as *const u32 as *const _);
    enc.set_bytes(8, 4, &n_rot as *const u32 as *const _);
    enc.set_bytes(9, 4, &cos_len as *const u32 as *const _);
    let nh = hidden / head_dim.max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: head_dim as u64,
            height: nh as u64,
            depth: (batch * seq) as u64,
        },
        metal::MTLSize {
            width: head_dim.min(16) as u64,
            height: nh.min(8) as u64,
            depth: 1,
        },
    );
}

fn encode_cumsum_bwd(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    dx: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    enc.set_compute_pipeline_state(&k.cumsum_bwd);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), dx as u64);
    enc.set_bytes(2, 4, &cols as *const u32 as *const _);
    let ex: u32 = if exclusive { 1 } else { 0 };
    enc.set_bytes(3, 4, &ex as *const u32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: rows as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_gather_bwd(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    indices: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
) {
    let n = outer * axis_dim * trailing;
    if n > 0 {
        enc.set_compute_pipeline_state(&k.gather_bwd_zero);
        enc.set_buffer(0, Some(buffer), dst as u64);
        enc.set_bytes(1, 4, &n as *const u32 as *const _);
        enc.dispatch_threads(
            metal::MTLSize {
                width: n as u64,
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
    enc.set_compute_pipeline_state(&k.gather_bwd_acc);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), indices as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &outer as *const u32 as *const _);
    enc.set_bytes(4, 4, &axis_dim as *const u32 as *const _);
    enc.set_bytes(5, 4, &num_idx as *const u32 as *const _);
    enc.set_bytes(6, 4, &trailing as *const u32 as *const _);
    enc.dispatch_threads(
        metal::MTLSize {
            width: outer as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
}

fn dequant_gguf_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::DequantMatMul { scheme } = &node.op
            && scheme.is_gguf()
        {
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            let total = node.shape.num_elements().unwrap();
            let m = total / n.max(1);
            let x_total = graph.node(node.inputs[0]).shape.num_elements().unwrap();
            let k = x_total / m.max(1);
            max = max.max(k * n * std::mem::size_of::<f32>());
        }
        if let Op::DequantGroupedMatMul { .. } = &node.op {
            let in_shape = &graph.node(node.inputs[0]).shape;
            let m = in_shape.dim(in_shape.rank() - 2).unwrap_static();
            let k = in_shape.dim(in_shape.rank() - 1).unwrap_static();
            let n = node.shape.dim(node.shape.rank() - 1).unwrap_static();
            max = max.max(k * n * 4 + m * k * 4 + m * n * 4);
        }
    }
    max
}

pub(crate) fn gguf_scheme_id(scheme: rlx_ir::quant::QuantScheme) -> u32 {
    use rlx_ir::quant::QuantScheme;
    match scheme {
        QuantScheme::GgufQ4K => 0,
        QuantScheme::GgufQ5K => 1,
        QuantScheme::GgufQ6K => 2,
        QuantScheme::GgufQ8K => 3,
        QuantScheme::GgufQ2K => 4,
        QuantScheme::GgufQ3K => 5,
        other => panic!("gguf_scheme_id: unsupported {other:?}"),
    }
}

pub(crate) fn encode_dequant_gguf(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    w_q: usize,
    dst: usize,
    scheme: rlx_ir::quant::QuantScheme,
    k_dim: usize,
    n_dim: usize,
) {
    let block_elems = scheme.gguf_block_size() as usize;
    let total = k_dim * n_dim;
    let num_blocks = total / block_elems.max(1);
    let scheme_id = gguf_scheme_id(scheme);
    let dst_f32 = (dst / 4) as u32;
    enc.set_compute_pipeline_state(&k.dequant_gguf);
    enc.set_buffer(0, Some(buffer), 0);
    let w_u = w_q as u32;
    enc.set_bytes(1, 4, &w_u as *const u32 as *const _);
    enc.set_bytes(2, 4, &dst_f32 as *const u32 as *const _);
    enc.set_bytes(3, 4, &scheme_id as *const u32 as *const _);
    let nb = num_blocks as u32;
    enc.set_bytes(4, 4, &nb as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: num_blocks as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(num_blocks) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_dequant_grouped_matmul_gguf(
    cmd_buf: &metal::CommandBufferRef,
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scratch_off: usize,
    input: usize,
    w_q: usize,
    expert_idx: usize,
    dst: usize,
    m: usize,
    k_dim: usize,
    n: usize,
    num_experts: usize,
    scheme: rlx_ir::quant::QuantScheme,
) {
    let block_elems = scheme.gguf_block_size() as usize;
    let block_bytes = scheme.gguf_block_bytes() as usize;
    let slab_bytes = (k_dim * n) / block_elems * block_bytes;

    let base = buffer.contents() as *const u8;
    unsafe {
        let x_host = std::slice::from_raw_parts(base.add(input) as *const f32, m * k_dim);
        let idx_host = std::slice::from_raw_parts(base.add(expert_idx) as *const f32, m);
        let (packed_in, original_pos, offsets) =
            rlx_cpu::gguf_matmul::grouped_moe_sort_plan(x_host, idx_host, m, k_dim, num_experts);

        let dequant_off = scratch_off;
        let pack_in_off = scratch_off + k_dim * n * 4;
        let pack_out_off = scratch_off + (k_dim * n + m * k_dim) * 4;

        std::ptr::copy_nonoverlapping(
            packed_in.as_ptr(),
            base.add(pack_in_off) as *mut f32,
            packed_in.len(),
        );

        for e in 0..num_experts {
            let count = offsets[e + 1] - offsets[e];
            if count == 0 {
                continue;
            }
            encode_dequant_gguf(
                enc,
                k,
                buffer,
                w_q + e * slab_bytes,
                dequant_off,
                scheme,
                k_dim,
                n,
            );
            let in_start = offsets[e];
            crate::mps_blas::encode_mps_sgemm_bt(
                cmd_buf,
                buffer,
                pack_in_off + in_start * k_dim * 4,
                dequant_off,
                pack_out_off + in_start * n * 4,
                count,
                k_dim,
                n,
            );
        }

        let pack_out_host = std::slice::from_raw_parts(base.add(pack_out_off) as *const f32, m * n);
        let mut out_host = vec![0f32; m * n];
        rlx_cpu::gguf_matmul::grouped_moe_unpermute_out(
            pack_out_host,
            &original_pos,
            &mut out_host,
            m,
            n,
        );
        std::ptr::copy_nonoverlapping(out_host.as_ptr(), base.add(dst) as *mut f32, out_host.len());
    }
}

fn gdn_ephemeral_state_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::GatedDeltaNet {
            carry_state,
            state_size,
            ..
        } = &node.op
            && !*carry_state
        {
            let q_shape = &graph.node(node.inputs[0]).shape;
            let elems = q_shape.dim(0).unwrap_static()
                * q_shape.dim(2).unwrap_static()
                * state_size
                * state_size;
            max = max.max(elems * std::mem::size_of::<f32>());
        }
    }
    max
}

fn encode_gated_delta_net(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    q: usize,
    k_off: usize,
    v: usize,
    g: usize,
    beta: usize,
    state: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    heads: u32,
    state_size: u32,
    use_carry: bool,
) {
    let f32_idx = |byte_off: usize| -> u32 { (byte_off / 4) as u32 };
    enc.set_compute_pipeline_state(&k.gated_delta_net);
    enc.set_buffer(0, Some(buffer), 0);
    let q_u = f32_idx(q);
    let k_u = f32_idx(k_off);
    let v_u = f32_idx(v);
    let g_u = f32_idx(g);
    let beta_u = f32_idx(beta);
    let state_u = f32_idx(state);
    let dst_u = f32_idx(dst);
    enc.set_bytes(1, 4, &q_u as *const u32 as *const _);
    enc.set_bytes(2, 4, &k_u as *const u32 as *const _);
    enc.set_bytes(3, 4, &v_u as *const u32 as *const _);
    enc.set_bytes(4, 4, &g_u as *const u32 as *const _);
    enc.set_bytes(5, 4, &beta_u as *const u32 as *const _);
    enc.set_bytes(6, 4, &state_u as *const u32 as *const _);
    enc.set_bytes(7, 4, &dst_u as *const u32 as *const _);
    let dims = [batch, seq, heads, state_size];
    enc.set_bytes(8, 16, dims.as_ptr() as *const _);
    let use_carry_u: u32 = if use_carry { 1 } else { 0 };
    enc.set_bytes(9, 4, &use_carry_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: (batch * heads) as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: state_size as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

fn conv_bwd_scratch_bytes(graph: &Graph) -> usize {
    let mut max = 0usize;
    for node in graph.nodes() {
        if let Op::Conv2dBackwardWeight {
            kernel_size,
            groups,
            ..
        } = &node.op
        {
            let x_shape = &graph.node(node.inputs[0]).shape;
            let dy_shape = &graph.node(node.inputs[1]).shape;
            if x_shape.rank() != 4 || dy_shape.rank() != 4 {
                continue;
            }
            let c_in = x_shape.dim(1).unwrap_static();
            let h_out = dy_shape.dim(2).unwrap_static();
            let w_out = dy_shape.dim(3).unwrap_static();
            let kh = kernel_size.first().copied().unwrap_or(1);
            let kw = kernel_size.get(1).copied().unwrap_or(1);
            let groups = (*groups).max(1);
            let c_in_per_g = c_in / groups;
            let n_dim = c_in_per_g * kh * kw;
            let k_dim = h_out * w_out;
            max = max.max(n_dim * k_dim * std::mem::size_of::<f32>());
        }
    }
    max
}

/// Implicit im2col+GEMM only when explicitly enabled — materialized im2col + MPS/simd
/// sgemm wins on Voxtral-scale conv weight backward (see bench-encoder).
fn conv_bwd_weight_use_implicit_gemm(m: usize, k: usize, n: usize) -> bool {
    if !rlx_ir::env::var("RLX_METAL_CONV_BWD_IMPLICIT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return false;
    }
    if !k.is_multiple_of(8) || n < 8 || m < 1 {
        return false;
    }
    !matches!(
        crate::cost::hw_model().pick_sgemm(m, k, n),
        crate::cost::SgemmVariant::Mps
            | crate::cost::SgemmVariant::Tiled
            | crate::cost::SgemmVariant::Naive
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_conv2d_bwd_weight_gemm(
    enc: &metal::ComputeCommandEncoderRef,
    kk: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    x: usize,
    dw: usize,
    m: usize,
    k: usize,
    n: usize,
    nchw: &[u32; 4],
    out_dims: &[u32; 4],
    kshape: &[u32; 4],
    padd: &[u32; 4],
) {
    let m_u = m as u32;
    let k_u = k as u32;
    let n_u = n as u32;
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), x as u64);
    enc.set_buffer(2, Some(buffer), dw as u64);
    enc.set_bytes(3, 4, &m_u as *const _ as *const _);
    enc.set_bytes(4, 4, &k_u as *const _ as *const _);
    enc.set_bytes(5, 4, &n_u as *const _ as *const _);
    enc.set_bytes(6, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(7, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(8, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(9, 16, padd.as_ptr() as *const _);

    let aligned_32 = m.is_multiple_of(32) && k.is_multiple_of(32) && n.is_multiple_of(32);
    if aligned_32 && m >= 32 && n >= 32 {
        enc.set_compute_pipeline_state(&kk.conv2d_bwd_weight_gemm_4x4);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: n.div_ceil(32) as u64,
                height: m.div_ceil(32) as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 512,
                height: 1,
                depth: 1,
            },
        );
    } else {
        enc.set_compute_pipeline_state(&kk.conv2d_bwd_weight_gemm);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: n.div_ceil(8) as u64,
                height: m.div_ceil(8) as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_im2col_group(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    col: usize,
    nchw: &[u32; 4],
    out_dims: &[u32; 4],
    kshape: &[u32; 4],
    padd: &[u32; 4],
    elems: u64,
) {
    let w1 = nchw[2] == 1 && out_dims[2] == 1;
    if w1 {
        enc.set_compute_pipeline_state(&k.im2col_group_w1);
    } else {
        enc.set_compute_pipeline_state(&k.im2col_group);
    }
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), col as u64);
    enc.set_bytes(2, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(3, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(4, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(5, 16, padd.as_ptr() as *const _);
    let tg_w = 512u64.min(elems).max(1);
    enc.dispatch_threads(
        metal::MTLSize {
            width: elems,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_conv2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let nch: [u32; 4] = [n, c_in, h, w];
    let out_dims: [u32; 4] = [c_out, h_out, w_out, groups];
    let kshape: [u32; 4] = [kh, kw, sh, sw];
    let padd: [u32; 4] = [ph, pw, dh, dw];
    let w1 = w == 1 && w_out == 1;
    if w1 {
        enc.set_compute_pipeline_state(&k.conv2d_w1);
    } else {
        enc.set_compute_pipeline_state(&k.conv2d);
    }
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 16, nch.as_ptr() as *const _);
    enc.set_bytes(4, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(5, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(6, 16, padd.as_ptr() as *const _);
    let grid = if w1 {
        metal::MTLSize {
            width: 1,
            height: h_out as u64,
            depth: (n * c_out) as u64,
        }
    } else {
        metal::MTLSize {
            width: w_out as u64,
            height: h_out as u64,
            depth: (n * c_out) as u64,
        }
    };
    let tg = if w1 {
        metal::MTLSize {
            width: 1,
            height: 8.min(h_out as u64),
            depth: 1,
        }
    } else {
        metal::MTLSize {
            width: 8.min(w_out as u64),
            height: 8.min(h_out as u64),
            depth: 1,
        }
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_group_norm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    num_groups: u32,
    eps: f32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    enc.set_compute_pipeline_state(&k.group_norm);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), b as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(5, 4, &num_groups as *const u32 as *const _);
    enc.set_bytes(6, 4, &eps as *const f32 as *const _);
    let groups = (n * num_groups) as u64;
    let tg = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    // Dispatch one threadgroup per (batch, group) along grid *width* so
    // `threadgroup_position_in_grid` (scalar .x) indexes 0..batch*num_groups-1.
    let grid = metal::MTLSize {
        width: groups.max(1),
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

fn encode_resize_nearest_2x(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    let w2 = w * 2;
    let h2 = h * 2;
    enc.set_compute_pipeline_state(&k.resize_nearest_2x);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 16, nchw.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w2 as u64,
        height: h2 as u64,
        depth: (n * c) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w2 as u64),
        height: 8.min(h2 as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_layer_norm2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    g: usize,
    b: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    eps: f32,
) {
    let nchw: [u32; 4] = [n, c, h, w];
    enc.set_compute_pipeline_state(&k.layer_norm2d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), g as u64);
    enc.set_buffer(2, Some(buffer), b as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(5, 4, &eps as *const f32 as *const _);
    let grid = metal::MTLSize {
        width: w as u64,
        height: h as u64,
        depth: n as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w as u64),
        height: 8.min(h as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_conv_transpose2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    weight: usize,
    dst: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w: u32,
    c_out: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    dh: u32,
    dw: u32,
    groups: u32,
) {
    let nch: [u32; 4] = [n, c_in, h, w];
    let out_dims: [u32; 4] = [c_out, h_out, w_out, groups];
    let kshape: [u32; 4] = [kh, kw, sh, sw];
    let padd: [u32; 4] = [ph, pw, dh, dw];
    enc.set_compute_pipeline_state(&k.conv_transpose2d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 16, nch.as_ptr() as *const _);
    enc.set_bytes(4, 16, out_dims.as_ptr() as *const _);
    enc.set_bytes(5, 16, kshape.as_ptr() as *const _);
    enc.set_bytes(6, 16, padd.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w_out as u64,
        height: h_out as u64,
        depth: (n * c_out) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_out as u64),
        height: 8.min(h_out as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_pool2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    n: u32,
    c: u32,
    h: u32,
    w: u32,
    h_out: u32,
    w_out: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
    kind: rlx_ir::op::ReduceOp,
) {
    use rlx_ir::op::ReduceOp;
    let kind_u: u32 = match kind {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    };
    let nchw: [u32; 4] = [n, c, h, w];
    let hw_out: [u32; 2] = [h_out, w_out];
    let khsw: [u32; 4] = [kh, kw, sh, sw];
    let pad: [u32; 2] = [ph, pw];
    enc.set_compute_pipeline_state(&k.pool2d);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 16, nchw.as_ptr() as *const _);
    enc.set_bytes(3, 8, hw_out.as_ptr() as *const _);
    enc.set_bytes(4, 16, khsw.as_ptr() as *const _);
    enc.set_bytes(5, 8, pad.as_ptr() as *const _);
    enc.set_bytes(6, 4, &kind_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: w_out as u64,
        height: h_out as u64,
        depth: (n * c) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_out as u64),
        height: 8.min(h_out as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_gather_axis(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    table: usize,
    idx: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    num_idx: u32,
    trailing: u32,
) {
    enc.set_compute_pipeline_state(&k.gather_axis);
    enc.set_buffer(0, Some(buffer), table as u64);
    enc.set_buffer(1, Some(buffer), idx as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &outer as *const u32 as *const _);
    enc.set_bytes(4, 4, &axis_dim as *const u32 as *const _);
    enc.set_bytes(5, 4, &num_idx as *const u32 as *const _);
    enc.set_bytes(6, 4, &trailing as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: trailing as u64,
        height: num_idx as u64,
        depth: outer as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(trailing as u64),
        height: 8.min(num_idx as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Swap of the last two axes with dense leading batch dims → `(batch, rows, cols)`.
fn detect_last2_batched_swap(out_dims: &[u32], in_strides: &[u32]) -> Option<(u32, u32, u32)> {
    let rank = out_dims.len();
    if rank < 3 || in_strides.len() < rank {
        return None;
    }
    let rows = out_dims[rank - 1];
    let cols = out_dims[rank - 2];
    if in_strides[rank - 2] != 1 || in_strides[rank - 1] != cols {
        return None;
    }
    let mut tail = cols.saturating_mul(rows);
    if rank >= 3 && in_strides[rank - 3] != tail {
        return None;
    }
    for i in (0..rank.saturating_sub(3)).rev() {
        let expected = tail.saturating_mul(out_dims[i + 1].max(1));
        if in_strides[i] != expected {
            return None;
        }
        tail = expected;
    }
    let mut batch_u64 = 1u64;
    for &d in &out_dims[..rank - 2] {
        batch_u64 = batch_u64.saturating_mul(d.max(1) as u64);
    }
    if batch_u64 == 0 || batch_u64 > u32::MAX as u64 {
        return None;
    }
    Some((batch_u64 as u32, rows, cols))
}

/// `[B, A, C, D] → [B, C, A, D]` (perm `[0, 2, 1, 3]`).
fn detect_swap12_batched_trailing(
    out_dims: &[u32],
    in_strides: &[u32],
) -> Option<(u32, u32, u32, u32)> {
    if out_dims.len() != 4 || in_strides.len() != 4 {
        return None;
    }
    let batch = out_dims[0];
    let rows = out_dims[1];
    let cols = out_dims[2];
    let trail = out_dims[3];
    if batch == 0 || rows == 0 || cols == 0 || trail == 0 {
        return None;
    }
    if in_strides[3] != 1 {
        return None;
    }
    if in_strides[1] != trail {
        return None;
    }
    if in_strides[2] != rows.saturating_mul(trail) {
        return None;
    }
    if in_strides[0] != cols.saturating_mul(rows).saturating_mul(trail) {
        return None;
    }
    Some((batch, rows, cols, trail))
}

fn encode_transpose_swap12_batched_trailing(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    batch: u32,
    rows: u32,
    cols: u32,
    trail: u32,
) {
    let use_tiled = rows >= 32 && cols >= 32;
    let depth = (batch as u64).saturating_mul(trail as u64);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &batch as *const u32 as *const _);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    enc.set_bytes(5, 4, &trail as *const u32 as *const _);
    if use_tiled {
        enc.set_compute_pipeline_state(&k.transpose_swap12_batched_trail_tiled_f32);
        let tg = metal::MTLSize {
            width: 32,
            height: 8,
            depth: 1,
        };
        let groups = metal::MTLSize {
            width: (rows as u64).div_ceil(32),
            height: (cols as u64).div_ceil(32),
            depth,
        };
        enc.dispatch_thread_groups(groups, tg);
        return;
    }
    enc.set_compute_pipeline_state(&k.transpose_swap12_batched_trail_f32);
    enc.dispatch_threads(
        metal::MTLSize {
            width: rows as u64,
            height: cols as u64,
            depth,
        },
        metal::MTLSize {
            width: 16.min(rows as u64),
            height: 16.min(cols as u64),
            depth: 1,
        },
    );
}

fn metal_host_slices_enabled() -> bool {
    matches!(
        std::env::var("RLX_METAL_HOST_SLICE").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    )
}

fn encode_transpose_2d(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    rows: u32,
    cols: u32,
) {
    let use_tiled = rows >= 32 && cols >= 32;
    enc.set_compute_pipeline_state(if use_tiled {
        &k.transpose_2d_tiled_f32
    } else {
        &k.transpose_2d_f32
    });
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &rows as *const u32 as *const _);
    enc.set_bytes(3, 4, &cols as *const u32 as *const _);
    if use_tiled {
        // 32x32 tile, threadgroup (32,8).
        let tg = metal::MTLSize {
            width: 32,
            height: 8,
            depth: 1,
        };
        let groups = metal::MTLSize {
            width: (rows as u64).div_ceil(32),
            height: (cols as u64).div_ceil(32),
            depth: 1,
        };
        enc.dispatch_thread_groups(groups, tg);
    } else {
        let tg_w = k.transpose_2d_f32.thread_execution_width().min(cols as u64);
        let tg_h = (256 / tg_w.max(1)).min(rows as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: rows as u64,
                height: cols as u64,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: tg_h,
                depth: 1,
            },
        );
    }
}

fn encode_transpose_last2_batched(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    batch: u32,
    rows: u32,
    cols: u32,
) {
    let use_tiled = rows >= 32 && cols >= 32;
    enc.set_compute_pipeline_state(if use_tiled {
        &k.transpose_last2_batched_tiled_f32
    } else {
        &k.transpose_last2_batched_f32
    });
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &batch as *const u32 as *const _);
    enc.set_bytes(3, 4, &rows as *const u32 as *const _);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
    if use_tiled {
        let tg = metal::MTLSize {
            width: 32,
            height: 8,
            depth: 1,
        };
        let groups = metal::MTLSize {
            width: (rows as u64).div_ceil(32),
            height: (cols as u64).div_ceil(32),
            depth: batch as u64,
        };
        enc.dispatch_thread_groups(groups, tg);
    } else {
        let tg_w = k
            .transpose_last2_batched_f32
            .thread_execution_width()
            .min(rows as u64);
        let tg_h = (256 / tg_w.max(1)).min(cols as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: rows as u64,
                height: cols as u64,
                depth: batch as u64,
            },
            metal::MTLSize {
                width: tg_w,
                height: tg_h,
                depth: 1,
            },
        );
    }
}

fn encode_transpose(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    total: u32,
    out_dims: &[u32],
    in_strides: &[u32],
) {
    let rank = out_dims.len() as u32;
    // Pack [out_dims..., in_strides...] into a single inline meta buffer.
    let mut meta: Vec<u32> = Vec::with_capacity(2 * out_dims.len());
    meta.extend_from_slice(out_dims);
    meta.extend_from_slice(in_strides);
    enc.set_compute_pipeline_state(&k.transpose_nd);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &rank as *const u32 as *const _);
    enc.set_bytes(3, 4, &total as *const u32 as *const _);
    enc.set_bytes(4, (meta.len() * 4) as u64, meta.as_ptr() as *const _);
    let tg_w = k.transpose_nd.thread_execution_width().min(total as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: total as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_elementwise_region(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    len: u32,
    num_inputs: u32,
    num_steps: u32,
    dst: usize,
    input_offs: &[u32; 16],
    chain: &[u32; 128],
    scalar_input_mask: u32,
    input_modulus: &[u32; 16],
    prologue: u32,
    out_n: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    prologue_input: u32,
) {
    enc.set_compute_pipeline_state(&k.elementwise_region);
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &len as *const u32 as *const _,
    );
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &num_inputs as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &num_steps as *const u32 as *const _,
    );
    let dst_u32 = (dst / 4) as u32;
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &dst_u32 as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        (input_offs.len() * 4) as u64,
        input_offs.as_ptr() as *const _,
    );
    enc.set_bytes(6, (chain.len() * 4) as u64, chain.as_ptr() as *const _);
    enc.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &scalar_input_mask as *const u32 as *const _,
    );
    enc.set_bytes(
        8,
        (input_modulus.len() * 4) as u64,
        input_modulus.as_ptr() as *const _,
    );
    enc.set_bytes(
        9,
        std::mem::size_of::<u32>() as u64,
        &prologue as *const u32 as *const _,
    );
    enc.set_bytes(
        10,
        std::mem::size_of::<u32>() as u64,
        &out_n as *const u32 as *const _,
    );
    enc.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &out_c as *const u32 as *const _,
    );
    enc.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        &out_h as *const u32 as *const _,
    );
    enc.set_bytes(
        13,
        std::mem::size_of::<u32>() as u64,
        &out_w as *const u32 as *const _,
    );
    enc.set_bytes(
        14,
        std::mem::size_of::<u32>() as u64,
        &prologue_input as *const u32 as *const _,
    );
    if prologue != 0 && out_h > 0 && out_w > 0 {
        let grid = metal::MTLSize {
            width: out_w as u64,
            height: out_h as u64,
            depth: (out_n as u64) * (out_c as u64),
        };
        let tg = metal::MTLSize {
            width: 8.min(out_w as u64),
            height: 8.min(out_h as u64),
            depth: 1,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        let tg_w = k
            .elementwise_region
            .thread_execution_width()
            .min(len as u64);
        enc.dispatch_threads(
            metal::MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            },
        );
    }
}

fn encode_batch_elementwise_region(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    slice_len: u32,
    num_batch: u32,
    num_steps: u32,
    base_dst: usize,
    slice_elems: u32,
    batch_input_offs: &[u32; 64],
    chain: &[u32; 128],
    scalar_input_mask: u32,
    input_modulus: &[u32; 16],
) {
    enc.set_compute_pipeline_state(&k.batch_elementwise_region);
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &slice_len as *const u32 as *const _,
    );
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &num_batch as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &num_steps as *const u32 as *const _,
    );
    let base_dst_u32 = (base_dst / 4) as u32;
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &base_dst_u32 as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &slice_elems as *const u32 as *const _,
    );
    enc.set_bytes(
        6,
        (batch_input_offs.len() * 4) as u64,
        batch_input_offs.as_ptr() as *const _,
    );
    enc.set_bytes(7, (chain.len() * 4) as u64, chain.as_ptr() as *const _);
    enc.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &scalar_input_mask as *const u32 as *const _,
    );
    enc.set_bytes(
        9,
        (input_modulus.len() * 4) as u64,
        input_modulus.as_ptr() as *const _,
    );
    let tg_w = k
        .batch_elementwise_region
        .thread_execution_width()
        .min(slice_len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: slice_len as u64,
            height: 1,
            depth: num_batch as u64,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_scatter_add(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    updates: usize,
    indices: usize,
    dst: usize,
    num_updates: u32,
    out_dim: u32,
    trailing: u32,
) {
    // Phase 0: zero the output buffer (out_dim * trailing u32 atomics).
    let out_total = out_dim * trailing;
    enc.set_compute_pipeline_state(&k.scatter_add_zero);
    enc.set_buffer(0, Some(buffer), dst as u64);
    enc.set_bytes(1, 4, &out_total as *const u32 as *const _);
    let tg_w0 = k
        .scatter_add_zero
        .thread_execution_width()
        .min(out_total as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: out_total as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w0,
            height: 1,
            depth: 1,
        },
    );

    // Phase 1: atomic accumulate.
    enc.set_compute_pipeline_state(&k.scatter_add_accumulate);
    enc.set_buffer(0, Some(buffer), updates as u64);
    enc.set_buffer(1, Some(buffer), indices as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &trailing as *const u32 as *const _);
    enc.set_bytes(4, 4, &num_updates as *const u32 as *const _);
    enc.set_bytes(5, 4, &out_dim as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: trailing as u64,
        height: num_updates as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 8.min(trailing as u64),
        height: 8.min(num_updates as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_grouped_matmul(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    input: usize,
    weight: usize,
    expert_idx: usize,
    dst: usize,
    m: u32,
    k_dim: u32,
    n: u32,
    num_experts: u32,
) {
    enc.set_compute_pipeline_state(&k.grouped_matmul);
    enc.set_buffer(0, Some(buffer), input as u64);
    enc.set_buffer(1, Some(buffer), weight as u64);
    enc.set_buffer(2, Some(buffer), expert_idx as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &m as *const u32 as *const _);
    enc.set_bytes(5, 4, &k_dim as *const u32 as *const _);
    enc.set_bytes(6, 4, &n as *const u32 as *const _);
    enc.set_bytes(7, 4, &num_experts as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n as u64,
        height: m as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 8.min(n as u64),
        height: 8.min(m as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_topk(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    outer: u32,
    axis_dim: u32,
    k_val: u32,
) {
    enc.set_compute_pipeline_state(&k.topk_lastax);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &axis_dim as *const u32 as *const _);
    enc.set_bytes(3, 4, &k_val as *const u32 as *const _);
    let tg_w = k.topk_lastax.thread_execution_width().min(outer as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: outer as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_reduce_axes(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    outer: u32,
    reduced: u32,
    inner: u32,
    op: rlx_ir::op::ReduceOp,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    use rlx_ir::op::ReduceOp;
    let op_kind: u32 = match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    };
    let pipeline = match dt {
        HalfFlag::F32 => &k.reduce_axes,
        HalfFlag::F16 => &k.reduce_axes_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(2, 4, &reduced as *const u32 as *const _);
    enc.set_bytes(3, 4, &inner as *const u32 as *const _);
    enc.set_bytes(4, 4, &op_kind as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: inner as u64,
        height: outer as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 16.min(inner as u64),
        height: 16.min(outer as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn encode_compare(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    lhs: usize,
    rhs: usize,
    dst: usize,
    len: u32,
    op: rlx_ir::op::CmpOp,
) {
    use rlx_ir::op::CmpOp;
    let op_kind: u32 = match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    };
    enc.set_compute_pipeline_state(&k.elem_compare);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &len as *const u32 as *const _);
    enc.set_bytes(4, 4, &op_kind as *const u32 as *const _);
    let tg_w = k.elem_compare.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

fn encode_where(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    cond: usize,
    on_true: usize,
    on_false: usize,
    dst: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(&k.elem_where);
    enc.set_buffer(0, Some(buffer), cond as u64);
    enc.set_buffer(1, Some(buffer), on_true as u64);
    enc.set_buffer(2, Some(buffer), on_false as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    let tg_w = k.elem_where.thread_execution_width().min(len as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}

/// Standalone softmax: one threadgroup per row, in-place exp+normalize.
/// Threadgroup size must be a power of 2 and ≤256 (the kernel's reduction
/// buffer). Picks the largest pow2 ≤ cols, capped at 256.
fn encode_softmax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    data: usize,
    rows: u32,
    cols: u32,
    dt: crate::thunk::HalfFlag,
) {
    use crate::thunk::HalfFlag;
    let pipeline = match dt {
        HalfFlag::F32 => &k.softmax_lastax,
        HalfFlag::F16 => &k.softmax_lastax_h,
    };
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), data as u64);
    enc.set_bytes(
        1,
        std::mem::size_of::<u32>() as u64,
        &cols as *const u32 as *const _,
    );
    // 1D dispatch: pack rows along width so threadgroup_position_in_grid.x
    // is the row index (the kernel's `row` parameter is a scalar uint, which
    // binds to .x — the same gotcha as encode_layer_norm).
    let grid = metal::MTLSize {
        width: tg_w * rows as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: tg_w,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

fn metal_concat_multi_enabled() -> bool {
    !rlx_ir::env::flag("RLX_METAL_CONCAT_MULTI")
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConcatSegGpu {
    src: u32,
    dst_col: u32,
    len: u32,
}

/// Dispatch a concat-along-last-axis. Uses one multi-segment kernel when possible.
fn encode_concat_lastax(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dst: usize,
    outer: u32,
    dst_axis: u32,
    dt: crate::thunk::HalfFlag,
    inputs: &[(usize, u32)],
) {
    use crate::thunk::HalfFlag;
    // GQA `repeat_kv` concats 16 head slices; `concat_lastax_multi{,4}` mis-copies
    // beyond 8 segments on Apple GPUs — use per-segment kernels instead.
    if inputs.len() >= 2
        && inputs.len() <= 8
        && matches!(dt, HalfFlag::F32)
        && metal_concat_multi_enabled()
        && inputs.len() <= NARROW_BATCH_MAX
    {
        let mut cum = 0u32;
        let segs: Vec<ConcatSegGpu> = inputs
            .iter()
            .map(|&(src_off, src_axis)| {
                let seg = ConcatSegGpu {
                    src: src_off as u32,
                    dst_col: cum,
                    len: src_axis,
                };
                cum += src_axis;
                seg
            })
            .collect();
        let num_seg = segs.len() as u32;
        let max_len = segs.iter().map(|s| s.len).max().unwrap_or(0);
        let use_vec4 = dst_axis.is_multiple_of(4)
            && cum == dst_axis
            && segs
                .iter()
                .all(|s| (s.dst_col % 4) == 0 && (s.len % 4) == 0 && s.len >= 4);
        if use_vec4 {
            let dst_axis4 = dst_axis / 4;
            let max_len4 = max_len / 4;
            enc.set_compute_pipeline_state(&k.concat_lastax_multi4);
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_bytes(1, 4, &(dst as u32) as *const u32 as *const _);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &dst_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
            enc.set_bytes(
                5,
                (segs.len() * std::mem::size_of::<ConcatSegGpu>()) as u64,
                segs.as_ptr() as *const _,
            );
            let grid = metal::MTLSize {
                width: max_len4 as u64,
                height: outer as u64,
                depth: num_seg as u64,
            };
            let tg = metal::MTLSize {
                width: 64.min(max_len4 as u64),
                height: 4.min(outer as u64),
                depth: num_seg as u64,
            };
            enc.dispatch_threads(grid, tg);
            return;
        }
        enc.set_compute_pipeline_state(&k.concat_lastax_multi);
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_bytes(1, 4, &(dst as u32) as *const u32 as *const _);
        enc.set_bytes(2, 4, &outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &dst_axis as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<ConcatSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let grid = metal::MTLSize {
            width: max_len as u64,
            height: outer as u64,
            depth: num_seg as u64,
        };
        let tg = metal::MTLSize {
            width: 32.min(max_len as u64),
            height: 8.min(outer as u64),
            depth: num_seg as u64,
        };
        enc.dispatch_threads(grid, tg);
        return;
    }

    let pipeline = match dt {
        HalfFlag::F32 => &k.concat_segment_lastax,
        HalfFlag::F16 => &k.concat_segment_lastax_h,
    };
    let mut cum: u32 = 0;
    for &(src_off, src_axis) in inputs {
        let use_vec4 = matches!(dt, HalfFlag::F32)
            && (src_axis % 4) == 0
            && dst_axis.is_multiple_of(4)
            && cum.is_multiple_of(4)
            && src_axis >= 4;
        if use_vec4 {
            let src_axis4 = src_axis / 4;
            let dst_axis4 = dst_axis / 4;
            let dst_col4 = cum / 4;
            enc.set_compute_pipeline_state(&k.concat_segment_lastax4);
            enc.set_buffer(0, Some(buffer), src_off as u64);
            enc.set_buffer(1, Some(buffer), dst as u64);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &dst_axis4 as *const u32 as *const _);
            enc.set_bytes(5, 4, &dst_col4 as *const u32 as *const _);
            let grid = metal::MTLSize {
                width: src_axis4 as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 64.min(src_axis4 as u64),
                height: 4.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        } else {
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(buffer), src_off as u64);
            enc.set_buffer(1, Some(buffer), dst as u64);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &dst_axis as *const u32 as *const _);
            enc.set_bytes(5, 4, &cum as *const u32 as *const _);
            let grid = metal::MTLSize {
                width: src_axis as u64,
                height: outer as u64,
                depth: 1,
            };
            let tg = metal::MTLSize {
                width: 16.min(src_axis as u64),
                height: 16.min(outer as u64),
                depth: 1,
            };
            enc.dispatch_threads(grid, tg);
        }
        cum += src_axis;
    }
}

/// Dispatch a FusedSwiGLU kernel. Picks the variant matching `(src_dt, dst_dt)`:
/// f32→f32, f16→f16, f32→f16 (cast), f16→f32 (cast).
fn encode_fused_swiglu(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    n_half: u32,
    total: u32,
    src_dt: crate::thunk::HalfFlag,
    dst_dt: crate::thunk::HalfFlag,
    gate_first: bool,
) {
    use crate::thunk::HalfFlag;
    let gate_first_u32 = u32::from(gate_first);
    let pipeline = match (src_dt, dst_dt) {
        (HalfFlag::F32, HalfFlag::F32) => &k.fused_swiglu,
        (HalfFlag::F16, HalfFlag::F16) => &k.fused_swiglu_h,
        (HalfFlag::F32, HalfFlag::F16) => &k.fused_swiglu_cast_f32_to_f16,
        (HalfFlag::F16, HalfFlag::F32) => &k.fused_swiglu_cast_f16_to_f32,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &n_half as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &total as *const u32 as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &gate_first_u32 as *const u32 as *const _,
    );
    let tg_w = pipeline.thread_execution_width().min(total as u64);
    enc.dispatch_threads(
        metal::MTLSize {
            width: total as u64,
            height: 1,
            depth: 1,
        },
        metal::MTLSize {
            width: tg_w,
            height: 1,
            depth: 1,
        },
    );
}
