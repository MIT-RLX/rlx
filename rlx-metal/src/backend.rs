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
}

unsafe impl Send for MetalExecutable {}

impl MetalExecutable {
    /// Compile at the requested precision.
    pub fn compile_with_precision(graph: Graph, precision: MetalPrecision) -> Self {
        // F16 compilation requires every kernel in the graph to have an f16
        // variant. Until they do, transparently fall back to F32 with a note.
        let effective = if precision == MetalPrecision::F16 {
            let verbose = std::env::var("RLX_VERBOSE")
                .ok()
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
        Self::compile_inner(graph, None)
    }

    /// Compile with an optional `PrecisionPolicy`. The pass runs *after*
    /// fusion to avoid breaking pattern-match-based fusion via interleaved
    /// Cast nodes.
    pub fn compile_with_policy(graph: Graph, policy: Option<rlx_opt::PrecisionPolicy>) -> Self {
        Self::compile_inner(graph, policy)
    }

    fn compile_inner(graph: Graph, policy: Option<rlx_opt::PrecisionPolicy>) -> Self {
        let verbose = std::env::var("RLX_VERBOSE")
            .ok()
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

        // Run fusion passes (shared with CPU backend — device-agnostic)
        let passes: Vec<&dyn rlx_opt::pass::Pass> = vec![
            // Lower DotGeneral first so the rest of the fusion pipeline only
            // sees MatMul (mirrors the CPU backend's pass list).
            &rlx_opt::LowerDotGeneral,
            &rlx_opt::fusion::FuseMatMulBiasAct,
            &rlx_opt::fusion::FuseResidualLN,
            &rlx_opt::fusion::FuseSharedInputMatMul,
            // Match silu(narrow)*narrow → FusedSwiGLU after the shared-input
            // matmul pass has produced the narrow pair.
            &rlx_opt::fusion::FuseSwiGLU,
            // PLAN L2: Metal has a native ElementwiseRegion thunk
            // (interpreted-chain MSL kernel) — collapse leftover
            // element-wise chains AFTER the big pattern fusions.
            &rlx_opt::MarkElementwiseRegions,
        ];
        let fused = rlx_opt::run_passes(graph, &passes, verbose);

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
        let plan = memory::plan_memory_aligned(&fused, 128);
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
                let len = fused.node(id).shape.num_elements().unwrap_or(0);
                (off, len)
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
        let mps_plan = if std::env::var("RLX_DISABLE_MPSGRAPH").is_ok() {
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

        // Optional ICB pre-encoding: opt-in via env var. Pre-encodes the
        // ICB-compatible thunks (small element-wise / norm / copy ops) into
        // an IndirectCommandBuffer at compile time so encode_and_run can
        // issue them as one `executeCommandsInBuffer` call instead of N
        // individual `set_pipeline + set_buffer + dispatch` round-trips.
        let icb_segments = if std::env::var("RLX_USE_ICB").is_ok() {
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
            icb_segments,
            pending_cmd_bufs: Vec::new(),
            active_extent: None,
            max_matmul_flops,
            mps_params_frozen: false,
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
        if self.mps_plan.is_none() {
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
        let cap_bytes = std::env::var("RLX_MPSGRAPH_PARAM_CONST_CAP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4 * 1024 * 1024);
        let arena_ptr = self.arena.buffer.contents() as *const u8;
        let mut param_bytes: HashMap<String, Vec<u8>> = HashMap::new();
        for (name, id) in &self.param_ids {
            let node = self.graph.node(*id);
            if !matches!(node.shape.dtype(), rlx_ir::DType::F32) {
                continue;
            }
            let n_elem = match node.shape.num_elements() {
                Some(n) => n,
                None => continue,
            };
            let len_bytes = n_elem * 4;
            if cap_bytes != 0 && len_bytes > cap_bytes {
                continue;
            }
            let off = self.arena.byte_offset(*id);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(arena_ptr.add(off), len_bytes).to_vec()
            };
            param_bytes.insert(name.clone(), bytes);
        }

        // Re-run lowering with the params marked as constants. Old
        // plan is dropped, which releases the old executable and
        // cached arrays.
        let new_plan = crate::mps_graph_lower::try_lower_with_constants(
            &self.graph,
            Some(&param_bytes),
        );
        if let Some(plan) = new_plan {
            self.mps_plan = Some(plan);
            // Re-bind the (now much smaller) feed list to the arena.
            self.bind_mps_executable_to_arena();
        }
    }

    fn bind_mps_executable_to_arena(&mut self) {
        let Some(plan) = self.mps_plan.as_mut() else { return };
        let Some(exec) = plan.executable.as_mut() else { return };
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
        if let Some(cmd_buf) = self.encode_commit(false, None) {
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
            if let Some(cmd_buf) = self.encode_commit(false, Some(&dests)) {
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
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name)
                && self.arena.has_buffer(id)
            {
                self.arena.write_from_f32(id, data);
            }
        }
        self.encode_and_run();
        // Read outputs as f32 regardless of native precision.
        self.graph
            .outputs
            .iter()
            .map(|&id| self.arena.read_as_f32(id))
            .collect()
    }

    fn encode_and_run(&mut self) {
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
            && self.mps_plan.is_some()
            && std::env::var("RLX_MPSGRAPH_PARAM_CONST").is_ok()
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
            let force = std::env::var("RLX_MPSGRAPH_FORCE").is_ok();
            let threshold = std::env::var("RLX_MPSGRAPH_MIN_FLOPS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1_000_000);
            if force || self.estimated_max_flops() >= threshold {
                self.run_via_mps_graph();
                return;
            }
        }
        // wait=true: synchronous, drop the buffer immediately after wait.
        // ICB segments (if any) are dispatched inline by encode_commit.
        let _ = self.encode_commit(true, None);
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
    ) -> Option<metal::CommandBuffer> {
        let trace = std::env::var("RLX_METAL_TRACE").is_ok();
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

        macro_rules! e {
            () => {{
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
        let mut i = 0usize;
        while i < thunks.len() {
            if active.is_none()
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
                    }
                }
                Thunk::ActivationInPlace { data, len, act, dt } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_activation(e!(), k, &self.arena.buffer, *data, len, *act, *dt);
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
                        let tg = metal::MTLSize {
                            width: 64.min(total_out as u64),
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
                        let lhs_len_in = inferred_input_len(&lhs_strides, &out_dims);
                        let rhs_len_in = inferred_input_len(&rhs_strides, &out_dims);
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
                Thunk::Copy { src, dst, len, dt } => {
                    let len = scale(*len);
                    if len == 0 {
                        continue;
                    }
                    encode_copy(e!(), k, &self.arena.buffer, *src, *dst, len, *dt);
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
                    dt,
                } => {
                    // PLAN L1: split seq into runtime-scaled bound +
                    // compile-time full-extent stride; safe at any batch.
                    let seq_stride = *seq;
                    let kv_stride = *kv_seq;
                    let seq = scale(*seq);
                    let kv_seq_eff = scale(*kv_seq);
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
                        kv_seq_eff,
                        kv_stride,
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
                    if *inner == 1 {
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
                        // Mid-shape concat (e.g. SAM windowed-attention pad
                        // along axis 1 or 2). The legacy kernel only does
                        // last-axis concat and was silently wrong here.
                        // Apple-Silicon unified memory makes the host
                        // copy cheap; total bytes is ≤ a few MB even for
                        // SAM's window-pad.
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
                        // For each outer row, copy each input's
                        // axis-slot contiguously.
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

                Thunk::Fft1d {
                    src,
                    dst,
                    outer,
                    n_complex,
                    inverse,
                    dtype,
                } => {
                    // Native MSL kernel is f32 + power-of-2 + N ≤ 2048
                    // (TG memory budget). Anything else falls through to
                    // the host fallback below — Apple GPUs have no f64,
                    // and a Bluestein Metal kernel would be its own
                    // future PR. Set RLX_METAL_FFT_HOST_FALLBACK=1 to
                    // force the host path for debugging.
                    let force_host =
                        std::env::var("RLX_METAL_FFT_HOST_FALLBACK").is_ok();
                    let n = *n_complex as usize;
                    let can_native = !force_host
                        && matches!(dtype, rlx_ir::DType::F32)
                        && n.is_power_of_two()
                        && n <= 2048
                        && n >= 2;
                    if can_native {
                        let enc = e!();
                        enc.set_compute_pipeline_state(&k.fft_radix2_f32);
                        enc.set_buffer(0, Some(&self.arena.buffer), 0);
                        let src_u32 = (*src as u64 / 4) as u32; // f32 index
                        let dst_u32 = (*dst as u64 / 4) as u32;
                        let n_u32 = n as u32;
                        let log2n: u32 = n.trailing_zeros();
                        let inv_u32 = if *inverse { 1u32 } else { 0u32 };
                        enc.set_bytes(1, 4, &src_u32 as *const u32 as *const _);
                        enc.set_bytes(2, 4, &dst_u32 as *const u32 as *const _);
                        enc.set_bytes(3, 4, &n_u32 as *const u32 as *const _);
                        enc.set_bytes(4, 4, &log2n as *const u32 as *const _);
                        enc.set_bytes(5, 4, &inv_u32 as *const u32 as *const _);
                        // One TG per row, TG size = min(N/2, 1024).
                        let tg_w = (n as u64 / 2).clamp(1, 1024);
                        let grid = metal::MTLSize {
                            width: *outer as u64 * tg_w,
                            height: 1,
                            depth: 1,
                        };
                        let tg = metal::MTLSize {
                            width: tg_w,
                            height: 1,
                            depth: 1,
                        };
                        enc.dispatch_threads(grid, tg);
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
                                    arena_ptr,
                                ),
                                rlx_ir::DType::F64 => rlx_cpu::thunk::execute_fft1d_f64(
                                    *src,
                                    *dst,
                                    *outer as usize,
                                    n,
                                    *inverse,
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
        let plan = self.mps_plan.as_ref().expect("plan present");
        let dev = metal_device().expect("Metal device");

        // Build feed lists: graph inputs + params, in the order the plan
        // captured them. For each, the arena offset is known via input_ids /
        // param_ids and the underlying graph node.
        let arena_buf = &self.arena.buffer;

        let mut feed_tensors: Vec<&crate::mps_graph::MpsTensor> = Vec::new();
        let mut feed_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut feed_offsets: Vec<usize> = Vec::new();
        let mut feed_shapes: Vec<Vec<usize>> = Vec::new();
        let mut feed_dtypes: Vec<u32> = Vec::new();

        for (name, t, shape, dt) in &plan.inputs {
            let id = self.input_ids.get(name).expect("input id");
            let off = self.arena.byte_offset(*id);
            feed_tensors.push(t);
            feed_buffers.push(arena_buf);
            feed_offsets.push(off);
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }
        for (name, t, shape, dt) in &plan.params {
            let id = self.param_ids.get(name).expect("param id");
            let off = self.arena.byte_offset(*id);
            feed_tensors.push(t);
            feed_buffers.push(arena_buf);
            feed_offsets.push(off);
            feed_shapes.push(shape.clone());
            feed_dtypes.push(*dt);
        }

        // Result outputs: write back into the arena at each output's offset.
        let mut out_tensors: Vec<&crate::mps_graph::MpsTensor> = Vec::new();
        let mut out_buffers: Vec<&metal::Buffer> = Vec::new();
        let mut out_offsets: Vec<usize> = Vec::new();
        let mut out_shapes: Vec<Vec<usize>> = Vec::new();
        let mut out_dtypes: Vec<u32> = Vec::new();
        for (id, t, shape, dt) in &plan.outputs {
            let off = self.arena.byte_offset(*id);
            out_tensors.push(t);
            out_buffers.push(arena_buf);
            out_offsets.push(off);
            out_shapes.push(shape.clone());
            out_dtypes.push(*dt);
        }

        if let Some(exec) = plan.executable.as_ref() {
            // bind_arena ran at compile — hot path is a single ObjC
            // dispatch with no per-call allocation.
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
    // 1D grid: row index lives in threadgroup_position_in_grid.x. The kernel
    // reads `row` as a uint scalar which binds to the .x component, so
    // packing rows along width is what makes the multi-row dispatch work.
    let tg_w = 256u64.min(h as u64);
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
    // f16 covers Add and Mul (the Nomic residual + SwiGLU patterns).
    // Other binaries silently fall back to f32 kernels in mixed
    // precision — same caveat as encode_activation.
    let pipeline = match (dt, op) {
        (HalfFlag::F16, BinaryOp::Add) => &k.elem_add_h,
        (HalfFlag::F16, BinaryOp::Mul) => &k.elem_mul_h,
        (_, BinaryOp::Add) => &k.elem_add,
        (_, BinaryOp::Mul) => &k.elem_mul,
        (_, BinaryOp::Sub) => &k.elem_sub,
        (_, BinaryOp::Div) => &k.elem_div,
        (_, BinaryOp::Max) => &k.elem_max,
        (_, BinaryOp::Min) => &k.elem_min,
        (_, BinaryOp::Pow) => &k.elem_pow,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), lhs as u64);
    enc.set_buffer(1, Some(buffer), rhs as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &len as *const u32 as *const _,
    );
    let tg_w = pipeline.thread_execution_width().min(len as u64);
    let grid = metal::MTLSize {
        width: len as u64,
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
    let pipeline = match dt {
        HalfFlag::F32 => &k.narrow_lastax,
        HalfFlag::F16 => &k.narrow_lastax_h,
    };
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &outer as *const u32 as *const _,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &src_axis as *const u32 as *const _,
    );
    enc.set_bytes(
        4,
        std::mem::size_of::<u32>() as u64,
        &start as *const u32 as *const _,
    );
    enc.set_bytes(
        5,
        std::mem::size_of::<u32>() as u64,
        &len as *const u32 as *const _,
    );
    let grid = metal::MTLSize {
        width: len as u64,
        height: outer as u64,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 16.min(len as u64),
        height: 16.min(outer as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
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
    // Same .x-binding gotcha as encode_layer_norm: row index must land in
    // threadgroup_position_in_grid.x, so we put `rows` in tg_count.width.
    let tg_w = 256u64.min(h as u64);
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
    kv_seq: u32,
    kv_stride: u32,
) {
    use crate::thunk::HalfFlag;
    // The two-pass `sdpa` / `sdpa_h` kernels store an [seq, seq] scores
    // matrix in threadgroup memory (`scores[64*64]`); they're correct
    // only for seq ≤ 64. For longer sequences (e.g. NomicVision's seq=257
    // = 256 patches + 1 CLS) we route to `sdpa_long`, an online-softmax
    // FA-v1 variant that's O(D) memory per query row and scales to any
    // seq length. F16 input/output isn't supported by sdpa_long yet —
    // that path falls through and would hit the seq-64 ceiling; today
    // no f16-tagged graph hits seq>64 in production.
    if seq > 64 && matches!(dt, HalfFlag::F32) {
        // Pick between the scalar online-softmax (`sdpa_long`) and the
        // tile-based flash-attention (`sdpa_fa_f32`). FA amortizes K/V
        // reads across an 8-query tile via threadgroup memory, so it
        // wins over `sdpa_long` (~35% faster) when Lk dominates. It
        // still lags MPSGraph's batched matmul decomp for SAM3 image
        // CA (Lq=201, Lk=5184, dh=16) because MPSGraph uses
        // simdgroup_float8x8 internally; opt-in via `RLX_METAL_FA=1`
        // for benchmarking until the kernel is upgraded to use
        // simdgroup matrix primitives.
        let use_fa = kv_seq >= 256 && head_dim <= 32 && std::env::var("RLX_METAL_FA").is_ok();
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
        if use_fa {
            // FA kernel: 1 TG per (q_tile, head, batch), 64 threads, Br=8.
            const BR: u32 = 8;
            let q_tiles = (seq + BR - 1) / BR;
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
    let nh = hidden / head_dim;
    let half = head_dim / 2;
    let grid = metal::MTLSize {
        width: half as u64,
        height: nh as u64,
        depth: (batch * seq) as u64,
    };
    let tg = metal::MTLSize {
        width: half.min(16) as u64,
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
    // Rows packed in width — same .x scalar binding gotcha as encode_layer_norm.
    let tg_w = 256u64.min(h as u64);
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
    enc.set_compute_pipeline_state(&k.conv2d);
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

/// Dispatch a concat-along-last-axis as N segment-kernel calls, one per
/// input tensor. Each segment writes its source buffer into the
/// corresponding slice of the destination's last dimension.
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
    let pipeline = match dt {
        HalfFlag::F32 => &k.concat_segment_lastax,
        HalfFlag::F16 => &k.concat_segment_lastax_h,
    };
    let mut cum: u32 = 0;
    for &(src_off, src_axis) in inputs {
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
) {
    use crate::thunk::HalfFlag;
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
