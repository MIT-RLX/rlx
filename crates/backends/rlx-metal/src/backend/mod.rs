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
use std::collections::HashMap;

use crate::arena::Arena;
use crate::device::metal_device;
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
    /// Arena tail scratch for in-graph onnx.QMatMul act dequant (f32).
    onnx_qmatmul_act_scratch_off: usize,
    /// Cached dequant f32 weights for in-graph onnx.QMatMul.
    qmatmul_weight_cache: std::cell::RefCell<crate::onnx_qmatmul::QMatMulWeightCache>,
    /// Persistent KV / state inputs (unified-memory `Vec`, fed into arena each run).
    gpu_handles: HashMap<String, Vec<f32>>,
    /// After each run, copy `graph.outputs[idx]` into the named handle.
    gpu_handle_feeds: HashMap<String, usize>,
    /// Handles whose arena input slots are authoritative (skip host mirror ping-pong).
    gpu_handle_resident: std::collections::HashSet<String>,
    /// `handle_name → output index` for the resident-KV *row* feed (decode graphs
    /// that emit the new token at the last bucket-padded output row, e.g. llama32
    /// `concat(past_k, k_new)`). Driven via [`feed_kv_row`]; kept separate from
    /// `gpu_handle_feeds` so the generic prefix propagation never fires for these.
    kv_row_feeds: HashMap<String, usize>,
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

mod bind;
mod compile;
mod encode;
mod output;
mod read;
mod run;
mod set;

impl MetalExecutable {
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

    pub(crate) fn estimated_max_flops(&self) -> u64 {
        self.max_matmul_flops
    }

    pub fn arena_ptr(&self) -> *const u8 {
        self.arena.buffer.contents() as *const u8
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

    /// Wait for every command buffer queued by `commit_no_wait`.
    pub fn sync_pending(&mut self) {
        for cb in self.pending_cmd_bufs.drain(..) {
            cb.wait_until_completed();
        }
    }

    /// Copy all named params from another executable with matching param layout.
    pub fn copy_params_from(&mut self, other: &Self) -> bool {
        if self.param_ids.len() != other.param_ids.len() {
            return false;
        }
        for (name, &dst_id) in &self.param_ids {
            let Some(&src_id) = other.param_ids.get(name) else {
                return false;
            };
            if !self.arena.has_buffer(dst_id) || !other.arena.has_buffer(src_id) {
                return false;
            }
            let dst_cap = *self.arena.element_counts.get(&dst_id).unwrap_or(&0);
            let src_cap = *other.arena.element_counts.get(&src_id).unwrap_or(&0);
            if dst_cap != src_cap {
                return false;
            }
            self.arena
                .copy_node_bytes_from(dst_id, &other.arena, src_id);
        }
        self.preload_qmatmul_weights();
        true
    }

    /// Warm the in-graph QMatMul weight dequant cache after all params are loaded.
    pub fn preload_qmatmul_weights(&mut self) {
        if !crate::onnx_qmatmul::ingraph_enabled() {
            return;
        }
        let arena_ptr = self.arena.buffer.contents() as *const u8;
        let mut cache = self.qmatmul_weight_cache.borrow_mut();
        let before = cache.len();
        for thunk in &self.schedule.thunks {
            let Thunk::CustomOp { kernel, inputs, .. } = thunk else {
                continue;
            };
            if kernel.name() != crate::onnx_qmatmul::KERNEL_NAME || inputs.len() < 6 {
                continue;
            }
            let read_input = |idx: usize| -> (&[u8], &rlx_ir::Shape) {
                let (off, len, shape) = &inputs[idx];
                let nbytes = (*len as usize) * shape.dtype().size_bytes();
                let data = unsafe { std::slice::from_raw_parts(arena_ptr.add(*off), nbytes) };
                (data, shape)
            };
            let (w_b, w_sh) = read_input(3);
            let (w_scale_b, _) = read_input(4);
            let (w_zp_b, w_zp_sh) = read_input(5);
            let k = w_sh.dim(0).unwrap_static().max(1);
            let n = w_sh.dim(1).unwrap_static().max(1);
            let w_scale = crate::onnx_qmatmul::read_f32_scalar(w_scale_b);
            let w_zp = crate::onnx_qmatmul::read_zp_i32(w_zp_b, w_zp_sh.dtype());
            cache.preload_weight(inputs[3].0, w_b, w_sh.dtype(), k, n, w_zp, w_scale);
        }
        let loaded = cache.len().saturating_sub(before);
        if loaded > 0 && rlx_ir::env::flag("KITTEN_RLX_TIMING") {
            eprintln!("[metal] QMatMul weight preload: {loaded} tiles");
        }
    }

    /// Current RNG compile/execute policy.
    pub fn rng(&self) -> rlx_ir::RngOptions {
        *self.schedule.rng.read().expect("rng lock")
    }

    /// True when every thunk in the schedule is safe for active-extent
    /// dispatch — guards `encode_commit`'s bypass of MPSGraph + ICB.
    pub(crate) fn all_safe_for_active(&self) -> bool {
        self.schedule
            .thunks
            .iter()
            .all(|t| t.safe_for_active_extent())
    }

    pub fn has_gpu_handle(&self, name: &str) -> bool {
        self.gpu_handles.contains_key(name)
    }

    /// Register a resident-KV *row* feed (vs the generic prefix feed): row
    /// `src_row` of output `output_index` is folded into handle `handle_name`'s
    /// input slot at `dst_row` by [`feed_kv_row`]. For decode graphs that emit
    /// the new token at the last bucket-padded output row (llama32).
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) {
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Fold each registered row feed's new-token row into its resident handle
    /// slot, in-place on the unified-memory arena. Call after a logits-only run.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) {
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            if out_idx >= self.graph.outputs.len() {
                continue;
            }
            let out_id = self.graph.outputs[out_idx];
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                self.arena.copy_node_f32_range(
                    in_id,
                    dst_row * row_elems,
                    out_id,
                    src_row * row_elems,
                    row_elems,
                );
            }
            self.gpu_handle_resident.insert(name.clone());
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    /// Clone into an independent executable (recompiles from the stored graph).
    pub fn clone_for_cache(&self) -> Self {
        let mut exe = Self::compile_from_fused(
            self.graph.clone(),
            None,
            None,
            rlx_ir::RngOptions::default(),
        );
        // `compile_from_fused` re-initializes `Op::Constant` slots but leaves the
        // fresh arena's `Op::Param` (weight) slots zeroed — params are uploaded
        // externally via `set_param`/`finalize_params` after the first compile and
        // are NOT part of the graph. Without copying them, a cached/reused clone
        // (e.g. the in-memory graph cache in a persistent TTS service) runs with
        // all-zero weights: embeddings/matmuls collapse to zero on the 2nd+ run.
        exe.copy_params_from(self);
        for (name, data) in &self.gpu_handles {
            if !data.is_empty() {
                exe.bind_gpu_handle(name, data);
            }
        }
        for (name, &idx) in &self.gpu_handle_feeds {
            exe.set_gpu_handle_feed(name, idx);
        }
        for (name, &idx) in &self.kv_row_feeds {
            exe.register_kv_row_feed(name, idx);
        }
        exe.set_active_extent(self.active_extent);
        exe
    }

    pub(crate) fn propagate_gpu_handle_feeds_in_arena(&mut self) {
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

    pub(crate) fn refresh_gpu_handles_from_outputs(&mut self) {
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

    pub(crate) fn dispatch_mps_plan(
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
        let flops = match &node.op {
            Op::MatMul | Op::FusedMatMulBiasAct { .. } => {
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
                (m_dim as u64) * (k_dim as u64) * (n_dim as u64)
            }
            // Conv (forward + gradients) is the bulk of a CNN's compute but was
            // invisible here — so a conv-heavy graph looked "tiny" (matmul-only)
            // and the adaptive dispatch skipped its own MPSGraph plan, losing the
            // ~2.6× fusion win. Count conv as out_elems × per-output MACs
            // (C_in/g·kH·kW = weight elems / C_out); the gradients are the same
            // order, so the forward conv alone is enough to cross the threshold.
            Op::Conv { .. } | Op::Conv2dBackwardInput { .. } | Op::Conv2dBackwardWeight { .. } => {
                let out_total: usize = match node.shape.num_elements() {
                    Some(v) => v,
                    None => continue,
                };
                // weight is input[1] for Conv / BackwardInput, input shapes vary
                // for BackwardWeight (output IS the weight) — use the largest
                // input's per-element fan to stay an order-of-magnitude estimate.
                let w_id = *node.inputs.last().unwrap_or(&node.inputs[0]);
                let w_shape = &graph.node(w_id).shape;
                let w_total: usize = match w_shape.num_elements() {
                    Some(v) => v,
                    None => continue,
                };
                let c_out = match w_shape.dim(0) {
                    d if d.is_static() => d.unwrap_static().max(1),
                    _ => 1,
                };
                (out_total as u64) * (w_total as u64 / c_out as u64).max(1)
            }
            _ => continue,
        };
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

fn widen_input_bytes_to_f32(data: &[u8], dt: rlx_ir::DType) -> Vec<f32> {
    use rlx_ir::DType;
    match dt {
        DType::F32 => {
            let n = data.len() / 4;
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) }.to_vec()
        }
        DType::F16 => {
            let n = data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, n) };
            s.iter().map(|h| h.to_f32()).collect()
        }
        DType::BF16 => {
            let n = data.len() / 2;
            let s = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, n) };
            s.iter().map(|h| h.to_f32()).collect()
        }
        // Integer/bool inputs widen to f32 — `widen_integer_activations_to_f32`
        // rewrites their arena slots to F32, so this matches the graph dtype.
        DType::I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::U32 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::Bool => data.iter().map(|&b| b as f32).collect(),
        other => panic!(
            "rlx-metal widen_input_bytes_to_f32: dtype {other:?} unsupported \
             (use direct byte write for F64/U8/I8 dtypes)"
        ),
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

fn encode_gelu_approx_out(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    len: u32,
) {
    debug_assert!(
        len.is_multiple_of(4) && len >= 4,
        "gelu_approx_out expects vec4 len"
    );
    let len4 = len / 4;
    enc.set_compute_pipeline_state(&k.gelu_approx_out4);
    enc.set_buffer(0, Some(buffer), 0);
    let src_u = src as u64;
    let dst_u = dst as u64;
    enc.set_bytes(1, 8, &src_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &dst_u as *const u64 as *const _);
    enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
    let tg_w = k.gelu_approx_out4.thread_execution_width().min(len4 as u64);
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
        && matches!(act, Activation::Gelu)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.gelu_inplace4);
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(2, 4, &len4 as *const u32 as *const _);
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
        && matches!(act, Activation::GeluApprox)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.gelu_approx_inplace4);
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(2, 4, &len4 as *const u32 as *const _);
        let tg_w = k
            .gelu_approx_inplace4
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
    if matches!(dt, HalfFlag::F32)
        && len.is_multiple_of(4)
        && len >= 4
        && matches!(act, Activation::Silu)
    {
        let len4 = len / 4;
        enc.set_compute_pipeline_state(&k.silu_inplace4);
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(2, 4, &len4 as *const u32 as *const _);
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
        (HalfFlag::F16, Activation::Gelu) => &k.gelu_inplace_h,
        (HalfFlag::F16, Activation::GeluApprox) => &k.gelu_approx_inplace_h,
        (HalfFlag::F16, Activation::Silu) => &k.silu_inplace_h,
        (_, Activation::Gelu) => &k.gelu_inplace,
        (_, Activation::GeluApprox) => &k.gelu_approx_inplace,
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
        (_, Activation::Round) => &k.round_inplace,
    };
    enc.set_compute_pipeline_state(pipeline);
    if matches!(dt, HalfFlag::F32)
        && matches!(
            act,
            Activation::Gelu | Activation::GeluApprox | Activation::Silu
        )
    {
        // Task #50: arena base + byte offset for activations past 4 GB.
        enc.set_buffer(0, Some(buffer), 0);
        let off = data_off as u64;
        enc.set_bytes(1, 8, &off as *const u64 as *const _);
        enc.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &len as *const u32 as *const _,
        );
    } else {
        enc.set_buffer(0, Some(buffer), data_off as u64);
        enc.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &len as *const u32 as *const _,
        );
    }
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
        enc.set_buffer(0, Some(buffer), 0);
        let a_u = a as u64;
        let b_u = b as u64;
        let d_u = dst as u64;
        enc.set_bytes(1, 8, &a_u as *const u64 as *const _);
        enc.set_bytes(2, 8, &b_u as *const u64 as *const _);
        enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
        enc.set_bytes(4, 4, &len4 as *const u32 as *const _);
        enc.set_bytes(5, 4, &op as *const u32 as *const _);
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
    enc.set_buffer(0, Some(buffer), 0);
    let a_u = a as u64;
    let b_u = b as u64;
    let d_u = dst as u64;
    enc.set_bytes(1, 8, &a_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &b_u as *const u64 as *const _);
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    enc.set_bytes(5, 4, &op as *const u32 as *const _);
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
    let dispatch_len = if use_vec4 { len / 4 } else { len };
    enc.set_compute_pipeline_state(pipeline);
    let use_arena_off = matches!(dt, HalfFlag::F32)
        && matches!(
            op,
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::Sub | BinaryOp::Div
        );
    if use_arena_off {
        // Task #50: arena base + byte offsets for tensors past 4 GB.
        enc.set_buffer(0, Some(buffer), 0);
        let lhs_u64 = lhs as u64;
        let rhs_u64 = rhs as u64;
        let dst_u64 = dst as u64;
        enc.set_bytes(1, 8, &lhs_u64 as *const u64 as *const _);
        enc.set_bytes(2, 8, &rhs_u64 as *const u64 as *const _);
        enc.set_bytes(3, 8, &dst_u64 as *const u64 as *const _);
        enc.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &dispatch_len as *const u32 as *const _,
        );
    } else {
        enc.set_buffer(0, Some(buffer), lhs as u64);
        enc.set_buffer(1, Some(buffer), rhs as u64);
        enc.set_buffer(2, Some(buffer), dst as u64);
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &dispatch_len as *const u32 as *const _,
        );
    }
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
        enc.set_buffer(0, Some(buffer), 0);
        let src_u64 = src as u64;
        let dst_u64 = dst as u64;
        enc.set_bytes(1, 8, &src_u64 as *const u64 as *const _);
        enc.set_bytes(2, 8, &dst_u64 as *const u64 as *const _);
        enc.set_bytes(3, 4, &len4 as *const u32 as *const _);
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
    enc.set_buffer(0, Some(buffer), 0);
    let src_u64 = src as u64;
    let dst_u64 = dst as u64;
    enc.set_bytes(1, 8, &src_u64 as *const u64 as *const _);
    enc.set_bytes(2, 8, &dst_u64 as *const u64 as *const _);
    enc.set_bytes(3, 4, &dispatch_len as *const u32 as *const _);
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
    // u64 for the same reason as ConcatSegGpu — task #50: ≥4 GB Q4 models
    // have activation byte offsets that exceed u32.
    dst: u64,
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
            dst: *dst as u64,
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
        // Bind to arena base + pass src byte offset (task #50).
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_bytes(2, 4, &batch.outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<NarrowSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let src_u64 = batch.src as u64;
        enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
        let grid = metal::MTLSize {
            width: max_len4 as u64,
            height: batch.outer as u64,
            depth: num_seg as u64,
        };
        // Task #50: cap total threads per threadgroup at 1024.
        let tg_depth = (1024u64 / (64 * 4)).min(num_seg as u64).max(1);
        let tg = metal::MTLSize {
            width: 64.min(max_len4 as u64),
            height: 4.min(batch.outer as u64),
            depth: tg_depth,
        };
        enc.dispatch_threads(grid, tg);
    } else {
        enc.set_compute_pipeline_state(&k.split_lastax);
        // Bind to arena base + pass src byte offset (task #50).
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_bytes(2, 4, &batch.outer as *const u32 as *const _);
        enc.set_bytes(3, 4, &batch.src_axis as *const u32 as *const _);
        enc.set_bytes(4, 4, &num_seg as *const u32 as *const _);
        enc.set_bytes(
            5,
            (segs.len() * std::mem::size_of::<NarrowSegGpu>()) as u64,
            segs.as_ptr() as *const _,
        );
        let src_u64 = batch.src as u64;
        enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
        let grid = metal::MTLSize {
            width: max_len as u64,
            height: batch.outer as u64,
            depth: num_seg as u64,
        };
        let tg_depth = (1024u64 / (32 * 8)).min(num_seg as u64).max(1);
        let tg = metal::MTLSize {
            width: 32.min(max_len as u64),
            height: 8.min(batch.outer as u64),
            depth: tg_depth,
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
            // Task #50: bind to arena base + pass byte offsets as ulong.
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &start4 as *const u32 as *const _);
            enc.set_bytes(5, 4, &len4 as *const u32 as *const _);
            let src_u64 = src as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
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
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &start as *const u32 as *const _);
            enc.set_bytes(5, 4, &len as *const u32 as *const _);
            let src_u64 = src as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
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
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &start as *const u32 as *const _);
            enc.set_bytes(5, 4, &len as *const u32 as *const _);
            let src_u64 = src as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
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
    score_scale: f32,
    attn_logit_softcap: f32,
) {
    // The kernels read these as constants right after `window`. Sentinel
    // `0.0` keeps the existing default (`1/sqrt(head_dim)`, no softcap).
    // Honour caller's score_scale (Gemma 4 sets 1.0); pass 0.0 (sentinel)
    // so the kernel computes the default itself. This matches the
    // historical MSL behaviour where `score_scale` was nonexistent.
    let kernel_score_scale: f32 = score_scale;
    let kernel_softcap: f32 = attn_logit_softcap;
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
        let use_decode_m1 = seq == 1
            && kv_seq != seq
            && head_dim <= 512
            && rlx_ir::env::var("RLX_METAL_SDPA_DECODE_M1").as_deref() != Some("0");
        let pipeline = if use_fa {
            &k.sdpa_fa_f32
        } else if use_decode_m1 {
            &k.sdpa_decode_m1
        } else {
            &k.sdpa_long
        };
        enc.set_compute_pipeline_state(pipeline);
        // Bind to arena base (offset 0) and pass byte offsets via inline
        // constants — large `set_buffer` offsets silently lose kernel writes
        // on M-series at offsets ≥ ~4 GB (task #50, same pattern as the
        // non-long `sdpa` path below).
        enc.set_buffer(0, Some(buffer), 0);
        enc.set_buffer(1, Some(buffer), 0);
        enc.set_buffer(2, Some(buffer), 0);
        enc.set_buffer(3, Some(buffer), 0);
        enc.set_buffer(4, Some(buffer), 0);
        enc.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &batch as *const u32 as *const _,
        );
        if use_decode_m1 {
            enc.set_bytes(
                6,
                std::mem::size_of::<u32>() as u64,
                &heads as *const u32 as *const _,
            );
            enc.set_bytes(
                7,
                std::mem::size_of::<u32>() as u64,
                &head_dim as *const u32 as *const _,
            );
            enc.set_bytes(
                8,
                std::mem::size_of::<u32>() as u64,
                &seq_stride as *const u32 as *const _,
            );
            enc.set_bytes(
                9,
                std::mem::size_of::<u32>() as u64,
                &mask_kind as *const u32 as *const _,
            );
            enc.set_bytes(
                10,
                std::mem::size_of::<u32>() as u64,
                &kv_seq as *const u32 as *const _,
            );
            enc.set_bytes(
                11,
                std::mem::size_of::<u32>() as u64,
                &kv_stride as *const u32 as *const _,
            );
            enc.set_bytes(
                12,
                std::mem::size_of::<u32>() as u64,
                &bhsd as *const u32 as *const _,
            );
            enc.set_bytes(
                13,
                std::mem::size_of::<u32>() as u64,
                &window as *const u32 as *const _,
            );
            enc.set_bytes(
                14,
                std::mem::size_of::<f32>() as u64,
                &kernel_score_scale as *const f32 as *const _,
            );
            enc.set_bytes(
                15,
                std::mem::size_of::<f32>() as u64,
                &kernel_softcap as *const f32 as *const _,
            );
            let long_offs_pack: [u64; 5] =
                [q as u64, k_off as u64, v as u64, mask as u64, out as u64];
            enc.set_bytes(
                16,
                (5 * std::mem::size_of::<u64>()) as u64,
                long_offs_pack.as_ptr() as *const _,
            );
            let total = (batch as u64) * (heads as u64);
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
            return;
        }
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
        enc.set_bytes(
            15,
            std::mem::size_of::<f32>() as u64,
            &kernel_score_scale as *const f32 as *const _,
        );
        enc.set_bytes(
            16,
            std::mem::size_of::<f32>() as u64,
            &kernel_softcap as *const f32 as *const _,
        );
        let long_offs_pack: [u64; 5] = [q as u64, k_off as u64, v as u64, mask as u64, out as u64];
        enc.set_bytes(
            17,
            (5 * std::mem::size_of::<u64>()) as u64,
            long_offs_pack.as_ptr() as *const _,
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
    // 12B Q4 GGUF activations sit at arena offsets > 4 GB. `set_buffer`'s
    // `offset` param is NSUInteger (64-bit) so the API takes it, but in
    // practice writes from kernels bound this way silently get dropped
    // (task #50 — sentinel `OUT[i]=7.0` doesn't reach the slot, even
    // though CPU writes at the same byte offset DO show up). Workaround:
    // bind to (buffer, 0) and pass byte offsets as ulong constants; the
    // kernel adds them itself. Q4K dequant uses this pattern and works
    // for offsets ≥ 14 GB.
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_buffer(1, Some(buffer), 0);
    enc.set_buffer(2, Some(buffer), 0);
    enc.set_buffer(3, Some(buffer), 0);
    enc.set_buffer(4, Some(buffer), 0);
    let q_off = q as u64;
    let k_off_u = k_off as u64;
    let v_off = v as u64;
    let m_off = mask as u64;
    let o_off = out as u64;
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
    enc.set_bytes(
        15,
        std::mem::size_of::<f32>() as u64,
        &kernel_score_scale as *const f32 as *const _,
    );
    enc.set_bytes(
        16,
        std::mem::size_of::<f32>() as u64,
        &kernel_softcap as *const f32 as *const _,
    );
    // Pack 5 byte-offsets into one inline-constant buffer (5×u64 = 40 bytes).
    // Setting them individually at buffer indices 17..21 turned out to bind
    // the SAME value to all five slots — Metal's argument table seemed to
    // alias them past index 16. Packing into one struct sidesteps that
    // (task #50, post-u64 dequant fix).
    let offs_pack: [u64; 5] = [q_off, k_off_u, v_off, m_off, o_off];
    enc.set_bytes(
        17,
        (5 * std::mem::size_of::<u64>()) as u64,
        offs_pack.as_ptr() as *const _,
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

/// Native block-quantized (int8 / int4) weight matmul over the unified-memory
/// arena. `out[m,n] = x[m,k] @ dequant(wq)`, one GPU thread per output element.
#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    zp: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
    block_size: u32,
    asym: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), zp as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &m as *const u32 as *const _);
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    enc.set_bytes(8, sz, &block_size as *const u32 as *const _);
    enc.set_bytes(9, sz, &asym as *const u32 as *const _);
    let total = (m * n) as u64;
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

#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul_fp8(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
    e5m2: u32,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &m as *const u32 as *const _);
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    enc.set_bytes(8, sz, &e5m2 as *const u32 as *const _);
    let total = (m * n) as u64;
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

#[allow(clippy::too_many_arguments)]
fn encode_dequant_matmul_nvfp4(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    scale: usize,
    global_scale: usize,
    dst: usize,
    m: u32,
    k: u32,
    n: u32,
) {
    use rlx_ir::NVFP4_GROUP_SIZE;
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), w_q as u64);
    enc.set_buffer(2, Some(buffer), scale as u64);
    enc.set_buffer(3, Some(buffer), global_scale as u64);
    enc.set_buffer(4, Some(buffer), dst as u64);
    let sz = std::mem::size_of::<u32>() as u64;
    enc.set_bytes(5, sz, &m as *const u32 as *const _);
    enc.set_bytes(6, sz, &k as *const u32 as *const _);
    enc.set_bytes(7, sz, &n as *const u32 as *const _);
    let gs = NVFP4_GROUP_SIZE as u32;
    enc.set_bytes(8, sz, &gs as *const u32 as *const _);
    let total = (m * n) as u64;
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
    cos_per_token: bool,
    interleaved: bool,
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
    let cos_per_token_u32: u32 = cos_per_token as u32;
    enc.set_bytes(
        11,
        std::mem::size_of::<u32>() as u64,
        &cos_per_token_u32 as *const u32 as *const _,
    );
    let interleaved_u32: u32 = interleaved as u32;
    enc.set_bytes(
        12,
        std::mem::size_of::<u32>() as u64,
        &interleaved_u32 as *const u32 as *const _,
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
    // Task #50: arena base + byte offsets for activations past 4 GB.
    enc.set_buffer(0, Some(buffer), 0);
    let src_u64 = src as u64;
    let g_u64 = g as u64;
    let b_u64 = b as u64;
    let dst_u64 = dst as u64;
    enc.set_bytes(1, 8, &src_u64 as *const u64 as *const _);
    enc.set_bytes(2, 8, &g_u64 as *const u64 as *const _);
    enc.set_bytes(3, 8, &b_u64 as *const u64 as *const _);
    enc.set_bytes(4, 8, &dst_u64 as *const u64 as *const _);
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

/// True when the native `fused_attn_block` MSL kernel can serve this
/// `Op::FusedAttentionBlock`: f32, no bias, rank-3, even head_dim, and a
/// `[seq,seq]` score matrix that fits the kernel's `threadgroup float[64*64]`
/// (`seq ≤ 64`). Everything else decomposes to the primitive chain.
fn fab_is_native(node: &rlx_ir::Node) -> bool {
    if let Op::FusedAttentionBlock {
        head_dim, has_bias, ..
    } = &node.op
    {
        let dims = node.shape.dims();
        dims.len() == 3
            && !*has_bias
            && node.shape.dtype() == rlx_ir::DType::F32
            && dims[1].unwrap_static() <= 64
            && *head_dim % 2 == 0
    } else {
        false
    }
}

/// Decompose non-native `Op::FusedAttentionBlock` nodes to the primitive chain
/// (via the shared `expand_attention_block`), leaving native-eligible blocks
/// intact for the `fused_attn_block` kernel. No rewrite when there is no FAB,
/// or when every FAB is already native.
fn lower_fab_for_metal(g: Graph) -> Graph {
    let has_fab = g
        .nodes()
        .iter()
        .any(|n| matches!(n.op, Op::FusedAttentionBlock { .. }));
    if !has_fab {
        return g;
    }
    let all_native = g
        .nodes()
        .iter()
        .all(|n| !matches!(n.op, Op::FusedAttentionBlock { .. }) || fab_is_native(n));
    if all_native {
        return g;
    }
    let mut out = Graph::new(g.name.clone());
    let mut id_map: std::collections::HashMap<NodeId, NodeId> = std::collections::HashMap::new();
    let nodes: Vec<rlx_ir::Node> = g.nodes().to_vec();
    for node in &nodes {
        let new_inputs: Vec<NodeId> = node.inputs.iter().map(|i| id_map[i]).collect();
        let new_id = if let Op::FusedAttentionBlock {
            num_heads,
            head_dim,
            has_bias,
            has_rope,
        } = &node.op
        {
            if fab_is_native(node) {
                out.add_node(node.op.clone(), new_inputs, node.shape.clone())
            } else {
                rlx_opt::unfuse::expand_attention_block(
                    &mut out,
                    &new_inputs,
                    *num_heads,
                    *head_dim,
                    *has_bias,
                    *has_rope,
                )
            }
        } else {
            out.add_node(node.op.clone(), new_inputs, node.shape.clone())
        };
        id_map.insert(node.id, new_id);
    }
    out.set_outputs(g.outputs.iter().map(|i| id_map[i]).collect());
    out
}

/// Per native-FAB-node `(qkv, attn)` BYTE offsets *relative to the FAB scratch
/// base*, plus the total scratch size in bytes. `qkv = [B,S,3*inner]` and
/// `attn = [B,S,inner]` (both f32), each block 128-byte aligned.
fn fab_scratch_layout(graph: &Graph) -> (usize, Vec<(NodeId, usize, usize)>) {
    let mut rel: Vec<(NodeId, usize, usize)> = Vec::new();
    let mut cur: usize = 0;
    for node in graph.nodes() {
        if !fab_is_native(node) {
            continue;
        }
        if let Op::FusedAttentionBlock {
            num_heads,
            head_dim,
            ..
        } = &node.op
        {
            let dims = node.shape.dims();
            let b = dims[0].unwrap_static();
            let s = dims[1].unwrap_static();
            let inner = num_heads * head_dim;
            cur = (cur + 127) & !127;
            let qkv_off = cur;
            cur += b * s * 3 * inner * 4;
            cur = (cur + 127) & !127;
            let attn_off = cur;
            cur += b * s * inner * 4;
            rel.push((node.id, qkv_off, attn_off));
        }
    }
    (cur, rel)
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

fn encode_cumsum(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    src: usize,
    dst: usize,
    rows: u32,
    cols: u32,
    exclusive: bool,
) {
    enc.set_compute_pipeline_state(&k.cumsum_fwd);
    enc.set_buffer(0, Some(buffer), src as u64);
    enc.set_buffer(1, Some(buffer), dst as u64);
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

/// Maps [`QuantScheme`] to the shared GPU `dequant_gguf` MSL kernel scheme id (0–23).
///
/// See [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md) for the full table.
pub(crate) fn gguf_scheme_id(scheme: rlx_ir::quant::QuantScheme) -> u32 {
    use rlx_ir::quant::QuantScheme;
    match scheme {
        QuantScheme::GgufQ4K => 0,
        QuantScheme::GgufQ5K => 1,
        QuantScheme::GgufQ6K => 2,
        QuantScheme::GgufQ8K => 3,
        QuantScheme::GgufQ2K => 4,
        QuantScheme::GgufQ3K => 5,
        QuantScheme::GgufIQ4NL => 6,
        QuantScheme::GgufIQ4XS => 7,
        QuantScheme::GgufTQ1_0 => 8,
        QuantScheme::GgufTQ2_0 => 9,
        QuantScheme::GgufMXFP4 => 10,
        QuantScheme::GgufNVFP4 => 11,
        QuantScheme::GgufIQ2XXS => 12,
        QuantScheme::GgufIQ2XS => 13,
        QuantScheme::GgufIQ2S => 14,
        QuantScheme::GgufIQ3XXS => 15,
        QuantScheme::GgufIQ3S => 16,
        QuantScheme::GgufIQ1S => 17,
        QuantScheme::GgufIQ1M => 18,
        QuantScheme::GgufQ4_0 => 19,
        QuantScheme::GgufQ8_0 => 20,
        QuantScheme::GgufQ4_1 => 21,
        QuantScheme::GgufQ5_0 => 22,
        QuantScheme::GgufQ5_1 => 23,
        other => panic!("gguf_scheme_id: unsupported {other:?} — use CPU dequant path"),
    }
}

/// Returns `true` when this scheme has a native on-device dequant kernel
/// in the `dequant_gguf` MSL shader, `false` when callers should route
/// through the CPU dequant path (`rlx_gguf::dequant_*`) instead.
///
/// Fused GEMV (`q4k_mv_f32`, `q4_0_mv_f32`, `q8_0_mv_f32`) is separate — see
/// [docs/gguf-backend-paths.md](../../../docs/gguf-backend-paths.md).
pub fn has_metal_dequant_kernel(scheme: rlx_ir::quant::QuantScheme) -> bool {
    use rlx_ir::quant::QuantScheme;
    matches!(
        scheme,
        QuantScheme::GgufQ4K
            | QuantScheme::GgufQ5K
            | QuantScheme::GgufQ6K
            | QuantScheme::GgufQ8K
            | QuantScheme::GgufQ2K
            | QuantScheme::GgufQ3K
            | QuantScheme::GgufIQ4NL
            | QuantScheme::GgufIQ4XS
            | QuantScheme::GgufTQ1_0
            | QuantScheme::GgufTQ2_0
            | QuantScheme::GgufMXFP4
            | QuantScheme::GgufNVFP4
            | QuantScheme::GgufIQ2XXS
            | QuantScheme::GgufIQ2XS
            | QuantScheme::GgufIQ2S
            | QuantScheme::GgufIQ3XXS
            | QuantScheme::GgufIQ3S
            | QuantScheme::GgufIQ1S
            | QuantScheme::GgufIQ1M
            | QuantScheme::GgufQ4_0
            | QuantScheme::GgufQ4_1
            | QuantScheme::GgufQ5_0
            | QuantScheme::GgufQ5_1
            | QuantScheme::GgufQ8_0
    )
}

/// Simdgroup-cooperative Q4_K_M GEMV. 32 threads share x reads and
/// produce 8 output columns each via `simd_sum`. Constraint:
/// `n_dim % 8 == 0` (caller enforces). Adapted from llama.cpp's
/// `kernel_mul_mv_q4_K_f32_impl`.
pub(crate) fn encode_q4k_mv_f32_sg(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4k_mv_f32_sg);
    enc.set_buffer(0, Some(buffer), 0);
    // u64 byte offsets (task #50) — 12B Q4 activations sit past 4 GB.
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    // 4 simdgroups per threadgroup → 32 outputs per threadgroup.
    // Threadgroup size = 128, grid sized to cover all output groups.
    const NSG: u64 = 4;
    let n_output_groups = (n_dim.div_ceil(8)) as u64;
    let n_threadgroups = n_output_groups.div_ceil(NSG);
    let grid = metal::MTLSize {
        width: n_threadgroups * NSG * 32,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: NSG * 32,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused Q4_K / Q6_K GEMM (`m > 1`, prefill): `C[m,n] = A[m,k] @ dequant(w)^T`
/// straight from the packed weight — no f32 scratch, no MPS sgemm. Grid is
/// `(n columns) × ceil(m / TM)` row-tiles; threadgroup = up to 64 columns.
/// Caller must guarantee `k_dim % 256 == 0`. Used for `m > 1` GgufQ4K/Q6K.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_qk_mm_f32(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    m_dim: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let m_u = m_dim as u32;
    enc.set_bytes(4, 4, &m_u as *const u32 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    // TM must match Q4K_MM_TM / Q6K_MM_TM in dequant_gguf.msl.
    const TM: u64 = 8;
    let row_tiles = (m_dim as u64).div_ceil(TM);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: row_tiles,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: (n_dim as u64).min(64),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused Q4_K_M GEMV: `dst[n] = sum_k x[k] * dequant(w[n,k])` in one
/// pass, skipping the f32 dequant scratch the dequant + MPS sgemm path
/// would write. Caller must guarantee `k_dim % 256 == 0`. Decode-only
/// (`m == 1`) — m > 1 still falls through to the legacy GPU path.
pub(crate) fn encode_q4k_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4k_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused decode MLP gate+up packed GEMV dispatch (`m == 1`).
#[allow(clippy::too_many_arguments)]
fn encode_fused_mlp_gate_up_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    gate_w: usize,
    up_w: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let g_u = gate_w as u64;
    enc.set_bytes(2, 8, &g_u as *const u64 as *const _);
    let u_u = up_w as u64;
    enc.set_bytes(3, 8, &u_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(4, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

/// Fused decode MLP gate+up packed GEMV with SwiGLU epilogue (`m == 1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fused_mlp_gate_up_swiglu(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scheme: rlx_ir::quant::QuantScheme,
    x: usize,
    gate_w: usize,
    up_w: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    use rlx_ir::quant::QuantScheme;
    let pipeline = match scheme {
        QuantScheme::GgufQ4K => &k.q4k_swiglu_mv_f32,
        QuantScheme::GgufQ5_0 => &k.q5_0_swiglu_mv_f32,
        other => panic!("encode_fused_mlp_gate_up_swiglu: unsupported {other:?}"),
    };
    encode_fused_mlp_gate_up_mv_f32(enc, pipeline, buffer, x, gate_w, up_w, dst, k_dim, n_dim);
}

/// Fused decode MLP gate+up packed GEMV with GELU-approx epilogue (`m == 1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_fused_mlp_gate_up_gelu(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    scheme: rlx_ir::quant::QuantScheme,
    x: usize,
    gate_w: usize,
    up_w: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    use rlx_ir::quant::QuantScheme;
    let pipeline = match scheme {
        QuantScheme::GgufQ4K => &k.q4k_gelu_mv_f32,
        QuantScheme::GgufQ5_0 => &k.q5_0_gelu_mv_f32,
        other => panic!("encode_fused_mlp_gate_up_gelu: unsupported {other:?}"),
    };
    encode_fused_mlp_gate_up_mv_f32(enc, pipeline, buffer, x, gate_w, up_w, dst, k_dim, n_dim);
}

/// Fused decode MLP down-projection GEMV + residual add (`m == 1`).
/// `dst[j] = res[j] + down(x)[j]`. `pipeline` selects Q4_K / Q5_0 / Q6_K.
/// one thread per output column. Caller guarantees `k_dim % 256 == 0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_q4k_mv_residual_f32(
    enc: &metal::ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    res: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let r_u = res as u64;
    enc.set_bytes(4, 8, &r_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(5, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(6, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_q4_0_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4_0_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_q4_1_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q4_1_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_q8_0_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.q8_0_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq4_nl_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq4_nl_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq2_xxs_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq2_xxs_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq2_xs_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq2_xs_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq3_xxs_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq3_xxs_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq2_s_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq2_s_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq3_s_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq3_s_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq1_s_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq1_s_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

pub(crate) fn encode_iq1_m_mv_f32(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_q: usize,
    dst: usize,
    k_dim: usize,
    n_dim: usize,
) {
    enc.set_compute_pipeline_state(&k.iq1_m_mv_f32);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = x as u64;
    enc.set_bytes(1, 8, &x_u as *const u64 as *const _);
    let w_u = w_q as u64;
    enc.set_bytes(2, 8, &w_u as *const u64 as *const _);
    let d_u = dst as u64;
    enc.set_bytes(3, 8, &d_u as *const u64 as *const _);
    let k_u = k_dim as u32;
    enc.set_bytes(4, 4, &k_u as *const u32 as *const _);
    let n_u = n_dim as u32;
    enc.set_bytes(5, 4, &n_u as *const u32 as *const _);
    enc.set_buffer(6, Some(k.iq_grid_buffer()), 0);
    let grid = metal::MTLSize {
        width: n_dim as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: 256.min(n_dim) as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
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
    // 12B Q4 GGUF activations sit at arena offsets > 4 GB. u32 cast on a
    // ~14 GB byte offset silently truncates the high bits and the dequant
    // kernel reads garbage from a wrap-around pointer — producing Q4K output
    // with values up to 1.2e11 and sparse NaN (task #50). Pass offsets as u64.
    let dst_f32 = (dst / 4) as u64;
    let w_u = w_q as u64;
    enc.set_compute_pipeline_state(&k.dequant_gguf);
    enc.set_buffer(0, Some(buffer), 0);
    enc.set_bytes(1, 8, &w_u as *const u64 as *const _);
    enc.set_bytes(2, 8, &dst_f32 as *const u64 as *const _);
    enc.set_bytes(3, 4, &scheme_id as *const u32 as *const _);
    let nb = num_blocks as u32;
    enc.set_bytes(4, 4, &nb as *const u32 as *const _);
    // buffer(5): IQ grid LUTs (KMASK | KSIGNS | KGRID_IQ2XXS | ... |
    // KGRID_IQ1S). Schemes 0..=11 ignore it. See `crate::kernels::iq_grid_buffer`.
    let lut = k.iq_grid_buffer();
    enc.set_buffer(5, Some(lut), 0);
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
    queue: &metal::CommandQueueRef,
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

        // Per expert: dequant the slab into `dequant_off` on a dedicated MSL
        // compute encoder, END that encoder, then MPS sgemm. The compute
        // encoder MUST be ended before the MPS call — MPS opens its own
        // encoder internally, and two live encoders on one command buffer is a
        // hard Metal abort (`A command encoder is already encoding...`).
        // Encoders execute serially in submission order, so expert e's sgemm
        // reads `dequant_off` before expert e+1's dequant overwrites it.
        let cmd_buf = queue.new_command_buffer();
        for e in 0..num_experts {
            let count = offsets[e + 1] - offsets[e];
            if count == 0 {
                continue;
            }
            let enc =
                cmd_buf.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Serial);
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
            enc.end_encoding();
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
        // Sgemm results must land before the host-side unpermute reads them.
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

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

/// Native MSL selective scan (f32, `state_size <= SSM_MAX_N = 128`). One
/// thread per `(batch, channel)`; each owns a private state vector and
/// scans sequentially over the seq axis. Matches `execute_selective_scan_f32`.
#[allow(clippy::too_many_arguments)]
fn encode_selective_scan(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
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
) {
    let f32_idx = |byte_off: usize| -> u32 { (byte_off / 4) as u32 };
    let p = &k.selective_scan;
    enc.set_compute_pipeline_state(p);
    enc.set_buffer(0, Some(buffer), 0);
    let offs = [
        f32_idx(x),
        f32_idx(delta),
        f32_idx(a),
        f32_idx(b),
        f32_idx(c),
        f32_idx(dst),
    ];
    for (i, off) in offs.iter().enumerate() {
        enc.set_bytes((i + 1) as u64, 4, off as *const u32 as *const _);
    }
    let dims = [batch, seq, hidden, state_size];
    enc.set_bytes(7, 16, dims.as_ptr() as *const _);
    let threads = (batch * hidden) as u64;
    let tg_w = p.thread_execution_width().min(threads.max(1));
    enc.dispatch_threads(
        metal::MTLSize {
            width: threads,
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

/// Native MSL forward LSTM (f32, `hidden <= LSTM_MAX_H = 1024`). One
/// threadgroup per batch item, `hidden` threads each.
#[allow(clippy::too_many_arguments)]
fn encode_lstm(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    bias: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
) {
    let f32_idx = |byte_off: usize| -> u32 { (byte_off / 4) as u32 };
    enc.set_compute_pipeline_state(&k.lstm);
    enc.set_buffer(0, Some(buffer), 0);
    let x_u = f32_idx(x);
    let wih_u = f32_idx(w_ih);
    let whh_u = f32_idx(w_hh);
    let bias_u = f32_idx(bias);
    let dst_u = f32_idx(dst);
    enc.set_bytes(1, 4, &x_u as *const u32 as *const _);
    enc.set_bytes(2, 4, &wih_u as *const u32 as *const _);
    enc.set_bytes(3, 4, &whh_u as *const u32 as *const _);
    enc.set_bytes(4, 4, &bias_u as *const u32 as *const _);
    enc.set_bytes(5, 4, &dst_u as *const u32 as *const _);
    let dims = [batch, seq, input_size, hidden];
    enc.set_bytes(6, 16, dims.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: hidden as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Native MSL GRU (f32, single-layer/unidir/no-carry, `hidden ≤ 1024`).
#[allow(clippy::too_many_arguments)]
fn encode_gru(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    b_ih: usize,
    b_hh: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
) {
    let f32_idx = |o: usize| -> u32 { (o / 4) as u32 };
    enc.set_compute_pipeline_state(&k.gru);
    enc.set_buffer(0, Some(buffer), 0);
    let offs = [
        f32_idx(x),
        f32_idx(w_ih),
        f32_idx(w_hh),
        f32_idx(b_ih),
        f32_idx(b_hh),
        f32_idx(dst),
    ];
    for (i, o) in offs.iter().enumerate() {
        enc.set_bytes((i + 1) as u64, 4, o as *const u32 as *const _);
    }
    let dims = [batch, seq, input_size, hidden];
    enc.set_bytes(7, 16, dims.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: hidden as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Native MSL Elman RNN (f32, single-layer/unidir/no-carry, `hidden ≤ 1024`).
#[allow(clippy::too_many_arguments)]
fn encode_rnn(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    w_ih: usize,
    w_hh: usize,
    bias: usize,
    dst: usize,
    batch: u32,
    seq: u32,
    input_size: u32,
    hidden: u32,
    relu: bool,
) {
    let f32_idx = |o: usize| -> u32 { (o / 4) as u32 };
    enc.set_compute_pipeline_state(&k.rnn);
    enc.set_buffer(0, Some(buffer), 0);
    let offs = [
        f32_idx(x),
        f32_idx(w_ih),
        f32_idx(w_hh),
        f32_idx(bias),
        f32_idx(dst),
    ];
    for (i, o) in offs.iter().enumerate() {
        enc.set_bytes((i + 1) as u64, 4, o as *const u32 as *const _);
    }
    let dims = [batch, seq, input_size, hidden];
    enc.set_bytes(6, 16, dims.as_ptr() as *const _);
    let relu_u: u32 = if relu { 1 } else { 0 };
    enc.set_bytes(7, 4, &relu_u as *const u32 as *const _);
    let grid = metal::MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let tg = metal::MTLSize {
        width: hidden as u64,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(grid, tg);
}

/// Native MSL Mamba-2 SSD scan (f32, `state_size ≤ 128`). One thread per
/// `(batch, head, head_dim_pos)`.
#[allow(clippy::too_many_arguments)]
fn encode_mamba2(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
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
) {
    let f32_idx = |o: usize| -> u32 { (o / 4) as u32 };
    let p = &k.mamba2;
    enc.set_compute_pipeline_state(p);
    enc.set_buffer(0, Some(buffer), 0);
    let offs = [
        f32_idx(x),
        f32_idx(dt),
        f32_idx(a),
        f32_idx(b),
        f32_idx(c),
        f32_idx(dst),
    ];
    for (i, o) in offs.iter().enumerate() {
        enc.set_bytes((i + 1) as u64, 4, o as *const u32 as *const _);
    }
    // dims.w packs head_dim (high 16) | state_size (low 16).
    let packed = (head_dim << 16) | (state_size & 0xffff);
    let dims = [batch, seq, heads, packed];
    enc.set_bytes(7, 16, dims.as_ptr() as *const _);
    let threads = (batch * heads * head_dim) as u64;
    let tg_w = p.thread_execution_width().min(threads.max(1));
    enc.dispatch_threads(
        metal::MTLSize {
            width: threads,
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
            // n==1 im2col path needs n_dim*k_dim f32; the batch-parallel
            // two-pass path needs N*C_out*c_in_per_g*kh*kw f32 partials.
            // Size the shared scratch for whichever is larger.
            let n_batch = x_shape.dim(0).unwrap_static();
            let c_out = dy_shape.dim(1).unwrap_static();
            let two_pass = n_batch * c_out * c_in_per_g * kh * kw;
            let need = (n_dim * k_dim).max(two_pass);
            max = max.max(need * std::mem::size_of::<f32>());
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

#[allow(clippy::too_many_arguments)]
fn encode_maxpool2d_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dx: usize,
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
) {
    let p0: [u32; 4] = [n, c, h, w];
    let p1: [u32; 4] = [h_out, w_out, kh, kw];
    let p2: [u32; 4] = [sh, sw, ph, pw];
    enc.set_compute_pipeline_state(&k.maxpool2d_backward);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 16, p0.as_ptr() as *const _);
    enc.set_bytes(4, 16, p1.as_ptr() as *const _);
    enc.set_bytes(5, 16, p2.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w as u64,
        height: h as u64,
        depth: (n * c) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w as u64),
        height: 8.min(h as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
fn encode_conv2d_backward_input(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dy: usize,
    w: usize,
    dx: usize,
    n: u32,
    c_in: u32,
    h: u32,
    w_in: u32,
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
    let a: [u32; 4] = [n, c_in, h, w_in];
    let b: [u32; 4] = [c_out, h_out, w_out, kh];
    let cc: [u32; 4] = [kw, sh, sw, ph];
    let d: [u32; 4] = [pw, dh, dw, groups];
    enc.set_compute_pipeline_state(&k.conv2d_backward_input);
    enc.set_buffer(0, Some(buffer), dy as u64);
    enc.set_buffer(1, Some(buffer), w as u64);
    enc.set_buffer(2, Some(buffer), dx as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, d.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: w_in as u64,
        height: h as u64,
        depth: (n * c_in) as u64,
    };
    let tg = metal::MTLSize {
        width: 8.min(w_in as u64),
        height: 8.min(h as u64),
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

#[allow(clippy::too_many_arguments)]
fn encode_conv2d_backward_weight(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    x: usize,
    dy: usize,
    dw: usize,
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
    dw_dil: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, h, w];
    let b: [u32; 4] = [c_out, h_out, w_out, kh];
    let cc: [u32; 4] = [kw, sh, sw, ph];
    let d: [u32; 4] = [pw, dh, dw_dil, groups];
    let c_in_per_g = c_in / groups;
    enc.set_compute_pipeline_state(&k.conv2d_backward_weight);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), dw as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, d.as_ptr() as *const _);
    let grid = metal::MTLSize {
        width: kw as u64,
        height: kh as u64,
        depth: (c_out * c_in_per_g) as u64,
    };
    let tg = metal::MTLSize {
        width: kw as u64,
        height: kh as u64,
        depth: 1,
    };
    enc.dispatch_threads(grid, tg);
}

// Two-pass batch-parallel conv2d weight-grad. `part_off` is the conv-bwd
// scratch slot, sized (by conv_bwd_scratch_bytes) to hold N*C_out*c_in_per_g*
// kh*kw f32 partials. Pass 1 fills it (one thread per per-sample weight elem),
// pass 2 reduces over N. Both run in the same serial encoder, so pass 2 sees
// pass 1's writes with no explicit barrier.
#[allow(clippy::too_many_arguments)]
fn encode_conv2d_backward_weight_2pass(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    part_off: usize,
    x: usize,
    dy: usize,
    dw: usize,
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
    dw_dil: u32,
    groups: u32,
) {
    let a: [u32; 4] = [n, c_in, h, w];
    let b: [u32; 4] = [c_out, h_out, w_out, kh];
    let cc: [u32; 4] = [kw, sh, sw, ph];
    let d: [u32; 4] = [pw, dh, dw_dil, groups];
    let c_in_per_g = c_in / groups;
    let wsz = c_in_per_g * kh * kw; // per-(n,co) slab
    let wslab = c_out * wsz; // per-sample slab

    // Pass 1: per-sample partials → scratch.
    enc.set_compute_pipeline_state(&k.conv2d_backward_weight_partial);
    enc.set_buffer(0, Some(buffer), x as u64);
    enc.set_buffer(1, Some(buffer), dy as u64);
    enc.set_buffer(2, Some(buffer), part_off as u64);
    enc.set_bytes(3, 16, a.as_ptr() as *const _);
    enc.set_bytes(4, 16, b.as_ptr() as *const _);
    enc.set_bytes(5, 16, cc.as_ptr() as *const _);
    enc.set_bytes(6, 16, d.as_ptr() as *const _);
    let grid1 = metal::MTLSize {
        width: wsz as u64,
        height: c_out as u64,
        depth: n as u64,
    };
    let tgw = 8.min(wsz as u64).max(1);
    let tg1 = metal::MTLSize {
        width: tgw,
        height: 8.min(c_out as u64).max(1),
        depth: 1,
    };
    enc.dispatch_threads(grid1, tg1);

    // Pass 2: reduce partials over the batch → dw.
    let dims: [u32; 2] = [n, wslab];
    enc.set_compute_pipeline_state(&k.conv2d_backward_weight_reduce);
    enc.set_buffer(0, Some(buffer), part_off as u64);
    enc.set_buffer(1, Some(buffer), dw as u64);
    enc.set_bytes(2, 8, dims.as_ptr() as *const _);
    let grid2 = metal::MTLSize {
        width: wslab as u64,
        height: 1,
        depth: 1,
    };
    let tg2 = metal::MTLSize {
        width: 64.min(wslab as u64).max(1),
        height: 1,
        depth: 1,
    };
    enc.dispatch_threads(grid2, tg2);
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
    // Apple simdgroups are 32 lanes. The previous 8×8 threadgroup left
    // 75% of each simdgroup idle when the gather had a single row
    // (num_idx == 1, the embedding-lookup hot path). Pick the largest
    // axis as the threadgroup-x dimension and pack to 32 — keeps the
    // 2-D case fast for general gathers while making the embed lookup
    // ~4× more parallel per simdgroup.
    let tg_x = 32.min(trailing as u64).max(1);
    let tg_y = (32 / tg_x).clamp(1, num_idx as u64).max(1);
    let tg = metal::MTLSize {
        width: tg_x,
        height: tg_y,
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

/// CPU unified-memory fallbacks for copy / elem / activation (debug only).
/// Default is GPU arena-base + u64 byte offsets (Task #50, >4 GiB arenas).
fn metal_host_fallback_enabled() -> bool {
    metal_host_slices_enabled() || rlx_ir::env::flag("RLX_METAL_HOST_FALLBACK")
}

/// Task #50: activations past 4 GiB need arena-base + u64 byte offsets.
const ARENA_LARGE_OFF: usize = 1usize << 32;

#[inline]
fn arena_off_large(off: usize) -> bool {
    off >= ARENA_LARGE_OFF
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

#[allow(clippy::too_many_arguments)]
fn encode_fma(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    a: usize,
    b: usize,
    c: usize,
    dst: usize,
    len: u32,
) {
    enc.set_compute_pipeline_state(&k.elem_fma);
    enc.set_buffer(0, Some(buffer), a as u64);
    enc.set_buffer(1, Some(buffer), b as u64);
    enc.set_buffer(2, Some(buffer), c as u64);
    enc.set_buffer(3, Some(buffer), dst as u64);
    enc.set_bytes(4, 4, &len as *const u32 as *const _);
    let tg_w = k.elem_fma.thread_execution_width().min(len as u64);
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

/// Fused dense softmax cross-entropy: one threadgroup per row, three
/// threadgroup reductions (row max, Σexp, Σtargets·logits). `cols` is
/// the class count C. Threadgroup width is the largest pow2 ≤ cols,
/// capped at 256 (the kernel's reduction buffer). f32 only.
#[allow(clippy::too_many_arguments)]
fn encode_softmax_cross_entropy_dense(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    logits: usize,
    targets: usize,
    dst: usize,
    rows: u32,
    cols: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&k.softmax_cross_entropy_dense);
    enc.set_buffer(0, Some(buffer), logits as u64);
    enc.set_buffer(1, Some(buffer), targets as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &cols as *const u32 as *const _,
    );
    // 1D dispatch: pack rows along width so threadgroup_position_in_grid.x
    // is the row index (same gotcha as encode_softmax / encode_layer_norm).
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

#[allow(clippy::too_many_arguments)]
fn encode_softmax_cross_entropy_with_logits(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    logits: usize,
    labels: usize,
    dst: usize,
    rows: u32,
    cols: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&k.softmax_cross_entropy_with_logits);
    enc.set_buffer(0, Some(buffer), logits as u64);
    enc.set_buffer(1, Some(buffer), labels as u64);
    enc.set_buffer(2, Some(buffer), dst as u64);
    enc.set_bytes(3, 4, &cols as *const u32 as *const _);
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

#[allow(clippy::too_many_arguments)]
fn encode_softmax_cross_entropy_backward(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    logits: usize,
    labels: usize,
    d_loss: usize,
    dlogits: usize,
    rows: u32,
    cols: u32,
) {
    let mut tg_w: u64 = 1;
    while tg_w * 2 <= cols as u64 && tg_w * 2 <= 256 {
        tg_w *= 2;
    }
    enc.set_compute_pipeline_state(&k.softmax_cross_entropy_backward);
    enc.set_buffer(0, Some(buffer), logits as u64);
    enc.set_buffer(1, Some(buffer), labels as u64);
    enc.set_buffer(2, Some(buffer), d_loss as u64);
    enc.set_buffer(3, Some(buffer), dlogits as u64);
    enc.set_bytes(4, 4, &cols as *const u32 as *const _);
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
    // u64 because for ≥4 GB models the source byte offset exceeds u32 and
    // u32-truncation wrap-around made `repeat_kv` write to the wrong slot,
    // leaving K_rep / V_rep as zeros and SDPA output as zero (task #50).
    src: u64,
    dst_col: u32,
    len: u32,
}

/// Dispatch a concat-along-last-axis. Uses one multi-segment kernel when possible.
/// Mid-axis concat (inner > 1) encoded entirely into the live command buffer
/// — one 1D dispatch per input segment, NO commit/wait. Replaces the
/// per-concat `commit + wait_until_completed` host-copy fallback that
/// serialized a decode step into one GPU submission per concat (the dominant
/// Metal decode cost on KV caches). Offsets are element offsets within the
/// f32/f16 arena; the kernel takes ulong byte offsets so it is correct on
/// >4 GiB arenas (task #50). `dst`/`inputs` offsets are byte offsets into the
/// arena (as stored in the thunk).
fn encode_concat_midaxis(
    enc: &metal::ComputeCommandEncoderRef,
    k: &crate::kernels::Kernels,
    buffer: &metal::Buffer,
    dst: usize,
    outer: u32,
    dst_axis: u32,
    inner: u32,
    dt: crate::thunk::HalfFlag,
    inputs: &[(usize, u32)],
) {
    use crate::thunk::HalfFlag;
    // `dst`/`src_off` are byte offsets into the arena (the kernel adds them to
    // a char* base, then indexes elements), so no element-size scaling here.
    let pipeline = match dt {
        HalfFlag::F32 => &k.concat_midaxis_seg,
        HalfFlag::F16 => &k.concat_midaxis_seg_h,
    };
    let inner_e = inner as u64;
    let mut axis_off: u32 = 0;
    for &(src_off, src_axis) in inputs {
        let total = outer as u64 * src_axis as u64 * inner_e;
        if total == 0 {
            axis_off += src_axis;
            continue;
        }
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(buffer), 0);
        let dst_byte = dst as u64; // already a byte offset
        let src_byte = src_off as u64;
        enc.set_bytes(1, 8, &dst_byte as *const u64 as *const _);
        enc.set_bytes(2, 8, &src_byte as *const u64 as *const _);
        enc.set_bytes(3, 4, &outer as *const u32 as *const _);
        enc.set_bytes(4, 4, &dst_axis as *const u32 as *const _);
        enc.set_bytes(5, 4, &src_axis as *const u32 as *const _);
        enc.set_bytes(6, 4, &inner as *const u32 as *const _);
        enc.set_bytes(7, 4, &axis_off as *const u32 as *const _);
        let tg = 256u64.min(total);
        let grid = metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let tgs = metal::MTLSize {
            width: tg.max(1),
            height: 1,
            depth: 1,
        };
        enc.dispatch_threads(grid, tgs);
        axis_off += src_axis;
    }
}

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
    // Historically `concat_lastax_multi{,4}` was reported to mis-copy beyond 8
    // segments on Apple GPUs and we fell through to the per-segment kernels for
    // GQA `repeat_kv` (16 head slices). Per-segment fallback uses `set_buffer`
    // at large arena offsets which silently drops writes on ≥4 GB models
    // (task #50). The multi kernel takes byte offsets from `ConcatSeg` (now
    // u64) and binds `arena` at offset 0 — works at any offset.
    if inputs.len() >= 2
        && inputs.len() <= NARROW_BATCH_MAX
        && matches!(dt, HalfFlag::F32)
        && metal_concat_multi_enabled()
    {
        let mut cum = 0u32;
        let segs: Vec<ConcatSegGpu> = inputs
            .iter()
            .map(|&(src_off, src_axis)| {
                let seg = ConcatSegGpu {
                    src: src_off as u64,
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
            let dst_u64 = dst as u64;
            enc.set_bytes(1, 8, &dst_u64 as *const u64 as *const _);
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
            // Task #50: total threads per threadgroup must be ≤ 1024 on
            // Apple Silicon. Width×height×depth previously was 64×4×num_seg —
            // exceeds the cap once num_seg≥5 (GQA repeat_kv concats 16 head
            // slices). Metal silently fails the dispatch when over the cap,
            // leaving the destination buffer zero, which manifested as the
            // long-standing K_rep / V_rep zero bug on Gemma 4 12B SWA layers.
            let tg_depth = (1024u64 / (64 * 4)).min(num_seg as u64).max(1);
            let tg = metal::MTLSize {
                width: 64.min(max_len4 as u64),
                height: 4.min(outer as u64),
                depth: tg_depth,
            };
            enc.dispatch_threads(grid, tg);
            return;
        }
        enc.set_compute_pipeline_state(&k.concat_lastax_multi);
        enc.set_buffer(0, Some(buffer), 0);
        let dst_u64 = dst as u64;
        enc.set_bytes(1, 8, &dst_u64 as *const u64 as *const _);
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
        // Task #50: cap total threads per threadgroup at 1024.
        let tg_depth = (1024u64 / (32 * 8)).min(num_seg as u64).max(1);
        let tg = metal::MTLSize {
            width: 32.min(max_len as u64),
            height: 8.min(outer as u64),
            depth: tg_depth,
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
            // Large set_buffer offsets silently drop kernel writes on
            // M-series at offsets ≥ ~4 GB (task #50). Bind to arena base
            // and pass byte offsets as ulong constants in buffers 6/7.
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis4 as *const u32 as *const _);
            enc.set_bytes(4, 4, &dst_axis4 as *const u32 as *const _);
            enc.set_bytes(5, 4, &dst_col4 as *const u32 as *const _);
            let src_off_u64 = src_off as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_off_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
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
            // Task #50: same arena-base + ulong-offset workaround.
            enc.set_buffer(0, Some(buffer), 0);
            enc.set_buffer(1, Some(buffer), 0);
            enc.set_bytes(2, 4, &outer as *const u32 as *const _);
            enc.set_bytes(3, 4, &src_axis as *const u32 as *const _);
            enc.set_bytes(4, 4, &dst_axis as *const u32 as *const _);
            enc.set_bytes(5, 4, &cum as *const u32 as *const _);
            let src_off_u64 = src_off as u64;
            let dst_u64 = dst as u64;
            enc.set_bytes(6, 8, &src_off_u64 as *const u64 as *const _);
            enc.set_bytes(7, 8, &dst_u64 as *const u64 as *const _);
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
