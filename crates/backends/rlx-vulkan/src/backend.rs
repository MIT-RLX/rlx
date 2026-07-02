// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// SPDX-License-Identifier: GPL-3.0-only

//! `VulkanExecutable` — compile an IR graph into a flat schedule of compute
//! dispatches over a single f32 arena buffer, then execute it.
//!
//! Design (mirrors rlx-cuda / rlx-wgpu): every tensor is an f32 slot in one
//! arena `VkBuffer`; each schedule [`Step`] is one compute pipeline + push
//! constants + a workgroup count. A single descriptor set binds the whole
//! arena; per-op offsets/dims ride in push constants. Between dispatches we
//! insert a global shader-memory barrier (every kernel reads/writes the shared
//! arena), submit once per `run`, and read outputs back from the host-visible
//! mapping.
//!
//! Op coverage is the transformer-inference hot path: elementwise (binary /
//! unary / compare / where), matmul, last-axis reduce, softmax, RMS/Layer
//! norm, RoPE, attention, gather, cumsum, and the shape ops (narrow / concat /
//! expand / transpose) via one strided-copy kernel. Fused ops, DotGeneral,
//! Fma, non-last-axis reduce, GroupNorm, etc. are decomposed to these
//! primitives by `legalize_or_rewrite_for_backend`. Anything left unsupported
//! (Conv, Pool, quantized matmul, SSM, …) fails loudly with a "pin to CPU"
//! diagnostic — like rlx-wgpu's stance for ops it can't lower.

use crate::buffer::Arena;
use crate::device::vulkan_device;
use crate::kernels::kernels;
use ash::vk;
use rlx_compile::memory::{BufferSlot, MemoryPlan};
use rlx_ir::op::{Activation, BinaryOp, CmpOp, MaskKind, ReduceOp, RopeStyle};
use rlx_ir::{DType, Graph, NodeId, Op, RngOptions};
use std::collections::{HashMap, HashSet};

/// OpKinds this backend lowers natively. Everything else is either decomposed
/// into this set by the rewrite pass or rejected at legalize time.
pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = {
    use rlx_ir::OpKind::*;
    &[
        Input,
        Param,
        Constant,
        Cast,
        StopGradient,
        Reshape, // structural / alias
        Binary,
        Compare,
        Where,
        Activation, // elementwise
        MatMul,
        Reduce,
        Softmax, // contraction / reduction
        LayerNorm,
        RmsNorm,
        LayerNorm2d, // normalization
        Rope,
        Attention, // transformer
        // Claimed so the block is a first-class op; `compile_rng` runs
        // `unfuse_attention_block` to lower it to the chain above (matmul
        // → narrow → rope → attention → matmul) before legalization.
        FusedAttentionBlock,
        Transpose,
        Narrow,
        Concat,
        Expand,
        Gather,
        Cumsum,
        Reverse, // shape / indexing
        ArgMax,
        ArgMin,
        Pool,
        ResizeNearest2x,
        Conv,          // reductions / vision
        GroupedMatMul, // MoE
        SelectiveScan, // SSM / Mamba
        Im2Col,
        ScatterAdd,
        TopK, // vision / indexing / generation
        // Host-fallback (run on the CPU reference against the mapped arena —
        // sequential / specialized families with no native SPIR-V kernel yet):
        Lstm,
        Gru,
        Rnn,
        Mamba2,
        GatedDeltaNet,
        // General Op::Scan (arbitrary-body recurrence, e.g. IIR biquad) — the
        // host fallback builds a one-op CPU graph and runs rlx-cpu's native
        // Scan against the mapped arena.
        Scan,
        ConvTranspose2d,
        Fft,
        DequantMatMul,
        DequantGroupedMatMul,
        DequantMoEWeights, // GGUF quant
        RngNormal,
        RngUniform,
        Sample, // RNG / generation
    ]
};

/// Ops with no native kernel that route to the CPU host-fallback path.
///
/// `DequantMatMul` is handled by its own scheduler arm: GGUF Q4_K/Q6_K decode
/// GEMV (`m == 1`) runs natively via the `dequant_matmul` shader; every other
/// scheme/shape still falls back to the CPU reference from that arm.
fn is_host_fallback(op: &Op) -> bool {
    matches!(
        op,
        Op::Lstm { .. }
            | Op::Gru { .. }
            | Op::Rnn { .. }
            | Op::Mamba2 { .. }
            | Op::GatedDeltaNet { .. }
            | Op::Scan { .. }
            | Op::ConvTranspose2d { .. }
            | Op::Fft { .. }
            | Op::DequantGroupedMatMul { .. }
            | Op::DequantMoEWeights { .. }
            | Op::RngNormal { .. }
            | Op::RngUniform { .. }
            | Op::Sample { .. }
    )
}

/// One scheduled step: either a GPU compute dispatch or a CPU host-fallback
/// op (for families with no native SPIR-V kernel yet).
#[derive(Clone)]
enum Step {
    Gpu {
        kernel: &'static str,
        push: Vec<u8>,
        groups: (u32, u32, u32),
    },
    Host {
        op: Op,
        out: NodeId,
        out_shape: rlx_ir::Shape,
        inputs: Vec<NodeId>,
    },
}

/// A pre-recorded execution segment. The schedule is partitioned into maximal
/// runs of GPU dispatches (each recorded ONCE into a reusable command buffer at
/// compile time) separated by CPU host-fallback ops. At run time a GPU segment
/// is a single `queue_submit` of its prebuilt command buffer — no per-step
/// allocation, recording, or fence churn. See [`record_segments`].
enum Segment {
    /// A prebuilt command buffer covering a run of consecutive GPU dispatches.
    Gpu(vk::CommandBuffer),
    /// A CPU host-fallback op, evaluated against the mapped arena between GPU
    /// segments (HOST_COHERENT memory, queue idle here — see `run_read_outputs`).
    Host {
        op: Op,
        out: NodeId,
        out_shape: rlx_ir::Shape,
        inputs: Vec<NodeId>,
    },
}

pub struct VulkanExecutable {
    /// Post-legalize, f32-uniform graph (kept for `clone_for_cache`).
    graph: Graph,
    arena: Arena,
    schedule: Vec<Step>,
    /// Pre-recorded segments (GPU command buffers + interleaved host ops). Built
    /// once from `schedule`; reused every `run`. Empty when caching is disabled
    /// (`RLX_VULKAN_NOCACHE=1`), in which case the legacy per-run record path
    /// drives `schedule` directly.
    segments: Vec<Segment>,
    /// Reusable fence for the cached submit path (reset after each wait).
    fence: vk::Fence,
    /// Whether the cached pre-recorded path is active.
    cached: bool,
    input_ids: HashMap<String, NodeId>,
    param_ids: HashMap<String, NodeId>,
    output_ids: Vec<NodeId>,
    output_dtypes: Vec<DType>,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    rng: RngOptions,
    active_extent: Option<(usize, usize)>,
    /// GPU-resident input handles (KV-cache style). Host mirror is kept only
    /// until the handle becomes resident (fed in-arena from an output), after
    /// which it is cleared — the value lives purely in the arena slot.
    gpu_handles: HashMap<String, Vec<f32>>,
    /// `handle_name → output index`: after each run, that output's arena slot
    /// is folded back into the handle's input slot (in-place, no host copy).
    gpu_handle_feeds: HashMap<String, usize>,
    /// Handles whose value is live in the arena (skip host re-upload).
    gpu_handle_resident: HashSet<String>,
    /// `handle_name → output index` for the *row* feed (decode graphs that emit
    /// the new K/V token at the LAST row of a bucket-padded output, e.g. llama32
    /// `concat(past_k, k_new)`). Driven explicitly via [`feed_kv_row`] after a
    /// logits-only run; kept separate from `gpu_handle_feeds` so the generic
    /// prefix propagation never fires for these.
    kv_row_feeds: HashMap<String, usize>,
}

unsafe impl Send for VulkanExecutable {}

// ── memory plan (f32-uniform bump allocator; same shape as rlx-cuda) ───────

fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
        if matches!(
            node.op,
            Op::Reshape { .. } | Op::Cast { .. } | Op::StopGradient
        ) {
            if let Some(in_id) = node.inputs.first()
                && let Some(slot) = assignments.get(in_id)
            {
                let aliased = slot.clone();
                assignments.insert(node.id, aliased);
                schedule.push(node.id);
                continue;
            }
        }
        let elems = node.shape.num_elements().unwrap_or(0);
        // f32-uniform arena: F32 (and widened F16/BF16/int) params occupy 4 bytes
        // per element, but U8/I8 packed quant weights are stored as RAW BYTES
        // (`set_param_bytes`) and read via byte addressing in the dequant kernel —
        // sizing them `elems*4` like f32 inflated the arena ~4× (≈10 GB on
        // Orpheus-3B Q4_K). Size by the real dtype. Slots stay `align`-aligned so
        // every f32 word offset is still exact and the GEMV's word-relative
        // weight addressing is unaffected.
        let elem_size = match node.shape.dtype() {
            DType::U8 | DType::I8 => 1,
            _ => 4,
        };
        let bytes = (elems * elem_size).max(4);
        let aligned = bytes.div_ceil(align) * align;
        assignments.insert(
            node.id,
            BufferSlot {
                offset: cursor,
                size: aligned,
            },
        );
        schedule.push(node.id);
        cursor += aligned;
    }
    MemoryPlan {
        arena_size: cursor.max(align),
        assignments,
        schedule,
    }
}

// ── small shape helpers ────────────────────────────────────────────────────

fn dims(graph: &Graph, id: NodeId) -> Vec<usize> {
    graph
        .node(id)
        .shape
        .dims()
        .iter()
        .map(|d| match d {
            rlx_ir::Dim::Static(s) => *s,
            _ => 0,
        })
        .collect()
}

fn numel(d: &[usize]) -> usize {
    d.iter()
        .product::<usize>()
        .max(if d.is_empty() { 1 } else { 0 })
}

/// Row-major contiguous strides for `d`.
fn contig_strides(d: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; d.len()];
    for i in (0..d.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * d[i + 1];
    }
    s
}

fn norm_axis(axis: i32, rank: usize) -> usize {
    if axis < 0 {
        (rank as i32 + axis).max(0) as usize
    } else {
        (axis as usize).min(rank.saturating_sub(1))
    }
}

// ── push-constant builder (std430, all 4-byte scalars / scalar arrays) ─────

#[derive(Default)]
struct Push {
    words: Vec<u32>,
}
impl Push {
    fn u(mut self, v: u32) -> Self {
        self.words.push(v);
        self
    }
    fn f(mut self, v: f32) -> Self {
        self.words.push(v.to_bits());
        self
    }
    fn us(mut self, vs: &[u32]) -> Self {
        self.words.extend_from_slice(vs);
        self
    }
    fn bytes(self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.words.len() * 4);
        for w in self.words {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }
}

fn ceil_div(n: usize, d: u32) -> u32 {
    (n as u64).div_ceil(d as u64) as u32
}

/// The `matmul_coop` kernel writes a full 16×16 output tile per workgroup, so M
/// and N must be 16-aligned — a partial output tile would store out of bounds.
/// K is unconstrained: the kernel zero-pads its final partial K-tile. Shapes
/// with non-16-aligned M/N fall back to the (fully general, fp32-exact) tiled
/// kernel, which is the better fit for them anyway.
fn coop_eligible(m: usize, _k: usize, n: usize) -> bool {
    m.is_multiple_of(16) && n.is_multiple_of(16)
}

/// Which matmul kernel to dispatch:
/// - default: `matmul_tiled` (shared-memory blocked **fp32**, exact) on native
///   drivers; `matmul` (scalar) on portability drivers (MoltenVK), where
///   tiling + barriers regress under Vulkan→Metal translation.
/// - `RLX_VULKAN_MATMUL=coop`: `matmul_coop`, the tensor-core path (f16·f16→f32
///   cooperative matrix). It is **opt-in** because f16 operands trade precision
///   for throughput (not fp32-exact), so it is never auto-selected — that would
///   silently degrade accuracy. Used only when the device advertises a usable
///   config (`coop_matmul`) and M,N are 16-aligned (K is arbitrary); otherwise
///   falls back to the exact tiled kernel (see `coop_eligible`).
/// - `RLX_VULKAN_MATMUL=scalar|tiled`: force that fp32 kernel (A/B benching).
fn matmul_kernel(m: usize, k: usize, n: usize) -> &'static str {
    let dev = vulkan_device();
    let portability = dev.map(|d| d.portability).unwrap_or(false);
    let coop = dev.map(|d| d.coop_matmul).unwrap_or(false);
    match std::env::var("RLX_VULKAN_MATMUL").ok().as_deref() {
        Some("scalar") => "matmul",
        Some("tiled") => "matmul_tiled",
        Some("coop") if coop && coop_eligible(m, k, n) => "matmul_coop",
        Some("coop") => "matmul_tiled",
        _ if portability => "matmul",
        _ => "matmul_tiled",
    }
}

/// 1-D workgroup count for `n` items at `local` threads/group. Assumes the
/// device's `maxComputeWorkGroupCount[0]` is large (true on desktop GPUs;
/// the Vulkan minimum of 65535 caps ~16M elements/dispatch — a follow-up
/// would switch to a grid-stride loop).
fn groups1d(n: usize, local: u32) -> (u32, u32, u32) {
    (ceil_div(n, local).max(1), 1, 1)
}

fn act_id(a: Activation) -> u32 {
    match a {
        Activation::Gelu => 0,
        Activation::GeluApprox => 1,
        Activation::Silu => 2,
        Activation::Relu => 3,
        Activation::Sigmoid => 4,
        Activation::Tanh => 5,
        Activation::Exp => 6,
        Activation::Log => 7,
        Activation::Sqrt => 8,
        Activation::Rsqrt => 9,
        Activation::Neg => 10,
        Activation::Abs => 11,
        Activation::Sin => 12,
        Activation::Cos => 13,
        Activation::Tan => 14,
        Activation::Atan => 15,
        Activation::Round => 16,
    }
}

fn binop_id(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Max => 4,
        BinaryOp::Min => 5,
        BinaryOp::Pow => 6,
    }
}

fn cmp_id(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

fn reduce_id(op: ReduceOp) -> u32 {
    match op {
        ReduceOp::Sum => 0,
        ReduceOp::Mean => 1,
        ReduceOp::Max => 2,
        ReduceOp::Min => 3,
        ReduceOp::Prod => 4,
    }
}

impl VulkanExecutable {
    pub fn compile(graph: Graph) -> Self {
        Self::compile_rng(graph, RngOptions::default())
    }

    /// Prepare the graph (legalize → primitive set), plan the arena, and build
    /// the dispatch schedule. Panics with a clear message if the graph
    /// contains an op no decomposition rule can reduce to [`SUPPORTED_OPS`].
    pub fn compile_rng(graph: Graph, rng: RngOptions) -> Self {
        use rlx_opt::pass::Pass as _;

        let graph = rlx_opt::LowerControlFlow.run(graph);
        // `FusedAttentionBlock` is claimed (so it legalizes), but there is
        // no monolithic fused-attention kernel — decompose it to primitives
        // first. FAB-only (not the whole-graph unfuse) so nothing else is
        // touched. No-op when no FAB node is present.
        let graph = rlx_opt::unfuse::unfuse_attention_block(graph);
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, SUPPORTED_OPS)
            .unwrap_or_else(|errs| panic!("{}", rlx_opt::format_legalize_error("vulkan", &errs)));
        // Materialize mid-axis broadcasts so Binary operands are equal-shaped
        // or trailing-broadcast (our kernels only do trailing modulus).
        let graph = rlx_opt::LegalizeBroadcast.run(graph);

        Self::build(graph, rng)
    }

    fn build(graph: Graph, rng: RngOptions) -> Self {
        let dev = vulkan_device().expect("rlx-vulkan: no device");
        let kern = kernels().expect("rlx-vulkan: no kernels");

        let plan = plan_f32_uniform(&graph, 16);
        let arena = Arena::from_plan(&plan);

        // Upload constants (widened to f32 — the arena is f32-uniform).
        for node in graph.nodes() {
            if let Op::Constant { data } = &node.op
                && arena.has(node.id)
                && !data.is_empty()
            {
                let f = widen_const_to_f32(data, node.shape.dtype());
                arena.write_f32(node.id, &f);
            }
        }

        let mut input_ids = HashMap::new();
        let mut param_ids = HashMap::new();
        for node in graph.nodes() {
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

        let output_ids = graph.outputs.clone();
        let output_dtypes = output_ids
            .iter()
            .map(|&id| graph.node(id).shape.dtype())
            .collect();

        let (schedule, deps) = build_schedule(&graph, &arena);

        // Descriptor set binding the whole arena to binding 0.
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let desc_pool = unsafe {
            dev.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .expect("vk descriptor_pool");
        let set_layouts = [kern.dsl];
        let desc_set = unsafe {
            dev.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .expect("vk descriptor_set")[0];
        let buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(arena.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(desc_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_info);
        unsafe { dev.device.update_descriptor_sets(&[write], &[]) };

        // Pre-record the static schedule into reusable command buffers (one per
        // maximal GPU run). The whole schedule — kernels, push constants,
        // workgroup counts — is fixed at compile time; per-step inputs are
        // memcpy'd into the host-visible arena, never the command stream. So a
        // single recording is valid for every `run`, turning each step into one
        // `queue_submit` instead of allocate → record → fence → free.
        let cached = std::env::var("RLX_VULKAN_NOCACHE").as_deref() != Ok("1");
        let (segments, fence) = if cached {
            let segs = record_segments(dev, kern, desc_set, &schedule, &deps);
            (segs, dev.create_reusable_fence())
        } else {
            (Vec::new(), vk::Fence::null())
        };

        if std::env::var_os("RLX_VULKAN_DEBUG").is_some() {
            let gpu = schedule
                .iter()
                .filter(|s| matches!(s, Step::Gpu { .. }))
                .count();
            let host = schedule.len() - gpu;
            let gpu_segs = segments
                .iter()
                .filter(|s| matches!(s, Segment::Gpu(_)))
                .count();
            let mut hist: HashMap<&'static str, usize> = HashMap::new();
            for s in &schedule {
                if let Step::Gpu { kernel, .. } = s {
                    *hist.entry(kernel).or_default() += 1;
                }
            }
            let mut by_count: Vec<_> = hist.into_iter().collect();
            by_count.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
            eprintln!(
                "[rlx-vulkan] schedule: {gpu} gpu dispatches, {host} host ops; \
                 cached={cached} ({gpu_segs} gpu submit(s)/run)"
            );
            eprintln!("[rlx-vulkan] dispatch histogram: {by_count:?}");
        }

        Self {
            graph,
            arena,
            schedule,
            segments,
            fence,
            cached,
            input_ids,
            param_ids,
            output_ids,
            output_dtypes,
            desc_pool,
            desc_set,
            rng,
            active_extent: None,
            gpu_handles: HashMap::new(),
            gpu_handle_feeds: HashMap::new(),
            gpu_handle_resident: HashSet::new(),
            kv_row_feeds: HashMap::new(),
        }
    }

    pub fn set_param(&mut self, name: &str, data: &[f32]) {
        if let Some(&id) = self.param_ids.get(name) {
            self.arena.write_f32(id, data);
        }
    }

    /// Raw-byte param upload (packed weights). The arena is f32-uniform, so
    /// callers should normally use [`set_param`]; this exists for symmetry.
    pub fn set_param_bytes(&mut self, name: &str, data: &[u8]) {
        if let Some(&id) = self.param_ids.get(name) {
            self.arena.write_bytes(id, data);
        }
    }

    pub fn output_dtypes(&self) -> Vec<DType> {
        self.output_dtypes.clone()
    }

    pub fn set_active_extent(&mut self, extent: Option<(usize, usize)>) {
        self.active_extent = extent;
    }

    /// Persistent input buffer for KV-cache style graphs. Writes `data` into the
    /// input's arena slot once; subsequent decode steps reuse it (and, with a
    /// feed wired, update it in place on-device). Returns false if `name` is not
    /// a graph input. Mirrors the rlx-metal handle semantics.
    pub fn bind_gpu_handle(&mut self, name: &str, data: &[f32]) -> bool {
        let Some(&id) = self.input_ids.get(name) else {
            return false;
        };
        // A fresh bind re-seeds from host, so it is no longer purely resident.
        self.gpu_handle_resident.remove(name);
        self.arena.write_f32(id, data);
        // Keep a host mirror only until the first in-arena feed makes it resident.
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

    /// Register a *row* feed (vs the generic prefix feed): after a decode run,
    /// row `src_row` of output `output_index` is folded into handle
    /// `handle_name`'s input slot at row `dst_row`. For decode graphs that emit
    /// the new K/V token at the last bucket-padded output row (llama32). Driven
    /// explicitly via [`feed_kv_row`]; does NOT trigger the auto-propagation in
    /// `run_read_outputs`.
    pub fn register_kv_row_feed(&mut self, handle_name: &str, output_index: usize) {
        self.kv_row_feeds
            .insert(handle_name.to_string(), output_index);
    }

    /// Fold each registered row-feed's new-token row into its resident handle
    /// slot, in-place on the arena (no host round-trip). Call after a
    /// logits-only `run_read_outputs(.., Some(&[0]))`. `row_elems` is kv_dim.
    pub fn feed_kv_row(&mut self, src_row: usize, dst_row: usize, row_elems: usize) {
        let feeds: Vec<(String, usize)> = self
            .kv_row_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            let Some(&out_id) = self.output_ids.get(out_idx) else {
                continue;
            };
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

    /// Read a handle back to host: from its fed output slot if wired, else the
    /// resident arena slot, else the host mirror. Used on bucket change / sync.
    pub fn read_gpu_handle(&self, name: &str) -> Option<Vec<f32>> {
        if let Some(&out_idx) = self.gpu_handle_feeds.get(name)
            && let Some(&out_id) = self.output_ids.get(out_idx)
        {
            let n = self.graph.node(out_id).shape.num_elements().unwrap_or(0);
            return Some(self.arena.read_f32(out_id, n));
        }
        if self.gpu_handle_resident.contains(name)
            && let Some(&id) = self.input_ids.get(name)
        {
            let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
            return Some(self.arena.read_f32(id, n));
        }
        self.gpu_handles.get(name).cloned()
    }

    /// Read one row (`row_inner` f32 elements at `row`) from graph output
    /// `out_idx`, directly from the arena. Used by resident KV decode to pull
    /// just the new-token K/V row to the host cache (for bucket transitions)
    /// without a full-output readback.
    pub fn read_output_row(
        &self,
        out_idx: usize,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        let id = *self.output_ids.get(out_idx)?;
        let base = self.arena.elem_offset(id) as usize + row * row_inner;
        Some(self.arena.read_f32_at_elem(base, row_inner))
    }

    /// Read one row (`row_inner` f32 elements at `row`) of a named handle,
    /// resolving the slot exactly as [`Self::read_gpu_handle`] does (fed
    /// output slot, else resident arena slot, else host mirror). Row-granular
    /// counterpart used by resident-KV decode at bucket transitions.
    pub fn read_gpu_handle_row(
        &self,
        name: &str,
        row: usize,
        row_inner: usize,
    ) -> Option<Vec<f32>> {
        if let Some(&out_idx) = self.gpu_handle_feeds.get(name)
            && let Some(&out_id) = self.output_ids.get(out_idx)
        {
            let base = self.arena.elem_offset(out_id) as usize + row * row_inner;
            return Some(self.arena.read_f32_at_elem(base, row_inner));
        }
        if self.gpu_handle_resident.contains(name)
            && let Some(&id) = self.input_ids.get(name)
        {
            let base = self.arena.elem_offset(id) as usize + row * row_inner;
            return Some(self.arena.read_f32_at_elem(base, row_inner));
        }
        self.gpu_handles.get(name).map(|v| {
            let start = (row * row_inner).min(v.len());
            let end = (start + row_inner).min(v.len());
            v[start..end].to_vec()
        })
    }

    /// Fold each fed output's arena slot back into its handle input slot,
    /// in-place (no host round-trip). The copy length honors `active_extent`
    /// `(actual_rows, upper)` so only the valid prefix (incl. the new token row)
    /// is carried — the rest of the bucket-padded slot stays zero.
    fn propagate_gpu_handle_feeds_in_arena(&mut self) {
        let extent = self.active_extent;
        let feeds: Vec<(String, usize)> = self
            .gpu_handle_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            let Some(&out_id) = self.output_ids.get(out_idx) else {
                continue;
            };
            let Some(&in_id) = self.input_ids.get(name.as_str()) else {
                continue;
            };
            if in_id != out_id {
                let out_elems = self.graph.node(out_id).shape.num_elements().unwrap_or(0);
                let copy_elems = match extent {
                    Some((actual, upper)) if upper > 0 => actual * (out_elems / (upper + 1)).max(1),
                    _ => out_elems,
                };
                self.arena
                    .copy_node_f32_prefix(in_id, out_id, copy_elems.min(out_elems));
            }
            self.gpu_handle_resident.insert(name.clone());
            // Drop the host mirror — the value now lives in the arena.
            self.gpu_handles.insert(name.clone(), Vec::new());
        }
    }

    /// Refresh host mirrors from fed outputs (only when all outputs are read).
    fn refresh_gpu_handles_from_outputs(&mut self) {
        let feeds: Vec<(String, usize)> = self
            .gpu_handle_feeds
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        for (name, out_idx) in feeds {
            let Some(&out_id) = self.output_ids.get(out_idx) else {
                continue;
            };
            let n = self.graph.node(out_id).shape.num_elements().unwrap_or(0);
            let src = self.arena.read_f32(out_id, n);
            self.gpu_handles.insert(name, src);
        }
    }

    pub fn set_rng(&mut self, rng: RngOptions) {
        self.rng = rng;
    }

    pub fn rng(&self) -> RngOptions {
        self.rng
    }

    pub fn run(&mut self, inputs: &[(&str, &[f32])]) -> Vec<Vec<f32>> {
        self.run_read_outputs(inputs, None)
    }

    pub fn run_read_outputs(
        &mut self,
        inputs: &[(&str, &[f32])],
        read_indices: Option<&[usize]>,
    ) -> Vec<Vec<f32>> {
        // Re-seed any GPU handle that is neither resident in the arena nor about
        // to be overwritten by an explicit input this step (first step after a
        // bind, or a bucket reinstall). Resident handles are skipped — their
        // value already lives in the arena from the previous step's feed.
        for (name, data) in &self.gpu_handles {
            if self.gpu_handle_resident.contains(name) || inputs.iter().any(|(n, _)| n == name) {
                continue;
            }
            if let Some(&id) = self.input_ids.get(name) {
                self.arena.write_f32(id, data);
            }
        }
        // Upload inputs.
        for &(name, data) in inputs {
            if let Some(&id) = self.input_ids.get(name) {
                self.arena.write_f32(id, data);
            }
        }

        // Execute the schedule in segments: runs of consecutive GPU dispatches
        // are submitted together; a host-fallback step flushes the queue, runs
        // on the CPU directly against the host-visible arena, and the next GPU
        // segment picks up its result (HOST_COHERENT memory).
        let dev = vulkan_device().expect("rlx-vulkan: no device");
        let kern = kernels().expect("rlx-vulkan: no kernels");
        let desc_set = self.desc_set;
        let layout = kern.pipeline_layout;

        if self.cached {
            // Fast path: each GPU segment is a single submit of its pre-recorded
            // command buffer; host segments run on the CPU between submits. Arena
            // reads/writes are `&self` (interior mutability via the mapped ptr),
            // so the whole loop borrows `self` immutably.
            let nseg = self.segments.len();
            for si in 0..nseg {
                match &self.segments[si] {
                    Segment::Gpu(cmd) => {
                        let cmd = *cmd;
                        dev.submit_recorded_wait(cmd, self.fence);
                    }
                    Segment::Host {
                        op,
                        out,
                        out_shape,
                        inputs: in_ids,
                    } => {
                        let in_specs: Vec<(rlx_ir::Shape, crate::host::HostBuf)> = in_ids
                            .iter()
                            .map(|&id| {
                                let sh = self.graph.node(id).shape.clone();
                                let nn = sh.num_elements().unwrap_or(0);
                                let buf = if matches!(sh.dtype(), DType::U8 | DType::I8) {
                                    crate::host::HostBuf::Bytes(self.arena.read_bytes(id, nn))
                                } else {
                                    crate::host::HostBuf::F32(self.arena.read_f32(id, nn))
                                };
                                (sh, buf)
                            })
                            .collect();
                        let result = crate::host::eval(op, out_shape, &in_specs);
                        self.arena.write_f32(*out, &result);
                    }
                }
            }
            // Fall through to the feed/readback tail below.
            return self.finish_run(read_indices);
        }

        let n = self.schedule.len();
        let mut i = 0;
        while i < n {
            let start = i;
            while i < n && matches!(self.schedule[i], Step::Gpu { .. }) {
                i += 1;
            }
            if i > start {
                let gpu = self.schedule[start..i].to_vec();
                dev.submit_and_wait(|cmd| unsafe {
                    dev.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &[desc_set],
                        &[],
                    );
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        );
                    for (j, step) in gpu.iter().enumerate() {
                        if let Step::Gpu {
                            kernel,
                            push,
                            groups,
                        } = step
                        {
                            let pipeline = kern.pipeline(kernel);
                            dev.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                pipeline,
                            );
                            dev.device.cmd_push_constants(
                                cmd,
                                layout,
                                vk::ShaderStageFlags::COMPUTE,
                                0,
                                push,
                            );
                            dev.device.cmd_dispatch(cmd, groups.0, groups.1, groups.2);
                            if j + 1 < gpu.len() {
                                dev.device.cmd_pipeline_barrier(
                                    cmd,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::DependencyFlags::empty(),
                                    &[barrier],
                                    &[],
                                    &[],
                                );
                            }
                        }
                    }
                });
            }
            if i < n {
                if let Step::Host {
                    op,
                    out,
                    out_shape,
                    inputs: in_ids,
                } = self.schedule[i].clone()
                {
                    let in_specs: Vec<(rlx_ir::Shape, crate::host::HostBuf)> = in_ids
                        .iter()
                        .map(|&id| {
                            let sh = self.graph.node(id).shape.clone();
                            let nn = sh.num_elements().unwrap_or(0);
                            // Packed quant weights (U8/I8) are read as raw bytes;
                            // everything else is f32 from the uniform arena.
                            let buf = if matches!(sh.dtype(), DType::U8 | DType::I8) {
                                crate::host::HostBuf::Bytes(self.arena.read_bytes(id, nn))
                            } else {
                                crate::host::HostBuf::F32(self.arena.read_f32(id, nn))
                            };
                            (sh, buf)
                        })
                        .collect();
                    let result = crate::host::eval(&op, &out_shape, &in_specs);
                    self.arena.write_f32(out, &result);
                }
                i += 1;
            }
        }

        self.finish_run(read_indices)
    }

    /// Shared post-execution tail for both the cached and legacy run paths: fold
    /// fed outputs (new-token K/V) back into their handle input slots in-place on
    /// the arena — the queue is idle here so the mapped memory is coherent. When
    /// all outputs are read back, also refresh host mirrors; for logits-only
    /// decode (`read_indices == Some([0])`) the K/V never leaves the arena, which
    /// is the whole point. Then read the requested outputs.
    fn finish_run(&mut self, read_indices: Option<&[usize]>) -> Vec<Vec<f32>> {
        if !self.gpu_handle_feeds.is_empty() {
            self.propagate_gpu_handle_feeds_in_arena();
            if read_indices.is_none() {
                self.refresh_gpu_handles_from_outputs();
            }
        }

        let want: Vec<usize> = match read_indices {
            Some(ix) => ix.to_vec(),
            None => (0..self.output_ids.len()).collect(),
        };
        want.into_iter()
            .filter_map(|i| {
                let id = *self.output_ids.get(i)?;
                let n = self.graph.node(id).shape.num_elements().unwrap_or(0);
                Some(self.arena.read_f32(id, n))
            })
            .collect()
    }

    /// Deep copy for `clone_box`: fresh arena/descriptors with the same params
    /// and constants already resident.
    pub fn clone_for_cache(&self) -> Self {
        let mut twin = Self::build(self.graph.clone(), self.rng);
        twin.active_extent = self.active_extent;
        // Copy the whole arena (params + constants, plus any resident K/V)
        // byte-for-byte, then carry the GPU-handle bookkeeping so the twin keeps
        // feeding/resident semantics identical to the source.
        self.arena.copy_into(&twin.arena);
        twin.gpu_handles = self.gpu_handles.clone();
        twin.gpu_handle_feeds = self.gpu_handle_feeds.clone();
        twin.gpu_handle_resident = self.gpu_handle_resident.clone();
        twin.kv_row_feeds = self.kv_row_feeds.clone();
        twin
    }
}

impl Drop for VulkanExecutable {
    fn drop(&mut self) {
        if let Some(dev) = vulkan_device() {
            // Free the pre-recorded command buffers and the reusable fence
            // before tearing down the pool they came from.
            let cmds: Vec<vk::CommandBuffer> = self
                .segments
                .iter()
                .filter_map(|s| match s {
                    Segment::Gpu(cmd) => Some(*cmd),
                    Segment::Host { .. } => None,
                })
                .collect();
            if !cmds.is_empty() {
                dev.free_cmds(&cmds);
            }
            if self.fence != vk::Fence::null() {
                dev.destroy_fence(self.fence);
            }
            unsafe {
                dev.device.destroy_descriptor_pool(self.desc_pool, None);
            }
        }
    }
}

/// Pre-record the static schedule into reusable command buffers. The schedule is
/// partitioned into maximal runs of consecutive GPU dispatches; each run is
/// recorded once into a primary command buffer that is resubmitted unchanged
/// every `run`. Host-fallback ops become `Segment::Host` markers, executed on the
/// CPU between GPU submits. Recorded WITHOUT `ONE_TIME_SUBMIT` so the buffers can
/// be resubmitted.
///
/// Barriers are placed only where a real memory hazard exists (per `deps`): a
/// dispatch that reads/writes a slot touched since the last barrier flushes with
/// one global shader-memory barrier, which both lets the driver overlap
/// independent dispatches and — decisively on MoltenVK, where each barrier forces
/// a Metal compute-encoder restart — slashes the barrier count for the typical
/// MLP/CNN graph (most of whose 100+ dispatches are independent elementwise/shape
/// glue). `RLX_VULKAN_FULLBARRIER=1` restores a barrier between every pair
/// (conservative fallback); `RLX_VULKAN_NOBARRIER=1` drops them all (unsafe —
/// diagnostic only).
fn record_segments(
    dev: &crate::device::VulkanDevice,
    kern: &crate::kernels::Kernels,
    desc_set: vk::DescriptorSet,
    schedule: &[Step],
    deps: &[StepDep],
) -> Vec<Segment> {
    let layout = kern.pipeline_layout;
    let no_barrier = std::env::var("RLX_VULKAN_NOBARRIER").as_deref() == Ok("1");
    let full_barrier = std::env::var("RLX_VULKAN_FULLBARRIER").as_deref() == Ok("1");
    let mut segments = Vec::new();
    let n = schedule.len();
    let mut i = 0;
    while i < n {
        let start = i;
        while i < n && matches!(schedule[i], Step::Gpu { .. }) {
            i += 1;
        }
        if i > start {
            let run = &schedule[start..i];
            let run_deps = &deps[start..i];
            let cmd = dev.alloc_primary_cmd();
            unsafe {
                dev.device
                    .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                    .expect("vk begin cmd");
                dev.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    &[desc_set],
                    &[],
                );
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                // Slots written / read since the last barrier (arena elem
                // offsets). A dispatch hazards on RAW (reads a written slot),
                // WAW (writes a written slot) or WAR (writes a read slot); on a
                // hazard we flush with one barrier and reset the sets.
                let mut wrote: HashSet<u32> = HashSet::new();
                let mut read: HashSet<u32> = HashSet::new();
                for (j, step) in run.iter().enumerate() {
                    if let Step::Gpu {
                        kernel,
                        push,
                        groups,
                    } = step
                    {
                        let dep = &run_deps[j];
                        let hazard = !wrote.is_empty()
                            && (dep.reads.iter().any(|r| wrote.contains(r))
                                || wrote.contains(&dep.write)
                                || read.contains(&dep.write));
                        let emit_barrier = j > 0 && !no_barrier && (full_barrier || hazard);
                        if emit_barrier {
                            dev.device.cmd_pipeline_barrier(
                                cmd,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &[barrier],
                                &[],
                                &[],
                            );
                            wrote.clear();
                            read.clear();
                        }
                        let pipeline = kern.pipeline(kernel);
                        dev.device
                            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                        dev.device.cmd_push_constants(
                            cmd,
                            layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            push,
                        );
                        dev.device.cmd_dispatch(cmd, groups.0, groups.1, groups.2);
                        wrote.insert(dep.write);
                        for &r in &dep.reads {
                            read.insert(r);
                        }
                    }
                }
                dev.device.end_command_buffer(cmd).expect("vk end cmd");
            }
            segments.push(Segment::Gpu(cmd));
        }
        if i < n {
            if let Step::Host {
                op,
                out,
                out_shape,
                inputs,
            } = &schedule[i]
            {
                segments.push(Segment::Host {
                    op: op.clone(),
                    out: *out,
                    out_shape: out_shape.clone(),
                    inputs: inputs.clone(),
                });
            }
            i += 1;
        }
    }
    segments
}

/// Widen a constant byte blob (any IR dtype) to f32 for the f32-uniform arena.
fn widen_const_to_f32(data: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F16 => data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => data
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::F64 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::I64 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        DType::I32 | DType::U32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        DType::I16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        DType::I8 => data.iter().map(|&b| b as i8 as f32).collect(),
        DType::U8 | DType::Bool => data.iter().map(|&b| b as f32).collect(),
        DType::C64 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

// ── schedule construction ──────────────────────────────────────────────────

/// Per-GPU-step memory footprint, used to place barriers only where a real data
/// hazard exists. The arena is a bump allocator with one unique slot per node
/// (only Reshape/Cast/StopGradient alias, and aliases share an offset), so
/// tracking arena *element offsets* captures aliasing for free. `reads` are the
/// node's input slot offsets; `write` is its output slot offset.
#[derive(Clone, Default)]
struct StepDep {
    reads: Vec<u32>,
    write: u32,
}

/// Build the dispatch schedule plus, in lockstep, the per-step dependency info
/// (`StepDep`) that [`record_segments`] uses to elide redundant barriers. Each
/// graph node contributes its node-level footprint to every `Step` it emits
/// (most nodes emit one; `Concat` emits one per input — conservatively sharing
/// the node footprint, which over-serializes only a concat's own sub-copies).
fn build_schedule(graph: &Graph, arena: &Arena) -> (Vec<Step>, Vec<StepDep>) {
    let mut steps = Vec::new();
    let mut deps: Vec<StepDep> = Vec::new();
    for node in graph.nodes() {
        let off = |id: NodeId| arena.elem_offset(id);
        let out = node.id;
        let before = steps.len();
        match &node.op {
            // Leaves / aliases — no dispatch.
            Op::Input { .. }
            | Op::Param { .. }
            | Op::Constant { .. }
            | Op::Reshape { .. }
            | Op::Cast { .. }
            | Op::StopGradient => {}

            Op::Binary(op) => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(graph, out));
                let an = numel(&dims(graph, a));
                let bn = numel(&dims(graph, b));
                let push = Push::default()
                    .u(n as u32)
                    .u(off(a))
                    .u(off(b))
                    .u(off(out))
                    .u(if an == n { 0 } else { an as u32 })
                    .u(if bn == n { 0 } else { bn as u32 })
                    .u(binop_id(*op))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "binary",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::Compare(op) => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(graph, out));
                let an = numel(&dims(graph, a));
                let bn = numel(&dims(graph, b));
                let push = Push::default()
                    .u(n as u32)
                    .u(off(a))
                    .u(off(b))
                    .u(off(out))
                    .u(if an == n { 0 } else { an as u32 })
                    .u(if bn == n { 0 } else { bn as u32 })
                    .u(cmp_id(*op))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "compare",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::Where => {
                let c = node.inputs[0];
                let a = node.inputs[1];
                let b = node.inputs[2];
                let n = numel(&dims(graph, out));
                let cn = numel(&dims(graph, c));
                let an = numel(&dims(graph, a));
                let bn = numel(&dims(graph, b));
                let push = Push::default()
                    .u(n as u32)
                    .u(off(c))
                    .u(off(a))
                    .u(off(b))
                    .u(off(out))
                    .u(if cn == n { 0 } else { cn as u32 })
                    .u(if an == n { 0 } else { an as u32 })
                    .u(if bn == n { 0 } else { bn as u32 })
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "where",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::Activation(act) => {
                let x = node.inputs[0];
                let n = numel(&dims(graph, out));
                let push = Push::default()
                    .u(n as u32)
                    .u(off(x))
                    .u(off(out))
                    .u(act_id(*act))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "unary",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::MatMul => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let ad = dims(graph, a);
                let bd = dims(graph, b);
                let od = dims(graph, out);
                let (m, k) = (ad[ad.len() - 2], ad[ad.len() - 1]);
                let n = bd[bd.len() - 1];
                let batch = if od.len() > 2 {
                    numel(&od[..od.len() - 2])
                } else {
                    1
                };
                let a_batch = if ad.len() > 2 {
                    numel(&ad[..ad.len() - 2])
                } else {
                    1
                };
                let b_batch = if bd.len() > 2 {
                    numel(&bd[..bd.len() - 2])
                } else {
                    1
                };
                let a_bs = if a_batch <= 1 { 0 } else { m * k };
                let b_bs = if b_batch <= 1 { 0 } else { k * n };
                let push = Push::default()
                    .u(m as u32)
                    .u(k as u32)
                    .u(n as u32)
                    .u(off(a))
                    .u(off(b))
                    .u(off(out))
                    .u(batch as u32)
                    .u(a_bs as u32)
                    .u(b_bs as u32)
                    .u((m * n) as u32)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: matmul_kernel(m, k, n),
                    push,
                    groups: (ceil_div(n, 16), ceil_div(m, 16), batch.max(1) as u32),
                });
            }

            Op::Reduce { op, axes, .. } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let rank = xd.len();
                // After LowerNonLastAxisReduce: last-axis single-axis reduce.
                let last = rank.saturating_sub(1);
                debug_assert!(
                    axes.as_slice() == [last] || (rank <= 1),
                    "rlx-vulkan: non-last-axis reduce should have been lowered"
                );
                let r = *xd.get(last).unwrap_or(&1);
                let outer = numel(&xd) / r.max(1);
                let push = Push::default()
                    .u(outer as u32)
                    .u(r as u32)
                    .u(off(x))
                    .u(off(out))
                    .u(reduce_id(*op))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "reduce",
                    push,
                    groups: groups1d(outer, 256),
                });
            }

            Op::Softmax { axis } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let ax = norm_axis(*axis, xd.len());
                let axis_len = xd[ax];
                let outer = numel(&xd[..ax]);
                let inner = numel(&xd[ax + 1..]);
                let push = Push::default()
                    .u(outer as u32)
                    .u(axis_len as u32)
                    .u(inner as u32)
                    .u(off(x))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "softmax",
                    push,
                    groups: groups1d(outer * inner, 256),
                });
            }

            Op::RmsNorm { axis, eps } => {
                // Op::RmsNorm carries (x, gamma, beta): y = x*rsqrt(ms+eps)*gamma + beta.
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let beta = node.inputs[2];
                let xd = dims(graph, x);
                let ax = norm_axis(*axis, xd.len());
                debug_assert_eq!(ax, xd.len().saturating_sub(1), "rmsnorm expects last axis");
                let n = xd[ax];
                let rows = numel(&xd) / n.max(1);
                let push = Push::default()
                    .u(rows as u32)
                    .u(n as u32)
                    .u(off(x))
                    .u(off(gamma))
                    .u(off(beta))
                    .u(off(out))
                    .f(*eps)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "rmsnorm",
                    push,
                    groups: groups1d(rows, 64),
                });
            }

            Op::LayerNorm { axis, eps } => {
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let has_beta = node.inputs.len() >= 3;
                let beta = if has_beta { node.inputs[2] } else { gamma };
                let xd = dims(graph, x);
                let ax = norm_axis(*axis, xd.len());
                let n = xd[ax];
                let rows = numel(&xd) / n.max(1);
                let push = Push::default()
                    .u(rows as u32)
                    .u(n as u32)
                    .u(off(x))
                    .u(off(gamma))
                    .u(off(beta))
                    .u(off(out))
                    .u(if has_beta { 1 } else { 0 })
                    .f(*eps)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "layernorm",
                    push,
                    groups: groups1d(rows, 64),
                });
            }

            Op::Rope {
                head_dim,
                n_rot,
                style,
            } => {
                let x = node.inputs[0];
                let cos = node.inputs[1];
                let sin = node.inputs[2];
                let xd = dims(graph, x);
                let (batch, seq, hidden) = if xd.len() >= 3 {
                    (xd[0], xd[1], xd[2])
                } else {
                    let total = numel(&xd);
                    (1, xd[0], total / xd[0].max(1))
                };
                let hd = *head_dim;
                let nh = hidden / hd.max(1);
                let tab_half = hd / 2;
                let cos_len = numel(&dims(graph, cos));
                let cos_rows = cos_len / tab_half.max(1);
                let per_token = (cos_rows == batch * seq && cos_rows != seq) as u32;
                let style_id = match style {
                    RopeStyle::NeoX => 0u32,
                    RopeStyle::GptJ => 1u32,
                };
                let push = Push::default()
                    .u(batch as u32)
                    .u(seq as u32)
                    .u(hidden as u32)
                    .u(hd as u32)
                    .u(*n_rot as u32)
                    .u(nh as u32)
                    .u(tab_half as u32)
                    .u(hidden as u32) // src_row_stride (no Narrow→Rope fusion)
                    .u(per_token)
                    .u(style_id)
                    .u(off(x))
                    .u(off(cos))
                    .u(off(sin))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "rope",
                    push,
                    groups: groups1d(batch * seq * nh, 64),
                });
            }

            Op::Attention {
                num_heads,
                head_dim,
                mask_kind,
                score_scale,
                ..
            } => {
                let q = node.inputs[0];
                let k = node.inputs[1];
                let v = node.inputs[2];
                let qd = dims(graph, q);
                let kd = dims(graph, k);
                let nh = *num_heads;
                let dh = *head_dim;
                let (batch, q_s, k_s, bhsd) = if qd.len() == 4 {
                    if qd[1] == nh {
                        (qd[0], qd[2], kd[2], 1u32) // [B,H,S,D]
                    } else {
                        (qd[0], qd[1], kd[1], 0u32) // [B,S,H,D]
                    }
                } else if qd.len() >= 3 {
                    (qd[0], qd[1], kd[1], 0u32)
                } else {
                    (1, qd[0], kd[0], 0u32)
                };
                let hs = (nh * dh) as u32;
                let (mask_kind_id, mask_off, window) = match mask_kind {
                    MaskKind::None => (0u32, 0u32, 0u32),
                    MaskKind::Causal => (1, 0, 0),
                    MaskKind::SlidingWindow(w) => (2, 0, *w as u32),
                    MaskKind::Custom => (3, off(node.inputs[3]), 0),
                    MaskKind::Bias => (4, off(node.inputs[3]), 0),
                };
                let scale = score_scale.unwrap_or((dh as f32).powf(-0.5));
                let push = Push::default()
                    .u(batch as u32)
                    .u(nh as u32)
                    .u(q_s as u32)
                    .u(k_s as u32)
                    .u(dh as u32)
                    .u(off(q))
                    .u(off(k))
                    .u(off(v))
                    .u(off(out))
                    .u(hs)
                    .u(hs)
                    .u(hs)
                    .u(bhsd)
                    .u(mask_kind_id)
                    .u(mask_off)
                    .u(window)
                    .f(scale)
                    .f(-1.0e30)
                    .f(0.5)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "attention",
                    push,
                    groups: groups1d(batch * nh * q_s, 64),
                });
            }

            Op::Transpose { perm } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let od = dims(graph, out);
                let in_str = contig_strides(&xd);
                let out_str = contig_strides(&od);
                let rank = od.len();
                let mut shape = [1u32; 6];
                let mut istr = [0u32; 6];
                let mut ostr = [0u32; 6];
                for ax in 0..rank {
                    shape[ax] = od[ax] as u32;
                    istr[ax] = in_str[perm[ax]] as u32;
                    ostr[ax] = out_str[ax] as u32;
                }
                let n = numel(&od);
                let push = Push::default()
                    .u(n as u32)
                    .u(rank as u32)
                    .u(off(x))
                    .u(off(out))
                    .us(&shape)
                    .us(&istr)
                    .us(&ostr)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "reindex",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::Narrow { axis, start, .. } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let od = dims(graph, out);
                let in_str = contig_strides(&xd);
                let out_str = contig_strides(&od);
                let rank = od.len();
                let mut shape = [1u32; 6];
                let mut istr = [0u32; 6];
                let mut ostr = [0u32; 6];
                for ax in 0..rank {
                    shape[ax] = od[ax] as u32;
                    istr[ax] = in_str[ax] as u32;
                    ostr[ax] = out_str[ax] as u32;
                }
                let in_off = off(x) + (*start * in_str[*axis]) as u32;
                let n = numel(&od);
                let push = Push::default()
                    .u(n as u32)
                    .u(rank as u32)
                    .u(in_off)
                    .u(off(out))
                    .us(&shape)
                    .us(&istr)
                    .us(&ostr)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "reindex",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::Expand { .. } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let od = dims(graph, out);
                let rank = od.len();
                // Right-align input dims to output rank.
                let pad = rank - xd.len();
                let in_str_full = contig_strides(&xd);
                let out_str = contig_strides(&od);
                let mut shape = [1u32; 6];
                let mut istr = [0u32; 6];
                let mut ostr = [0u32; 6];
                for ax in 0..rank {
                    shape[ax] = od[ax] as u32;
                    ostr[ax] = out_str[ax] as u32;
                    if ax < pad {
                        istr[ax] = 0;
                    } else {
                        let xi = ax - pad;
                        istr[ax] = if xd[xi] == 1 && od[ax] != 1 {
                            0
                        } else {
                            in_str_full[xi] as u32
                        };
                    }
                }
                let n = numel(&od);
                let push = Push::default()
                    .u(n as u32)
                    .u(rank as u32)
                    .u(off(x))
                    .u(off(out))
                    .us(&shape)
                    .us(&istr)
                    .us(&ostr)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "reindex",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::Concat { axis } => {
                let od = dims(graph, out);
                let out_str = contig_strides(&od);
                let rank = od.len();
                let mut axis_cursor = 0usize;
                for &inp in &node.inputs {
                    let id_dims = dims(graph, inp);
                    let in_str = contig_strides(&id_dims);
                    let mut shape = [1u32; 6];
                    let mut istr = [0u32; 6];
                    let mut ostr = [0u32; 6];
                    for ax in 0..rank {
                        shape[ax] = *id_dims.get(ax).unwrap_or(&1) as u32;
                        istr[ax] = *in_str.get(ax).unwrap_or(&0) as u32;
                        ostr[ax] = out_str[ax] as u32;
                    }
                    let out_off = off(out) + (axis_cursor * out_str[*axis]) as u32;
                    let n = numel(&id_dims);
                    let push = Push::default()
                        .u(n as u32)
                        .u(rank as u32)
                        .u(off(inp))
                        .u(out_off)
                        .us(&shape)
                        .us(&istr)
                        .us(&ostr)
                        .bytes();
                    steps.push(Step::Gpu {
                        kernel: "reindex",
                        push,
                        groups: groups1d(n, 256),
                    });
                    axis_cursor += *id_dims.get(*axis).unwrap_or(&1);
                }
            }

            Op::Gather { axis } => {
                let data = node.inputs[0];
                let idx = node.inputs[1];
                let dd = dims(graph, data);
                let ax = *axis;
                let out_outer = numel(&dd[..ax]);
                let axis_dim = dd[ax];
                let out_inner = numel(&dd[ax + 1..]);
                let n_idx = numel(&dims(graph, idx));
                let total = out_outer * n_idx * out_inner;
                let push = Push::default()
                    .u(out_outer as u32)
                    .u(n_idx as u32)
                    .u(out_inner as u32)
                    .u(axis_dim as u32)
                    .u(off(data))
                    .u(off(idx))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "gather",
                    push,
                    groups: groups1d(total, 256),
                });
            }

            Op::Cumsum { axis, exclusive } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let ax = norm_axis(*axis, xd.len());
                debug_assert_eq!(ax, xd.len().saturating_sub(1), "cumsum expects last axis");
                let cols = *xd.get(ax).unwrap_or(&1);
                let rows = numel(&xd) / cols.max(1);
                let push = Push::default()
                    .u(rows as u32)
                    .u(cols as u32)
                    .u(off(x))
                    .u(off(out))
                    .u(if *exclusive { 1 } else { 0 })
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "cumsum",
                    push,
                    groups: groups1d(rows, 64),
                });
            }

            Op::Reverse { axes } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let rank = xd.len();
                let mut shape = [1u32; 6];
                let mut flip = [0u32; 6];
                for ax in 0..rank {
                    shape[ax] = xd[ax] as u32;
                    flip[ax] = if axes.contains(&ax) { 1 } else { 0 };
                }
                let n = numel(&xd);
                let push = Push::default()
                    .u(n as u32)
                    .u(rank as u32)
                    .u(off(x))
                    .u(off(out))
                    .us(&shape)
                    .us(&flip)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "reverse",
                    push,
                    groups: groups1d(n, 256),
                });
            }

            Op::ArgMax { axis, .. } | Op::ArgMin { axis, .. } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let ax = (*axis).min(xd.len().saturating_sub(1));
                let axis_len = xd[ax];
                let outer = numel(&xd[..ax]);
                let inner = numel(&xd[ax + 1..]);
                let op_id = if matches!(node.op, Op::ArgMax { .. }) {
                    0
                } else {
                    1
                };
                let push = Push::default()
                    .u(outer as u32)
                    .u(axis_len as u32)
                    .u(inner as u32)
                    .u(off(x))
                    .u(off(out))
                    .u(op_id)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "argreduce",
                    push,
                    groups: groups1d(outer * inner, 256),
                });
            }

            Op::LayerNorm2d { eps } => {
                // x [N,C,H,W], gamma, beta [C].
                let x = node.inputs[0];
                let gamma = node.inputs[1];
                let beta = node.inputs[2];
                let xd = dims(graph, x);
                let (nn, cc, hw) = (xd[0], xd[1], xd[2] * xd[3]);
                let positions = nn * hw;
                let push = Push::default()
                    .u(positions as u32)
                    .u(cc as u32)
                    .u(hw as u32)
                    .u(off(x))
                    .u(off(gamma))
                    .u(off(beta))
                    .u(off(out))
                    .f(*eps)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "layernorm2d",
                    push,
                    groups: groups1d(positions, 64),
                });
            }

            Op::Pool {
                kind,
                kernel_size,
                stride,
                padding,
            } => {
                // 2-D pooling on NCHW (kernel_size.len() == 2).
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let od = dims(graph, out);
                let (nn, cc, hh, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let (oh, ow) = (od[2], od[3]);
                let (kh, kw) = (kernel_size[0], kernel_size[1]);
                let (sh, sw) = (stride[0], stride[1]);
                let (ph, pw) = (padding[0], padding[1]);
                let kind_id = reduce_id(*kind); // Max=2, Mean=1
                let push = Push::default()
                    .us(&[nn as u32, cc as u32, hh as u32, ww as u32])
                    .us(&[oh as u32, ow as u32])
                    .us(&[
                        kh as u32, kw as u32, sh as u32, sw as u32, ph as u32, pw as u32,
                    ])
                    .u(off(x))
                    .u(off(out))
                    .u(kind_id)
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "pool2d",
                    push,
                    groups: groups1d(nn * cc * oh * ow, 64),
                });
            }

            Op::ResizeNearest2x => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let (nn, cc, hh, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let push = Push::default()
                    .us(&[nn as u32, cc as u32, hh as u32, ww as u32])
                    .u(off(x))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "resize2x",
                    push,
                    groups: groups1d(nn * cc * hh * 4 * ww, 256),
                });
            }

            Op::GroupedMatMul => {
                // inputs: [input [M,K], weight [E,K,N], expert_idx [M]] → [M,N]
                let input = node.inputs[0];
                let weight = node.inputs[1];
                let idx = node.inputs[2];
                let id = dims(graph, input);
                let wd = dims(graph, weight);
                let (m, k) = (id[id.len() - 2], id[id.len() - 1]);
                let n = wd[wd.len() - 1];
                let push = Push::default()
                    .u(m as u32)
                    .u(k as u32)
                    .u(n as u32)
                    .u(off(input))
                    .u(off(weight))
                    .u(off(idx))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "grouped_matmul",
                    push,
                    groups: (ceil_div(n, 16), ceil_div(m, 16), 1),
                });
            }

            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                // 2-D conv (kernel_size.len() == 2). inputs: [x, weight, bias?].
                let x = node.inputs[0];
                let weight = node.inputs[1];
                let has_bias = node.inputs.len() > 2;
                let bias = if has_bias { node.inputs[2] } else { weight };
                let xd = dims(graph, x);
                let od = dims(graph, out);
                let (nn, cin, hh, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let (cout, oh, ow) = (od[1], od[2], od[3]);
                let (kh, kw) = (kernel_size[0], kernel_size[1]);
                let (sh, sw) = (stride[0], stride[1]);
                let (ph, pw) = (padding[0], padding[1]);
                let (dh, dw) = (dilation[0], dilation[1]);
                let push = Push::default()
                    .us(&[nn as u32, cin as u32, hh as u32, ww as u32])
                    .us(&[cout as u32, kh as u32, kw as u32])
                    .us(&[oh as u32, ow as u32])
                    .us(&[
                        sh as u32, sw as u32, ph as u32, pw as u32, dh as u32, dw as u32,
                    ])
                    .u(*groups as u32)
                    .u(if has_bias { 1 } else { 0 })
                    .u(off(x))
                    .u(off(weight))
                    .u(off(bias))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "conv2d",
                    push,
                    groups: groups1d(nn * cout * oh * ow, 64),
                });
            }

            Op::SelectiveScan { state_size } => {
                // inputs: [x, delta, a, b, c]; x,delta [B,S,H], a [H,N], b,c [B,S,N]
                let x = node.inputs[0];
                let delta = node.inputs[1];
                let a = node.inputs[2];
                let bmat = node.inputs[3];
                let cmat = node.inputs[4];
                let xd = dims(graph, x);
                let (bb, ss, hh) = (xd[0], xd[1], xd[2]);
                let nn = *state_size;
                let push = Push::default()
                    .u(bb as u32)
                    .u(ss as u32)
                    .u(hh as u32)
                    .u(nn as u32)
                    .u(off(x))
                    .u(off(delta))
                    .u(off(a))
                    .u(off(bmat))
                    .u(off(cmat))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "selective_scan",
                    push,
                    groups: groups1d(bb * hh, 64),
                });
            }

            Op::Im2Col {
                kernel_size,
                stride,
                padding,
                dilation,
            } => {
                // x [N,Cin,H,W] → [N*Ho*Wo, Cin*kH*kW]. out dims give Ho*Wo / cols.
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let (nn, cin, hh, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let (kh, kw) = (kernel_size[0], kernel_size[1]);
                let (sh, sw) = (stride[0], stride[1]);
                let (ph, pw) = (padding[0], padding[1]);
                let (dh, dw) = (dilation[0], dilation[1]);
                let eff_h = dh * (kh - 1) + 1;
                let eff_w = dw * (kw - 1) + 1;
                let ho = (hh + 2 * ph - eff_h) / sh + 1;
                let wo = (ww + 2 * pw - eff_w) / sw + 1;
                let push = Push::default()
                    .us(&[nn as u32, cin as u32, hh as u32, ww as u32])
                    .us(&[ho as u32, wo as u32])
                    .us(&[
                        kh as u32, kw as u32, sh as u32, sw as u32, ph as u32, pw as u32,
                        dh as u32, dw as u32,
                    ])
                    .u(off(x))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "im2col",
                    push,
                    groups: groups1d(nn * ho * wo * cin * kh * kw, 256),
                });
            }

            Op::ScatterAdd => {
                // updates [U, ...trailing], indices [U] → out [out_dim, ...trailing]
                let updates = node.inputs[0];
                let indices = node.inputs[1];
                let ud = dims(graph, updates);
                let od = dims(graph, out);
                let num_updates = ud[0];
                let trailing = numel(&ud[1..]);
                let out_dim = od[0];
                let push = Push::default()
                    .u(out_dim as u32)
                    .u(trailing as u32)
                    .u(num_updates as u32)
                    .u(off(updates))
                    .u(off(indices))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "scatter_add",
                    push,
                    groups: groups1d(out_dim * trailing, 256),
                });
            }

            Op::TopK { k } => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let n = *xd.last().unwrap_or(&1);
                let rows = numel(&xd) / n.max(1);
                let push = Push::default()
                    .u(rows as u32)
                    .u(n as u32)
                    .u(*k as u32)
                    .u(off(x))
                    .u(off(out))
                    .bytes();
                steps.push(Step::Gpu {
                    kernel: "topk",
                    push,
                    groups: groups1d(rows, 64),
                });
            }

            // GGUF K-quant dequant + matmul. Decode GEMV (m == 1) for the
            // Q4_K / Q6_K schemes runs natively; everything else (prefill
            // m > 1, other GGUF schemes) keeps the CPU host-fallback path.
            Op::DequantMatMul { scheme } => {
                use rlx_ir::quant::QuantScheme;
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let od = dims(graph, out);
                let n = *od.last().unwrap_or(&1);
                let m = numel(&od) / n.max(1);
                let k = numel(&xd) / m.max(1);
                let gpu_scheme = match scheme {
                    QuantScheme::GgufQ4K => Some(0u32),
                    QuantScheme::GgufQ6K => Some(1u32),
                    _ => None,
                };
                match gpu_scheme {
                    Some(sc) if m == 1 && k.is_multiple_of(256) && n >= 1 => {
                        let w = node.inputs[1];
                        let push = Push::default()
                            .u(n as u32)
                            .u(k as u32)
                            .u(off(x))
                            .u(off(w))
                            .u(off(out))
                            .u(sc)
                            .bytes();
                        steps.push(Step::Gpu {
                            kernel: "dequant_matmul",
                            push,
                            groups: groups1d(n, 64),
                        });
                    }
                    _ => {
                        steps.push(Step::Host {
                            op: node.op.clone(),
                            out: node.id,
                            out_shape: node.shape.clone(),
                            inputs: node.inputs.clone(),
                        });
                    }
                }
            }

            op if is_host_fallback(op) => {
                steps.push(Step::Host {
                    op: node.op.clone(),
                    out: node.id,
                    out_shape: node.shape.clone(),
                    inputs: node.inputs.clone(),
                });
            }

            other => panic!(
                "rlx-vulkan: op {:?} reached the scheduler but has no kernel \
                 (should have been rejected at legalize). Pin this graph to Device::Cpu.",
                other.kind()
            ),
        }

        // Attach the node's memory footprint to each Step it just produced. GPU
        // steps read the node's input slots and write its output slot; host
        // steps get an entry too (kept parallel to `steps`, unused at record
        // time since host ops sit on their own segment boundary).
        let added = steps.len() - before;
        if added > 0 {
            let reads: Vec<u32> = node
                .inputs
                .iter()
                .filter(|&&id| arena.has(id))
                .map(|&id| arena.elem_offset(id))
                .collect();
            let write = if arena.has(out) {
                arena.elem_offset(out)
            } else {
                0
            };
            for _ in 0..added {
                deps.push(StepDep {
                    reads: reads.clone(),
                    write,
                });
            }
        }
    }
    (steps, deps)
}
