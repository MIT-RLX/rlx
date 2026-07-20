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

use crate::buffer::{Arena, SHARD_STAGE_RESERVE, is_weight_elem, raw_elem_off};
use crate::device::vulkan_device;
use crate::kernels::kernels;
use ash::vk;
use rlx_compile::memory::MemoryPlan;
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
        // DiT modulation — claimed for fusion; `unfuse_dit_modulation`
        // expands forward Ada/Gated to primitives before SPIR-V.
        AdaLayerNorm,
        GatedResidual,
        // Packed DiT reverse — native SPIR-V (`ada_layer_norm_backward` /
        // `gated_residual_backward` shaders).
        AdaLayerNormBackward,
        GatedResidualBackward,
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
        ScatterNd,
        ScatterElements,
        GatherNd,
        GatherElements,
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
        ScanBackward,
        ScanBackwardXs,
        ConvTranspose2d,
        Fft,
        DequantMatMul,
        DequantGroupedMatMul,
        DequantMoEWeights, // GGUF quant
        // Native low-precision scaled GEMM (FP8/FP6/FP4 + parameterized `fNeXmY`
        // minifloats). No FP8/FP4 matrix HW on the current SPIR-V path, so these
        // decode-and-accumulate on the CPU reference against the mapped arena.
        ScaledMatMul,
        ScaledQuantScale,
        ScaledQuantize,
        ScaledDequantize,
        RngNormal,
        RngUniform,
        Sample, // RNG / generation
        // Core Riemannian / SPD-manifold ops — no native SPIR-V eigen kernel,
        // so they host-fallback to `rlx_cpu::spd` (F64) against the mapped
        // arena. See `crate::spd`.
        BiMap,
        ReEig,
        LogEig,
        SpdBatchNorm,
        SpdKarcherMean,
        SpdKarcherMeanWeighted,
        SpdLogMap,
        SpdExpMap,
        SpdParallelTransport,
        SpdMatrixFnBatch,
        ReEigBackward,
        LogEigBackward,
        SpdBatchNormBackwardX,
        SpdBatchNormBackwardG,
        SpdLogMapBackward,
        SpdExpMapBackward,
        SpdParallelTransportBackward,
        SpdMatrixFnBatchBackward,
        Eigh,
        EighBackward,
        EighBatch,
        EighBatchBackward,
        // In-graph collectives (`collective.*` Custom ops) — no SPIR-V kernel;
        // claimed so legalize passes, then `is_host_fallback` routes to
        // `Step::Host` → rlx-cpu (rlx-collectives kernel). Required for
        // data-parallel trainers (rlx-vision-bench) on Vulkan.
        Custom,
    ]
};

/// Ops with no native kernel that route to the CPU host-fallback path.
///
/// `DequantMatMul` is handled by its own scheduler arm: Q1_0 prefill uses
/// tiled `dequant_gemm_q1_0`; Q4_K / Q6_K / Q1_0 decode (and Q4/Q6 prefill)
/// use row-loop `dequant_matmul` GEMV. Other schemes fall back to CPU.
fn is_host_fallback(op: &Op) -> bool {
    // Any `Op::Custom` (in-graph collectives, onnx.* host kernels, …) — no
    // SPIR-V kernel; `host::eval` runs the registered rlx-cpu kernel against
    // the mapped arena. Collective names are listed only for documentation —
    // the catch-all keeps legalize (`OpKind::Custom` claimed) and schedule in
    // sync when new Custom ops appear.
    if matches!(op, Op::Custom { .. }) {
        return true;
    }
    matches!(
        op,
        Op::Lstm { .. }
            | Op::Gru { .. }
            | Op::Rnn { .. }
            | Op::Mamba2 { .. }
            | Op::GatedDeltaNet { .. }
            | Op::ConvTranspose2d { .. }
            | Op::Fft { .. }
            | Op::DequantGroupedMatMul { .. }
            | Op::DequantMoEWeights { .. }
            | Op::ScaledMatMul { .. }
            | Op::ScaledQuantScale { .. }
            | Op::ScaledQuantize { .. }
            | Op::ScaledDequantize { .. }
            | Op::RngNormal { .. }
            | Op::RngUniform { .. }
            | Op::Sample { .. } // ScanBackward* uses dedicated `Step::HostOp` / `HostOpDesc`.
    )
}

/// `RLX_VULKAN_HOST_OPS=conv,matmul,reindex,binary,unary,reduce,gather,norm,attn,scatter,all`
/// forces listed GPU op families through the CPU host fallback (diagnosis / parity).
fn host_ops_forced(op: &Op) -> bool {
    let Ok(raw) = std::env::var("RLX_VULKAN_HOST_OPS") else {
        return false;
    };
    let tags: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let all = tags.iter().any(|t| t == "all");
    let hit = |names: &[&str]| all || tags.iter().any(|t| names.contains(&t.as_str()));
    match op {
        Op::Conv { .. } => hit(&["conv", "conv2d"]),
        Op::MatMul | Op::GroupedMatMul => hit(&["matmul", "gemm"]),
        Op::Transpose { .. } | Op::Narrow { .. } | Op::Expand { .. } | Op::Concat { .. } => hit(&[
            "reindex",
            "shape",
            "transpose",
            "narrow",
            "expand",
            "concat",
        ]),
        Op::Binary(_) | Op::Compare(_) | Op::Where => hit(&["binary", "elemwise"]),
        Op::Activation(_) | Op::Cast { .. } => hit(&["unary", "activation", "cast"]),
        Op::Reduce { .. } => hit(&["reduce"]),
        Op::Softmax { .. } => hit(&["softmax"]),
        Op::ArgMax { .. } | Op::ArgMin { .. } => hit(&["reduce", "argreduce"]),
        Op::Gather { .. } => hit(&["gather"]),
        Op::Cumsum { .. } | Op::Reverse { .. } => hit(&["cumsum", "reverse", "reindex"]),
        Op::Pool { .. } | Op::Im2Col { .. } | Op::ResizeNearest2x => {
            hit(&["pool", "im2col", "vision", "conv"])
        }
        Op::RmsNorm { .. } | Op::LayerNorm { .. } | Op::LayerNorm2d { .. } => {
            hit(&["norm", "rmsnorm", "layernorm"])
        }
        Op::Attention { .. } | Op::Rope { .. } => hit(&["attn", "attention", "rope"]),
        Op::ScatterAdd | Op::TopK { .. } => hit(&["scatter", "topk"]),
        Op::Fft { .. } => hit(&["fft"]),
        Op::SelectiveScan { .. } => hit(&["scan", "selective_scan"]),
        _ => all,
    }
}

/// One scheduled step: either a GPU compute dispatch or a CPU host-fallback
/// op (for families with no native SPIR-V kernel yet).
#[derive(Clone)]
enum Step {
    /// Host-visible memcpy staging for cross-shard operands (sharded arenas).
    ActCopy {
        src_byte: usize,
        dst_byte: usize,
        bytes: usize,
    },
    Gpu {
        kernel: &'static str,
        push: Vec<u8>,
        groups: (u32, u32, u32),
        /// Activation descriptor set index (binding 0) for this dispatch.
        act_shard: u32,
    },
    Host {
        op: Op,
        out: NodeId,
        out_shape: rlx_ir::Shape,
        inputs: Vec<NodeId>,
    },
    /// A core Riemannian / SPD-manifold op (BiMap / ReEig / LogEig /
    /// SpdBatchNorm / SpdKarcherMean + backwards) evaluated on the CPU
    /// reference. Kept distinct from `Host` because the SPD kernels compute in
    /// F64 (`rlx_cpu::spd`) while the arena is f32 — the f32↔f64 widening lives
    /// in [`crate::spd::eval`], not the generic `host::eval` thunk path.
    SpdHost {
        op: Op,
        out: NodeId,
        out_shape: rlx_ir::Shape,
        inputs: Vec<NodeId>,
    },
    /// General `Op::Scan` via `execute_scan_host_desc` on the mapped arena.
    ScanHost { desc: rlx_cpu::thunk::ScanHostDesc },
    /// Nested-body AD (`ScanBackward` / `ScanBackwardXs`) via shared HostOpDesc.
    HostOp { desc: rlx_cpu::thunk::HostOpDesc },
}

/// A pre-recorded execution segment. The schedule is partitioned into maximal
/// runs of GPU dispatches (each recorded ONCE into a reusable command buffer at
/// compile time) separated by CPU host-fallback ops. At run time a GPU segment
/// is a single `queue_submit` of its prebuilt command buffer — no per-step
/// allocation, recording, or fence churn. See [`record_segments`].
enum Segment {
    /// A prebuilt command buffer covering a run of consecutive GPU dispatches.
    Gpu(vk::CommandBuffer),
    /// Cross-shard staging copy on the host-visible arena mapping.
    ActCopy {
        src_byte: usize,
        dst_byte: usize,
        bytes: usize,
    },
    /// A CPU host-fallback op, evaluated against the mapped arena between GPU
    /// segments (HOST_COHERENT memory, queue idle here — see `run_read_outputs`).
    Host {
        op: Op,
        out: NodeId,
        out_shape: rlx_ir::Shape,
        inputs: Vec<NodeId>,
    },
    /// A core SPD-manifold op evaluated on the CPU reference in F64 (see the
    /// [`Step::SpdHost`] note).
    SpdHost {
        op: Op,
        out: NodeId,
        out_shape: rlx_ir::Shape,
        inputs: Vec<NodeId>,
    },
    /// General `Op::Scan` on the mapped arena (see [`Step::ScanHost`]).
    ScanHost { desc: rlx_cpu::thunk::ScanHostDesc },
    /// Nested-body AD on the mapped arena (see [`Step::HostOp`]).
    HostOp { desc: rlx_cpu::thunk::HostOpDesc },
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
    /// Per-shard descriptor sets (binding 0 = activation shard, binding 1 = weights).
    act_desc_sets: Vec<vk::DescriptorSet>,
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

// ── memory plan (f32-uniform + liveness reuse; same as rlx-wgpu) ───────────
//
// The previous bump allocator summed every intermediate (~5 GiB for Kokoro's
// longer decoder), which exceeds Vulkan's typical `maxStorageBufferRange`
// (~4 GiB). Binding / addressing past that range yields silent zeros. Reuse
// drops the peak live set under the limit the same way wgpu does.

fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    rlx_compile::memory::plan_memory_f32_uniform(graph, align)
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

/// Number of f32 lanes a node occupies in the f32-uniform arena's host-readback
/// view. Complex is simulated on f32 lanes — C64 = 2 lanes/elem, C128 = 4 lanes
/// (df64); every OTHER dtype is exactly ONE f32 lane per element (I64/Bool/… are
/// widened to a single lane, so `size_bytes()` must NOT be blanket-applied here).
/// Used to read a complex output back with ALL its lanes rather than just
/// `num_elements` (which would truncate to the real parts).
fn arena_lane_count(shape: &rlx_ir::Shape) -> usize {
    let elems = shape.num_elements().unwrap_or(0);
    match shape.dtype() {
        DType::C64 => elems * 2,
        DType::C128 => elems * 4,
        _ => elems,
    }
}

/// Cast op ids for the `unary` shader (`unary.comp` cases 100–106). Kept in
/// sync with rlx-cuda / rlx-rocm (unary.cu) and rlx-oneapi (unary.cl).
const CAST_F32_TO_I8: u32 = 100;
const CAST_F32_TO_I16: u32 = 101;
const CAST_F32_TO_I32: u32 = 102;
const CAST_F32_TO_I64: u32 = 103;
const CAST_F32_TO_U8: u32 = 104;
const CAST_F32_TO_U32: u32 = 105;
const CAST_TO_BOOL: u32 = 106;
/// `unary` op id whose default branch is a value-preserving f32 copy.
const CAST_IDENTITY_COPY: u32 = 255;

/// How an `Op::Cast` lowers on the f32-uniform arena.
enum CastLower {
    /// Value-preserving relabel: a same-slot no-op, or (distinct slot) an
    /// identity f32 copy. Covers int→float, float→float (F16/BF16/F64 are all
    /// f32-stored here), int→int, bool→int/float, same-dtype.
    Identity,
    /// A real elementwise conversion via the `unary` shader with this op id
    /// (float→int trunc-saturate, or →Bool `x != 0`).
    Kernel(u32),
    /// A complex cast (real↔C64, real↔C128, C64↔C128) — pure f32-lane moves via
    /// the standalone `complex_cast` shader. Carries the mode (0..5, see
    /// `complex_cast.comp`). Needs its own (complex-sized) slot, not an alias.
    Complex(u32),
    /// Not representable in an f32 arena (an F64 real component has no lane
    /// storage here) — reject at lowering.
    Reject,
}

/// Classify a `Cast(src → dst)` on the f32-uniform arena. float→int truncates
/// toward zero + saturates (Rust `as` / rlx-cpu); →Bool is `x != 0`. F16/BF16/
/// F64 are demoted to f32 storage so casts to/from them are identity copies.
/// Complex casts (real↔C64, real↔C128, C64↔C128) are pure f32-lane moves on the
/// simulated-complex arena (C64 = 2 lanes/elem, C128 = 4 lanes df64); only a
/// complex cast touching the one non-lane-storable real component (F64) rejects.
fn classify_cast(src: DType, dst: DType) -> CastLower {
    if src == dst {
        return CastLower::Identity; // pure relabel (also covers C64→C64 / C128→C128)
    }
    if src.is_complex() || dst.is_complex() {
        // F64 is the one component type with no f32-lane storage here, so a
        // complex cast touching it (real side) is still rejected.
        if src == DType::F64 || dst == DType::F64 {
            return CastLower::Reject;
        }
        let mode = match (src, dst) {
            (s, DType::C64) if !s.is_complex() => 0,  // real → C64
            (DType::C64, d) if !d.is_complex() => 1,  // C64 → real
            (s, DType::C128) if !s.is_complex() => 2, // real → C128
            (DType::C128, d) if !d.is_complex() => 3, // C128 → real
            (DType::C64, DType::C128) => 4,
            (DType::C128, DType::C64) => 5,
            _ => return CastLower::Reject,
        };
        return CastLower::Complex(mode);
    }
    if dst == DType::Bool {
        return CastLower::Kernel(CAST_TO_BOOL);
    }
    if src.is_float() && dst.is_int() {
        return CastLower::Kernel(match dst {
            DType::I8 => CAST_F32_TO_I8,
            DType::I16 => CAST_F32_TO_I16,
            DType::I32 => CAST_F32_TO_I32,
            DType::I64 => CAST_F32_TO_I64,
            DType::U8 => CAST_F32_TO_U8,
            DType::U32 => CAST_F32_TO_U32,
            _ => unreachable!("is_int() covers all integer dtypes"),
        });
    }
    CastLower::Identity
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
        Some("tiled-unsafe") => "matmul_tiled", // opt-in for benching the raw tiled path
        _ if portability => "matmul",
        // The tiled kernel is exact for aligned K, but mishandles a *trailing
        // partial* K-tile (full tiles followed by a k%16 remainder), e.g.
        // 32x50x64 → max|Δ|~0.2 vs CPU while 32x48x64 is exact. Until that shader
        // bug is root-caused, route non-16-aligned-K shapes to the fully general,
        // bounds-checked (fp32-exact) scalar kernel. Aligned K keeps the fast path.
        _ if !k.is_multiple_of(16) => "matmul",
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
        Self::compile_rng_with_options(graph, rng, 64)
    }

    pub fn compile_rng_with_options(
        graph: Graph,
        rng: RngOptions,
        scan_unroll_max_length: u32,
    ) -> Self {
        use rlx_opt::pass::Pass as _;

        let graph = rlx_opt::LowerControlFlow.run(graph);
        // `FusedAttentionBlock` is claimed (so it legalizes), but there is
        // no monolithic fused-attention kernel — decompose it to primitives
        // first. FAB-only (not the whole-graph unfuse) so nothing else is
        // touched. No-op when no FAB node is present.
        let graph = rlx_opt::unfuse::unfuse_attention_block(graph);
        let graph = rlx_opt::unfuse::unfuse_dit_modulation(graph);
        let graph = rlx_opt::legalize_or_rewrite_for_backend(graph, SUPPORTED_OPS)
            .unwrap_or_else(|errs| panic!("{}", rlx_opt::format_legalize_error("vulkan", &errs)));
        let graph = rlx_cpu::rlx_maybe_unroll_scans!(graph, scan_unroll_max_length);
        let graph = rlx_opt::maybe_unroll_scans_budget(graph, 4096);
        // Materialize mid-axis broadcasts so Binary operands are equal-shaped
        // or trailing-broadcast (our kernels only do trailing modulus).
        let graph = rlx_opt::LegalizeBroadcast.run(graph);

        Self::build(graph, rng)
    }

    fn build(graph: Graph, rng: RngOptions) -> Self {
        let dev = vulkan_device().expect("rlx-vulkan: no device");
        let kern = kernels().expect("rlx-vulkan: no kernels");

        let plan = plan_f32_uniform(&graph, 16);
        let max_range = dev.limits.max_storage_buffer_range as usize;
        let arena = if plan.arena_size > max_range {
            Arena::from_plan_split(&graph)
        } else {
            Arena::from_plan(&plan)
        };

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
        if std::env::var("RLX_VULKAN_CHECK_CAST").as_deref() == Ok("1") {
            let mut bad = 0usize;
            let mut ok = 0usize;
            for node in graph.nodes() {
                if !matches!(node.op, Op::Cast { .. }) {
                    continue;
                }
                if node.inputs.is_empty() || !arena.has(node.id) || !arena.has(node.inputs[0]) {
                    continue;
                }
                let same = arena.byte_offset(node.id) == arena.byte_offset(node.inputs[0])
                    && arena.slot_elems(node.id) == arena.slot_elems(node.inputs[0]);
                let in_dt = graph.node(node.inputs[0]).shape.dtype();
                let out_dt = node.shape.dtype();
                if same {
                    ok += 1;
                } else {
                    bad += 1;
                    if bad <= 12 {
                        eprintln!(
                            "[rlx-vulkan] Cast NOT aliased: {:?} -> {:?} elems_in={} elems_out={} off_in={} off_out={}",
                            in_dt,
                            out_dt,
                            arena.slot_elems(node.inputs[0]),
                            arena.slot_elems(node.id),
                            arena.byte_offset(node.inputs[0]),
                            arena.byte_offset(node.id),
                        );
                    }
                }
            }
            eprintln!("[rlx-vulkan] Cast alias check: ok={ok} bad={bad}");
        }

        // Descriptor sets: binding 0 = activation shard(s), binding 1 = weights.
        let n_act_sets = arena.shard_count();
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count((n_act_sets * 2) as u32)];
        let desc_pool = unsafe {
            dev.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(n_act_sets as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .expect("vk descriptor_pool");
        let set_layouts = vec![kern.dsl; n_act_sets];
        let act_desc_sets = unsafe {
            dev.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .expect("vk descriptor_set");
        let weight_info = vk::DescriptorBufferInfo::default()
            .buffer(arena.weight_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let mut act_infos = Vec::with_capacity(n_act_sets);
        for i in 0..n_act_sets {
            act_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(arena.act_buffer(i))
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
            );
        }
        let mut writes = Vec::with_capacity(n_act_sets * 2);
        for (i, &set) in act_desc_sets.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&act_infos[i])),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&weight_info)),
            );
        }
        unsafe { dev.device.update_descriptor_sets(&writes, &[]) };

        // Pre-record the static schedule into reusable command buffers (one per
        // maximal GPU run). The whole schedule — kernels, push constants,
        // workgroup counts — is fixed at compile time; per-step inputs are
        // memcpy'd into the host-visible arena, never the command stream. So a
        // single recording is valid for every `run`, turning each step into one
        // `queue_submit` instead of allocate → record → fence → free.
        let cached = std::env::var("RLX_VULKAN_NOCACHE").as_deref() != Ok("1");
        let (segments, fence) = if cached {
            let segs = record_segments(dev, kern, &act_desc_sets, &schedule, &deps);
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
            act_desc_sets,
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
        let base = self
            .arena
            .elem_offset(id)
            .wrapping_add((row * row_inner) as u32);
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
            let base = self
                .arena
                .elem_offset(out_id)
                .wrapping_add((row * row_inner) as u32);
            return Some(self.arena.read_f32_at_elem(base, row_inner));
        }
        if self.gpu_handle_resident.contains(name)
            && let Some(&id) = self.input_ids.get(name)
        {
            let base = self
                .arena
                .elem_offset(id)
                .wrapping_add((row * row_inner) as u32);
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
        let desc_sets = &self.act_desc_sets;
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
                        // Discrete GPUs + host ConvTranspose: make shader
                        // writes visible before the CPU reads the arena.
                        self.arena.sync_host_after_gpu();
                    }
                    Segment::ActCopy {
                        src_byte,
                        dst_byte,
                        bytes,
                    } => {
                        self.arena.copy_bytes_range(*src_byte, *dst_byte, *bytes);
                        self.arena.sync_gpu_after_host();
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
                        match crate::host::eval(op, out_shape, &in_specs) {
                            crate::host::HostOut::F32(v) => self.arena.write_f32(*out, &v),
                            crate::host::HostOut::Bytes(b) => self.arena.write_bytes(*out, &b),
                        }
                        self.arena.sync_gpu_after_host();
                    }
                    Segment::SpdHost {
                        op,
                        out,
                        out_shape,
                        inputs: in_ids,
                    } => {
                        self.run_spd_host(op, *out, out_shape, in_ids);
                        self.arena.sync_gpu_after_host();
                    }
                    Segment::ScanHost { desc } => unsafe {
                        rlx_cpu::rlx_execute_scan_on_bytes!(self.arena.mapped_ptr(), desc);
                        self.arena.sync_gpu_after_host();
                    },
                    Segment::HostOp { desc } => unsafe {
                        rlx_cpu::rlx_execute_host_op_on_bytes!(self.arena.mapped_ptr(), desc);
                        self.arena.sync_gpu_after_host();
                    },
                }
            }
            // Fall through to the feed/readback tail below.
            return self.finish_run(read_indices);
        }

        let n = self.schedule.len();
        let mut i = 0;
        while i < n {
            if let Step::ActCopy {
                src_byte,
                dst_byte,
                bytes,
            } = &self.schedule[i]
            {
                self.arena.copy_bytes_range(*src_byte, *dst_byte, *bytes);
                self.arena.sync_gpu_after_host();
                i += 1;
                continue;
            }
            let start = i;
            while i < n && matches!(self.schedule[i], Step::Gpu { .. }) {
                i += 1;
            }
            if i > start {
                let gpu = self.schedule[start..i].to_vec();
                dev.submit_and_wait(|cmd| unsafe {
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
                            act_shard,
                        } = step
                        {
                            let desc_set = desc_sets
                                .get(*act_shard as usize)
                                .copied()
                                .unwrap_or(desc_sets[0]);
                            dev.device.cmd_bind_descriptor_sets(
                                cmd,
                                vk::PipelineBindPoint::COMPUTE,
                                layout,
                                0,
                                &[desc_set],
                                &[],
                            );
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
                self.arena.sync_host_after_gpu();
            }
            if i < n {
                match self.schedule[i].clone() {
                    Step::ActCopy { .. } => {}
                    Step::Host {
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
                        match crate::host::eval(&op, &out_shape, &in_specs) {
                            crate::host::HostOut::F32(v) => self.arena.write_f32(out, &v),
                            crate::host::HostOut::Bytes(b) => self.arena.write_bytes(out, &b),
                        }
                        self.arena.sync_gpu_after_host();
                    }
                    Step::SpdHost {
                        op,
                        out,
                        out_shape,
                        inputs: in_ids,
                    } => {
                        self.run_spd_host(&op, out, &out_shape, &in_ids);
                        self.arena.sync_gpu_after_host();
                    }
                    Step::ScanHost { desc } => unsafe {
                        rlx_cpu::rlx_execute_scan_on_bytes!(self.arena.mapped_ptr(), &desc);
                        self.arena.sync_gpu_after_host();
                    },
                    Step::HostOp { desc } => unsafe {
                        rlx_cpu::rlx_execute_host_op_on_bytes!(self.arena.mapped_ptr(), &desc);
                        self.arena.sync_gpu_after_host();
                    },
                    Step::Gpu { .. } => {}
                }
                i += 1;
            }
        }

        self.finish_run(read_indices)
    }

    /// Evaluate one SPD-manifold op (BiMap / ReEig / … + backwards) on the CPU
    /// reference against the mapped arena. Reads each operand as f32, hands the
    /// f32 buffers + declared shapes to [`crate::spd::eval`] (which widens to
    /// F64, calls `rlx_cpu::spd`, and narrows back), then writes the f32 result
    /// into the output slot. The queue is idle here (this runs between GPU
    /// segments), and the arena is HOST_COHERENT, so the plain mapped I/O is
    /// visible to the next dispatch.
    fn run_spd_host(&self, op: &Op, out: NodeId, out_shape: &rlx_ir::Shape, in_ids: &[NodeId]) {
        let inputs: Vec<(rlx_ir::Shape, Vec<f32>)> = in_ids
            .iter()
            .map(|&id| {
                let sh = self.graph.node(id).shape.clone();
                let nn = sh.num_elements().unwrap_or(0);
                (sh, self.arena.read_f32(id, nn))
            })
            .collect();
        let y = crate::spd::eval(op, out_shape, &inputs);
        self.arena.write_f32(out, &y);
    }

    /// Shared post-execution tail for both the cached and legacy run paths: fold
    /// fed outputs (new-token K/V) back into their handle input slots in-place on
    /// the arena — the queue is idle here so the mapped memory is coherent. When
    /// all outputs are read back, also refresh host mirrors; for logits-only
    /// decode (`read_indices == Some([0])`) the K/V never leaves the arena, which
    /// is the whole point. Then read the requested outputs.
    fn finish_run(&mut self, read_indices: Option<&[usize]>) -> Vec<Vec<f32>> {
        // Last GPU segment may have left the host mapping stale.
        self.arena.sync_host_after_gpu();
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
        // NaN/Inf output-boundary scan (RLX_DEBUG_NANS). GPU segments are
        // pre-recorded command buffers with no per-op host boundary; the arena
        // is HOST_COHERENT so reading outputs back is free. Scan them here and
        // point provenance at the offending output node. Host-fallback ops that
        // run on the CPU can be localized per-op by replaying on the CPU backend.
        let scanner = rlx_ir::numeric_check::DebugScanner::from_env("vulkan");
        // Full-graph culprit scan: read EVERY node's slot (the arena is
        // HOST_COHERENT and has no slot reuse) with its real operands, so the
        // first node whose output is non-finite while its inputs are finite is
        // the true source (Inf as well as NaN). The default output-only scan
        // passes empty inputs, mislabelling every non-finite output a "culprit".
        if scanner.enabled() && std::env::var("RLX_VULKAN_SCAN_ALL").is_ok() {
            for node in self.graph.nodes() {
                let id = node.id;
                let n = node.shape.num_elements().unwrap_or(0);
                if n == 0 || !self.arena.has(id) {
                    continue;
                }
                let out = self.arena.read_f32(id, n);
                let ins: Vec<(NodeId, Vec<f32>)> = node
                    .inputs
                    .iter()
                    .filter_map(|&inp| {
                        let m = self.graph.node(inp).shape.num_elements().unwrap_or(0);
                        if m == 0 || !self.arena.has(inp) {
                            return None;
                        }
                        Some((inp, self.arena.read_f32(inp, m)))
                    })
                    .collect();
                let refs: Vec<(NodeId, &[f32])> =
                    ins.iter().map(|(i, b)| (*i, b.as_slice())).collect();
                if scanner.check(&self.graph, id, &out, &refs).is_some() {
                    break;
                }
            }
        }
        want.into_iter()
            .filter_map(|i| {
                let id = *self.output_ids.get(i)?;
                // Lane count, not element count: a complex output occupies 2 (C64)
                // / 4 (C128) f32 lanes per element, so reading `num_elements`
                // would truncate the readback to the real parts (and a slot-sizing
                // regression would surface here as a short buffer). Every other
                // dtype is one lane per element, so this is `num_elements` there.
                let n = arena_lane_count(&self.graph.node(id).shape);
                let buf = self.arena.read_f32(id, n);
                if scanner.enabled() {
                    scanner.check(&self.graph, id, &buf, &[]);
                }
                Some(buf)
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
                    Segment::Host { .. }
                    | Segment::SpdHost { .. }
                    | Segment::ScanHost { .. }
                    | Segment::HostOp { .. }
                    | Segment::ActCopy { .. } => None,
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
    act_desc_sets: &[vk::DescriptorSet],
    schedule: &[Step],
    deps: &[StepDep],
) -> Vec<Segment> {
    let layout = kern.pipeline_layout;
    let no_barrier = std::env::var("RLX_VULKAN_NOBARRIER").as_deref() == Ok("1");
    let full_barrier = std::env::var("RLX_VULKAN_FULLBARRIER").as_deref() == Ok("1");
    let mut segments = Vec::new();
    let n = schedule.len();
    let mut i = 0;
    let mut dep_i = 0usize;
    while i < n {
        if matches!(schedule[i], Step::ActCopy { .. }) {
            if let Step::ActCopy {
                src_byte,
                dst_byte,
                bytes,
            } = &schedule[i]
            {
                segments.push(Segment::ActCopy {
                    src_byte: *src_byte,
                    dst_byte: *dst_byte,
                    bytes: *bytes,
                });
            }
            i += 1;
            dep_i += 1;
            continue;
        }
        let start = i;
        while i < n && matches!(schedule[i], Step::Gpu { .. }) {
            i += 1;
        }
        if i > start {
            let run = &schedule[start..i];
            let run_deps = &deps[dep_i..dep_i + run.len()];
            dep_i += run.len();
            let cmd = dev.alloc_primary_cmd();
            unsafe {
                dev.device
                    .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                    .expect("vk begin cmd");
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                let mut wrote: Vec<SlotSpan> = Vec::new();
                let mut read: Vec<SlotSpan> = Vec::new();
                for (j, step) in run.iter().enumerate() {
                    if let Step::Gpu {
                        kernel,
                        push,
                        groups,
                        act_shard,
                    } = step
                    {
                        let dep = &run_deps[j];
                        let raw = dep
                            .reads
                            .iter()
                            .any(|r| wrote.iter().any(|w| r.overlaps(*w)));
                        let waw = wrote.iter().any(|w| dep.write.overlaps(*w));
                        let war = read.iter().any(|r| dep.write.overlaps(*r));
                        let hazard = raw || waw || war;
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
                        let desc_set = act_desc_sets
                            .get(*act_shard as usize)
                            .copied()
                            .unwrap_or(act_desc_sets[0]);
                        dev.device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            layout,
                            0,
                            &[desc_set],
                            &[],
                        );
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
                        if dep.write.len > 0 {
                            wrote.push(dep.write);
                        }
                        for &r in &dep.reads {
                            if r.len > 0 {
                                read.push(r);
                            }
                        }
                    }
                }
                dev.device.end_command_buffer(cmd).expect("vk end cmd");
            }
            segments.push(Segment::Gpu(cmd));
        }
        if i < n {
            match &schedule[i] {
                Step::ActCopy { .. } => {}
                Step::Host {
                    op,
                    out,
                    out_shape,
                    inputs,
                } => segments.push(Segment::Host {
                    op: op.clone(),
                    out: *out,
                    out_shape: out_shape.clone(),
                    inputs: inputs.clone(),
                }),
                Step::SpdHost {
                    op,
                    out,
                    out_shape,
                    inputs,
                } => segments.push(Segment::SpdHost {
                    op: op.clone(),
                    out: *out,
                    out_shape: out_shape.clone(),
                    inputs: inputs.clone(),
                }),
                Step::ScanHost { desc } => segments.push(Segment::ScanHost { desc: desc.clone() }),
                Step::HostOp { desc } => segments.push(Segment::HostOp { desc: desc.clone() }),
                Step::Gpu { .. } => {}
            }
            if !matches!(schedule[i], Step::ActCopy { .. }) {
                dep_i += 1;
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
        // C64 = 2 interleaved f32 lanes `[re, im]`; the host already stores it as
        // f32 pairs, so widening is a pure reinterpret (N complex → 2N lanes).
        DType::C64 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // C128 = 4 f32 lanes df64 `[re_hi, re_lo, im_hi, im_lo]`, host-stored as
        // 2×f64 (16 B/elem). This is the df64 SPLIT boundary: each f64 `v` →
        // `hi=(f32)v` + `lo=(f32)(v-(f64)hi)`, so `(f64)hi+(f64)lo` reconstructs
        // `v` to double precision. Bit-identical to the shared
        // `rlx_runtime::backend::widen_bytes_to_f32` (the CPU↔GPU boundary the
        // complex-cast kernels round-trip against).
        DType::C128 => {
            let split = |v: f64| -> [f32; 2] {
                let hi = v as f32;
                let lo = (v - hi as f64) as f32;
                [hi, lo]
            };
            let mut out = Vec::with_capacity((data.len() / 16) * 4);
            for elem in data.chunks_exact(16) {
                let re = f64::from_le_bytes(elem[0..8].try_into().unwrap());
                let im = f64::from_le_bytes(elem[8..16].try_into().unwrap());
                out.extend_from_slice(&split(re));
                out.extend_from_slice(&split(im));
            }
            out
        }
    }
}

// ── schedule construction ──────────────────────────────────────────────────

/// Per-GPU-op activation shard binding + cross-shard staging (mirrors rlx-wgpu).
struct ActBinder {
    bind_shard: u32,
    bind_base: usize,
    scratch: usize,
    begun: bool,
    pending_copies: Vec<(usize, usize, usize)>,
}

impl ActBinder {
    fn new() -> Self {
        Self {
            bind_shard: 0,
            bind_base: 0,
            scratch: 0,
            begun: false,
            pending_copies: Vec::new(),
        }
    }

    fn drain_copies(&mut self, steps: &mut Vec<Step>, deps: &mut Vec<StepDep>) {
        for (src_byte, dst_byte, bytes) in self.pending_copies.drain(..) {
            steps.push(Step::ActCopy {
                src_byte,
                dst_byte,
                bytes,
            });
            deps.push(StepDep::default());
        }
    }

    fn reset_op(&mut self) {
        self.bind_shard = 0;
        self.bind_base = 0;
        self.scratch = 0;
        self.begun = false;
        self.pending_copies.clear();
    }

    fn act_shard(&self) -> u32 {
        self.bind_shard
    }

    fn begin_op(&mut self, arena: &Arena, act_ids: &[NodeId]) {
        if !arena.is_sharded() {
            self.bind_shard = 0;
            self.begun = true;
            return;
        }
        let s = arena.shard_size;
        let pick = act_ids
            .iter()
            .find(|&&id| !arena.is_weight_node(id))
            .map(|&id| arena.byte_offset(id))
            .unwrap_or(0);
        self.bind_shard = (pick / s) as u32;
        self.bind_base = self.bind_shard as usize * s;
        self.scratch = arena.shard_stage_off(pick);
        self.begun = true;
    }

    fn off(&mut self, arena: &Arena, id: NodeId) -> u32 {
        let raw = arena.elem_offset(id);
        if is_weight_elem(raw) {
            return raw;
        }
        if !arena.is_sharded() {
            return raw;
        }
        if !self.begun {
            self.begin_op(arena, &[id]);
        }
        let byte = arena.byte_offset(id);
        let len = arena.byte_len(id).max(4);
        let s = arena.shard_size;
        let end = byte.saturating_add(len);
        let shard_lo = byte / s;
        let shard_hi = end.saturating_sub(1) / s;
        if shard_lo == self.bind_shard as usize && shard_hi == self.bind_shard as usize {
            let rebase_elem = (self.bind_base / 4) as u32;
            return raw_elem_off(raw).saturating_sub(rebase_elem);
        }
        self.stage_activation(arena, id, byte, len)
    }

    fn stage_activation(&mut self, arena: &Arena, id: NodeId, byte: usize, len: usize) -> u32 {
        if len > SHARD_STAGE_RESERVE {
            eprintln!(
                "[rlx-vulkan] cross-shard staging: tensor {id:?} is {len} bytes \
                 (>{SHARD_STAGE_RESERVE} reserve); cannot run this op on sharded arena"
            );
            panic!(
                "rlx-vulkan: cannot stage {len} bytes for {id:?} across activation shards \
                 (reserve {SHARD_STAGE_RESERVE})"
            );
        }
        let stage_begin = arena.shard_stage_off(self.bind_base);
        let stage_end = stage_begin.saturating_add(SHARD_STAGE_RESERVE);
        let aligned = len.div_ceil(256) * 256;
        if self.scratch.saturating_add(aligned) > stage_end {
            self.scratch = stage_begin;
        }
        let dst = self.scratch;
        self.scratch = self.scratch.saturating_add(aligned);
        self.pending_copies.push((byte, dst, len));
        ((dst.saturating_sub(self.bind_base)) / 4) as u32
    }
}

/// Half-open arena span `[start, start+len)` in f32 elements.
#[derive(Clone, Copy, Debug, Default)]
struct SlotSpan {
    start: u32,
    len: u32,
}

impl SlotSpan {
    fn overlaps(self, other: SlotSpan) -> bool {
        let self_w = is_weight_elem(self.start);
        let other_w = is_weight_elem(other.start);
        if self_w != other_w {
            return false;
        }
        let a = crate::buffer::raw_elem_off(self.start);
        let b = crate::buffer::raw_elem_off(other.start);
        self.len > 0
            && other.len > 0
            && a < b.saturating_add(other.len)
            && b < a.saturating_add(self.len)
    }
}

/// Per-GPU-step memory footprint, used to place barriers only where a real data
/// hazard exists. Arena slots are assigned by liveness reuse (`plan_memory_f32_uniform`),
/// so distinct nodes may share storage across non-overlapping lifetimes, and a
/// free-list pack can place a later tensor inside a previously larger span.
/// Hazards therefore compare **byte/element ranges**, not base offsets alone —
/// base-only tracking misses RAW/WAR when spans overlap without sharing a start.
#[derive(Clone, Default)]
struct StepDep {
    reads: Vec<SlotSpan>,
    write: SlotSpan,
}

fn push_gpu_step(
    binder: &mut ActBinder,
    steps: &mut Vec<Step>,
    deps: &mut Vec<StepDep>,
    kernel: &'static str,
    push: Vec<u8>,
    groups: (u32, u32, u32),
) {
    binder.drain_copies(steps, deps);
    steps.push(Step::Gpu {
        kernel,
        push,
        groups,
        act_shard: binder.act_shard(),
    });
}

/// Build the dispatch schedule plus, in lockstep, the per-step dependency info
/// (`StepDep`) that [`record_segments`] uses to elide redundant barriers. Each
/// graph node contributes its node-level footprint to every `Step` it emits
/// (most nodes emit one; `Concat` emits one per input — conservatively sharing
/// the node footprint, which over-serializes only a concat's own sub-copies).
fn build_schedule(graph: &Graph, arena: &Arena) -> (Vec<Step>, Vec<StepDep>) {
    let mut steps = Vec::new();
    let mut deps: Vec<StepDep> = Vec::new();
    let mut binder = ActBinder::new();
    if std::env::var("RLX_VULKAN_DUMP_OPS").as_deref() == Ok("1") {
        let mut hist: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut host_n = 0usize;
        for node in graph.nodes() {
            let k = match &node.op {
                Op::Input { .. }
                | Op::Param { .. }
                | Op::Constant { .. }
                | Op::Reshape { .. }
                | Op::Cast { .. }
                | Op::StopGradient => continue,
                Op::Binary(_) => "binary",
                Op::Compare(_) => "compare",
                Op::Where => "where",
                Op::Activation(_) => "unary",
                Op::MatMul | Op::GroupedMatMul => "matmul",
                Op::Conv { .. } => "conv2d",
                Op::ConvTranspose2d { .. } => "conv_transpose(host)",
                Op::Transpose { .. }
                | Op::Narrow { .. }
                | Op::Expand { .. }
                | Op::Concat { .. } => "reindex",
                Op::Gather { .. } => "gather",
                Op::Reduce { .. } => "reduce",
                Op::Softmax { .. } => "softmax",
                Op::RmsNorm { .. } | Op::LayerNorm { .. } | Op::LayerNorm2d { .. } => "norm",
                Op::Fft { .. } => "fft(host)",
                Op::Cumsum { .. } => "cumsum",
                Op::Pool { .. } => "pool",
                Op::Im2Col { .. } => "im2col",
                Op::Attention { .. } => "attention",
                Op::Rope { .. } => "rope",
                Op::ScatterAdd => "scatter_add",
                other => {
                    if is_host_fallback(other) {
                        host_n += 1;
                        "other_host"
                    } else {
                        "other_gpu"
                    }
                }
            };
            if k.contains("host") {
                host_n += 1;
            }
            *hist.entry(k).or_default() += 1;
        }
        eprintln!("[rlx-vulkan] op hist ({host_n} host-ish): {hist:?}");
    }
    for node in graph.nodes() {
        binder.reset_op();
        let out = node.id;
        let before = steps.len();
        match &node.op {
            // Leaves / view aliases — no dispatch when the planner colocated
            // the output with its parent. `Cast` is handled below: same-slot is
            // a no-op, but Bool→F32 (and other dtype-changing casts) get their
            // own slot and must copy (see wgpu's Cast lowering — silent zeros
            // otherwise kill TTS masks / Kokoro decoder).
            Op::Input { .. }
            | Op::Param { .. }
            | Op::Constant { .. }
            | Op::Reshape { .. }
            | Op::StopGradient => {}

            Op::Cast { to } => {
                let x = node.inputs[0];
                // Same-slot (planner-aliased same-dtype view) is a pure no-op.
                // A distinct out slot means the planner saw a dtype-changing
                // cast. Complex casts (real↔C64, real↔C128, C64↔C128) route to
                // the standalone `complex_cast` shader (pure f32-lane moves,
                // dispatched over the complex-element index — the fused/unary
                // scalar-per-lane path can't re-pair `[re, im]` lanes). Every
                // other cast emits the `unary` shader: an identity f32 copy
                // (255) for value-preserving casts, or a real conversion
                // (float→int trunc-saturate, →Bool).
                if arena.has(out) && arena.has(x) && arena.byte_offset(out) != arena.byte_offset(x)
                {
                    let src_dtype = graph.node(x).shape.dtype();
                    if let CastLower::Complex(mode) = classify_cast(src_dtype, *to) {
                        // `n` is the complex-element count (identical for src and
                        // dst — a cast preserves element count, only the lane
                        // width changes). The kernel reads/writes the right lanes
                        // per `mode`. Falls through to the node footprint attach
                        // (the standalone complex slot is sized 2N/4N lanes by the
                        // planner, so the barrier span it records is correct).
                        let n = numel(&dims(graph, out));
                        let push = Push::default()
                            .u(n as u32)
                            .u(binder.off(arena, x)) // in_off (f32-lane start)
                            .u(binder.off(arena, out)) // out_off
                            .u(mode)
                            .bytes();
                        push_gpu_step(
                            &mut binder,
                            &mut steps,
                            &mut deps,
                            "complex_cast",
                            push,
                            groups1d(n, 256),
                        );
                    } else {
                        let op = match classify_cast(src_dtype, *to) {
                            CastLower::Identity => CAST_IDENTITY_COPY,
                            CastLower::Kernel(op) => op,
                            CastLower::Complex(_) => unreachable!("handled above"),
                            CastLower::Reject => panic!(
                                "rlx-vulkan: Cast {src_dtype:?} → {to:?} involves an F64 \
                                 real component, which has no f32-lane storage in the \
                                 uniform arena — run this cast on CPU"
                            ),
                        };
                        let n = numel(&dims(graph, out))
                            .min(arena.slot_elems(out))
                            .min(arena.slot_elems(x));
                        let push = Push::default()
                            .u(n as u32)
                            .u(binder.off(arena, x))
                            .u(binder.off(arena, out))
                            .u(op)
                            .bytes();
                        push_gpu_step(
                            &mut binder,
                            &mut steps,
                            &mut deps,
                            "unary",
                            push,
                            groups1d(n, 256),
                        );
                    }
                }
            }

            // Diagnosis: force GPU families onto the host fallback first so
            // dedicated arms below do not take precedence.
            op if host_ops_forced(op) => {
                steps.push(Step::Host {
                    op: node.op.clone(),
                    out: node.id,
                    out_shape: node.shape.clone(),
                    inputs: node.inputs.clone(),
                });
            }

            Op::Binary(op) if node.shape.dtype().is_complex() => {
                // C64 add/sub/mul/div reads BOTH `[re, im]` lanes per element, so
                // it cannot ride the scalar-per-thread `binary` kernel — lower to
                // a standalone `binary_c64` dispatch over the complex-element
                // index. C128 arithmetic is out of scope (rlx-cpu has none
                // either) → reject; C64 max/min/pow are undefined for complex →
                // reject (matches the CPU). Broadcast is carried by per-operand
                // complex-element counts (`k % n_x`), matching the CPU modulo
                // fallback. Mirrors rlx-wgpu / rlx-cuda binary_c64 lowering.
                let out_dt = node.shape.dtype();
                if out_dt == DType::C128 {
                    panic!(
                        "rlx-vulkan: Binary on C128: complex-f64 arithmetic is \
                         unsupported (rlx-cpu has none either) — only C64 \
                         add/sub/mul/div are wired"
                    );
                }
                let op_code = binop_id(*op);
                if op_code > 3 {
                    panic!(
                        "rlx-vulkan: C64 Binary: {op:?} is undefined for complex \
                         (only Add/Sub/Mul/Div); matches rlx-cpu rejection"
                    );
                }
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(graph, out)); // output complex-element count
                let na = numel(&dims(graph, a));
                let nb = numel(&dims(graph, b));
                let push = Push::default()
                    .u(n as u32)
                    .u(binder.off(arena, a)) // a_off (f32-lane start)
                    .u(binder.off(arena, b)) // b_off
                    .u(binder.off(arena, out)) // c_off
                    .u(op_code)
                    .u(na.max(1) as u32) // n_a (broadcast, complex-element units)
                    .u(nb.max(1) as u32) // n_b
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "binary_c64",
                    push,
                    groups1d(n, 256),
                );
            }

            Op::Binary(op) => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(graph, out));
                let an = numel(&dims(graph, a));
                let bn = numel(&dims(graph, b));
                // Trailing-broadcast check: a (or b) must equal n or evenly
                // tile n. Mid-axis broadcasts that slip past LegalizeBroadcast
                // silently corrupt TTS decoders (Kokoro cos≈0.09).
                let trailing_ok = |m: usize| m == 0 || m == n || n.is_multiple_of(m);
                if std::env::var("RLX_VULKAN_CHECK_BCAST").as_deref() == Ok("1")
                    && (!trailing_ok(an) || !trailing_ok(bn))
                {
                    eprintln!(
                        "[rlx-vulkan] non-trailing Binary {:?} out_n={n} a_n={an} b_n={bn} \
                         a_dims={:?} b_dims={:?} out_dims={:?}",
                        op,
                        dims(graph, a),
                        dims(graph, b),
                        dims(graph, out)
                    );
                }
                let push = Push::default()
                    .u(n as u32)
                    .u(binder.off(arena, a))
                    .u(binder.off(arena, b))
                    .u(binder.off(arena, out))
                    .u(if an == n { 0 } else { an as u32 })
                    .u(if bn == n { 0 } else { bn as u32 })
                    .u(binop_id(*op))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "binary",
                    push,
                    groups1d(n, 256),
                );
            }

            Op::Compare(op) => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let n = numel(&dims(graph, out));
                let an = numel(&dims(graph, a));
                let bn = numel(&dims(graph, b));
                let push = Push::default()
                    .u(n as u32)
                    .u(binder.off(arena, a))
                    .u(binder.off(arena, b))
                    .u(binder.off(arena, out))
                    .u(if an == n { 0 } else { an as u32 })
                    .u(if bn == n { 0 } else { bn as u32 })
                    .u(cmp_id(*op))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "compare",
                    push,
                    groups1d(n, 256),
                );
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
                    .u(binder.off(arena, c))
                    .u(binder.off(arena, a))
                    .u(binder.off(arena, b))
                    .u(binder.off(arena, out))
                    .u(if cn == n { 0 } else { cn as u32 })
                    .u(if an == n { 0 } else { an as u32 })
                    .u(if bn == n { 0 } else { bn as u32 })
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "where",
                    push,
                    groups1d(n, 256),
                );
            }

            Op::Activation(act) => {
                let x = node.inputs[0];
                let n = numel(&dims(graph, out));
                let push = Push::default()
                    .u(n as u32)
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .u(act_id(*act))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "unary",
                    push,
                    groups1d(n, 256),
                );
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
                    .u(binder.off(arena, a))
                    .u(binder.off(arena, b))
                    .u(binder.off(arena, out))
                    .u(batch as u32)
                    .u(a_bs as u32)
                    .u(b_bs as u32)
                    .u((m * n) as u32)
                    .bytes();
                let kernel = if is_weight_elem(binder.off(arena, a))
                    || is_weight_elem(binder.off(arena, b))
                {
                    "matmul"
                } else {
                    matmul_kernel(m, k, n)
                };
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    kernel,
                    push,
                    (ceil_div(n, 16), ceil_div(m, 16), batch.max(1) as u32),
                );
            }

            Op::Reduce { .. } => {
                // SPIR-V reduce is last-axis only and had no legalize for other
                // axes; host every Reduce until a general kernel / rewrite lands
                // (Supertonic: non-last Reduce → cos≈0.24).
                steps.push(Step::Host {
                    op: node.op.clone(),
                    out: node.id,
                    out_shape: node.shape.clone(),
                    inputs: node.inputs.clone(),
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "softmax",
                    push,
                    groups1d(outer * inner, 256),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, gamma))
                    .u(binder.off(arena, beta))
                    .u(binder.off(arena, out))
                    .f(*eps)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "rmsnorm",
                    push,
                    groups1d(rows, 64),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, gamma))
                    .u(binder.off(arena, beta))
                    .u(binder.off(arena, out))
                    .u(if has_beta { 1 } else { 0 })
                    .f(*eps)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "layernorm",
                    push,
                    groups1d(rows, 64),
                );
            }

            Op::AdaLayerNormBackward { norm, eps } => {
                use rlx_ir::ada_modulation_launch;
                use rlx_ir::op::AdaNormKind;
                let x = node.inputs[0];
                let scale = node.inputs[1];
                // inputs[2] = shift (unused in reverse kernel; same shape as scale)
                let dy = node.inputs[3];
                let x_dims = dims(graph, x);
                let mod_dims = dims(graph, scale);
                let inner = *x_dims.last().unwrap_or(&1) as u32;
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &mod_dims);
                let layer_norm = matches!(norm, AdaNormKind::LayerNorm) as u32;
                let push = Push::default()
                    .u(mod_rows)
                    .u(seq_per_mod)
                    .u(inner)
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, scale))
                    .u(binder.off(arena, dy))
                    .u(binder.off(arena, out))
                    .u(layer_norm)
                    .f(*eps)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "ada_layer_norm_backward",
                    push,
                    groups1d(mod_rows as usize, 64),
                );
            }

            Op::GatedResidualBackward => {
                use rlx_ir::ada_modulation_launch;
                let _x = node.inputs[0];
                let y = node.inputs[1];
                let gate = node.inputs[2];
                let dy = node.inputs[3];
                let x_dims = dims(graph, dy);
                let gate_dims = dims(graph, gate);
                let inner = *x_dims.last().unwrap_or(&1) as u32;
                let (mod_rows, seq_per_mod) = ada_modulation_launch(&x_dims, &gate_dims);
                let push = Push::default()
                    .u(mod_rows)
                    .u(seq_per_mod)
                    .u(inner)
                    .u(binder.off(arena, y))
                    .u(binder.off(arena, gate))
                    .u(binder.off(arena, dy))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "gated_residual_backward",
                    push,
                    groups1d(mod_rows as usize, 64),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, cos))
                    .u(binder.off(arena, sin))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "rope",
                    push,
                    groups1d(batch * seq * nh, 64),
                );
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
                    MaskKind::Custom => (3, binder.off(arena, node.inputs[3]), 0),
                    MaskKind::Bias => (4, binder.off(arena, node.inputs[3]), 0),
                };
                let scale = score_scale.unwrap_or((dh as f32).powf(-0.5));
                let push = Push::default()
                    .u(batch as u32)
                    .u(nh as u32)
                    .u(q_s as u32)
                    .u(k_s as u32)
                    .u(dh as u32)
                    .u(binder.off(arena, q))
                    .u(binder.off(arena, k))
                    .u(binder.off(arena, v))
                    .u(binder.off(arena, out))
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
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "attention",
                    push,
                    groups1d(batch * nh * q_s, 64),
                );
            }

            Op::Transpose { perm } => {
                let x = node.inputs[0];
                let mut xd = dims(graph, x);
                let mut od = dims(graph, out);
                // Complex tensors pack `lanes` contiguous f32 per element
                // (C64=2 [re,im], C128=4 df64). The `reindex` kernel copies one
                // f32 per "element" via reindexed strides, so append an INNERMOST
                // lane axis to input+output dims and extend the permutation so the
                // lane axis maps to ITSELF (never permuted). `contig_strides` then
                // yields lane-unit strides on the element axes and stride-1 on the
                // lane axis in BOTH istr/ostr, so each thread copies a whole
                // complex element's lanes as a group instead of shattering [re,im].
                // lanes=1 (real/int) ⇒ dims/perm/rank unchanged ⇒ strict no-op.
                let lanes: usize = match node.shape.dtype() {
                    DType::C64 => 2,
                    DType::C128 => 4,
                    _ => 1,
                };
                let mut perm: Vec<usize> = perm.to_vec();
                if lanes > 1 {
                    // The fixed [u32; 6] push arrays cap rank at 6; the appended
                    // lane axis therefore limits complex Transpose to element-rank
                    // ≤ 5. Guard rather than overflow.
                    assert!(
                        od.len() < 6,
                        "rlx-vulkan: complex Transpose element-rank {} exceeds 5 \
                         (lane axis + [u32; 6] cap)",
                        od.len()
                    );
                    // Lane axis is input axis `xd.len()` (appended innermost);
                    // it maps to itself as the output's innermost axis.
                    let lane_ax = xd.len();
                    xd.push(lanes);
                    od.push(lanes);
                    perm.push(lane_ax);
                }
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .us(&shape)
                    .us(&istr)
                    .us(&ostr)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "reindex",
                    push,
                    groups1d(n, 256),
                );
            }

            Op::Narrow { axis, start, .. } => {
                let x = node.inputs[0];
                let mut xd = dims(graph, x);
                let mut od = dims(graph, out);
                // Complex packs `lanes` contiguous f32 per element; the `reindex`
                // kernel copies one f32 per "element". Append an INNERMOST lane
                // axis to input+output dims so `contig_strides`/`numel` follow in
                // lane units: `axis`/`start` stay element-indexed, but `in_str
                // [*axis]` is now lane-scaled so the source offset lands on the
                // right complex element. lanes=1 ⇒ dims/rank unchanged ⇒ no-op.
                let lanes: usize = match node.shape.dtype() {
                    DType::C64 => 2,
                    DType::C128 => 4,
                    _ => 1,
                };
                if lanes > 1 {
                    assert!(
                        od.len() < 6,
                        "rlx-vulkan: complex Narrow element-rank {} exceeds 5 \
                         (lane axis + [u32; 6] cap)",
                        od.len()
                    );
                    xd.push(lanes);
                    od.push(lanes);
                }
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
                let in_off = binder.off(arena, x) + (*start * in_str[*axis]) as u32;
                let n = numel(&od);
                let push = Push::default()
                    .u(n as u32)
                    .u(rank as u32)
                    .u(in_off)
                    .u(binder.off(arena, out))
                    .us(&shape)
                    .us(&istr)
                    .us(&ostr)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "reindex",
                    push,
                    groups1d(n, 256),
                );
            }

            Op::Expand { .. } => {
                let x = node.inputs[0];
                let mut xd = dims(graph, x);
                let mut od = dims(graph, out);
                // Complex tensors pack `lanes` contiguous f32 per element
                // (C64=2 [re,im], C128=4 df64). The `reindex` kernel copies one
                // f32 per "element", so append an INNERMOST lane axis (xd==od==
                // lanes, never a 1→N broadcast). `contig_strides` then yields
                // lane-unit strides on the element axes and stride-1 on the lane
                // axis in BOTH istr/ostr, so each thread copies a whole complex
                // element's lanes as a contiguous group instead of shattering
                // [re,im]. `rank` and `n = numel(&od)` follow automatically.
                // lanes=1 (real/int) ⇒ dims/rank unchanged ⇒ strict no-op.
                let lanes: usize = match node.shape.dtype() {
                    DType::C64 => 2,
                    DType::C128 => 4,
                    _ => 1,
                };
                if lanes > 1 {
                    // The fixed [u32; 6] push arrays cap rank at 6; the appended
                    // lane axis therefore limits complex Expand to element-rank
                    // ≤ 5 (fine for real graphs). Guard rather than overflow.
                    assert!(
                        od.len() < 6,
                        "rlx-vulkan: complex Expand element-rank {} exceeds 5 \
                         (lane axis + [u32; 6] cap)",
                        od.len()
                    );
                    xd.push(lanes);
                    od.push(lanes);
                }
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .us(&shape)
                    .us(&istr)
                    .us(&ostr)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "reindex",
                    push,
                    groups1d(n, 256),
                );
            }

            Op::Concat { axis } => {
                // Complex packs `lanes` contiguous f32 per element; the `reindex`
                // kernel copies one f32 per "element". Append an INNERMOST lane
                // axis to the output AND each input's dims so `contig_strides`/
                // `numel` follow in lane units: `axis`/`axis_cursor` stay element-
                // indexed, but `out_str[*axis]` is now lane-scaled so each input
                // lands at the right complex offset. lanes=1 ⇒ dims unchanged.
                let lanes: usize = match node.shape.dtype() {
                    DType::C64 => 2,
                    DType::C128 => 4,
                    _ => 1,
                };
                let mut od = dims(graph, out);
                if lanes > 1 {
                    assert!(
                        od.len() < 6,
                        "rlx-vulkan: complex Concat element-rank {} exceeds 5 \
                         (lane axis + [u32; 6] cap)",
                        od.len()
                    );
                    od.push(lanes);
                }
                let out_str = contig_strides(&od);
                let rank = od.len();
                let mut axis_cursor = 0usize;
                for &inp in &node.inputs {
                    let mut id_dims = dims(graph, inp);
                    if lanes > 1 {
                        id_dims.push(lanes);
                    }
                    let in_str = contig_strides(&id_dims);
                    let mut shape = [1u32; 6];
                    let mut istr = [0u32; 6];
                    let mut ostr = [0u32; 6];
                    for ax in 0..rank {
                        shape[ax] = *id_dims.get(ax).unwrap_or(&1) as u32;
                        istr[ax] = *in_str.get(ax).unwrap_or(&0) as u32;
                        ostr[ax] = out_str[ax] as u32;
                    }
                    let out_off = binder.off(arena, out) + (axis_cursor * out_str[*axis]) as u32;
                    let n = numel(&id_dims);
                    let push = Push::default()
                        .u(n as u32)
                        .u(rank as u32)
                        .u(binder.off(arena, inp))
                        .u(out_off)
                        .us(&shape)
                        .us(&istr)
                        .us(&ostr)
                        .bytes();
                    push_gpu_step(
                        &mut binder,
                        &mut steps,
                        &mut deps,
                        "reindex",
                        push,
                        groups1d(n, 256),
                    );
                    axis_cursor += *id_dims.get(*axis).unwrap_or(&1);
                }
            }

            Op::Gather { axis } => {
                let data = node.inputs[0];
                let idx = node.inputs[1];
                let dd = dims(graph, data);
                let ax = *axis;
                // Complex packs `lanes` contiguous f32 per element. Index values
                // select ELEMENTS (indices stay unscaled, one f32 lane each), but
                // each gathered element is `lanes` contiguous f32 — so the inner
                // contiguous copy span (`out_inner`) and the total copy count
                // scale by lanes. `out_outer`/`axis_dim`/`n_idx` stay element
                // units. lanes=1 (real/int) ⇒ strict no-op.
                let lanes: usize = match node.shape.dtype() {
                    DType::C64 => 2,
                    DType::C128 => 4,
                    _ => 1,
                };
                let out_outer = numel(&dd[..ax]);
                let axis_dim = dd[ax];
                let out_inner = numel(&dd[ax + 1..]) * lanes;
                let n_idx = numel(&dims(graph, idx));
                let total = out_outer * n_idx * out_inner;
                let push = Push::default()
                    .u(out_outer as u32)
                    .u(n_idx as u32)
                    .u(out_inner as u32)
                    .u(axis_dim as u32)
                    .u(binder.off(arena, data))
                    .u(binder.off(arena, idx))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "gather",
                    push,
                    groups1d(total, 256),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .u(if *exclusive { 1 } else { 0 })
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "cumsum",
                    push,
                    groups1d(rows, 64),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .us(&shape)
                    .us(&flip)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "reverse",
                    push,
                    groups1d(n, 256),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .u(op_id)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "argreduce",
                    push,
                    groups1d(outer * inner, 256),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, gamma))
                    .u(binder.off(arena, beta))
                    .u(binder.off(arena, out))
                    .f(*eps)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "layernorm2d",
                    push,
                    groups1d(positions, 64),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .u(kind_id)
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "pool2d",
                    push,
                    groups1d(nn * cc * oh * ow, 64),
                );
            }

            Op::ResizeNearest2x => {
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let (nn, cc, hh, ww) = (xd[0], xd[1], xd[2], xd[3]);
                let push = Push::default()
                    .us(&[nn as u32, cc as u32, hh as u32, ww as u32])
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "resize2x",
                    push,
                    groups1d(nn * cc * hh * 4 * ww, 256),
                );
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
                    .u(binder.off(arena, input))
                    .u(binder.off(arena, weight))
                    .u(binder.off(arena, idx))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "grouped_matmul",
                    push,
                    (ceil_div(n, 16), ceil_div(m, 16), 1),
                );
            }

            Op::Conv {
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
            } => {
                // Opt-in CPU path for diagnosing SPIR-V conv2d (Kokoro ISTFTNet).
                if std::env::var("RLX_VULKAN_HOST_CONV").as_deref() == Ok("1") {
                    steps.push(Step::Host {
                        op: node.op.clone(),
                        out: node.id,
                        out_shape: node.shape.clone(),
                        inputs: node.inputs.clone(),
                    });
                } else {
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
                        .u(binder.off(arena, x))
                        .u(binder.off(arena, weight))
                        .u(binder.off(arena, bias))
                        .u(binder.off(arena, out))
                        .bytes();
                    push_gpu_step(
                        &mut binder,
                        &mut steps,
                        &mut deps,
                        "conv2d",
                        push,
                        groups1d(nn * cout * oh * ow, 64),
                    );
                }
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, delta))
                    .u(binder.off(arena, a))
                    .u(binder.off(arena, bmat))
                    .u(binder.off(arena, cmat))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "selective_scan",
                    push,
                    groups1d(bb * hh, 64),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "im2col",
                    push,
                    groups1d(nn * ho * wo * cin * kh * kw, 256),
                );
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
                    .u(binder.off(arena, updates))
                    .u(binder.off(arena, indices))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "scatter_add",
                    push,
                    groups1d(out_dim * trailing, 256),
                );
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
                    .u(binder.off(arena, x))
                    .u(binder.off(arena, out))
                    .bytes();
                push_gpu_step(
                    &mut binder,
                    &mut steps,
                    &mut deps,
                    "topk",
                    push,
                    groups1d(rows, 64),
                );
            }

            // GGUF K-quant / Q1_0 dequant + matmul. Q1_0 prefill (m>1) uses the
            // tiled GEMM kernel (Metal/wgpu parity). Decode GEMV (m==1) and
            // Q4_K/Q6_K prefill keep the row-loop GEMV. Other schemes → host.
            Op::DequantMatMul { scheme } => {
                use rlx_ir::quant::QuantScheme;
                let x = node.inputs[0];
                let xd = dims(graph, x);
                let od = dims(graph, out);
                let n = *od.last().unwrap_or(&1);
                let m = numel(&od) / n.max(1);
                let k = numel(&xd) / m.max(1);
                let gpu_scheme = match scheme {
                    QuantScheme::GgufQ4K => Some((0u32, 256usize)),
                    QuantScheme::GgufQ6K => Some((1u32, 256usize)),
                    QuantScheme::GgufQ1_0 => Some((2u32, 128usize)),
                    _ => None,
                };
                match gpu_scheme {
                    // Q1_0 prefill: one tiled GEMM (1-D grid — MoltenVK dropped
                    // the Y dimension of the earlier 2-D dispatch).
                    Some((2, blk)) if m > 1 && k.is_multiple_of(blk) && n >= 1 => {
                        let w = node.inputs[1];
                        const TM: usize = 8;
                        let n_row_tiles = m.div_ceil(TM);
                        let total = n * n_row_tiles;
                        let push = Push::default()
                            .u(m as u32)
                            .u(k as u32)
                            .u(n as u32)
                            .u(binder.off(arena, x))
                            .u(binder.off(arena, w))
                            .u(binder.off(arena, out))
                            .bytes();
                        push_gpu_step(
                            &mut binder,
                            &mut steps,
                            &mut deps,
                            "dequant_gemm_q1_0",
                            push,
                            groups1d(total, 64),
                        );
                    }
                    // Decode (m==1) or Q4_K/Q6_K prefill: row-loop GEMV on-device.
                    Some((sc, blk)) if k.is_multiple_of(blk) && n >= 1 && m >= 1 => {
                        let w = node.inputs[1];
                        let x_base = binder.off(arena, x);
                        let out_base = binder.off(arena, out);
                        for r in 0..m {
                            let push = Push::default()
                                .u(n as u32)
                                .u(k as u32)
                                .u(x_base + (r * k) as u32)
                                .u(binder.off(arena, w))
                                .u(out_base + (r * n) as u32)
                                .u(sc)
                                .bytes();
                            push_gpu_step(
                                &mut binder,
                                &mut steps,
                                &mut deps,
                                "dequant_matmul",
                                push,
                                groups1d(n, 64),
                            );
                        }
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

            // Native on-device FFT for f32 power-of-two rows up to 1024
            // (one workgroup per batch row, radix-2 in shared memory). Larger
            // n / non-f32 fall back to the host path. This keeps `Op::Fft` off
            // the host-fallback path, which crashes on discrete GPUs.
            Op::Fft { inverse, norm } => {
                let x = node.inputs[0];
                let in_shape = graph.node(x).shape.clone();
                let meta = rlx_ir::fft_meta(&in_shape);
                let native = matches!(in_shape.dtype(), DType::F32)
                    && meta.n_complex.is_power_of_two()
                    && meta.n_complex >= 2
                    && meta.n_complex <= 1024
                    && !is_weight_elem(binder.off(arena, x))
                    && !is_weight_elem(binder.off(arena, out));
                if native {
                    let scale = norm.output_scale(meta.n_complex, *inverse) as f32;
                    let push = Push::default()
                        .u(binder.off(arena, x))
                        .u(binder.off(arena, out))
                        .u(meta.n_complex as u32)
                        .u((meta.n_complex as u32).trailing_zeros())
                        .u(if *inverse { 1 } else { 0 })
                        .f(scale)
                        .u(meta.outer as u32)
                        .u(0)
                        .bytes();
                    push_gpu_step(
                        &mut binder,
                        &mut steps,
                        &mut deps,
                        "fft",
                        push,
                        (meta.outer as u32, 1, 1),
                    );
                } else {
                    steps.push(Step::Host {
                        op: node.op.clone(),
                        out: node.id,
                        out_shape: node.shape.clone(),
                        inputs: node.inputs.clone(),
                    });
                }
            }

            // Core SPD-manifold ops run on the CPU reference in F64 (their
            // eigen-decompositions have no SPIR-V kernel). Distinct from the
            // generic host-fallback below because the f32↔f64 conversion lives
            // in `crate::spd::eval`, not the thunk-based `host::eval`.
            op if crate::spd::is_spd_host(op) => {
                steps.push(Step::SpdHost {
                    op: node.op.clone(),
                    out: node.id,
                    out_shape: node.shape.clone(),
                    inputs: node.inputs.clone(),
                });
            }

            Op::Scan { .. } => {
                steps.push(Step::ScanHost {
                    desc: rlx_cpu::rlx_scan_host_desc!(graph, node, |id| arena.byte_offset(id)),
                });
            }

            Op::ScanBackward { .. }
            | Op::ScanBackwardXs { .. }
            | Op::ScatterNd { .. }
            | Op::ScatterElements { .. }
            | Op::GatherNd { .. }
            | Op::GatherElements { .. } => {
                steps.push(Step::HostOp {
                    desc: rlx_cpu::rlx_host_op_desc!(graph, node, |id| arena.byte_offset(id)),
                });
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
            let span = |id: NodeId| -> SlotSpan {
                if !arena.has(id) {
                    return SlotSpan::default();
                }
                SlotSpan {
                    // Barriers compare ranges in the *activation* buffer only;
                    // strip the weight tag so param spans don't look like 2 GiB+.
                    start: crate::buffer::raw_elem_off(arena.elem_offset(id)),
                    len: arena.slot_elems(id) as u32,
                }
            };
            let reads: Vec<SlotSpan> = node
                .inputs
                .iter()
                .filter(|&&id| arena.has(id))
                .map(|&id| span(id))
                .collect();
            let write = span(out);
            for step in &steps[before..] {
                if matches!(step, Step::ActCopy { .. }) {
                    continue;
                }
                deps.push(StepDep {
                    reads: reads.clone(),
                    write,
                });
            }
        }
        binder.reset_op();
    }
    binder.drain_copies(&mut steps, &mut deps);
    debug_assert_eq!(steps.len(), deps.len(), "schedule/deps length mismatch");
    (steps, deps)
}
