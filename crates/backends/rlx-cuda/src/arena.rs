// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA device-memory arena.
//!
//! Mirrors rlx-wgpu's `Arena`: one big device buffer allocated at
//! compile time, per-node byte offsets carved out by the planner.
//! Activations live as f32 in the main `buffer` (Bool / I32 widen on
//! access) — same f32-uniform convention as rlx-wgpu, so we can share
//! kernel logic.
//!
//! Optional **half-precision side-buffer** (`half_buffer`, raw `u16`
//! storage) stores params (weights) as f16 or bf16. Activations stay
//! f32 — this is the standard inference setup: 2× weight memory
//! savings + Tensor Core compute via cublasGemmEx, full precision on
//! the bandwidth-sensitive softmax / norm / residual paths.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice};
use rlx_ir::{DType, Graph, NodeId, Op};
use rlx_opt::memory::{BufferSlot, MemoryPlan};

/// Half-precision dtype tag. Bit-identical layouts (16 bits each) but
/// different exponent/mantissa splits — kernels need to know which one
/// to interpret. Stored alongside each half-arena offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfDtype {
    F16,
    Bf16,
}

/// One contiguous f32 device buffer + per-node offsets, plus an
/// optional u16 side-buffer for f16/bf16 params.
pub struct Arena {
    /// Underlying CUDA allocation for f32 activations + un-promoted
    /// params. Sized by the memory plan; lives as long as the executable.
    /// Returned to a process-wide pool on drop.
    pub buffer: ManuallyDrop<CudaSlice<f32>>,
    /// Per-node byte offset into `buffer`.
    pub offsets: HashMap<NodeId, usize>,
    /// Per-node byte length (data, not slot).
    pub lens: HashMap<NodeId, usize>,
    /// Total arena size in bytes.
    pub size: usize,

    /// Optional half-precision side-buffer (raw u16 bits — `f16` or
    /// `bf16` per-node tag). Allocated lazily on first
    /// `register_half_param`. Backends that consume half-precision
    /// (cublasGemmEx, matmul_wmma) read from here using the half
    /// offsets; other backends fall back to the f32 `buffer`.
    pub half_buffer: Option<CudaSlice<u16>>,
    /// Per-node `(half_offset_in_u16_elements, HalfDtype)`.
    pub half_offsets: HashMap<NodeId, (usize, HalfDtype)>,
    /// Inverse lookup keyed by the param's f32-arena offset (in f32
    /// elements). Lets the matmul dispatch ask "is this input
    /// half-stored?" given only the `*_off_f32` it has at hand.
    pub half_by_f32_off: HashMap<u32, (usize, HalfDtype)>,
    /// Total half-buffer size in u16 elements.
    pub half_size: usize,
}

static F32_ARENA_POOL: OnceLock<Mutex<Vec<(usize, CudaSlice<f32>)>>> = OnceLock::new();

/// Max pooled f32 buffers retained (default 2 — enough for double-buffering,
/// avoids pinning multiple 10+ GiB Orpheus arenas on 16 GiB GPUs).
fn pool_enabled() -> bool {
    rlx_ir::env::flag("RLX_CUDA_ARENA_POOL")
}

/// Whether to zero arena buffers on acquisition (default ON). An un-zeroed arena
/// lets an op read a slot's alignment padding or a not-yet-written slot and pick
/// up driver/stale garbage, which intermittently collapses CUDA *training* to
/// chance while CPU/MLX are fine. Set `RLX_CUDA_NO_ZERO_ARENA=1` to opt out
/// (reproduce the bug / benchmark the memset cost).
fn zero_arena_enabled() -> bool {
    !rlx_ir::env::flag("RLX_CUDA_NO_ZERO_ARENA")
}

fn pool_max_buffers() -> usize {
    rlx_ir::env::var("RLX_CUDA_ARENA_POOL_MAX")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
        .max(1)
}

/// Do not retain individual arenas larger than this in the pool (bytes).
fn pool_max_chunk_bytes() -> usize {
    rlx_ir::env::var("RLX_CUDA_ARENA_POOL_CHUNK_BYTES")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512 * 1024 * 1024)
        .max(1024 * 1024)
}

/// Drop all pooled device buffers — call between large graph compiles on
/// memory-constrained GPUs (Orpheus 3B prefill → decode handoff).
pub fn trim_f32_arena_pool() {
    let mut pool = f32_arena_pool()
        .lock()
        .expect("rlx-cuda: arena pool lock poisoned");
    pool.clear();
}

fn f32_arena_pool() -> &'static Mutex<Vec<(usize, CudaSlice<f32>)>> {
    F32_ARENA_POOL.get_or_init(|| Mutex::new(Vec::new()))
}

fn pool_acquire_f32(ctx: &Arc<CudaContext>, n_f32: usize) -> CudaSlice<f32> {
    let need = n_f32.max(4);
    if let Some(mut buf) = try_pool_take(need) {
        // A pooled buffer still holds the previous graph's data. Reusing it dirty
        // is what makes the stale-read collapse *intermittent* (each run/retry
        // grabs a different recycled buffer). Zero it so reuse matches a fresh
        // `alloc_zeros`. Async memset on the arena's own stream → ordered before
        // any kernel that later reads this buffer.
        if zero_arena_enabled() {
            let _ = ctx.default_stream().memset_zeros(&mut buf);
        }
        return buf;
    }
    match try_alloc_f32(ctx, need) {
        Ok(buf) => buf,
        Err(_) => {
            trim_f32_arena_pool();
            try_alloc_f32(ctx, need).unwrap_or_else(|_| {
                panic!(
                    "rlx-cuda: device allocation failed for {} f32 ({:.3} GiB)",
                    need,
                    (need as f64 * 4.0) / (1u64 << 30) as f64
                )
            })
        }
    }
}

fn try_pool_take(need: usize) -> Option<CudaSlice<f32>> {
    let mut pool = f32_arena_pool()
        .lock()
        .expect("rlx-cuda: arena pool lock poisoned");
    if let Some(idx) = pool.iter().position(|(cap, _)| *cap >= need) {
        let (_, buf) = pool.swap_remove(idx);
        return Some(buf);
    }
    None
}

/// Managed (unified) arena — pages migrate host↔device on demand, so the arena
/// may **oversubscribe VRAM**: a pipeline stage larger than the GPU still runs
/// (paged over PCIe). Opt in with `RLX_CUDA_UNIFIED=1` to run big models on small
/// GPUs (e.g. a 42 GB stage on a 16 GB card). Slower than resident VRAM, but it
/// runs where a plain `cudaMalloc` would OOM.
fn unified_arena_enabled() -> bool {
    rlx_ir::env::flag("RLX_CUDA_UNIFIED")
}

fn try_alloc_f32(ctx: &Arc<CudaContext>, n_f32: usize) -> Result<CudaSlice<f32>, ()> {
    // Managed/pageable path: cuMemAllocManaged, wrapped as a CudaSlice via
    // `upgrade_device_ptr` so the rest of the backend is unchanged (same type,
    // same device pointer — the pages just migrate on access).
    if unified_arena_enabled() {
        let bytes = n_f32 * std::mem::size_of::<f32>();
        let cu = unsafe {
            cudarc::driver::result::malloc_managed(
                bytes,
                cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL,
            )
        }
        .map_err(|_| ())?;
        // Prefer HOST for the oversubscribed arena so it does NOT greedily migrate
        // into the small VRAM. A stage far larger than the GPU (e.g. 50 GB on 16 GB)
        // would otherwise fill VRAM with resident pages, leaving no room for the
        // forward's device-only allocations (cuBLAS/cuDNN workspaces, `CudaSlice`
        // clones) → CUDA_ERROR_OUT_OF_MEMORY mid-forward. With host-preferred pages,
        // the GPU still reads them (migrated on access, then evictable back to host
        // under VRAM pressure), so device allocs always find free VRAM. Best-effort.
        unsafe {
            const CU_DEVICE_CPU: i32 = -1;
            let _ = cudarc::driver::sys::cuMemAdvise(
                cu,
                bytes,
                cudarc::driver::sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                CU_DEVICE_CPU,
            );
        }
        let mut slice = unsafe { ctx.default_stream().upgrade_device_ptr::<f32>(cu, n_f32) };
        // Managed memory is uninitialized — zero it (same determinism as below).
        let _ = ctx.default_stream().memset_zeros(&mut slice);
        return Ok(slice);
    }
    // Zero on allocation (default). An un-zeroed arena lets any op that reads a
    // slot's alignment padding, or a slot before its producer has written it,
    // pick up whatever garbage the driver handed back — which intermittently
    // (~1/3 of runs) collapses CUDA training to chance (loss pinned at ln k)
    // while CPU/MLX train fine. `alloc_zeros` (async memset on the stream) makes
    // every arena start deterministic. f32 is `ValidAsZeroBits`.
    if zero_arena_enabled() {
        ctx.default_stream().alloc_zeros(n_f32).map_err(|_| ())
    } else {
        unsafe { ctx.default_stream().alloc(n_f32).map_err(|_| ()) }
    }
}

fn pool_release_f32(cap_f32: usize, buffer: CudaSlice<f32>) {
    if !pool_enabled() {
        return;
    }
    let cap_bytes = cap_f32.saturating_mul(4);
    if cap_bytes > pool_max_chunk_bytes() {
        return;
    }
    let mut pool = f32_arena_pool()
        .lock()
        .expect("rlx-cuda: arena pool lock poisoned");
    while pool.len() >= pool_max_buffers() {
        pool.sort_by_key(|(cap, _)| *cap);
        pool.remove(0);
    }
    pool.push((cap_f32.max(4), buffer));
}

/// Plan memory using f32-sized slots regardless of declared IR dtype.
/// Byte size of a node's f32-arena slot. Packed GGUF params use U8/I8 shapes
/// `[bytes.len()]` — reserve byte storage, not `elems * 4` (that inflated Orpheus
/// 3B arenas ~4×). Bool is NOT byte-packed here: it is a compare/mask output
/// written as f32 (1.0/0.0) into the f32 arena, so it needs the full `elems * 4`
/// — sizing it as `elems` let the f32 compare kernel overrun its slot.
///
/// Empty tensors (`num_elements() == 0`) still get a 1-element slot so every
/// NodeId has an offset — matching rlx-wgpu. Skipping them left `arena.offset`
/// panicking on LuxTTS-sized CFM graphs (ScatterND / empty ConstantOfShape
/// consumers still lower a kernel that touches the id).
fn arena_slot_bytes(node: &rlx_ir::Node, align: usize) -> usize {
    let elems = node.shape.num_elements().unwrap_or(0).max(1);
    let bytes = match node.shape.dtype() {
        DType::U8 | DType::I8 => elems,
        // Complex simulates on f32 lanes: C64 = 2 lanes/elem (8 B), C128 = 4
        // lanes/elem (16 B, df64). Sizing these `elems * 4` (as the `_` arm did)
        // truncated the imaginary/low lanes — the same class of bug as blanket-
        // applying `size_bytes()`, which would break U8/Bool (1 native byte but
        // stored as one f32 lane). So scope the multi-lane widening to complex.
        DType::C64 => elems * 8,
        DType::C128 => elems * 16,
        _ => elems * 4,
    };
    bytes.div_ceil(align) * align
}

/// Number of f32 lanes a node occupies in the f32-uniform arena's host-readback
/// view. Complex is simulated on f32 lanes (C64 = 2 lanes/elem, C128 = 4); every
/// other dtype is one f32 lane per element (I64/Bool/… widen to a single lane).
/// Used to size + read the host staging slot so a complex output reads back ALL
/// its lanes, not just `num_elements` (which would truncate to the real parts).
pub(crate) fn arena_lane_count(shape: &rlx_ir::Shape) -> usize {
    let elems = shape.num_elements().unwrap_or(0);
    match shape.dtype() {
        DType::C64 => elems * 2,
        DType::C128 => elems * 4,
        _ => elems,
    }
}

/// Cast op ids for the shared unary kernel (`unary.cu` cases 100–106). Kept
/// in sync with rlx-rocm (same kernel) and rlx-vulkan / rlx-oneapi.
pub(crate) const CAST_F32_TO_I8: u32 = 100;
pub(crate) const CAST_F32_TO_I16: u32 = 101;
pub(crate) const CAST_F32_TO_I32: u32 = 102;
pub(crate) const CAST_F32_TO_I64: u32 = 103;
pub(crate) const CAST_F32_TO_U8: u32 = 104;
pub(crate) const CAST_F32_TO_U32: u32 = 105;
pub(crate) const CAST_TO_BOOL: u32 = 106;

/// How an `Op::Cast` lowers on the f32-uniform arena.
pub(crate) enum CastLower {
    /// Value-preserving relabel — alias the input slot. Covers same-dtype,
    /// int→float, float→float (F16/BF16/F64 are all f32-stored here), int→int,
    /// and bool→int/float.
    Identity,
    /// A real elementwise conversion via the unary kernel with this op id
    /// (float→int trunc-saturate, or →Bool `x != 0`).
    Kernel(u32),
    /// A complex cast (real↔C64, real↔C128, C64↔C128) — pure f32-lane moves via
    /// the standalone `complex_cast` kernel. Carries the mode (0..5, see
    /// `complex_cast.cu`). Needs its own (complex-sized) slot, not an alias.
    Complex(u32),
    /// Not representable in an f32 arena (F64 has no lane storage) — reject.
    Reject,
}

/// Classify a `Cast(src → dst)` on the f32-uniform arena. float→int truncates
/// toward zero + saturates (Rust `as` / rlx-cpu); →Bool is `x != 0`. F16/BF16/
/// F64 are demoted to f32 storage so casts to/from them are identity relabels;
/// only complex (C64) conversions are rejected.
pub(crate) fn classify_cast(src: DType, dst: DType) -> CastLower {
    if src == dst {
        return CastLower::Identity; // pure relabel (also covers C64→C64 / C128→C128)
    }
    // Complex casts (real↔C64, real↔C128, C64↔C128) are pure f32-lane moves on
    // the simulated-complex arena. F64 is the one component type with no f32-lane
    // storage here, so a complex cast touching F64 (real side) is still rejected.
    if src.is_complex() || dst.is_complex() {
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

/// True when a Cast node needs its own slot + a conversion kernel (float→int /
/// →Bool), or must be rejected — i.e. it is *not* an identity relabel and so
/// must NOT alias the input slot.
pub(crate) fn cast_is_kernel(graph: &Graph, node: &rlx_ir::Node) -> bool {
    match &node.op {
        Op::Cast { to } => !matches!(
            classify_cast(graph.node(node.inputs[0]).shape.dtype(), *to),
            CastLower::Identity
        ),
        _ => false,
    }
}

/// A view (arena-aliased, no kernel): Reshape / StopGradient always, and Cast
/// only when it is an identity relabel. float→int / →Bool casts get their own
/// slot (see [`cast_is_kernel`]).
#[inline]
fn is_arena_view(graph: &Graph, node: &rlx_ir::Node) -> bool {
    match &node.op {
        Op::Reshape { .. } | Op::StopGradient => true,
        Op::Cast { .. } => !cast_is_kernel(graph, node),
        _ => false,
    }
}

/// Same f32-uniform layout as rlx-wgpu (every tensor is f32; Reshape/Cast/
/// StopGradient alias the input slot — a zero-copy relabel), but with
/// **liveness-aware slot reuse**: buffers whose live ranges don't overlap share
/// device memory instead of each getting a fresh offset.
///
/// The old planner gave every node a unique sequential slot, so the arena grew
/// to the *sum* of every intermediate. On an unrolled 28-layer GGUF decode graph
/// that summed to ~3.9 GiB for a 0.6B model and OOM'd the device (rlx-cuda only
/// ever reached the model on CPU/Vulkan). Reuse drops it to roughly the peak live
/// set + resident weights.
///
/// Safe for CUDA specifically: kernels dispatch in schedule order on the default
/// stream (in-order), and `Op::Narrow` is materialised as a real copy (not an
/// alias), so — unlike the wgpu planner — no view keeps a parent buffer live past
/// its last real read. Reads are attributed to the alias root so a Reshape/Cast
/// consumer keeps the underlying buffer alive. Set `RLX_CUDA_ARENA_NO_REUSE=1` to
/// fall back to the old unique-slot-per-node behaviour.
pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let align = align.max(1);
    let nodes = graph.nodes();
    let n = nodes.len();
    let schedule: Vec<NodeId> = nodes.iter().map(|nd| nd.id).collect();

    // Resolve each node to the buffer-owning ancestor it aliases (view chains).
    let mut root_of: HashMap<NodeId, NodeId> = HashMap::with_capacity(n);
    for node in nodes {
        let root = if is_arena_view(graph, node) {
            match node.inputs.first() {
                Some(in_id) => *root_of.get(in_id).unwrap_or(in_id),
                None => node.id,
            }
        } else {
            node.id
        };
        root_of.insert(node.id, root);
    }
    let root = |id: NodeId| -> NodeId { root_of.get(&id).copied().unwrap_or(id) };

    let no_reuse = rlx_ir::env::flag("RLX_CUDA_ARENA_NO_REUSE");

    // Liveness per owning buffer: birth = first step it (or an alias) is produced;
    // death = last step any node reads it (reads charged to the alias root).
    let mut birth: HashMap<NodeId, usize> = HashMap::new();
    let mut death: HashMap<NodeId, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let r = root(node.id);
        birth.entry(r).and_modify(|b| *b = (*b).min(i)).or_insert(i);
        death.entry(r).and_modify(|d| *d = (*d).max(i)).or_insert(i);
        for &inp in &node.inputs {
            let ri = root(inp);
            death
                .entry(ri)
                .and_modify(|d| *d = (*d).max(i))
                .or_insert(i);
            birth.entry(ri).or_insert(0);
        }
    }
    // Elided broadcast (see `Step::BinaryBroadcast`): the CUDA backend folds an
    // `Op::Expand` that feeds only `Op::Binary` into a stride-aware read of the
    // Expand's *input* at the Binary. That input must therefore stay live as long
    // as the Expand's output would — otherwise its slot is recycled before the
    // Binary reads it (silent wrong values). Extend it for every Expand input;
    // over-extending a materialized Expand's input is harmless (they're small
    // broadcast tensors) and keeps this independent of the compile-time elide set.
    for node in nodes {
        if matches!(node.op, Op::Expand { .. }) {
            if let Some(&pin) = node.inputs.first() {
                let de = *death.get(&root(node.id)).unwrap_or(&n);
                let p = root(pin);
                death
                    .entry(p)
                    .and_modify(|d| *d = (*d).max(de))
                    .or_insert(de);
            }
        }
    }

    // Params / Inputs / Constants are resident for the whole execution; graph
    // outputs must survive to the final read-back. Never let these be reused.
    for node in nodes {
        if matches!(
            node.op,
            Op::Param { .. } | Op::Input { .. } | Op::Constant { .. }
        ) {
            let r = root(node.id);
            birth.insert(r, 0);
            death.insert(r, n);
        }
    }
    for &out in &graph.outputs {
        death.insert(root(out), n);
    }

    // Buffer-owning nodes to place (alias nodes borrow their root's slot later).
    struct Buf {
        id: NodeId,
        size: usize,
        birth: usize,
        death: usize,
    }
    let mut bufs: Vec<Buf> = Vec::new();
    for node in nodes {
        if root(node.id) != node.id {
            continue;
        }
        // A float→int / →Bool Cast writes f32 lanes via the unary kernel, so its
        // own slot must be f32-sized even when the dst dtype (I8/U8) would
        // otherwise byte-pack. A COMPLEX cast, however, produces a genuine
        // multi-lane (C64=2, C128=4) output — it must keep its complex-sized
        // slot (`arena_slot_bytes`), not the `elems * 4` single-lane sizing.
        let size = if cast_is_kernel(graph, node) && !node.shape.dtype().is_complex() {
            let elems = node.shape.num_elements().unwrap_or(0).max(1);
            (elems * 4).div_ceil(align) * align
        } else {
            arena_slot_bytes(node, align)
        };
        bufs.push(Buf {
            id: node.id,
            size,
            birth: *birth.get(&node.id).unwrap_or(&0),
            death: *death.get(&node.id).unwrap_or(&n),
        });
    }

    let plan_t0 = Instant::now();
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::with_capacity(bufs.len());
    let mut arena_size = 0usize;

    if no_reuse {
        // Unique sequential slots (debug / parity).
        bufs.sort_by_key(|b| std::cmp::Reverse(b.size));
        for b in &bufs {
            let off = arena_size.div_ceil(align) * align;
            assignments.insert(
                b.id,
                BufferSlot {
                    offset: off,
                    size: b.size,
                },
            );
            arena_size = off + b.size;
        }
    } else {
        // O(B log B) free-list allocator. Birth-order placement with a min-heap
        // of retirements replaces the old O(B²) "scan all live placed buffers"
        // planner — fusion-disabled packed Qwen35 graphs have ~90k buffers and
        // the quadratic path dominated the ~200s CUDA compile.
        bufs.sort_by(|a, b| {
            a.birth
                .cmp(&b.birth)
                .then_with(|| b.size.cmp(&a.size))
                .then_with(|| a.id.0.cmp(&b.id.0))
        });

        // Free interval keyed by start offset → length.
        let mut free: BTreeMap<usize, usize> = BTreeMap::new();
        #[derive(Eq, PartialEq)]
        struct Retire {
            death: usize,
            offset: usize,
            size: usize,
        }
        impl Ord for Retire {
            fn cmp(&self, other: &Self) -> Ordering {
                // Min-heap on death via reverse ordering.
                other
                    .death
                    .cmp(&self.death)
                    .then_with(|| self.offset.cmp(&other.offset))
            }
        }
        impl PartialOrd for Retire {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        let mut retire: BinaryHeap<Retire> = BinaryHeap::new();

        let insert_free = |free: &mut BTreeMap<usize, usize>, mut off: usize, mut size: usize| {
            if size == 0 {
                return;
            }
            // Merge with previous interval if contiguous.
            if let Some((&po, &ps)) = free.range(..=off).next_back() {
                if po + ps == off {
                    free.remove(&po);
                    off = po;
                    size += ps;
                } else if po + ps > off {
                    // Overlap should not happen; clamp to end of prev.
                    let end = off + size;
                    let start = po + ps;
                    if end <= start {
                        return;
                    }
                    off = start;
                    size = end - start;
                }
            }
            let end = off + size;
            // Merge with next interval if contiguous.
            if let Some((&no, &ns)) = free.range(off..).next() {
                if no == end {
                    free.remove(&no);
                    size += ns;
                } else if no < end {
                    let merged_end = no + ns;
                    free.remove(&no);
                    size = merged_end.saturating_sub(off);
                }
            }
            if size > 0 {
                free.insert(off, size);
            }
        };

        for b in &bufs {
            // Release slots whose live range ended before this birth.
            while let Some(top) = retire.peek() {
                if top.death >= b.birth {
                    break;
                }
                let r = retire.pop().unwrap();
                insert_free(&mut free, r.offset, r.size);
            }

            // Best-fit among free holes (aligned).
            let mut best: Option<(usize, usize, usize)> = None; // waste, map_key, aligned_off
            for (&fo, &fs) in &free {
                let a = fo.div_ceil(align) * align;
                if a.saturating_add(b.size) <= fo.saturating_add(fs) {
                    let waste = fo.saturating_add(fs).saturating_sub(a + b.size) + (a - fo);
                    let cand = (waste, fo, a);
                    best = Some(match best {
                        None => cand,
                        Some(cur) => {
                            if cand.0 < cur.0 || (cand.0 == cur.0 && cand.2 < cur.2) {
                                cand
                            } else {
                                cur
                            }
                        }
                    });
                }
            }

            let off = if let Some((_, key, aligned)) = best {
                let fs = free.remove(&key).unwrap();
                let hole_end = key + fs;
                if aligned > key {
                    insert_free(&mut free, key, aligned - key);
                }
                let used_end = aligned + b.size;
                if used_end < hole_end {
                    insert_free(&mut free, used_end, hole_end - used_end);
                }
                aligned
            } else {
                let a = arena_size.div_ceil(align) * align;
                arena_size = a + b.size;
                a
            };

            assignments.insert(
                b.id,
                BufferSlot {
                    offset: off,
                    size: b.size,
                },
            );
            retire.push(Retire {
                death: b.death,
                offset: off,
                size: b.size,
            });
            arena_size = arena_size.max(off + b.size);
        }
    }

    if rlx_ir::env::flag("RLX_CUDA_COMPILE_TIMING") || rlx_ir::env::flag("RLX_CUDA_ARENA_DEBUG") {
        eprintln!(
            "[cuda-arena] planned {} bufs → {:.3} GiB in {:.2?}",
            bufs.len(),
            arena_size as f64 / (1u64 << 30) as f64,
            plan_t0.elapsed()
        );
    }

    if rlx_ir::env::flag("RLX_CUDA_ARENA_DEBUG") {
        let node_by_id: HashMap<NodeId, &rlx_ir::Node> =
            nodes.iter().map(|nd| (nd.id, nd)).collect();
        let (mut resident, mut transient) = (0usize, 0usize);
        for b in &bufs {
            match node_by_id.get(&b.id).map(|nd| &nd.op) {
                Some(Op::Param { .. } | Op::Input { .. } | Op::Constant { .. }) => {
                    resident += b.size
                }
                _ => transient += b.size,
            }
        }
        let gib = |x: usize| x as f64 / (1u64 << 30) as f64;
        eprintln!(
            "[cuda-arena] {} bufs: resident(param/input/const)={:.3} GiB, transient={:.3} GiB, peak={:.3} GiB",
            bufs.len(),
            gib(resident),
            gib(transient),
            gib(arena_size)
        );
        let mut by_size: Vec<&Buf> = bufs.iter().collect();
        by_size.sort_by_key(|b| std::cmp::Reverse(b.size));
        for b in by_size.iter().take(14) {
            if let Some(nd) = node_by_id.get(&b.id) {
                let label = match &nd.op {
                    Op::Param { name } => format!("Param({name})"),
                    Op::Input { name } => format!("Input({name})"),
                    Op::Constant { .. } => "Constant".to_string(),
                    other => format!("{other:?}").chars().take(40).collect(),
                };
                let dims: Vec<i64> = (0..nd.shape.rank())
                    .map(|i| {
                        let d = nd.shape.dim(i);
                        if d.is_static() {
                            d.unwrap_static() as i64
                        } else {
                            -1
                        }
                    })
                    .collect();
                eprintln!(
                    "  [{:.3} GiB] {label} dtype={:?} shape={:?} live[{}..{}]",
                    gib(b.size),
                    nd.shape.dtype(),
                    dims,
                    b.birth,
                    b.death
                );
            }
        }
    }

    // Alias nodes (Reshape/Cast/StopGradient) borrow their root's slot.
    for node in nodes {
        let r = root(node.id);
        if r != node.id
            && let Some(root_slot) = assignments.get(&r).cloned()
        {
            assignments.insert(node.id, root_slot);
        }
    }

    // Every NodeId must have a slot — compile lowers kernels that touch ids
    // even when the tensor is empty / an unresolved view chain. Give orphans a
    // dedicated aligned byte at the end of the arena (LuxTTS fm_decoder panic).
    let mut orphan_off = arena_size.div_ceil(align) * align;
    for node in nodes {
        if assignments.contains_key(&node.id) {
            continue;
        }
        let size = arena_slot_bytes(node, align).max(align);
        assignments.insert(
            node.id,
            BufferSlot {
                offset: orphan_off,
                size,
            },
        );
        orphan_off += size;
    }
    arena_size = orphan_off.max(arena_size);

    MemoryPlan {
        arena_size,
        assignments,
        schedule,
    }
}

impl Arena {
    pub fn from_plan(ctx: &Arc<CudaContext>, plan: &MemoryPlan) -> Self {
        let n_f32 = plan.arena_size.div_ceil(4);
        let buffer = ManuallyDrop::new(pool_acquire_f32(ctx, n_f32));
        let mut offsets = HashMap::new();
        let mut lens = HashMap::new();
        for (id, slot) in &plan.assignments {
            offsets.insert(*id, slot.offset);
            lens.insert(*id, slot.size);
        }
        Self {
            buffer,
            offsets,
            lens,
            size: plan.arena_size,
            half_buffer: None,
            half_offsets: HashMap::new(),
            half_by_f32_off: HashMap::new(),
            half_size: 0,
        }
    }

    pub fn has(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id)
    }

    #[inline]
    pub fn f32_buf(&self) -> &CudaSlice<f32> {
        &self.buffer
    }

    #[inline]
    pub fn f32_buf_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.buffer
    }

    #[inline]
    pub fn f32_buf_and_size(&mut self) -> (&mut CudaSlice<f32>, usize) {
        let size = self.size;
        (self.f32_buf_mut(), size)
    }

    /// Byte offset of `id`'s slot in the f32 arena. NOTE: a `usize` that
    /// **routinely exceeds `u32::MAX`** — arenas are often > 4 GiB (batch-16 conv
    /// codec ~5.6 GiB, NO_REUSE graphs ~15 GiB). A `Step` field that stashes this
    /// must keep it `u64` (byte offset) or store the f32-element index `offset / 4`
    /// in `u32` — NEVER `offset as u32`, which truncates past 4 GiB and was bug #4.
    /// See the `Step` enum doc for the full convention.
    pub fn offset(&self, id: NodeId) -> usize {
        match self.offsets.get(&id) {
            Some(&off) => off,
            None => panic!(
                "rlx-cuda arena: no offset for {id:?} ({} slots). \
                 Usually a zero-size / unresolved view node was skipped by the planner.",
                self.offsets.len()
            ),
        }
    }
    pub fn len_of(&self, id: NodeId) -> usize {
        self.lens[&id]
    }
    pub fn set_actual_len(&mut self, id: NodeId, bytes: usize) {
        self.lens.insert(id, bytes);
    }

    /// Reserve a slot in the half-precision side-buffer for `id` with
    /// `n_elems` u16 elements. Returns the offset (in u16 elements).
    /// Allocates / grows the underlying CudaSlice as needed. The
    /// caller passes the param's `f32_off` (in f32 elements) so the
    /// inverse `half_by_f32_off` map is kept consistent for the
    /// matmul dispatch's "is this input half-stored?" check.
    pub fn register_half_param(
        &mut self,
        ctx: &Arc<CudaContext>,
        id: NodeId,
        f32_off: u32,
        n_elems: usize,
        dtype: HalfDtype,
    ) -> usize {
        let off = self.half_size;
        self.half_size += n_elems;
        self.half_offsets.insert(id, (off, dtype));
        self.half_by_f32_off.insert(f32_off, (off, dtype));
        // (Re)allocate a buffer that fits the new total size. Cheap
        // because params are only registered at compile / load time —
        // not on the run() hot path.
        let stream = ctx.default_stream();
        // The half side-buffer holds BF16 params (e.g. all the MoE scales of a big
        // stage → several GB). On an oversubscribed GPU it must be MANAGED like the
        // main arena, or this device alloc — plus the transient old+new during a
        // grow — OOMs the small VRAM. Managed + host-preferred keeps it off VRAM.
        let n_u16 = self.half_size.max(4);
        let mut new_buf = if unified_arena_enabled() {
            let bytes = n_u16 * std::mem::size_of::<u16>();
            let cu = unsafe {
                cudarc::driver::result::malloc_managed(
                    bytes,
                    cudarc::driver::sys::CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL,
                )
            }
            .expect("rlx-cuda: managed half-arena allocation failed");
            unsafe {
                const CU_DEVICE_CPU: i32 = -1;
                let _ = cudarc::driver::sys::cuMemAdvise(
                    cu,
                    bytes,
                    cudarc::driver::sys::CUmem_advise_enum::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                    CU_DEVICE_CPU,
                );
            }
            let mut s = unsafe { stream.upgrade_device_ptr::<u16>(cu, n_u16) };
            let _ = stream.memset_zeros(&mut s);
            s
        } else {
            stream
                .alloc_zeros::<u16>(n_u16)
                .expect("rlx-cuda: half-arena allocation failed")
        };
        if let Some(old) = self.half_buffer.take() {
            // Copy old contents into the new buffer's prefix. Best-effort. Copy INTO
            // `new_buf` directly — the old code cloned it into a throwaway temporary,
            // which both lost the copy AND allocated a second full-size buffer (a
            // 2× VRAM spike that OOMs once the half side-buffer holds many params,
            // e.g. all the BF16 MoE scales of a big oversubscribed stage).
            let n = old.len().min(new_buf.len());
            let _ = stream.memcpy_dtod(&old.slice(0..n), &mut new_buf.slice_mut(0..n));
        }
        self.half_buffer = Some(new_buf);
        off
    }

    /// True iff `id` has an entry in the half-precision side-buffer.
    pub fn is_half(&self, id: NodeId) -> bool {
        self.half_offsets.contains_key(&id)
    }

    /// `(offset_in_u16_elements, dtype)` for a half-stored node.
    pub fn half_off(&self, id: NodeId) -> Option<(usize, HalfDtype)> {
        self.half_offsets.get(&id).copied()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let cap_f32 = self.size.div_ceil(4).max(4);
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) };
        pool_release_f32(cap_f32, buffer);
    }
}
