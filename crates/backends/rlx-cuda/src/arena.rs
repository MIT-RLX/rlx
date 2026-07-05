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

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock};

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
    if let Some(buf) = try_pool_take(need) {
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

fn try_alloc_f32(ctx: &Arc<CudaContext>, n_f32: usize) -> Result<CudaSlice<f32>, ()> {
    unsafe { ctx.default_stream().alloc(n_f32).map_err(|_| ()) }
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
/// Same logic as rlx-wgpu — keeps every tensor as f32 in the arena.
/// Reshape and Cast alias the input slot (zero-copy relabel in our
/// row-major f32 layout).
pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
        if matches!(
            node.op,
            Op::Reshape { .. } | Op::Cast { .. } | Op::StopGradient
        ) && let Some(in_id) = node.inputs.first()
            && let Some(slot) = assignments.get(in_id)
        {
            let aliased = slot.clone();
            assignments.insert(node.id, aliased);
            schedule.push(node.id);
            continue;
        }
        let elems = node.shape.num_elements().unwrap_or(0);
        // Packed GGUF params use U8/I8 shapes `[bytes.len()]` — reserve byte
        // storage, not `elems * 4` (that inflated Orpheus 3B arenas ~4×). Bool
        // is NOT byte-packed here: it is a compare/mask output written as f32
        // (1.0/0.0) into the f32 arena, so it needs the full `elems * 4` — sizing
        // it as `elems` let the f32 compare kernel overrun its slot.
        let bytes = match node.shape.dtype() {
            DType::U8 | DType::I8 => elems,
            _ => elems * 4,
        };
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
        arena_size: cursor,
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

    pub fn offset(&self, id: NodeId) -> usize {
        self.offsets[&id]
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
        let new_buf = stream
            .alloc_zeros::<u16>(self.half_size.max(4))
            .expect("rlx-cuda: half-arena allocation failed");
        if let Some(old) = self.half_buffer.take() {
            // Copy old contents into the new buffer's prefix. Best-effort.
            let _ = stream.memcpy_dtod(&old, &mut { new_buf.clone() });
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
