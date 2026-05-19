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
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice};
use rlx_ir::{Graph, NodeId, Op};
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
    pub buffer: CudaSlice<f32>,
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

/// Plan memory using f32-sized slots regardless of declared IR dtype.
/// Same logic as rlx-wgpu — keeps every tensor as f32 in the arena.
/// Reshape and Cast alias the input slot (zero-copy relabel in our
/// row-major f32 layout).
pub fn plan_f32_uniform(graph: &Graph, align: usize) -> MemoryPlan {
    let mut assignments: HashMap<NodeId, BufferSlot> = HashMap::new();
    let mut schedule = Vec::with_capacity(graph.nodes().len());
    let mut cursor = 0usize;
    for node in graph.nodes() {
        if matches!(node.op, Op::Reshape { .. } | Op::Cast { .. })
            && let Some(in_id) = node.inputs.first()
            && let Some(slot) = assignments.get(in_id)
        {
            let aliased = slot.clone();
            assignments.insert(node.id, aliased);
            schedule.push(node.id);
            continue;
        }
        let elems = node.shape.num_elements().unwrap_or(0);
        let bytes = elems * 4;
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
        let stream = ctx.default_stream();
        // alloc_zeros gives a deterministic starting state — Constants
        // get patched in afterward; everything else is overwritten by
        // its kernel.
        let buffer = stream
            .alloc_zeros::<f32>(n_f32.max(4))
            .expect("rlx-cuda: device allocation failed");
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
