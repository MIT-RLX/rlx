// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU arena allocator — single Metal buffer with sub-region offsets.
//!
//! Mirrors rlx-cpu's arena: one big allocation, all intermediate tensors
//! get byte offsets within it. Apple Silicon unified memory means the same
//! buffer is accessible from both CPU and GPU with zero copy.

use crate::device::metal_device;
use metal::Buffer;
use rlx_ir::{DType, Graph, NodeId};
use rlx_opt::memory::MemoryPlan;
use std::collections::HashMap;

pub struct Arena {
    pub buffer: Buffer,
    pub size_bytes: usize,
    pub offsets: HashMap<NodeId, usize>, // byte offsets per node
    pub element_counts: HashMap<NodeId, usize>, // element counts per node
    pub dtypes: HashMap<NodeId, DType>,  // per-node dtype (for f16 vs f32 dispatch)
}

impl Arena {
    pub fn from_plan(plan: MemoryPlan) -> Self {
        Self::from_plan_with_graph(plan, None)
    }

    /// Build arena from memory plan, recording per-node dtype from the graph.
    /// If `graph` is None, all buffers are assumed F32.
    pub fn from_plan_with_graph(plan: MemoryPlan, graph: Option<&Graph>) -> Self {
        let dev = metal_device().expect("Metal device required for rlx-metal arena");
        if rlx_ir::env::flag("RLX_METAL_ARENA_DIAG") {
            eprintln!(
                "[rlx-metal] arena: {} bytes ({:.1} MiB) over {} slots{}",
                plan.arena_size,
                plan.arena_size as f64 / (1024.0 * 1024.0),
                plan.assignments.len(),
                if plan.arena_size >= (1u64 << 32) as usize {
                    "  <-- >=4GiB: forces thunks_only_big_arena (no MPSGraph fusion)"
                } else {
                    ""
                }
            );
        }
        let buffer = dev.alloc_shared(plan.arena_size.max(64));

        let mut offsets = HashMap::with_capacity(plan.assignments.len());
        let mut element_counts = HashMap::with_capacity(plan.assignments.len());
        let mut dtypes = HashMap::with_capacity(plan.assignments.len());
        for (node_id, slot) in &plan.assignments {
            offsets.insert(*node_id, slot.offset);
            // Element count derived from byte size and dtype
            let dt = graph
                .map(|g| g.node(*node_id).shape.dtype())
                .unwrap_or(DType::F32);
            let elem_size = dt.size_bytes();
            element_counts.insert(*node_id, slot.size / elem_size.max(1));
            dtypes.insert(*node_id, dt);
        }
        Self {
            buffer,
            size_bytes: plan.arena_size,
            offsets,
            element_counts,
            dtypes,
        }
    }

    pub fn has_buffer(&self, id: NodeId) -> bool {
        self.offsets.contains_key(&id)
    }

    pub fn byte_offset(&self, id: NodeId) -> usize {
        *self.offsets.get(&id).expect("node not in arena")
    }

    pub fn dtype(&self, id: NodeId) -> DType {
        self.dtypes.get(&id).copied().unwrap_or(DType::F32)
    }

    /// Get a CPU-side mutable slice for the node's region as f32. Only valid
    /// when the node's dtype is F32 (debug-asserted).
    pub fn slice_mut(&mut self, id: NodeId) -> &mut [f32] {
        debug_assert_eq!(self.dtype(id), DType::F32);
        let off = self.byte_offset(id);
        let len = *self.element_counts.get(&id).unwrap_or(&0);
        unsafe {
            let ptr = self.buffer.contents() as *mut u8;
            std::slice::from_raw_parts_mut(ptr.add(off) as *mut f32, len)
        }
    }

    pub fn slice(&self, id: NodeId) -> &[f32] {
        debug_assert_eq!(self.dtype(id), DType::F32);
        let off = self.byte_offset(id);
        let len = *self.element_counts.get(&id).unwrap_or(&0);
        unsafe {
            let ptr = self.buffer.contents() as *const u8;
            std::slice::from_raw_parts(ptr.add(off) as *const f32, len)
        }
    }

    /// Read the node's data as f32 regardless of native precision (converts
    /// f16 → f32 on the fly). Used at graph output boundary.
    pub fn read_as_f32(&self, id: NodeId) -> Vec<f32> {
        let dt = self.dtype(id);
        let off = self.byte_offset(id);
        let len = *self.element_counts.get(&id).unwrap_or(&0);
        unsafe {
            let base = (self.buffer.contents() as *const u8).add(off);
            match dt {
                DType::F32 => std::slice::from_raw_parts(base as *const f32, len).to_vec(),
                DType::F16 => {
                    let src = std::slice::from_raw_parts(base as *const half::f16, len);
                    src.iter().map(|h| h.to_f32()).collect()
                }
                // Host f32-lane interface for interleaved complex storage:
                // C64 = 2 lanes/elem, C128 = 4 lanes/elem.
                DType::C64 | DType::C128 => {
                    let lanes = len * (dt.size_bytes() / 4).max(1);
                    std::slice::from_raw_parts(base as *const f32, lanes).to_vec()
                }
                _ => std::slice::from_raw_parts(base as *const f32, len).to_vec(),
            }
        }
    }

    /// Write f32 data, converting to the node's native dtype.
    /// Used at graph input/param boundary.
    pub fn write_from_f32(&mut self, id: NodeId, data: &[f32]) {
        let dt = self.dtype(id);
        let off = self.byte_offset(id);
        let cap = *self.element_counts.get(&id).unwrap_or(&0);
        unsafe {
            let base = (self.buffer.contents() as *mut u8).add(off);
            match dt {
                DType::F32 => {
                    let len = data.len().min(cap);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), base as *mut f32, len);
                }
                DType::F16 => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base as *mut half::f16, len);
                    if len >= 1 << 20 {
                        use rayon::prelude::*;
                        dst.par_iter_mut()
                            .zip(&data[..len])
                            .for_each(|(d, &v)| *d = half::f16::from_f32(v));
                    } else {
                        for (i, &v) in data.iter().take(len).enumerate() {
                            dst[i] = half::f16::from_f32(v);
                        }
                    }
                }
                DType::BF16 => {
                    // Parallel f32→bf16 for large params (MXFP4 expert scales:
                    // ~20M elem/param, serial was ~190ms each = the dominant expert
                    // upload cost). Disjoint element writes into the unified-memory
                    // arena → safe + bit-identical to the serial loop.
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base as *mut half::bf16, len);
                    if len >= 1 << 20 {
                        use rayon::prelude::*;
                        dst.par_iter_mut()
                            .zip(&data[..len])
                            .for_each(|(d, &v)| *d = half::bf16::from_f32(v));
                    } else {
                        for (i, &v) in data.iter().take(len).enumerate() {
                            dst[i] = half::bf16::from_f32(v);
                        }
                    }
                }
                // Integer-typed inputs (token IDs, position indices) get
                // cast from f32 → int. The previous fallthrough memcpy
                // bit-pattern-reinterpreted the floats as ints, which
                // produced stable garbled-token streams from gather/take.
                DType::I32 => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base as *mut i32, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = v as i32;
                    }
                }
                DType::I64 => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base as *mut i64, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = v as i64;
                    }
                }
                DType::U32 => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base as *mut u32, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = v as u32;
                    }
                }
                DType::I16 => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base as *mut i16, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = v as i16;
                    }
                }
                DType::I8 => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base as *mut i8, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = v as i8;
                    }
                }
                DType::U8 => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = v as u8;
                    }
                }
                DType::Bool => {
                    let len = data.len().min(cap);
                    let dst = std::slice::from_raw_parts_mut(base, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = if v != 0.0 { 1 } else { 0 };
                    }
                }
                // Interleaved complex: host feeds f32 lanes (C64=2, C128=4
                // per element). `cap` is the complex-element count.
                DType::C64 | DType::C128 => {
                    let lane_cap = cap * (dt.size_bytes() / 4).max(1);
                    let len = data.len().min(lane_cap);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), base as *mut f32, len);
                }
                // F64: two f32 lanes per element when fed via the f32 host path.
                DType::F64 => {
                    let lane_cap = cap * 2;
                    let len = data.len().min(lane_cap);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), base as *mut f32, len);
                }
            }
        }
    }

    /// Copy one arena node's f32 payload into another (unified-memory memcpy).
    pub fn copy_node_f32(&self, dst: NodeId, src: NodeId) {
        let dst_len = *self.element_counts.get(&dst).unwrap_or(&0);
        let src_len = *self.element_counts.get(&src).unwrap_or(&0);
        self.copy_node_f32_prefix(dst, src, dst_len.min(src_len));
    }

    /// Copy `n` f32 from `src` (starting at element `src_elem`) into `dst`
    /// (starting at `dst_elem`), clamped to both node element counts. Used by the
    /// resident KV *row* feed to drop one new-token row (output row `upper`) into
    /// the resident `past_k_*` slot at the active row, in unified memory.
    pub fn copy_node_f32_range(
        &self,
        dst: NodeId,
        dst_elem: usize,
        src: NodeId,
        src_elem: usize,
        n: usize,
    ) {
        let dst_cap = *self.element_counts.get(&dst).unwrap_or(&0);
        let src_cap = *self.element_counts.get(&src).unwrap_or(&0);
        if n == 0 || dst_elem + n > dst_cap || src_elem + n > src_cap {
            return;
        }
        // Byte-width per element from the node dtype (f16=2, f32=4). The KV feed
        // copies same-dtype tensors (new K/V output row → past K/V input row), so
        // a raw byte copy is exact; sizing by the actual dtype is what makes an
        // f16 KV cache work — the old `*f32` (4-byte) offset math corrupted it.
        let elem_bytes = self.dtype(dst).size_bytes().max(1);
        debug_assert_eq!(
            self.dtype(dst),
            self.dtype(src),
            "copy_node_f32_range: src/dst dtype mismatch (raw byte copy assumes equal dtype)"
        );
        let dst_off = self.byte_offset(dst);
        let src_off = self.byte_offset(src);
        unsafe {
            let base = self.buffer.contents() as *mut u8;
            let src_p = base.add(src_off + src_elem * elem_bytes) as *const u8;
            let dst_p = base.add(dst_off + dst_elem * elem_bytes);
            if !std::ptr::eq(src_p, dst_p) {
                std::ptr::copy(src_p, dst_p, n * elem_bytes);
            }
        }
    }

    /// Copy the first `elems` floats from `src` into `dst` (KV prefix after active-extent).
    pub fn copy_node_f32_prefix(&self, dst: NodeId, src: NodeId, elems: usize) {
        if elems == 0 {
            return;
        }
        let dst_off = self.byte_offset(dst);
        let src_off = self.byte_offset(src);
        let dst_cap = *self.element_counts.get(&dst).unwrap_or(&0);
        let src_cap = *self.element_counts.get(&src).unwrap_or(&0);
        let len = elems.min(dst_cap).min(src_cap);
        if len == 0 {
            return;
        }
        unsafe {
            let base = self.buffer.contents() as *mut u8;
            std::ptr::copy(
                base.add(src_off) as *const f32,
                base.add(dst_off) as *mut f32,
                len,
            );
        }
    }

    /// Copy raw bytes into the node's arena slot (U8/I8 packed weights).
    pub fn write_bytes(&mut self, id: NodeId, data: &[u8]) {
        let off = self.byte_offset(id);
        let cap = *self.element_counts.get(&id).unwrap_or(&0);
        let len = data.len().min(cap);
        unsafe {
            let base = (self.buffer.contents() as *mut u8).add(off);
            std::ptr::copy_nonoverlapping(data.as_ptr(), base, len);
        }
    }

    /// Copy raw bytes into a sub-range of the node's arena slot, starting
    /// `byte_offset` bytes into it. Used for incremental per-slot uploads of a
    /// large packed-expert residency buffer (write one changed slot instead of the
    /// whole buffer). Zero-copy on Apple unified memory — the GPU reads the same
    /// bytes. Bounded by the slot's element/byte capacity.
    pub fn write_bytes_at(&mut self, id: NodeId, byte_offset: usize, data: &[u8]) {
        let off = self.byte_offset(id);
        // element_counts is in ELEMENTS; convert to a byte cap so this is correct
        // for U8 codes and BF16 scales alike.
        let byte_cap =
            *self.element_counts.get(&id).unwrap_or(&0) * self.dtype(id).size_bytes().max(1);
        if byte_offset >= byte_cap {
            return;
        }
        let len = data.len().min(byte_cap - byte_offset);
        unsafe {
            let base = (self.buffer.contents() as *mut u8).add(off + byte_offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), base, len);
        }
    }

    /// Copy a node's byte payload from another arena (packed param sharing).
    pub fn copy_node_bytes_from(&self, dst: NodeId, src_arena: &Arena, src: NodeId) {
        let dst_off = self.byte_offset(dst);
        let src_off = src_arena.byte_offset(src);
        let dst_cap = *self.element_counts.get(&dst).unwrap_or(&0);
        let src_cap = src_arena.element_counts.get(&src).copied().unwrap_or(0);
        let elems = dst_cap.min(src_cap);
        if elems == 0 {
            return;
        }
        // `element_counts` are ELEMENT counts, so scale by the dtype width to get
        // bytes — otherwise an F32 param copies only 1 of every 4 bytes (a scalar
        // scale becomes a denormal ≈0 → div-by-zero → NaN on a reused/cloned graph).
        let elem_size = self.dtype(dst).size_bytes().max(1);
        let bytes = elems * elem_size;
        unsafe {
            let dst_base = self.buffer.contents() as *mut u8;
            let src_base = src_arena.buffer.contents() as *const u8;
            std::ptr::copy(src_base.add(src_off), dst_base.add(dst_off), bytes);
        }
    }
}
