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
        let len = data.len().min(cap);
        unsafe {
            let base = (self.buffer.contents() as *mut u8).add(off);
            match dt {
                DType::F32 => {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), base as *mut f32, len);
                }
                DType::F16 => {
                    let dst = std::slice::from_raw_parts_mut(base as *mut half::f16, len);
                    for (i, &v) in data.iter().take(len).enumerate() {
                        dst[i] = half::f16::from_f32(v);
                    }
                }
                _ => {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), base as *mut f32, len);
                }
            }
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
}
