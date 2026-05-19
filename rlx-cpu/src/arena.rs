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

//! Arena allocator — ONE allocation, zero per-call overhead.
//!
//! The memory planner computes the total arena size and per-buffer offsets
//! at compile time. At runtime, the arena is allocated once and slices
//! are handed out by offset. Between forward calls, just reset the
//! generation counter — no deallocation, no reallocation.

use rlx_ir::NodeId;
use rlx_opt::memory::MemoryPlan;

/// Pre-allocated memory arena for graph execution.
#[derive(Clone)]
pub struct Arena {
    buf: Vec<u8>,
    plan: MemoryPlan,
}

impl Arena {
    /// Allocate arena from a memory plan.
    pub fn from_plan(plan: MemoryPlan) -> Self {
        let buf = vec![0u8; plan.arena_size];
        Self { buf, plan }
    }

    /// Total arena size in bytes.
    pub fn size(&self) -> usize {
        self.plan.arena_size
    }

    /// Get a mutable f32 slice for a node's buffer.
    ///
    /// # Panics
    /// Panics if the node has no buffer assignment.
    pub fn slice_mut(&mut self, id: NodeId) -> &mut [f32] {
        let slot = self
            .plan
            .assignments
            .get(&id)
            .unwrap_or_else(|| panic!("no buffer for {id}"));
        let bytes = &mut self.buf[slot.offset..slot.offset + slot.size];
        // SAFETY: buf is aligned to at least 1, but we need f32 alignment.
        // The memory planner aligns to 64 bytes, so this is safe.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, slot.size / 4) }
    }

    /// Get a read-only f32 slice for a node's buffer.
    pub fn slice(&self, id: NodeId) -> &[f32] {
        let slot = self
            .plan
            .assignments
            .get(&id)
            .unwrap_or_else(|| panic!("no buffer for {id}"));
        let bytes = &self.buf[slot.offset..slot.offset + slot.size];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, slot.size / 4) }
    }

    /// Get a mutable f64 slice for a node's buffer.
    ///
    /// # Panics
    /// Panics if the node has no buffer assignment, or if the slot's
    /// byte size is not 8-aligned.
    pub fn slice_mut_f64(&mut self, id: NodeId) -> &mut [f64] {
        let slot = self
            .plan
            .assignments
            .get(&id)
            .unwrap_or_else(|| panic!("no buffer for {id}"));
        debug_assert!(
            slot.size.is_multiple_of(8),
            "slice_mut_f64: slot {} has size {} not divisible by 8",
            id,
            slot.size
        );
        let bytes = &mut self.buf[slot.offset..slot.offset + slot.size];
        // SAFETY: planner aligns slots to 64 bytes ⇒ f64-aligned.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f64, slot.size / 8) }
    }

    /// Get a read-only f64 slice for a node's buffer.
    pub fn slice_f64(&self, id: NodeId) -> &[f64] {
        let slot = self
            .plan
            .assignments
            .get(&id)
            .unwrap_or_else(|| panic!("no buffer for {id}"));
        debug_assert!(
            slot.size.is_multiple_of(8),
            "slice_f64: slot {} has size {} not divisible by 8",
            id,
            slot.size
        );
        let bytes = &self.buf[slot.offset..slot.offset + slot.size];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, slot.size / 8) }
    }

    /// Check if a node has a buffer assignment.
    pub fn has_buffer(&self, id: NodeId) -> bool {
        self.plan.assignments.contains_key(&id)
    }

    /// Get a raw pointer + length for a node's buffer.
    /// SAFETY: caller must ensure no aliasing writes to the same buffer.
    pub fn raw_ptr(&self, id: NodeId) -> (*mut f32, usize) {
        let slot = self
            .plan
            .assignments
            .get(&id)
            .unwrap_or_else(|| panic!("no buffer for {id}"));
        let ptr = unsafe { self.buf.as_ptr().add(slot.offset) as *mut f32 };
        (ptr, slot.size / 4)
    }

    /// The execution schedule from the memory plan.
    pub fn schedule(&self) -> &[NodeId] {
        &self.plan.schedule
    }

    /// Byte offset of a node's buffer within the arena.
    pub fn byte_offset(&self, id: NodeId) -> usize {
        self.plan
            .assignments
            .get(&id)
            .map(|s| s.offset)
            .unwrap_or(usize::MAX)
    }

    /// Raw mutable access to the arena buffer (for thunk executor).
    pub fn raw_buf_mut(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    /// Read-only access to the arena buffer (for typed reads).
    pub fn raw_buf(&self) -> &[u8] {
        &self.buf
    }

    /// Raw pointer to arena start (for zero-copy output reads).
    pub fn raw_buf_mut_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_opt::memory::BufferSlot;
    use std::collections::HashMap;

    #[test]
    fn arena_slice_access() {
        let plan = MemoryPlan {
            arena_size: 1024,
            assignments: {
                let mut m = HashMap::new();
                m.insert(
                    NodeId(0),
                    BufferSlot {
                        offset: 0,
                        size: 256,
                    },
                );
                m.insert(
                    NodeId(1),
                    BufferSlot {
                        offset: 256,
                        size: 512,
                    },
                );
                m
            },
            schedule: vec![NodeId(0), NodeId(1)],
        };

        let mut arena = Arena::from_plan(plan);
        let s0 = arena.slice_mut(NodeId(0));
        assert_eq!(s0.len(), 64); // 256 bytes / 4 bytes per f32
        s0[0] = 42.0;

        let s1 = arena.slice_mut(NodeId(1));
        assert_eq!(s1.len(), 128); // 512 / 4

        // s0's data persists
        let s0_read = arena.slice(NodeId(0));
        assert_eq!(s0_read[0], 42.0);
    }
}
