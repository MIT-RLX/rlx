// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Arena allocator — ONE allocation, zero per-call overhead.
//!
//! The memory planner computes the total arena size and per-buffer offsets
//! at compile time. At runtime, the arena is allocated once and slices
//! are handed out by offset. Between forward calls, just reset the
//! generation counter — no deallocation, no reallocation.

use rlx_ir::NodeId;
use rlx_ir::bytes::AlignedBytes;
use rlx_opt::memory::MemoryPlan;

/// Pre-allocated memory arena for graph execution.
///
/// The backing store is [`AlignedBytes`], not `Vec<u8>`: slots are handed out
/// as `&[f32]` / `&[f64]`, and `Vec<u8>` only promises alignment 1. Aligning
/// the planner's *offsets* to 64 does nothing for that — an aligned offset
/// from an unaligned base is still an unaligned address. `AlignedBytes` pins
/// the base at 64 bytes, so `base + offset` is genuinely aligned and the
/// reinterprets below are sound rather than allocator luck.
#[derive(Clone)]
pub struct Arena {
    buf: Store,
    plan: MemoryPlan,
}

/// The arena's bytes: a heap buffer, or a page-aligned mapping whose large
/// weight slots may be shared with other executables
/// ([`crate::shared_params`]).
enum Store {
    Heap(AlignedBytes),
    Mapped(crate::shared_params::MappedArena),
}

impl Store {
    fn slice_of<T: rlx_ir::bytes::LeBytes>(&self, offset: usize, len_bytes: usize) -> &[T] {
        match self {
            Store::Heap(b) => b.slice_of(offset, len_bytes),
            Store::Mapped(m) => {
                let bytes = &m.as_slice()[offset..offset + len_bytes];
                assert_eq!(
                    offset % std::mem::align_of::<T>(),
                    0,
                    "offset {offset} misaligned"
                );
                // SAFETY: the mapping is page-aligned, `offset` is aligned for
                // `T` (asserted), the range is in bounds, `T: LeBytes`.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), len_bytes / T::SIZE) }
            }
        }
    }

    fn slice_of_mut<T: rlx_ir::bytes::LeBytes>(
        &mut self,
        offset: usize,
        len_bytes: usize,
    ) -> &mut [T] {
        match self {
            Store::Heap(b) => b.slice_of_mut(offset, len_bytes),
            Store::Mapped(m) => {
                assert_eq!(
                    offset % std::mem::align_of::<T>(),
                    0,
                    "offset {offset} misaligned"
                );
                let bytes = &mut m.as_mut_slice()[offset..offset + len_bytes];
                // SAFETY: as `slice_of`, and `&mut self` makes it exclusive.
                unsafe {
                    std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), len_bytes / T::SIZE)
                }
            }
        }
    }
}

impl std::ops::Deref for Store {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Store::Heap(b) => b,
            Store::Mapped(m) => m.as_slice(),
        }
    }
}

impl std::ops::DerefMut for Store {
    fn deref_mut(&mut self) -> &mut [u8] {
        match self {
            Store::Heap(b) => b,
            Store::Mapped(m) => m.as_mut_slice(),
        }
    }
}

/// A clone is a private copy: shared weight pages are copied, not shared.
impl Clone for Store {
    fn clone(&self) -> Self {
        Store::Heap(AlignedBytes::from_slice(self))
    }
}

impl Arena {
    /// Allocate arena from a memory plan.
    pub fn from_plan(plan: MemoryPlan) -> Self {
        if rlx_ir::env::var_os("RLX_ARENA_CHECK").is_some() {
            let mut worst = 0usize;
            for (id, s) in &plan.assignments {
                let end = s.offset + s.size;
                if end > plan.arena_size {
                    eprintln!(
                        "[arena] node {id} slot [{}..{}] EXCEEDS arena_size {} by {}",
                        s.offset,
                        end,
                        plan.arena_size,
                        end - plan.arena_size
                    );
                }
                worst = worst.max(end);
            }
            eprintln!(
                "[arena] {} slots, arena_size={}, max slot-end={worst} ({} slack)",
                plan.assignments.len(),
                plan.arena_size,
                plan.arena_size as isize - worst as isize
            );
        }
        Self::from_plan_filled(plan, rlx_ir::env::flag("RLX_POISON_ARENA"))
    }

    /// [`Self::from_plan`] with the fill spelled out. `poison` fills every
    /// byte with `0xFF` — NaN as f32/f64, −1 as integers — instead of zero.
    ///
    /// A zeroed arena makes a slot that nothing wrote look like a plausible
    /// result — an unfed input, a constant a harness forgot to write, an op
    /// that runs as a no-op. Poisoned, it reads NaN. Also exposes a kernel that
    /// reads its own output before writing it (accumulating into it, say).
    /// The CPU executable re-applies the fill at its per-run scratch reset;
    /// GPU host fallbacks (`rlx-gpu-host`) use their own buffers and are not
    /// covered.
    pub fn from_plan_filled(plan: MemoryPlan, poison: bool) -> Self {
        let mut buf = AlignedBytes::zeroed(plan.arena_size);
        if poison {
            buf.as_mut_slice().fill(0xFF);
        }
        Self {
            buf: Store::Heap(buf),
            plan,
        }
    }

    /// [`Self::from_plan`] whose `shared` slots — `(name, dtype tag, bytes,
    /// offset)`, each offset page-aligned — are backed by the `scope`'s shared
    /// copy of that weight ([`crate::shared_params`]). Returns the arena and,
    /// per shared slot, whether its values were already there (another
    /// executable made the copy). `None` when the platform cannot map shared
    /// pages; the caller then builds a plain arena.
    pub fn from_plan_shared(
        plan: MemoryPlan,
        scope: &str,
        shared: &[(String, u8, usize, usize)],
    ) -> Option<(Self, Vec<bool>)> {
        let mut m = crate::shared_params::MappedArena::new(plan.arena_size)?;
        if rlx_ir::env::flag("RLX_POISON_ARENA") {
            m.as_mut_slice().fill(0xFF);
        }
        let mut present = Vec::with_capacity(shared.len());
        for (name, tag, nbytes, offset) in shared {
            // A slot that cannot be mapped stays private (and is uploaded).
            present.push(
                m.map_shared(scope, name, *tag, *nbytes, *offset)
                    .unwrap_or(false),
            );
        }
        Some((
            Self {
                buf: Store::Mapped(m),
                plan,
            },
            present,
        ))
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
        // 64-aligned base + a planner-aligned offset ⇒ f32-aligned address.
        // `slice_of_mut` re-checks the offset and panics rather than handing
        // back a misaligned `&mut [f32]`.
        self.buf.slice_of_mut::<f32>(slot.offset, slot.size)
    }

    /// Get a read-only f32 slice for a node's buffer.
    pub fn slice(&self, id: NodeId) -> &[f32] {
        let slot = self
            .plan
            .assignments
            .get(&id)
            .unwrap_or_else(|| panic!("no buffer for {id}"));
        self.buf.slice_of::<f32>(slot.offset, slot.size)
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
        self.buf.slice_of_mut::<f64>(slot.offset, slot.size)
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
        self.buf.slice_of::<f64>(slot.offset, slot.size)
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
        assert_eq!(
            slot.offset % std::mem::align_of::<f32>(),
            0,
            "raw_ptr: slot offset {} is not 4-aligned",
            slot.offset
        );
        // SAFETY: the arena base is 64-byte aligned and `offset` is 4-aligned
        // (asserted), so the result is a valid, f32-aligned pointer into the
        // allocation. Aliasing is the caller's obligation, as documented.
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

    /// Byte size of a node's arena slot (0 if unassigned / aliased view).
    pub fn byte_size(&self, id: NodeId) -> usize {
        self.plan.assignments.get(&id).map(|s| s.size).unwrap_or(0)
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

    /// The arena hands out `&[f32]` / `&[f64]` views of its byte store, so the
    /// base must be over-aligned. A `Vec<u8>` base (alignment 1 by contract)
    /// made those reinterprets UB no matter how the planner aligned offsets.
    #[test]
    fn arena_base_is_over_aligned_for_typed_slices() {
        let plan = MemoryPlan {
            arena_size: 1024,
            assignments: {
                let mut m = HashMap::new();
                m.insert(
                    NodeId(0),
                    BufferSlot {
                        offset: 0,
                        size: 64,
                    },
                );
                m.insert(
                    NodeId(1),
                    BufferSlot {
                        offset: 64,
                        size: 64,
                    },
                );
                m
            },
            schedule: vec![NodeId(0), NodeId(1)],
        };
        let arena = Arena::from_plan(plan);
        assert_eq!(arena.raw_buf().as_ptr() as usize % 64, 0);
        for id in [NodeId(0), NodeId(1)] {
            assert_eq!(arena.slice(id).as_ptr() as usize % align_of::<f32>(), 0);
            assert_eq!(arena.slice_f64(id).as_ptr() as usize % align_of::<f64>(), 0);
        }
    }

    #[test]
    fn poisoned_arena_reads_nan_until_written() {
        let plan = MemoryPlan {
            arena_size: 128,
            assignments: {
                let mut m = HashMap::new();
                m.insert(
                    NodeId(0),
                    BufferSlot {
                        offset: 0,
                        size: 64,
                    },
                );
                m
            },
            schedule: vec![NodeId(0)],
        };
        let mut arena = Arena::from_plan_filled(plan.clone(), true);
        assert!(arena.slice(NodeId(0)).iter().all(|v| v.is_nan()));
        arena.slice_mut(NodeId(0))[3] = 1.5;
        assert_eq!(arena.slice(NodeId(0))[3], 1.5);
        let clean = Arena::from_plan_filled(plan, false);
        assert!(clean.slice(NodeId(0)).iter().all(|&v| v == 0.0));
    }
}
