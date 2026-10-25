// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A byte range in the activation arena, and the one definition of overlap.
//!
//! This lives in `rlx-ir` because every backend already depends on it, and the
//! question "do these two arena slots share a byte?" was being answered by four
//! separate hand-rolled copies — `rlx-compile`'s in-place safety pass and
//! `RLX_MEM_VERIFY`, `rlx-metal`'s `mlp_f32_ranges_overlap`, `rlx-wgpu`'s
//! `arena_tensors_overlap`, and two inline comparisons in its hazard scanner.
//!
//! They did not agree, and the disagreement is not cosmetic. Some compare
//! offsets for equality, which cannot see a partial overlap at all; some treat a
//! zero-length slot as one byte so it still conflicts, and some treat it as
//! conflicting with nothing. A guard built on the wrong one of those reports
//! clean on a real hazard — which is how `rlx-wgpu`'s split-QKV elision wrote Q,
//! K and V into a single offset while its own scanner said "0 not
//! view-explained".
//!
//! The convention here is **half-open and zero-length-empty**: `[off, off+len)`,
//! and a zero-length range overlaps nothing, including itself. Two sites
//! deliberately keep the opposite convention and say so at the call site —
//! `rlx-compile`'s planner uses `.max(1)` so a zero-size buffer still occupies a
//! byte and cannot be handed out twice. That is a planning decision, not a
//! disagreement about geometry, and `rlx-wgpu`'s `empty_tensor_arena_slot` test
//! depends on it.

/// A half-open byte range `[off, off + len)` in the activation arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ByteRange {
    /// Offset from the start of the arena.
    pub off: usize,
    /// Length in bytes. A zero-length range overlaps nothing.
    pub len: usize,
}

impl ByteRange {
    /// Construct from an offset and a length in bytes.
    pub fn new(off: usize, len: usize) -> Self {
        Self { off, len }
    }

    /// Construct from an offset and a count of `f32` elements.
    ///
    /// Backends carry activation extents as element counts far more often than
    /// as byte counts, and the `* 4` was being written at each call site.
    pub fn f32_elems(off: usize, elems: usize) -> Self {
        Self {
            off,
            len: elems * 4,
        }
    }

    /// Half-open end offset, saturating.
    pub fn end(&self) -> usize {
        self.off.saturating_add(self.len)
    }

    /// True iff these ranges share at least one byte.
    ///
    /// Zero-length ranges overlap nothing — see the module note for the one
    /// place that deliberately wants the opposite.
    pub fn overlaps(&self, other: &ByteRange) -> bool {
        self.len != 0 && other.len != 0 && self.off < other.end() && other.off < self.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_open_means_abutting_is_not_overlapping() {
        assert!(!ByteRange::new(0, 64).overlaps(&ByteRange::new(64, 64)));
        assert!(ByteRange::new(63, 2).overlaps(&ByteRange::new(64, 2)));
    }

    #[test]
    fn zero_length_overlaps_nothing() {
        assert!(!ByteRange::new(0, 0).overlaps(&ByteRange::new(0, 64)));
        assert!(!ByteRange::new(0, 64).overlaps(&ByteRange::new(0, 0)));
        assert!(!ByteRange::new(0, 0).overlaps(&ByteRange::new(0, 0)));
    }

    /// Equal offsets with non-zero lengths always overlap — the reason
    /// `rlx-metal`'s old `a_off == b_off => true` shortcut was redundant rather
    /// than load-bearing, so dropping it preserved behaviour.
    #[test]
    fn equal_offsets_overlap_without_a_special_case() {
        assert!(ByteRange::new(4096, 8).overlaps(&ByteRange::new(4096, 2048)));
        assert!(ByteRange::new(4096, 2048).overlaps(&ByteRange::new(4096, 8)));
    }

    /// Containment in both directions, which an offset-equality test misses.
    #[test]
    fn containment_is_overlap() {
        let outer = ByteRange::new(1000, 500);
        let inner = ByteRange::new(1200, 50);
        assert!(outer.overlaps(&inner));
        assert!(inner.overlaps(&outer));
    }

    #[test]
    fn end_saturates_instead_of_wrapping() {
        assert_eq!(ByteRange::new(usize::MAX, 8).end(), usize::MAX);
    }

    #[test]
    fn f32_elems_scales_by_four() {
        assert_eq!(ByteRange::f32_elems(16, 32), ByteRange::new(16, 128));
        assert_eq!(ByteRange::f32_elems(16, 0).len, 0);
    }
}
