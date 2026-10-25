// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inter-thunk parallelism (`wave-exec` feature).
//!
//! The executor runs thunks one after another and parallelizes only *inside*
//! each kernel. On dispatch-bound schedules (decode, small models) a single
//! kernel is too small to occupy every core, while its independent siblings
//! — the per-branch projections, norms and activations of a block — wait
//! their turn.
//!
//! This module groups thunks into *waves*: within a run of thunks whose arena
//! accesses are known exactly, each thunk is placed one level after the last
//! earlier thunk it conflicts with (read-after-write, write-after-read or
//! write-after-write on overlapping byte ranges). Every thunk in a level is
//! then independent of the rest of its level, so the level runs concurrently
//! on the rayon pool; levels run in order.
//!
//! Byte ranges, not node ids, carry the dependencies, so the memory planner's
//! slot reuse is handled by construction: a reused slot shows up as an
//! overlap and serializes the two thunks that share it.
//!
//! Any thunk without an access summary in [`thunk_access`] is a barrier and
//! runs on the sequential path. The list is deliberately short — an access
//! summary that under-states what a kernel touches is a data race, so a kind
//! is only listed when its summary mirrors the slices its `exec_*` builds.

use crate::thunk::*;
use std::sync::Arc;

/// Half-open byte range `[lo, hi)` in the arena.
type Span = (usize, usize);

#[inline]
fn span_f32(off: usize, elems: usize) -> Span {
    (off, off.saturating_add(elems.saturating_mul(4)))
}

#[inline]
fn overlaps(a: Span, b: Span) -> bool {
    a.0 < b.1 && b.0 < a.1
}

/// Reads and the single written range of a thunk, or `None` when the kind
/// has no exact summary (it then acts as a barrier).
fn thunk_access(t: &Thunk) -> Option<(Vec<Span>, Span)> {
    Some(match t {
        Thunk::Sgemm { a, b, c, m, k, n } => {
            let (m, k, n) = (*m as usize, *k as usize, *n as usize);
            (
                vec![span_f32(*a, m * k), span_f32(*b, k * n)],
                span_f32(*c, m * n),
            )
        }
        Thunk::DequantMatMulGguf {
            x,
            w_q,
            dst,
            m,
            k,
            n,
            scheme,
        } => {
            let (m, k, n) = (*m as usize, *k as usize, *n as usize);
            let block_elems = scheme.gguf_block_size() as usize;
            if block_elems == 0 {
                return None;
            }
            let w_bytes = (k * n) / block_elems * scheme.gguf_block_bytes() as usize;
            (
                vec![span_f32(*x, m * k), (*w_q, w_q + w_bytes)],
                span_f32(*dst, m * n),
            )
        }
        Thunk::DequantMatMul {
            x,
            w_q,
            scale,
            zp,
            dst,
            m,
            k,
            n,
            block_size,
            is_asymmetric,
        } => {
            let (m, k, n, bs) = (*m as usize, *k as usize, *n as usize, *block_size as usize);
            let nb = k.div_ceil(bs.max(1)) * n;
            let mut r = vec![
                span_f32(*x, m * k),
                (*w_q, w_q + k * n),
                span_f32(*scale, nb),
            ];
            if *is_asymmetric {
                r.push(span_f32(*zp, nb));
            }
            (r, span_f32(*dst, m * n))
        }
        Thunk::DequantMatMulInt4 {
            x,
            w_q,
            scale,
            zp,
            dst,
            m,
            k,
            n,
            block_size,
            is_asymmetric,
        } => {
            let (m, k, n, bs) = (*m as usize, *k as usize, *n as usize, *block_size as usize);
            let nb = k.div_ceil(bs.max(1)) * n;
            let mut r = vec![
                span_f32(*x, m * k),
                (*w_q, w_q + (k * n).div_ceil(2)),
                span_f32(*scale, nb),
            ];
            if *is_asymmetric {
                r.push(span_f32(*zp, nb));
            }
            (r, span_f32(*dst, m * n))
        }
        Thunk::DequantMatMulFp8 {
            x,
            w_q,
            scale,
            dst,
            m,
            k,
            n,
            ..
        } => {
            let (m, k, n) = (*m as usize, *k as usize, *n as usize);
            (
                vec![
                    span_f32(*x, m * k),
                    (*w_q, w_q + k * n),
                    span_f32(*scale, n),
                ],
                span_f32(*dst, m * n),
            )
        }
        Thunk::RmsNorm {
            src,
            g,
            b,
            dst,
            rows,
            h,
            ..
        }
        | Thunk::LayerNorm {
            src,
            g,
            b,
            dst,
            rows,
            h,
            ..
        } => {
            let (rows, h) = (*rows as usize, *h as usize);
            (
                vec![span_f32(*src, rows * h), span_f32(*g, h), span_f32(*b, h)],
                span_f32(*dst, rows * h),
            )
        }
        Thunk::Softmax {
            data, rows, cols, ..
        } => {
            let s = span_f32(*data, *rows as usize * *cols as usize);
            (vec![s], s)
        }
        Thunk::ActivationInPlace { data, len, .. } => {
            let s = span_f32(*data, *len as usize);
            (vec![s], s)
        }
        Thunk::Rope {
            src,
            cos,
            sin,
            dst,
            batch,
            seq,
            hidden,
            cos_len,
            src_row_stride,
            ..
        } => {
            let rows = *batch as usize * *seq as usize;
            let cl = *cos_len as usize;
            (
                vec![
                    span_f32(*src, rows * *src_row_stride as usize),
                    span_f32(*cos, cl),
                    span_f32(*sin, cl),
                ],
                span_f32(*dst, rows * *hidden as usize),
            )
        }
        _ => return None,
    })
}

/// Run one wave member. Covers exactly the kinds [`thunk_access`] summarizes.
fn exec_member(t: &Thunk, base: *mut u8, arena_len: usize) {
    match t {
        Thunk::Sgemm { .. } => exec_sgemm(t, base, arena_len, false),
        Thunk::DequantMatMulGguf { .. } => exec_dequant_mat_mul_gguf(t, base),
        Thunk::DequantMatMul { .. } => exec_dequant_mat_mul(t, base),
        Thunk::DequantMatMulInt4 { .. } => exec_dequant_mat_mul_int4(t, base),
        Thunk::DequantMatMulFp8 { .. } => exec_dequant_mat_mul_fp8(t, base),
        Thunk::RmsNorm { .. } => exec_rms_norm(t, base),
        Thunk::LayerNorm { .. } => exec_layer_norm(t, base),
        Thunk::Softmax { .. } => exec_softmax(t, base),
        Thunk::ActivationInPlace { .. } => exec_activation_in_place(t, base),
        Thunk::Rope { .. } => exec_rope(t, base),
        _ => unreachable!("thunk kind without an access summary in a wave"),
    }
}

/// A run of summarized thunks `[start, end)` and its levels (thunk indices).
#[derive(Debug)]
struct Segment {
    start: usize,
    end: usize,
    levels: Vec<Vec<u32>>,
}

/// Segments that contain at least one level with two or more thunks.
#[derive(Debug, Default)]
pub struct WavePlan {
    segments: Vec<Segment>,
}

/// Longest run levelled as one segment. Levelling is quadratic in the run, and
/// a longer run rarely finds parallelism a shorter one misses.
const MAX_SEGMENT: usize = 256;

impl WavePlan {
    pub fn build(thunks: &[Thunk]) -> Self {
        let mut segments = Vec::new();
        let mut i = 0;
        while i < thunks.len() {
            if thunk_access(&thunks[i]).is_none() {
                i += 1;
                continue;
            }
            let start = i;
            // (thunk index, reads, write, level)
            let mut members: Vec<(usize, Vec<Span>, Span, usize)> = Vec::new();
            while i < thunks.len() && i - start < MAX_SEGMENT {
                if matches!(thunks[i], Thunk::Nop) {
                    i += 1;
                    continue;
                }
                let Some((reads, write)) = thunk_access(&thunks[i]) else {
                    break;
                };
                let level = members
                    .iter()
                    .filter(|(_, r, w, _)| {
                        overlaps(*w, write)
                            || reads.iter().any(|s| overlaps(*w, *s))
                            || r.iter().any(|s| overlaps(write, *s))
                    })
                    .map(|m| m.3 + 1)
                    .max()
                    .unwrap_or(0);
                members.push((i, reads, write, level));
                i += 1;
            }
            let n_levels = members.iter().map(|m| m.3 + 1).max().unwrap_or(0);
            let mut levels = vec![Vec::new(); n_levels];
            for (idx, _, _, l) in &members {
                levels[*l].push(*idx as u32);
            }
            if levels.iter().any(|l| l.len() > 1) {
                segments.push(Segment {
                    start,
                    end: i,
                    levels,
                });
            }
        }
        WavePlan { segments }
    }

    /// Number of thunks that run concurrently with a sibling.
    pub fn parallel_thunks(&self) -> usize {
        self.segments
            .iter()
            .flat_map(|s| s.levels.iter())
            .filter(|l| l.len() > 1)
            .map(|l| l.len())
            .sum()
    }

    /// Index of the segment starting at thunk `i`, searched from `*cursor`.
    #[inline]
    fn segment_at(&self, i: usize, cursor: &mut usize) -> Option<&Segment> {
        while *cursor < self.segments.len() && self.segments[*cursor].start < i {
            *cursor += 1;
        }
        self.segments.get(*cursor).filter(|s| s.start == i)
    }
}

/// Whether wave execution is on (`RLX_CPU_WAVES=0` turns it off for A/B runs).
pub(crate) fn waves_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        rlx_ir::env::flag_or("RLX_CPU_WAVES", true) && crate::pool::num_threads() > 1
    })
}

/// The schedule's plan, built on first use.
pub(crate) fn plan_for(schedule: &ThunkSchedule) -> &Arc<WavePlan> {
    schedule
        .waves
        .get_or_init(|| Arc::new(WavePlan::build(&schedule.thunks)))
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// If a segment starts at thunk `i`, run it and return its end.
pub(crate) fn try_run_segment(
    plan: &WavePlan,
    cursor: &mut usize,
    i: usize,
    thunks: &[Thunk],
    base: *mut u8,
    arena_len: usize,
) -> Option<usize> {
    let seg = plan.segment_at(i, cursor)?;
    let ptr = SendPtr(base);
    for level in &seg.levels {
        match level.as_slice() {
            [] => {}
            [only] => exec_member(&thunks[*only as usize], base, arena_len),
            [first, rest @ ..] => {
                rayon::scope(|s| {
                    for &j in rest {
                        let t = &thunks[j as usize];
                        s.spawn(move |_| {
                            let p = ptr;
                            exec_member(t, p.0, arena_len)
                        });
                    }
                    exec_member(&thunks[*first as usize], ptr.0, arena_len);
                });
            }
        }
    }
    Some(seg.end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(src: usize, dst: usize) -> Thunk {
        Thunk::RmsNorm {
            src,
            g: 10_000,
            b: 10_100,
            dst,
            rows: 1,
            h: 16,
            eps: 1e-6,
        }
    }

    #[test]
    fn independent_thunks_share_a_level() {
        // Two norms of the same input into disjoint outputs, then a third
        // reading the first output.
        let plan = WavePlan::build(&[rms(0, 1000), rms(0, 2000), rms(1000, 3000)]);
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].levels, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn reused_slot_serializes() {
        // The second thunk overwrites the slot the first one reads (WAR).
        let plan = WavePlan::build(&[rms(0, 1000), rms(5000, 0)]);
        assert!(plan.segments.is_empty());
    }

    #[test]
    fn unsummarized_thunk_is_a_barrier() {
        let copy = Thunk::Copy {
            src: 0,
            dst: 9000,
            len: 4,
        };
        let plan = WavePlan::build(&[rms(0, 1000), copy, rms(0, 2000)]);
        assert!(plan.segments.is_empty());
    }
}
