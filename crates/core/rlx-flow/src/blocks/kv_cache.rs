// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The decode KV-cache append, shared by every family's decode layer.
//!
//! A decode layer grows its K/V cache by exactly one row per token. Doing that
//! with `concat` re-copies the WHOLE cache every step, so the cost is O(context)
//! for one row of real work — measured at 16% of decode GPU time at ctx 256,
//! 30% at 1024 and ~33% at 2048 (qwen3-0.6B Q4_K_M, Metal): a constant 2
//! dispatches per layer whose time grows with the context while the useful
//! write stays 4 KB. `Op::KvAppend` writes just that row into the cache buffer,
//! which the planner aliases to the output, making the append O(1).
//!
//! Two things must line up for that to be correct, and they live in different
//! crates: the FLOW declares the cache with a spare row, and the LAYER emits the
//! op. [`kv_spare_rows`] and [`append_kv_row`] share one predicate so the two
//! halves cannot drift apart — which they did once already, silently.

use rlx_ir::HirGraphExt;
use rlx_ir::hir::{HirMut, HirNodeId};

/// Opt out of in-place KV append everywhere (`=1`). Shared across families so
/// one switch flips the whole decode path back to `concat` for a bisect.
pub const NO_INPLACE_KV: &str = "RLX_NO_INPLACE_KV";

/// Whether `Op::KvAppend`'s `[..pos + 1]` output can alias a cache of `dims`
/// when appending along `axis`.
///
/// Only when every dim BEFORE the axis is 1. Otherwise the prefix is strided:
/// `[batch, seq, kv_dim]` at axis 1 with `batch > 1` interleaves the sequences,
/// so no single offset into the cache describes the output, and
/// `[batch, heads, seq, dim]` at axis 2 never can. The same line is drawn by
/// `rlx-compile/tests/kv_append_portability.rs`.
pub fn kv_append_can_alias(dims: &[usize], axis: usize) -> bool {
    dims.iter().take(axis).all(|d| *d == 1)
}

/// Spare rows a decode flow should add to its K/V cache, given the dims that
/// PRECEDE the sequence axis (`&[batch]` for the usual `[batch, seq, kv_dim]`).
///
/// `1` opts the flow into the in-place append; `0` keeps `concat`. The result
/// must go into THREE places that have to agree: the declared cache shape, the
/// `past_len` contract (`ModelFlow::bind_decode_inputs_with_capacity`), and the
/// row count the host pads its KV feed to.
pub fn kv_spare_rows(dims_before_seq: &[usize]) -> usize {
    let n = dims_before_seq.len();
    (kv_append_can_alias(dims_before_seq, n) && !rlx_ir::env::flag(NO_INPLACE_KV)) as usize
}

/// Append `row` to the `past` K/V cache along `axis`, in place where the cache
/// contract allows it and by `concat` otherwise.
///
/// `past_len` is [`crate::DecodeBindings::past_len`]: `Some` only when the flow
/// bound its inputs with `bind_decode_inputs_with_capacity`, which is the only
/// signal that the cache carries a spare row — the shape alone cannot say
/// whether its last row is real history or headroom.
///
/// This is deliberately the single decision point for every family. The
/// in-place path landed once and then silently reverted: the layer began gating
/// on the contract while no flow was ever migrated to declare it, so the fast
/// path just stopped being emitted. Nothing failed — `concat` is CORRECT, only
/// O(context) — and decode quietly got slower for everyone. Sharing the
/// predicate with [`kv_spare_rows`] is also what stops the shape and the op from
/// disagreeing, which is the opposite failure and corrupts output rather than
/// merely costing time: a spare row with `concat` puts the new token at
/// `past_len + 1` while every reader looks at `past_len`, so the cache never
/// grows and decode repeats the previous token.
pub fn append_kv_row(
    gb: &mut HirMut<'_>,
    past: HirNodeId,
    row: HirNodeId,
    axis: usize,
    past_len: Option<usize>,
) -> HirNodeId {
    if let Some(pos) = past_len {
        // Read the layout out before mutating: `shape` borrows the module.
        let (cap, outer) = {
            let dims = gb.shape(past);
            let cap = dims.dim(axis).unwrap_static();
            let outer: Vec<usize> = (0..axis).map(|i| dims.dim(i).unwrap_static()).collect();
            (cap, outer)
        };
        if pos < cap && kv_append_can_alias(&outer, axis) && !rlx_ir::env::flag(NO_INPLACE_KV) {
            return gb.kv_append(past, row, axis, pos);
        }
    }
    gb.concat_(vec![past, row], axis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_contiguous_prefix_can_alias() {
        // `[1, seq, kv_dim]` at axis 1 — nothing precedes the axis.
        assert!(kv_append_can_alias(&[1], 1));
        // `[batch > 1, seq, kv_dim]` at axis 1 — sequences interleave.
        assert!(!kv_append_can_alias(&[2], 1));
        // `[1, heads, seq, dim]` at axis 2 — heads interleave.
        assert!(!kv_append_can_alias(&[1, 8], 2));
        // axis 0 is always a contiguous prefix.
        assert!(kv_append_can_alias(&[4, 8], 0));
    }

    #[test]
    fn spare_rows_track_the_alias_predicate() {
        assert_eq!(kv_spare_rows(&[1]), 1, "batch 1 opts in");
        assert_eq!(kv_spare_rows(&[3]), 0, "batched stays on concat");
        assert_eq!(kv_spare_rows(&[1, 8]), 0, "head-major layout cannot alias");
    }
}
