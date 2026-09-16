// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The resident-KV row write must stay in place, and stay in its dtype.**
//!
//! MLX was the one backend whose `feed_kv_row` did not write a row. It read the
//! whole output and the whole cache back to the host as f32, memcpy'd one row
//! between them, and rebuilt the cache with
//! `Array::from_f32_slice(.., DType::F32)`. Three defects rode along:
//!
//!   1. **O(cache) per handle per token, through the host** — the exact
//!      O(context) traffic residency exists to remove, and worse than the
//!      `concat` it replaced. Every other backend does a single-row device
//!      write (rlx-cuda / rlx-rocm D2D, rlx-metal / rlx-vulkan in-arena memcpy).
//!   2. **dtype was hard-coded to F32.** Latent, not live: `bind_gpu_handle`
//!      only builds F32 handles today, so nothing could observe it. What it
//!      did do was make the dtype of a resident cache a property of
//!      `feed_kv_row` rather than of the cache — a trap set for the first
//!      non-F32 handle.
//!   3. the cache was *replaced*, not written, so nothing else holding the
//!      handle saw the update.
//!
//! Measured on an M4 Pro with a 4096x1024 F32 cache (Qwen3-0.6B-shaped
//! per-layer KV), both arms warmed before timing: **751us/feed -> 11.3us/feed,
//! ~60x** (57-67x across runs — the spread is all in the old arm, which moved
//! ~33MB of host traffic per feed).
//!
//! There is no per-row dtype penalty: warm steady state is ~13us/row for an F32
//! leaf, an F16 leaf and a cast-built F16 handle alike. A cast-built handle
//! pays one 19.6ms materialization on its FIRST write and nothing after.
//!
//! `rlx-runtime/tests/kv_resident_row_feed.rs` covers accumulation end-to-end,
//! but builds its cache as F32 and only checks values — so it saw neither (2)
//! nor (3). This file pins what that test cannot: the dtype survives a write,
//! and the write is visible through a shared handle.

#![cfg(target_vendor = "apple")]

use rlx_ir::DType;
use rlx_mlx::Array;

/// `to_bytes` returns the array's *native* bytes, so its length over the
/// element count is the itemsize — the only dtype probe the Rust wrapper
/// currently offers.
fn itemsize(a: &Array) -> usize {
    a.to_bytes().expect("to_bytes").len() / a.num_elements().expect("num_elements")
}

/// An F16 cache stays F16 across a row write.
///
/// Scope, stated precisely: this exercises the `copy_row_inplace` primitive,
/// not `feed_kv_row` itself — the primitive is new, so it has no "before" to
/// have failed against. What it pins is the property `feed_kv_row` now leans
/// on. The old rebuild spelled the dtype out as a literal (`DType::F32`) and
/// could not have preserved F16 by construction; here the dtype is never named
/// at all, because the shim scales element offsets by the array's own
/// `itemsize()`.
#[test]
fn a_row_write_preserves_a_half_precision_cache() {
    let row_elems = 4usize;
    let rows = 3usize;

    // Cache [3, 4] F16, all zeros; row source [1, 4] F16.
    let mut cache = Array::from_f32_slice(
        &vec![0.0f32; rows * row_elems],
        &[rows, row_elems],
        DType::F16,
    )
    .expect("cache");
    let src =
        Array::from_f32_slice(&[1.5, 2.5, 3.5, 4.5], &[1, row_elems], DType::F16).expect("src");

    assert_eq!(itemsize(&cache), 2, "cache did not start as F16");

    // Destination is row 1 of the cache.
    let dst_off = row_elems;
    cache
        .copy_row_inplace(dst_off, &src, 0, row_elems)
        .expect("copy_row_inplace");

    assert_eq!(
        itemsize(&cache),
        2,
        "the cache widened to F32 on a row write — the resident handle changed dtype \
         underneath its owner, which is what the old from_f32_slice(.., DType::F32) \
         rebuild did on every fed token"
    );

    let got = cache.to_f32().expect("to_f32");
    assert_eq!(
        got,
        vec![0.0, 0.0, 0.0, 0.0, 1.5, 2.5, 3.5, 4.5, 0.0, 0.0, 0.0, 0.0],
        "row 1 should hold the source row and nothing else should move"
    );
}

/// The write lands in the buffer, so a handle sharing it sees the change.
///
/// `clone_handle` shares the underlying MLX array rather than copying it. That
/// is what makes a *resident* cache resident — the old code replaced the array
/// instead, so any other holder kept seeing the pre-feed contents.
#[test]
fn a_row_write_is_visible_through_a_shared_handle() {
    let row_elems = 2usize;
    let mut cache =
        Array::from_f32_slice(&[0.0, 0.0, 0.0, 0.0], &[2, row_elems], DType::F32).expect("cache");
    let observer = cache.clone_handle().expect("clone_handle");
    let src = Array::from_f32_slice(&[7.0, 8.0], &[1, row_elems], DType::F32).expect("src");

    cache
        .copy_row_inplace(row_elems, &src, 0, row_elems)
        .expect("copy_row_inplace");

    assert_eq!(
        observer.to_f32().expect("to_f32"),
        vec![0.0, 0.0, 7.0, 8.0],
        "the shared handle did not see the write, so this was a replace and not an \
         in-place update"
    );
}

/// Out-of-range writes are refused rather than corrupting a neighbour.
///
/// The whole point of writing straight into the buffer is that there is no
/// bounds check between here and someone else's memory.
#[test]
fn an_out_of_range_row_is_rejected() {
    let mut cache = Array::from_f32_slice(&[0.0; 4], &[2, 2], DType::F32).expect("cache");
    let src = Array::from_f32_slice(&[1.0, 2.0], &[1, 2], DType::F32).expect("src");

    // dst row 2 does not exist in a 2-row cache.
    assert!(
        cache.copy_row_inplace(4, &src, 0, 2).is_err(),
        "a write past the end of the cache was accepted"
    );
    assert!(
        cache.copy_row_inplace(0, &src, 2, 2).is_err(),
        "a read past the end of the source was accepted"
    );
}

/// A dtype mismatch is refused rather than reinterpreting bits.
#[test]
fn a_dtype_mismatch_is_rejected() {
    let mut cache = Array::from_f32_slice(&[0.0; 4], &[2, 2], DType::F32).expect("cache");
    let src = Array::from_f32_slice(&[1.0, 2.0], &[1, 2], DType::F16).expect("src");
    assert!(
        cache.copy_row_inplace(0, &src, 0, 2).is_err(),
        "an F16 row was copied into an F32 cache as raw bytes"
    );
}
