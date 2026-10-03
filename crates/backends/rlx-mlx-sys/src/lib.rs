// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native MLX dependency for RLX: vendored C++ (`vendor/mlx`), static
//! `libmlx.a`, and the `rlx_mlx_shim` C ABI compiled in `build.rs`.
//!
//! Higher-level graph lowering lives in [`rlx-mlx`](../rlx-mlx).

// `rlx_mlx_host` is set by this crate's own build.rs — the one place that
// decides whether libmlx was cross-compiled at all. Gating on it rather than on
// a repeated `target_os` list is what keeps this module from disagreeing with
// the archive that is (or is not) there to link against.
#[cfg(rlx_mlx_host)]
pub mod ffi;

/// Ensures this crate is linked so `build.rs` native artifacts propagate.
#[inline]
pub fn link_ensure() {}
