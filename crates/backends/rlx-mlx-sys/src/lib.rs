// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native MLX dependency for RLX: vendored C++ (`vendor/mlx`), static
//! `libmlx.a`, and the `rlx_mlx_shim` C ABI compiled in `build.rs`.
//!
//! Higher-level graph lowering lives in [`rlx-mlx`](../rlx-mlx).

#[cfg(any(
    target_os = "macos",
    target_os = "linux",
    target_os = "windows",
    target_os = "ios"
))]
pub mod ffi;

/// Ensures this crate is linked so `build.rs` native artifacts propagate.
#[inline]
pub fn link_ensure() {}
