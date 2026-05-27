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

//! Native MLX dependency for RLX: vendored C++ (`vendor/mlx`), static
//! `libmlx.a`, and the `rlx_mlx_shim` C ABI compiled in `build.rs`.
//!
//! Higher-level graph lowering lives in [`rlx-mlx`](../rlx-mlx).

#[cfg(target_os = "macos")]
pub mod ffi;

/// Ensures this crate is linked so `build.rs` native artifacts propagate.
#[inline]
pub fn link_ensure() {}
