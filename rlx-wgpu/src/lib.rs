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

//! RLX wgpu backend — cross-platform GPU execution via the `wgpu`
//! Rust crate (Metal on macOS, Vulkan on Linux, DX12 on Windows,
//! WebGPU in browsers).
//!
//! Compared to rlx-metal: same overall shape (device singleton, buffer
//! arena, per-op compute pipelines, command-buffer-per-forward-pass)
//! but with WGSL kernels and the wgpu Rust API instead of MSL + the
//! `metal` crate. Pure Rust deps — no FFI / submodules to manage.
//!
//! Layout:
//! - `device`   — wgpu instance/adapter/device singleton (sync wrapper)
//! - `buffer`   — typed GPU buffer + arena
//! - `kernels`  — WGSL source strings + per-kernel pipeline cache
//! - `backend`  — Backend trait impl + per-op dispatch

pub mod backend;
pub mod buffer;
pub mod device;
pub mod gdn_host;
pub mod llada2_gate_host;
pub mod gguf_host;
pub mod training_bwd_host;
pub mod kernels;
pub mod unfuse;
#[cfg(feature = "splat")]
pub mod splat;
#[cfg(feature = "native-splat")]
pub mod splat_native;

/// True if a wgpu adapter is reachable on this system. Always
/// available at the crate level; the runtime registry only registers
/// the backend when this returns `true` so tests on weird CI machines
/// without a GPU don't trip up.
pub fn is_available() -> bool {
    device::wgpu_device().is_some()
}

pub use device::{is_vulkan_available, select_vulkan_backend};
