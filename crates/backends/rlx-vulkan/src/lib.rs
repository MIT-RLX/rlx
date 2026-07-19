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

//! RLX native Vulkan compute backend.
//!
//! A from-scratch Vulkan compute backend built directly on `ash` (raw
//! Vulkan) with hand-written GLSL compute kernels compiled to SPIR-V at
//! build time and embedded in the binary. Unlike `rlx-wgpu` (which can
//! reach Vulkan via the wgpu portability layer), this backend owns the
//! Vulkan instance/device/queue, its own arena `VkBuffer`, descriptor
//! sets, and compute pipelines — the dedicated `Device::Vulkan` path.
//!
//! Layout mirrors the other native GPU backends (rlx-cuda / rlx-rocm):
//! - `device`  — Vulkan instance/physical-device/device/queue singleton
//!   (dynamic-loaded; gracefully unavailable with no driver)
//! - `shaders` — embedded SPIR-V blobs (built from `shaders/*.comp`)
//! - `kernels` — per-kernel compute-pipeline cache
//! - `buffer`  — host-visible f32 arena + memory plan mapping
//! - `backend` — `VulkanExecutable`: compile a graph → schedule → run

pub mod backend;
pub mod buffer;
pub mod device;
pub mod host;
pub mod kernels;
pub mod shaders;
pub mod spd;

/// True if a Vulkan compute device is reachable on this system. The
/// runtime registry only registers `Device::Vulkan` when this returns
/// `true`, so hosts with no Vulkan driver (e.g. macOS without MoltenVK)
/// fall through cleanly instead of panicking.
pub fn is_available() -> bool {
    device::vulkan_device().is_some()
}

/// Human-readable name of the selected Vulkan physical device, if any.
pub fn device_name() -> Option<String> {
    device::vulkan_device().map(|d| d.name.clone())
}
