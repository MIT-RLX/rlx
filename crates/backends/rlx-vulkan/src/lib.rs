// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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
pub mod host_stage;
pub mod kernels;
pub mod shaders;
pub mod spd;
pub mod unfuse;
pub mod vmath;

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
