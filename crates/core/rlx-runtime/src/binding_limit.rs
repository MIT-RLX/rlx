// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whether a graph keeps its arena in one buffer binding — for a caller that
//! can split its work (tiles, windows) to stay there. Ungated, unlike
//! [`arena_budget`](crate::arena_budget), so the browser has it.

use rlx_driver::Device;
use rlx_ir::Graph;

/// The largest arena `device` keeps in one buffer binding, where the backend
/// has such a limit and its device is up: wgpu's
/// `maxStorageBufferBindingSize` (128 MiB guaranteed, 1–4 GiB typical). An
/// arena past it still compiles, striped, but every op whose operands span
/// two stripes runs on the host — so keep [`binding_arena_bytes`] under this.
pub fn device_binding_limit(device: Device) -> Option<u64> {
    match device {
        #[cfg(feature = "gpu")]
        Device::Gpu => rlx_wgpu::arena_binding_limit(),
        _ => None,
    }
}

/// Bytes at the end of each stripe of a striped arena that hold no tensors
/// (rlx-wgpu's staging reserve: 768 MiB natively, 64 MiB in wasm). A tensor
/// fits a stripe when it is at most the binding limit less this.
pub fn device_stripe_reserve(device: Device) -> Option<u64> {
    match device {
        #[cfg(feature = "gpu")]
        Device::Gpu => Some(rlx_wgpu::buffer::shard_stage_reserve() as u64),
        _ => None,
    }
}

/// The arena `graph` plans to on `device`, by that backend's own passes and
/// planner, where [`device_binding_limit`] has a limit for it.
pub fn binding_arena_bytes(graph: &Graph, device: Device) -> Option<usize> {
    match device {
        #[cfg(feature = "gpu")]
        Device::Gpu => Some(rlx_wgpu::planned_arena_bytes(graph.clone())),
        _ => {
            let _ = graph;
            None
        }
    }
}

/// The largest single tensor `graph` holds on `device` (an activation's
/// slot, by that backend's planner — half-packed activations at two bytes),
/// where [`device_binding_limit`] has a limit for it: a striped arena needs
/// each tensor inside one stripe.
pub fn binding_largest_tensor_bytes(graph: &Graph, device: Device) -> Option<usize> {
    match device {
        #[cfg(feature = "gpu")]
        Device::Gpu => Some(rlx_wgpu::planned_largest_tensor_bytes(graph.clone())),
        _ => {
            let _ = graph;
            None
        }
    }
}

/// The longest texture side `graph` needs on WebGL2, where every tensor is
/// one texture and a browser caps each side at `MAX_TEXTURE_SIZE` (16384 on
/// most desktops, 4096–8192 on some phones). `None` when the plan fails.
#[cfg(feature = "opengl")]
pub fn webgl_texture_side(graph: &Graph) -> Option<usize> {
    rlx_webgl::build_plan(graph)
        .ok()
        .map(|p| p.max_texture_side())
}
