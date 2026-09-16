// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **A schedule target built from a live device, not a table.**
//!
//! CUDA and Metal capabilities are ISA facts: `mma.sync.m16n8k16` and
//! `simdgroup_half8x8` are fixed by the architecture, so a compile-time table
//! is the right shape for them.
//!
//! Vulkan is not like that. `VK_KHR_cooperative_matrix` **enumerates** the
//! supported `(MSize, NSize, KSize, AType, BType, CType, scope)` combinations
//! per physical device via `VkCooperativeMatrixPropertiesKHR`. Which shapes a
//! device supports is not derivable from the API version or the feature bit.
//!
//! rlx's current check is:
//!
//! ```ignore
//! if !feats.contains(SHADER_F16) || !feats.contains(EXPERIMENTAL_COOPERATIVE_MATRIX) {
//!     return None;
//! }
//! ```
//!
//! — the feature bit is present, therefore 8x8 f16 is assumed to work. That is
//! an assumption a conformant device may violate, and it is why
//! [`rlx_ir::kernel_schedule::Features`] distinguishes a queried capability
//! list from a static one.
//!
//! # What this can and cannot see
//!
//! wgpu does not surface `VkCooperativeMatrixPropertiesKHR` — it exposes a
//! single `EXPERIMENTAL_COOPERATIVE_MATRIX` boolean. So [`target_for_device`]
//! records the 8x8x8 shape the WGSL kernel is written against **and marks the
//! set truncated**, because the device may support other shapes this query
//! cannot enumerate.
//!
//! That is the honest encoding of the situation: a `false` from
//! [`rlx_ir::kernel_schedule::Target::has`] here means "not known to be
//! supported", not "known to be unsupported", and
//! [`rlx_ir::kernel_schedule::Target::capabilities_may_be_incomplete`] says so.
//! Closing that gap needs a raw Vulkan query beneath wgpu, which is a larger
//! change than this module.

use rlx_ir::kernel_schedule::{Feature, FeatureSet, Features, Target};

/// Tile edge the `matmul_coop16` WGSL kernel is written against.
pub const COOP_TILE: u32 = 8;

/// Build a schedule target from a live wgpu device.
///
/// `max_shared_bytes` comes from the device's real
/// `max_compute_workgroup_storage_size` rather than a guess, so a schedule that
/// fits the adapter in hand is accepted and one that does not is rejected with
/// the adapter's own number.
pub fn target_for_device(device: &wgpu::Device) -> Target {
    let limits = device.limits();
    let feats = device.features();

    let mut set = FeatureSet::new();
    // Only claim the shape the shipped kernel actually uses. wgpu cannot
    // enumerate the rest, and inventing entries would be worse than admitting
    // the list is partial.
    if feats.contains(wgpu::Features::SHADER_F16)
        && feats.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        set.push(Feature::CoopMatrix {
            m: COOP_TILE,
            n: COOP_TILE,
            k: COOP_TILE,
        });
    }
    // Mark the list partial regardless: wgpu exposes a boolean where Vulkan has
    // a table, so this set is never known to be complete.
    set.truncated = true;

    Target {
        name: "wgpu-device",
        max_shared_bytes: limits.max_compute_workgroup_storage_size as usize,
        // wgpu speaks in invocations, not warps. 32 is the common subgroup
        // width; this is a floor, not a measurement.
        max_warps: (limits.max_compute_invocations_per_workgroup as usize).div_ceil(32),
        features: Features::Queried(set),
    }
}

/// A target for a device with no cooperative-matrix support, for testing the
/// rejection path without needing such a device present.
pub fn target_without_coop_matrix(max_shared_bytes: usize) -> Target {
    let mut set = FeatureSet::new();
    set.truncated = true;
    Target {
        name: "wgpu-device (no coop matrix)",
        max_shared_bytes,
        max_warps: 8,
        features: Features::Queried(set),
    }
}
