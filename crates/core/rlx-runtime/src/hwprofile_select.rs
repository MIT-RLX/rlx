// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! VRAM-aware device selection, backed by [`rlx_hwprofile`] (feature
//! `hwprofile`).
//!
//! [`crate::fastest_device`] picks by static backend priority alone — it can't
//! tell whether the chosen GPU actually has the memory a model needs. This
//! module folds host GPU/VRAM detection into the choice: a GPU backend whose
//! detected VRAM is below a floor is skipped, walking down the priority list
//! to CPU. The raw [`HostProfile`] is also re-exported so a caller can build a
//! `rlx-protocol` topology `NodeSpec` (which needs `vram_bytes`) straight from
//! a detected [`GpuInfo`].

use crate::device_ext::{available_devices, fastest_among};
use rlx_driver::Device;
use rlx_hwprofile::{BackendKind, HostProfile};

pub use rlx_hwprofile::{GpuInfo, HostProfile as Profile, detect};

/// Map an rlx [`Device`] to the hardware backend the profiler reports it under.
/// Apple's Metal / MLX / ANE all execute on the same unified-memory GPU, so
/// they share the `Metal` profile entry. Non-GPU or unprofiled backends → `None`.
fn device_backend(device: Device) -> Option<BackendKind> {
    match device {
        Device::Cuda => Some(BackendKind::Cuda),
        Device::Rocm => Some(BackendKind::Rocm),
        Device::Metal | Device::Mlx | Device::Ane => Some(BackendKind::Metal),
        Device::Vulkan => Some(BackendKind::Vulkan),
        Device::Cpu => Some(BackendKind::Cpu),
        _ => None,
    }
}

fn device_vram_bytes_in(profile: &HostProfile, device: Device) -> Option<u64> {
    let backend = device_backend(device)?;
    profile
        .gpus
        .iter()
        .filter(|g| g.backend == backend && g.vram_bytes > 0)
        .map(|g| g.vram_bytes)
        .max()
}

/// Detected VRAM (bytes) available to `device` on this host, if the profiler
/// can see it. Apple unified-memory devices report total system RAM (shared
/// with the CPU). Returns `None` when the backend isn't a profiled GPU or its
/// memory couldn't be read.
pub fn device_vram_bytes(device: Device) -> Option<u64> {
    device_vram_bytes_in(&detect(), device)
}

/// Like [`crate::fastest_device`], but skips GPU backends whose detected VRAM
/// is below `min_vram_bytes`, walking down the priority list and ending at
/// CPU. A backend whose VRAM can't be probed (unknown, or the CPU) is *kept* —
/// better to try it than to wrongly demote it on a host the profiler can't
/// fully see.
pub fn fastest_device_with_vram(min_vram_bytes: u64) -> Device {
    let profile = detect();
    let fits = |d: Device| match device_vram_bytes_in(&profile, d) {
        Some(v) => v >= min_vram_bytes,
        None => true,
    };
    let candidates: Vec<Device> = available_devices()
        .into_iter()
        .filter(|&d| fits(d))
        .collect();
    if candidates.is_empty() {
        return Device::Cpu;
    }
    fastest_among(&candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_and_select_never_panics() {
        // On any host (incl. CI with no GPU) detection returns a profile and
        // the VRAM-floored pick yields a real device (CPU in the worst case).
        let profile = detect();
        assert!(!profile.available_backends.is_empty());
        let d = fastest_device_with_vram(1 << 30); // 1 GiB floor
        // CPU has no VRAM entry, so it's never demoted — a valid fallback.
        let _ = device_backend(d);
    }
}
