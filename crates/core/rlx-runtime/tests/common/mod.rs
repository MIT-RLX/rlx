// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared helpers for integration tests that touch GPU backends.

use rlx_runtime::Device;
use std::sync::{Mutex, MutexGuard};

static GPU_TEST_MUTEX: Mutex<()> = Mutex::new(());

/// Serialize GPU backend use within one integration-test binary.
///
/// Metal, wgpu, and CUDA/ROCm adapters are not safe to init/teardown from
/// multiple test threads at once on Apple Silicon (parallel runs may SIGSEGV).
pub struct GpuTestGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl GpuTestGuard {
    pub fn acquire(device: Device) -> Option<Self> {
        if matches!(
            device,
            Device::Metal | Device::Gpu | Device::Cuda | Device::Rocm | Device::Mlx
        ) {
            Some(Self(
                GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner()),
            ))
        } else {
            None
        }
    }
}

impl Drop for GpuTestGuard {
    fn drop(&mut self) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            rlx_metal::device::drain_command_queue();
            rlx_metal::mps_blas::invalidate_caches();
        }
    }
}
