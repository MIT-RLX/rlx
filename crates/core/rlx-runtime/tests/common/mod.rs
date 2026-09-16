// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared helpers for integration tests that touch GPU backends.

// Each test binary declaring `mod common;` uses only the helpers it needs, so
// the rest are dead code in that binary — 25 of them use only
// `skip_unless_available`, and `just lint` runs with `-D warnings`.
#![allow(dead_code)]

use rlx_runtime::Device;
use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

static GPU_TEST_MUTEX: Mutex<()> = Mutex::new(());

thread_local! {
    /// How many guards this thread currently holds. See [`GpuTestGuard`].
    static GPU_GUARD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Does this device need serializing? On macOS `Vulkan` is Metal via MoltenVK,
/// so it is the same physical device as the rest of the list.
fn needs_serializing(device: Device) -> bool {
    matches!(
        device,
        Device::Metal | Device::Gpu | Device::Cuda | Device::Rocm | Device::Mlx | Device::Vulkan
    )
}

/// Serialize GPU backend use within one integration-test binary.
///
/// Metal, wgpu, Vulkan and CUDA/ROCm adapters are not safe to init/teardown
/// from multiple test threads at once on Apple Silicon (parallel runs may
/// SIGSEGV, or return garbage — `elementwise_backend_parity` produced
/// `worst_rel=1.0` on Vulkan about one run in six, and `vulkan_parity`'s
/// `fma_decomposed` did the same, both clean at `--test-threads=1`).
///
/// **Re-entrant on purpose.** A plain `Mutex` deadlocks the moment a guarded
/// helper is called from a guarded test, which is exactly the shape these files
/// have — a `run_on` chokepoint plus per-test `Session::new` sites. Making
/// re-acquisition on the same thread a no-op is what lets the guard be added at
/// every site mechanically, without auditing each call graph for nesting.
pub struct GpuTestGuard {
    // `None` when this is a re-entrant acquisition: the outermost guard on this
    // thread owns the lock and releases it.
    #[allow(dead_code)]
    lock: Option<MutexGuard<'static, ()>>,
    counted: bool,
}

impl GpuTestGuard {
    pub fn acquire(device: Device) -> Option<Self> {
        if !needs_serializing(device) {
            return None;
        }
        let depth = GPU_GUARD_DEPTH.with(|d| {
            let n = d.get();
            d.set(n + 1);
            n
        });
        if depth > 0 {
            // Already held further up this thread's stack.
            return Some(Self {
                lock: None,
                counted: true,
            });
        }
        Some(Self {
            lock: Some(GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())),
            counted: true,
        })
    }
}

/// Take the GPU serialization lock for the rest of the enclosing scope,
/// whatever devices the caller goes on to use.
///
/// [`GpuTestGuard::acquire`] needs a `Device`, which is awkward at the top of a
/// test that loops over several. There is one global lock, so "serialize this
/// whole test against every other GPU test in the binary" is the useful
/// granularity — and because the guard is re-entrant, inner `acquire` calls
/// inside helpers cost nothing.
///
/// ```ignore
/// #[test]
/// fn my_gpu_parity() {
///     let _gpu = common::serialize_gpu();
///     for dev in devices() { /* ... */ }
/// }
/// ```
pub fn serialize_gpu() -> Option<GpuTestGuard> {
    GpuTestGuard::acquire(Device::Gpu)
}

impl Drop for GpuTestGuard {
    fn drop(&mut self) {
        if self.counted {
            GPU_GUARD_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        }
        // Only the outermost guard drains — an inner one releasing the device's
        // queues while the outer scope is still using them defeats the point.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if self.lock.is_some() {
            rlx_metal::device::drain_command_queue();
            rlx_metal::mps_blas::invalidate_caches();
        }
    }
}

/// True when `device` cannot be used on this host and the test should return.
///
/// `Session::new` asserts on an unavailable device, so a parity test that does
/// not check first reports a *numerical failure* for something that is not one.
/// The case that matters in practice is not a missing driver but a present GPU
/// with no memory left: on a box already running a training job, wgpu's
/// `request_device` fails with "Not enough memory left", `is_available` goes
/// false, and an unguarded suite turns 37 tests red at once — indistinguishable
/// at a glance from a real regression in the kernels.
///
/// The opposite risk is that a skip is *silent*: a rig where the device should
/// work reports `ok` for tests that never ran. `RLX_REQUIRE_DEVICE=1` turns
/// every skip here into a failure. The GPU recipes in the `Justfile` set it for
/// you — asking for `just test-gpu` is asserting a GPU is present — so opting
/// back out is `just require_device=0 test-gpu`, and it is read with
/// [`rlx_ir::env::flag`] so that `=0` actually means off.
pub fn skip_unless_available(device: Device, label: &str) -> bool {
    // One implementation, in `rlx-ir`, because the backend crates need the same
    // discipline and cannot depend on rlx-runtime. A second copy here is the
    // silent-divergence hazard this tree keeps paying for — `mask_strides_for_shape`
    // and `is_static_weight_tensor` both drifted that way.
    rlx_ir::env::skip_unless_device(
        label,
        rlx_runtime::feature_compiled(device),
        rlx_runtime::is_available(device),
    )
}

/// [`skip_unless_available`] with the label derived from the device.
///
/// Most device-gated tests loop over a `&[Device]` or take one as a parameter,
/// so there is no literal to name — `if !is_available(dev) { return }` was the
/// single most common unmigrated shape in this tree. `{dev:?}` is a perfectly
/// good label there ("Metal", "Cuda", …), and demanding a hand-written string
/// is what kept those call sites on the raw form.
pub fn skip_unless(device: Device) -> bool {
    skip_unless_available(device, &format!("{device:?}"))
}
