// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared GPU serialization for this crate's integration tests.

// Not every binary declaring `mod common;` uses every item, and `just lint`
// runs with `-D warnings`.
#![allow(dead_code)]

use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

static GPU_TEST_MUTEX: Mutex<()> = Mutex::new(());

thread_local! {
    /// How many guards this thread holds. See [`GpuTestGuard`].
    static GPU_GUARD_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Serialize Vulkan use within one integration-test binary.
///
/// The device itself is a process-wide `OnceLock`, so this is not about
/// concurrent *creation* — it is concurrent *use* of the one shared device.
/// `VulkanDevice` locks submission (`submit_lock`) but not the allocation,
/// descriptor-set and pipeline paths around it, and Rust runs the tests in a
/// binary on parallel threads by default. Measured on Apple Silicon (MoltenVK):
/// `cargo test -p rlx-vulkan --release` crashed **4 of 10 runs** with SIGTRAP /
/// SIGABRT during setup — before any test printed — and **0 of 10** with
/// `--test-threads=1`. Because it dies in setup the target varies run to run,
/// which is why it reads as an unrelated failure each time and never
/// reproduces when a single target is run on its own.
///
/// This mirrors `rlx-runtime/tests/common/mod.rs`, whose own notes record the
/// same class for `vulkan_parity::fma_decomposed`. That guard lives in a
/// test-only module of another crate, so it cannot be imported here.
///
/// **Re-entrant on purpose**, for the same reason as the runtime copy: a plain
/// `Mutex` deadlocks as soon as a guarded helper is called from a guarded
/// test, so re-acquisition on the same thread is a no-op and the guard can be
/// added at every site mechanically.
pub struct GpuTestGuard {
    // `None` for a re-entrant acquisition: the outermost guard on this thread
    // owns the lock and releases it.
    #[allow(dead_code)]
    lock: Option<MutexGuard<'static, ()>>,
}

impl GpuTestGuard {
    pub fn acquire() -> Self {
        let depth = GPU_GUARD_DEPTH.with(|d| {
            let n = d.get();
            d.set(n + 1);
            n
        });
        if depth > 0 {
            return Self { lock: None };
        }
        Self {
            // A test that panicked while holding the lock poisons it; the next
            // test should still serialize rather than inherit the poison and
            // fail for an unrelated reason.
            lock: Some(GPU_TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())),
        }
    }
}

impl Drop for GpuTestGuard {
    fn drop(&mut self) {
        GPU_GUARD_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Take the Vulkan serialization lock for the rest of the enclosing scope.
///
/// ```ignore
/// #[test]
/// fn my_vulkan_test() {
///     let _gpu = common::serialize_gpu();
///     // …
/// }
/// ```
pub fn serialize_gpu() -> GpuTestGuard {
    GpuTestGuard::acquire()
}
