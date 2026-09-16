// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **The MPSGraph crash mitigation must still be in force.**
//!
//! `metal-mpsgraph-init-segv` in the evolution ledger: with a nil
//! `MPSGraphCompilationDescriptor`, MPSGraph defers building `GPURegionRuntime`
//! onto its own `MPSGraphExecutable_queue`, and that deferred path faulted at a
//! measured **~2% per executable initialization** — a null dereference inside
//! MetalPerformanceShadersGraph with no rlx frames on the faulting thread. Full
//! writeup and measurements in `docs/apple-feedback-mpsgraph-crash.md`.
//!
//! The mitigation is to pass a real descriptor with
//! `waitForCompilationCompletion = YES`, keeping the work on the calling
//! thread. Building that descriptor has to be best-effort — an OS without the
//! class or the setter must get the old behaviour rather than a hard failure —
//! and that is exactly the problem this file exists for: **"best-effort
//! fallback" and "silently back on the crashing path" are the same code path.**
//! Nothing about a `respondsToSelector:` returning false is visible at runtime.
//!
//! ## Why this is a *structural* check and not a soak
//!
//! The fault does not currently reproduce on this OS. Re-measured on macOS
//! 26.4.1 (25E253), ~700 executable inits with the mitigation **disabled**
//! produced 0 crashes where the 2% baseline predicts ~14 (p ≈ 6e-7). So a soak
//! test would pass today whether or not the mitigation was applied, and would
//! be a vacuous gate — the exact failure mode the corpus's own
//! `vacuous-check-empty-fold` entry records.
//!
//! What is still checkable, cheaply and deterministically, is whether the
//! mitigation is *in force*. That is what regressed silently if a future macOS
//! drops the selector, and it is the only part a test can honestly assert.
//! `crates/backends/rlx-metal/scripts/mpsgraph-soak.sh` remains the way to
//! re-measure the rate itself on an OS where it reproduces.

#![cfg(target_vendor = "apple")]

use rlx_metal::mps_graph::{SyncCompile, mps_graph_supported, sync_compile_status};

/// Serializes the two tests below.
///
/// `RLX_MPSGRAPH_NO_SYNC_COMPILE` is process-global and
/// `the_env_override_still_selects_the_control_arm` sets it, while
/// `sync_compile_mitigation_is_in_force` reads it — and cargo runs both on
/// threads of one process. Without this, the reader intermittently observes the
/// writer's arm and fails claiming the mitigation is off, which is the opposite
/// of what it found.
static ENV_ARM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take [`ENV_ARM_LOCK`], ignoring poisoning: it orders env access and guards
/// no invariant a panicking test could leave broken.
fn env_arm_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_ARM_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The mitigation applies on this host.
///
/// A failure here is not a test bug: it means new MPSGraph executables are
/// being compiled through the path that faulted at ~2%, and nothing else would
/// have said so.
#[test]
fn sync_compile_mitigation_is_in_force() {
    let _lock = env_arm_lock();
    if !mps_graph_supported() {
        eprintln!("MPSGraph unavailable on this host — nothing to check");
        return;
    }
    match sync_compile_status() {
        SyncCompile::Applied => {}
        SyncCompile::DisabledByEnv => {
            panic!(
                "RLX_MPSGRAPH_NO_SYNC_COMPILE=1 is set in this environment. That is the \
                 deliberate A/B arm for re-measuring the crash rate (scripts/mpsgraph-soak.sh), \
                 not something a test run should inherit — every MPSGraph compile in this \
                 process is on the nil-descriptor path."
            )
        }
        SyncCompile::ClassMissing => panic!(
            "MPSGraphCompilationDescriptor is missing on this OS, so compiles fall back to \
             the nil-descriptor path that faulted at ~2% per executable init. See \
             docs/apple-feedback-mpsgraph-crash.md — the mitigation needs a new mechanism \
             on this OS version, not a deleted assertion."
        ),
        SyncCompile::SelectorMissing => panic!(
            "MPSGraphCompilationDescriptor no longer responds to \
             setWaitForCompilationCompletion:. The descriptor is still passed, but it no \
             longer requests synchronous compilation — the ~2% crash path is live again and \
             the code would not have told you. See docs/apple-feedback-mpsgraph-crash.md."
        ),
    }
}

/// The env A/B arm still reaches the un-mitigated path.
///
/// Without this, a refactor that made `sync_compile_status` return `Applied`
/// unconditionally would keep the test above green while silently deleting the
/// control arm that `mpsgraph-soak.sh` needs to measure anything.
#[test]
fn the_env_override_still_selects_the_control_arm() {
    let _lock = env_arm_lock();
    // SAFETY: `set_var` mutates process-global state, so being single-threaded
    // *within this test* is not the property that matters — the sibling test
    // reads the same variable from another thread. `ENV_ARM_LOCK` is what makes
    // this sound: no other test in this binary reads the value while it is set.
    unsafe { std::env::set_var("RLX_MPSGRAPH_NO_SYNC_COMPILE", "1") };
    let status = sync_compile_status();
    unsafe { std::env::remove_var("RLX_MPSGRAPH_NO_SYNC_COMPILE") };
    assert_eq!(
        status,
        SyncCompile::DisabledByEnv,
        "RLX_MPSGRAPH_NO_SYNC_COMPILE=1 no longer selects the nil-descriptor arm, so \
         scripts/mpsgraph-soak.sh would be measuring the mitigated path under both labels"
    );
}
