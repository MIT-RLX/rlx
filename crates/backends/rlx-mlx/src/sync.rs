// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serialize access to the MLX C++ runtime.
//!
//! MLX's device context, allocator/refcount tables, and `mlx::compile`
//! trace builder are not safe under concurrent use from multiple Rust
//! threads **or processes**. Integration tests run in parallel by default
//! (`cargo test -jN` launches multiple test binaries at once); without
//! this lock, compiled-mode conv repro / autodiff conv parity tests can
//! exit with SIGTRAP and result arrays freed on one thread
//! (`Array::drop` → `rlx_mlx_array_free`) can SIGSEGV against another
//! thread's in-flight `eval()`.
//!
//! The lock is **reentrant** on purpose: `Array::drop` takes it, and
//! intermediate arrays are dropped constantly *inside* an already-guarded
//! `run_*` call on the same thread — a non-reentrant `Mutex` would
//! self-deadlock there. Nested acquisitions on the owning thread only
//! bump a depth counter. Single-threaded inference (the hot path) only
//! ever sees the uncontended-or-reentrant case.
//!
//! On Unix the outer acquisition also takes an advisory `flock` so
//! concurrent `cargo test` binaries (separate processes sharing one GPU)
//! serialize the same way.

use std::cell::Cell;
use std::fs::File;
use std::sync::{Mutex, MutexGuard, OnceLock};

static MLX_RUNTIME_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

thread_local! {
    /// Reentrancy depth for *this* thread. >0 means we already hold the
    /// cross-thread `Mutex`, so a nested `runtime_guard()` must not re-lock
    /// (that would self-deadlock); it just bumps the count.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Reentrant guard over [`MLX_RUNTIME_LOCK`] (+ process flock on Unix).
/// The outermost acquisition on a thread owns the real `MutexGuard` and
/// optional process lock; nested ones hold `None` and only manage the
/// depth counter. Dropping decrements the depth and, at zero, releases
/// the mutex / flock (the `_outer` / `_process` fields drop after this
/// `Drop` body runs).
pub(crate) struct RuntimeGuard {
    // Drop order is reverse declaration: release mutex before flock so
    // acquisition (flock → mutex) and release (mutex → flock) nest cleanly.
    _process: Option<ProcessLock>,
    _outer: Option<MutexGuard<'static, ()>>,
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get() - 1));
    }
}

/// Hold for the duration of any MLX FFI that builds, executes, or frees
/// graphs/arrays. Reentrant: safe to take while already held on this thread
/// (e.g. `Array::drop` firing inside a guarded `run_*`).
pub(crate) fn runtime_guard() -> RuntimeGuard {
    let depth = DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v
    });
    if depth == 0 {
        // Process lock first so two cargo-test binaries can't both pass
        // the in-process mutex and race on Metal.
        let process = ProcessLock::acquire();
        let outer = MLX_RUNTIME_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("mlx runtime lock poisoned");
        RuntimeGuard {
            _process: process,
            _outer: Some(outer),
        }
    } else {
        RuntimeGuard {
            _process: None,
            _outer: None,
        }
    }
}

/// Advisory flock held for the lifetime of an outer [`RuntimeGuard`].
/// Closing the fd releases the lock.
struct ProcessLock(File);

impl ProcessLock {
    fn acquire() -> Option<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let path = std::env::temp_dir().join("rlx-mlx-runtime.lock");
            let file = match File::create(&path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("rlx-mlx: warning: could not create process lock {path:?}: {e}");
                    return None;
                }
            };
            let fd = file.as_raw_fd();
            loop {
                let rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
                if rc == 0 {
                    break;
                }
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                eprintln!("rlx-mlx: warning: flock failed on {path:?}: {err}");
                return None;
            }
            Some(ProcessLock(file))
        }
        #[cfg(not(unix))]
        {
            None
        }
    }
}

impl Drop for ProcessLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self.0.as_raw_fd();
            unsafe {
                libc::flock(fd, libc::LOCK_UN);
            }
        }
    }
}
