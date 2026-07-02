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

//! Serialize access to the MLX C++ runtime.
//!
//! MLX's device context, allocator/refcount tables, and `mlx::compile`
//! trace builder are not safe under concurrent use from multiple Rust
//! threads. Integration tests run in parallel by default; without this
//! lock, compiled-mode conv repro tests can exit with SIGTRAP and result
//! arrays freed on one thread (`Array::drop` → `rlx_mlx_array_free`) can
//! SIGSEGV against another thread's in-flight `eval()`.
//!
//! The lock is **reentrant** on purpose: `Array::drop` takes it, and
//! intermediate arrays are dropped constantly *inside* an already-guarded
//! `run_*` call on the same thread — a non-reentrant `Mutex` would
//! self-deadlock there. `ReentrantLock` lets the owning thread re-enter
//! for free while still excluding all other threads. Single-threaded
//! inference (the hot path) only ever sees the uncontended-or-reentrant
//! case, so the overhead is a thread-id check, not cross-core contention.

use std::cell::Cell;
use std::sync::{Mutex, MutexGuard, OnceLock};

static MLX_RUNTIME_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

thread_local! {
    /// Reentrancy depth for *this* thread. >0 means we already hold the
    /// cross-thread `Mutex`, so a nested `runtime_guard()` must not re-lock
    /// (that would self-deadlock); it just bumps the count.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Reentrant guard over [`MLX_RUNTIME_LOCK`]. The outermost acquisition on a
/// thread owns the real `MutexGuard`; nested ones hold `None` and only manage
/// the depth counter. Dropping decrements the depth and, at zero, releases the
/// mutex (the `_outer` field drops after this `Drop` body runs).
pub(crate) struct RuntimeGuard {
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
        let outer = MLX_RUNTIME_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("mlx runtime lock poisoned");
        RuntimeGuard {
            _outer: Some(outer),
        }
    } else {
        RuntimeGuard { _outer: None }
    }
}
