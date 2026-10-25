// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Turning Rust panics into [`ApiError`]s at the binding boundary.
//!
//! rlx asserts its invariants with `panic!` — a matmul with mismatched inner
//! dimensions, a reshape that changes the element count. That is right for
//! Rust callers and fatal for a host language: a typo in a script would abort
//! the process. [`catch`] turns one into an ordinary error of kind
//! [`ErrorKind::Panic`]. Every generated op runs under it, so an assertion in
//! an op nobody thought to test is still a catchable error in every language.
//!
//! # Caveats
//!
//! * A caught panic leaves the graph **possibly half-modified**. The shape
//!   assertions fire before any mutation in practice, but a caller that catches
//!   one should build a fresh graph.
//! * Under `panic = "abort"` (the `apple` profile used for iOS archives) there
//!   is no unwinding and this does nothing, on purpose.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, PanicHookInfo};
use std::sync::OnceLock;

use crate::{ApiError, ApiResult, ErrorKind};

thread_local! {
    /// How many [`catch`] frames are active on this thread. While non-zero the
    /// panic hook stays quiet: the message is about to reach the caller as an
    /// error instead.
    static GUARDED: Cell<usize> = const { Cell::new(0) };
}

/// Install a hook that suppresses panic output only inside [`catch`].
///
/// Panics from anywhere else still print — swallowing those would hide real
/// bugs in the host. Idempotent.
pub fn install_hook() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info: &PanicHookInfo<'_>| {
            if GUARDED.with(|g| g.get()) == 0 {
                previous(info);
            }
        }));
    });
}

/// The message out of a panic payload, for the two types `panic!` produces.
pub fn message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panicked with a non-string payload".to_string()
    }
}

/// Run `f`, converting a panic into an [`ErrorKind::Panic`] error.
///
/// `what` names the operation, since a bare assertion message from deep in
/// shape inference does not say which call produced it.
pub fn catch<T>(what: &str, f: impl FnOnce() -> T) -> ApiResult<T> {
    GUARDED.with(|g| g.set(g.get() + 1));
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    GUARDED.with(|g| g.set(g.get() - 1));
    result.map_err(|payload| {
        ApiError::new(
            ErrorKind::Panic,
            format!(
                "{what}: {} (this was a Rust assertion inside rlx, surfaced here \
                 rather than aborting; the graph may be partially built — start a \
                 fresh one)",
                message(payload.as_ref())
            ),
        )
    })
}
