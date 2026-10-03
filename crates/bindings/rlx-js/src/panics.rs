// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Turning Rust panics into JavaScript exceptions at the boundary.
//!
//! rlx asserts its invariants: a matmul with mismatched inner dimensions, a
//! reshape that changes the element count, a `wrt` parameter the loss does not
//! depend on — each is a `panic!`, which is right for Rust callers and fatal for
//! an embedded engine. A script could abort its host with a typo:
//!
//! ```text
//! g.matmul(g.input("a", [2, 3], "f32"), g.input("b", [5, 7], "f32"))
//! → thread 'main' panicked at rlx-ir/src/shape.rs
//! ```
//!
//! [`catch`] converts that into an ordinary `Error` the script can catch. It is
//! a class fix rather than a per-op one: every op the `graph_ops!` table
//! generates is wrapped, so an assertion in an op nobody thought to test is
//! still a catchable error.
//!
//! # Caveats
//!
//! * A caught panic leaves the graph **possibly half-modified**. In practice the
//!   shape assertions fire before any mutation, but a script that catches one
//!   should build a fresh graph rather than carry on with that one.
//! * Under the `apple` profile (`panic = "abort"`, used for the iOS static
//!   library) there is no unwinding and this does nothing. That profile aborts
//!   on purpose.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, PanicHookInfo};
use std::sync::OnceLock;

use quickrs_core::context::Context;
use quickrs_core::value::JsResult;

thread_local! {
    /// How many [`catch`] frames are active on this thread. While non-zero the
    /// panic hook stays quiet, because the message is about to be delivered to
    /// the script as an exception instead.
    static GUARDED: Cell<usize> = const { Cell::new(0) };
}

/// Install a hook that suppresses output only inside [`catch`].
///
/// Panics from anywhere else still print normally — swallowing those would hide
/// real bugs in the host.
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
fn message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panicked with a non-string payload".to_string()
    }
}

/// Run `f`, converting a panic into a thrown JavaScript error.
///
/// `what` names the operation, since a bare assertion message from deep in
/// shape inference does not say which call produced it.
pub fn catch<T>(ctx: &mut Context, what: &str, f: impl FnOnce() -> T) -> JsResult<T> {
    GUARDED.with(|g| g.set(g.get() + 1));
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    GUARDED.with(|g| g.set(g.get() - 1));
    match result {
        Ok(value) => Ok(value),
        Err(payload) => ctx.throw_internal(&format!(
            "{what}: {} (this was a Rust assertion inside rlx, surfaced here \
             rather than aborting; the graph may be partially built — start a \
             fresh one)",
            message(payload.as_ref())
        )),
    }
}
