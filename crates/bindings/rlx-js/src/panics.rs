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
//! a class fix rather than a per-op one: every op the `rlx-bind` table
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

use quickrs_core::context::Context;
use quickrs_core::value::JsResult;

/// Install the shared hook that keeps a caught panic quiet (see
/// [`rlx_bind::panics::install_hook`]).
pub fn install_hook() {
    rlx_bind::panics::install_hook();
}

/// Run `f`, converting a panic into a thrown JavaScript `InternalError`.
///
/// `what` names the operation, since a bare assertion message from deep in
/// shape inference does not say which call produced it.
pub fn catch<T>(ctx: &mut Context, what: &str, f: impl FnOnce() -> T) -> JsResult<T> {
    let r = rlx_bind::panics::catch(what, f);
    crate::api::bind::lift(ctx, r)
}
