// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! UI (compile-fail) tests locking in the `rlx!` DSL's error diagnostics — each
//! `tests/ui/*.rs` case must fail with the message in its paired `.stderr`.
//! These pin the *spans* of our own `compile_error!`s (not downstream rustc
//! type errors), complementing the message-content checks in the proc macro's
//! own unit tests. Run: `cargo test -p rlx-tensor --features dsl --test ui`
//! (regenerate snapshots with `TRYBUILD=overwrite`).
#![cfg(feature = "dsl")]

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
