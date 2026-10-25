// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The language-neutral core of the rlx bindings.
//!
//! Every language binding — rlx-js, pyrlx, the C ABI and what sits on it
//! (Swift, and any FFI language) — is generated from what lives here:
//!
//! * [`for_each_graph_op!`] — **the** graph-builder table. One row per op.
//! * [`ops`] — a typed function per row, with arguments decoded, node ids
//!   checked and panics caught. Bindings convert their own values to the wire
//!   types of [`kinds`] and call these.
//! * [`manifest()`] — the table as data, for the generators (`rlx-bindgen`)
//!   that print foreign source (C header, Swift, `.pyi`), and for drift checks.
//! * [`api`] — the runtime API table (`api.idl`): every class, method and
//!   function past the graph builders, named in every language.
//! * [`enums`] — the one copy of every string → enum parser.
//! * [`ApiError`] — the error every binding maps onto its own exceptions.
//!
//! Adding an op means adding one row to the table; see `docs/bindings.md`.

pub mod api;
pub mod enums;
mod error;
pub mod kinds;
pub mod manifest;
pub mod ops;
pub mod panics;
#[cfg(feature = "runtime")]
pub mod runtime;
pub mod shapes;
mod table;

pub use error::{ApiError, ApiResult, ErrorKind};
pub use manifest::{ArgSpec, OpSpec};

/// Every op-table row as data.
pub fn manifest() -> &'static [OpSpec] {
    manifest::GRAPH_OPS
}
