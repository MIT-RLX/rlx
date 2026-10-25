// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The runtime surface every binding exposes, as plain Rust.
//!
//! Each binding wraps these types in its own classes — a JS `rlx.Session`, a
//! Python `pyrlx.Session`, a C `RlxSession*` — and converts only its own
//! values. Tensors cross as slices (`&[f32]`, typed bytes); structured options
//! and reports cross as JSON ([`serde_json::Value`]), so the shape of every
//! options object is decided once, here.

pub mod autodiff;
pub mod check;
pub mod compose;
pub mod devices;
pub mod dist;
#[cfg(feature = "fpga")]
pub mod fpga;
pub mod io;
#[cfg(feature = "models")]
pub mod model;
pub mod opts;
pub mod policy;
pub mod session;
#[cfg(feature = "text")]
pub mod text;
#[cfg(feature = "training")]
pub mod train;

pub use serde_json::{self, Value};
pub use session::{CompileRequest, Compiled, FlexibleSession, Session};
