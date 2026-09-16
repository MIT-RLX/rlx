// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vendored pure-Rust ONNX protobuf types — no `protoc`, no build script.
//!
//! [`onnx`] is the checked-in `rust-protobuf` output for the ONNX schema.
//! The crates.io `onnx` crate code-generates the same module at build time
//! by shelling out to an external `protoc` binary; carrying the generated
//! source instead keeps `cargo build` hermetic and toolchain-free, which
//! matters for the [`rlx-onnx-import`] path on machines that have no
//! protobuf compiler installed.
//!
//! The module layout mirrors the old `onnx` crate — `onnx::onnx::ModelProto`
//! and friends — so consumers only change the crate name:
//!
//! ```
//! use rlx_onnx_proto::onnx::ModelProto;
//!
//! let model = ModelProto::new();
//! assert_eq!(model.get_ir_version(), 0);
//! ```
//!
//! # Regenerating
//!
//! Only needed when the ONNX schema itself changes. Run `rust-protobuf` 1.x
//! over `onnx.proto`, then rewrite the bare `::std::any::Any` trait-object
//! references to `dyn ::std::any::Any` so the output builds on a modern
//! edition.
//!
//! [`rlx-onnx-import`]: https://docs.rs/rlx-onnx-import

pub mod onnx;
