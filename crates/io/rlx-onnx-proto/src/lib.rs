// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vendored pure-Rust ONNX protobuf types — no `protoc`, no build script.
//!
//! [`onnx`] is the checked-in `rust-protobuf` 3.7.2 output for the ONNX
//! schema in `onnx.proto`.
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
//! assert_eq!(model.ir_version(), 0);
//! ```
//!
//! # Regenerating
//!
//! Only needed when the ONNX schema or the `protobuf` version changes:
//! `cargo run --manifest-path crates/io/rlx-onnx-proto/regen/Cargo.toml`
//! (pure-Rust parser, no `protoc`). See the crate README.
//!
//! [`rlx-onnx-import`]: https://docs.rs/rlx-onnx-import

pub mod onnx;
