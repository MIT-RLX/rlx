// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
// Pure-Rust ONNX protobuf types — checked-in rust-protobuf output, no protoc.
// Mirrors the layout of the old `onnx` crate (`onnx::onnx::ModelProto`, …) so
// consumers only change the crate name.
extern crate protobuf;

pub mod onnx;
