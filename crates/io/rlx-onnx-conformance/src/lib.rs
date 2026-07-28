// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! ONNX op-level conformance harness (ORT reference vs RLX import).

pub mod backend_runner;
pub mod harness;
pub mod onnx_op_registry;
pub mod synthetic;

pub use harness::{ConformanceResult, OrtSession, compare_tensors};
