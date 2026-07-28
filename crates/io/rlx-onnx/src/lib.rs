// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Run ONNX (`.onnx`) models through **native RLX** (`Session::compile`) on the
//! chosen [`Device`] and compile level.
//!
//! ONNX is imported to HIR (see `rlx-onnx-import`), compiled, and executed on CPU /
//! Metal / CUDA / etc. Enable optional `ort-fallback` for ONNX Runtime parity.
//!
//! ## Example
//!
//! ```no_run
//! use rlx_onnx::{OnnxCompileLevel, OnnxModel};
//! use rlx_runtime::Device;
//!
//! let mut model = OnnxModel::load_native("model.onnx", Device::Cpu, OnnxCompileLevel::Level3, 128)?;
//! model.print_io();
//! let inputs = model.zero_inputs_sized(32)?;
//! let outputs = model.run(&inputs)?;
//! assert!(!outputs.is_empty());
//! # Ok::<(), anyhow::Error>(())
//! ```

#[cfg(feature = "ort")]
pub mod backend;
pub mod io;
pub mod level;
pub mod native;
pub mod session;

#[cfg(feature = "ort")]
pub mod session_ort;

#[cfg(feature = "ort")]
pub use backend::{OrtSession, build_onnx_session, execution_providers_for, validate_device};
pub use io::{IoDesc, OnnxElementType, OnnxTensor};
pub use level::OnnxCompileLevel;
pub use session::{OnnxExecBackend, OnnxModel};
