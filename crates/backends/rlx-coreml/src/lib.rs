// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

//! # rlx-coreml
//!
//! Apple **CoreML / Neural Engine (ANE)** backend for RLX.
//!
//! The backend lowers an RLX IR [`Graph`](rlx_ir::Graph) to a CoreML
//! **ML Program** (the MIL dialect), serialises it into a `.mlpackage`
//! bundle, and runs it through `CoreML.framework`. CoreML's own planner
//! then schedules each op across the CPU, GPU and Neural Engine.
//!
//! ## Layers
//!
//! | module        | role                                                   |
//! |---------------|--------------------------------------------------------|
//! | [`proto`]     | prost-generated CoreML protobuf types                  |
//! | [`mil`]       | IR → MIL `Program` lowering (host-portable, no FFI)    |
//! | [`mlpackage`] | `.mlpackage` bundle writer                             |
//! | `chip`        | ANE / chip introspection ([`ane_available`], [`chip_info`]) |
//! | `ffi`/`backend` | CoreML.framework execution (Apple platforms only)    |
//!
//! The lowering and `.mlpackage` writing are pure Rust and build on every
//! host, so MIL emission can be unit-tested anywhere. Only execution is
//! gated behind `all(target_vendor = "apple", not(target_os = "watchos"))`.

pub mod host_exec;
pub mod hybrid;
pub mod mil;
pub use mil::{LowerOptions, lower_graph, lower_graph_with_options};
pub mod mlpackage;
pub mod op_registry;
pub mod supported_ops;
pub use op_registry::{CoremlKernel, lookup_coreml_kernel, register_coreml_kernel};
#[cfg(feature = "training")]
pub use supported_ops::SUPPORTED_OPS_TRAINING;
pub use supported_ops::{BACKWARD_OPS, NATIVE_BACKWARD_OPS, SUPPORTED_OPS};

pub mod collective;

/// prost-generated CoreML protobuf types (`package coreml`).
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/coreml.rs"));
}

mod chip;
pub use chip::{ChipInfo, ane_available, chip_info, is_available};

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
mod ffi;

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
pub mod backend;

#[cfg(all(target_vendor = "apple", not(target_os = "watchos")))]
pub use backend::{
    CoremlExecutable, default_compute_units, default_lower_options, promote_c64_to_interleaved_f32,
};

/// Which CoreML compute units the model may use. Mirrors `MLComputeUnits`.
///
/// Runtime selection for a compiled graph goes through
/// [`default_compute_units`] (env + dtype), **not** this enum's `Default`
/// impl — that only provides a serde/placeholder default of [`All`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeUnits {
    /// CPU + GPU + Neural Engine, planner's choice (`MLComputeUnitsAll`).
    #[default]
    All,
    /// CPU only.
    CpuOnly,
    /// CPU + GPU (default for **fp32** graphs — BNNS-safe on large ML Programs).
    CpuAndGpu,
    /// CPU + Neural Engine (default for **f16** graphs / `RLX_COREML_UNITS=ane`).
    CpuAndNeuralEngine,
}

impl ComputeUnits {
    /// The integer code understood by the Objective-C shim.
    pub(crate) fn code(self) -> i32 {
        match self {
            ComputeUnits::All => 0,
            ComputeUnits::CpuOnly => 1,
            ComputeUnits::CpuAndGpu => 2,
            ComputeUnits::CpuAndNeuralEngine => 3,
        }
    }
}

/// Errors raised while lowering, packaging, or running a CoreML model.
#[derive(Debug)]
pub enum CoremlError {
    /// An IR op has no MIL lowering yet.
    Unsupported(String),
    /// A tensor shape was dynamic where a static extent is required.
    DynamicShape(String),
    /// Filesystem error writing the `.mlpackage`.
    Io(std::io::Error),
    /// CoreML.framework rejected the model or prediction.
    Runtime(String),
    /// The serialized model exceeds a hard CoreML/protobuf size limit
    /// (e.g. the ~2 GiB protobuf message cap). Carries `(actual, limit)`
    /// byte counts and a human-readable hint.
    TooLarge {
        what: String,
        bytes: usize,
        limit: usize,
    },
}

impl std::fmt::Display for CoremlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoremlError::Unsupported(s) => write!(f, "unsupported for CoreML: {s}"),
            CoremlError::DynamicShape(s) => write!(f, "dynamic shape unsupported: {s}"),
            CoremlError::Io(e) => write!(f, "io error: {e}"),
            CoremlError::Runtime(s) => write!(f, "CoreML runtime error: {s}"),
            CoremlError::TooLarge { what, bytes, limit } => write!(
                f,
                "{what} is {bytes} bytes, exceeding the {limit}-byte CoreML limit \
                 (~{:.2} GiB). Weights ≥10 elements already live in weight.bin; \
                 reduce inline constants, fold ops, or split the graph.",
                *limit as f64 / (1u64 << 30) as f64
            ),
        }
    }
}

impl std::error::Error for CoremlError {}

impl From<std::io::Error> for CoremlError {
    fn from(e: std::io::Error) -> Self {
        CoremlError::Io(e)
    }
}

/// Convenience result alias for this crate.
pub type Result<T> = std::result::Result<T, CoremlError>;
