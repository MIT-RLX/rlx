// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX FPGA backend — per-graph datapath synthesis.
//!
//! Pipeline:
//!
//! ```text
//!   rlx-ir Graph
//!     → Model::from_graph   (INT8 QConv2d / QMatMul / Pool / Relu / TopK)
//!     → optimize(model, tune)
//!     → codegen            (target-agnostic SystemVerilog + .mem)
//!     → optional synth.sh  (generic Yosys, or ecp5 / ice40 / xilinx7)
//! ```
//!
//! First-class entry points:
//! * [`export::export_graph`] — Graph → RTL
//! * [`codegen::emit_with_config`] — Model → RTL with [`export_config::FpgaExportConfig`]
//! * [`model::Model::from_graph`] — IR recognition
//!
//! Does **not** implement the runtime `Backend` trait (synth takes minutes).

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// The `std` prelude names a `no_std` build loses. Every module glob-imports
/// it, so the crate reads the same with and without the `std` feature.
mod prelude {
    pub use alloc::format;
    pub use alloc::string::{String, ToString};
    pub use alloc::vec;
    pub use alloc::vec::Vec;
}

pub mod codegen;
pub mod estimate;
// Graph import/export needs `std` (files, hash maps, rlx-ir's full surface).
#[cfg(feature = "std")]
pub mod export;
pub mod export_config;
#[cfg(feature = "std")]
pub mod from_graph;
pub mod ir;
#[cfg(feature = "std")]
pub mod legalize;
pub mod model;
pub mod pack;
pub mod passes;
pub mod quant;
pub mod reference;
// The sequential target emits files and formats text, so it needs `std`.
#[cfg(feature = "std")]
pub mod seq;
#[cfg(feature = "std")]
pub mod sideband;
pub mod tune;
pub mod verilog;
pub mod weights;
pub mod xc7;

pub use codegen::{Artifact, collect_artifacts, collect_artifacts_opt};
#[cfg(feature = "std")]
pub use codegen::{emit_model, emit_model_tuned, emit_with_config};
pub use estimate::{Estimate, estimate};
#[cfg(feature = "std")]
pub use export::{ExportResult, export_graph, export_model};
pub use export_config::{
    FpgaExportConfig, GraphIoBind, HwTarget, InputIface, IoConfig, OutputIface, OutputKind,
    PortNames, SidebandSpec,
};
#[cfg(feature = "std")]
pub use from_graph::FromGraphOptions;
#[cfg(feature = "std")]
pub use legalize::{ExportQuantMode, LegalizeOptions, prepare_model};
pub use model::{Layer, Model, WeightEncoding, tinyconv_mnist_from_cortexm};
pub use tune::{OptTarget, RequantPrecision, Tune};
