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
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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

pub mod codegen;
pub mod estimate;
pub mod export;
pub mod export_config;
pub mod from_graph;
pub mod ir;
pub mod legalize;
pub mod model;
pub mod pack;
pub mod passes;
pub mod quant;
pub mod reference;
pub mod sideband;
pub mod tune;
pub mod verilog;
pub mod weights;

pub use codegen::{
    Artifact, collect_artifacts, collect_artifacts_opt, emit_model, emit_model_tuned,
    emit_with_config,
};
pub use estimate::{Estimate, estimate};
pub use export::{ExportResult, export_graph, export_model};
pub use export_config::{
    FpgaExportConfig, GraphIoBind, HwTarget, InputIface, IoConfig, OutputIface, OutputKind,
    PortNames, SidebandSpec,
};
pub use from_graph::FromGraphOptions;
pub use legalize::{ExportQuantMode, LegalizeOptions, prepare_model};
pub use model::{Layer, Model, WeightEncoding, tinyconv_mnist_from_cortexm};
pub use tune::{OptTarget, RequantPrecision, Tune};
