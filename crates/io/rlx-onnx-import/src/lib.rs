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

//! Load ONNX-exported RLX bundles (or raw `.onnx`) and lower them to [`rlx_ir::hir::HirModule`].
//!
//! Bundles are produced by downstream export tooling (quant fusion, shape inference,
//! safetensors weights). Raw ONNX uses [`prepare_onnx_file`] plus graph rewrites in
//! [`rewrite`] before [`lower`].

pub mod bundle;
pub mod control_flow;
pub mod coverage;
pub mod emit_codegen;
pub mod emit_runtime;
pub mod layout;
pub mod lower;
pub mod onnx_file;
pub mod ops;
pub mod random;
pub mod rewrite;
pub mod shape_propagate;
pub mod strict;
pub mod tensor_data;

#[cfg(feature = "runtime")]
pub mod onnx_scatter;

pub use bundle::{BundleManifest, BundleNode, IoMeta, RlxBundle, load_bundle, onnx_bundle_dir};
pub use control_flow::{
    ALIGNMENT_FRAME_COUNT, CONCAT_FROM_SEQUENCE_OUTPUT, DURATION_CARRY, DurationAlignInputs,
    SAMPLES_PER_ALIGNMENT_FRAME, alignment_buffer_upper_bound, alignment_frame_count,
    alignment_frame_upper_bound, concat_alignment_durations, resolve_duration_align_inputs,
    tensor_traces_alignment_length, tensor_traces_concat_output,
};
pub use coverage::{LOWERED_OPS, REWRITTEN_OPS, op_is_supported, registry_op_count};
pub use lower::{
    DurationLoopLowering, ImportOptions, ImportReport, build_hir_from_bundle, build_hir_from_parts,
};
pub use onnx_file::{
    build_hir_from_onnx_file, install_if_branches, install_scalar_consts, prepare_onnx_file,
    take_if_branches, take_scalar_consts,
};
pub use ops::{OpCategory, format_bundle_category_report, format_registry_dashboard};
pub use tensor_data::TypedParams;
