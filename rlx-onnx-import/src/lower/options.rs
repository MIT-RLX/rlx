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

use std::collections::HashMap;

use rlx_ir::Shape;

/// Optional per-node output shape correction during lowering (model bundles set this).
pub type OutputShapeFix = fn(&str, &Shape) -> Option<Shape>;

/// Hook after bundle `propagate_shapes` (model crates fix ORT trace dims before quant rewrites).
pub type PostShapePropagate = fn(&mut [crate::bundle::BundleNode], &ImportOptions);

/// Hook before bundle `propagate_shapes` (seed seq axes before sym env binds ORT trace dims).
pub type PreShapePropagate = fn(&mut [crate::bundle::BundleNode], &ImportOptions);

/// Per-node import statistics.
#[derive(Debug, Default)]
pub struct ImportReport {
    pub lowered: usize,
    pub skipped: usize,
    pub stubbed: usize,
    pub stubbed_nodes: Vec<String>,
    pub unsupported: HashMap<String, usize>,
}

/// Options controlling ONNX → HIR lowering.
#[derive(Debug, Clone)]
pub struct ImportOptions {
    pub sequence_length: usize,
    pub max_waveform_samples: usize,
    /// When true, keep int8 weights and lower to `DequantMatMul` / `QConv2d` instead of
    /// expanding everything to F32 at import time.
    pub use_quantized_kernels: bool,
    /// Bind ONNX `sequence_length` / seq-like `unk__*` axes to dynamic seq for runtime specialize.
    pub dynamic_sequence: bool,
    /// Fail import when any node is stubbed or unsupported (default: true).
    pub strict: bool,
    /// Apply quant-bundle rewrites (`QMatMul`, DQL alias, epilogue bypass) before lowering.
    pub quantize_bundle_rewrites: bool,
    /// Upper bound multiplier for `Loop` / `ConcatFromSequence` static shapes (`seq * N`).
    pub max_frames_per_token: usize,
    /// When true, lower ONNX `Random*` to `Op::Custom` instead of native
    /// [`Op::RngNormal`] / [`Op::RngUniform`]. Kept for callers that still
    /// route random ops through a backend custom-kernel registry.
    pub lower_random_as_custom: bool,
    /// Correct inferred output shapes for known-bad ONNX metadata (set by downstream model crates).
    pub output_shape_fix: Option<OutputShapeFix>,
    /// Model-specific bundle node patch after shape propagation (before quant rewrites).
    pub post_shape_propagate: Option<PostShapePropagate>,
    /// Model-specific bundle node patch before shape propagation.
    pub pre_shape_propagate: Option<PreShapePropagate>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            sequence_length: 128,
            max_waveform_samples: 48_000,
            use_quantized_kernels: true,
            dynamic_sequence: false,
            strict: true,
            quantize_bundle_rewrites: false,
            max_frames_per_token: 24,
            lower_random_as_custom: false,
            output_shape_fix: None,
            post_shape_propagate: None,
            pre_shape_propagate: None,
        }
    }
}

impl ImportOptions {
    /// Quantized bundle import defaults (quant rewrites, relaxed strict until control-flow parity).
    pub fn quant_bundle() -> Self {
        Self {
            quantize_bundle_rewrites: true,
            strict: false,
            ..Self::default()
        }
    }
}
