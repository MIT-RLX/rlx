// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compatibility shim: ScatterElements lives in `rlx_cpu::onnx_indexing`.

pub use rlx_cpu::onnx_indexing::SCATTER_ELEMENTS;

/// Register the shared CPU `onnx.ScatterElements` kernel (and sibling indexing ops).
pub fn register_onnx_scatter_elements_kernel() {
    rlx_cpu::onnx_indexing::register_onnx_indexing_kernels();
}
