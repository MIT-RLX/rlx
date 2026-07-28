// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-infer active token count — forwarded to CPU ONNX control-flow kernels.

/// Hint the active padded token count for this infer.
pub fn set_active_token_count(count: Option<usize>) {
    rlx_cpu::onnx_control_flow::set_active_token_count(count);
}

/// Active token count when set by [`set_active_token_count`] or [`crate::CompiledGraph::set_active_extent`].
pub fn active_token_count() -> Option<usize> {
    rlx_cpu::onnx_control_flow::active_token_count()
}
