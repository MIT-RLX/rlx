// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-infer active token count — forwarded to CPU ONNX control-flow kernels.

/// Hint the active padded token count for this infer. No-op unless the `cpu`
/// backend (which owns the ONNX control-flow kernels) is compiled in.
pub fn set_active_token_count(count: Option<usize>) {
    #[cfg(feature = "cpu")]
    rlx_cpu::onnx_control_flow::set_active_token_count(count);
    #[cfg(not(feature = "cpu"))]
    let _ = count;
}

/// Active token count when set by [`set_active_token_count`] or [`crate::CompiledGraph::set_active_extent`].
pub fn active_token_count() -> Option<usize> {
    #[cfg(feature = "cpu")]
    {
        rlx_cpu::onnx_control_flow::active_token_count()
    }
    #[cfg(not(feature = "cpu"))]
    {
        None
    }
}
