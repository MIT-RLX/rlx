// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Precision selection for graph execution.
//!
//! Each backend can compile a graph at f32 (default — accurate) or f16
//! (half precision — 2× peak FLOPs and ½ memory bandwidth on supported
//! hardware). The IR remains dtype-agnostic; the backend decides how to
//! materialize buffers and pick kernels.
//!
//! Mixed precision: f16 inference typically keeps reductions (LayerNorm
//! mean/var, attention softmax) in f32 to avoid catastrophic accuracy
//! loss while keeping matmul + element-wise in f16.

/// Numeric precision for graph compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Full single precision. Always supported; accurate; baseline.
    #[default]
    F32,
    /// Half precision (IEEE 754 binary16). Native on Apple Silicon GPU
    /// and many CPUs (NEON `vfmaq_f16`). 2× FLOPs / 0.5× memory vs F32.
    /// Reductions are still computed in F32 for numerical stability.
    F16,
    /// Brain-float: 8-bit exponent, 7-bit mantissa. Same range as F32,
    /// less precision. Used in many LLMs. Accelerator-dependent.
    BF16,
}

impl Precision {
    /// Bytes per scalar at this precision.
    pub fn size_bytes(self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F16 | Precision::BF16 => 2,
        }
    }

    /// Backward-compatible alias used in older code.
    pub fn bytes(self) -> usize {
        self.size_bytes()
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::F32 => write!(f, "f32"),
            Precision::F16 => write!(f, "f16"),
            Precision::BF16 => write!(f, "bf16"),
        }
    }
}
