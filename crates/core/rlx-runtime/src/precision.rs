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

/// Does `Session::new_with_precision(device, precision)` change how `device`
/// computes? Only these combinations do; every other one compiles through the
/// backend's f32 path, with the same results as `Precision::F32`:
///
/// - `F32` everywhere;
/// - `F16` on [`Device::Ane`] (CoreML lowers floats to f16 when no
///   [`PrecisionPolicy`](crate::PrecisionPolicy) is set);
/// - `BF16` on [`Device::Xdna`] (MatMuls run through the bf16 microkernel).
///
/// For the numerics of a reduced format on any backend (rounded, but still
/// computed in f32), see `RLX_NUMBER_FORMAT` / `CompileOptions::number_format`.
pub fn precision_honored(device: crate::Device, precision: Precision) -> bool {
    match precision {
        Precision::F32 => true,
        Precision::F16 => device == crate::Device::Ane,
        Precision::BF16 => device == crate::Device::Xdna,
    }
}

/// One stderr line per process for each `(device, precision)` request that
/// [`precision_honored`] says runs in f32 anyway. Silent before this existed,
/// which made a timing comparison of F16 against F32 measure noise.
pub(crate) fn warn_if_ignored(device: crate::Device, precision: Precision) {
    use std::sync::Mutex;
    static WARNED: Mutex<Vec<(crate::Device, Precision)>> = Mutex::new(Vec::new());
    if precision_honored(device, precision) {
        return;
    }
    let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    if warned.contains(&(device, precision)) {
        return;
    }
    warned.push((device, precision));
    eprintln!(
        "[rlx] {precision} requested on {device}, which compiles in f32: results and \
         speed match Precision::F32. Set RLX_NUMBER_FORMAT={precision} to emulate \
         {precision} rounding (still computed in f32)."
    );
}

#[cfg(test)]
mod honored_tests {
    use super::*;
    use crate::Device;

    #[test]
    fn f32_is_always_honored() {
        for d in [
            Device::Cpu,
            Device::Metal,
            Device::Cuda,
            Device::Ane,
            Device::Xdna,
        ] {
            assert!(precision_honored(d, Precision::F32));
        }
    }

    #[test]
    fn half_formats_only_where_a_backend_reads_them() {
        assert!(precision_honored(Device::Ane, Precision::F16));
        assert!(precision_honored(Device::Xdna, Precision::BF16));
        for d in [
            Device::Cpu,
            Device::Metal,
            Device::Cuda,
            Device::Rocm,
            Device::Mlx,
        ] {
            assert!(!precision_honored(d, Precision::F16), "{d}");
            assert!(!precision_honored(d, Precision::BF16), "{d}");
        }
        assert!(!precision_honored(Device::Ane, Precision::BF16));
        assert!(!precision_honored(Device::Xdna, Precision::F16));
    }
}
