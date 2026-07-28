// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! HIP kernel source strings (shared CUDA/HIP `.cu` + AMD-only MFMA).

pub use rlx_gpu_kernels::*;

pub use rlx_gpu_kernels::rocm::MATMUL_MFMA_CU;

/// HIP entry points compiled from shared sources (+ `matmul_mfma`).
/// Includes the on-device complex-simulation kernels `complex_cast` +
/// `binary_c64` + Wirtinger trio (`complex_norm_sq` /
/// `complex_norm_sq_backward` / `conjugate_c64`) (shared CUDA-C, compiled
/// via hipRTC) and real-INT8 `q_matmul` / `q_conv2d`.
pub const KERNEL_COUNT: usize = 74;
