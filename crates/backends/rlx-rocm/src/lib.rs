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

//! RLX ROCm/HIP backend — AMD GPUs via the HIP runtime.
#![allow(clippy::too_many_arguments)]
//
// FFI shim helpers (cudnn_conv*, hipblaslt_matmul_fused, etc.)
// inherently take many arguments — they mirror the underlying C API
// surface. Suppressing the lint at crate scope avoids drowning out
// signal warnings.
//!
//! Sister crate to `rlx-cuda`. Kernel sources live in `rlx-gpu-kernels`
//! (CUDA/HIP-compatible `.cu` files). The dispatch ladder mirrors
//! `rlx-cuda`'s: hipBLASLt → hipBLAS → MIOpen-conv → custom kernels
//! via hipRTC, falling through on any setup error.

pub mod arena;
pub mod backend;
pub mod calibrate;
pub mod device;
pub mod fft_dispatch;
pub mod fft_host;
pub mod gdn_host;
pub mod gguf_gpu;
pub mod gguf_host;
pub mod hip;
pub mod hipblas;
pub mod hipblaslt;
pub mod host_misc;
pub mod host_staging;
pub mod im2col_host;
pub mod kernels;
pub mod launch;
pub mod llada2_gate_host;
pub mod log_mel_backward_host;
pub mod log_mel_host;
pub mod lstm_host;
pub mod miopen;
pub mod ms_deform_attn_host;
pub mod rng_host;
pub mod roctx;
pub mod scan_host;
pub mod splat_host;
#[cfg(feature = "native-splat")]
pub mod splat_native;
pub mod training_bwd_host;
pub mod umap_knn_host;
pub mod unfuse;
pub mod welch_peaks_dispatch;
pub mod welch_peaks_host;

// Dev-only HIP-CPU validation FFI. Bindings are reused verbatim from
// rlx-cuda via `#[path]`; only compiled under `--features hip-cpu-validate`.
#[cfg(feature = "hip-cpu-validate")]
pub mod cpu_dispatch;

pub use backend::{CompileMode, ExecMode, RocmExecutable};

/// True when a HIP-capable AMD GPU is reachable via libamdhip64.
pub fn is_available() -> bool {
    device::rocm_context().is_some()
}
