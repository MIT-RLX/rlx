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

//! RLX CUDA backend — NVIDIA GPUs via the pure-Rust `cudarc` crate.
#![allow(clippy::too_many_arguments)]
//
// FFI shim helpers (cudnn_conv*, cublaslt_matmul_fused, etc.) inherently
// take many arguments — they mirror the underlying C API surface.
// Suppressing the lint at crate scope avoids drowning out signal warnings.
//!
//! Same overall shape as rlx-wgpu (device singleton, arena buffer, per-op
//! kernels, command-stream-per-forward-pass) but targeting CUDA via
//! `cudarc::driver` for memory + dispatch and `cudarc::cublas` for matmul.
//! Element-wise / reduction / shape kernels are CUDA C++ source strings
//! compiled at init time via NVRTC — same pattern as rlx-wgpu's WGSL
//! kernels.
//!
//! The crate uses `cudarc`'s `fallback-dynamic-loading` feature so it
//! compiles on Mac (and any other host without a CUDA SDK). `is_available()`
//! returns false when libcuda can't be dlopen()'d — every other entry
//! point checks this and degrades cleanly.
//!
//! Layout:
//! - `device`   — `CudaContext` singleton (per-process), driver init
//! - `arena`    — device buffer + per-node offsets
//! - `kernels`  — CUDA C++ source strings + NVRTC compile + cuModule cache
//! - `backend`  — `CudaExecutable`: IR lowering, schedule, run

pub mod arena;
pub mod backend;
pub mod config;
pub mod supported_ops;
pub use config::{CudaRuntimeConfig, install_runtime_config, reload_runtime_config, runtime_config};
pub use supported_ops::SUPPORTED_OPS;
pub mod calibrate;
pub mod collective_host;
#[cfg(feature = "nccl")]
pub mod distributed;
#[cfg(feature = "cufft")]
pub mod cufft_dispatch;
pub mod device;
pub mod fft_dispatch;
pub mod fft_host {
    pub use super::host_ops::run_fft1d;
}
pub mod eigh_native;
pub mod gdn_host {
    pub use super::host_ops::run_gated_delta_net;
}
pub mod gguf_gpu;
pub mod gguf_host;
pub mod host_misc;
pub mod host_ops;
pub mod host_stage;
pub mod host_staging;
pub mod im2col_host {
    pub use super::host_ops::run_im2col;
}
pub mod iq_grid;
pub mod kernels;
pub mod llada2_gate_host {
    pub use super::host_ops::run_llada2_group_limited_gate;
}
pub mod log_mel_backward_host {
    pub use super::host_ops::run_log_mel_backward;
}
pub mod log_mel_host {
    pub use super::host_ops::run_log_mel;
}
pub mod lstm_host {
    pub use super::host_ops::run_lstm;
}
pub mod lstm_gpu;
pub mod ms_deform_attn_host {
    pub use super::host_ops::run_ms_deform_attn;
}
pub mod cuda_gpu_kernels;
#[cfg(feature = "native-cuda-fft")]
pub mod native_fft_dispatch;
pub mod onnx_custom_host;
pub mod rng_host;
pub mod sam_ops_host;
pub mod scan_host;
pub mod spd;
pub mod spd_host;
pub mod splat_host {
    pub use super::host_ops::{
        run_gaussian_splat_prepare, run_gaussian_splat_rasterize, run_gaussian_splat_render,
        run_gaussian_splat_render_backward,
    };
}
#[cfg(feature = "native-splat")]
pub mod splat_native;
pub mod training_bwd_host;
pub mod umap_knn_host {
    pub use super::host_ops::run_umap_knn;
}
pub mod unfuse;
pub mod welch_peaks_dispatch;
pub mod welch_peaks_host {
    pub use super::host_ops::run_welch_peaks;
}

pub use backend::{CompileMode, CudaExecutable, ExecMode};

/// Release pooled CUDA arena buffers (see [`arena::trim_f32_arena_pool`]).
pub fn trim_device_memory_pool() {
    arena::trim_f32_arena_pool();
}

/// HIP-CPU validation path — runs `.cu` kernels on CPU threads so we
/// can numerically validate them on Mac/Docker without renting a CUDA
/// box. Strictly a dev feature; never enabled in production.
#[cfg(feature = "hip-cpu-validate")]
pub mod cpu_dispatch;

/// True if a CUDA driver is reachable. With `dynamic-loading`, this
/// returns false on hosts without `libcuda` (Mac, headless boxes, CI
/// runners without GPUs) — the crate still compiled, but no kernel
/// dispatch is possible.
pub fn is_available() -> bool {
    device::cuda_context().is_some()
}
