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
//! via hipRTC, falling through on any setup error. Batched symmetric
//! eigendecomposition (`n ≤ 32`) uses hipSOLVER `SsyevjBatched` when
//! `libhipsolver` is available.

pub mod arena;
pub mod backend;
pub mod supported_ops;
pub use supported_ops::SUPPORTED_OPS;
pub mod calibrate;
pub mod collective_host;
pub mod dense_solve_native;
pub mod device;
pub mod eigh_native;
pub mod fft_dispatch;
pub mod fft_host {
    pub use super::host_ops::run_fft1d;
}
pub mod gdn_host {
    pub use super::host_ops::run_gated_delta_net;
}
pub mod gguf_gpu;
pub mod gguf_host;
pub mod hip;
pub mod hipblas;
pub mod hipblaslt;
pub mod hipsolver;
pub mod host_misc;
pub mod host_ops;
pub mod host_stage;
pub mod host_staging;
pub mod im2col_host {
    pub use super::host_ops::run_im2col;
}
pub mod kernels;
pub mod launch;
pub mod rocm_gpu_kernels;
pub mod vmath;
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
pub mod gru_host {
    pub use super::host_ops::run_gru;
}
pub mod gru_gpu;
pub mod rnn_host {
    pub use super::host_ops::run_rnn;
}
pub mod rnn_gpu;
pub mod mamba2_host {
    pub use super::host_ops::run_mamba2;
}
pub mod mamba2_gpu;
pub mod miopen;
pub mod ms_deform_attn_host {
    pub use super::host_ops::run_ms_deform_attn;
}
pub mod rng_host;
pub mod roctx;
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
pub mod training_bwd_host {
    pub use super::host_ops::{
        run_cumsum_backward, run_gather_backward, run_rms_norm_backward_beta,
        run_rms_norm_backward_gamma, run_rms_norm_backward_input, run_rope_backward,
    };
}
pub mod umap_knn_host {
    pub use super::host_ops::run_umap_knn;
}
pub mod unfuse;
pub mod welch_peaks_dispatch;
pub mod welch_peaks_host {
    pub use super::host_ops::run_welch_peaks;
}

// Dev-only HIP-CPU validation FFI. Bindings are reused verbatim from
// rlx-cuda via `#[path]`; only compiled under `--features hip-cpu-validate`.
#[cfg(feature = "hip-cpu-validate")]
pub mod cpu_dispatch;

pub use backend::{CompileMode, ExecMode, RocmExecutable};

/// True when a HIP-capable AMD GPU is reachable via libamdhip64.
pub fn is_available() -> bool {
    device::rocm_context().is_some()
}
