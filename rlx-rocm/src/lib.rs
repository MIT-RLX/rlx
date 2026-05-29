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
//! Sister crate to `rlx-cuda`. The CUDA C++ kernel sources in
//! Kernel sources live in `rlx-gpu-kernels` (CUDA/HIP-compatible `.cu`
//! files). The dispatch
//! ladder mirrors `rlx-cuda`'s: hipBLASLt → hipBLAS → MIOpen-conv
//! → custom kernels via hipRTC, falling through on any setup error.
//!
//! ## Status: Mac-iterable scaffold
//!
//! This crate currently compiles on any host but has no real HIP
//! dispatch wired up — `is_available()` always returns `false`,
//! `RocmExecutable::compile()` panics with a clear message. The
//! intent is to ship the structural skeleton (workspace member,
//! IR plumbing, kernel-source plumbing, basic-test harness) so we
//! can iterate the real HIP runtime bindings + dispatch path as
//! drop-in additions, the same way `rlx-cuda` started.

pub mod arena;
pub mod backend;
pub mod device;
pub mod fft_dispatch;
pub mod fft_host;
pub mod gdn_host;
pub mod gguf_host;
pub mod hip;
pub mod hipblas;
pub mod hipblaslt;
pub mod kernels;
pub mod launch;
pub mod llada2_gate_host;
pub mod miopen;
pub mod roctx;
pub mod splat_host;
#[cfg(feature = "native-splat")]
pub mod splat_native;
pub mod training_bwd_host;
pub mod umap_knn_host;
pub mod unfuse;

// Dev-only HIP-CPU validation FFI. Bindings are reused verbatim from
// rlx-cuda via `#[path]`; only compiled under `--features hip-cpu-validate`.
#[cfg(feature = "hip-cpu-validate")]
pub mod cpu_dispatch;

pub use backend::{CompileMode, ExecMode, RocmExecutable};

/// True if a HIP-capable AMD GPU is reachable. Returns false today
/// because the HIP runtime bindings haven't landed yet — once they do,
/// this will probe `libamdhip64` via the same panic-catching dlopen
/// pattern `rlx-cuda` uses for `libcuda`.
pub fn is_available() -> bool {
    device::rocm_context().is_some()
}
