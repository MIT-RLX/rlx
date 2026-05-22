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

// `objc` crate's `class!` / `msg_send!` macros expand to
// `cfg(feature = "cargo-clippy")` checks that aren't recognized by
// modern rustc. The warnings are third-party noise (~78 across this
// crate); they say nothing about our code. Silence at the crate root.
#![allow(unexpected_cfgs)]

//! RLX Metal backend — Apple Silicon GPU execution.
//!
//! Compiles RLX IR graphs to Metal compute pipelines + MPS matrix kernels.
//!
//! Architecture mirrors rlx-cpu:
//! - `device` — Metal device discovery and properties
//! - `arena`  — GPU buffer allocation from memory plan
//! - `blas`   — MPS matrix multiplication (analog of cblas_sgemm)
//! - `kernels`— custom MSL compute shaders (analog of NEON kernels)
//! - `thunk`  — pre-compiled command buffer with arena offsets
//! - `backend`— ExecutableGraph implementation
//!
//! Apple Silicon advantages:
//! - Unified memory: zero-copy CPU↔GPU
//! - 16-core GPU on M4 Pro: ~1.4 TFLOP/s peak
//! - 273 GB/s memory bandwidth (vs 120 on CPU)
//! - MPSMatrixMultiplication uses dedicated matmul hardware

#[cfg(target_os = "macos")]
pub mod device;

#[cfg(target_os = "macos")]
pub mod arena;

#[cfg(target_os = "macos")]
pub mod blas;

#[cfg(target_os = "macos")]
pub mod mps_blas;

#[cfg(target_os = "macos")]
pub mod mps_graph;

#[cfg(target_os = "macos")]
pub mod mps_graph_lower;
pub mod mps_graph_hybrid;

#[cfg(target_os = "macos")]
pub mod icb;

#[cfg(target_os = "macos")]
pub mod kernels;

#[cfg(target_os = "macos")]
pub mod llada2_gate;

#[cfg(target_os = "macos")]
pub mod cost;

#[cfg(target_os = "macos")]
pub mod calibrate;

#[cfg(target_os = "macos")]
pub mod thunk;

#[cfg(target_os = "macos")]
pub mod backend;

#[cfg(all(feature = "native-splat", target_os = "macos"))]
pub mod splat_native;
#[cfg(all(feature = "native-splat", target_os = "macos"))]
pub mod splat_training;
#[cfg(all(feature = "native-splat", target_os = "macos"))]
pub mod splat_adam;
#[cfg(all(feature = "native-splat", target_os = "macos"))]
pub mod splat_training_pipeline;

#[cfg(target_os = "macos")]
pub mod async_copy;

#[cfg(target_os = "macos")]
pub mod op_registry;

/// PLAN: Schedule splitting for the Metal MPSGraph path. Splits the
/// schedule at attention boundaries so the broken slice-of-computed
/// MPSGraph attention pattern is replaced by the parity-correct
/// thunk path; everything else still gets the MPSGraph dispatch-
/// overhead reduction. Scaffolding only today (data model +
/// segmenter + 3 unit tests); executor wiring + per-segment plan
/// compilation is the next chunk.
pub mod segmented;

/// Stub when not on macOS — Metal is only available on Apple platforms.
#[cfg(not(target_os = "macos"))]
pub fn is_available() -> bool {
    false
}

#[cfg(target_os = "macos")]
pub fn is_available() -> bool {
    device::has_metal_device()
}
