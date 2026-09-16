// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

#[cfg(rlx_metal_host)]
pub mod device;

#[cfg(rlx_metal_host)]
pub mod arena;

#[cfg(rlx_metal_host)]
pub mod blas;

#[cfg(rlx_metal_host)]
pub mod mps_blas;

#[cfg(rlx_metal_host)]
pub mod mps_graph;

#[cfg(rlx_metal_host)]
pub mod mps_graph_hybrid;
#[cfg(rlx_metal_host)]
pub mod mps_graph_lower;

#[cfg(rlx_metal_host)]
pub mod mps_gelu;

/// Hand-rolled Metal bindings, replacing the `metal` crate. Public so downstream
/// `MetalGpuKernel` authors can name the encoder/buffer types they are handed:
/// `use rlx_metal::mtl as metal;` keeps existing custom kernels source-compatible.
#[cfg(rlx_metal_host)]
pub mod mtl;

#[cfg(rlx_metal_host)]
pub mod icb;

/// Gated on `rlx_metal_host` because it reads `crate::kernels::RLX_KERNELS_MSL`
/// to scan the shipping MSL for `simdgroup_load` strides — and `kernels` is
/// host-gated. Leaving this ungated compiled on macOS and broke the *Linux*
/// build of this crate, which is the `linux_workspace_test_gate` trap: a
/// workspace `cargo test` builds every crate, Apple-only ones included, so one
/// ungated Apple module stops **every** test on the ROCm rig from running.
#[cfg(rlx_metal_host)]
pub mod kernel_schedule_port;

/// The Apple kernel knobs, in one place — builder, config and CLI, one parser.
///
/// Five parameters, each targeting a limiter measured on Apple silicon, and a
/// single `RLX_METAL_PARAMS` variable rather than one per knob.
pub mod apple_params;

/// Does more threadgroup memory pay for itself on this Apple GPU?
///
/// Apple hides memory latency with occupancy, not with software pipelining, so
/// the CAKE-shaped question "how deep should the pipeline be" has an
/// Apple-shaped answer that is usually "shallower". Calibrated from measured
/// runs; refuses to predict on chips it has not seen.
pub mod occupancy;

/// Generate the tiled sgemm entry point *from* a typed schedule rather than
/// from the hand-written MSL in [`kernels`].
///
/// Feature-gated because it is a second implementation of a shipping kernel:
/// until it is measured at least as fast on Apple hardware, `kernels.rs` stays
/// the default and this is opt-in.
#[cfg(feature = "schedule-codegen")]
pub mod kernel_schedule_emit;
#[cfg(rlx_metal_host)]
pub mod kernels;

pub mod vmath;

#[cfg(rlx_metal_host)]
pub mod fft_dispatch;

/// CPU host-fallback for the core Riemannian / SPD-manifold ops (BiMap /
/// ReEig / LogEig / SpdBatchNorm / SpdKarcherMean + backwards). No MSL
/// eigen kernel; they run `rlx_cpu::spd` (F64) against the unified-memory
/// arena between GPU segments, like `Op::Fft`. See `crate::spd`.
#[cfg(rlx_metal_host)]
pub mod spd;

#[cfg(rlx_metal_host)]
pub mod hc_sinkhorn_gate;
#[cfg(rlx_metal_host)]
pub mod llada2_gate;
#[cfg(rlx_metal_host)]
pub mod ms_deform_attn;

#[cfg(rlx_metal_host)]
pub mod config;
#[cfg(rlx_metal_host)]
pub use config::{
    MetalRuntimeConfig, install_runtime_config, reload_runtime_config, runtime_config,
};

#[cfg(rlx_metal_host)]
pub mod cost;

#[cfg(rlx_metal_host)]
pub mod calibrate;

#[cfg(rlx_metal_host)]
pub mod thunk;

#[cfg(rlx_metal_host)]
pub mod backend;

#[cfg(rlx_metal_host)]
pub mod attention_bwd_gpu;

/// Device-side span of the last run, from the command buffer's own
/// `GPUStartTime`/`GPUEndTime` — the part a kernel change can actually move.
#[cfg(rlx_metal_host)]
pub mod gpu_span;

#[cfg(rlx_metal_host)]
pub mod thunk_profile;

#[cfg(rlx_metal_host)]
pub mod mps_profile;

#[cfg(all(feature = "native-splat", rlx_metal_host))]
pub mod splat_adam;
#[cfg(all(feature = "native-splat", rlx_metal_host))]
pub mod splat_native;
#[cfg(all(feature = "native-splat", rlx_metal_host))]
pub mod splat_training;
#[cfg(all(feature = "native-splat", rlx_metal_host))]
pub mod splat_training_pipeline;

/// Double-single (2× f32 ≈ f64) reductions — near-f64 precision on Metal, which
/// has no native f64. Compiled with precise math (fast-math breaks EFT).
#[cfg(rlx_metal_host)]
pub mod double_single;
#[cfg(rlx_metal_host)]
pub mod pipeline_cache;

#[cfg(rlx_metal_host)]
pub mod onnx_qmatmul;

#[cfg(rlx_metal_host)]
pub mod async_copy;

#[cfg(rlx_metal_host)]
pub mod op_registry;

#[cfg(rlx_metal_host)]
pub mod prefill_stats;

#[cfg(rlx_metal_host)]
pub mod collective;

/// Legalization op claim — always available (no Metal device required).
pub mod supported_ops;

/// Dispatch-table persistence + the arch key.
///
/// Gated on `rlx_metal_host` like `cost` and `calibrate`, because it reads
/// `cost::hw_model()` to name the GPU family. Leaving it ungated compiled fine on
/// macOS and broke the *Linux* build of this crate — which then failed the whole
/// workspace `cargo test` on the ROCm rig, so **zero** tests ran there. That is
/// the `linux_workspace_test_gate` trap: a workspace test run builds every crate,
/// Apple-only ones included, and a green macOS check says nothing about it.
#[cfg(rlx_metal_host)]
pub mod tuning;
pub use supported_ops::SUPPORTED_OPS;

/// PLAN: Schedule splitting for the Metal MPSGraph path. Splits the
/// schedule at attention boundaries so the broken slice-of-computed
/// MPSGraph attention pattern is replaced by the parity-correct
/// thunk path; everything else still gets the MPSGraph dispatch-
/// overhead reduction. Scaffolding only today (data model +
/// segmenter + 3 unit tests); executor wiring + per-segment plan
/// compilation is the next chunk.
pub mod segmented;

/// Typed kernel planning, precision legality, token bucketing, and two-stage
/// build/launch configuration for Metal kernel families.
pub mod kernel_plan;

/// Whether a usable Metal device is present. `rlx-metal` is a Metal-only
/// dependency (its consumers gate it to `cfg(all(target_vendor = "apple",
/// not(target_os = "watchos")))` — every Apple platform with Metal: macOS,
/// iOS, tvOS, visionOS), so this is never a non-Apple `false` stub — callers
/// on other platforms (and watchOS) report Metal availability via the
/// runtime's own device-feature check, not this crate.
#[cfg(rlx_metal_host)]
pub fn is_available() -> bool {
    device::has_metal_device()
}
