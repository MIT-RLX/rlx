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

//! RLX Runtime — the user-facing API.
//!
//! Provides a unified [`Session`] that compiles and executes IR graphs
//! on the selected backend. Backend selection is via Cargo features:
//!
//! ```toml
//! [dependencies]
//! rlx-runtime = { version = "0.1", features = ["cpu"] }                # CPU (default)
//! rlx-runtime = { version = "0.1", features = ["blas-accelerate"] }    # CPU + Apple Accelerate
//! rlx-runtime = { version = "0.1", features = ["blas-mkl"] }           # CPU + Intel MKL
//! # rlx-runtime = { version = "0.1", features = ["gpu"] }             # GPU via wgpu
//! # rlx-runtime = { version = "0.1", features = ["cuda"] }            # GPU via CUDA
//! ```
//!
//! # Example
//! ```rust,no_run
//! use rlx_runtime::*;
//! use rlx_ir::*;
//!
//! // Build a graph
//! let mut g = Graph::new("example");
//! let x = g.input("x", Shape::new(&[2, 4], DType::F32));
//! let w = g.param("w", Shape::new(&[4, 3], DType::F32));
//! let b = g.param("b", Shape::new(&[3], DType::F32));
//! let mm = g.matmul(x, w, Shape::new(&[2, 3], DType::F32));
//! let out = g.binary(op::BinaryOp::Add, mm, b, Shape::new(&[2, 3], DType::F32));
//! g.set_outputs(vec![out]);
//!
//! // Compile and execute
//! let session = Session::new(Device::Cpu);
//! let mut compiled = session.compile(g);
//! compiled.set_param("w", &[1.0f32; 12]);
//! compiled.set_param("b", &[0.0f32; 3]);
//! let result = compiled.run(&[("x", &[1.0f32; 8])]);
//! ```

// Driver-layer concerns (device, arena, handle, stream, buffer)
// live in rlx-driver as of plan #58; re-exported below so
// existing callers compile unchanged.
pub mod aot_cache;
pub mod attn_mask;
pub mod backend;
pub mod backends_manifest;
pub mod compile_cache;
pub mod compile_config;
pub mod compiled;
// On-device (ANE) gradient-based training + MLUpdateTask eligibility probe.
#[cfg(all(feature = "coreml", feature = "training"))]
pub mod coreml_training;
pub mod cost;
mod cpu_low_precision;
mod deferred;
pub mod device_bench;
pub mod device_ext;
pub mod device_parse;
pub mod device_policy;
pub mod expert_pool;
pub mod graph_io;
pub mod jacfwd;
pub mod kernel_trace;
pub mod kv_cache;
pub mod lora_scheduler;
pub mod memory_estimate;
pub mod model_pipeline;
pub mod moe_expert_store;
#[cfg(feature = "cpu")]
pub mod onnx_active;
pub mod op_registry;
pub mod options;
pub mod paged_kv;
pub mod precision;
pub mod precompile;
pub mod quantized_kv;
pub mod record_replay;
pub mod reflect;
pub mod registry;
pub mod router;
pub mod samplers;
pub mod session;
pub mod stages;
pub mod subgraph;
pub mod trace;
pub mod weight_registry;
pub mod weights;
pub mod worker_pool;
/// PLAN L3 — Perfetto / chrome-trace JSON tracing. Lives in `rlx-ir`
/// (alongside the `Tick` cycle counter it depends on) so every backend
/// can instrument per-thunk without crate-dep gymnastics. Re-exported
/// here so callers see one consistent `rlx_runtime::perfetto::TraceSpan`.
pub use rlx_ir::perfetto;
pub mod custom_ops;
pub mod device_router;
pub mod flexible_session;
pub mod graph_devices;
pub mod hwinfo;
pub mod lm;
pub mod logit_verify;
pub mod nan_check;
pub mod phase;
pub mod spec_decode;
pub mod telemetry;
pub mod validators;

// Always-available now that serde is a non-optional dep + #32
// router consumes the OpenAI-shaped structs unconditionally.
pub mod mock_requests;

// Driver-layer types — re-exported from rlx-driver (plan #58).
pub use rlx_driver::{Buffer, BufferHandle, CommandStream, Device, DeviceArena, SyncStream};
// Symmetric-memory primitives (plan #49) — foundation for #12.
pub use rlx_driver::{
    CollectiveError, LocalTransport, Rank, SymmetricBuffer, SymmetricHeap, SymmetricTransport,
};
// Collective ops (plan #12).
pub use aot_cache::{AotCache, AotCacheError};
pub use backend::{Backend, ExecutableGraph, compile_hir, compile_module};
pub use backends_manifest::BackendsManifest;
pub use compile_cache::{
    BucketedCompileCache, CacheRunInput, CompileCache, DynamicDimCompileCache, pad_rows, slice_rows,
};
pub use compile_config::{
    COMPILE_OUTPUT_CAP_ENV, COMPILE_OUTPUT_CAP_ENV_MLX, DEFAULT_COMPILE_OUTPUT_CAP,
    compile_output_cap, device_has_compile_output_cap, reset_compile_output_cap,
    set_compile_output_cap,
};
pub use compiled::CompiledGraph;
#[cfg(all(feature = "coreml", feature = "training"))]
pub use coreml_training::{
    CoremlTrainingSession, MlUpdateEligibility, Optimizer, StepReport, TrainConfig, UpdatePath,
};
pub use cost::fastest_device_for;
pub use device_bench::{DeviceBenchResult, benchmark_devices, warm_all};
#[cfg(feature = "apple")]
pub use device_ext::available_apple_devices;
pub use device_ext::{
    available_devices, devices_for, dispatch_report_for_device,
    dispatch_report_for_device_with_options, fastest_device, first_unsupported_op,
    first_unsupported_op_with_options, full_name, is_available, legalize_graph_for_device,
    legalize_graph_for_device_with_options, legalize_graph_for_device_with_report, supports,
    supports_graph, supports_graph_with_options, supports_run_slots,
};
pub use device_parse::{ParseDeviceError, device_label, parse_device, parse_device_list};
pub use device_policy::{
    DeviceCandidate, DeviceFallbackError, DevicePickStrategy, DevicePolicy, device_chain_from_env,
    device_chain_from_env_key, device_from_env, device_from_env_key, device_report,
    devices_for_with_policy, resolve_device, resolve_device_chain, run_with_fallback,
};
pub use device_router::DeviceRouter;
pub use expert_pool::{
    ExpertPool, ExpertPoolConfig, ExpertPoolStats, ExpertRefreshPolicy, ExpertRefreshResult,
    MoEExecMode, gpu_expert_budget_from_vram,
};
pub use flexible_session::FlexibleSession;
pub use graph_devices::{GraphDevices, graph_param_names};
pub use kv_cache::LayerKvCache;
pub use lm::{
    ConfigSource, LmRunner, LmRunnerBuilder, MirostatMode, ModelRegistration,
    PACKED_GGUF_AUTO_THRESHOLD_BYTES, SampleOpts, WeightFormat, auto_runner_name,
    registered_models,
};
pub use memory_estimate::{
    DEFAULT_SOFT_MEMORY_FRACTION, MoeOffloadEstimate, available_unified_memory,
    estimate_moe_offload, llama_decode_bucket_compile_peak_bytes,
    llama_decode_oneshot_compile_peak_bytes, memory_headroom_bytes, process_rss_bytes,
    soft_memory_budget_bytes, soft_memory_fraction, would_exceed_soft_budget,
};
pub use model_pipeline::ModelCompilePipeline;
pub use options::CompileOptions;
pub use precision::Precision;
pub use reflect::{ModelReflection, load_hir_template_with_extensions, specialize_entry};
pub use registry::{BackendFactory, backend_for, register_backend, registered_devices};

/// Alias for [`COMPILE_OUTPUT_CAP_ENV`].
pub const MLX_COMPILE_OUTPUT_CAP_ENV: &str = COMPILE_OUTPUT_CAP_ENV;

/// Alias for [`DEFAULT_COMPILE_OUTPUT_CAP`].
pub const DEFAULT_MLX_COMPILE_OUTPUT_CAP: usize = DEFAULT_COMPILE_OUTPUT_CAP;

/// Alias for [`compile_output_cap`].
#[inline]
pub fn mlx_compile_output_cap() -> usize {
    compile_output_cap()
}

/// Alias for [`set_compile_output_cap`].
#[inline]
pub fn set_mlx_compile_output_cap(cap: usize) {
    set_compile_output_cap(cap);
}

/// Alias for [`reset_compile_output_cap`].
#[inline]
pub fn reset_mlx_compile_output_cap() {
    reset_compile_output_cap();
}

#[cfg(feature = "cpu")]
pub use rlx_cpu::moe_residency::MoeResidencyStats;
#[cfg(feature = "cpu")]
pub use rlx_cpu::moe_topk_capture::MoeTopkCapture;
pub use rlx_driver::{
    DEFAULT_HEAP_BYTES, NetTransport, ProcessGroup, TcpTransport, ThunderboltTransport, Transport,
};
pub use rlx_driver::{ReduceKind, all_gather, all_reduce, reduce_scatter};
pub use rlx_ir::env::{self, RlxEnv, RuntimeOverrides};
pub use session::Session;
pub use stages::{
    compile_graph_stages, compile_graph_stages_for_backend, compile_hir_stages,
    compile_module_stages, fusion_target_for, graph_from_lir, maybe_log_fusion,
    options_with_supported_ops, pipeline_for,
};
pub use subgraph::{SubgraphCache, run_if, run_while};

pub use expert_pool::{merged_resident_mask, per_layer_resident_masks};
pub use moe_expert_store::{ExpertStackF32, LayerMoeWeights, MoeExpertStore};
pub use weight_registry::{WeightEntry, WeightHandle, WeightKind, WeightRegistry};
pub use weights::{BytesWeightLoader, WeightLoader};

// Cycle-accurate timing primitive lives in rlx-ir (lowest crate); re-export
// here so `rlx_runtime::Tick` works without forcing callers to add the IR
// crate as a direct dep.
pub use rlx_ir::{AsyncCopy, BarrierToken, DoubleBuffer, SyncCopy};
pub use rlx_ir::{CacheBuster, Tick, time_ns};

// Re-export precision policy from rlx-opt for convenience
pub use rlx_ir::{
    inspect_graph, inspect_hir, inspect_hir_stats, inspect_lir, inspect_mir, inspect_mir_stats,
};
pub use rlx_opt::{OpKind, PrecisionPolicy};
pub use rlx_opt::{PipelineInspect, inspect_pipeline};

// Re-export IR types for convenience
pub use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
pub use rlx_ir::op;
pub use rlx_ir::{
    BindingManifest, CompilationMode, DType, Graph, HirExtensionFn, HirReflection, IoBindingEntry,
    ManifestDiff, ModelComponent, ModelPhase, ModelVariant, Node, NodeId, Op, RngBackend,
    RngOptions, Shape, WeightBlock, apply_hir_extensions, register_hir_extension,
    registered_hir_extensions,
};

// Re-export proc macro
pub use rlx_macros::pipeline_schedule;
pub use rlx_macros::rlx_model;
