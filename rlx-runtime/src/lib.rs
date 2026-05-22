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
pub mod backend;
pub mod compile_cache;
pub mod compiled;
pub mod cost;
pub mod device_ext;
pub mod jacfwd;
pub mod kernel_trace;
pub mod lora_scheduler;
pub mod expert_pool;
pub mod moe_expert_store;
pub mod memory_estimate;
pub mod model_pipeline;
pub mod reflect;
pub mod op_registry;
pub mod options;
pub mod paged_kv;
pub mod precision;
pub mod record_replay;
pub mod registry;
pub mod router;
pub mod stages;
pub mod session;
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
pub mod hwinfo;
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
pub use backend::{Backend, ExecutableGraph, compile_hir, compile_module};
pub use stages::{
    compile_graph_stages, compile_graph_stages_for_backend, compile_hir_stages,
    compile_module_stages, fusion_target_for, graph_from_lir, maybe_log_fusion,
    options_with_supported_ops, pipeline_for,
};
pub use aot_cache::{AotCache, AotCacheError};
pub use compile_cache::{BucketedCompileCache, CompileCache, DynamicDimCompileCache, pad_rows, slice_rows};
pub use model_pipeline::ModelCompilePipeline;
pub use reflect::{load_hir_template_with_extensions, specialize_entry, ModelReflection};
pub use compiled::CompiledGraph;
#[cfg(feature = "apple")]
pub use device_ext::available_apple_devices;
pub use device_ext::{
    available_devices, dispatch_report_for_device, dispatch_report_for_device_with_options,
    first_unsupported_op, first_unsupported_op_with_options, full_name, is_available,
    legalize_graph_for_device,
    legalize_graph_for_device_with_options, legalize_graph_for_device_with_report, supports,
    supports_graph, supports_graph_with_options,
};
pub use options::CompileOptions;
pub use rlx_ir::env::{self, RlxEnv, RuntimeOverrides};
pub use precision::Precision;
pub use registry::{BackendFactory, backend_for, register_backend, registered_devices};
pub use rlx_driver::{ReduceKind, all_gather, all_reduce, reduce_scatter};
pub use session::Session;
pub use subgraph::{SubgraphCache, run_if, run_while};
pub use expert_pool::{
    ExpertPool, ExpertPoolConfig, ExpertPoolStats, ExpertRefreshPolicy, ExpertRefreshResult,
    MoEExecMode, gpu_expert_budget_from_vram,
};
pub use memory_estimate::{MoeOffloadEstimate, estimate_moe_offload};
#[cfg(feature = "cpu")]
pub use rlx_cpu::moe_topk_capture::MoeTopkCapture;
#[cfg(feature = "cpu")]
pub use rlx_cpu::moe_residency::MoeResidencyStats;

pub use expert_pool::{merged_resident_mask, per_layer_resident_masks};
pub use moe_expert_store::{
    ExpertStackF32, LayerMoeWeights, MoeExpertStore,
};
pub use weight_registry::{WeightEntry, WeightHandle, WeightKind, WeightRegistry};
pub use weights::{BytesWeightLoader, WeightLoader};

// Cycle-accurate timing primitive lives in rlx-ir (lowest crate); re-export
// here so `rlx_runtime::Tick` works without forcing callers to add the IR
// crate as a direct dep.
pub use rlx_ir::{AsyncCopy, BarrierToken, DoubleBuffer, SyncCopy};
pub use rlx_ir::{CacheBuster, Tick, time_ns};

// Re-export precision policy from rlx-opt for convenience
pub use rlx_opt::{OpKind, PrecisionPolicy};
pub use rlx_ir::{
    inspect_graph, inspect_hir, inspect_hir_stats, inspect_lir, inspect_mir, inspect_mir_stats,
};
pub use rlx_opt::{inspect_pipeline, PipelineInspect};

// Re-export IR types for convenience
pub use rlx_ir::op;
pub use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
pub use rlx_ir::{
    apply_hir_extensions, register_hir_extension, registered_hir_extensions, BindingManifest,
    CompilationMode, DType, Graph, HirExtensionFn, HirReflection, IoBindingEntry, ManifestDiff,
    ModelComponent, ModelPhase, ModelVariant, Node, NodeId, Op, Shape, WeightBlock,
};

// Re-export proc macro
pub use rlx_macros::pipeline_schedule;
pub use rlx_macros::rlx_model;
