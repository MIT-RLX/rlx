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

//! # RLX
//!
//! A small ML compiler + runtime for transformer inference and training,
//! with a JAX-shaped IR + autodiff + transforms (`jvp`, `hvp`, `vmap`)
//! on top of CPU / Apple Silicon (Metal / MLX) / NVIDIA (CUDA) / AMD
//! (ROCm) / Google TPU / cross-platform GPU (wgpu) / FPGA / Cortex-M
//! backends.
//!
//! This is the **prelude crate** — pulls in the framework-level
//! workspace members and re-exports the common types so a one-line
//! `use rlx::prelude::*;` covers most usage.
//!
//! ## Three usage patterns
//!
//! ### 1. Build + run a graph by hand
//!
//! ```ignore
//! use rlx::prelude::*;
//!
//! let mut g = Graph::new("hello");
//! let x = g.input("x", Shape::new(&[1, 4], DType::F32));
//! let w = g.param("w", Shape::new(&[4, 2], DType::F32));
//! let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
//! let scaled = g.mul(x, g.constant(2.0, DType::F32)); // GraphExt literal
//! g.set_outputs(vec![y, scaled]);
//!
//! let mut compiled = Session::new(Device::Cpu).compile(g);
//! compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
//! let out = compiled.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
//! ```
//!
//! ## Module map
//!
//! Every workspace crate is reachable as a module on `rlx`:
//!
//! | path            | crate           | what                                                                            |
//! |-----------------|-----------------|---------------------------------------------------------------------------------|
//! | `rlx::ir`       | `rlx-ir`        | IR types, ops, graph builder                                                    |
//! | `rlx::opt`      | `rlx-opt`       | facade: `rlx-fusion` + `rlx-autodiff` + `rlx-compile`                           |
//! | `rlx::driver`   | `rlx-driver`    | `Device` enum, registries                                                       |
//! | `rlx::runtime`  | `rlx-runtime`   | `Session`, `CompiledGraph`                                                      |
//! | `rlx::macros`   | `rlx-macros`    | `#[rlx_model]` proc macro                                                       |
//! | `rlx::collectives` | `rlx-collectives` | in-graph collective ops + mesh/planner *(feature `distributed`)*             |
//! | `rlx::gguf`     | `rlx-gguf`      | GGUF parser + dequant *(feature `gguf`)*                                        |
//! | `rlx::onnx`     | `rlx-onnx`      | ONNX Runtime `.onnx` inference *(feature `onnx`)*                               |
//! | `rlx::bench`    | `rlx-bench`     | benchmark harness *(feature `bench`)*                                           |
//! | `rlx::sparse`   | `rlx-sparse`    | downstream: sparse linalg *(feature `sparse`)*                                  |
//! | `rlx::splat`    | `rlx-splat`     | 3D Gaussian splatting *(feature `splat`)* — `register()`, decomposed IR ops      |
//! | `rlx::linalg`   | `rlx-linalg`    | downstream: dense linalg via LAPACK *(feature `linalg`)*                        |
//! | `rlx::cortexm`  | `rlx-cortexm`   | INT8 ARMv7E-M kernels *(feature `cortexm`)* — no `Backend` impl, kernels only   |
//! | `rlx::fpga`     | `rlx-fpga`      | IR → SystemVerilog export *(feature `fpga`)* — target-agnostic RTL; no `Backend` |
//!
//! ## Convenience namespaces
//!
//! Grouped re-exports for related concerns — use these when you want
//! one focused subset without star-importing the whole prelude:
//!
//! | namespace            | what                                                                          |
//! |----------------------|-------------------------------------------------------------------------------|
//! | [`rlx::quant`]       | `QuantScheme`, `QuantMap` (IR quantization metadata)                          |
//! | [`rlx::ops`]         | `Activation`, `BinaryOp`, `CmpOp`, `MaskKind`, `ChainStep`, `ChainOperand`    |
//! | [`rlx::autodiff`]    | `jvp`, `hvp`, `vmap` + the autodiff entry points                              |
//! | `rlx::distributed`   | transports + in-graph collectives + ship-graph train/infer *(feature `distributed`)* |
//! | [`rlx::prelude`]     | star-import target covering the 95% case                                      |
//!
//! ## Backend feature gates
//!
//! Pick the ones that match your hardware. Multiple backends can be
//! enabled at once; the runtime picks one per `Session`.
//!
//! | feature             | backend                              | platform                  |
//! |---------------------|--------------------------------------|---------------------------|
//! | `cpu` *(default)*   | NEON / AVX + Accelerate / OpenBLAS   | every host                |
//! | `metal`             | Metal Performance Shaders + MSL      | macOS (Apple Silicon)     |
//! | `mlx`               | Apple MLX (vendored)                 | macOS (Apple Silicon)     |
//! | `gpu`               | wgpu (Vulkan / DX12 / WebGPU / Metal)| cross-platform            |
//! | `cuda`              | cuBLAS / cuDNN / NVRTC               | Linux / Windows + NVIDIA  |
//! | `rocm`              | hipBLAS / MIOpen                     | Linux + AMD               |
//! | `tpu`               | libtpu PJRT plugin                   | Linux + GCP TPU           |
//! | `blas-accelerate`   | macOS Accelerate                     | macOS                     |
//! | `blas-mkl`          | Intel MKL                            | Intel / AMD CPUs          |
//! | `blas-openblas`     | OpenBLAS                             | cross-platform CPU        |
//!
//! ## Convenience aggregates
//!
//! Single-flag setups for common platforms. Each composes the
//! fragments most users want for that target.
//!
//! | feature           | expands to                                  |
//! |-------------------|---------------------------------------------|
//! | `apple-silicon`   | `cpu` + `metal` + `blas-accelerate`         |
//! | `nvidia`          | `cpu` + `cuda`                              |
//! | `edge`            | `cpu` + `cortexm`                           |
//! | `all-cpu`         | `cpu` + `gguf` + `linalg`                   |
//!
//! `mlx` and `rocm` aren't in any aggregate because their crates
//! aren't on crates.io (vendor-bundled submodule / workspace-
//! relative kernel sources). To opt in, depend on the workspace via
//! git and add the feature explicitly:
//!
//! ```toml
//! rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["apple-silicon", "mlx"] }
//! ```

#![doc(html_root_url = "https://docs.rs/rlx/0.2.1")]

// ── Module re-exports ───────────────────────────────────────────

/// Tensor IR — types, shapes, ops, graph builder.
/// See [`rlx-ir`](https://crates.io/crates/rlx-ir).
pub use rlx_ir as ir;

/// Symbolic tensor DSL — operator-overloaded graph building.
/// Available with the `tensor` feature (on by default).
#[cfg(feature = "tensor")]
pub use rlx_tensor as tensor;

/// Graph rewrites + autodiff + vmap.
/// See [`rlx-opt`](https://crates.io/crates/rlx-opt).
pub use rlx_opt as opt;

/// Device enum + cross-cutting types.
/// See [`rlx-driver`](https://crates.io/crates/rlx-driver).
pub use rlx_driver as driver;

/// User-facing `Session` / `CompiledGraph`.
/// See [`rlx-runtime`](https://crates.io/crates/rlx-runtime).
pub use rlx_runtime as runtime;

/// Procedural macros (`#[rlx_model]`, `pipeline_schedule!`).
/// See [`rlx-macros`](https://crates.io/crates/rlx-macros).
pub use rlx_macros as macros;

#[cfg(feature = "gguf")]
/// GGUF v1 / v2 / v3 parser + dequant + quant encoders + writer.
/// See [`rlx-gguf`](https://crates.io/crates/rlx-gguf).
pub use rlx_gguf as gguf;

#[cfg(feature = "gguf-convert")]
/// safetensors / ONNX → GGUF conversion with per-tensor quantization.
/// Useful at first inference load to shrink memory + disk footprint.
/// See [`rlx-gguf-convert`](https://crates.io/crates/rlx-gguf-convert).
pub use rlx_gguf_convert as gguf_convert;

#[cfg(feature = "bench")]
/// Uniform benchmark harness.
/// See [`rlx-bench`](https://crates.io/crates/rlx-bench).
pub use rlx_bench as bench;

#[cfg(feature = "sparse")]
/// Downstream: sparse linear algebra (custom-op scaffold).
/// See [`rlx-sparse`](https://crates.io/crates/rlx-sparse).
pub use rlx_sparse as sparse;

#[cfg(feature = "linalg")]
/// Downstream: dense linalg via LAPACK (custom-op scaffold).
/// See [`rlx-linalg`](https://crates.io/crates/rlx-linalg).
pub use rlx_linalg as linalg;

#[cfg(feature = "splat")]
/// Downstream: 3D Gaussian splatting (CPU reference render custom op).
/// See [`rlx-splat`](https://crates.io/crates/rlx-splat).
pub use rlx_splat as splat;

#[cfg(feature = "umap")]
/// Downstream: UMAP / fast-umap custom ops (k-NN from pairwise distances).
pub use rlx_umap as umap;

#[cfg(feature = "optim")]
/// Training-step optimizers (Adam, AdamW, NAdamW, RAdam, QHAdamW,
/// LAMB, Adafactor, Lion, SOAP, Kron-PSGD, Muon, Sophia, MARS). See
/// [`rlx-optim`](https://crates.io/crates/rlx-optim).
pub use rlx_optim as optim;

#[cfg(feature = "cortexm")]
/// `no_std` ARMv7E-M INT8 kernels (Cortex-M4F / M7). Doesn't
/// implement `Backend` — call the kernels (`dense`, `conv2d`,
/// `maxpool`, `relu`, `argmax`) directly.
/// See [`rlx-cortexm`](https://crates.io/crates/rlx-cortexm).
pub use rlx_cortexm as cortexm;

#[cfg(feature = "fpga")]
/// IR → SystemVerilog datapath synthesis + runtime [`export`](rlx_runtime::export).
///
/// Prefer the prelude when the `fpga` feature is on:
///
/// ```ignore
/// use rlx::prelude::*;
///
/// let arts = ExportSession::fpga("hw/out")
///     .hw_target(HwTarget::Generic)
///     .export_model(&tinyconv_mnist_from_cortexm())?;
/// ```
///
/// Entry via module path: `rlx::fpga::export_graph` / `emit_with_config`.
/// Soft-port RTL by default (`HwTarget::Generic`); optional ECP5/iCE40/Xilinx7
/// synth scripts. See [`rlx-fpga`](https://crates.io/crates/rlx-fpga).
pub use rlx_fpga as fpga;

#[cfg(feature = "onnx")]
/// ONNX Runtime inference for `.onnx` files on RLX [`Device`] backends.
/// See [`rlx-onnx`](https://crates.io/crates/rlx-onnx).
pub use rlx_onnx as onnx;

#[cfg(feature = "distributed")]
/// In-graph collective ops (`collective.all_reduce`, all-gather, reduce-scatter,
/// broadcast, all-to-all, ppermute, send/recv, the Megatron `f`/`g` operators),
/// the group registry, and the device-mesh / placement planner. The unified
/// [`rlx::distributed`](crate::distributed) namespace folds this together with
/// the `rlx-driver` transports and the `rlx-runtime::dist` ship-graph API.
/// See [`rlx-collectives`](https://crates.io/crates/rlx-collectives).
pub use rlx_collectives as collectives;

// ── Error types ─────────────────────────────────────────────────
//
// The whole stack returns `anyhow::Result<T>` — `rlx::Result` /
// `rlx::Error` make that the obvious choice for downstream code
// without forcing an explicit `anyhow` dep at the call site.

/// Crate-wide result type — alias of `anyhow::Result<T>`. Use this
/// in `main()` and library boundaries.
pub type Result<T, E = anyhow::Error> = std::result::Result<T, E>;

/// Crate-wide error type — alias of `anyhow::Error`.
pub type Error = anyhow::Error;

// ── Flat re-exports for the most-common types ───────────────────
//
// These cover ~90% of user code: build a graph with rlx_ir types,
// compile + run it through Session, then read back outputs. Less
// common types stay reachable via the module re-exports above.

pub use rlx_driver::Device;
#[cfg(feature = "fpga")]
pub use rlx_fpga::{
    ExportQuantMode, FpgaExportConfig, GraphIoBind, HwTarget, InputIface, IoConfig, OutputIface,
    OutputKind, PortNames, SidebandSpec, tinyconv_mnist_from_cortexm,
};
pub use rlx_ir::quant::QuantScheme;
pub use rlx_ir::{
    DType, Element, FusionPolicy, Graph, GraphExt, GraphModule, GraphStage, HirModule, HirOp,
    LirModule, MirModule, Node, NodeId, Op, OpKind, Shape, Tick, scalar_constant_bytes,
};
pub use rlx_ir::{
    NodeOrigin, inspect_graph, inspect_graph_diff, inspect_hir, inspect_hir_stats, inspect_lir,
    inspect_mir, inspect_mir_diff, inspect_mir_stats, node_label,
};
pub use rlx_opt::{
    CalibrationRecord, CompilePipeline, CompileResult, FusionOptions, FusionReport, FusionTarget,
    MissReason, MissedFusion, Pass, PipelineInspect, Precision, PrecisionPolicy, fusion_passes,
    fusion_passes_for_supported, hvp, inspect_pipeline, jvp, maybe_dump_pipeline,
    supported_for_target, supports_op, vmap,
};
pub use rlx_runtime::{
    BackendsManifest, CompiledGraph, DeviceBenchResult, DeviceCandidate, DeviceFallbackError,
    DevicePickStrategy, DevicePolicy, DeviceRouter, FlexibleSession, GraphDevices,
    ParseDeviceError, Session, available_devices, benchmark_devices, device_chain_from_env,
    device_from_env, device_label, device_report, devices_for, devices_for_with_policy,
    fastest_device, fastest_device_for, graph_param_names, is_available, parse_device,
    parse_device_list, resolve_device, resolve_device_chain, run_with_fallback,
};
#[cfg(feature = "fpga")]
pub use rlx_runtime::{
    ExportOptions, ExportSession, ExportTarget, ExportedArtifacts, export_graph,
    export_tinyconv_mnist,
};

// ── Grouped namespaces ──────────────────────────────────────────

/// Quantization metadata — schemes the IR carries per-tensor, plus
/// the `QuantMap` graph-level annotation. Use these when wiring
/// `Op::DequantMatMul` or attaching quant info to your own ops.
///
/// ```ignore
/// use rlx::quant::QuantScheme;
///
/// let scheme = QuantScheme::GgufQ4K;   // GGUF Q4_K super-block
/// assert!(scheme.is_gguf());
/// assert_eq!(scheme.gguf_block_bytes(), 144);
/// ```
pub mod quant {
    pub use rlx_ir::quant::{QuantMap, QuantScheme};
}

/// Op-builder helper enums — the variants the graph builder methods
/// (`g.binary`, `g.compare`, `g.activation`, `g.attention_kind`, …)
/// take as their first argument, plus the fused-chain primitives
/// used by `Op::ElementwiseRegion`.
///
/// ```ignore
/// use rlx::{Graph, GraphExt, Shape, DType};
/// use rlx::ops::{Activation, BinaryOp};
///
/// let mut g = Graph::new("ex");
/// let x = g.input("x", Shape::new(&[4], DType::F32));
/// let y = g.input("y", Shape::new(&[4], DType::F32));
/// let s = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
/// let r = g.activation(Activation::Silu, s, Shape::new(&[4], DType::F32));
/// let scaled = g.mul(x, g.constant(2.0, DType::F32));
/// g.set_outputs(vec![r, scaled]);
/// ```
pub mod ops {
    pub use rlx_ir::op::{Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind};
}

/// Autodiff + transforms — re-exports the public entry points from
/// `rlx_opt`. Use these when computing gradients or doing
/// `vmap` / `jvp` / `hvp` over a graph.
///
/// ```ignore
/// use rlx::autodiff::{jvp, vmap};
/// ```
pub mod autodiff {
    pub use rlx_opt::{hvp, jvp, vmap};
}

/// Distributed training + inference — the single front door over all three
/// layers, which otherwise live in separate crates: the transport layer
/// (`rlx-driver`: `ProcessGroup`, transports, `Node` discovery, `ReduceMode`),
/// the in-graph collective op builders + placement planner (`rlx-collectives`),
/// and the ship-graph worker/coordinator + heterogeneous placement
/// (`rlx-runtime::dist` / `::hetero`). Feature `distributed`.
///
/// ```ignore
/// use rlx::distributed::*;
///
/// register(); // install the in-graph collective kernel once
/// // reproducible + precise cross-rank gradient reduce, baked into the graph:
/// let g = all_reduce_op_mode(&mut bwd, grad, gid, ReduceKind::Mean, ReduceMode::Deterministic);
/// // ship-graph data-parallel training on a heterogeneous cluster:
/// run_train(&group, rank, &spec, resolve, reduce)?;
/// // one-machine-vs-cluster divergence diagnostic:
/// let d = backend_divergence(&graph, &inputs)?;
/// ```
///
/// The collective ops and mesh/planner are also reachable directly on
/// [`rlx::collectives`](crate::collectives); the ship-graph API on
/// [`rlx::runtime::dist`](crate::runtime) (always present, no feature).
#[cfg(feature = "distributed")]
pub mod distributed {
    // Transport + in-graph collective ops + device mesh / planner
    // (rlx-driver + rlx-collectives), via the collectives prelude.
    pub use rlx_collectives::prelude::*;
    // Ship-graph inference / training / diagnostics (rlx-runtime::dist).
    pub use crate::runtime::dist::{
        BackendDivergence, DataRef, StageSpec, TrainMetrics, TrainSpec, WeightCache, WeightRef,
        WorkerStage, backend_divergence, pull_shards, push_shards, recv_activation, recv_stage,
        recv_train, report_backend_divergence, resolve_weight_bytes, resolve_weight_uri, run_train,
        send_activation, serve_stage, serve_stage_uri, ship_stage, ship_train, uri_resolver,
    };
    // Heterogeneous multi-backend placement (rlx-runtime::hetero).
    pub use crate::runtime::{DeviceMap, HeteroExecutable};
}

// ── Prelude — single `use rlx::prelude::*;` for the 95% case ────
//
// Includes the graph-building / runtime types, common IR helper
// enums, and autodiff entry points. Skips less-common
// types — those stay reachable via the module re-exports above.

/// Star-import target covering the 95% case:
///
/// ```ignore
/// use rlx::prelude::*;
///
/// // graph building
/// let mut g = Graph::new("ex");
/// let x = g.input("x", Shape::new(&[1, 4], DType::F32));
/// let y = g.mul(x, g.constant(2.0, DType::F32));
/// g.set_outputs(vec![y]);
///
/// // compile + run (auto-pick fastest, or choose any compatible backend)
/// let mut runner = GraphDevices::new(g);
/// let device = runner.fastest(); // or pick from runner.devices()
/// let out = runner.run(device, &[("x", &[1.0; 4])]).unwrap();
///
/// ```
pub mod prelude {
    // Tensor DSL (expression-style graph building) — feature `tensor`.
    #[cfg(feature = "tensor")]
    pub use crate::tensor::{GraphScope, Tensor, ax, graph, graph_with, ix, rg, s, shape, tail};
    // Core graph + runtime
    pub use crate::{
        BackendsManifest, CompiledGraph, DType, Device, DeviceBenchResult, DeviceCandidate,
        DeviceFallbackError, DevicePickStrategy, DevicePolicy, DeviceRouter, Element, Error,
        FlexibleSession, Graph, GraphDevices, GraphExt, GraphModule, GraphStage, Node, NodeId, Op,
        OpKind, ParseDeviceError, Result, Session, Shape, Tick, available_devices,
        benchmark_devices, device_chain_from_env, device_from_env, device_label, device_report,
        devices_for, devices_for_with_policy, fastest_device, fastest_device_for,
        graph_param_names, is_available, parse_device, parse_device_list, resolve_device,
        resolve_device_chain, run_with_fallback, scalar_constant_bytes,
    };
    // IR builder helpers
    pub use crate::ops::{Activation, BinaryOp, CmpOp, MaskKind};
    // Quant metadata
    pub use crate::QuantScheme;
    // Autodiff
    pub use crate::{hvp, jvp, vmap};
    // Optimizer types — useful when configuring passes / precision
    pub use crate::ir::env::{self, RlxEnv, RuntimeOverrides, flag, set, unset, var};
    pub use crate::{CalibrationRecord, Pass, Precision, PrecisionPolicy};

    // FPGA / ASIC SystemVerilog export (feature `fpga`)
    #[cfg(feature = "fpga")]
    pub use crate::{
        ExportOptions, ExportQuantMode, ExportSession, ExportTarget, ExportedArtifacts,
        FpgaExportConfig, GraphIoBind, HwTarget, InputIface, IoConfig, OutputIface, OutputKind,
        PortNames, SidebandSpec, export_graph, export_tinyconv_mnist, tinyconv_mnist_from_cortexm,
    };

    // 3D Gaussian splatting (`rlx-splat` — call `register()` once per process)
    #[cfg(feature = "splat")]
    pub use crate::splat::{
        gaussian_splat_render_common_ir, gaussian_splat_render_decomposed,
        gaussian_splat_render_reference, register,
    };
    #[cfg(feature = "splat")]
    pub use rlx_ir::ops::splat::{
        GaussianSplatInputs, GaussianSplatRenderParams, gaussian_splat_prep_packed_len,
        gaussian_splat_tile_count,
    };
    #[cfg(feature = "splat")]
    pub use rlx_splat::prep_layout::{prep_packed_len, tile_count};
}

/// Register optional custom backends and companion custom-op crates.
///
/// Builtins (CPU, Metal, CUDA, …) register automatically on first
/// [`Session`] use. Call this at process startup when you ship extra
/// backends or custom-op libraries:
///
/// ```ignore
/// rlx::register_backends! {
///     splat => rlx::splat::register,
///     sparse => rlx::sparse::register,
/// }
/// ```
#[macro_export]
macro_rules! register_backends {
    () => {};
    ( $( $name:ident => $register:path ),* $(,)? ) => {
        $( $register(); )*
    };
}
