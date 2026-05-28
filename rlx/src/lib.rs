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
//! g.set_outputs(vec![y]);
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
//! | `rlx::gguf`     | `rlx-gguf`      | GGUF parser + dequant *(feature `gguf`)*                                        |
//! | `rlx::bench`    | `rlx-bench`     | benchmark harness *(feature `bench`)*                                           |
//! | `rlx::sparse`   | `rlx-sparse`    | downstream: sparse linalg *(feature `sparse`)*                                  |
//! | `rlx::splat`    | `rlx-splat`     | 3D Gaussian splatting *(feature `splat`)* — `register()`, decomposed IR ops      |
//! | `rlx::linalg`   | `rlx-linalg`    | downstream: dense linalg via LAPACK *(feature `linalg`)*                        |
//! | `rlx::cortexm`  | `rlx-cortexm`   | INT8 ARMv7E-M kernels *(feature `cortexm`)* — no `Backend` impl, kernels only   |
//! | `rlx::fpga`     | `rlx-fpga`      | IR → SystemVerilog datapath synthesis *(feature `fpga`)* — no `Backend` impl    |
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
/// GGUF v1 / v2 / v3 parser + dequant.
/// See [`rlx-gguf`](https://crates.io/crates/rlx-gguf).
pub use rlx_gguf as gguf;

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

#[cfg(feature = "cortexm")]
/// `no_std` ARMv7E-M INT8 kernels (Cortex-M4F / M7). Doesn't
/// implement `Backend` — call the kernels (`dense`, `conv2d`,
/// `maxpool`, `relu`, `argmax`) directly.
/// See [`rlx-cortexm`](https://crates.io/crates/rlx-cortexm).
pub use rlx_cortexm as cortexm;

#[cfg(feature = "fpga")]
/// IR → SystemVerilog datapath synthesis. Doesn't implement
/// `Backend` — synth + P&R takes minutes; the entry point is
/// `rlx::fpga::codegen::emit_model`.
/// See [`rlx-fpga`](https://crates.io/crates/rlx-fpga).
pub use rlx_fpga as fpga;

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
pub use rlx_ir::quant::QuantScheme;
pub use rlx_ir::{
    DType, Element, FusionPolicy, Graph, GraphModule, GraphStage, HirModule, HirOp, LirModule,
    MirModule, Node, NodeId, Op, OpKind, Shape, Tick,
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
pub use rlx_runtime::{CompiledGraph, Session};

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
/// use rlx::{Graph, Shape, DType};
/// use rlx::ops::{Activation, BinaryOp};
///
/// let mut g = Graph::new("ex");
/// let x = g.input("x", Shape::new(&[4], DType::F32));
/// let y = g.input("y", Shape::new(&[4], DType::F32));
/// let s = g.binary(BinaryOp::Add, x, y, Shape::new(&[4], DType::F32));
/// let r = g.activation(Activation::Silu, s, Shape::new(&[4], DType::F32));
/// g.set_outputs(vec![r]);
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
///
/// // compile + run
/// let mut compiled = Session::new(Device::Cpu).compile(g);
/// let out = compiled.run(&[("x", &[1.0; 4])]);
///
/// ```
pub mod prelude {
    // Core graph + runtime
    pub use crate::{
        CompiledGraph, DType, Device, Element, Error, Graph, GraphModule, GraphStage, Node, NodeId,
        Op, OpKind, Result, Session, Shape, Tick,
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
