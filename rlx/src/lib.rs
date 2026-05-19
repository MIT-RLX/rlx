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
//! ### 2. Run a model by name (Qwen3, SAM 1/2/3, DINOv2)
//!
//! Requires the `models` feature.
//!
//! ```ignore
//! use rlx::prelude::*;
//!
//! let mut runner = Qwen3Runner::builder()
//!     .weights("Qwen3-0.6B-Q4_K_M.gguf")   // safetensors or gguf
//!     .device(Device::Metal)
//!     .max_seq(128)
//!     .build()?;
//! runner.generate(&prompt_ids, 32, |tok| print!(" {tok}"))?;
//! ```
//!
//! ### 3. Plug your own runner into the CLI / dispatch surface
//!
//! ```ignore
//! use rlx::prelude::*;
//!
//! struct WhisperRunner;
//! impl ModelRunner for WhisperRunner {
//!     fn name(&self) -> &'static str { "whisper" }
//!     fn description(&self) -> &'static str { "OpenAI Whisper" }
//!     fn run(&self, args: &[String]) -> Result<()> { /* … */ Ok(()) }
//! }
//!
//! fn main() -> Result<()> {
//!     register_runner(Box::new(WhisperRunner));
//!     dispatch(&std::env::args().skip(1).collect::<Vec<_>>())
//! }
//! ```
//!
//! ## Module map
//!
//! Every workspace crate is reachable as a module on `rlx`:
//!
//! | path            | crate           | what                                                                            |
//! |-----------------|-----------------|---------------------------------------------------------------------------------|
//! | `rlx::ir`       | `rlx-ir`        | IR types, ops, graph builder                                                    |
//! | `rlx::opt`      | `rlx-opt`       | passes, autodiff, vmap                                                          |
//! | `rlx::driver`   | `rlx-driver`    | `Device` enum, registries                                                       |
//! | `rlx::runtime`  | `rlx-runtime`   | `Session`, `CompiledGraph`                                                      |
//! | `rlx::macros`   | `rlx-macros`    | `#[rlx_model]` proc macro                                                       |
//! | `rlx::models`   | `rlx-models`    | Qwen3 / SAM / DINOv2 / BERT / Nomic builders *(feature `models`)*               |
//! | `rlx::gguf`     | `rlx-gguf`      | GGUF parser + dequant *(feature `gguf`)*                                        |
//! | `rlx::bench`    | `rlx-bench`     | benchmark harness *(feature `bench`)*                                           |
//! | `rlx::sparse`   | `rlx-sparse`    | downstream: sparse linalg *(feature `sparse`)*                                  |
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
//! | [`rlx::weights`]     | `WeightLoader`, `WeightMap`, `GgufLoader`, HF↔GGUF name mappers *(`models`)*  |
//! | [`rlx::run`]         | `Qwen3Runner`, `SamRunner`, `DinoV2Runner` + dispatch / registry *(`models`)* |
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
//! | `all-cpu`         | `cpu` + `models` + `gguf` + `linalg`        |
//!
//! `mlx` and `rocm` aren't in any aggregate because their crates
//! aren't on crates.io (vendor-bundled submodule / workspace-
//! relative kernel sources). To opt in, depend on the workspace via
//! git and add the feature explicitly:
//!
//! ```toml
//! rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["apple-silicon", "mlx"] }
//! ```

#![doc(html_root_url = "https://docs.rs/rlx/0.2.0")]

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

#[cfg(feature = "models")]
/// Qwen3 / SAM / DINOv2 / BERT / Nomic / vision builders.
/// See [`rlx-models`](https://crates.io/crates/rlx-models).
pub use rlx_models as models;

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
pub use rlx_ir::{DType, Element, Graph, Node, NodeId, Op, OpKind, Shape, Tick};
pub use rlx_opt::{CalibrationRecord, Pass, Precision, PrecisionPolicy, hvp, jvp, vmap};
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
    pub use rlx_ir::op::{
        Activation, BinaryOp, ChainOperand, ChainStep, CmpOp, MaskKind,
    };
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

/// Weight loaders — pluggable `WeightLoader` trait plus the two
/// built-in adapters (`WeightMap` for safetensors, `GgufLoader`
/// for `.gguf`), the HF↔GGUF tensor-name mapper, and the MTP-head
/// detection helper. Available only with the `models` feature.
///
/// ```ignore
/// use rlx::weights::{GgufLoader, hf_to_gguf_name};
///
/// let mut wm = GgufLoader::from_file("model.gguf")?;
/// let (bytes, shape) = wm.take("model.embed_tokens.weight")?;
/// assert_eq!(hf_to_gguf_name("lm_head.weight").as_deref(), Some("output.weight"));
/// ```
#[cfg(feature = "models")]
pub mod weights {
    pub use rlx_models::weight_loader::{
        GgufLoader, WeightLoader, gguf_to_hf_name, hf_to_gguf_name, is_mtp_weight,
        load_from_path,
    };
    pub use rlx_models::weight_map::WeightMap;
}

// ── High-level model runners + dispatch registry ────────────────
//
// The `rlx::run` namespace re-exports the builder-style entry points
// from `rlx_models::run` plus the `ModelRunner` plug-in registry.
// Lets a user do:
//
//   use rlx::run::{Qwen3Runner, Qwen3Precision};
//   let r = Qwen3Runner::builder()
//       .weights("model.gguf")
//       .device(rlx::Device::Metal)
//       .precision(Qwen3Precision::F16LmHead)
//       .build()?;
//   r.generate(&prompt_ids, 32, |tok| print!(" {tok}"))?;
//
// `Precision` is renamed to `Qwen3Precision` here to avoid clashing
// with `rlx::Precision` (the autodiff precision policy). The other
// runner types — `Qwen3Runner` etc. — keep their original names.
//
// Gated behind the `models` cargo feature (default-off in the
// minimum prelude build; enabled by `default-features`).
#[cfg(feature = "models")]
pub mod run {
    pub use rlx_models::run::{
        ConfigSource, DinoV2Output, DinoV2Runner, DinoV2RunnerBuilder, DinoV2Variant,
        ModelRunner, Precision as Qwen3Precision, Qwen3Runner, Qwen3RunnerBuilder, SamArch,
        SamPredictionAny, SamRunner, SamRunnerBuilder, WeightFormat, debug_resolve_name,
        dispatch, dispatch_help, list_mtp_keys, open_loader, register_runner,
        registered_runners, run_registered,
    };
}

// ── Prelude — single `use rlx::prelude::*;` for the 95% case ────
//
// Includes the graph-building / runtime types, common IR helper
// enums, autodiff entry points, and (when the `models` feature is
// on) every runner + the plug-in registry. Skips less-common
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
/// // (with models feature) high-level runner
/// let mut runner = Qwen3Runner::builder()
///     .weights("model.gguf")
///     .device(Device::Metal)
///     .build()?;
/// ```
pub mod prelude {
    // Core graph + runtime
    pub use crate::{
        CompiledGraph, DType, Device, Element, Error, Graph, Node, NodeId, Op, OpKind, Result,
        Session, Shape, Tick,
    };
    // IR builder helpers
    pub use crate::ops::{Activation, BinaryOp, CmpOp, MaskKind};
    // Quant metadata
    pub use crate::QuantScheme;
    // Autodiff
    pub use crate::{hvp, jvp, vmap};
    // Optimizer types — useful when configuring passes / precision
    pub use crate::{CalibrationRecord, Pass, Precision, PrecisionPolicy};

    // Model runners + plug-in registry
    #[cfg(feature = "models")]
    pub use crate::run::{
        DinoV2Output, DinoV2Runner, DinoV2Variant, ModelRunner, Qwen3Precision, Qwen3Runner,
        SamArch, SamPredictionAny, SamRunner, WeightFormat, dispatch, dispatch_help,
        register_runner, registered_runners,
    };

    // Weight loaders
    #[cfg(feature = "models")]
    pub use crate::weights::{GgufLoader, WeightLoader, WeightMap, hf_to_gguf_name};
}
