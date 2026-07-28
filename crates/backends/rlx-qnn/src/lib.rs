// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX Qualcomm QNN (AI Engine Direct) backend for the Hexagon NPU — two
//! surfaces onto the same toolchain, sharing one Rust parity oracle:
//!
//! * **FFI runtime backend** (feature `runtime`) — the live
//!   `Device::Hexagon` path `../rlx-models` dispatches to. Lowers an `rlx-ir`
//!   graph straight to the QNN C API and executes it in-process by `dlopen`ing a
//!   backend library (`libQnnCpu.so` / `libQnnHtp.so`), like `rlx-cuda` /
//!   `rlx-coreml`.
//! * **Codegen / export tool** (default) — emits the `qnn_wrapper_api` C++
//!   (`qnn_model.cpp`) plus a host harness that builds it with
//!   `qnn-model-lib-generator` and runs it under `qnn-net-run`. No `Device`; an
//!   offline artifact and the input to the context-binary perf path.
//!
//! ```text
//!   rlx-ir Graph
//!     ├─ runtime  → QNN C API (dlopen libQnn*.so) → in-process execute    [Device::Hexagon]
//!     └─ codegen  → model → qnn_model.cpp → qnn-model-lib-generator → qnn-net-run [offline]
//! ```
//!
//! ## Why the QNN model-C++ path (and not the converters / Genie)
//!
//! Qualcomm exposes several ingestion surfaces; only one is a *compute backend
//! we can target from our own IR and validate without hardware*:
//!
//! * **Genie / hosted inference** — serves Qualcomm-packaged models, not your
//!   graph. Not a backend.
//! * **`qnn-onnx-converter` / `qnn-pytorch-converter`** — need an ONNX/TF/Torch
//!   frontend; they *consume* a model, they don't let us emit one from rlx-ir.
//!   (Their output is exactly the wrapper-API C++ we emit directly here.)
//! * **QNN AI Engine Direct model C++** — compose a graph with the
//!   `qnn_wrapper_api` surface, build it into a `.so` with
//!   `qnn-model-lib-generator`, and run it with `qnn-net-run` against a backend
//!   library. The **x86-64 CPU reference backend (`libQnnCpu.so`)** and the HTP
//!   emulation run on a commodity Linux host — *no Snapdragon NPU required* — so
//!   this is the one path RLX can target and validate end-to-end.
//!
//! That x86 reference backend is to QNN what the fabric simulator is to
//! [`rlx-cerebras`](../rlx_cerebras/index.html): the no-hardware loop-closer.
//! On real silicon the same `qnn_model.cpp` is built against `libQnnHtp.so` to
//! run on the Hexagon tensor accelerator — that is where the perf lives, and is
//! the milestone after numerical parity. Adreno GPU compute is already covered
//! by [`rlx-vulkan`](../rlx_vulkan/index.html), so this crate aims squarely at
//! the NPU.
//!
//! ## Status
//!
//! * **FFI runtime (feature `runtime`).** A general `rlx-ir` → QNN graph
//!   lowering covering a complete modern-LLM forward pass plus embedding and
//!   vision models: MatMul, element-wise binary (Add/Sub/Mul/Div), activations
//!   (Relu/Gelu/Sigmoid/Tanh/Neg), Reshape, Transpose, Narrow, Concat, Gather
//!   (int32 indices), Softmax, Reduce (Mean/Sum/Max), LayerNorm, RmsNorm, RoPE,
//!   Attention (MHA/GQA; causal / sliding-window / none; optional softcap),
//!   Conv2d, and Quantize/Dequantize (int8 `SFIXED_POINT_8`). Ops that map to
//!   several QNN nodes (RmsNorm, RoPE, Attention, Conv2d, GQA) are decomposed
//!   into intermediate `NATIVE` tensors. Dispatched through
//!   `Session::new(Device::Hexagon)` with static weights and executed in-process
//!   on the CPU reference backend (`libQnnCpu.so`) — 17/17 FFI ops validated
//!   bit-exact in Docker.
//! * **Codegen / export (default).** [`model::Layer::MatMul`],
//!   [`model::Layer::Linear`], [`model::Layer::LinearRelu`],
//!   [`model::Layer::MatMulSoftmax`], [`model::Model::mlp2`], and
//!   [`model::Layer::LinearStatic`] (STATIC weight/bias) with runtime
//!   `APP_WRITE` activations, emitted as `qnn_model.cpp`, compiled by
//!   `qnn-model-lib-generator` against the real QNN headers and run on
//!   `libQnnCpu.so` via `qnn-net-run` with numpy parity (atol/rtol 1e-3).
//!   Feeds the context-binary perf path (FFI save/load + offline
//!   `run_qnn_context.sh` → `qnn-context-binary-generator`).
//! * **Remaining.** HTP/on-device (`libQnnHtp.so` — needs Snapdragon silicon).
//!
//! Both paths are reproducible in Docker via `crates/rlx-qnn/docker/`; the FFI
//! runtime design + milestones live in `docs/ffi-runtime-backend.md`.
//!
//! ## What's here
//!
//! * `cpp`       — pure-Rust C/C++ source writer (buffer + indent), the analog
//!                 of `rlx_cerebras::csl::Csl` / `rlx_fpga::verilog::V`.
//! * `model`     — lightweight [`model::Layer`] / [`model::Model`] description
//!                 plus [`model::Model::from_graph`] (rlx-ir → model).
//! * `codegen`   — emit the `qnn_model.cpp` / `verify.py` / `run_qnn.sh`
//!                 artifacts.
//! * `reference` — Rust forward pass; the parity oracle for the emitted model.
//! * `runtime`   — (feature `runtime`) in-process FFI execution on a QNN
//!                 backend library — the `Device::Hexagon` path that
//!                 `../rlx-models` consumes.

pub mod codegen;
pub mod cpp;
pub mod model;
pub mod reference;
pub mod supported_ops;

/// In-process FFI runtime backend (the `Device::Hexagon` path). Off by
/// default; needs `QNN_SDK_ROOT` at build time. See [`runtime`].
#[cfg(feature = "runtime")]
pub mod runtime;

/// Host GGUF / MLX dequant helpers (`DequantMatMul` → f32 MatMul).
/// Enabled by `host-dequant` (no SDK) or `runtime`.
#[cfg(feature = "host-dequant")]
pub mod dequant;

/// Host INT8 `QMatMul` (no f32 weight bake) used by the FFI runtime.
#[cfg(feature = "runtime")]
pub mod qmatmul;

pub use codegen::{Artifact, collect_artifacts, emit_model};
pub use model::{Layer, Model};
pub use supported_ops::SUPPORTED_OPS;
