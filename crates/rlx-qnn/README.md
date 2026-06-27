# rlx-qnn

Qualcomm **QNN (AI Engine Direct)** backend for RLX — runs an `rlx-ir` graph on
the Qualcomm Hexagon NPU. Two surfaces onto the same toolchain, sharing one Rust
parity oracle:

- **FFI runtime backend** (feature `runtime`) — a real `Device::Hexagon`
  `rlx-runtime` backend. Lowers a graph straight to the QNN C API and executes
  it **in-process** by `dlopen`ing a backend library (`libQnnCpu.so` /
  `libQnnHtp.so`), exactly like `rlx-cuda` / `rlx-coreml`. This is the live
  inference path `../rlx-models` dispatches to.
- **Codegen / export tool** (default) — emits the `qnn_wrapper_api` C++
  (`qnn_model.cpp`) plus a host harness that builds it with
  `qnn-model-lib-generator` and runs it under `qnn-net-run`. An offline artifact
  and the input to the context-binary perf path; no `Device`.

Both validate **without a Snapdragon device**: QNN's x86-64 CPU reference
backend (`libQnnCpu.so`) runs on a commodity Linux host — the QNN analog of
`rlx-cerebras`'s fabric simulator. On real silicon the same graph runs against
`libQnnHtp.so` on the Hexagon tensor accelerator, where the perf lives. Adreno
GPU compute is already covered by `rlx-vulkan`, so this crate targets the NPU.

## FFI runtime backend (`Device::Hexagon`)

Dispatched through the normal `Session` API — the same path every other backend
uses (Metal, CUDA, ANE):

```rust
use rlx_runtime::{Device, Session};

// rlx-runtime built with `qnn`; rlx-qnn with `runtime`.
// dlopens libQnnCpu.so / libQnnHtp.so (resolved from QNN_SDK_ROOT or
// RLX_QNN_BACKEND_LIB), builds the graph via the QNN C API, executes in-process.
let mut hex = Session::new(Device::Hexagon).compile(graph);
let out = hex.run(&[("in0", &in0), ("in1", &in1)]);
```

A general `rlx-ir` → QNN graph lowering. Supported ops — enough for a complete
modern-LLM forward pass (embedding → blocks → norm → lm_head) plus embedding and
vision models:

| Group | Ops |
| --- | --- |
| Linear / element-wise | `MatMul`, `Binary` (Add/Sub/Mul/Div), `Neg` |
| Activations | `Relu`, `Gelu`, `Sigmoid`, `Tanh` |
| Shape | `Reshape`, `Transpose`, `Narrow` (→ StridedSlice), `Concat` |
| Indexing | `Gather` (embedding lookup, int32 indices) |
| Normalization | `LayerNorm`, `RmsNorm` |
| Attention | `Softmax`, `RoPE` (NeoX), `Attention` — MHA/GQA, causal / sliding-window / none, optional logit softcap |
| Reduction | `Reduce` (Mean/Sum/Max — e.g. mean-pool for BERT/nomic) |
| Vision | `Conv2d` (NCHW↔NHWC, stride/pad/dilation/group) |
| Quantization | `Quantize` / `Dequantize` (int8 `SFIXED_POINT_8`, scale/offset) |

Ops that map to several QNN nodes — `RmsNorm`, `RoPE` (7 nodes), `Attention`
(~13 nodes), `Conv2d`, GQA — are decomposed into the underlying QNN primitives
with intermediate `NATIVE` tensors. Static weights (`Param` / `Constant`) are
staged as `QNN_TENSOR_TYPE_STATIC`. **17/17 FFI ops validated bit-exact** against
`libQnnCpu.so` in Docker.

Layout: a thin C shim (`runtime/rlx_qnn_shim.{c,h}`), compiled by `build.rs`
against the real SDK headers, `dlopen`s the backend lib and drives the
`QnnInterface` vtable; `src/runtime.rs` (`QnnExecutable`) builds the tensor/node
plan from `rlx-ir` and binds it. The build is **driverless** — `dlopen` at
runtime, nothing linked — so `cargo build -p rlx-qnn --features runtime`
compiles on any host with no SDK present. Design + milestones:
[`docs/ffi-runtime-backend.md`](docs/ffi-runtime-backend.md).

Remaining: quantized *matmul* deployment (per-channel/block scales, int4) and
HTP/on-device + the context-binary perf path (`libQnnHtp.so`, needs Snapdragon
silicon — where the perf, the north star, lives).

## Codegen / export tool

For the offline `qnn-net-run` path (and the eventual context-binary perf
artifact), the crate also emits QNN model C++ directly from `rlx-ir`.

### Why the QNN model-C++ path

Qualcomm exposes several ingestion surfaces; only one lets RLX emit a graph from
its own IR *and* validate it without a Snapdragon device:

| Surface | What it is | Usable as an RLX backend? |
| --- | --- | --- |
| Genie / hosted inference | Serves Qualcomm-packaged models | No — not your graph |
| `qnn-onnx`/`-pytorch`-converter | Consume ONNX/TF/Torch → model C++ | No — needs a frontend; we'd be on the wrong side of it |
| **QNN model C++ (`qnn_wrapper_api`)** | Compose a graph, build a `.so`, run with `qnn-net-run` | **Yes — x86 `libQnnCpu.so` runs without hardware** |

The converters' *output* is exactly the wrapper-API C++ this crate emits
directly. On real silicon the same `qnn_model.cpp` builds against `libQnnHtp.so`.

### Pipeline

```
rlx-ir Graph
  → rlx-qnn::model    (recognize the supported subgraph, read shapes)
  → rlx-qnn::codegen  (emit qnn_model.cpp + verify.py + run_qnn.sh)
  → qnn-model-lib-generator + qnn-net-run   (QNN SDK, Linux host)
```

### Status — milestone 1

A single rank-2 `MatMul`: one node, two runtime activation inputs (`in0[M,K]`,
`in1[K,N]`), one output (`out[M,N]`), f32. Validated end-to-end —
`qnn-model-lib-generator` compiles the emitted `qnn_model.cpp` against the real
QNN headers and `qnn-net-run` executes it on `libQnnCpu.so` with numpy parity
(atol/rtol 1e-3), reproducible in Docker. It is also the graph-construction
reference and numerical oracle the FFI runtime reuses; multi-op codegen lands
with the context-binary perf path.

### Emit

```sh
cargo run -p rlx-qnn --bin rlx-qnn-emit -- 32 64 32 ./qnn-out
# then, on a Linux host with the QNN SDK (QNN_SDK_ROOT set):
cd qnn-out && bash run_qnn.sh   # build + qnn-net-run → "SUCCESS!"
```

## Validation (Docker)

Both paths reproduce on a commodity host via [`docker/`](docker/) — no QNN SDK
install, no Snapdragon device. The harness pulls the public Qualcomm AI Runtime
Community SDK, builds against the real headers, and runs on `libQnnCpu.so` under
x86_64 (Rosetta/emulation on Apple Silicon). See
[`docker/README.md`](docker/README.md) and the `qnn-*` recipes in the workspace
`Justfile`.
