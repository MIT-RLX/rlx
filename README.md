# RLX

![status](https://img.shields.io/badge/status-0.2.14-blue)
![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)
![rust](https://img.shields.io/badge/rust-edition%202024-orange)
[![repo](https://img.shields.io/badge/github-MIT--RLX%2Frlx-black)](https://github.com/MIT-RLX/rlx)

RLX is an ML compiler and runtime for neural-network inference **and**
training. At its core is a small, serializable tensor IR with JAX-shaped
autodiff and transforms (`grad`, `jvp`, `hvp`, `vmap`); a compile pipeline
(HIR → MIR → LIR) legalizes, fuses, and memory-plans the graph, then runs
it through backend-specific kernels across CPU, Apple Silicon (Metal / MLX
/ ANE), NVIDIA CUDA, AMD (ROCm + XDNA NPU), Intel oneAPI, Google TPU,
Qualcomm Hexagon, cross-platform GPU (wgpu / Vulkan / WebGL / WASM),
microcontrollers (Cortex-M), FPGA, and the Cerebras WSE. It imports models
from ONNX, PyTorch (`torch.export`), and GGUF / safetensors; does
quantization (GGUF K/IQ/TQ/MX, INT8/INT4, QAT) and multi-node distributed
execution; and keeps the core model-agnostic — model crates live in
sibling repos.

## Table of Contents

- [Why another one](#why-another-one)
- [Install](#install)
- [Quickstart](#quickstart)
- [Examples](#examples)
  - [Simple example](#simple-example)
  - [Advanced example](#advanced-example)
  - [Training example](#training-example)
  - [Distributed example](#distributed-example)
  - [Extending RLX](#extending-rlx)
- [Import a PyTorch model](#import-a-pytorch-model)
- [Running models](#running-models)
- [Workspace layout](#workspace-layout)
- [Building from source](#building-from-source)
- [Documentation](#documentation)
- [Kernel dispatch and transparency](#kernel-dispatch-and-transparency)
- [Training](#training)
- [Status by area](#status-by-area)
- [Citing RLX](#citing-rlx)
- [Contributing](#contributing)

## Why another one

Most ML stacks pick a side: either a graph compiler (XLA, TVM, MLIR) or
a kernel runtime (cuDNN, MPS, MLX). RLX is both, end-to-end, in one
language, with a vocabulary modelled on `jax.lax`. The IR knows about
`Op::Scan`, `Op::DenseSolve`, `Op::FakeQuantize`, attention with
`MaskKind`, and an `Op::Custom` / `Op::CustomFn` extension surface;
the optimizer knows about fusion, AMP precision policy, autodiff in
both directions, vmap, broadcast legalization, and PTQ insertion; the
runtime knows about backend dispatch, compile caching, and
heterogeneous cost-based device selection.

**Non-goals.** RLX is inference-first and edge-first: the aim is the fastest,
most precise path for *each* backend × hardware pair when running — and
fine-tuning / training-in-the-loop — transformer- and vision-scale models,
not a from-scratch trainer for frontier-scale pretraining. Model definitions
live downstream in [`rlx-models`](#running-models); core stays
application-agnostic.

## Install

The `rlx` prelude crate is the recommended entry point — it pulls in
the IR, optimizer, runtime, and re-exports the common types:

```toml
[dependencies]
rlx = { version = "0.2", features = ["cpu"] }
```

For Apple Silicon GPU acceleration:

```toml
rlx = { version = "0.2", features = ["cpu", "metal"] }
```

> **`mlx` and `rocm` build from the workspace git tree** — they resolve
> workspace-relative submodule / kernel-source paths that a crates.io
> package can't carry. Depend on the repo directly for those features:
>
> ```toml
> rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["mlx"] }
> ```

### Backend features

| feature             | backend                              | platform                  |
|---------------------|--------------------------------------|---------------------------|
| `cpu` *(default)*   | NEON / AVX + Accelerate / OpenBLAS   | every host                |
| `metal`             | Metal Performance Shaders + MSL      | macOS (Apple Silicon)     |
| `mlx`               | Apple MLX (vendored)                 | macOS (Apple Silicon)     |
| `coreml`            | CoreML / Neural Engine (`Device::Ane`)| macOS / iOS (Apple)      |
| `gpu`               | wgpu (Vulkan / DX12 / WebGPU / Metal)| cross-platform            |
| `vulkan`            | native Vulkan compute (`ash` + SPIR-V)| Linux / Windows (+ MoltenVK)|
| `oneapi`            | Intel oneAPI Level Zero (SPIR-V)     | Linux + Intel GPU         |
| `cuda`              | cuBLAS / cuDNN / NVRTC               | Linux / Windows + NVIDIA  |
| `rocm`              | hipBLAS / MIOpen                     | Linux + AMD               |
| `tpu`               | libtpu PJRT plugin                   | Linux + GCP TPU           |
| `blas-accelerate`   | macOS Accelerate                     | macOS                     |
| `blas-mkl`          | Intel MKL                            | Intel / AMD CPUs          |
| `blas-openblas`     | OpenBLAS                             | cross-platform CPU        |

Convenience aggregates roll up the common setup for a target:
`apple-silicon` (`cpu` + `metal` + `blas-accelerate`), `nvidia` (`cpu` +
`cuda`), `edge` (`cpu` + `cortexm` + `fpga`), and `all-cpu` (`cpu` + `gguf`
+ `linalg`). Qualcomm Hexagon / QNN (`Device::Hexagon`) ships via
`rlx-runtime`'s `qnn` feature — see [Specialty crates](#specialty-crates).

### Companion crate features

Off by default; enable per workload:

| feature        | what                                                     |
|----------------|----------------------------------------------------------|
| `tensor` *(default)* | NumPy-style lazy `Tensor` DSL (`rlx::tensor`)      |
| `gguf`         | GGUF v1 / v2 / v3 parser + dequant                       |
| `gguf-convert` | safetensors / ONNX → GGUF with per-tensor quant          |
| `onnx`         | run `.onnx` models on RLX backends via ONNX Runtime EPs  |
| `optim`        | training-step optimizers (Adam/Lion/Muon/…, `rlx::optim`)|
| `distributed`  | in-graph collectives + `rlx::distributed` front door     |
| `umap`         | UMAP dimensionality reduction (`rlx::umap`)              |
| `bench`        | uniform benchmark harness                                |
| `sparse`       | sparse linear algebra (custom-op scaffold)               |
| `linalg`       | dense linalg via LAPACK (custom-op scaffold)             |

### Specialty crates

The `Backend` model doesn't fit microcontrollers or hardware synthesis.
For those, depend on the standalone crates directly — they're not
exposed through the prelude:

- `rlx-cortexm` — `no_std` ARMv7E-M INT8 kernels.
- `rlx-fpga` — IR → SystemVerilog → bitstream (first-class export via
  `rlx_runtime::export` / `pyrlx.export_fpga` under the `fpga` feature).
- `rlx-qnn` — Qualcomm Hexagon / QNN codegen + FFI runtime (`Device::Hexagon`;
  enable `rlx-runtime`'s `qnn` feature, not on the umbrella prelude).

## Quickstart

A single `use rlx::prelude::*;` covers the common surface: graph
types, `Session`, `Device`, ops + activations, and `Result`.

```rust
use rlx::prelude::*;

let mut g = Graph::new("hello");
let x = g.input("x", Shape::new(&[1, 4], DType::F32));
let w = g.param("w", Shape::new(&[4, 2], DType::F32));
let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
g.set_outputs(vec![y]);

let mut compiled = Session::new(Device::Cpu).compile(g);
compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
let out = compiled.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
assert_eq!(out[0], vec![4.0, 6.0]); // `run` returns one Vec<f32> per graph output
```

Prefer NumPy-style expressions? The `rlx::tensor` DSL ([`rlx-tensor`](crates/core/rlx-tensor/README.md))
builds the same IR with operator-overloaded, lazy `Tensor` handles —
`(&a + &b).relu()` traces instead of executing, then fuses + memory-plans
across any backend when you call `.to_vec()`:

```rust
use rlx_tensor::Tensor; // crate `rlx-tensor`, feature `eval`

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
let c = (&a + &Tensor::ones([3])).relu();
assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0]); // auto-picks the fastest backend
```

Or declare the graph in a compact little language with `rlx!` (feature `tensor`,
on by default) — `@` is matmul with NumPy precedence, shapes/wiring/outputs are
inferred, and it lowers to the same IR:

```rust
use rlx::rlx;

let g = rlx! {
    graph "mlp";
    input x: [?, 784];              // `?` = dynamic batch
    param w1: [784, 256];  param b1: [256];
    param w2: [256, 10];   param b2: [10];

    let h = gelu(x @ w1 + b1);
    let y = h @ w2 + b2;
    out y;                          // → rlx_ir::Graph
};
```

Domain-specific namespaces if you want narrower star-imports:
`rlx::ops::*` (IR helper enums), `rlx::quant::*`, `rlx::autodiff::*`.
Or the full per-crate surface
via `rlx::ir::…` / `rlx::opt::…` / `rlx::runtime::…` etc. — every
workspace crate is reachable as a module on `rlx`. Or depend on each
crate directly (`rlx-ir`, `rlx-opt`, `rlx-runtime`, …) for the
smallest possible dep tree.

## Examples

Each example below is a minimal, faithful sketch; runnable end-to-end versions
live under each crate's `examples/` (e.g. `cargo run -p rlx-runtime --example
int8_qat_train --features cpu`).

### Simple example

Build a 2-layer MLP, compile it for the CPU, and run a forward pass. `run`
returns one `Vec<f32>` per graph output.

```rust
use rlx::prelude::*;
use rlx_ir::op::Activation;

let mut g = Graph::new("mlp");
let x  = g.input("x",  Shape::new(&[1, 4], DType::F32));
let w1 = g.param("w1", Shape::new(&[4, 8], DType::F32));
let w2 = g.param("w2", Shape::new(&[8, 2], DType::F32));

let h = g.matmul(x, w1, Shape::new(&[1, 8], DType::F32));
let h = g.activation(Activation::Relu, h, Shape::new(&[1, 8], DType::F32));
let y = g.matmul(h, w2, Shape::new(&[1, 2], DType::F32));
g.set_outputs(vec![y]);

let mut net = Session::new(Device::Cpu).compile(g);
net.set_param("w1", &w1_weights); // &[f32], len 4*8
net.set_param("w2", &w2_weights); // &[f32], len 8*2
let out = net.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
println!("{:?}", out[0]);          // the 2 outputs
```

### Advanced example

Differentiate a graph (reverse mode), let RLX pick the fastest backend actually
present on the machine, and print what the compiler did per op.

```rust
use rlx::prelude::*;
use rlx_autodiff::grad;
use rlx_runtime::fastest_device;

// `forward` is a graph with input `x` (node `x_id`); build ∂output/∂x:
let grad_graph = grad(&forward, &[x_id]);

let device = fastest_device();                 // Metal / CUDA / wgpu / … else CPU
std::env::set_var("RLX_DISPATCH_REPORT", "1"); // native vs common-ir per OpKind
let mut compiled = Session::new(device).compile(grad_graph);
let grads = compiled.run(&[("x", &x_values)]);
```

Mixed precision, compile caching, and per-op dispatch overrides all go through
[`CompileOptions`](crates/core/rlx-runtime/src/options.rs) — see
[Kernel dispatch and transparency](#kernel-dispatch-and-transparency).

### Training example

Reverse-mode autodiff plus an optimizer from
[`rlx-optim`](crates/core/rlx-optim). `grad_with_loss` builds a backward graph
whose outputs are `[loss, ∂loss/∂param…]` and which takes a unit loss-seed input
named `d_output`.

```rust
use rlx::prelude::*;
use rlx_autodiff::grad_with_loss;
use rlx_optim::{Adam, Optimizer};

// `forward`: a graph whose FIRST output is a scalar loss, with trainable param
// `w` (node `w_id`). See examples/int8_qat_train.rs for a full forward graph.
let backward = grad_with_loss(&forward, &[w_id]);  // outputs: [loss, dW]
let mut compiled = Session::new(Device::Cpu).compile(backward);

let mut opt = Adam::new(1e-3);
let mut w = init_weights();                         // Vec<f32>
for step in 0..200 {
    let out = compiled.run(&[
        ("x", &x), ("target", &target),
        ("w", &w),                                 // feed the current params in
        ("d_output", &[1.0]),                      // ∂loss/∂loss = 1
    ]);
    let (loss, grad) = (out[0][0], &out[1]);
    opt.step("w", &[w.len()], &mut w, grad);       // in-place Adam update
    opt.end_iteration();
    if step % 20 == 0 { println!("step {step}: loss {loss:.6}"); }
}
```

### Distributed example

Data-parallel training across machines, gradients reduced on the host. With
`TOPOLOGY=iroh` each rank is one process and relays + pkarr discovery traverse
NAT and firewalls — no `ip:port` wiring. Run one process per box with a shared
seed:

```bash
# rank 0 — e.g. a Mac on Metal
RANK=0 WORLD=2 TOPOLOGY=iroh RLX_IROH_SEED=cafe1234 RLX_DEVICE=metal \
  cargo run -p rlx-vision-bench --features iroh -- --model cnn --epochs 5

# rank 1 — e.g. a Linux CUDA box, even behind an inbound-blocking firewall
RANK=1 WORLD=2 TOPOLOGY=iroh RLX_IROH_SEED=cafe1234 RLX_DEVICE=cuda \
  cargo run -p rlx-vision-bench --features iroh -- --model cnn --epochs 5
```

In-graph collectives (`collective.all_reduce`, `moe_dispatch` /
`moe_combine`, …) come from the `distributed` feature (`rlx::distributed`);
WideEP expert-parallel MoE + EPLB live in `rlx-collectives`, with an optional
CUDA NCCL path (`rlx-cuda --features nccl`) for device-resident
`all_reduce` / `all_to_all`. The ship-graph fabric ships a `TrainSpec` to
model-agnostic workers that train on each node's fastest hardware. See
[`docs/distributed.md`](docs/distributed.md) and
[`docs/iroh-transport.md`](docs/iroh-transport.md).

### Extending RLX

Add a new architecture **block** without editing core: implement
[`LayerStage`](crates/core/rlx-flow) — compose primitives through `FlowCtx`'s
builders and drop it into any flow with `ModelFlow::layer_stage`. It lowers to
ordinary primitives, so it stays visible to fusion and hits the fast path on
every backend.

```rust
use rlx_extend::prelude::*;

struct GatedMlp { gate: String, up: String, down: String }

impl LayerStage for GatedMlp {
    fn name(&self) -> &str { "gated_mlp" }

    fn emit_layer(&self, ctx: &mut FlowCtx<'_>, x: FlowValue)
        -> anyhow::Result<(FlowValue, StageArtifacts)>
    {
        let g = ctx.linear(&x, &self.gate, false)?;  // curated primitive builders
        let g = ctx.silu(&g);
        let u = ctx.linear(&x, &self.up, false)?;
        let h = ctx.mul(&g, &u);
        let out = ctx.linear(&h, &self.down, false)?;
        Ok((out.clone(), StageArtifacts::hidden_only(out.shape().clone())))
    }
}

let flow = ModelFlow::new("my_model").input("x", shape).layer_stage(GatedMlp { /* … */ });
```

For a new **op** (a custom primitive with its own shape / autodiff / lowering
rules), implement [`OpExtension`](crates/core/rlx-ir) + `register_op` instead;
`cargo rlx new-op` / `new-model` scaffold either seam. Full guide with all four
seams: [`docs/extending.md`](docs/extending.md) and
[`rlx-extend`](crates/core/rlx-extend/README.md).

## Import a PyTorch model

Already have a model in PyTorch? Convert it straight to RLX — no ONNX in the
loop. [`rlx-torch-import`](crates/io/rlx-torch-import/README.md) runs
`torch.export`, maps each ATen op directly onto RLX ops, and emits a runnable
**bundle** (serialized HIR graph + weights) and/or a **standalone RLX crate**,
then verifies numeric parity against PyTorch (cosine + max abs err).

**Python** — one call (`pip install pyrlx`):

```python
import pyrlx, torch
from torchvision.models import resnet18

model = resnet18(weights="DEFAULT").eval()
example = (torch.randn(1, 3, 224, 224),)

pyrlx.from_torch(model, example, out_dir="out/", verify=True)
# → out/bundle/            runnable HIR graph + weights   (parity: cosine 1.000000)
#   out/rlx-resnet18/      standalone RLX crate you can ship / edit
```

Options: `emit=("bundle","crate")`, `emit_style="graph"|"tensor"|"flow"` (the
crate's authoring layer), `decomposition="aten"|"high"|"core"`, `verify=True`.

**CLI** — a `model.py` exposing `model` + `example_inputs`:

```bash
rlx-torch-import model.py -o out/ --emit bundle,crate --verify --device cuda
```

**A second example** — a HuggingFace LLM exports the same way (weight names stay
HF-canonical, so RLX's loaders consume them directly):

```python
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("…").eval()
pyrlx.from_torch(model, (torch.randint(0, 32000, (1, 16)),), out_dir="llama/", verify=True)
```

Verified end-to-end at **cosine 1.000000** on both **CPU and CUDA** (NVIDIA GPU)
across MLP, encoder-decoder, CNN, MNIST, LLaMA (rotary + GQA + causal mask), DINO
/ ViT, and the FLUX diffusion transformer (MMDiT). Runnable, per-model, per-form
examples live in
[`crates/bindings/pyrlx/examples/`](crates/bindings/pyrlx/examples/README.md)
(`python mlp.py`, `python llama.py`, …). Training is out of scope — `torch.export`
captures the inference graph; import the *forward* and use RLX's own autodiff +
[`rlx-optim`](crates/core/rlx-optim) for on-device training.

## Running models

RLX core is model-agnostic — the graph builders, ops, and backends live here;
concrete model definitions live in the sibling **[`rlx-models`](https://github.com/MIT-RLX/rlx-models)**
workspace (Qwen3 / Qwen3.5, BERT / Nomic embeddings, SAM, …), which pins a
published `rlx`. That repo ships turnkey builders (`Qwen3Runner`, `SamRunner`, …)
and the `rlx-run` CLI:

```bash
# from the rlx-models workspace
rlx-run qwen3   --weights qwen3-0.6b.gguf --prompt "hello"   # GGUF or safetensors
rlx-run qwen35  --weights qwen3.5.gguf    --prompt "hello"
rlx-run inspect --weights model.gguf                         # tensors + metadata
```

Quantized weights are first-class from Python too — `pyrlx` reads, writes, and
converts GGUF (every llama.cpp scheme) and quantizes safetensors:

```python
import pyrlx
gguf = pyrlx.load_gguf("qwen3-0.6b-Q4_K_M.gguf")        # inspect tensors + metadata
pyrlx.convert_to_gguf("model.safetensors", "out.gguf", quant="Q4_K_M")
```

Qwen3 runs **end-to-end on Metal at 100% top-1 parity vs HuggingFace** and
matches/beats Python MPS on most prefill shapes (see [Status by area](#status-by-area)
and [`docs/gguf-backend-paths.md`](docs/gguf-backend-paths.md)).

## Workspace layout

The umbrella `rlx` crate is a prelude that re-exports the framework; the
first-party workspace is 61 crates under
`crates/{core,backends,io,numerics,tooling,bindings}/`, each with its own
`README.md`.

```
rlx                  umbrella prelude — re-exports the framework + common types

core/  — IR, compiler, autodiff, runtime
  rlx-ir             tensor IR: types, shapes, Op enum, verifier, serialization (leaf)
  rlx-tensor         NumPy-style symbolic Tensor DSL (lazy: trace → fuse → any backend)
  rlx-flow           block assembly-line API for model builders (fusion-first)
  rlx-autodiff       JAX-shaped transforms on the MIR — grad / jvp / hvp / vmap
  rlx-fusion         MIR fusion passes + fused-op decomposition
  rlx-unfuse         shared IR unfuse/decompose pass for the GPU-family backends
  rlx-compile        HIR → MIR → LIR pipeline (legalize, memory plan, precision)
  rlx-opt            facade — re-exports fusion + autodiff + compile
  rlx-optim          training-step optimizers (Adam / Lion / Muon / SOAP / Sophia / …)
  rlx-driver         Device enum + driver layer (handles, arenas, buffers, streams)
  rlx-runtime        user-facing Session / CompiledGraph; feature-gated backends
  rlx-macros         proc macros — #[rlx_model] AOT model compilation
  rlx-extend         one prelude for extending rlx downstream (custom ops / blocks / DSL)
  rlx-collectives    in-graph collectives (all-reduce, WideEP moe_dispatch/combine, EPLB)
  rlx-distributed    multi-node pipeline parallel — partition + shard + transport, pool RAM

backends/  — one crate per hardware target
  rlx-cpu            CPU: SIMD (NEON/AVX) + BLAS dispatch + threaded arena executor
  rlx-metal          Apple Metal (custom MSL + MPSGraph + indirect command buffers)
  rlx-mlx            Apple MLX via hand-rolled C++ shim (eager + lazy)
  rlx-mlx-sys        low-level MLX C++ build + C-ABI shim (vendored mlx)
  rlx-coreml         Apple CoreML / Neural Engine — IR → MIL ML Program (Device::Ane)
  rlx-cuda           NVIDIA CUDA — cuBLAS + cuDNN + NVRTC kernels (opt. NCCL) via cudarc
  rlx-rocm           AMD ROCm/HIP — shares rlx-cuda's .cu sources, dispatched via HIP
  rlx-xdna           AMD XDNA / Ryzen AI NPU — Device::Xdna detection + AIE execution
  rlx-oneapi         Intel oneAPI Level Zero (Arc / Data Center Max) + SPIR-V kernels
  rlx-tpu            Google TPU via libtpu's PJRT plugin
  rlx-vulkan         native Vulkan compute (raw ash + embedded SPIR-V)
  rlx-wgpu           cross-platform GPU via wgpu (Metal / Vulkan / DX12 / WebGPU)
  rlx-webgl          WebGL2 GPGPU — render-to-texture graph execution in-browser
  rlx-qnn            Qualcomm Hexagon NPU — IR → QNN model (Device::Hexagon)
  rlx-cortexm        Cortex-M INT8 kernels for ARMv7E-M microcontrollers (no_std)
  rlx-fpga           FPGA — per-graph datapath synthesis (IR → Verilog → bitstream)
  rlx-cerebras       Cerebras Wafer-Scale Engine — IR → CSL → fabric simulator
  rlx-gpu-host       backend-agnostic host-fallback kernels (D2H → CPU → H2D)
  rlx-gpu-kernels    shared CUDA/HIP C++ kernel sources (NVRTC / hipRTC)

io/  — model + weight loading, conversion, packaging
  rlx-gguf           standalone GGUF v1/2/3 parser + dequant (every llama.cpp scheme)
  rlx-gguf-convert   safetensors / ONNX → GGUF with per-tensor quantization
  rlx-onnx           ONNX inference — native compile by default, optional ORT fallback
  rlx-onnx-import    ONNX → RLX HIR import (protobuf + op lowering)
  rlx-onnx-proto     vendored pure-Rust ONNX protobuf types (protoc-free)
  rlx-onnx-conformance   ONNX op-level conformance tests (ORT vs RLX)
  rlx-torch-import   PyTorch (torch.export) → RLX HIR (aten→rlx registry + gen crate)
  rlx-nemo           native loader for NVIDIA NeMo .nemo files
  rlx-dduf           HuggingFace DDUF (.dduf) package loader
  rlx-mlx-io         MLX weight layouts (mlx-community safetensors / .npz / affine+mxfp)
  rlx-hub            HuggingFace shard-aware download + verify (resumable, layer→shard)
  rlx-pkg            RLX package format (.rlxp) — flat hybrid mmap packs
  rlx-bake           offline bake: graph + weights → optimized *.rlx (optional encrypt)
  rlx-text           tokenizers, chat templates, sampling for downstream LM apps

numerics/  — downstream domain packages (register against the custom-op scaffold)
  rlx-linalg         dense linear algebra via LAPACK (eigh / svd / qr / cholesky / solve)
  rlx-sparse         sparse linear algebra — CSR LU, mat-vec, conjugate gradient
  rlx-vq             fused vector-quantization kernel (nearest-codebook assignment)
  rlx-umap           parametric UMAP (fit / transform + k-NN)
  rlx-fdm            force density method — pin-jointed form-finding
  rlx-bbo            black-box optimization + FMQ/QGBS search
  rlx-rl             flow-map generative policies (FMQ / QGBS)

tooling/
  rlx-check          device-free static graph checker (shape / dispatch / fusion / NaN)
  rlx-bench          benchmark harness + `rlx-gpu` GPU telemetry/control CLI
  rlx-hwprofile      host hardware profiler — GPU/VRAM detection for device selection
  rlxsl              scalar-expr manifest → per-language activation kernels (WGSL/CUDA/MSL/…)

bindings/
  pyrlx              Python bindings (PyO3) — run RLX graphs + HF models on any backend
  rlx-web            WebAssembly entry point — run models in-browser (CPU; WebGPU bring-up)
```

Each crate carries its own `README.md` covering public surface, build
commands, and internal gotchas. Model crates (`rlx-qwen3`, `rlx-nomic`, …)
and `rlx-splat` (3D Gaussian splatting) live in sibling repos, keeping RLX
core model-agnostic.

## Building from source

```sh
cargo build --release                         # cpu only
cargo build --release --features metal,mlx    # apple silicon GPU
cargo test  --release --workspace             # full workspace suite
```

For Apple Silicon, MLX is a git submodule under `rlx-mlx-sys`:

```sh
git submodule update --init rlx-mlx-sys/vendor/mlx
# or: git clone --recurse-submodules …
```

(`.gitmodules` sets `shallow = true` — only the pinned MLX commit is fetched.)

## Documentation

Topic guides live in [`docs/`](docs/README.md); the most useful entry points:

| Doc | Covers |
|-----|--------|
| [`op-coverage.md`](docs/op-coverage.md) | per-op × per-backend support matrix (single source of truth) |
| [`backend-selection.md`](docs/backend-selection.md) | runtime device switching — `GraphDevices`, `DeviceRouter`, cost-based picking |
| [`gguf-backend-paths.md`](docs/gguf-backend-paths.md) | GGUF quant schemes and each backend's dequant / fused-GEMV path |
| [`weight-compute-caching.md`](docs/weight-compute-caching.md) | computing weight-derived tensors once, not every forward |
| [`rlx-bake.md`](docs/rlx-bake.md) | offline bake: graph + weights → optimized / encrypted `*.rlx` |
| [`fpga-export.md`](docs/fpga-export.md) | IR → SystemVerilog → bitstream export |
| [`scaled-matmul-fp8.md`](docs/scaled-matmul-fp8.md) | native FP8 / FP6 / FP4 GEMM + `fNeXmY` minifloats |
| [`distributed.md`](docs/distributed.md) / [`iroh-transport.md`](docs/iroh-transport.md) | data / tensor / expert-parallel training + NAT-traversing transport |
| [`nan-debugging.md`](docs/nan-debugging.md) | `RLX_DEBUG_NANS` / `RLX_LINT_NUMERICS` NaN / Inf localization |
| [`extending.md`](docs/extending.md) | extension seams for downstream crates (custom ops / stages) |
| [`development.md`](docs/development.md) | contributor workflow, gates, and conventions |

## Kernel dispatch and transparency

RLX keeps **native fast paths** as the default while still allowing
**transparent fallback** when a backend has not wired an op yet.

| Path | When | Effect |
|------|------|--------|
| **Native** | `OpKind` is in the backend's `supported_ops` claim | Backend thunk (MSL, CUDA, CPU ref, …) |
| **Common IR** | Registered logical kernel, not in `supported_ops` | Lowered to primitive MIR (`MatMul`, `Reduce`, …) — portable, often slower |
| **Rewritten** | Structural unfuse / lower (e.g. fused matmul → primitives) | Same semantics, different graph shape |
| **Unsupported** | Still illegal after rewrite | Compile fails with a diagnostic report |

Policy (default `PreferNative`): native if claimed, else common IR.
Override globally with `RLX_KERNEL_DISPATCH=common|native`, or per compile
via [`CompileOptions::kernel_dispatch`](crates/core/rlx-runtime/src/options.rs) and
`force_common_kinds` / `force_native_kinds`.

**See what a compile will do** — set `RLX_DISPATCH_REPORT=1` or
`RLX_VERBOSE=1` before `Session::compile`; the runtime prints a per-kind
summary (native / common-ir / rewritten / missing). On failure, the error
includes both legalization details and the dispatch report.

```rust
use rlx::prelude::*;
use rlx::runtime::{
    dispatch_report_for_device, legalize_graph_for_device_with_options, CompileOptions,
    ModelReflection,
};
use rlx::opt::format_dispatch_report;
use rlx_flow::ModelExecutionConfig;
use rlx_ir::CompilationMode;

// Unified component (variant + dispatch + eager/lazy/AOT + profile + layer stack)
let config = ModelExecutionConfig::qwen35_prefill(1, 512)
    .with_compilation_mode(CompilationMode::Lazy);
let _key = config.cache_key();

// Static probe (common-ir kinds only; no unfuse)
let report = dispatch_report_for_device(&graph, Device::Metal)?;
eprintln!("{}", format_dispatch_report(&report));

// Full rewrite + legalize probe (same path as compile)
let opts = CompileOptions::new(); // or compile_options_for_device(&config, Device::Metal)
let (graph, report) =
    legalize_graph_for_device_with_options(graph, Device::Metal, &opts)?;
```

[`supports_graph`](crates/core/rlx-runtime/src/device_ext.rs) uses the backend
`supported_ops` claim set when a backend is registered, so device
picking stays aligned with compile rather than hand-maintained op tables.

More detail: [`rlx-ir/README-logical-kernels.md`](crates/core/rlx-ir/README-logical-kernels.md)
(registered logical kernels, splat example, API table).

To speed up a workload: implement the native thunk, add the `OpKind` to
that backend's `supported_ops`, and re-run with `RLX_DISPATCH_REPORT=1`
until the kind moves from **common-ir** to **native**.

## Development workflow

- **Fast local gate**: `just ci` (build, workspace tests, lint, pyrlx pytest).
- **Always gate benches on throttle.** `scripts/check-throttle.sh` refuses
  to proceed under thermal pressure (`pmset -g therm`). Silent 10×
  slowdowns are a real failure mode on Macs. `--warn` mode for CI;
  `RLX_ALLOW_THROTTLE=1` for one-off bypass.
- **Use `rlx_ir::Tick` for sub-ms timing** (CNTVCT_EL0 directly, not
  `Instant::now`). Re-exported from `rlx_runtime` for convenience.
- **Touch every backend when you add an Op.** New ops mean: rlx-ir
  (op.rs, infer.rs, graph.rs, verify.rs), every backend's thunks +
  cost models (rlx-cpu, rlx-metal, rlx-mlx, rlx-cuda, rlx-rocm, rlx-tpu,
  rlx-wgpu — sister-crate ports are usually mechanical), the optimizer
  fusion patterns, and ideally a cross-backend parity test. Use
  `RLX_DISPATCH_REPORT=1` after compile to confirm native vs common-ir.
- **Bench every change.** [`rlx-bench`](crates/tooling/rlx-bench) is the
  in-tree harness — `cargo run -p rlx-bench --release --example bench_all`
  sweeps every (pattern × device) cell.
- **Record bench deltas.** Performance-affecting changes should note the
  before/after in the PR — even "within noise" is data worth recording.

The per-release feature history — every landed Op, backend, and fusion —
lives in [`CHANGELOG.md`](CHANGELOG.md).

## Training

RLX trains as well as infers. Autodiff runs in both directions on the MIR
([`rlx-autodiff`](crates/core/rlx-autodiff)): reverse-mode `grad` (through
attention, Scan / SSM, SPD manifolds, DiT modulation, eigendecomposition, …),
forward-mode `jvp` / `hvp`, and `vmap`. Training-step optimizers live in
[`rlx-optim`](crates/core/rlx-optim) — Adam / AdamW / NAdamW / RAdam / Lion /
LAMB / Adafactor / SOAP / Muon / Sophia / MARS.

- **On-device training** — gradients run on the accelerator; the CoreML / ANE
  backend does on-device gradient training behind the `training` feature.
- **Quantization-aware training** — PTQ + straight-through estimator + LSQ
  (EMA / Fixed / PerBatch scale modes).
- **Distributed data- / tensor- / expert-parallel** training with NAT-traversing
  transport — WideEP MoE (`moe_dispatch` / `moe_combine` + EPLB) and optional
  CUDA NCCL (`--features nccl`); see [`docs/distributed.md`](docs/distributed.md)
  and [`docs/iroh-transport.md`](docs/iroh-transport.md) (the `rlx-vision-bench`
  trainer is the multi-dataset, multi-machine harness).
- **NaN / Inf localization** during training via `RLX_DEBUG_NANS` —
  [`docs/nan-debugging.md`](docs/nan-debugging.md).

## Status by area

| Area                         | State                                         |
|------------------------------|-----------------------------------------------|
| CPU forward + backward       | Mature; full **153/`OpKind`** claim (fused/control expand before thunks) |
| Apple Metal forward          | Mature; full OpKind claim; native fused Gru/Rnn/Mamba2 + training bwd |
| Apple MLX forward + backward | Mature; full OpKind claim; tier-1/2/3 backward parity |
| NVIDIA CUDA                  | Full **153** claim; shared `.cu` training/QAT/RNN/I8 depth; DenseSolve via cuSOLVER; optional NCCL (`--features nccl`); less field mileage than Metal |
| AMD ROCm                     | Sister-crate parity to CUDA (shared kernels + hipSOLVER DenseSolve/Eigh) |
| TPU                          | Full **153** claim; HLO compose for norms/QAT/conv-bwd/MaxPool/Attention bwd; host for DenseSolve/SPD/`FftButterflyStage`; MiniLM-L6 E2E via PJRT |
| WGPU                         | Full OpKind claim; coop-matrix paths under test; on-device C64 + native Gru/Rnn/Mamba2 / FftButterfly / act bwd |
| Apple CoreML / ANE (`Device::Ane`) | Full OpKind claim; transformer block on-device (IR → MIL); Complex*/FftButterfly MIL; backward + on-device gradient training behind the `training` feature |
| Native Vulkan (`Device::Vulkan`) | Full **153** claim; broad SPIR-V set (norms/fused/RNN/vision-bwd/I8 quant/FFT butterfly) + host/unfuse for specialty ops; C64 arithmetic; parity on MoltenVK / lavapipe |
| Intel oneAPI (`Device::OneApi`) | Full **153** claim; OpenCL-C SPIR-V kernel set mirrors Vulkan depth when `RLX_ONEAPI_BUILD_KERNELS=1`; else CPU reference; DenseSolve stays HostOpDesc→LAPACK |
| Qualcomm Hexagon (QNN)       | Codegen + FFI runtime (`Device::Hexagon`): on-device INT8/INT4 `QMatMul`, `DequantMatMul`, `FusedAttentionBlock`, persistent session + context-binary save/load; x86 HTP functional sim (`just qnn-htp-sim`) |
| Cortex-M (INT8)              | Production: 96.6% MNIST on nRF52840 hardware  |
| FPGA                         | First-class SystemVerilog / bitstream export (`rlx_runtime::export`, `pyrlx.export_fpga`); Int8/Int4/Fp4, soft-port RTL + ECP5 / iCE40 / Xilinx7 synth |
| Reverse-mode AD              | Phase 1–9 complete; SelectiveScan, FusedTL    |
| Forward-mode AD (`jvp`/`hvp`)| Functional; thin public API                   |
| `vmap`                       | MVP — leading-axis batching                   |
| QAT (PTQ + STE + LSQ)        | Complete: EMA, Fixed, PerBatch, propagation; native FakeQuantize/LSQ on CUDA/ROCm/Vulkan/Metal/wgpu |
| OpKind coverage matrix       | **153/153** on CPU / Metal / MLX / wgpu / ANE / CUDA / ROCm / Vulkan / oneAPI / TPU — [`docs/op-coverage.md`](docs/op-coverage.md) (`just gen-op-coverage`) |
| Qwen3 LM (safetensors + GGUF)| End-to-end on Metal: 100% top-1 parity vs HF; matches/beats Python MPS on most prefill shapes. Q4_K_M GGUF loads + runs |
| Op::DequantMatMul GGUF schemes | All llama.cpp schemes (incl. Q4_1, Q5_0, Q5_1, IQ/TQ/MX). GPU dequant on Metal/CUDA/ROCm/WGPU (shared scheme ids 0–23); **Metal fused GEMV** for Q4_K, Q4_0/1, Q8_0, IQ4NL, IQ2/3/1 families (`m=1` prefill); WGPU grouped MoE GPU when scratch fits; ANE MIL constexpr for K/IQ/TQ/MX; TPU compile-time + runtime Param bake. **pyrlx:** `quantize`, `load_gguf`, `convert_to_gguf`. See [docs/gguf-backend-paths.md](docs/gguf-backend-paths.md). |
| Sampler chain                  | `SamplerChain` in `rlx-runtime::samplers`: Temperature, DynamicTemperature, TopK, TopP, TopNSigma, TypicalP, Mirostat v1/v2, XTC, DRY, RepetitionPenalty. Wired into `SampleOpts::into_chain()`; classic top-k/top-p stay on the fast path via `is_classic()`. |
| Quantized KV cache             | Per-layer K/V stored as q4_0 / q5_0 / q8_0 / f16 blocks via `rlx-runtime::quantized_kv`. Optional `mmap-kv` feature spills to a file-backed mapping for long contexts. |
| DiT modulation (`AdaLayerNorm` / `GatedResidual`) | First-class fused ops on every backend; fused **forward + backward** (packed reverse kernels), exact JVP + `vmap`; ONNX adaLN import fuses after param specialization. See [`docs/op-coverage.md`](docs/op-coverage.md). |
| Eigendecomposition (`Op::Eigh` / `EighBatch`) | Differentiable symmetric eigensolver on all backends: CPU-native (LAPACK `dsyevd` f64), native on-device CUDA/ROCm batched Jacobi (cuSOLVER / hipSOLVER `SsyevjBatched`, `n ≤ 32`), f64 host-fallback elsewhere. Full reverse mode (`EighBackward`). |
| SPD-manifold Riemannian primitives | `SpdKarcherMeanWeighted` / `SpdLogMap` / `SpdExpMap` / `SpdParallelTransport` / `SpdMatrixFnBatch` as first-class ops on all backends, fully differentiable (analytic Riemannian VJPs, incl. the base point). |
| Static graph checker            | `rlx_runtime::check` folds shape/dtype verify + backend dispatch + missed-fusion hints + provable-NaN/Inf lint into one `CheckReport`; `cargo rlx check` (`--json`) and `#[rlx_model(check)]` (`RLX_CHECK`). |
| Weight-compute caching          | Compile-time `CompileOptions::param_bindings` (bake + broadcasting `ConstantFolding`) and run-time `cache_param_invariant` / `RLX_CACHE_PARAM_INVARIANT=1` (split prepare graph) compute weight-derived tensors once, not every forward. Validated CPU + CUDA. |
| Offline bake (`rlx-bake`)       | Merge graph + weights into a deployable `*.rlx` (skip-zero / ternary TQ2_0 / opt-in Q8_0, optional encrypt). CLI + crate; walkthrough [`docs/rlx-bake.md`](docs/rlx-bake.md). |
| Complex dtypes (`C64` / `C128`) | `DType::C64` arithmetic (add/sub/mul/div) + cast matrix on CPU and on-device CUDA / ROCm / Vulkan / wgpu / oneAPI; `DType::C128` cast plumbing (arithmetic remains C64-only). Parity tests per backend. |
| WideEP MoE + NCCL               | Host `moe_dispatch` / `moe_combine` + EPLB (`rlx-collectives::{moe_ep,eplb}`); CUDA NCCL scaffold for device-resident `all_reduce` / `all_to_all` (`rlx-cuda --features nccl`). Toy: `wide_ep_toy` / `nccl_all_reduce_toy`. See [`docs/distributed.md`](docs/distributed.md). |

## Authors

Eugene Hauptmann, Nataliya Kosmyna ([MIT-RLX](https://github.com/MIT-RLX)).

## Citing RLX

If you use RLX in academic work, please cite it. Machine-readable metadata is
in [`CITATION.cff`](CITATION.cff); a plain-text form:

```bibtex
@software{rlx,
  title  = {RLX: a small ML compiler and runtime for transformer inference and training},
  author = {Hauptmann, Eugene and Kosmyna, Nataliya},
  year   = {2026},
  url    = {https://github.com/MIT-RLX/rlx}
}
```

## Contributing

PRs are welcome. Keep changes focused and note any before/after bench
delta in the PR (even "within noise" is data worth recording). The
[development workflow](#development-workflow) above covers the technical
loop; [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`docs/development.md`](docs/development.md) go deeper, and each crate's
`README.md` is its canonical reference.

**Get started**

```sh
git clone --recurse-submodules https://github.com/MIT-RLX/rlx
just install-git-hooks     # symlink the repo pre-commit hook
just ci                    # build · test · fmt-check · lint · wasm · pyrlx — the gate every PR must pass
```

**A PR is ready when** `just ci` is green, new behavior is covered by a
test — kernels get a **CPU-parity test**, since the CPU path is the
numerical oracle every backend is checked against — and it adds a one-line
entry to [`CHANGELOG.md`](CHANGELOG.md)'s `[Unreleased]` section. Adding an
`Op` means landing it on *every* backend (see the dev-workflow note above),
not just the one you use.

**You may not need to touch core.** Custom ops, `LayerStage` blocks, and the
flow DSL are exposed through [`rlx-extend`](crates/core/rlx-extend), so
downstream crates (new models, domain numerics) can add capability without
editing rlx's closed enums — reach for that seam before patching the core.

Security issues: please follow [`SECURITY.md`](SECURITY.md) rather than
opening a public issue.

## License

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](./LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([`LICENSE-MIT`](./LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
