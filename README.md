# RLX

![status](https://img.shields.io/badge/status-0.2.19-blue)
![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green)
![rust](https://img.shields.io/badge/rust-edition%202024-orange)
[![repo](https://img.shields.io/badge/github-MIT--RLX%2Frlx-black)](https://github.com/MIT-RLX/rlx)
[![arXiv](https://img.shields.io/badge/arXiv-2609.37916-b31b1b)](https://arxiv.org/abs/2609.37916)

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

The design, IR, and cross-backend benchmarks are written up in
[*RLX: A Unified Multi-Backend Tensor Compiler and Distributed Runtime in
Rust*](https://arxiv.org/abs/2609.37916) (arXiv:2609.37916).

## Table of Contents

- [Why another one](#why-another-one)
- [Install](#install)
- [Quickstart](#quickstart)
- [Language bindings](#language-bindings)
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
- [Development workflow](#development-workflow)
- [Training](#training)
- [Status by area](#status-by-area)
- [Authors](#authors)
- [Citing RLX](#citing-rlx)
- [Contributing](#contributing)
- [License](#license)

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
| `metal`             | Metal Performance Shaders + MSL      | macOS / iOS / tvOS / visionOS |
| `mlx`               | Apple MLX (vendored)                 | macOS / iOS / tvOS / visionOS |
| `mlx-cuda`          | Apple MLX's CUDA backend             | Linux + NVIDIA            |
| `coreml`            | CoreML / Neural Engine (`Device::Ane`)| macOS / iOS / tvOS / visionOS |
| `gpu`               | wgpu (Vulkan / DX12 / WebGPU / Metal)| cross-platform            |
| `vulkan`            | native Vulkan compute (`ash` + SPIR-V)| Linux / Windows (+ MoltenVK)|
| `oneapi`            | Intel oneAPI Level Zero (SPIR-V)     | Linux + Intel GPU         |
| `cuda`              | cuBLAS / cuDNN / NVRTC               | Linux / Windows + NVIDIA  |
| `rocm`              | hipBLAS / MIOpen                     | Linux + AMD               |
| `tpu`               | libtpu PJRT plugin                   | Linux + GCP TPU           |
| `egpu`              | external GPU on a USB4/TB PCIe tunnel| macOS (discovery + transport)|
| `blas-accelerate`   | macOS Accelerate                     | macOS                     |
| `blas-mkl`          | Intel MKL                            | Intel / AMD CPUs          |
| `blas-openblas`     | OpenBLAS                             | cross-platform CPU        |

**Apple platforms.** macOS, iOS, tvOS and visionOS carry CPU + Metal +
CoreML/ANE + MLX. **watchOS is CPU/Accelerate only** — it has no public Metal API
and no CoreML runtime model-compile, so there is no backend to build there.
`just check-apple` cross-compiles all four OSes (device *and* simulator) and
`just test-apple-sim` runs the backend smoke + parity tests on all four
simulators. Metal, MLX and wgpu skip on simulators — a headless `simctl spawn`
exposes no Metal device — so they are validated on hardware instead: all four
backends match the CPU reference at `max|Δ| = 1.49e-8` on an iPad Pro (iOS 26.5).
tvOS, watchOS and visionOS hardware remains untested. See
[`ios/README.md`](ios/README.md) for the slice matrix, the deployment floors, and
packaging.

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
| `onnx`         | run `.onnx` models — native compile, optional ORT fallback|
| `optim`        | training-step optimizers (Adam/Lion/Muon/…, `rlx::optim`)|
| `distributed`  | in-graph collectives + `rlx::distributed` front door     |
| `umap`         | UMAP dimensionality reduction (`rlx::umap`)              |
| `geo`          | exact integer Delaunay + discrete Voronoi (`rlx::geo`)   |
| `bench`        | uniform benchmark harness                                |
| `sparse`       | sparse linear algebra (custom-op scaffold)               |
| `linalg`       | dense linalg via LAPACK (custom-op scaffold)             |
| `splat`        | 3D Gaussian splat opcodes + executor registry (`rlx::splat`) — renderer is the out-of-tree `rlx-splat` |
| `peft`         | LoRA / IA3 / AdaLoRA / DoRA / OFT adapters (`rlx::peft`) |
| `rng`          | bit-exact numpy / PyTorch random streams (`rlx::rng`)    |

### Specialty crates

The `Backend` model doesn't fit microcontrollers or hardware synthesis.
For those, depend on the standalone crates directly — they're not
exposed through the prelude:

- `rlx-cortexm` — `no_std` ARMv7E-M INT8 kernels.
- `rlx-fpga` — IR → SystemVerilog → bitstream (first-class export via
  `rlx_runtime::export` / `pyrlx.export_fpga` under the `fpga` feature);
  `no_std` + `alloc` without its default `std` feature.
- `rlx-qnn` — Qualcomm Hexagon / QNN codegen + FFI runtime (`Device::Hexagon`;
  enable `rlx-runtime`'s `qnn` feature, not on the umbrella prelude).
- `rlx-cerebras` — Cerebras WSE: IR → CSL → fabric simulator. No `Device`
  variant; the fabric is targeted through codegen, not a runtime backend.
- `rlx-ffi` — C ABI over the distributed node, for the Apple platforms
  (staticlib → one xcframework spanning iOS / tvOS / watchOS / visionOS) and
  embedded hosts (cdylib).

## Quickstart

A single `use rlx::prelude::*;` covers the common surface: graph
types, `Session`, `Device`, ops + activations, and `Result`.

```rust
use rlx::prelude::*;

let mut g = Graph::new("hello");
let x = g.input("x", Shape::new(&[1, 4], DType::F32));
let w = g.param("w", Shape::new(&[4, 2], DType::F32));
let y = g.mm(x, w); // GraphExt builders infer the output shape
let y = g.relu(y);
g.set_outputs(vec![y]);

let mut compiled = Session::new(Device::Cpu).compile(g);
compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
let out = compiled.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
assert_eq!(out[0], vec![4.0, 6.0]); // `run` returns one Vec<f32> per graph output
```

Mistakes at the boundary are reported by name: a misspelled input
(`the graph has no input `X` (did you mean `x`?)`) or a buffer of the wrong
length panics in `run` / `set_param`, and `try_run` returns the same as an
`IoError`. Inputs left unsupplied and params never set are named in a one-time
warning; `check_bindings` returns them as an error.

`g.matmul(x, w, shape)` and the other explicit-shape builders remain for
graphs whose output shape you want to state yourself.

Prefer NumPy-style expressions? The `rlx::tensor` DSL ([`rlx-tensor`](crates/core/rlx-tensor/README.md))
builds the same IR with operator-overloaded, lazy `Tensor` handles —
`(&a + &b).relu()` traces instead of executing, then fuses + memory-plans
across any backend when you call `.to_vec()`:

```rust
use rlx::prelude::*; // features `cpu` + `tensor`

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
let c = (&a + &Tensor::ones([3])).relu();
assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0]); // runs on the fastest backend built in
```

Or declare the graph in a compact little language with `rlx!` (feature `tensor`;
the `rlx` crate has no default features, so enable it explicitly) — `@` is matmul with NumPy precedence, shapes/wiring/outputs are
inferred, and it lowers to the same IR:

```rust
use rlx::prelude::*; // feature `tensor`

let g = rlx! {
    graph "mlp";
    input x: [?, 784];              // `?` = dynamic batch
    param w1: [784, 256];  param b1: [256];
    param w2: [256, 10];   param b2: [10];

    let h = gelu(x @ w1 + b1);
    let y = h @ w2 + b2;
    out y;                          // → rlx_ir::Graph
};
assert_eq!(g.outputs.len(), 1);
```

Domain-specific namespaces if you want narrower star-imports:
`rlx::ops::*` (IR helper enums), `rlx::quant::*`, `rlx::autodiff::*`.
Or the full per-crate surface
via `rlx::ir::…` / `rlx::opt::…` / `rlx::runtime::…` etc. — every
workspace crate is reachable as a module on `rlx`. Or depend on each
crate directly (`rlx-ir`, `rlx-opt`, `rlx-runtime`, …) for the
smallest possible dep tree.

## Language bindings

RLX is the same API in every language: the same graph builders and the same
runtime (sessions, device selection, autodiff, training, weights, text,
distributed nodes), generated from one source and tested one-to-one against
it. Each language's package has the same six runnable examples — inference,
training, loading, distributed, backends, models — printing the same numbers.

| Language | Package | Examples |
|----------|---------|----------|
| Python | [`crates/bindings/pyrlx`](crates/bindings/pyrlx) | [`pyrlx/examples/`](crates/bindings/pyrlx/examples/README.md) |
| JavaScript | [`crates/bindings/rlx-js`](crates/bindings/rlx-js) | [`rlx-js/examples/`](crates/bindings/rlx-js/examples/README.md) |
| Swift (iOS, macOS) | [`crates/bindings/rlx-swift`](crates/bindings/rlx-swift) | [`rlx-swift/Examples/`](crates/bindings/rlx-swift/Examples/README.md) |
| Go | [`crates/bindings/rlx-go`](crates/bindings/rlx-go/README.md) | [`rlx-go/examples/`](crates/bindings/rlx-go/examples) |
| Java (FFM) | [`crates/bindings/rlx-java`](crates/bindings/rlx-java/README.md) | [`rlx-java/examples/`](crates/bindings/rlx-java/examples/org/rlx/examples) |
| C# / .NET | [`crates/bindings/rlx-csharp`](crates/bindings/rlx-csharp/README.md) | [`rlx-csharp/examples/`](crates/bindings/rlx-csharp/examples/Rlx.Examples) |
| Ruby | [`crates/bindings/rlx-ruby`](crates/bindings/rlx-ruby/README.md) | [`rlx-ruby/examples/`](crates/bindings/rlx-ruby/examples) |
| Julia | [`crates/bindings/rlx-julia`](crates/bindings/rlx-julia/README.md) | [`rlx-julia/examples/`](crates/bindings/rlx-julia/examples) |
| Zig | [`crates/bindings/rlx-zig`](crates/bindings/rlx-zig/README.md) | [`rlx-zig/examples/`](crates/bindings/rlx-zig/examples) |
| C++ | [`crates/bindings/rlx-cpp`](crates/bindings/rlx-cpp/README.md) | [`rlx-cpp/examples/`](crates/bindings/rlx-cpp/examples) |
| Node.js / TypeScript | [`crates/bindings/rlx-node`](crates/bindings/rlx-node/README.md) | [`rlx-node/examples/`](crates/bindings/rlx-node/examples) |
| Kotlin (JVM, Android) | [`crates/bindings/rlx-kotlin`](crates/bindings/rlx-kotlin/README.md) | [`rlx-kotlin/examples/`](crates/bindings/rlx-kotlin/examples) |
| Dart / Flutter | [`crates/bindings/rlx-dart`](crates/bindings/rlx-dart/README.md) | [`rlx-dart/example/`](crates/bindings/rlx-dart/example) |
| Lua | [`crates/bindings/rlx-lua`](crates/bindings/rlx-lua/README.md) | [`rlx-lua/examples/`](crates/bindings/rlx-lua/examples) |
| MATLAB / Octave | [`crates/bindings/rlx-matlab`](crates/bindings/rlx-matlab/README.md) | [`rlx-matlab/examples/`](crates/bindings/rlx-matlab/examples) |
| R | [`crates/bindings/rlx-r`](crates/bindings/rlx-r/README.md) | [`rlx-r/inst/examples/`](crates/bindings/rlx-r/inst/examples) |
| PHP | [`crates/bindings/rlx-php`](crates/bindings/rlx-php/README.md) | [`rlx-php/examples/`](crates/bindings/rlx-php/examples) |
| Nim | [`crates/bindings/rlx-nim`](crates/bindings/rlx-nim/README.md) | [`rlx-nim/examples/`](crates/bindings/rlx-nim/examples) |
| D | [`crates/bindings/rlx-d`](crates/bindings/rlx-d/README.md) | [`rlx-d/examples/`](crates/bindings/rlx-d/examples) |
| V | [`crates/bindings/rlx-v`](crates/bindings/rlx-v/README.md) | [`rlx-v/examples/`](crates/bindings/rlx-v/examples) |
| Haskell | [`crates/bindings/rlx-haskell`](crates/bindings/rlx-haskell/README.md) | [`rlx-haskell/examples/`](crates/bindings/rlx-haskell/examples) |
| OCaml | [`crates/bindings/rlx-ocaml`](crates/bindings/rlx-ocaml/README.md) | [`rlx-ocaml/examples/`](crates/bindings/rlx-ocaml/examples) |
| Elixir | [`crates/bindings/rlx-elixir`](crates/bindings/rlx-elixir/README.md) | [`rlx-elixir/examples/`](crates/bindings/rlx-elixir/examples) |
| Odin | [`crates/bindings/rlx-odin`](crates/bindings/rlx-odin/README.md) | [`rlx-odin/examples/`](crates/bindings/rlx-odin/examples) |
| C / any FFI language | [`crates/bindings/rlx-capi`](crates/bindings/rlx-capi) | [`rlx-capi/examples/`](crates/bindings/rlx-capi/examples/README.md) |
| Browser (wasm) | [`crates/bindings/rlx-web`](crates/bindings/rlx-web) | [`rlx-web/README.md`](crates/bindings/rlx-web/README.md) |

Start at [`crates/bindings/README.md`](crates/bindings/README.md) for build
commands and a side-by-side snippet per language;
[`docs/bindings-api.md`](docs/bindings-api.md) maps every function's name across
languages, and [`docs/bindings.md`](docs/bindings.md) covers adding an op, a
runtime function or a new language.

## Examples

Each example below is a minimal, faithful sketch; runnable end-to-end versions
live under each crate's `examples/` (e.g. `cargo run -p rlx-runtime --example
int8_qat_train --features cpu`).

### Simple example

Build a 2-layer MLP, compile it for the CPU, and run a forward pass. `run`
returns one `Vec<f32>` per graph output.

```rust
use rlx::prelude::*;

let mut g = Graph::new("mlp");
let x  = g.input("x",  Shape::new(&[1, 4], DType::F32));
let w1 = g.param("w1", Shape::new(&[4, 8], DType::F32));
let w2 = g.param("w2", Shape::new(&[8, 2], DType::F32));
let h = g.mm(x, w1);
let h = g.relu(h);
let y = g.mm(h, w2);
g.set_outputs(vec![y]);

// `try_compile` / `try_run` return errors instead of panicking.
let mut net = Session::new(Device::Cpu).try_compile(g).expect("compiles");
net.set_param("w1", &[0.1; 4 * 8]);
net.set_param("w2", &[0.1; 8 * 2]);
let out = net.try_run(&[("x", &[1.0, 2.0, 3.0, 4.0])]).expect("inputs match");
assert_eq!(out[0].len(), 2); // one Vec<f32> per graph output
println!("{:?}", out[0]);

// Mistakes are named: `the graph has no input `X` (did you mean `x`?)`.
assert!(net.try_run(&[("X", &[0.0; 4])]).is_err());
```

### Browser example — a training cluster across tabs and devices

A standalone static page (no backend) where **every browser tab is a
data-parallel rank**: each compiles an RLX backward graph in a Web Worker,
computes gradients over its own shard, and the cohort averages them. Tabs in one
browser pair over a `BroadcastChannel`; other devices join a WebRTC mesh by
scanning a QR code, with signalling over public relays and every gradient
travelling peer-to-peer.

```sh
just build-cluster-site          # → ./examples/tab-cluster  (deployable as-is)
just serve-web --open cluster.html
```

It is the same compiler the native backends use, so the page also shows the
choices that normally stay invisible:

| Control | What it exposes |
|---------|-----------------|
| backend | `cpu` / `webgpu` / `webgl` per rank, and a mixed cohort is visible in the roster |
| batch | doubling steps, 2 → 65,536; WebGPU only beats CPU above ~64 |
| wire | `f32` / `f16` / `f8-e4m3` / `f8-e5m2` payloads — 3.1× less traffic at f8, with the error that costs |

The page ends with the same model built three ways — Rust, JavaScript
(`WebGraph` over wasm) and Python (`pyrlx`) — each describing the graph, letting
autodiff rewrite it, and training it. Details, measurements and the
failure modes that shaped it are in
[`crates/bindings/rlx-web/README.md`](crates/bindings/rlx-web/README.md).

### Advanced example

Differentiate a graph (reverse mode), let RLX pick the fastest backend actually
present on the machine, and print what the compiler did per op.

```rust
use rlx::prelude::*;

// f(x) = Σ tanh(x)², and its gradient ∂f/∂x = 2·tanh(x)·(1 − tanh(x)²).
let mut f = Graph::new("f");
let x = f.input("x", Shape::new(&[4], DType::F32));
let t = f.tanh(x);
let sq = f.mul(t, t);
let s = f.sum(sq, vec![0], false);
f.set_outputs(vec![s]);

let df = grad(&f, &[x]);                       // reverse mode; outputs [∂f/∂x]
let device = fastest_device();                 // Metal / CUDA / wgpu / … else CPU
let mut c = Session::new(device).try_compile(df).expect("compiles");
let xs = [0.5f32, -1.0, 2.0, 0.0];
let g = c.run(&[("x", &xs), ("d_output", &[1.0])]); // `d_output`: the seed ∂f/∂f
for (gi, xi) in g[0].iter().zip(xs) {
    let th = xi.tanh();
    assert!((gi - 2.0 * th * (1.0 - th * th)).abs() < 1e-4);
}
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
use rlx::prelude::*; // feature `optim` for the optimizers

// Fit y = x·w to targets made with w* = [2, -3]: loss = Σ (x·w − y)².
let mut f = Graph::new("linreg");
let x = f.input("x", Shape::new(&[4, 2], DType::F32));
let y = f.input("target", Shape::new(&[4, 1], DType::F32));
let w = f.param("w", Shape::new(&[2, 1], DType::F32));
let pred = f.mm(x, w);
let err = f.sub(pred, y);
let sq = f.mul(err, err);
let loss = f.sum(sq, vec![0, 1], false);
f.set_outputs(vec![loss]);

let backward = grad_with_loss(&f, &[w]);          // outputs: [loss, ∂loss/∂w]
let mut c = Session::new(Device::Cpu).try_compile(backward).expect("compiles");

let xs = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
let ys = [2.0f32, -3.0, -1.0, 7.0];
let mut wv = vec![0.0f32; 2];
let mut opt = Adam::new(0.1);
let mut last = f32::MAX;
for step in 0..300 {
    c.set_param("w", &wv);                             // upload the current weights
    let out = c.run(&[("x", &xs), ("target", &ys), ("d_output", &[1.0])]); // seed ∂loss/∂loss = 1
    last = out[0][0];
    opt.step("w", &[2, 1], &mut wv, &out[1]);          // in-place Adam update
    opt.end_iteration();
    if step % 100 == 0 { println!("step {step}: loss {last:.5}"); }
}
assert!(last < 1e-3 && (wv[0] - 2.0).abs() < 0.05 && (wv[1] + 3.0).abs() < 0.05);
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

let flow = ModelFlow::new("my_model")
    .input("x", Shape::new(&[1, 16], DType::F32))
    .layer_stage(GatedMlp { gate: "gate".into(), up: "up".into(), down: "down".into() });
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
first-party workspace is 85 crates under
`crates/{core,backends,io,numerics,tooling,bindings}/`, each with its own
`README.md`.

```text
rlx                  umbrella prelude — re-exports the framework + common types

core/  — IR, compiler, autodiff, runtime
  rlx-ir             tensor IR: types, shapes, Op enum, verifier, serialization (leaf; no_std core)
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
  rlx-fpga           FPGA — per-graph datapath synthesis (IR → Verilog → bitstream; no_std)
  rlx-cerebras       Cerebras Wafer-Scale Engine — IR → CSL → fabric simulator
  rlx-egpu           external GPU over USB4/Thunderbolt — IOKit PCI discovery + DriverKit
  rlx-gpu-host       backend-agnostic host-fallback kernels (D2H → CPU → H2D)
  rlx-gpu-kernels    shared CUDA/HIP C++ kernel sources (NVRTC / hipRTC)
  rlx-gpu-dispatch   shape-keyed kernel-variant dispatch table shared by all GPU backends
  rlx-cortexm-trainer  native fp32 trainer that emits rlx-cortexm's INT8 weights

io/  — model + weight loading, conversion, packaging
  rlx-gguf           standalone GGUF v1/2/3 parser + dequant (every llama.cpp scheme)
  rlx-gguf-convert   safetensors / ONNX → GGUF with per-tensor quantization
  rlx-onnx           ONNX inference — native compile by default, optional ORT fallback
  rlx-onnx-import    ONNX → RLX HIR import (protobuf + op lowering)
  rlx-onnx-proto     vendored pure-Rust ONNX protobuf types (protoc-free)
  rlx-onnx-conformance   ONNX op-level conformance tests (ORT vs RLX)
  rlx-torch-import   PyTorch (torch.export) → RLX HIR (aten→rlx registry + gen crate)
  rlx-torch-ckpt     native PyTorch checkpoint reader (.pt / .pth / .bin) — no libtorch
  rlx-nemo           native loader for NVIDIA NeMo .nemo files
  rlx-dduf           HuggingFace DDUF (.dduf) package loader
  rlx-mlx-io         MLX weight layouts (mlx-community safetensors / .npz / affine+mxfp)
  rlx-hub            HuggingFace shard-aware download + verify (resumable, layer→shard)
  rlx-pkg            RLX package format (.rlxp) — flat hybrid mmap packs
  rlx-bake           offline bake: graph + weights → optimized *.rlx (optional encrypt)
  rlx-stablehlo      graph → StableHLO MLIR for IREE / MLIR toolchains (+ parameter files)
  rlx-program        package host logic as WebAssembly (tokenize, decode loops) — guest SDK
  rlx-text           tokenizers, chat templates, sampling for downstream LM apps

numerics/  — downstream domain packages (register against the custom-op scaffold)
  rlx-linalg         dense linear algebra via LAPACK (eigh / svd / qr / cholesky / solve)
  rlx-sparse         sparse linear algebra — CSR LU, mat-vec, conjugate gradient
  rlx-vq             fused vector-quantization kernel (nearest-codebook assignment)
  rlx-umap           parametric UMAP (fit / transform + k-NN)
  rlx-fdm            force density method — pin-jointed form-finding
  rlx-fem            nonlinear scalar P1 FEM on triangles (Poisson family)
  rlx-geo            exact integer Delaunay + discrete Voronoi custom ops
  rlx-lbm            moment-encoded lattice Boltzmann (HOME-LBM) — D2Q9 / D3Q27
  rlx-bbo            black-box optimization + FMQ/QGBS search
  rlx-rl             flow-map generative policies (FMQ / QGBS)
  rlx-peft           parameter-efficient adaptation — LoRA / IA3 / AdaLoRA / DoRA / OFT
  rlx-rng            bit-exact numpy + PyTorch random streams (MT19937 / PCG64 / randperm)
  rlx-numpy          numpy host numerics bit-for-bit (pairwise sum, half-even round, …)
  rlx-search         decode-side search: argmax / top-k, beam, CTC, forced alignment

tooling/
  rlx-check          device-free static graph checker (shape / dispatch / fusion / NaN)
  rlx-bench          benchmark harness + `rlx-gpu` GPU telemetry/control CLI
  rlx-hwprofile      host hardware profiler — GPU/VRAM detection for device selection
  rlx-corpus         named kernel corpus every compiler change must clear, by family
  rlx-opscope        data-pattern recorder — inject stat taps, sweep, mine exploitable structure
  rlx-megakernel     standalone study: how far MoE composes toward one device program
  rlxsl              scalar-expr manifest → per-language activation kernels (WGSL/CUDA/MSL/…)
  rlx-bindgen        generates the C header, C glue, Swift and `.pyi` from rlx-bind's tables

bindings/            (one API in every language — see crates/bindings/README.md)
  rlx-bind           the language-neutral core: op table + runtime API table (api.idl)
  pyrlx              Python bindings (PyO3) — run RLX graphs + HF models on any backend
  rlx-js             JavaScript bindings — an embedded QuickJS runtime with `rlx.*`
  rlx-capi           C ABI (generated `rlx.h`) — the hub for Swift and any FFI language
  rlx-swift          Swift package on the C ABI — iOS / macOS, `RLX.xcframework`
  rlx-go, rlx-java, rlx-csharp, rlx-ruby, rlx-julia, rlx-zig
                     generated bindings on the C ABI, one package per language
  rlx-ffi            C ABI for the distributed node — Apple xcframework + embedded hosts
  rlx-micro          C ABI for the Cortex-M kernels — no_std, no heap (MCU firmware)
  rlx-model-plugin   reference model plugin (a provider in a shared library)
  rlx-web            WebAssembly entry point — run models in-browser (CPU; WebGPU bring-up)
```

Each crate carries its own `README.md` covering public surface, build
commands, and internal gotchas. Model crates (`rlx-qwen3`, `rlx-nomic`, …)
and `rlx-splat` (3D Gaussian splatting) live in sibling repos, keeping RLX
core model-agnostic.

### Generated example folders

`./examples/` holds assembled, deployable demos — currently
`examples/tab-cluster/`, built by `just build-cluster-site` from
`crates/bindings/rlx-web/web/`. The page sources are committed so the folder is
browsable; its `pkg/` wasm bundle is gitignored as an 8.4 MB build artifact, so
rebuild it before deploying from a fresh clone.

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
| [`rlx-stablehlo`](crates/io/rlx-stablehlo/README.md) | graph → StableHLO MLIR for IREE / MLIR toolchains |
| [`scaled-matmul-fp8.md`](docs/scaled-matmul-fp8.md) | native FP8 / FP6 / FP4 GEMM + `fNeXmY` minifloats |
| [`distributed.md`](docs/distributed.md) / [`iroh-transport.md`](docs/iroh-transport.md) | data / tensor / expert-parallel training + NAT-traversing transport |
| [`troubleshooting.md`](docs/troubleshooting.md) | **start here when something is wrong** — run/compile errors, wrong numbers on one device, NaN, slowness |
| [`nan-debugging.md`](docs/nan-debugging.md) | `RLX_DEBUG_NANS` / `RLX_LINT_NUMERICS` NaN / Inf localization |
| [`extending.md`](docs/extending.md) | extension seams for downstream crates (custom ops / stages) |
| [`development.md`](docs/development.md) | contributor workflow, gates, and conventions |
| [`adr/`](docs/adr/README.md) | architecture decision records — why core is shaped the way it is |

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
use rlx::opt::format_dispatch_report;
use rlx::runtime::{dispatch_report_for_device, legalize_graph_for_device_with_report};

let mut g = Graph::new("probe");
let x = g.input("x", Shape::new(&[2, 8], DType::F32));
let w = g.param("w", Shape::new(&[8, 8], DType::F32));
let h = g.mm(x, w);
let y = g.gelu(h);
g.set_outputs(vec![y]);

// Static probe: which op kinds run natively, lower to common IR, or are missing.
let report = dispatch_report_for_device(&g, Device::Cpu).expect("cpu is built in");
eprintln!("{}", format_dispatch_report(&report));

// The full rewrite + legalize path `compile` takes, without compiling.
let (_legal, report) =
    legalize_graph_for_device_with_report(g, Device::Cpu).expect("legal on cpu");
assert!(report.compile_ready);
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
| CPU forward + backward       | Mature; the reference surface — **176/187** `OpKind`s claimed (fused/control expand before thunks) |
| Apple Metal forward          | Mature; widest claim of any backend (**179/187**); native fused Gru/Rnn/Mamba2 + training bwd |
| Apple MLX forward + backward | Mature; **168/187**; tier-1/2/3 backward parity |
| NVIDIA CUDA                  | **173/187**; shared `.cu` training/QAT/RNN/I8 depth; DenseSolve via cuSOLVER; optional NCCL (`--features nccl`); less field mileage than Metal |
| AMD ROCm                     | **171/187** — sister-crate parity to CUDA (shared kernels + hipSOLVER DenseSolve/Eigh) |
| TPU                          | **163/187**; HLO compose for norms/QAT/conv-bwd/MaxPool/Attention bwd; host for DenseSolve/SPD/`FftButterflyStage`; MiniLM-L6 E2E via PJRT |
| WGPU                         | **170/187**; coop-matrix paths under test; on-device C64 + native Gru/Rnn/Mamba2 / FftButterfly / act bwd |
| Apple CoreML / ANE (`Device::Ane`) | **164/187**; transformer block on-device (IR → MIL); Complex*/FftButterfly MIL; backward + on-device gradient training behind the `training` feature |
| Native Vulkan (`Device::Vulkan`) | **170/187**; broad SPIR-V set (norms/fused/RNN/vision-bwd/I8 quant/FFT butterfly) + host/unfuse for specialty ops; C64 arithmetic; parity on MoltenVK / lavapipe |
| Intel oneAPI (`Device::OneApi`) | **161/187**; OpenCL-C SPIR-V kernel set mirrors Vulkan depth when `RLX_ONEAPI_BUILD_KERNELS=1`; else CPU reference; DenseSolve stays HostOpDesc→LAPACK |
| Qualcomm Hexagon (QNN)       | Codegen + FFI runtime (`Device::Hexagon`): on-device INT8/INT4 `QMatMul`, `DequantMatMul`, `FusedAttentionBlock`, persistent session + context-binary save/load; x86 HTP functional sim (`just qnn-htp-sim`) |
| Cortex-M (INT8)              | Production: 96.6% MNIST on nRF52840 hardware  |
| FPGA                         | First-class SystemVerilog / bitstream export (`rlx_runtime::export`, `pyrlx.export_fpga`); Int8/Int4/Fp4, soft-port RTL + ECP5 / iCE40 / Xilinx7 synth |
| Reverse-mode AD              | Phase 1–9 complete; SelectiveScan, FusedTL    |
| Forward-mode AD (`jvp`/`hvp`)| Functional; thin public API                   |
| `vmap`                       | MVP — leading-axis batching                   |
| QAT (PTQ + STE + LSQ)        | Complete: EMA, Fixed, PerBatch, propagation; native FakeQuantize/LSQ on CUDA/ROCm/Vulkan/Metal/wgpu |
| OpKind coverage matrix       | **187** `OpKind`s; per-backend claims run **161–179**, generated from each backend's `SUPPORTED_OPS` — [`docs/op-coverage.md`](docs/op-coverage.md) (`just gen-op-coverage`, gated by `just check-op-coverage`) |
| Qwen3 LM (safetensors + GGUF)| End-to-end on Metal: 100% top-1 parity vs HF; matches/beats Python MPS on most prefill shapes. Q4_K_M GGUF loads + runs |
| Op::DequantMatMul GGUF schemes | All llama.cpp schemes (incl. Q4_1, Q5_0, Q5_1, IQ/TQ/MX). GPU dequant on Metal/CUDA/ROCm/WGPU (shared scheme ids 0–23); **Metal fused GEMV** for Q4_K, Q4_0/1, Q8_0, IQ4NL, IQ2/3/1 families (`m=1` prefill); WGPU grouped MoE GPU when scratch fits; ANE MIL constexpr for K/IQ/TQ/MX; TPU compile-time + runtime Param bake. **pyrlx:** `quantize`, `load_gguf`, `convert_to_gguf`. See [docs/gguf-backend-paths.md](docs/gguf-backend-paths.md). |
| Sampler chain                  | `SamplerChain` in `rlx-runtime::samplers`: Temperature, DynamicTemperature, TopK, TopP, TopNSigma, TypicalP, Mirostat v1/v2, XTC, DRY, RepetitionPenalty. Wired into `SampleOpts::into_chain()`; classic top-k/top-p stay on the fast path via `is_classic()`. |
| Quantized KV cache             | Per-layer K/V stored as q4_0 / q5_0 / q8_0 / f16 blocks via `rlx-runtime::quantized_kv`. Optional `mmap-kv` feature spills to a file-backed mapping for long contexts. |
| DiT modulation (`AdaLayerNorm` / `GatedResidual`) | First-class fused ops on every backend; fused **forward + backward** (packed reverse kernels), exact JVP + `vmap`; ONNX adaLN import fuses after param specialization. See [`docs/op-coverage.md`](docs/op-coverage.md). |
| Eigendecomposition (`Op::Eigh` / `EighBatch`) | Differentiable symmetric eigensolver on all backends: CPU-native (LAPACK `dsyevd` f64), native on-device CUDA/ROCm batched Jacobi (cuSOLVER / hipSOLVER `SsyevjBatched`, `n ≤ 32`), f64 host-fallback elsewhere. Full reverse mode (`EighBackward`). |
| SPD-manifold Riemannian primitives | `SpdKarcherMeanWeighted` / `SpdLogMap` / `SpdExpMap` / `SpdParallelTransport` / `SpdMatrixFnBatch` as first-class ops on all backends, fully differentiable (analytic Riemannian VJPs, incl. the base point). |
| StableHLO / IREE export         | `rlx-stablehlo` / `ExportTarget::StableHlo`: shared rlx-tpu lowering printed as StableHLO; NCHW kept; weights as args or a safetensors parameter file; int8 weights as `i8`; multi-length specializations. IREE round trip (llvm-cpu) matches CPU on MLP, conv, Llama and BERT layers. |
| Run / compile errors            | `CompiledGraph::try_run`, `Session::try_compile` / `try_new` return `IoError` / `CompileError` that name the input, node, or device; `run` panics with the same messages. See [`docs/troubleshooting.md`](docs/troubleshooting.md). |
| Static graph checker            | `rlx_runtime::check` folds shape/dtype verify + backend dispatch + missed-fusion hints + provable-NaN/Inf lint into one `CheckReport`; `cargo rlx check` (`--json`) and `#[rlx_model(check)]` (`RLX_CHECK`). |
| Weight-compute caching          | Compile-time `CompileOptions::param_bindings` (bake + broadcasting `ConstantFolding`) and run-time `cache_param_invariant` / `RLX_CACHE_PARAM_INVARIANT=1` (split prepare graph) compute weight-derived tensors once, not every forward. Validated CPU + CUDA. |
| Offline bake (`rlx-bake`)       | Merge graph + weights into a deployable `*.rlx` (skip-zero / ternary TQ2_0 / opt-in Q8_0, optional encrypt). CLI + crate; walkthrough [`docs/rlx-bake.md`](docs/rlx-bake.md). |
| Complex dtypes (`C64` / `C128`) | `DType::C64` arithmetic (add/sub/mul/div) + cast matrix on CPU and on-device CUDA / ROCm / Vulkan / wgpu / oneAPI; `DType::C128` cast plumbing (arithmetic remains C64-only). Parity tests per backend. |
| WideEP MoE + NCCL               | Host `moe_dispatch` / `moe_combine` + EPLB (`rlx-collectives::{moe_ep,eplb}`); CUDA NCCL scaffold for device-resident `all_reduce` / `all_to_all` (`rlx-cuda --features nccl`). Toy: `wide_ep_toy` / `nccl_all_reduce_toy`. See [`docs/distributed.md`](docs/distributed.md). |

## Authors

Eugene Hauptmann, Nataliya Kosmyna ([MIT-RLX](https://github.com/MIT-RLX)).

## Citing RLX

If you use RLX in academic work, please cite the paper
([arXiv:2609.37916](https://arxiv.org/abs/2609.37916)):

```bibtex
@misc{rlx2026,
  title         = {RLX: A Unified Multi-Backend Tensor Compiler and Distributed Runtime in Rust},
  author        = {Hauptmann, Eugene and Kosmyna, Nataliya},
  year          = {2026},
  eprint        = {2609.37916},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC},
  url           = {https://arxiv.org/abs/2609.37916}
}
```

To cite a specific version of the implementation instead, machine-readable
metadata is in [`CITATION.cff`](CITATION.cff); a plain-text form:

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
