# rlx-runtime

User-facing API for RLX — `Session::new(Device).compile(graph)` →
`CompiledGraph`, which holds the executable, the arena, the weights,
and the device handle.

## What's here

- **`Session`** — entry point; selects a backend via `Device`.
- **`CompiledGraph`** (`compiled.rs`) — `run` / `run_typed` / `set_param`;
  `inputs()` / `check_bindings` describe and check the boundary. Zero
  allocation per call.
- **`Backend` trait** + `ExecutableGraph` — every backend (CPU, Metal,
  MLX, CUDA, ROCm, wgpu, TPU) implements these. Every backend declares
  its supported `OpKind`s, and `legalize_for_backend` rejects
  unsupported graphs at compile time.
- **`registry.rs` / `op_registry.rs`** — backend factory + per-op
  registration plumbing for downstream extension.
- **`Device`** lives in `rlx-driver::device`; this crate just consumes
  it. Variants: `Cpu`, `Metal`, `Mlx`, `Ane`, `Cuda`, `Rocm`, `Xdna`,
  `Egpu`, `OneApi`, `Tpu`, `Hexagon`, `Gpu` (wgpu), `Vulkan`, `OpenGl`,
  `DirectX`, `WebGpu`.
- **`device_ext.rs`** — `Device::is_available()` lookup against the
  registry (keeps the runtime→driver dep direction one-way).
- **`hwinfo.rs`** — host + GPU introspection. `HwSnapshot` (CPU topology /
  caches / Apple thermal state) plus live **GPU telemetry** and **control**:
  `device_thermal(device, index)` / `all_gpu_thermal()` read
  temperature / power / clock / fan across CUDA & ROCm (unprivileged), and
  `set_power_cap` / `set_locked_clocks` / `set_fan_percent` (+ resets,
  `power_cap_range`) are root-only controls returning a typed `ThermalError`.
  Drives the `rlx-gpu` CLI (in `rlx-bench`).
- **`arena.rs`** — device-side arena buffer.
- **`CompileCache`** (`compile_cache.rs`) — graph-fingerprint →
  compiled-artifact cache.
- **Control flow** — `Op::While` runs as a loop, unbounded or not: natively
  on the CPU, through the one-op host fallback on Metal / CUDA / ROCm / wgpu;
  other backends unroll it. `subgraph.rs` compiles nested bodies lazily and
  caches them.
- **`PrecisionPolicy`** — re-export from `rlx-opt`. AMP / always-f16 /
  always-f32 / always-bf16.
- **`trace.rs`** — runtime tracing (verbose env-gated).
- **`cost.rs`** — heterogeneous cost model that picks Cpu vs. Metal vs.
  MLX per graph; CUDA / ROCm / wgpu use on-disk calibration caches.
- **Multi-backend runtime** — `GraphDevices`, `DeviceRouter`,
  `FlexibleSession`, `DevicePolicy`, `BackendsManifest`. See
  [`docs/backend-selection.md`](../../../docs/backend-selection.md).
- **FFT dispatch** — `Op::Fft` on CPU / Metal / MLX / CUDA / ROCm / wgpu /
  TPU. Pow-2 f32 uses native GPU kernels where available; other shapes and
  dtypes use partial host sync. Graph helpers (`rfft`, `irfft`, `stft`, …)
  live in `rlx_ir::ops::fft_ops`.
- **`stream.rs`** — async command stream (Metal-side; CPU is sync).
- **`paged_kv`** — paged KV cache + continuous batching primitives.
- **`quantized_kv`** — block-quantized KV cache (q4_0 / q5_0 / q8_0 / f16
  per-layer). Cuts decode-history memory ~2–4× vs f16. Opt-in mmap path
  behind the `mmap-kv` feature spills the K/V buffers to a file-backed
  mapping so the kernel pages cold history in/out on demand — useful
  for 100k-token decodes that exceed RAM.
- **`samplers`** — backend-agnostic sampler chain. Implements Temperature,
  DynamicTemperature, TopK, TopP, TopNSigma, TypicalP, Mirostat v1 / v2,
  XTC, DRY, RepetitionPenalty. Compose via [`SamplerChain::builder()`] or
  via the flat fields on [`SampleOpts`] — `SampleOpts::into_chain()` builds
  the canonical llama.cpp-ordered chain from the configured fields. Use
  [`SampleOpts::is_classic()`] to short-circuit the fast path when only
  legacy top-k/top-p/temperature are configured.
- **In-graph RNG** — [`Op::RngNormal`] / [`Op::RngUniform`] (ONNX
  `Random*`) with [`CompileOptions::rng`] and runtime
  [`CompiledGraph::set_rng`]. Backends: Philox (default), Ort (CPU parity),
  Zero (deterministic tests). CUDA / ROCm / Metal / wgpu generate on device
  (Philox); TPU lowers to XLA `rng`.
- **`model`** *(feature `model-io`)* — one interface over whatever runs a
  checkpoint: `model::open(request)` returns a `Model` (named typed tensors in
  and out, plus `generate_text`, `chat`, `transcribe`, `synthesize`, `embed`
  where they apply). `ModelProvider`s register statically or at run time; model
  plugins (`model-plugins`, or `linked-plugins` on iOS / Android / wasm) load a
  separately built collection through a versioned C vtable. `.rlxp` packages
  open the same way (`pkg.rs`), and `model::trainer::open_trainer` trains a package's
  functions.
- **`program`** *(feature `program`)* — a package's host logic (tokenize, a
  decode loop) as a sandboxed WebAssembly module (`rlx-program`).
- **`debug`** — `debug_graph(graph, &DebugOptions)` compiles and runs a graph
  on a device and on the CPU, classifies the failure (`unsupported-op`,
  `run-panic`, `non-finite`, `mismatch`, …) and probes which compile option
  gets past it.
- **`export`** — artifacts instead of a run: `ExportTarget::Fpga` (feature
  `fpga`) and `ExportTarget::StableHlo` (feature `stablehlo`, for IREE).
- **`placement` / `job_slots`** — measured placement (time a few real steps on
  the cost model's GPU pick and the CPU, keep the GPU only at a 1.5× margin;
  `RLX_PLACEMENT_CACHE` remembers decisions), GPU sharing between concurrent
  jobs (`RLX_GPU_JOBS`), and a pool bounding concurrent work by count and
  memory.
- **Number formats** — `CompileOptions::number_format` runs an f32 graph as
  f64, double-single or a narrower float format (see `rlx-compile`).

Re-exports: `Tick`, `time_ns` from `rlx_ir::measure`. Use these for any
sub-ms timing in the user-facing layer.

## Cargo features

| feature             | backend                              |
|---------------------|--------------------------------------|
| `cpu` *(default)*   | `rlx-cpu`                            |
| `metal`             | `rlx-metal` (macOS)                  |
| `mlx`               | `rlx-mlx` (macOS)                    |
| `gpu`               | `rlx-wgpu` (cross-platform)          |
| `cuda`              | `rlx-cuda`                           |
| `rocm`              | `rlx-rocm`                           |
| `tpu`               | `rlx-tpu`                            |
| `vulkan`            | `rlx-vulkan` (native Vulkan)         |
| `coreml` / `ane`    | `rlx-coreml` (Apple Neural Engine)   |
| `apple`             | `metal` + `mlx` + `gpu` + `coreml`   |
| `oneapi` / `qnn` / `xdna` / `egpu` | Intel GPU / Qualcomm / AMD NPU / eGPU backends |
| `opengl` / `directx` / `webgpu` / `browser` | wgpu sub-targets |
| `fpga` / `stablehlo` | `export` targets (RTL, StableHLO MLIR) |
| `training`          | autodiff + optimizers                |
| `model-io`          | `model` (packages, providers)        |
| `model-plugins` / `linked-plugins` | load / link model plugins |
| `program`           | WebAssembly package programs (wasmi) |
| `blas-accelerate`   | macOS Accelerate                     |
| `blas-mkl`          | Intel MKL                            |
| `blas-openblas`     | OpenBLAS                             |
| `cpu-wave-exec`     | `rlx-cpu`'s inter-thunk parallelism  |
| `assoc-scan`        | log-depth affine `Scan` rewrite      |
| `mmap-kv`           | file-backed quantized KV (memmap2)   |

## Install

```toml
[dependencies]
rlx-runtime = { version = "0.2.19", features = ["cpu"] }
```

> **Heads-up.** The `mlx` and `rocm` features pull in `rlx-mlx` and
> `rlx-rocm`, which **aren't on crates.io** (workspace-relative submodule /
> kernel-source paths). Enabling those features on a crates.io build of
> `rlx-runtime` may fail to resolve. Use a git source on the whole workspace
> instead:
>
> ```toml
> rlx-runtime = { git = "https://github.com/MIT-RLX/rlx", features = ["mlx"] }
> ```

Most users want the [`rlx`](https://crates.io/crates/rlx) prelude
crate; it re-exports `rlx_runtime::Session` and friends at the top
level.

## Quickstart

```rust
use rlx_ir::{DType, Graph, Shape};
use rlx_runtime::{Device, Session};

let mut g = Graph::new("hello");
let x = g.input("x", Shape::new(&[1, 4], DType::F32));
let w = g.param("w", Shape::new(&[4, 2], DType::F32));
let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
g.set_outputs(vec![y]);

let mut compiled = Session::new(Device::Cpu).compile(g);
compiled.set_param("w", &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
let out = compiled.run(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
```

### Errors instead of panics

```rust
use rlx_runtime::{CompileError, IoError, Session, Device};

let session = Session::try_new(Device::Metal)?;          // CompileError::DeviceUnavailable
let mut compiled = session.try_compile(g)?;              // Invalid / Unsupported / Backend
println!("{:?}", compiled.inputs());                     // name, dtype, static shape
let out = compiled.try_run(&[("x", &x)])?;               // IoError: unknown name, wrong length
compiled.check_bindings(&["x"])?;                        // inputs / params left unbound
```

What each error means and what to do: [`docs/troubleshooting.md`](../../../docs/troubleshooting.md).

### Multi-backend (optional)

```rust
use rlx_runtime::{DevicePolicy, GraphDevices, DeviceRouter};

let mut runner = GraphDevices::with_policy(g, DevicePolicy::only([Device::Cpu, Device::Metal]));
runner.set_param("w", &weights);
let out = runner.run_resolved_with_inputs(None, &inputs)?;

let mut router = DeviceRouter::from_env(g)?;
let (device, out) = router.run(&inputs, None)?;
```

Full guide: [`docs/backend-selection.md`](../../../docs/backend-selection.md).

## Build / test

```sh
cargo build -p rlx-runtime --features cpu                       # CPU only
cargo build -p rlx-runtime --features cpu,metal                 # +Metal
cargo test  -p rlx-runtime --release
```

## Gotchas

- Backend selection is feature-gated. `--features metal` is mandatory
  to instantiate `Device::Metal`; otherwise `Session::new(Metal)` panics
  (`Session::try_new` returns the reason). Same for `cuda`, `rocm`, `mlx`, `gpu`.
- `set_param` / `run` take `&[f32]` of the declared shape's element count; a
  mismatch, or an input name the graph does not declare, panics with the name
  and shape (`try_run` returns it). A param passed to `run` is uploaded as if
  by `set_param`.
- Inputs never supplied and params never set read zeros; the first `run`
  names them on stderr once. `allow_unbound(&[…])` declares the ones meant
  to read zeros (an empty KV state).
- Compile cache key includes the graph fingerprint **and** the precision
  policy — bumping precision invalidates entries.
- For long-running serving paths, prefer `CompileCache` over recompiling
  per request.

## License

MIT OR Apache-2.0.
