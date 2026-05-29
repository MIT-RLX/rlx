# rlx-runtime

User-facing API for RLX — `Session::new(Device).compile(graph)` →
`CompiledGraph`, which holds the executable, the arena, the weights,
and the device handle.

## What's here

- **`Session`** — entry point; selects a backend via `Device`.
- **`CompiledGraph`** (`compiled.rs`) — `run` / `set_param` /
  `set_input`. Zero allocation per call.
- **`Backend` trait** + `ExecutableGraph` — every backend (CPU, Metal,
  MLX, CUDA, ROCm, wgpu, TPU) implements these. Every backend declares
  its supported `OpKind`s, and `legalize_for_backend` rejects
  unsupported graphs at compile time.
- **`registry.rs` / `op_registry.rs`** — backend factory + per-op
  registration plumbing for downstream extension.
- **`Device`** lives in `rlx-driver::device`; this crate just consumes
  it. Variants: `Cpu`, `Metal`, `Mlx`, `Ane`, `Cuda`, `Rocm`, `Tpu`,
  `Gpu` (wgpu), `Vulkan`, `OpenGl`, `DirectX`, `WebGpu`.
- **`device_ext.rs`** — `Device::is_available()` lookup against the
  registry (keeps the runtime→driver dep direction one-way).
- **`weights.rs`** — `WeightLoader` trait + `BytesWeightLoader`. Promote
  to registry per plan #24 / #56.
- **`arena.rs`** — device-side arena buffer.
- **`CompileCache`** (`compile_cache.rs`) — graph-fingerprint →
  compiled-artifact cache.
- **`subgraph.rs`** — `run_if` / `run_while` helpers; the IR has
  If/While ops but executor wiring is pending (see Op::If/While
  docstring).
- **`PrecisionPolicy`** — re-export from `rlx-opt`. AMP / always-f16 /
  always-f32 / always-bf16.
- **`trace.rs`** — runtime tracing (verbose env-gated).
- **`cost.rs`** — heterogeneous cost model that picks Cpu vs. Metal vs.
  MLX per graph.
- **FFT dispatch** — `Op::Fft` on CPU / Metal / MLX / CUDA / ROCm / wgpu /
  TPU. Pow-2 f32 uses native GPU kernels where available; other shapes and
  dtypes use partial host sync. Graph helpers (`rfft`, `irfft`, `stft`, …)
  live in `rlx_ir::ops::fft_ops`.
- **`stream.rs`** — async command stream (Metal-side; CPU is sync).
- **`paged_kv`** — paged KV cache + continuous batching primitives.

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
| `blas-accelerate`   | macOS Accelerate                     |
| `blas-mkl`          | Intel MKL                            |
| `blas-openblas`     | OpenBLAS                             |

## Install

```toml
[dependencies]
rlx-runtime = { version = "0.2", features = ["cpu"] }
```

> **Heads-up.** The `mlx` and `rocm` features pull in `rlx-mlx` and
> `rlx-rocm`, which **aren't on crates.io for 0.1.0** (workspace-
> relative submodule / kernel-source paths). Enabling those features
> on a crates.io build of `rlx-runtime` will fail to resolve. Use a
> git source on the whole workspace instead:
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

## Build / test

```sh
cargo build -p rlx-runtime --features cpu                       # CPU only
cargo build -p rlx-runtime --features cpu,metal                 # +Metal
cargo test  -p rlx-runtime --release
```

## Gotchas

- Backend selection is feature-gated. `--features metal` is mandatory
  to instantiate `Device::Metal`; otherwise `Session::new(Metal)` panics
  at registry lookup. Same applies to `cuda`, `rocm`, `mlx`, `wgpu`.
- `set_param` accepts `&[f32]` of the declared shape's element count.
  Mismatched len is a runtime panic, not a compile-time error.
- Compile cache key includes the graph fingerprint **and** the precision
  policy — bumping precision invalidates entries.
- For long-running serving paths, prefer `CompileCache` over recompiling
  per request.

## License

GPL-3.0-only.
