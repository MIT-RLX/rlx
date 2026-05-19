# RLX

A small ML compiler and runtime for transformer inference and training.
JAX-shaped IR + autodiff + transforms (`jvp`, `hvp`, `vmap`) on top of
backend-specific kernels for CPU, Apple Silicon (Metal / MLX), NVIDIA
(CUDA), AMD (ROCm), Google TPU, cross-platform GPU (wgpu), and
microcontrollers (Cortex-M).

> Status: **0.1.0**, Apple-Silicon-first. The CPU and Apple GPU paths
> are mature; CUDA / ROCm / TPU / WGPU work but have seen less mileage;
> Cortex-M is a separate INT8 product.

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

## Install

The `rlx` prelude crate is the recommended entry point — it pulls in
the IR, optimizer, runtime, and re-exports the common types:

```toml
[dependencies]
rlx = { version = "0.1", features = ["cpu"] }
```

For Apple Silicon GPU acceleration (note: `mlx` is git-only for 0.1.0
— see below):

```toml
rlx = { version = "0.1", features = ["cpu", "metal"] }
```

> **`mlx` and `rocm` aren't on crates.io for 0.1.0** (workspace-
> relative submodule / kernel-source paths). For those features, use
> the workspace git tree:
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
| `gpu`               | wgpu (Vulkan / DX12 / WebGPU / Metal)| cross-platform            |
| `cuda`              | cuBLAS / cuDNN / NVRTC               | Linux / Windows + NVIDIA  |
| `rocm`              | hipBLAS / MIOpen                     | Linux + AMD               |
| `tpu`               | libtpu PJRT plugin                   | Linux + GCP TPU           |
| `blas-accelerate`   | macOS Accelerate                     | macOS                     |
| `blas-mkl`          | Intel MKL                            | Intel / AMD CPUs          |
| `blas-openblas`     | OpenBLAS                             | cross-platform CPU        |

### Companion crate features

Off by default; enable per workload:

| feature   | what                                                     |
|-----------|----------------------------------------------------------|
| `models`  | BERT / Nomic / vision graph builders                     |
| `gguf`    | GGUF v1 / v2 / v3 parser + dequant                       |
| `bench`   | uniform benchmark harness                                |
| `sparse`  | sparse linear algebra (custom-op scaffold)               |
| `linalg`  | dense linalg via LAPACK (custom-op scaffold)             |

### Specialty crates

The `Backend` model doesn't fit microcontrollers or hardware synthesis.
For those, depend on the standalone crates directly — they're not
exposed through the prelude:

- `rlx-cortexm` — `no_std` ARMv7E-M INT8 kernels.
- `rlx-fpga` — IR → SystemVerilog → bitstream.

## Quickstart

A single `use rlx::prelude::*;` covers the common surface: graph
types, `Session`, `Device`, ops + activations, `Result`, the model
runners (Qwen3 / SAM / DINOv2) under the `models` feature.

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
```

Domain-specific namespaces if you want narrower star-imports:
`rlx::ops::*` (IR helper enums), `rlx::quant::*`, `rlx::weights::*`,
`rlx::autodiff::*`, `rlx::run::*`. Or the full per-crate surface
via `rlx::ir::…` / `rlx::opt::…` / `rlx::runtime::…` etc. — every
workspace crate is reachable as a module on `rlx`.

Or depend on each crate directly (`rlx-ir`, `rlx-opt`, `rlx-runtime`,
…) for the smallest possible dep tree.

### Running a model (Qwen3, SAM 1/2/3) in one call

With the `models` feature, the `rlx::run` namespace exposes a
builder-style runner. Format / config / dtype histogram are
auto-detected from the weights file:

```rust
use rlx::run::{Qwen3Runner, Qwen3Precision};
use rlx::Device;

let mut runner = Qwen3Runner::builder()
    .weights("Qwen3-0.6B-Q4_K_M.gguf")  // .safetensors or .gguf
    .device(Device::Metal)
    .max_seq(128)
    .precision(Qwen3Precision::F16LmHead)  // optional; default F32
    .max_memory_gb(16.0)                    // soft cap
    .stream(true)
    .build()?;

runner.generate(&prompt_ids, 32, |tok| print!(" {tok}"))?;
```

The same surface is wired into a `rlx-run` CLI:

```sh
cargo run --release -p rlx-models --bin rlx-run --features metal -- \
    qwen3 --weights model.gguf --device metal --max-tokens 32 \
          --max-seq 128 --prompt-ids 1,17,42,314
cargo run --release -p rlx-models --bin rlx-run -- inspect model.gguf
```

## Workspace layout

```
rlx            prelude — re-exports framework crates + common types
rlx-ir         leaf — types, shape, op enum, autodiff hooks
rlx-opt        graph rewrites: fusion, precision, memory plan, AD, vmap
rlx-driver     Device enum + cross-cutting types
rlx-cpu        CPU kernels (NEON / AVX / Accelerate / OpenBLAS)
rlx-metal      Apple Metal native (MSL + MPSGraph + ICB)
rlx-mlx        Apple MLX (vendored, hand-rolled C++ shim)
rlx-cuda       NVIDIA CUDA (cuBLAS + cuDNN + NVRTC + Graphs)
rlx-rocm       AMD ROCm/HIP (hipBLAS + MIOpen + hipGraph)
rlx-tpu        Google TPU via libtpu PJRT
rlx-wgpu       Cross-platform GPU via wgpu
rlx-cortexm    ARMv7E-M INT8 kernels (no_std)
rlx-fpga       IR → Verilog → bitstream
rlx-runtime    user-facing Session / CompiledGraph
rlx-models     concrete model graph builders (BERT, Nomic, vision, Qwen3, SAM/2/3, DINOv2)
rlx-gguf       standalone GGUF parser + dequant (incl. Q4_K / Q5_K / Q6_K / Q8_K)
rlx-macros     #[rlx_model] AOT macro
rlx-bench      benchmark harness
rlx-sparse     downstream: CSR LU / mat-vec / CG (custom-op scaffold)
rlx-linalg     downstream: dense linalg via LAPACK (custom-op scaffold)
pyrlx          Python bindings via PyO3
```

Each crate has its own `README.md` covering public surface, build
commands, and internal gotchas.

## Building from source

```sh
cargo build --release                         # cpu only
cargo build --release --features metal,mlx    # apple silicon GPU
cargo test  --release --workspace             # 865 tests
```

For Apple Silicon, MLX is vendored as a git submodule:

```sh
git submodule update --init
```

## Development workflow

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
  fusion patterns, and ideally a parity test in burnembed.
- **Bench every change in burnembed.** The integration testbed at
  `/Users/Shared/burnembed` is the canonical bench loop:
  `cargo run --release --example bench_rlx_single --features ndarray,blas-accelerate,rlx,hf-download -- --model minilm6`.
  Models pulled live from HF.
- **PLAN.md** drives priorities; the `## Landed` section at the bottom
  tracks what's already in tree, with bench deltas. PRs targeting plan
  items are expected to add a delta line — even "within noise" is data
  worth recording.

Recent phases (from git log) — A → J: dtype dispatch, AutoMixed
precision, cast-tax elimination, segmented ICB, f16 reduction kernels,
MPSGraph extension. K → L: rlx-cuda full stack (cuBLAS/Lt + cuDNN +
WMMA + CUDA Graphs + multi-stream + mixed-precision GemmEx + NVRTC
disk cache + NVTX), followed by rlx-rocm sister crate at parity.

## Versioning

Pre-1.0; `0.x` minor bumps may include breaking IR changes. The `Op`
enum and the `Graph` builder API in particular are still evolving as
new ops land. Pin exact versions in production until 1.0.

## Status by area

| Area                         | State                                         |
|------------------------------|-----------------------------------------------|
| CPU forward + backward       | Mature; 26 unit tests + integration suites    |
| Apple Metal forward          | Mature; 78-warning third-party noise silenced |
| Apple MLX forward + backward | Mature; tier-1/2/3 backward parity            |
| NVIDIA CUDA                  | Functional; less battle-tested                |
| AMD ROCm                     | Sister-crate parity to CUDA                   |
| TPU                          | Real-model E2E parity (MiniLM-L6) via PJRT    |
| WGPU                         | Functional; coop-matrix paths under test      |
| Cortex-M (INT8)              | Production: 96.6% MNIST on nRF52840 hardware  |
| FPGA                         | Per-graph datapath + bitstream emit           |
| Reverse-mode AD              | Phase 1–9 complete; SelectiveScan, FusedTL    |
| Forward-mode AD (`jvp`/`hvp`)| Functional; thin public API                   |
| `vmap`                       | MVP — leading-axis batching                   |
| QAT (PTQ + STE + LSQ)        | Complete: EMA, Fixed, PerBatch, propagation   |
| Qwen3 LM (safetensors + GGUF)| End-to-end on Metal: 100% top-1 parity vs HF; matches/beats Python MPS on most prefill shapes. Q4_K_M GGUF loads + runs |
| Op::DequantMatMul GGUF schemes | CPU: Q4_K / Q5_K / Q6_K / Q8_K supported (dequant scratch + sgemm — keeps arena packed). Metal: TBD; the per-op thunk path dequants to F32 once at load |
| `rlx::run` runner DX | Qwen3 + SAM 1/2/3 builders with autodetected weight format, device pick, memory ceiling, streaming. `rlx-run` CLI tool ships in `rlx-models` |

## Authors

Eugene Hauptmann, Nataliya Kosmyna ([MIT-RLX](https://github.com/MIT-RLX)).

## Contributing

PRs welcome; the roadmap (`PLAN.md`) drives priorities. Per-crate
`README.md` files document build commands and gotchas; treat them as
the canonical "how does this crate work" reference.

## License

GPL-3.0-only. See [`LICENSE`](./LICENSE).
