# RLX

A small ML compiler and runtime for transformer inference and training.
JAX-shaped IR + autodiff + transforms (`jvp`, `hvp`, `vmap`) on top of
backend-specific kernels for CPU, Apple Silicon (Metal / MLX), NVIDIA
(CUDA), AMD (ROCm), Google TPU, cross-platform GPU (wgpu), and
microcontrollers (Cortex-M).

> Status: **0.2.12**, Apple-Silicon-first. The CPU and Apple GPU paths
> are mature; CUDA / ROCm / TPU / WGPU work but have seen less mileage;
> Cortex-M is a separate INT8 product. Multi-backend runtime helpers
> (`GraphDevices`, `DeviceRouter`) — see [`docs/backend-selection.md`](docs/backend-selection.md).
> **0.2.12** landed FIR / RIR / IIR digital filters + fused `Op::PartitionedConv`
> (batched-GEMM convolution reverb), a native Vulkan FFT compute kernel (`Op::Fft`
> on-device — fixes the discrete-GPU host-fallback crash), and parameterized
> `fNeXmY` minifloats (`ScaledFormat::Custom`) — see [`CHANGELOG.md`](CHANGELOG.md).
> **0.2.11** landed full GGUF IQ / TQ / MX backend parity, Metal fused IQ
> GEMV, `FusedAttentionBlock` on every inference backend (with native CUDA /
> Metal fused-attention kernels), and pyrlx GGUF load / save / convert — see
> [`docs/gguf-backend-paths.md`](docs/gguf-backend-paths.md) and [`CHANGELOG.md`](CHANGELOG.md).
> **0.2.10** added tensor-parallel collectives, native `Gru` / `Rnn` /
> `Mamba2`, and dynamic-shape specialization — see [`docs/op-coverage.md`](docs/op-coverage.md).

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
rlx = { version = "0.2", features = ["cpu"] }
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
| `gguf`    | GGUF v1 / v2 / v3 parser + dequant                       |
| `bench`   | uniform benchmark harness                                |
| `sparse`  | sparse linear algebra (custom-op scaffold)               |
| `linalg`  | dense linalg via LAPACK (custom-op scaffold)             |
| `splat`   | 3D Gaussian splatting (CPU reference render custom op)   |

### Specialty crates

The `Backend` model doesn't fit microcontrollers or hardware synthesis.
For those, depend on the standalone crates directly — they're not
exposed through the prelude:

- `rlx-cortexm` — `no_std` ARMv7E-M INT8 kernels.
- `rlx-fpga` — IR → SystemVerilog → bitstream.

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
```

Prefer NumPy-style expressions? The `rlx::tensor` DSL ([`rlx-tensor`](rlx-tensor/README.md))
builds the same IR with operator-overloaded, lazy `Tensor` handles —
`(&a + &b).relu()` traces instead of executing, then fuses + memory-plans
across any backend when you call `.to_vec()`:

```rust
use rlx_tensor::Tensor; // crate `rlx-tensor`, feature `eval`

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]);
let c = (&a + &Tensor::ones([3])).relu();
assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0]); // auto-picks the fastest backend
```

Domain-specific namespaces if you want narrower star-imports:
`rlx::ops::*` (IR helper enums), `rlx::quant::*`, `rlx::autodiff::*`.
Or the full per-crate surface
via `rlx::ir::…` / `rlx::opt::…` / `rlx::runtime::…` etc. — every
workspace crate is reachable as a module on `rlx`.

### FFT (0.2.2)

`Op::Fft` is a first-class IR primitive with CPU, Metal, MLX, CUDA,
ROCm, wgpu, and TPU lowering. Graph helpers in `rlx_ir::Graph` cover
real-input spectra and signal-processing workflows:

- `fft_real` / `rfft` / `irfft` — Hermitian `irfft` mirrors the conjugate half
- `fftfreq` / `rfftfreq` — sample-frequency constants
- `psd` / `psd_real` — power spectral density
- `stft`, `fft_conv1d` — short-time FFT (a single batched `rfft` over all frames)
  and frequency-domain convolution

Pow-2 **f32** transforms run native GPU kernels (CUDA / ROCm / Metal / wgpu);
non-pow2 uses Bluestein/chirp-z, and f64 / C64 run on the host CPU path (wgpu —
whose arena is f32-only — rejects them). The `native-gpu-fft` feature adds the
on-chip single-kernel radix-2/4/8 path (Metal / wgpu), CPU radix-4, and rayon
batch parallelism. Runtime toggles: `RLX_FFT_FAST`, `RLX_FFT_RADIX`,
`RLX_FFT_CPU_PARALLEL`, `RLX_FFT_RADIX4`, `RLX_FFT_FUSE_REAL`.

Benchmark one size across backends with
`cargo run -p rlx-bench --release --example bench_fft --features metal,gpu`, or
the full variant × precision × size × backend matrix (with per-backend CPU-parity
checks) via `bench_fft_matrix`. Python bindings: `pyrlx.Graph.fft`, `.rfft`,
`.irfft`, `.fftfreq` (see `crates/pyrlx/tests/test_fft.py`).


Or depend on each crate directly (`rlx-ir`, `rlx-opt`, `rlx-runtime`,
…) for the smallest possible dep tree.

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

Verified end-to-end at **cosine 1.000000** on both **CPU and CUDA** (RTX 3080 Ti)
across MLP, encoder-decoder, CNN, MNIST, LLaMA (rotary + GQA + causal mask), DINO
/ ViT, and the FLUX diffusion transformer (MMDiT). Runnable, per-model, per-form
examples live in
[`crates/bindings/pyrlx/examples/`](crates/bindings/pyrlx/examples/README.md)
(`python mlp.py`, `python llama.py`, …). Training is out of scope — `torch.export`
captures the inference graph; import the *forward* and use RLX's own autodiff +
[`rlx-optim`](crates/core/rlx-optim) for on-device training.

## Workspace layout

```
rlx            prelude — re-exports framework crates + common types
rlx-ir         leaf — types, shape, op enum, verifier, HIR hooks
rlx-tensor     NumPy-style symbolic Tensor DSL (lazy, trace → fuse → any backend)
rlx-flow       block assembly-line API for model builders
rlx-fusion     MIR fusion passes + unfuse for AD
rlx-autodiff   grad / jvp / hvp / vmap on MIR
rlx-compile    CompilePipeline, legalization, memory plan, precision
rlx-opt        facade — re-exports fusion + autodiff + compile
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
rlx-gguf       standalone GGUF parser + dequant (every llama.cpp scheme: Q4_0..Q8_0, Q2_K..Q8_K, IQ1..IQ4, TQ1/TQ2, MXFP4, NVFP4)
rlx-macros     #[rlx_model] AOT macro
rlx-bench      benchmark harness
rlx-sparse     downstream: CSR LU / mat-vec / CG (custom-op scaffold)
rlx-linalg     downstream: dense linalg via LAPACK (custom-op scaffold)
rlx-splat      downstream: 3D Gaussian splatting (self-contained; `rlx_splat::register()`)
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

For Apple Silicon, MLX is a git submodule under `rlx-mlx-sys`:

```sh
git submodule update --init rlx-mlx-sys/vendor/mlx
# or: git clone --recurse-submodules …
```

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
via [`CompileOptions::kernel_dispatch`](rlx-runtime/src/options.rs) and
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

[`supports_graph`](rlx-runtime/src/device_ext.rs) uses the backend
`supported_ops` claim set when a backend is registered, so device
picking stays aligned with compile rather than hand-maintained op tables.

More detail: [`rlx-ir/README-logical-kernels.md`](rlx-ir/README-logical-kernels.md)
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
  fusion patterns, and ideally a parity test in burnembed. Use
  `RLX_DISPATCH_REPORT=1` after compile to confirm native vs common-ir.
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
| Op::DequantMatMul GGUF schemes | All llama.cpp schemes (incl. Q4_1, Q5_0, Q5_1, IQ/TQ/MX). GPU dequant on Metal/CUDA/ROCm/WGPU (shared scheme ids 0–23); **Metal fused GEMV** for Q4_K, Q4_0/1, Q8_0, IQ4NL, IQ2/3/1 families (`m=1` prefill); WGPU grouped MoE GPU when scratch fits; ANE MIL constexpr for K/IQ/TQ/MX; TPU compile-time + runtime Param bake. **pyrlx:** `quantize`, `load_gguf`, `convert_to_gguf`. See [docs/gguf-backend-paths.md](docs/gguf-backend-paths.md). |
| Sampler chain                  | `SamplerChain` in `rlx-runtime::samplers`: Temperature, DynamicTemperature, TopK, TopP, TopNSigma, TypicalP, Mirostat v1/v2, XTC, DRY, RepetitionPenalty. Wired into `SampleOpts::into_chain()`; classic top-k/top-p stay on the fast path via `is_classic()`. |
| Quantized KV cache             | Per-layer K/V stored as q4_0 / q5_0 / q8_0 / f16 blocks via `rlx-runtime::quantized_kv`. Optional `mmap-kv` feature spills to a file-backed mapping for long contexts. |

## Authors

Eugene Hauptmann, Nataliya Kosmyna ([MIT-RLX](https://github.com/MIT-RLX)).

## Contributing

PRs welcome; the roadmap (`PLAN.md`) drives priorities. Per-crate
`README.md` files document build commands and gotchas; treat them as
the canonical "how does this crate work" reference.

## License

GPL-3.0-only. See [`LICENSE`](./LICENSE).
