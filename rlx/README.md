# rlx

A small ML compiler and runtime for transformer inference and training.
JAX-shaped IR + autodiff + transforms (`jvp`, `hvp`, `vmap`) on top of
backend-specific kernels for CPU, Apple Silicon (Metal / MLX), NVIDIA
(CUDA), AMD (ROCm), Google TPU, and cross-platform GPU (wgpu).

This is the **prelude crate** — pulls in `rlx-ir` / `rlx-opt` /
`rlx-runtime` and re-exports the common types. Most code only needs
one `use rlx::prelude::*;`.

## Install

```toml
[dependencies]
rlx = { version = "0.2", features = ["cpu"] }
```

For common platforms, single-flag aggregates compose the right
fragments:

```toml
rlx = { version = "0.2", features = ["apple-silicon"] }   # cpu + metal + Accelerate
rlx = { version = "0.2", features = ["nvidia"] }          # cpu + cuda
rlx = { version = "0.2", features = ["edge"] }            # cpu + cortexm
rlx = { version = "0.2", features = ["all-cpu"] }         # cpu + models + gguf + linalg
```

> **`mlx` and `rocm` features.** `rlx-mlx` and `rlx-rocm` aren't on
> crates.io (vendor-bundled submodule / workspace-relative kernel
> sources). Enabling those features on a crates.io build will fail
> to resolve. Use a git source instead:
>
> ```toml
> rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["apple-silicon", "mlx"] }
> ```

## Three usage patterns

### 1. Build + run a graph by hand

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

### 2. Run a model by name (`models` feature)

```rust,ignore
use rlx::prelude::*;

let mut runner = Qwen3Runner::builder()
    .weights("Qwen3-0.6B-Q4_K_M.gguf")   // safetensors OR gguf
    .device(Device::Metal)
    .max_seq(128)
    .build()?;
runner.generate(&prompt_ids, 32, |tok| print!(" {tok}"))?;
```

### 3. Plug your own runner into the dispatch surface (`models` feature)

```rust,ignore
use rlx::prelude::*;

struct WhisperRunner;
impl ModelRunner for WhisperRunner {
    fn name(&self) -> &'static str { "whisper" }
    fn description(&self) -> &'static str { "OpenAI Whisper" }
    fn run(&self, args: &[String]) -> Result<()> { /* … */ Ok(()) }
}

fn main() -> Result<()> {
    register_runner(Box::new(WhisperRunner));
    dispatch(&std::env::args().skip(1).collect::<Vec<_>>())
}
```

## Prelude + namespaces

| import                       | gives you                                                                |
|------------------------------|--------------------------------------------------------------------------|
| `use rlx::prelude::*;`       | `Graph`, `Session`, `DType`, `Device`, `Result`, `Activation`, `BinaryOp`, `Qwen3Runner`, `SamRunner`, `DinoV2Runner`, … |
| `use rlx::ops::*;`           | IR helper enums: `Activation`, `BinaryOp`, `CmpOp`, `MaskKind`, `ChainStep`, `ChainOperand` |
| `use rlx::quant::*;`         | `QuantScheme`, `QuantMap`                                                |
| `use rlx::weights::*;`       | `WeightLoader`, `WeightMap`, `GgufLoader`, HF↔GGUF name mappers (`models`) |
| `use rlx::run::*;`           | All runner builders + dispatch / plug-in registry (`models`)             |
| `use rlx::autodiff::*;`      | `jvp`, `hvp`, `vmap`                                                     |
| `use rlx::ir::…`             | full `rlx-ir` surface (everything the prelude doesn't lift)              |
| `use rlx::runtime::…`        | full `rlx-runtime` surface (backends, custom Session config)             |

`rlx::Result<T>` and `rlx::Error` are aliases of `anyhow::Result<T>`
and `anyhow::Error` — the whole stack returns those.

## Feature matrix

### Backends

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

### Companion crates

Off by default; turn on per workload:

| feature    | what                                                              |
|------------|-------------------------------------------------------------------|
| `models`   | BERT / Nomic / vision graph builders → `rlx::models`              |
| `gguf`     | GGUF v1 / v2 / v3 parser + dequant → `rlx::gguf`                  |
| `bench`    | uniform benchmark harness → `rlx::bench`                          |
| `sparse`   | sparse linear algebra (custom-op scaffold) → `rlx::sparse`        |
| `linalg`   | dense linalg via LAPACK (custom-op scaffold) → `rlx::linalg`      |
| `cortexm`  | INT8 ARMv7E-M kernels → `rlx::cortexm` (no `Backend` impl)        |
| `fpga`     | IR → SystemVerilog datapath synthesis → `rlx::fpga` (no `Backend`)|

`cortexm` and `fpga` don't go through the `Session` / `Backend`
pipeline — they're specialty targets exposed for direct use.

### Convenience aggregates

| feature           | expands to                              |
|-------------------|-----------------------------------------|
| `apple-silicon`   | `cpu` + `metal` + `blas-accelerate`     |
| `nvidia`          | `cpu` + `cuda`                          |
| `edge`            | `cpu` + `cortexm`                       |
| `all-cpu`         | `cpu` + `models` + `gguf` + `linalg`    |

`mlx` and `rocm` aren't in any aggregate (vendor-bundled). To opt
in, add the feature explicitly to a git-source dep:

```toml
rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["apple-silicon", "mlx"] }
```

## Documentation

- API reference: <https://docs.rs/rlx>
- Workspace overview + per-crate READMEs: <https://github.com/MIT-RLX/rlx>

## License

GPL-3.0-only. See [`LICENSE`](https://github.com/MIT-RLX/rlx/blob/main/LICENSE).
