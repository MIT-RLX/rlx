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
rlx = { version = "0.2", features = ["all-cpu"] }         # cpu + gguf + linalg
```

> **`mlx` and `rocm` features.** `rlx-mlx` and `rlx-rocm` aren't on
> crates.io (vendor-bundled submodule / workspace-relative kernel
> sources). Enabling those features on a crates.io build will fail
> to resolve. Use a git source instead:
>
> ```toml
> rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["apple-silicon", "mlx"] }
> ```

## Quickstart

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

Prefer to write the math? The `rlx!` DSL (feature `tensor`, on by default)
builds the same graph in a compact little language — `@` is matmul, shapes and
outputs are inferred:

```rust
use rlx::rlx;

let g = rlx! {
    input x: [1, 4];
    param w: [4, 2];
    let y = x @ w;      // → rlx_ir::Graph
};
```

### Multi-backend runtime

```rust
let mut runner = GraphDevices::with_policy(g, DevicePolicy::only([Device::Cpu, Device::Metal]));
let out = runner.run_resolved_with_inputs(None, &inputs)?;

let mut router = DeviceRouter::from_env(g)?;
let (device, out) = router.run(&inputs, None)?;
```

See [`docs/backend-selection.md`](../docs/backend-selection.md).

## Prelude + namespaces

| import                       | gives you                                                                |
|------------------------------|--------------------------------------------------------------------------|
| `use rlx::prelude::*;`       | `Graph`, `Session`, `CompileOptions`, `Package`, `GraphDevices`, `DeviceRouter`, `DevicePolicy`, `FlexibleSession`, `DType`, `Dim`, `Device`, `Result`, `Activation`, `BinaryOp`, `ReduceOp`, `FftNorm`, `QuantScheme`, `grad` / `jvp` / `vmap`, `verify`, `open_rlxp` / `compile_rlxp`, `rlx_model!`, `rlx!`, … |
| `use rlx::ops::*;`           | IR helper enums: `Activation`, `BinaryOp`, `CmpOp`, `MaskKind`, `ReduceOp`, `InterpMode`, `FftNorm`, `ChainStep`, `ChainOperand` |
| `use rlx::quant::*;`         | `QuantScheme`, `QuantMap`                                                |
| `use rlx::pkg::*;`           | `.rlxp` — `Package`, `Placement`, `open_rlxp`, `compile_rlxp`, `compile_rlxp_bind_params`, … |
| `use rlx::compile::*;`       | `CompileOptions`, `Precision`, `PrecisionPolicy`, fusion / pass helpers  |
| `use rlx::gguf::*;`          | GGUF parser + dequant (`gguf` feature)                                   |
| `use rlx::autodiff::*;`      | `grad`, `grad_with_loss`, `jvp`, `hvp`, `vmap`, `nth_order_grad`, …       |
| `use rlx::ir::…`             | full `rlx-ir` surface (everything the prelude doesn't lift)              |
| `use rlx::runtime::…`        | full `rlx-runtime` surface (backends, custom Session config)             |

With `features = ["optim"]`, the prelude also star-imports `Adam`, `AdamW`,
`Optimizer`, and the rest of `rlx-optim`.

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
| `gguf`     | GGUF v1 / v2 / v3 parser + dequant → `rlx::gguf`                  |
| `bench`    | uniform benchmark harness → `rlx::bench`                          |
| `sparse`   | sparse linear algebra (custom-op scaffold) → `rlx::sparse`        |
| `linalg`   | dense linalg via LAPACK (custom-op scaffold) → `rlx::linalg`      |
| `cortexm`  | INT8 ARMv7E-M kernels → `rlx::cortexm` (no `Backend` impl)        |
| `fpga`     | IR → SystemVerilog export → `rlx::fpga` + `rlx_runtime::export`   |

`cortexm` kernels and `fpga` export don't go through `Session` / `Backend`
execution — they're specialty / offline targets. FPGA RTL is **target-agnostic**
by default (`HwTarget::Generic`); see [`docs/fpga-export.md`](../../docs/fpga-export.md).

### Convenience aggregates

| feature           | expands to                              |
|-------------------|-----------------------------------------|
| `apple-silicon`   | `cpu` + `metal` + `blas-accelerate`     |
| `nvidia`          | `cpu` + `cuda`                          |
| `edge`            | `cpu` + `cortexm` + `fpga`              |
| `all-cpu`         | `cpu` + `gguf` + `linalg`               |

`mlx` and `rocm` aren't in any aggregate (vendor-bundled). To opt
in, add the feature explicitly to a git-source dep:

```toml
rlx = { git = "https://github.com/MIT-RLX/rlx", features = ["apple-silicon", "mlx"] }
```

## Documentation

- API reference: <https://docs.rs/rlx>
- Workspace overview + per-crate READMEs: <https://github.com/MIT-RLX/rlx>

## License

MIT OR Apache-2.0, at your option. See
[`LICENSE-MIT`](https://github.com/MIT-RLX/rlx/blob/main/LICENSE-MIT) and
[`LICENSE-APACHE`](https://github.com/MIT-RLX/rlx/blob/main/LICENSE-APACHE).
