# Backends

`pyrlx` exposes every RLX backend through one Python string. Backend
*availability* is set at compile time via cargo features that pass
straight through to `rlx-runtime`.

## Cargo feature → device string

| `--features ...`                    | `device=` string  | What runs                            |
| ----------------------------------- | ----------------- | ------------------------------------ |
| `cpu`                               | `"cpu"`           | NEON/AVX SIMD + thread pool          |
| `cpu,blas-accelerate`               | `"cpu"`           | + Apple Accelerate (AMX-aware SGEMM) |
| `cpu,blas-mkl`                      | `"cpu"`           | + Intel MKL                          |
| `cpu,blas-openblas`                 | `"cpu"`           | + OpenBLAS                           |
| `metal`                             | `"metal"`         | Apple Metal native (MPS + MSL)       |
| `mlx`                               | `"mlx"`           | Apple MLX (lazy graph)               |
| `cuda`                              | `"cuda"`          | NVIDIA cuBLAS + cuDNN + WMMA         |
| `rocm`                              | `"rocm"`          | AMD hipBLAS + MIOpen + hipGraph      |
| `gpu`                               | `"gpu"` / `"wgpu"`| Cross-platform via wgpu              |

> **Picking BLAS:** at most one of `blas-accelerate` / `blas-mkl` /
> `blas-openblas`. They're mutually exclusive at link time.

## Build matrix

```sh
# Apple Silicon — everything that fits
maturin develop --release \
  --features cpu,blas-accelerate,metal,mlx,gpu,embed,hf-download

# Linux + NVIDIA
maturin develop --release \
  --features cpu,cuda,embed,hf-download

# Linux + AMD
maturin develop --release \
  --features cpu,rocm,embed,hf-download

# Cross-platform GPU only (Vulkan / DX12 / WebGPU via wgpu)
maturin develop --release \
  --features cpu,gpu,embed,hf-download
```

## Behavior contract

- `Session(device="metal")` raises `RuntimeError` if `metal` wasn't
  compiled in — the message names the cargo feature to enable.
- The same graph + same inputs across two backends produces the
  *same* output up to numerical precision. See
  `pyrlx/examples/cross_backend_parity.py` for the canonical check.
- `Session(precision="f16")` requests reduced-precision compute;
  backends that don't support the requested precision fall back to
  F32 silently (this matches the Rust contract).
- The runtime registry is per-process. Calling
  `pyrlx.available_devices()` after construction is fine; backends
  register at first use, not at import.

## Aliases

| You write          | Maps to            |
| ------------------ | ------------------ |
| `"nvidia"`         | `"cuda"`           |
| `"amd"` / `"hip"`  | `"rocm"`           |
| `"wgpu"`           | `"gpu"`            |
| `"vk"`             | `"vulkan"`         |
| `"dx12"` / `"d3d12"` | `"directx"`      |
| `"mtl"`            | `"metal"`          |
