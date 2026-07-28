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
  --features cpu,blas-accelerate,metal,mlx,gpu

# Linux + NVIDIA
maturin develop --release \
  --features cpu,cuda

# Linux + AMD
maturin develop --release \
  --features cpu,rocm

# Cross-platform GPU only (Vulkan / DX12 / WebGPU via wgpu)
maturin develop --release \
  --features cpu,gpu
```

## Behavior contract

- `Session(device="metal")` raises `RuntimeError` if `metal` wasn't
  compiled in — the message names the cargo feature to enable.
- The same graph + same inputs across two backends produces the
  *same* output up to numerical precision. See
  `examples/cross_backend_parity.py` for the canonical check.
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

## Multi-backend runtime (0.2.3+)

After `maturin develop`, Python exposes the same multi-backend surface as Rust:

| Class | Role |
|-------|------|
| `DevicePolicy` | Allow / deny / prefer backends; `from_env()` reads `RLX_*` |
| `GraphDevices` | Lazy per-device compile cache; `run`, `run_chain`, `run_resolved` |
| `FlexibleSession` | Defer backend until `compile_resolved` |
| `DeviceRouter` | Warm-all on init; serving + fallback chain |

```python
import json, pyrlx as rlx

print(json.loads(rlx.backends_manifest()))
runner = rlx.GraphDevices(g, policy=rlx.DevicePolicy.only(["cpu", "metal"]))
router = rlx.DeviceRouter(g, policy=rlx.DevicePolicy.from_env())
```

Full reference: [`docs/backend-selection.md`](../../docs/backend-selection.md).

Tests: `pytest tests/test_graph_devices.py`.

## GGUF cargo features

| Feature | Purpose |
|---------|---------|
| `gguf-convert` (default) | `convert_to_gguf` via `rlx-gguf-convert` (safetensors) |
| `gguf-onnx` | ONNX checkpoint import for `convert_to_gguf` |
| `gguf-pt` | PyTorch `.pt` / `.pth` import for `convert_to_gguf` |

Pack/unpack and file I/O need no compute backend:

```python
import pyrlx as rlx

packed = rlx.quantize(weights, dtype="Q4_K")
f = rlx.load_gguf("model.gguf")
w = f.dequant_tensor("token_embd.weight")
rlx.convert_to_gguf("in.safetensors", "out.gguf", "Q4_K", architecture="llama")
```

Tests: `tests/test_gguf_quantize.py`, `test_gguf_file.py`, `test_gguf_convert.py`.
Runtime dequant on Metal / CUDA / WGPU: [docs/gguf-backend-paths.md](../../docs/gguf-backend-paths.md).

## License

MIT OR Apache-2.0.
