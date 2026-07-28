# MLX weight loading

RLX can consume MLX / mlx-lm weight layouts without going through Python.

## Supported inputs

| Path | Notes |
|------|--------|
| mlx-community directory | `config.json` + `model*.safetensors` (sharded index OK) |
| `.safetensors` | Single file; optional sibling `config.json` |
| `.npz` / `.npy` | `mx.savez` / `mx.save` / `nn.Module.save_weights` |

## Quantization

`config.json` → `quantization.mode`:

| Mode | `QuantScheme` | Default group |
|------|---------------|---------------|
| `affine` | `MlxAffine { bits, group_size }` | 64 / 4-bit |
| `mxfp4` | `MlxMxfp4 { group_size }` | 32 |
| `nvfp4` | `MlxMxfp4 { group_size }` (same nibble pack; FP8 E4M3 scales) | 16 |
| `mxfp8` | `MlxMxfp8 { group_size }` | 32 |

mlx-lm `nvfp4` is **not** GGUF/NVIDIA `Nvfp4Block` (different layout). Packed
`DequantMatMul` uses `MlxMxfp4` with `group_size` from config.

Loaders can **dequant to f32** (`into_f32_map` / `import-mlx`) or keep packs for
`Op::DequantMatMul`:

| Backend | Path |
|---------|------|
| CPU | Native `DequantMatMulMlx` |
| Metal | On-device GEMV (`m=1`) / tiled GEMM (`m>1`, TM=8) |
| CUDA / ROCm | Shared `dequant_matmul_mlx.cu` (`_gemv` / `_gemm`) |
| wgpu | `dequant_matmul_mlx.wgsl` (tiled TM=8) |
| Vulkan | `dequant_matmul_mlx.comp` — 1-D dispatch (MoltenVK-safe) |
| TPU | Host-dequant at HLO emit → `dot_general` |
| MLX (default) | `MlxAffine` → native `quantized_matmul`; mxfp → **Rust** host dequant + cache + matmul |
| MLX + `native-mxfp` | mxfp → MLX C++ `quantized_matmul` (`mode=mxfp4/mxfp8/nvfp4`) |

Shared launch args: `QuantScheme::mlx_gpu_launch()` → `(kind, bits, group_size)`.

Force host path on GPU backends: `RLX_MLX_DEQUANT_GPU_DISABLE=1`
(also honors `RLX_METAL_DEQUANT_GPU_DISABLE`). Host reference:
`rlx-gpu-host::run_dequant_matmul_mlx`.

Enable MLX C++ mxfp FFI: Cargo feature `rlx-mlx/native-mxfp` or
`rlx-runtime/mlx-native-mxfp`.
Packed import: `rlx-pkg import-mlx … --keep-packed` stores
`{base}.weight/.scales/.biases` with `mlx_affine/…` / `mlx_mxfp*` schemes
(and optional parallel `DequantMatMul` graph). Default still dequants to f32.

Bake: `--weights-policy f32|packed|auto` — see [`rlx-bake` README](../crates/io/rlx-bake/README.md).
Affine odd bits (3/5/6) are supported in CPU/Metal/CUDA/wgpu/Vulkan dequant kernels.

Graph helpers: `rlx_mlx_io::{collect_packed_linears, build_mlp_chain_graph,
build_parallel_dequant_graph, param_bindings_for}`.

## CLI / APIs

```sh
rlx-pkg import-mlx path/to/mlx-model -o model.rlxp --no-graph
rlx-pkg import-mlx path/to/mlx-model -o model.rlxp --keep-packed --no-graph
rlx-bake graph.json -o model.rlxp --weights path/to/weights.npz
```

```rust
use rlx_mlx_io::{load_path, collect_packed_linears, build_mlp_chain_graph};
let mut w = load_path("model-dir")?;
let linears = collect_packed_linears(&mut w)?;
let g = build_mlp_chain_graph("mlp", &linears, /*batch=*/1)?;
```

Dist URI: `mlx://<path>#<tensor>` (alias `npz://`).

## MLX C++ load (Apple / MLX builds)

`rlx-mlx-sys` exposes `rlx_mlx_load_safetensors` and `rlx_mlx_load_npy`
(wrapping `mx::load_safetensors` / `mx::load`). Prefer `rlx-mlx-io` for
cross-platform import; use the shim when arrays should stay on-device in MLX.
## License

MIT OR Apache-2.0.
