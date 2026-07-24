# rlx-mlx-io

Load **MLX weight layouts** into RLX:

- HuggingFace `mlx-community` directories (`config.json` + `model*.safetensors`)
- Single `.safetensors`
- `.npz` / `.npy` (`mx.savez`, `nn.Module.save_weights`)

Quantized mlx-lm packs (`affine`, `mxfp4`, `nvfp4`, `mxfp8`) can be dequantized to
f32 or kept for `Op::DequantMatMul` with `QuantScheme::MlxAffine` /
`MlxMxfp4` / `MlxMxfp8`. mlx-lm `nvfp4` maps to `MlxMxfp4` (not NVIDIA
`Nvfp4Block`).

```rust
use rlx_mlx_io::load_path;
let w = load_path("path/to/mlx-model")?;
let f32_map = w.into_f32_map()?;
```

GPL-3.0-only.
