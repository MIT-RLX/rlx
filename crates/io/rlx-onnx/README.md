# rlx-onnx

Run ONNX (`.onnx`) models on RLX backends. Default path: import via
[`rlx-onnx-import`](../rlx-onnx-import), compile with
[`rlx-runtime`](../rlx-runtime) `Session`, execute on CPU / Metal / CUDA /
ROCm / MLX / wgpu. Optional ONNX Runtime (`ort`) backend for parity and
execution-provider fallback.

## Install

```toml
[dependencies]
rlx-onnx = { version = "0.2", default-features = true, features = ["cpu"] }

# GPU backends (pick what your host supports)
# rlx-onnx = { version = "0.2", features = ["cuda"] }
# rlx-onnx = { version = "0.2", features = ["metal", "mlx"] }

# ORT fallback + EPs
# rlx-onnx = { version = "0.2", features = ["ort-fallback", "ort-cuda"] }
```

From the [`rlx`](../rlx) prelude:

```toml
rlx = { version = "0.2", features = ["cpu", "onnx"] }
```

## Usage

```rust
use rlx_onnx::{OnnxCompileLevel, OnnxModel};
use rlx_runtime::Device;

let mut model = OnnxModel::load_native("model.onnx", Device::Cpu, OnnxCompileLevel::Level3, 128)?;
model.print_io();
let inputs = model.zero_inputs_sized(32)?;
let outputs = model.run(&inputs)?;
```

- **`OnnxCompileLevel`** — maps ONNX graph-opt tiers 0–3 to
  [`CompileOptions`](../rlx-runtime) (DCE / constant folding / full pipeline).
- **`OnnxExecBackend::Native`** (default) — RLX compile + execute.
- **`OnnxExecBackend::Ort`** — requires `ort` or `ort-fallback` feature.

## CLI

```bash
cargo run -p rlx-onnx --features native --bin rlx-onnx-run -- model.onnx \
  --device cpu --level 3 --seq-len 128 --warmup 1 --iters 5

# List inputs/outputs only
cargo run -p rlx-onnx --bin rlx-onnx-run -- model.onnx --list-io

# ORT path (rebuild with ort-fallback)
cargo run -p rlx-onnx --features ort-fallback --bin rlx-onnx-run -- \
  model.onnx --exec ort --device cuda
```

## Features

| Feature | Purpose |
|---------|---------|
| `native` (default) | RLX import + compile path |
| `cpu` / `cuda` / `metal` / `mlx` / `rocm` / `gpu` | Delegate to `rlx-runtime` backend |
| `ort` / `ort-fallback` | ONNX Runtime session |
| `ort-cuda`, `ort-coreml`, `ort-rocm`, `ort-directml` | ORT execution providers |
| `all-ep` | Convenience bundle of GPU + ORT EPs |

## Tests

```bash
cargo test -p rlx-onnx
```

## License

MIT OR Apache-2.0.
