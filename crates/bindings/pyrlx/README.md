# pyrlx

Python bindings for RLX via [PyO3](https://pyo3.rs/) +
[maturin](https://maturin.rs/). Run any RLX backend from Python.

## API overview

| Style | Entry | When to use |
|-------|-------|-------------|
| Explicit | `rlx.Graph("name")` | Full IR, integer node ids, tests mirroring Rust |
| DSL | `with rlx.graph("name") as g:` | Notebooks, `(x @ w + b).gelu()`, scalar literals |
| Execute | `Session.compile(g)` or `compile(g.raw)` | Graph is consumed; cache `g.raw` first in DSL |
| Typed I/O | `rlx.set_param` / `rlx.run` | f64 / integers without manual `tobytes` |

## Features

- **Build graphs from Python** — `Graph` (explicit ids) or the
  `graph()` / `Node` DSL (`(x @ w + b).gelu()`, scalar literals). ~200
  builders mirroring `rlx_ir::Graph`: conv 1d/2d/3d (+ transposed), pooling,
  interpolation, attention variants (bias, v-dim, score scale / softcap), RoPE
  (`style="neox" | "gptj"`), batch / adaLN / fused residual norms, LSTM / GRU /
  RNN / Mamba-2 / Gated DeltaNet, `scan` over a body graph, scatter / gather
  ND, linalg factorizations, per-channel / block-scaled quantization,
  `dequant_matmul` (int8, MLX, GGUF K- and IQ-quants), VQ, KAN splines,
  FIR / IIR / SOS / Hilbert DSP, spectrogram features and SPD manifold layers.
  `activation(name, x)` / `binary(name, a, b)` take every IR activation and
  binary op by name.
- **Static checks** — `rlx.check(g, backends=["cpu", "cuda", "metal"])` runs
  `cargo rlx check`'s device-free diagnostics (op legality per backend, missed
  fusions, numeric hazards, memory-plan safety).
- **Precision policies** — `Session(device, policy="mixed")` or a per-op-kind
  dict (`{"compute": "f16"}`); also `compile_with(..., policy=...)`.
- **Compile + run on any backend** — `Session(device="cpu" | "metal" |
  "mlx" | …)`.
- **Multi-backend runtime** — `GraphDevices`, `DeviceRouter`,
  `DevicePolicy`, `FlexibleSession`, `backends_manifest()`,
  `parse_device()`. See [`docs/backend-selection.md`](../../../docs/backend-selection.md).
- **In-graph RNG** — ONNX `Random*` via upstream `rlx-runtime`
  (`CompileOptions::rng`, `CompiledGraph::set_rng`). See [`CHANGELOG.md`](../../../CHANGELOG.md).
- **Combine and split graphs** — `splice(target, source, bind=…, prefix=…)`
  copies one graph into another, binding named inputs or params to existing
  nodes; `sequential` / `parallel` compose two graphs; `extract` cuts out the
  subgraph behind chosen outputs.
- **FFT helpers** — `fft`, `fft_norm`, `rfft`, `irfft`, `fftfreq`,
  `rfftfreq`, `psd_real` on `Graph` (see `tests/test_fft.py`).
- **Autodiff** — `pyrlx.grad(graph, wrt=[…])` returns the backward
  graph, ready to compile.
- **JVP / vmap** — `pyrlx.jvp` + `pyrlx.vmap` for forward-mode AD and
  batched function transforms.
- **GGUF** — `rlx.quantize(weights, dtype="IQ2_XXS")` and
  `rlx.dequant(packed, dtype=…)` pack/unpack via `rlx-gguf` (no backend required).
- **GGUF file I/O** — `rlx.load_gguf(path)`, `rlx.write_gguf(path, tensors, …)`,
  and `rlx.GgufFile` for tensor metadata / dequant / raw bytes.
- **GGUF convert** — `rlx.convert_to_gguf("model.safetensors", "out.gguf", "Q4_K")`
  wraps `rlx-gguf-convert` (safetensors by default; `gguf-onnx` / `gguf-pt` cargo
  features for ONNX / PyTorch checkpoints).
  See `tests/test_gguf_quantize.py`, `test_gguf_file.py`, `test_gguf_convert.py`.
- **Training** *(feature `training`, default)* — `rlx.Trainer(loss_graph,
  wrt=[...], init={...}, optimizer={"kind": "adamw", "lr": 3e-4})` with
  `step` / `evaluate` / `run(steps, batch, lr=..., on_step=...)`, freezing,
  gradient clipping, GGUF checkpoints (`save` / `load`, the same layout as
  rlx-js), and the on-device AdamW path (`resident=True`, Metal / wgpu).
  `Trainer.from_package(path)` trains a function of an `.rlxp` package and
  `save_package` writes the trained weights back.
  `rlx.Optimizer("muon", lr=...)` drives `rlx-optim` over your own arrays.
- **Text** *(feature `text`, default)* — `load_tokenizer` (streaming
  `decode_incremental`), `Sampler` (carried RNG + repetition history),
  `sample_next`, `argmax`, `render_chat` (GGUF / HF config / `.jinja` / inline).
- **RLXP packages** — `rlx.load_rlxp(path)` (manifest summary),
  `rlx.verify_rlxp(path)` (checksums), and
  `rlx.convert_gguf_to_rlxp(gguf, out, include_graph=False)`;
  `rlx.to_rlxp(from_, src, dst)` converts any supported weight format, and
  `rlx.load_weights(format, path)` reads one into a dict. `rlx.Model` runs a
  packaged model by function name. Spec: [`docs/rlxp.md`](../../../docs/rlxp.md).
- **Distributed nodes** — `rlx.Group.connect(rank, world)` / `from_env()` joins
  a mesh of rlx nodes: `send` / `recv`, `all_reduce`, `all_gather`,
  `broadcast`, `barrier`. Calls release the GIL, so ranks can be threads
  (`loopback_peers(world)` for a local mesh).

## Install

```sh
pip install pyrlx         # CPU; on macOS also Metal + Accelerate
pip install pyrlx-cu12    # CPU + CUDA (Linux x64, Windows x64; CUDA 12.3+ driver)
```

Both install the same `import pyrlx`: pick one.

## Install (from source)

```sh
cd crates/bindings/pyrlx
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop --features cpu,gguf-convert   # add metal,mlx,cuda,… as needed
```

From the repo root you can also use
`maturin develop --release -m crates/bindings/pyrlx/Cargo.toml`
inside an activated virtualenv.

## Tests

Run from `crates/bindings/pyrlx/` after `maturin develop` (not from the repo root without a
venv — the bare top-level `pyrlx/` namespace package lacks `Graph`):

```sh
cd crates/bindings/pyrlx && source .venv/bin/activate
pytest tests/ -q
```

## Examples

[`examples/`](examples/README.md): `inference`, `training`, `loading`,
`distributed`, `backends` and `models` — the same scripts exist for JS and
Swift and print the same numbers; `tests/test_examples.py` runs them.

## Install (PyPI)

PyPI wheels are cut from the same source on release. See the project's
GitHub Releases page for the current wheel set:

<https://github.com/MIT-RLX/rlx/releases>

## Quickstart

```python
import numpy as np
import pyrlx as rlx

# Explicit builder
g = rlx.Graph("hello")
x = g.input("x", [1, 4], "f32")
w = g.param("w", [4, 2], "f32")
y = g.matmul(x, w)
g.set_outputs([y])

# Or the Pythonic DSL
with rlx.graph("hello_dsl") as g:
    x = g.input("x", [1, 4], "f32")
    w = g.param("w", [4, 2], "f32")
    g.outputs = [(x @ w * g.constant(2.0)).relu()]
    graph = g.raw

compiled = rlx.Session("cpu").compile(graph)
rlx.set_param(compiled, "w", np.eye(4, 2, dtype=np.float32))
out, = rlx.run(compiled, x=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
```

### Multi-backend

```python
policy = rlx.DevicePolicy.only(["cpu", "metal"])
runner = rlx.GraphDevices(g, policy=policy)
device, outs = runner.run_chain({"x": x})

router = rlx.DeviceRouter(g, policy=policy)
device, outs = router.run({"x": x})
```

See [`docs/backend-selection.md`](../../../docs/backend-selection.md),
[`docs/dsl.md`](docs/dsl.md) (DSL reference), and
[`docs/backends.md`](docs/backends.md).

## Graph builder notes

- **Shape inference** — builders infer output shapes (`matmul`, `add`,
  `conv2d`, `layer_norm`, …).
- **Literals** — `g.constant(2.0)` or `x * 2.0` in the DSL; rank-0, NumPy-broadcastable.
- **Reserved names** — `where_`, `eq_`, `lt_`, `gt_`, `ge_`, `ne_` (trailing underscore).
- **FFT / attention / conv** — on `Graph` and via DSL proxy forwarding; see
  `tests/test_fft.py`, `tests/test_ir_parity.py`.

## Status

Surface follows the Rust crates closely and matches the embedded JS binding
(`rlx-js`) feature for feature, under snake_case names: both are generated
from the op table in `rlx-bind` and wrap its shared runtime, as are the C ABI
(`rlx-capi`) and Swift (`rlx-swift`) — see [`docs/bindings.md`](../../../docs/bindings.md). Not bound: the
backward-only kernels autodiff emits — reach those through `grad` / `jvp`.

## License

MIT OR Apache-2.0.