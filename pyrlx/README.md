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
  `graph()` / `Node` DSL (`(x @ w + b).gelu()`, scalar literals).
- **Compile + run on any backend** — `Session(device="cpu" | "metal" |
  "mlx" | …)`.
- **Multi-backend runtime** — `GraphDevices`, `DeviceRouter`,
  `DevicePolicy`, `FlexibleSession`, `backends_manifest()`,
  `parse_device()`. See [`docs/backend-selection.md`](../docs/backend-selection.md).
- **FFT helpers** — `fft`, `fft_norm`, `rfft`, `irfft`, `fftfreq`,
  `rfftfreq`, `psd_real` on `Graph` (see `pyrlx/tests/test_fft.py`).
- **Autodiff** — `pyrlx.grad(graph, wrt=[…])` returns the backward
  graph, ready to compile.
- **JVP / vmap** — `pyrlx.jvp` + `pyrlx.vmap` for forward-mode AD and
  batched function transforms.

## Install (from source)

```sh
cd pyrlx
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest
maturin develop --features cpu   # add metal,mlx,cuda,… as needed
```

From the repo root you can also use `maturin develop --release -m pyrlx/Cargo.toml`
inside an activated virtualenv.

## Tests

Run from `pyrlx/` after `maturin develop` (not from the repo root without a
venv — the bare `pyrlx/` directory is a namespace package and lacks `Graph`):

```sh
cd pyrlx && source .venv/bin/activate
pytest tests/ -q
```

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

See [`docs/backend-selection.md`](../docs/backend-selection.md),
[`pyrlx/docs/dsl.md`](docs/dsl.md) (DSL reference), and
[`pyrlx/docs/backends.md`](docs/backends.md).

## Graph builder notes

- **Shape inference** — prefer `matmul`, `add`, `conv2d`, `layer_norm`, etc.
  over `*_with_shape` unless you need a fixed output layout.
- **Literals** — `g.constant(2.0)` or `x * 2.0` in the DSL; rank-0, NumPy-broadcastable.
- **Reserved names** — `where_`, `eq_`, `lt_`, `gt_`, `ge_`, `ne_` (trailing underscore).
- **FFT / attention / conv** — on `Graph` and via DSL proxy forwarding; see
  `tests/test_fft.py`, `tests/test_ir_parity.py`.

## Status

Surface follows the Rust crates closely. DSL + scalar literals + expanded
`Graph` bindings (conv, norm, `stop_gradient`, …) ship alongside multi-backend
helpers (`GraphDevices`, `DeviceRouter`).

## License

GPL-3.0-only.