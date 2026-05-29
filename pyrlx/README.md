# pyrlx

Python bindings for RLX via [PyO3](https://pyo3.rs/) +
[maturin](https://maturin.rs/). Run any RLX backend from Python.

## Features

- **Build graphs from Python** — `Graph` + `Tensor` mirrors of
  `rlx_ir::Graph`.
- **Compile + run on any backend** — `Session(device="cpu" | "metal" |
  "mlx" | …)`.
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
import pyrlx as rlx
g = rlx.Graph("hello")
x = g.input("x", (1, 4), "f32")
w = g.param("w", (4, 2), "f32")
y = g.matmul(x, w)
g.set_outputs([y])

session = rlx.Session("cpu")
compiled = session.compile(g)
compiled.set_param("w", [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
out = compiled.run({"x": [1.0, 2.0, 3.0, 4.0]})
```

## Status

Surface follows the Rust crates closely; ergonomics layer is minimal at
0.2.2 — expect more `__repr__`, NumPy interop, and dunder support to
land in subsequent minor versions.

## License

GPL-3.0-only.