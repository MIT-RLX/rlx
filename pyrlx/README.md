# pyrlx

Python bindings for RLX via [PyO3](https://pyo3.rs/) +
[maturin](https://maturin.rs/). Run any RLX backend from Python.

## Features

- **Build graphs from Python** — `Graph` + `Tensor` mirrors of
  `rlx_ir::Graph`.
- **Compile + run on any backend** — `Session(device="cpu" | "metal" |
  "mlx" | …)`.
- **Autodiff** — `pyrlx.grad(graph, wrt=[…])` returns the backward
  graph, ready to compile.
- **JVP / vmap** — `pyrlx.jvp` + `pyrlx.vmap` for forward-mode AD and
  batched function transforms.
- **Embed**: `pyrlx.RlxEmbed.from_pretrained("…")` mirroring the Rust
  `rlx_models::embed::RlxEmbed`.

## Install (from source)

```sh
pip install maturin
maturin develop --release -m pyrlx/Cargo.toml
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
0.1.0 — expect more `__repr__`, NumPy interop, and dunder support to
land in subsequent minor versions.

## License

GPL-3.0-only.