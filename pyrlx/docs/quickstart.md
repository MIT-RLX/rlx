# Quickstart

Build with the backends you care about, then run a graph.

## 1. Install (Apple Silicon, full stack)

```sh
uv venv && source .venv/bin/activate
uv pip install maturin
uv pip install -e . --no-build-isolation \
  --config-settings=build-args='--features cpu,blas-accelerate,metal'
```

## 2. Confirm what shipped

```python
import pyrlx as rlx
print(rlx.available_devices())     # e.g. ['cpu', 'metal']
print(rlx.is_available("cuda"))    # False on a mac
```

## 3. Build, compile, run

```python
import numpy as np
import pyrlx as rlx

g = rlx.Graph("mlp")
x   = g.input("x", [128, 768], "f32")
w   = g.param("w", [768, 768], "f32")
b   = g.param("b", [768],      "f32")
out = g.gelu(g.add(g.matmul(x, w), b))   # shapes inferred
g.set_outputs([out])

sess     = rlx.Session(device="metal")            # cpu / metal / mlx / cuda / rocm / gpu
compiled = sess.compile(g)                        # consumes g

rng = np.random.default_rng(0)
compiled.set_param("w", rng.standard_normal((768, 768)).astype(np.float32) / 768**0.5)
compiled.set_param("b", np.zeros(768, dtype=np.float32))

[y] = compiled.run({"x": rng.standard_normal((128, 768)).astype(np.float32)})
print(y.shape)   # (128, 768)
```

## 4. Switch backends

```python
for dev in rlx.available_devices():
    [y] = rlx.Session(device=dev).compile(_make_graph()).run(inputs)
    print(dev, y.mean())
```

The compiled output of every backend is the same up to numerical
precision — that's the parity test (`pyrlx/examples/cross_backend_parity.py`).
