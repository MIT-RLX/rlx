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
import json
import pyrlx as rlx
print(rlx.available_devices())     # e.g. ['cpu', 'metal']
print(rlx.is_available("cuda"))    # False on a mac
print(json.loads(rlx.backends_manifest()))  # compile-time features
```

## 3. Build, compile, run

### Explicit builder

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

### Pythonic DSL

Same graph with operator syntax and scalar literals:

```python
with rlx.graph("mlp_dsl") as g:
    x = g.input("x", [128, 768], "f32")
    w = g.param("w", [768, 768], "f32")
    b = g.param("b", [768], "f32")
    g.outputs = [(x @ w + b).gelu()]
    graph = g.raw

compiled = rlx.Session("metal").compile(graph)
# rlx.set_param(compiled, "w", w_array)  — or compiled.set_param for f32
# out, = rlx.run(compiled, x=x_array)
```

Use `g.constant(2.0)` or `x * 2.0` for broadcastable literals. Forwarded
builders (`g.conv2d`, `g.attention_kind`, FFT, …) accept `Node` arguments.

## 4. Switch backends

Single-session loop (parity check):

```python
for dev in rlx.available_devices():
    [y] = rlx.Session(device=dev).compile(_make_graph()).run(inputs)
    print(dev, y.mean())
```

Multi-backend helpers (lazy compile cache, env-driven pick, fallback):

```python
runner = rlx.GraphDevices(g, policy=rlx.DevicePolicy.from_env())
device, outs = runner.run_chain(inputs)

router = rlx.DeviceRouter(g)  # warm-all on init
device, outs = router.run(inputs)
```

See [`docs/backend-selection.md`](../../docs/backend-selection.md).

The compiled output of every backend is the same up to numerical
precision — that's the parity test (`pyrlx/examples/cross_backend_parity.py`).

Full DSL reference: [`dsl.md`](dsl.md). Runnable demo: `python examples/dsl_quickstart.py`.

## License

GPL-3.0-only.
