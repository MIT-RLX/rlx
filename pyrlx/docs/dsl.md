# Python DSL (`graph` / `Node`)

The DSL is a pure-Python layer in `pyrlx/dsl.py` on top of the native
`Graph` binding. It does not add new IR ops — it wraps integer node ids
with operator syntax and dtype-aware scalar promotion.

## When to use which builder

| Style | Entry | Best for |
|-------|-------|----------|
| Explicit | `g = rlx.Graph("name")` | Full IR surface, parity tests, Rust-mirrored code |
| DSL | `with rlx.graph("name") as g:` | Notebooks, `(x @ w + b).gelu()`, scalar literals |

Both styles can mix in one graph via `g.raw` inside a `graph()` block.

## Quick example

```python
import numpy as np
import pyrlx as rlx

with rlx.graph("mlp") as g:
    x = g.input("x", [2, 4], "f32")
    w = g.param("w", [4, 3], "f32")
    b = g.param("b", [3], "f32")
    g.outputs = [(x @ w + b).gelu() * 2.0]

compiled = rlx.Session("cpu").compile(g.raw)
rlx.set_param(compiled, "w", np.ones((4, 3), dtype=np.float32))
rlx.set_param(compiled, "b", np.zeros(3, dtype=np.float32))
out, = rlx.run(compiled, x=np.ones((2, 4), dtype=np.float32))
```

See also `pyrlx/examples/dsl_quickstart.py`.

## Scalar promotion

Python scalars (`bool`, `int`, `float`) in **elementwise** binary ops are
inserted as rank-0 `Graph.constant` nodes and broadcast:

| Syntax | IR |
|--------|-----|
| `x * 2.0` | `mul(x, constant(2.0, f32))` |
| `1.0 + x` | `add(constant(1.0), x)` |
| `x < 1.0` | `compare(lt, x, constant(1.0))` → bool tensor |

Dtype follows the non-scalar operand (`x.dtype`). Integer graph dtypes
(`i32`, …) promote **both** `int` and `float` literals to the matching
constant dtype (`x + 2.0` on `i8` → `constant(2.0, i8)`, not `f32`).

### Integer literal rules

| Check | Example | Result |
|-------|---------|--------|
| Out of range | `x + 300` on `i8` input | `ValueError: … out of range for dtype i8` |
| Non-integral | `x + 2.5` on `i32` input | `ValueError: … must be integral for dtype i32` |
| `abs(int) > 2**53` | `g.constant(10**30, "i64")` | `ValueError: … exceeds exact float range` |

Python `int` literals larger than `2**53` are rejected before `float()`
conversion — otherwise they would round silently. Use an explicit `float`
only when the value is within exact range.

**Not promoted as matrix operands:** `@` requires rank ≥ 2 for real matmul.
When either side of `@` is a scalar, the DSL scales elementwise instead
(`mul`) — same effect as `x * 2.0` for a vector/matrix.

**Not promoted:** keyword arguments on forwarded proxy calls (`g.fft(x,
inverse=False)`). Only positional operands in the binary-op set (`add`,
`sub`, `mul`, `div`, `compare`, …) accept scalars.

## Operators on `Node`

| Category | Syntax |
|----------|--------|
| Arithmetic | `+`, `-`, `*`, `/`, `@`, `**`, unary `-` |
| Comparison | `<`, `<=`, `>`, `>=`, `==`, `!=` → bool `Node` |
| Activations | `.relu()`, `.gelu()`, `.gelu_approx()`, `.silu()`, `.tanh()`, `.exp()`, `.sqrt()`, `.softmax()` |
| Shape | `.reshape()`, `.transpose()`, `.T`, `.cast()`, `.narrow()`, `.sum()`, `.mean()`, `.cumsum()`, `.gather()` |
| Norm / conv | `.layer_norm(g, b)`, `.rms_norm(g, b)`, `.layer_norm2d(g, b)`, `.group_norm(g, b, groups)`, `.conv2d(w, …)`, `.conv_transpose2d(w, …)` |
| Attention / RoPE | `.attention_kind(k, v, heads, dim, mask_kind=...)`, `.rope(cos, sin, head_dim)`, `.rope_n(...)` |
| Autodiff | `.stop_gradient()` |
| Select | `cond.where_(on_true, on_false)` |

Ops without a `Node` method (FFT, `dense_solve`, `custom_fn`, …) remain on
the proxy: `g.fft(x, inverse=True)` or `g.raw.fft(...)`.

## Proxy-only conveniences

Inside `with rlx.graph(...) as g:`:

- `g.input` / `g.param` / `g.constant` return `Node`.
- `g.outputs = [y, z]` is sugar for `set_outputs`.
- `g.raw` is the native `Graph` for `Session.compile` (which consumes it).
- All other `Graph` methods are forwarded; `Node` args unwrap to ints and
  int returns re-wrap as `Node`.

## Compile + run

```python
graph = g.raw                          # capture before compile
compiled = rlx.Session("metal").compile(graph)

# f32 fast path
compiled.set_param("w", w_array)
outs = compiled.run({"x": x_array})

# any dtype
rlx.set_param(compiled, "w", w_f64)
outs = rlx.run(compiled, x=x_f64)
```

## Errors

- Out-of-range scalar for a dtype → `ValueError` from `Graph.constant`.
- Non-integral value for an integer dtype → `ValueError` (`must be integral`).
- Python `int` larger than `2**53` → `ValueError` (`exceeds exact float range`).
- Invalid comparison op string → `ValueError` from `Graph.compare`.
- Mixing `Node` with unrelated Python types → `TypeError` with the
  expected types in the message.

## Related

- [`quickstart.md`](quickstart.md) — install + first graph
- [`backends.md`](backends.md) — maturin feature matrix
- [`../../docs/backend-selection.md`](../../docs/backend-selection.md) — multi-device runtime
