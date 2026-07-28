# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pythonic graph DSL — ``Node``, ``graph()``, and dtype-aware I/O helpers.

Use ``with pyrlx.graph("name") as g:`` for operator syntax and chained
methods; pass ``g.raw`` to ``Session.compile``. See ``pyrlx/docs/dsl.md``.
"""

from __future__ import annotations

import contextlib as _contextlib

from ._pyrlx import Graph  # type: ignore[attr-defined]


class Node:
    """Operator-overloaded handle for a symbolic tensor in a ``graph()`` block.

    Arithmetic (``+``, ``-``, ``*``, ``/``, ``@``, ``**``, unary ``-``) accepts
    another ``Node`` or a Python scalar; scalars become broadcastable
    ``Graph.constant`` nodes typed to match ``node.dtype``.

    Comparisons (``<``, ``<=``, ``==``, …) return bool-dtyped ``Node`` tensors.
    A scalar on either side of ``@`` scales elementwise (``mul``) because rank-0
    literals cannot participate in matrix multiply.

    Unary activations (``.relu()``, ``.gelu()``, …) and shape helpers
    (``.reshape()``, ``.sum()``, ``.cast()``) chain directly. Multi-input ops
    (``.layer_norm(g, b)``, ``.conv2d(w, …)``, ``.attention_kind(k, v, …)``)
    are available as ``Node`` methods as well as via the graph proxy.
    """

    __slots__ = ("_g", "id")

    def __init__(self, g: Graph, node_id: int) -> None:
        object.__setattr__(self, "_g", g)
        object.__setattr__(self, "id", int(node_id))

    @property
    def shape(self) -> tuple:
        """Inferred output shape as a tuple (empty for scalars)."""
        dims, _ = self._g.shape_of(self.id)
        return tuple(dims)

    @property
    def dtype(self) -> str:
        """Inferred element dtype label."""
        _, dt = self._g.shape_of(self.id)
        return dt

    # ── arithmetic ─────────────────────────────────────────────────
    def __add__(self, other):     return Node(self._g, self._g.add(self.id, _as_operand(self, other)))
    def __radd__(self, other):    return Node(self._g, self._g.add(_as_operand(self, other), self.id))
    def __sub__(self, other):     return Node(self._g, self._g.sub(self.id, _as_operand(self, other)))
    def __rsub__(self, other):    return Node(self._g, self._g.sub(_as_operand(self, other), self.id))
    def __mul__(self, other):     return Node(self._g, self._g.mul(self.id, _as_operand(self, other)))
    def __rmul__(self, other):    return Node(self._g, self._g.mul(_as_operand(self, other), self.id))
    def __truediv__(self, other): return Node(self._g, self._g.div(self.id, _as_operand(self, other)))
    def __rtruediv__(self, other):return Node(self._g, self._g.div(_as_operand(self, other), self.id))
    def __pow__(self, other):     return Node(self._g, self._g.binary("pow", self.id, _as_operand(self, other)))
    def __rpow__(self, other):    return Node(self._g, self._g.binary("pow", _as_operand(self, other), self.id))
    def __neg__(self):            return Node(self._g, self._g.neg(self.id))

    def __matmul__(self, other):
        if isinstance(other, (bool, int, float)):
            return self._scale_by_scalar(other)
        return Node(self._g, self._g.matmul(self.id, _as_id(other)))

    def __rmatmul__(self, other):
        if isinstance(other, (bool, int, float)):
            return self._scale_by_scalar(other)
        return Node(self._g, self._g.matmul(_as_id(other), self.id))

    def _scale_by_scalar(self, scalar) -> Node:
        """``@`` with a scalar is elementwise scale (rank-0 cannot matmul)."""
        sid = _promote_scalar(self._g, scalar, self.dtype)
        return Node(self._g, self._g.mul(self.id, sid))

    # ── comparisons (bool output) ──────────────────────────────────
    def _compare(self, op: str, other) -> Node:
        return Node(
            self._g,
            self._g.compare(op, self.id, _as_operand(self, other)),
        )

    def __eq__(self, other):  # noqa: D105
        if not isinstance(other, (Node, int, float, bool)):
            return NotImplemented
        return self._compare("eq", other)

    def __ne__(self, other):  # noqa: D105
        if not isinstance(other, (Node, int, float, bool)):
            return NotImplemented
        return self._compare("ne", other)

    def __lt__(self, other):
        if not isinstance(other, (Node, int, float, bool)):
            return NotImplemented
        return self._compare("lt", other)

    def __le__(self, other):
        if not isinstance(other, (Node, int, float, bool)):
            return NotImplemented
        return self._compare("le", other)

    def __gt__(self, other):
        if not isinstance(other, (Node, int, float, bool)):
            return NotImplemented
        return self._compare("gt", other)

    def __ge__(self, other):
        if not isinstance(other, (Node, int, float, bool)):
            return NotImplemented
        return self._compare("ge", other)

    # ── activations ────────────────────────────────────────────────
    def relu(self):    return Node(self._g, self._g.relu(self.id))
    def gelu(self):    return Node(self._g, self._g.gelu(self.id))
    def gelu_approx(self):
        """Tanh-approximation GELU (PyTorch default)."""
        return Node(self._g, self._g.gelu_approx(self.id))
    def silu(self):    return Node(self._g, self._g.silu(self.id))
    def tanh(self):    return Node(self._g, self._g.tanh(self.id))
    def exp(self):     return Node(self._g, self._g.exp(self.id))
    def sqrt(self):    return Node(self._g, self._g.sqrt(self.id))
    def softmax(self, axis: int = -1):
        return Node(self._g, self._g.softmax(self.id, axis))

    # ── shape ops ──────────────────────────────────────────────────
    def reshape(self, *new_shape):
        if len(new_shape) == 1 and not isinstance(new_shape[0], int):
            new_shape = tuple(new_shape[0])
        return Node(self._g, self._g.reshape(self.id, list(new_shape)))

    def transpose(self, *perm):
        if not perm:
            ndim = len(self.shape)
            perm = tuple(range(ndim - 1, -1, -1))
        elif len(perm) == 1 and not isinstance(perm[0], int):
            perm = tuple(perm[0])
        return Node(self._g, self._g.transpose(self.id, list(perm)))

    @property
    def T(self):
        """Matrix transpose — swaps the last two axes."""
        return self.transpose()

    def cast(self, to: str):
        return Node(self._g, self._g.cast(self.id, to))

    def narrow(self, axis: int, start: int, length: int):
        return Node(self._g, self._g.narrow(self.id, axis, start, length))

    def sum(self, axes, keep_dim: bool = False):
        return Node(self._g, self._g.sum(self.id, list(axes), keep_dim))

    def mean(self, axes, keep_dim: bool = False):
        return Node(self._g, self._g.mean(self.id, list(axes), keep_dim))

    def cumsum(self, axis: int = -1, exclusive: bool = False):
        return Node(self._g, self._g.cumsum(self.id, axis, exclusive))

    def gather(self, indices, axis: int = 0):
        return Node(self._g, self._g.gather(self.id, _as_id(indices), axis))

    def stop_gradient(self):
        """Identity forward, zero backward (``jax.lax.stop_gradient``)."""
        return Node(self._g, self._g.stop_gradient(self.id))

    def where_(self, on_true, on_false):
        """``where(cond, on_true, on_false)`` with this node as ``cond``."""
        return Node(
            self._g,
            self._g.where_(self.id, _as_id(on_true), _as_id(on_false)),
        )

    # ── conv / norm / attention / rope ─────────────────────────────
    def conv2d(
        self,
        weight,
        kernel_size,
        stride,
        padding,
        dilation,
        groups: int = 1,
    ):
        return Node(
            self._g,
            self._g.conv2d(
                self.id,
                _as_id(weight),
                list(kernel_size),
                list(stride),
                list(padding),
                list(dilation),
                groups,
            ),
        )

    def conv_transpose2d(
        self,
        weight,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        groups: int = 1,
    ):
        return Node(
            self._g,
            self._g.conv_transpose2d(
                self.id,
                _as_id(weight),
                list(kernel_size),
                list(stride),
                list(padding),
                list(dilation),
                list(output_padding),
                groups,
            ),
        )

    def layer_norm(self, gamma, beta, axis: int = -1, eps: float = 1e-5):
        return Node(
            self._g,
            self._g.layer_norm(self.id, _as_id(gamma), _as_id(beta), axis, eps),
        )

    def rms_norm(self, gamma, beta, eps: float = 1e-5):
        return Node(
            self._g,
            self._g.rms_norm(self.id, _as_id(gamma), _as_id(beta), eps),
        )

    def layer_norm2d(self, gamma, beta, eps: float = 1e-5):
        return Node(
            self._g,
            self._g.layer_norm2d(self.id, _as_id(gamma), _as_id(beta), eps),
        )

    def group_norm(self, gamma, beta, num_groups: int, eps: float = 1e-5):
        return Node(
            self._g,
            self._g.group_norm(
                self.id, _as_id(gamma), _as_id(beta), num_groups, eps
            ),
        )

    def attention_kind(
        self,
        k,
        v,
        num_heads: int,
        head_dim: int,
        mask_kind: str = "causal",
    ):
        return Node(
            self._g,
            self._g.attention_kind(
                self.id, _as_id(k), _as_id(v), num_heads, head_dim, mask_kind
            ),
        )

    def rope(self, cos, sin, head_dim: int):
        return Node(
            self._g,
            self._g.rope(self.id, _as_id(cos), _as_id(sin), head_dim),
        )

    def rope_n(self, cos, sin, head_dim: int, n_rot: int):
        return Node(
            self._g,
            self._g.rope_n(
                self.id, _as_id(cos), _as_id(sin), head_dim, n_rot
            ),
        )

    # ── int coercion + repr ────────────────────────────────────────
    def __int__(self) -> int:
        return self.id

    def __index__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        try:
            dims, dt = self._g.shape_of(self.id)
            return f"<pyrlx.Node id={self.id} shape={tuple(dims)} dtype={dt}>"
        except Exception:
            return f"<pyrlx.Node id={self.id}>"


def _as_id(x):
    """Coerce `Node | int` → int. Other types fall through to a clear error."""
    if isinstance(x, Node):
        return x.id
    if isinstance(x, int):
        return x
    raise TypeError(
        f"expected pyrlx.Node or int node id, got {type(x).__name__}"
    )


_INT_DTYPES = frozenset({"i8", "i16", "i32", "i64", "u8", "u32"})
_FLOAT_DTYPES = frozenset({"f32", "f64", "f16", "bf16"})
_MAX_EXACT_INT = 2**53


def _float_from_int(value: int) -> float:
    """Convert a Python int to f64 without silent rounding past 2**53."""
    if abs(value) > _MAX_EXACT_INT:
        raise ValueError(
            f"integer literal {value} exceeds exact float range (2**53); "
            "pass a float to g.constant() or use a smaller integer"
        )
    return float(value)


def _promote_scalar(g: Graph, value, dtype_hint: str | None = None) -> int:
    """Insert a broadcastable rank-0 constant for a Python scalar."""
    if isinstance(value, bool):
        return g.constant(1.0 if value else 0.0, "bool")
    if isinstance(value, int):
        fv = _float_from_int(value)
        dt = dtype_hint or "f32"
        if dt in _INT_DTYPES:
            return g.constant(fv, dt)
        return g.constant(fv, "f32")
    if isinstance(value, float):
        dt = dtype_hint or "f32"
        if dt in _INT_DTYPES:
            return g.constant(value, dt)
        if dt in _FLOAT_DTYPES:
            return g.constant(value, dt)
        return g.constant(value, "f32")
    raise TypeError(
        f"expected pyrlx.Node, int node id, or scalar, got {type(value).__name__}"
    )


def _as_operand(node: Node, other):
    """Coerce arithmetic operands: `Node`, raw id, or Python scalar → node id."""
    if isinstance(other, Node):
        return other.id
    if isinstance(other, (bool, int, float)):
        return _promote_scalar(node._g, other, node.dtype)
    raise TypeError(
        f"expected pyrlx.Node, int node id, or scalar, got {type(other).__name__}"
    )


def _dtype_hint_from_args(args) -> str | None:
    for arg in args:
        if isinstance(arg, Node):
            return arg.dtype
    return None


def _unwrap_value(v):
    """Recursively unwrap `Node` handles for native `Graph` calls."""
    if isinstance(v, Node):
        return v.id
    if isinstance(v, (list, tuple)):
        return type(v)(_unwrap_value(x) for x in v)
    return v


_BINARY_GRAPH_OPS = frozenset({
    "add", "sub", "mul", "div", "binary", "compare", "eq_", "lt_",
})


def _unwrap_binary_arg(g_ref: Graph, v, dtype_hint: str | None):
    if isinstance(v, Node):
        return v.id
    if isinstance(v, (bool, int, float)):
        return _promote_scalar(g_ref, v, dtype_hint)
    if isinstance(v, (list, tuple)):
        return type(v)(_unwrap_binary_arg(g_ref, x, dtype_hint) for x in v)
    return v


class _GraphProxy:
    """Fluent façade over ``Graph`` used inside ``pyrlx.graph(...)``.

    - ``g.input`` / ``g.param`` / ``g.constant`` return ``Node``.
    - Other ``Graph`` methods are forwarded: ``Node`` args are unwrapped to
      ints; int returns (and int tuples from ``fft_real`` / ``rfft``) become
      ``Node`` again.
    - Binary forwarded calls (``g.add``, ``g.mul``, …) also accept Python
      scalars on the operand side.
    - ``g.outputs = [y]`` is write-only sugar for ``set_outputs``.
    - ``g.raw`` is the native ``Graph`` for ``Session.compile`` (which takes
      ownership).
    """

    __slots__ = ("_g",)

    def __init__(self, g: Graph) -> None:
        object.__setattr__(self, "_g", g)

    @property
    def raw(self) -> Graph:
        """The underlying native `Graph` — pass to `Session.compile`."""
        return self._g

    def __setattr__(self, name: str, value) -> None:
        if name == "outputs":
            object.__getattribute__(self, "_g").set_outputs(
                [_as_id(n) for n in value]
            )
            return
        object.__setattr__(self, name, value)

    def input(self, name: str, shape, dtype: str = "f32") -> Node:
        """Declare a runtime input (shape is a sequence of ints)."""
        return Node(self._g, self._g.input(name, list(shape), dtype))

    def param(self, name: str, shape, dtype: str = "f32") -> Node:
        """Declare a weight / buffer filled via ``set_param`` before ``run``."""
        return Node(self._g, self._g.param(name, list(shape), dtype))

    def constant(self, value, dtype: str = "f32") -> Node:
        """Rank-0 literal broadcastable in binary ops (see also ``x * 2.0``)."""
        fv = _float_from_int(value) if isinstance(value, int) else float(value)
        return Node(self._g, self._g.constant(fv, dtype))

    def __getattr__(self, name: str):
        if name == "outputs":
            raise AttributeError(
                "graph outputs are write-only — assign with `g.outputs = [...]`"
            )
        attr = getattr(self._g, name)
        if not callable(attr):
            return attr

        g_ref = self._g

        def wrapped(*args, **kwargs):
            if name in _BINARY_GRAPH_OPS:
                hint = _dtype_hint_from_args(args)
                args = tuple(_unwrap_binary_arg(g_ref, a, hint) for a in args)
                kwargs = {k: _unwrap_value(v) for k, v in kwargs.items()}
            else:
                args = tuple(_unwrap_value(a) for a in args)
                kwargs = {k: _unwrap_value(v) for k, v in kwargs.items()}
            result = attr(*args, **kwargs)
            if isinstance(result, int):
                return Node(g_ref, result)
            if isinstance(result, tuple) and result and all(
                isinstance(r, int) for r in result
            ):
                return tuple(Node(g_ref, r) for r in result)
            return result

        return wrapped


@_contextlib.contextmanager
def graph(name: str):
    """Context manager for the Pythonic graph builder.

    Parameters
    ----------
    name :
        Debug label stored in the IR (shows up in fusion / dispatch reports).

    Yields
    ------
    _GraphProxy
        Builder whose methods return ``Node`` handles.

    Notes
    -----
    Capture ``g.raw`` before ``compile`` — ``Session.compile`` moves the
    underlying ``Graph`` out of the proxy. Mixing DSL nodes with explicit
    ``g.raw.matmul(int(x), …)`` calls in the same graph is supported.

    Examples
    --------
    >>> with pyrlx.graph("mlp") as g:
    ...     x = g.input("x", [2, 4], "f32")
    ...     w = g.param("w", [4, 3], "f32")
    ...     g.outputs = [(x @ w + 0.5).gelu()]
    >>> compiled = pyrlx.Session("cpu").compile(g.raw)
    """
    raw = Graph(name)
    yield _GraphProxy(raw)


# ── dtype helpers + smart set_param / run wrappers ─────────────────

_DTYPE_TO_NP_STR = {
    "f32": "float32", "float32": "float32",
    "f16": "float16", "float16": "float16",
    "f64": "float64", "float64": "float64",
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "i8":  "int8",  "u8":  "uint8",
    "i16": "int16", "i32": "int32",
    "u32": "uint32","i64": "int64",
    "bool": "bool",
}


def numpy_dtype(s: str):
    """Map an RLX dtype string (e.g. ``"f32"``) to a NumPy dtype."""
    import numpy as np
    label = _DTYPE_TO_NP_STR.get(s, s)
    try:
        return np.dtype(label)
    except TypeError as e:
        raise TypeError(f"no numpy dtype for rlx dtype {s!r}: {e}") from None


def dtype_str(np_dtype) -> str:
    """Map a NumPy dtype to the matching RLX dtype string."""
    import numpy as np
    np_dtype = np.dtype(np_dtype)
    mapping = [
        (np.float32, "f32"), (np.float16, "f16"), (np.float64, "f64"),
        (np.int8,   "i8"),   (np.uint8,   "u8"),
        (np.int16,  "i16"),  (np.int32,   "i32"),
        (np.uint32, "u32"),  (np.int64,   "i64"),
        (np.bool_,  "bool"),
    ]
    for ty, label in mapping:
        if np_dtype == np.dtype(ty):
            return label
    raise TypeError(f"no rlx dtype label for numpy dtype {np_dtype!r}")


def set_param(compiled, name: str, array) -> None:
    """Upload a graph param from any NumPy dtype."""
    import numpy as np
    arr = np.ascontiguousarray(array)
    if arr.dtype == np.float32:
        compiled.set_param(name, arr)
    else:
        compiled.set_param_typed(name, arr.tobytes(), dtype_str(arr.dtype))


def run(compiled, **inputs):
    """Execute a compiled graph; keyword args are input name → ndarray."""
    import numpy as np
    arrs = {k: np.ascontiguousarray(v) for k, v in inputs.items()}
    all_f32 = all(a.dtype == np.float32 for a in arrs.values())
    if all_f32:
        return list(compiled.run(arrs))
    payload = {k: (a.tobytes(), dtype_str(a.dtype)) for k, a in arrs.items()}
    outs = compiled.run_typed(payload)
    return [np.frombuffer(raw, dtype=numpy_dtype(dt)) for raw, dt in outs]
