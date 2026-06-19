# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""pyrlx — Python bindings for the RLX ML compiler.

RLX is a JAX-shaped tensor compiler: build a symbolic graph, compile it for a
backend (CPU, Metal, MLX, CUDA, ROCm, wgpu, …), then execute with NumPy I/O.

Quick probe
-----------
>>> import pyrlx as rlx
>>> rlx.available_devices()
['cpu', 'metal']

Two graph builders
------------------
**Explicit** — ``rlx.Graph`` returns integer node ids; compose with
``g.add(g.matmul(x, w), b)``. Full IR surface (FFT, attention, conv2d,
``dense_solve``, ``custom_fn``, …).

**DSL** — ``with rlx.graph("mlp") as g:`` yields a proxy whose methods return
``rlx.Node`` handles. Supports operator syntax (``x @ w + b``), scalar literals
(``x * 2.0``), comparisons (``x < y``), and ``g.outputs = [...]``. Pass
``g.raw`` to ``Session.compile``.

Execution
---------
- ``rlx.Session(device, precision).compile(graph)`` — consumes the graph.
- ``compiled.set_param`` / ``compiled.run`` — f32 fast path.
- ``rlx.set_param`` / ``rlx.run`` — dtype-aware wrappers (f64, i32, … via
  ``run_typed`` / ``set_param_typed`` under the hood).
- ``Session.compile_with(graph, fusion_options=...)`` — FKL / fusion toggles.

Multi-backend
-------------
``GraphDevices``, ``DeviceRouter``, ``FlexibleSession``, ``DevicePolicy`` —
lazy per-device compile caches, env-driven device chains, benchmark pick.
See ``docs/backend-selection.md``.

Transforms
----------
``grad``, ``jvp``, ``hvp``, ``vmap``, ``nth_order_grad`` — build derivative /
batched graphs from a forward graph before compile.

Further reading: ``pyrlx/README.md``, ``pyrlx/docs/quickstart.md``,
``pyrlx/docs/dsl.md``, ``pyrlx/docs/backends.md``.
"""

from __future__ import annotations

from ._pyrlx import (        # type: ignore[attr-defined]
    available_devices,
    is_available,
    parse_device_py as parse_device,
    backends_manifest,
    fastest_device_for_py as fastest_device_for,
    device_report_py as device_report,
    grad,
    jvp,
    hvp,
    nth_order_grad,
    directional_nth_grad,
    vmap,
    Graph,
    Session,
    CompiledGraph,
    FusionOptions,
    DevicePolicy,
    GraphDevices,
    DeviceCandidate,
    DeviceBenchResult,
    FlexibleSession,
    DeviceRouter,
    __version__,
)

from .dsl import (
    Node,
    dtype_str,
    graph,
    numpy_dtype,
    run,
    set_param,
)

__all__ = [
    "available_devices",
    "is_available",
    "parse_device",
    "backends_manifest",
    "fastest_device_for",
    "device_report",
    "grad",
    "jvp",
    "hvp",
    "nth_order_grad",
    "directional_nth_grad",
    "vmap",
    "jacfwd",
    "Graph",
    "Session",
    "CompiledGraph",
    "FusionOptions",
    "DevicePolicy",
    "GraphDevices",
    "DeviceCandidate",
    "DeviceBenchResult",
    "FlexibleSession",
    "DeviceRouter",
    "batch_narrow_relu_graph",
    "Node",
    "graph",
    "set_param",
    "run",
    "numpy_dtype",
    "dtype_str",
    "__version__",
]


def batch_narrow_relu_graph(
    name: str,
    batch_n: int,
    channels: int,
    height: int,
    width: int,
    dtype: str = "f32",
) -> Graph:
    """Build narrow-slice + relu + concat graph for FKL batch fusion tests.

    With ``FusionOptions.native_fk()`` and a GPU or TPU backend, the compile pipeline
    can fuse this into ``BatchElementwiseRegion``.
    """
    g = Graph(name)
    batch = g.input("batch", [batch_n, channels, height, width], dtype)
    slices = []
    for i in range(batch_n):
        sl = g.narrow(batch, 0, i, 1)
        slices.append(g.relu(sl))
    g.set_outputs([g.concat(slices, 0)])
    return g


def jacfwd(
    compiled_jvp,
    primals: dict,
    wrt_name: str,
    wrt_shape,
    dtype: str = "f64",
):
    """Forward-mode Jacobian by repeated JVP evaluation.

    Materializes a full Jacobian by running a compiled JVP graph once per
    standard-basis unit vector. Use when the input dimension is small; for
    larger batches prefer building a ``vmap``-wrapped JVP graph and compiling
    that once.

    Parameters
    ----------
    compiled_jvp : pyrlx.CompiledGraph
        A graph compiled from ``rlx.jvp(forward, [wrt])``. Has Inputs:
        the originals plus ``f"tangent_{wrt_name}"``.
    primals : dict[str, numpy.ndarray]
        Values for the original (non-tangent) inputs of the JVP graph,
        keyed by name. Arrays must already be the right dtype.
    wrt_name : str
        Name of the input whose Jacobian we're building. The tangent
        input is ``f"tangent_{wrt_name}"``.
    wrt_shape : tuple[int, ...]
        Shape of the wrt input — used to enumerate the standard basis.
        Total elements = number of JVP runs = number of Jacobian columns.
    dtype : str
        Element dtype string for both inputs and outputs. Defaults to
        ``"f64"``.

    Returns
    -------
    list[numpy.ndarray]
        One array per primal output, with shape
        ``(*output_shape, *wrt_shape)``. Element ``[i_out..., j_in...]``
        is ``∂output[i_out...]/∂wrt[j_in...]``.

    Notes
    -----
    The compiled JVP graph holds two outputs per primal output —
    ``[primal_0, ..., primal_{k-1}, tangent_0, ..., tangent_{k-1}]``.
    ``jacfwd`` reads only the tangent half on each run.
    """
    import numpy as np

    np_dtype = {
        "f32": np.float32, "float32": np.float32,
        "f64": np.float64, "float64": np.float64,
    }.get(dtype)
    if np_dtype is None:
        raise ValueError(f"jacfwd: dtype {dtype!r} not supported (use 'f32' or 'f64')")

    primal_payload = {
        name: (np.ascontiguousarray(arr, dtype=np_dtype).tobytes(), dtype)
        for name, arr in primals.items()
    }

    n_in = 1
    for d in wrt_shape:
        n_in *= int(d)
    tangent_key = f"tangent_{wrt_name}"
    out_columns = []

    flat = np.zeros(n_in, dtype=np_dtype)
    for j in range(n_in):
        flat[j] = 1.0
        if j > 0:
            flat[j - 1] = 0.0
        payload = dict(primal_payload)
        payload[tangent_key] = (flat.reshape(wrt_shape).tobytes(), dtype)
        outs = compiled_jvp.run_typed(payload)
        n_outs = len(outs) // 2
        column = []
        for raw, dt in outs[n_outs:]:
            arr = np.frombuffer(raw, dtype=np_dtype)
            column.append(arr)
        out_columns.append(column)
    flat[n_in - 1] = 0.0

    result = []
    if not out_columns:
        return result
    n_outs = len(out_columns[0])
    for o in range(n_outs):
        cols = [c[o] for c in out_columns]
        out_n = cols[0].shape[0]
        stacked = np.stack(cols, axis=0).reshape(n_in, out_n).T
        result.append(stacked.reshape((out_n,) + tuple(wrt_shape)))
    return result
