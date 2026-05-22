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

>>> import pyrlx as rlx
>>> rlx.available_devices()
['cpu', 'metal']

Layered API
-----------
- ``rlx.available_devices()`` / ``rlx.is_available(name)`` —
  query the backends compiled into this wheel.
- ``rlx.Graph`` — minimal graph builder over ``rlx_ir::Graph``.
- ``rlx.Session(device, precision)`` + ``rlx.CompiledGraph`` —
  the hot-path execution surface, with NumPy I/O.

See ``pyrlx/README.md`` for installation, backend feature combos,
troubleshooting, and a side-by-side comparison with PyTorch / JAX.
"""

from __future__ import annotations

# The native module is importable as ``pyrlx._pyrlx``.
from ._pyrlx import (        # type: ignore[attr-defined]
    available_devices,
    is_available,
    grad,
    jvp,
    vmap,
    Graph,
    Session,
    CompiledGraph,
    __version__,
)

__all__ = [
    "available_devices",
    "is_available",
    "grad",
    "jvp",
    "vmap",
    "jacfwd",
    "Graph",
    "Session",
    "CompiledGraph",
    "__version__",
]


# ── Python-side helpers built atop the native bindings ──────────────

def jacfwd(
    compiled_jvp,
    primals: dict,
    wrt_name: str,
    wrt_shape,
    dtype: str = "f64",
):
    """Forward-mode Jacobian by repeated JVP evaluation.

    Pre-vmap convenience that materializes a full Jacobian by running
    a compiled JVP graph once per standard-basis unit vector. Use this
    when the input dimension is small (Circulax component groups have
    handfuls of params) — once an `rlx.vmap` lands, this becomes a
    one-call vectorised JVP.

    Parameters
    ----------
    compiled_jvp : pyrlx.CompiledGraph
        A graph compiled from `rlx.jvp(forward, [wrt])`. Has Inputs:
        the originals plus `f"tangent_{wrt_name}"`.
    primals : dict[str, numpy.ndarray]
        Values for the original (non-tangent) inputs of the JVP graph,
        keyed by name. Arrays must already be the right dtype.
    wrt_name : str
        Name of the input whose Jacobian we're building. The tangent
        input is `f"tangent_{wrt_name}"`.
    wrt_shape : tuple[int, ...]
        Shape of the wrt input — used to enumerate the standard basis.
        Total elements = number of JVP runs = number of Jacobian columns.
    dtype : str
        Element dtype string for both inputs and outputs. Defaults to
        ``"f64"`` since that's the only fully-supported numerical dtype
        on the rlx-cpu backend today.

    Returns
    -------
    list[numpy.ndarray]
        One array per primal output, with shape
        ``(*output_shape, *wrt_shape)``. Element ``[i_out..., j_in...]``
        is `∂output[i_out...]/∂wrt[j_in...]`.

    Notes
    -----
    The compiled JVP graph holds two outputs per primal output —
    `[primal_0, ..., primal_{k-1}, tangent_0, ..., tangent_{k-1}]`.
    `jacfwd` reads only the tangent half on each run.
    """
    import numpy as np  # local import keeps top-level lazy-friendly

    np_dtype = {
        "f32": np.float32, "float32": np.float32,
        "f64": np.float64, "float64": np.float64,
    }.get(dtype)
    if np_dtype is None:
        raise ValueError(f"jacfwd: dtype {dtype!r} not supported (use 'f32' or 'f64')")

    # Encode primals once — they don't change across runs.
    primal_payload = {
        name: (np.ascontiguousarray(arr, dtype=np_dtype).tobytes(), dtype)
        for name, arr in primals.items()
    }

    n_in = 1
    for d in wrt_shape:
        n_in *= int(d)
    tangent_key = f"tangent_{wrt_name}"
    out_columns = []  # list of list[ndarray] — outer = unit vector index

    flat = np.zeros(n_in, dtype=np_dtype)
    for j in range(n_in):
        flat[j] = 1.0
        if j > 0:
            flat[j - 1] = 0.0
        payload = dict(primal_payload)
        payload[tangent_key] = (flat.reshape(wrt_shape).tobytes(), dtype)
        outs = compiled_jvp.run_typed(payload)
        # Outputs are [primals..., tangents...] — keep the tangent half.
        n_outs = len(outs) // 2
        column = []
        for raw, dt in outs[n_outs:]:
            arr = np.frombuffer(raw, dtype=np_dtype)
            column.append(arr)
        out_columns.append(column)
    flat[n_in - 1] = 0.0  # restore for caller's safety, harmless

    # Stack: for each output, build (n_outputs_elems, n_in) then reshape.
    result = []
    if not out_columns:
        return result
    n_outs = len(out_columns[0])
    for o in range(n_outs):
        cols = [c[o] for c in out_columns]
        out_n = cols[0].shape[0]
        # rows = n_in (one per unit vector); each row holds the tangent
        # output for that perturbation. Stack to (n_in, out_n) then
        # transpose so axis 0 is the output index.
        stacked = np.stack(cols, axis=0).reshape(n_in, out_n).T  # (out_n, n_in)
        # Reshape n_in dim back to wrt_shape for caller convenience.
        result.append(stacked.reshape((out_n,) + tuple(wrt_shape)))
    return result
