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
"""Hello Resistor end-to-end at f64 through pyrlx.

Mirrors the Rust integration test at
``rlx-runtime/tests/cpu_f64_dense_solve.rs``. Validates the Python
surface for `grad`, `dense_solve`, `set_param_typed`, and `run_typed`.

To run after the new bindings land in the wheel:

    cd pyrlx
    maturin develop --release --features cpu,blas-accelerate
    pytest tests/test_dense_solve_f64.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

import pyrlx as rlx


def _build_forward(n: int) -> tuple[rlx.Graph, int, int, int]:
    """Build the forward graph: x = solve(A, b), loss = sum(x).
    Returns (graph, A_id, b_id, loss_id).
    """
    g = rlx.Graph("hello_resistor")
    a = g.input("A", [n, n], "f64")
    b = g.input("b", [n], "f64")
    x = g.dense_solve(a, b, [n], "f64")
    loss = g.reduce(x, "sum", [0], False)
    g.set_outputs([loss])
    return g, a, b, loss


def test_forward_dense_solve_f64() -> None:
    n = 3
    g = rlx.Graph("solve_only")
    a = g.input("A", [n, n], "f64")
    b = g.input("b", [n], "f64")
    x = g.dense_solve(a, b, [n], "f64")
    g.set_outputs([x])

    sess = rlx.Session(device="cpu")
    compiled = sess.compile(g)

    A = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ], dtype=np.float64)
    b_vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    outs = compiled.run_typed({
        "A": (A.tobytes(), "f64"),
        "b": (b_vec.tobytes(), "f64"),
    })
    assert len(outs) == 1
    raw, dt = outs[0]
    assert dt == "f64"
    x_got = np.frombuffer(raw, dtype=np.float64)

    # Residual check — works for any well-conditioned A.
    residual = A @ x_got
    np.testing.assert_allclose(residual, b_vec, atol=1e-10)


def test_hello_resistor_gradient_f64() -> None:
    """Forward: A param, b input, x = solve(A,b), loss = sum(x).
    Backward: dA, db verified against analytic + finite-difference."""
    n = 3

    g = rlx.Graph("hello_resistor")
    a = g.param("A", [n, n], "f64")
    b = g.input("b", [n], "f64")
    x = g.dense_solve(a, b, [n], "f64")
    loss = g.reduce(x, "sum", [0], False)
    g.set_outputs([loss])

    bwd = rlx.grad(g, [a, b])

    sess = rlx.Session(device="cpu")
    compiled = sess.compile(bwd)

    A = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ], dtype=np.float64)
    b_vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    d_out = np.array([1.0], dtype=np.float64)

    compiled.set_param_typed("A", A.tobytes(), "f64")
    outs = compiled.run_typed({
        "b":        (b_vec.tobytes(), "f64"),
        "d_output": (d_out.tobytes(), "f64"),
    })

    assert len(outs) == 3
    loss_bytes, dA_bytes, db_bytes = (raw for raw, _ in outs)
    loss_v = np.frombuffer(loss_bytes, dtype=np.float64)
    dA = np.frombuffer(dA_bytes, dtype=np.float64).reshape(n, n)
    db = np.frombuffer(db_bytes, dtype=np.float64)

    # Analytic ground truth.
    x_ref = np.linalg.solve(A, b_vec)
    loss_ref = x_ref.sum()
    db_ref = np.linalg.solve(A.T, np.ones(n))
    dA_ref = -np.outer(db_ref, x_ref)

    np.testing.assert_allclose(loss_v[0], loss_ref, atol=1e-10)
    np.testing.assert_allclose(db, db_ref, atol=1e-10)
    np.testing.assert_allclose(dA, dA_ref, atol=1e-10)

    # Finite-difference cross-check on db.
    h = 1e-6
    fd = np.empty(n)
    for k in range(n):
        bp = b_vec.copy(); bp[k] += h
        bm = b_vec.copy(); bm[k] -= h
        fd[k] = (np.linalg.solve(A, bp).sum() - np.linalg.solve(A, bm).sum()) / (2 * h)
    np.testing.assert_allclose(db, fd, atol=1e-7)


def test_jvp_dense_solve_b_only() -> None:
    """JVP of x = solve(A, b) w.r.t. b. Closed form: tangent_x = solve(A, t_b)."""
    n = 3
    g = rlx.Graph("jvp_b")
    a = g.input("A", [n, n], "f64")
    b = g.input("b", [n], "f64")
    x = g.dense_solve(a, b, [n], "f64")
    g.set_outputs([x])

    jg = rlx.jvp(g, [b])
    sess = rlx.Session(device="cpu")
    compiled = sess.compile(jg)

    A = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ], dtype=np.float64)
    b_vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tb = np.array([0.5, -0.25, 1.0], dtype=np.float64)

    outs = compiled.run_typed({
        "A":         (A.tobytes(), "f64"),
        "b":         (b_vec.tobytes(), "f64"),
        "tangent_b": (tb.tobytes(), "f64"),
    })
    assert len(outs) == 2
    primal = np.frombuffer(outs[0][0], dtype=np.float64)
    tangent = np.frombuffer(outs[1][0], dtype=np.float64)
    np.testing.assert_allclose(primal, np.linalg.solve(A, b_vec), atol=1e-10)
    np.testing.assert_allclose(tangent, np.linalg.solve(A, tb), atol=1e-10)


def test_jacfwd_dense_solve_recovers_inverse() -> None:
    """∂x/∂b = A⁻¹ when x = solve(A, b). Use jacfwd to materialize it."""
    n = 3
    g = rlx.Graph("jac_inverse")
    a = g.input("A", [n, n], "f64")
    b = g.input("b", [n], "f64")
    x = g.dense_solve(a, b, [n], "f64")
    g.set_outputs([x])

    jg = rlx.jvp(g, [b])
    compiled = rlx.Session(device="cpu").compile(jg)

    A = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ], dtype=np.float64)
    b_vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    jac_list = rlx.jacfwd(
        compiled,
        primals={"A": A, "b": b_vec},
        wrt_name="b",
        wrt_shape=(n,),
        dtype="f64",
    )
    assert len(jac_list) == 1, "one primal output, one Jacobian"
    jac = jac_list[0]
    # Shape: (output_size, *wrt_shape) = (3, 3).
    assert jac.shape == (n, n), f"expected (3,3), got {jac.shape}"
    np.testing.assert_allclose(jac, np.linalg.inv(A), atol=1e-10)
