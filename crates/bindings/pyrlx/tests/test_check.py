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
"""basic tests for pyrlx — mirror rlx-runtime's end_to_end_session test
and exercise the same flow on every backend present in this build.
"""

import numpy as np
import pytest

import pyrlx as rlx


def test_available_devices_includes_cpu():
    devs = rlx.available_devices()
    assert "cpu" in devs, devs
    assert rlx.is_available("cpu")


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        rlx.is_available("definitely-not-a-device")


def test_unsupported_device_error_is_actionable():
    """The error must tell the user how to fix it."""
    if rlx.is_available("cuda"):
        pytest.skip("cuda is available — no error expected")
    with pytest.raises(RuntimeError, match=r"--features cuda"):
        rlx.Session(device="cuda")


def _matmul_bias_gelu_graph():
    g = rlx.Graph("matmul_bias_gelu")
    x   = g.input("x", [2, 4], "f32")
    w   = g.param("w", [4, 3], "f32")
    b   = g.param("b", [3],    "f32")
    mm  = g.matmul(x, w)
    add = g.add(mm, b)
    out = g.gelu(add)
    g.set_outputs([out])
    return g


def _set_canonical_params(compiled):
    compiled.set_param("w", np.array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
    ], dtype=np.float32))
    compiled.set_param("b", np.array([0.5, -0.5, 0.0], dtype=np.float32))


def test_end_to_end_cpu_matches_rust_test():
    sess     = rlx.Session(device="cpu")
    compiled = sess.compile(_matmul_bias_gelu_graph())
    _set_canonical_params(compiled)

    x = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    [y] = compiled.run({"x": x})
    assert y.shape == (2, 3), y.shape

    # Gelu values from rlx-runtime's end_to_end_session test
    assert abs(y[0, 0] - 1.399) < 0.01
    assert abs(y[0, 1] - -0.154) < 0.01
    assert abs(y[0, 2]) < 0.01
    assert abs(y[1, 0] - 0.346) < 0.01
    assert abs(y[1, 1] - 0.346) < 0.01


def test_compile_consumes_graph():
    g = _matmul_bias_gelu_graph()
    sess = rlx.Session(device="cpu")
    sess.compile(g)
    with pytest.raises(RuntimeError):
        sess.compile(g)


def test_session_repr():
    sess = rlx.Session(device="cpu", precision="f32")
    assert "cpu" in repr(sess)
    assert sess.device == "cpu"
    assert sess.precision == "f32"


@pytest.mark.parametrize("dev", ["metal", "mlx", "cuda", "rocm", "gpu"])
def test_optional_backend_round_trip(dev):
    """If the backend is available, the same graph must produce the
    same result as CPU (within numerical tolerance)."""
    if not rlx.is_available(dev):
        pytest.skip(f"{dev} not built into this pyrlx")

    sess     = rlx.Session(device=dev)
    compiled = sess.compile(_matmul_bias_gelu_graph())
    _set_canonical_params(compiled)
    x = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    [y] = compiled.run({"x": x})
    assert y.shape == (2, 3)
    assert abs(y[0, 0] - 1.399) < 0.05  # looser tol for f16-ish backends
