# RLX - versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""FKL resize prologue via pyrlx Graph + Session.compile_with."""

import numpy as np
import pytest

import pyrlx as rlx


def _resize_relu_add_graph():
    g = rlx.Graph("fk_resize")
    x = g.input("x", [1, 3, 8, 8], "f32")
    a = g.input("a", [1, 3, 16, 16], "f32")
    up = g.resize_nearest_2x(x)
    r = g.relu(up)
    g.set_outputs([g.add(r, a)])
    return g


def _primitive_resize_relu_add():
    g = rlx.Graph("fk_ref")
    x = g.input("x", [1, 3, 8, 8], "f32")
    a = g.input("a", [1, 3, 16, 16], "f32")
    up = g.resize_nearest_2x(x)
    r = g.relu(up)
    g.set_outputs([g.add(r, a)])
    return g


def test_resize_nearest_2x_shape_inference():
    g = rlx.Graph("t")
    x = g.input("x", [1, 3, 4, 4], "f32")
    up = g.resize_nearest_2x(x)
    g.set_outputs([up])
    # compile on cpu only to exercise builder (no shape API on Graph)
    out = rlx.Session(device="cpu").compile(g).run(
        {"x": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    )[0]
    assert out.shape == (1, 3, 8, 8)


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_resize_relu_add_matches_primitives(device):
    if not rlx.is_available(device):
        pytest.skip(f"{device} not in this build")

    x = np.linspace(-0.5, 0.5, 1 * 3 * 8 * 8, dtype=np.float32).reshape(1, 3, 8, 8)
    a = np.linspace(0, 1, 1 * 3 * 16 * 16, dtype=np.float32).reshape(1, 3, 16, 16)
    inp = {"x": x, "a": a}

    ref = rlx.Session(device="cpu").compile(_primitive_resize_relu_add()).run(inp)[0]

    g = _resize_relu_add_graph()
    kd = "native" if device != "cpu" else None
    out = rlx.Session(device=device).compile_with(
        g, fusion_options=rlx.FusionOptions(), kernel_dispatch=kd
    ).run(inp)[0]

    np.testing.assert_allclose(ref, out, rtol=0, atol=1e-4)


@pytest.mark.skipif(not rlx.is_available("metal"), reason="metal required")
def test_flexible_session_compile_with_resolved_metal():
    x = np.linspace(-0.5, 0.5, 1 * 3 * 8 * 8, dtype=np.float32).reshape(1, 3, 8, 8)
    a = np.linspace(0, 1, 1 * 3 * 16 * 16, dtype=np.float32).reshape(1, 3, 16, 16)
    inp = {"x": x, "a": a}

    ref = rlx.Session(device="cpu").compile(_primitive_resize_relu_add()).run(inp)[0]

    g = _resize_relu_add_graph()
    fs = rlx.FlexibleSession()
    out = fs.compile_with_resolved(
        g, device="metal", fusion_options=rlx.FusionOptions(), kernel_dispatch="native"
    ).run(inp)[0]
    np.testing.assert_allclose(ref, out, rtol=0, atol=1e-4)
