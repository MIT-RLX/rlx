# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# RLX — pyrlx binding coverage for recently added Graph / DSL surfaces.

from __future__ import annotations

import numpy as np
import pytest

import pyrlx as rlx


def _compile(g: rlx.Graph):
    return rlx.Session(device="cpu").compile(g)


def test_gelu_approx_shape():
    g = rlx.Graph("gelu_approx")
    x = g.input("x", [4], "f32")
    y = g.gelu_approx(x)
    assert g.shape_of(y) == ([4], "f32")


def test_stop_gradient_preserves_shape():
    g = rlx.Graph("stop")
    x = g.input("x", [2, 3], "f32")
    y = g.stop_gradient(x)
    assert g.shape_of(y) == ([2, 3], "f32")


def test_conv2d_inferred_shape():
    g = rlx.Graph("conv")
    x = g.input("x", [1, 4, 8, 8], "f32")
    w = g.param("w", [8, 2, 3, 3], "f32")
    y = g.conv2d(x, w, [3, 3], [1, 1], [1, 1], [1, 1], groups=2)
    assert g.shape_of(y) == ([1, 8, 8, 8], "f32")


def test_layer_norm2d_and_group_norm_shapes():
    g = rlx.Graph("norm2d")
    x = g.input("x", [1, 4, 8, 8], "f32")
    gamma = g.param("g", [4], "f32")
    beta = g.param("b", [4], "f32")
    assert g.shape_of(g.layer_norm2d(x, gamma, beta)) == ([1, 4, 8, 8], "f32")
    assert g.shape_of(g.group_norm(x, gamma, beta, num_groups=2)) == ([1, 4, 8, 8], "f32")


def test_rope_n_shape():
    g = rlx.Graph("rope_n")
    x = g.input("x", [1, 4, 8], "f32")
    cos = g.param("cos", [4, 4], "f32")
    sin = g.param("sin", [4, 4], "f32")
    y = g.rope_n(x, cos, sin, head_dim=8, n_rot=4)
    assert g.shape_of(y) == ([1, 4, 8], "f32")


def test_graph_compare_shortcuts_execute():
    g = rlx.Graph("cmp_short")
    x = g.input("x", [3], "f32")
    y = g.input("y", [3], "f32")
    g.set_outputs([g.gt_(x, y), g.ge_(x, y), g.ne_(x, y)])
    compiled = _compile(g)
    payload = {
        "x": (np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes(), "f32"),
        "y": (np.array([2.0, 2.0, 1.0], dtype=np.float32).tobytes(), "f32"),
    }
    outs = compiled.run_typed(payload)
    def _bool(raw):
        return np.frombuffer(raw, dtype=np.uint8)[:3].astype(np.bool_)
    gt_out = _bool(outs[0][0])
    ge_out = _bool(outs[1][0])
    ne_out = _bool(outs[2][0])
    np.testing.assert_array_equal(gt_out, [False, False, True])
    np.testing.assert_array_equal(ge_out, [False, True, True])
    np.testing.assert_array_equal(ne_out, [True, False, True])


def test_proxy_add_with_scalar_literal():
    with rlx.graph("proxy_scalar") as g:
        x = g.input("x", [3], "f32")
        g.outputs = [g.add(x, 1.0)]
    compiled = _compile(g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_dsl_pow_with_scalar():
    with rlx.graph("pow") as g:
        x = g.input("x", [3], "f32")
        g.outputs = [x ** 2.0]
    compiled = _compile(g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([1.0, 4.0, 9.0], dtype=np.float32))


def test_dsl_gelu_approx_runs():
    with rlx.graph("ga") as g:
        x = g.input("x", [2], "f32")
        g.outputs = [x.gelu_approx()]
    compiled = _compile(g.raw)
    out, = rlx.run(compiled, x=np.array([0.0, 1.0], dtype=np.float32))
    assert out.shape == (2,)
    assert out[1] > out[0]
