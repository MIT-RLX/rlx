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
"""Pythonic DSL: `pyrlx.graph(...)` + `Node` operator overloading.

Verifies that the fluent surface mirrors the native `Graph` builder for
shape inference, execution, typed I/O, and every backend compiled into
this wheel.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import pyrlx as rlx


def _build_dsl_graph() -> rlx.Graph:
    """Build the same graph as `_matmul_bias_gelu_graph` via the DSL."""
    with rlx.graph("matmul_bias_gelu_dsl") as g:
        x = g.input("x", [2, 4], "f32")
        w = g.param("w", [4, 3], "f32")
        b = g.param("b", [3], "f32")
        g.outputs = [(x @ w + b).gelu()]
    return g.raw


def _set_canonical_params(compiled) -> None:
    rlx.set_param(compiled, "w", np.array([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        0, 0, 0,
    ], dtype=np.float32))
    rlx.set_param(compiled, "b", np.array([0.5, -0.5, 0.0], dtype=np.float32))


def _compile_dsl(build_fn) -> rlx.CompiledGraph:
    return rlx.Session(device="cpu").compile(build_fn())


# ── Canonical matmul+bias+gelu ─────────────────────────────────────

def test_dsl_matches_explicit_form_cpu():
    sess = rlx.Session(device="cpu")
    compiled = sess.compile(_build_dsl_graph())
    _set_canonical_params(compiled)

    x = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)
    [y] = rlx.run(compiled, x=x)

    assert y.shape == (2, 3), y.shape
    assert abs(y[0, 0] - 1.399) < 0.01
    assert abs(y[0, 1] - -0.154) < 0.01
    assert abs(y[1, 0] - 0.346) < 0.01


@pytest.mark.parametrize("dev", ["metal", "mlx", "cuda", "rocm", "gpu"])
def test_dsl_optional_backend_round_trip(dev):
    """DSL graphs must match CPU numerics on every compiled-in backend."""
    if not rlx.is_available(dev):
        pytest.skip(f"{dev} not built into this pyrlx")

    compiled = rlx.Session(device=dev).compile(_build_dsl_graph())
    _set_canonical_params(compiled)
    x = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    [y] = rlx.run(compiled, x=x)
    assert y.shape == (2, 3)
    assert abs(y[0, 0] - 1.399) < 0.05


# ── Node surface ──────────────────────────────────────────────────

def test_node_int_coercion():
    """`Node` must coerce to `int` so it interops with native methods."""
    with rlx.graph("coerce") as g:
        x = g.input("x", [2, 2], "f32")
        assert isinstance(x, rlx.Node)
        assert isinstance(int(x), int)
        assert int(x) == x.id
        y = g.raw.relu(x)
        assert isinstance(y, int)


def test_node_shape_dtype_introspection():
    with rlx.graph("introspect") as g:
        x = g.input("x", [2, 4], "f32")
        w = g.param("w", [4, 3], "f32")
        y = x @ w
        assert y.shape == (2, 3)
        assert y.dtype == "f32"


def test_node_repr_includes_shape():
    with rlx.graph("repr") as g:
        x = g.input("x", [3, 5], "f32")
        r = repr(x)
        assert "id=" in r
        assert "(3, 5)" in r
        assert "f32" in r


def test_outputs_setter_rejects_get():
    with rlx.graph("write_only") as g:
        x = g.input("x", [2], "f32")
        g.outputs = [x]
        with pytest.raises(AttributeError, match="write-only"):
            _ = g.outputs


def test_unwrap_node_args_to_native_ops():
    with rlx.graph("forward") as g:
        x = g.input("x", [2, 4], "f32")
        w = g.param("w", [4, 3], "f32")
        y = g.matmul(x, w)
        assert isinstance(y, rlx.Node)
        assert y.shape == (2, 3)


def test_node_arithmetic_chain():
    with rlx.graph("arith") as g:
        x = g.input("x", [2], "f32")
        y = g.input("y", [2], "f32")
        g.outputs = [(x + y) * x - y]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(
        compiled,
        x=np.array([2.0, 3.0], dtype=np.float32),
        y=np.array([1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(out, np.array([5.0, 13.0], dtype=np.float32))


def test_scalar_literals_auto_promote_in_arithmetic():
    with rlx.graph("scalar_mul") as g:
        x = g.input("x", [3], "f32")
        g.outputs = [x * 2.0]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([2.0, 4.0, 6.0], dtype=np.float32))


def test_scalar_literal_on_left_hand_side():
    with rlx.graph("scalar_lhs") as g:
        x = g.input("x", [2], "f32")
        g.outputs = [3.0 - x]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([2.0, -1.0], dtype=np.float32))


def test_graph_constant_native():
    with rlx.graph("const") as g:
        x = g.input("x", [2], "f32")
        c = g.constant(0.5)
        assert c.shape == ()
        assert c.dtype == "f32"
        g.outputs = [x + c]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([1.5, 2.5], dtype=np.float32))


def test_dsl_int_literal_matches_tensor_dtype():
    with rlx.graph("i32") as g:
        x = g.input("x", [2], "i32")
        g.outputs = [x + 1]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(compiled, x=np.array([4, 10], dtype=np.int32))
    np.testing.assert_array_equal(out, np.array([5, 11], dtype=np.int32))


def test_dsl_divide_by_scalar():
    with rlx.graph("div") as g:
        x = g.input("x", [3], "f32")
        g.outputs = [x / 2.0]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(compiled, x=np.array([2.0, 4.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_dsl_scalar_add_sub_chain():
    with rlx.graph("chain") as g:
        x = g.input("x", [4], "f32")
        g.outputs = [(x + 1.0) / 2.0]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


@pytest.mark.parametrize("dev", ["cpu", "metal"])
def test_dsl_scalar_literals_on_available_backends(dev):
    if not rlx.is_available(dev):
        pytest.skip(f"{dev} not built into this pyrlx")
    with rlx.graph("be_scalar") as g:
        x = g.input("x", [3], "f32")
        g.outputs = [x * 3.0 + 1.0]
    compiled = rlx.Session(device=dev).compile(g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([4.0, 7.0, 10.0], dtype=np.float32))


def test_node_negation():
    with rlx.graph("neg") as g:
        x = g.input("x", [2], "f32")
        g.outputs = [-x]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(compiled, x=np.array([1.0, -2.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([-1.0, 2.0], dtype=np.float32))


def test_node_fluent_reductions():
    with rlx.graph("red") as g:
        x = g.input("x", [2, 3, 4], "f32")
        assert x.sum([1]).shape == (2, 4)
        assert x.mean([2], keep_dim=True).shape == (2, 3, 1)
        assert x.cumsum(axis=-1).shape == (2, 3, 4)


def test_node_transpose_and_reshape():
    with rlx.graph("shape") as g:
        x = g.input("x", [2, 3, 4], "f32")
        assert x.transpose(0, 2, 1).shape == (2, 4, 3)
        assert x.T.shape == (4, 3, 2)
        assert x.reshape(6, 4).shape == (6, 4)


# ── Typed I/O helpers ─────────────────────────────────────────────

def test_run_auto_dtype_f32():
    with rlx.graph("identity") as g:
        x = g.input("x", [2, 2], "f32")
        g.outputs = [x.relu()]
    compiled = rlx.Session(device="cpu").compile(g.raw)
    out, = rlx.run(compiled, x=np.array([[1.0, -1.0], [-2.0, 3.0]], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32))


def test_set_param_auto_dtype_f32():
    with rlx.graph("scale") as g:
        x = g.input("x", [3], "f32")
        w = g.param("w", [3], "f32")
        g.outputs = [x * w]
    compiled = rlx.Session(device="cpu").compile(g.raw)
    rlx.set_param(compiled, "w", np.array([2.0, 3.0, 4.0], dtype=np.float32))
    out, = rlx.run(compiled, x=np.array([1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(out, np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_run_typed_f64():
    with rlx.graph("f64") as g:
        x = g.input("x", [3], "f64")
        s = g.param("s", [3], "f64")
        g.outputs = [x * s]
    compiled = rlx.Session(device="cpu").compile(g.raw)
    rlx.set_param(compiled, "s", np.array([2.0, 3.0, 4.0], dtype=np.float64))
    out, = rlx.run(compiled, x=np.array([1.0, 1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(out, np.array([2.0, 3.0, 4.0], dtype=np.float64))


def test_dtype_str_round_trips():
    assert rlx.dtype_str(np.float32) == "f32"
    assert rlx.dtype_str(np.float64) == "f64"
    assert rlx.dtype_str(np.int32) == "i32"
    assert rlx.numpy_dtype("f32") == np.dtype(np.float32)
    assert rlx.numpy_dtype("f64") == np.dtype(np.float64)


# ── IR parity via DSL (shape inference) ──────────────────────────

def test_dsl_io_shape_inference():
    with rlx.graph("io") as g:
        x = g.input("x", [4, 8], "f32")
        w = g.param("w", [8, 16], "f32")
        assert x.shape == (4, 8)
        assert w.shape == (8, 16)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_dsl_binary_inferred(op):
    with rlx.graph("bin") as g:
        x = g.input("x", [2, 3], "f32")
        y = g.input("y", [2, 3], "f32")
        z = g.binary(op, x, y)
        assert z.shape == (2, 3)


@pytest.mark.parametrize("kind", ["gelu", "silu", "relu", "tanh", "exp", "sqrt", "neg"])
def test_dsl_activation_inferred(kind):
    with rlx.graph("act") as g:
        x = g.input("x", [2, 3], "f32")
        y = g.activation(kind, x)
        assert y.shape == (2, 3)


def test_dsl_compare_and_where():
    with rlx.graph("cmp") as g:
        x = g.input("x", [3], "f32")
        y = g.input("y", [3], "f32")
        c = g.input("cond", [3], "bool")
        lt = g.lt_(x, y)
        assert lt.dtype == "bool"
        assert lt.shape == (3,)
        out = g.where_(c, x, y)
        assert out.shape == (3,)


def _f32_payload(**inputs):
    return {
        k: (np.ascontiguousarray(v, dtype=np.float32).tobytes(), "f32")
        for k, v in inputs.items()
    }


def _decode_bool_out(raw: bytes, count: int) -> np.ndarray:
    """Bool outputs are byte-packed; buffer is zero-padded to a 4-byte boundary."""
    return np.frombuffer(raw, dtype=np.uint8)[:count].astype(np.bool_)


def test_dsl_compare_executes_cpu():
    with rlx.graph("cmp_exec") as g:
        x = g.input("x", [3], "f32")
        y = g.input("y", [3], "f32")
        g.outputs = [g.lt_(x, y)]
        graph = g.raw
    compiled = rlx.Session("cpu").compile(graph)
    (raw, dt), = compiled.run_typed(_f32_payload(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([2.0, 2.0, 1.0]),
    ))
    assert dt == "bool"
    np.testing.assert_array_equal(_decode_bool_out(raw, 3), [True, False, False])


def test_dsl_where_executes_cpu():
    with rlx.graph("where_exec") as g:
        x = g.input("x", [3], "f32")
        y = g.input("y", [3], "f32")
        cond = x < y
        g.outputs = [cond.where_(x, y)]
        graph = g.raw
    compiled = rlx.Session("cpu").compile(graph)
    (raw, dt), = compiled.run_typed(_f32_payload(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([2.0, 2.0, 1.0]),
    ))
    assert dt == "f32"
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype=np.float32), [1.0, 2.0, 1.0]
    )


def test_dsl_node_compare_operators_execute_cpu():
    with rlx.graph("cmp_ops_exec") as g:
        x = g.input("x", [3], "f32")
        y = g.input("y", [3], "f32")
        g.outputs = [x < y, x >= y, x == y]
        graph = g.raw
    compiled = rlx.Session("cpu").compile(graph)
    payload = _f32_payload(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([2.0, 2.0, 1.0]),
    )
    outs = compiled.run_typed(payload)
    lt_out = _decode_bool_out(outs[0][0], 3)
    ge_out = _decode_bool_out(outs[1][0], 3)
    eq_out = _decode_bool_out(outs[2][0], 3)
    np.testing.assert_array_equal(lt_out, [True, False, False])
    np.testing.assert_array_equal(ge_out, [False, True, True])
    np.testing.assert_array_equal(eq_out, [False, True, False])


def test_dsl_shape_ops_via_proxy():
    with rlx.graph("shape_ops") as g:
        a = g.input("a", [2, 3], "f32")
        b = g.input("b", [2, 4], "f32")
        assert g.concat([a, b], axis=1).shape == (2, 7)
        x = g.input("x", [10, 4], "f32")
        assert x.narrow(0, 2, 5).shape == (5, 4)
        tb = g.param("tb", [10, 4], "f32")
        ix = g.input("ix", [3], "i64")
        assert g.gather(tb, ix, axis=0).shape == (3, 4)
        assert x.cast("f16").dtype == "f16"


def test_dsl_normalization_shapes():
    with rlx.graph("norm") as g:
        x = g.input("x", [2, 4], "f32")
        gm = g.param("g", [4], "f32")
        bt = g.param("b", [4], "f32")
        assert g.layer_norm(x, gm, bt).shape == (2, 4)
        assert g.rms_norm(x, gm, bt).shape == (2, 4)


def test_dsl_attention_and_rope_shapes():
    with rlx.graph("attn") as g:
        q = g.input("q", [1, 2, 4, 8], "f32")
        k = g.input("k", [1, 2, 4, 8], "f32")
        v = g.input("v", [1, 2, 4, 8], "f32")
        out = g.attention_kind(q, k, v, num_heads=2, head_dim=8, mask_kind="causal")
        assert out.shape == (1, 2, 4, 8)
    with rlx.graph("rope") as g:
        x = g.input("x", [1, 4, 8], "f32")
        cos = g.param("cos", [4, 4], "f32")
        sin = g.param("sin", [4, 4], "f32")
        assert g.rope(x, cos, sin, head_dim=8).shape == (1, 4, 8)


def test_dsl_resize_nearest_2x_shape():
    with rlx.graph("resize") as g:
        x = g.input("x", [1, 3, 4, 4], "f32")
        assert g.resize_nearest_2x(x).shape == (1, 3, 8, 8)


def test_dsl_fft_ops_shapes_and_tuple_returns():
    with rlx.graph("fft") as g:
        x = g.input("x", [8], "f32")
        assert g.fft_norm(x, inverse=False, norm="forward").shape == (8,)
        re, im = g.fft_real(x, norm="forward")
        assert isinstance(re, rlx.Node)
        assert isinstance(im, rlx.Node)
        assert re.shape == (8,)
        assert im.shape == (8,)
        re2, im2 = g.rfft(x, norm="forward")
        assert g.irfft(re2, im2, 8, norm="forward").shape == (8,)
        assert g.fftfreq(8).shape == (8,)
        assert g.rfftfreq(8).shape == (5,)


# ── Functional pipelines via DSL ─────────────────────────────────

def test_dsl_full_pipeline_runs_on_cpu():
    with rlx.graph("pipeline") as g:
        x = g.input("x", [2, 4], "f32")
        w1 = g.param("w1", [4, 8], "f32")
        w2 = g.param("w2", [8, 4], "f32")
        h = ((x @ w1).gelu() @ w2).softmax()
        g.outputs = [h]
    compiled = _compile_dsl(lambda: g.raw)
    rlx.set_param(compiled, "w1", np.zeros((4, 8), dtype=np.float32))
    rlx.set_param(compiled, "w2", np.zeros((8, 4), dtype=np.float32))
    out, = rlx.run(compiled, x=np.ones((2, 4), dtype=np.float32))
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(2), rtol=1e-4)


def test_dsl_fft_round_trip_via_proxy():
    with rlx.graph("fft_rt") as g:
        x = g.input("x", [8], "f32")
        y = g.fft_norm(x, inverse=False, norm="forward")
        z = g.fft_norm(y, inverse=True, norm="forward")
        g.outputs = [z]
    compiled = _compile_dsl(lambda: g.raw)
    signal = [1.0, 0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0]
    x_bytes = b"".join(struct.pack("<f", v) for v in signal)
    raw, _ = compiled.run_typed({"x": (x_bytes, "f32")})[0]
    got = struct.unpack("<8f", raw)
    for a, b in zip(got, signal):
        assert abs(a - b) < 1e-4


def test_dsl_concat_runs_on_cpu():
    with rlx.graph("concat_run") as g:
        a = g.input("a", [2, 3], "f32")
        b = g.input("b", [2, 4], "f32")
        g.outputs = [g.concat([a, b], axis=1)]
    compiled = _compile_dsl(lambda: g.raw)
    out, = rlx.run(
        compiled,
        a=np.ones((2, 3), dtype=np.float32),
        b=np.zeros((2, 4), dtype=np.float32),
    )
    assert out.shape == (2, 7)
    np.testing.assert_array_equal(out[:, :3], np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(out[:, 3:], np.zeros((2, 4), dtype=np.float32))


@pytest.mark.parametrize("dev", ["cpu", "metal"])
def test_dsl_fluent_matmul_on_available_backends(dev):
    if not rlx.is_available(dev):
        pytest.skip(f"{dev} not built into this pyrlx")
    with rlx.graph("mm") as g:
        x = g.input("x", [2, 2], "f32")
        g.outputs = [(x @ x.T).relu()]
    compiled = rlx.Session(device=dev).compile(g.raw)
    out, = rlx.run(compiled, x=np.eye(2, dtype=np.float32))
    np.testing.assert_array_equal(out, np.eye(2, dtype=np.float32))


def test_dsl_node_comparison_operators():
    with rlx.graph("cmp_ops") as g:
        x = g.input("x", [3], "f32")
        y = g.input("y", [3], "f32")
        assert (x < y).dtype == "bool"
        assert (x <= 1.0).shape == (3,)
        assert (x == y).dtype == "bool"
        assert (x != 0.0).dtype == "bool"
        g.outputs = [x.where_(x > 0.0, y)]


def test_dsl_matmul_scalar_scales_elementwise():
    with rlx.graph("mm_scale") as g:
        x = g.input("x", [2, 3], "f32")
        g.outputs = [x @ 2.0]
    compiled = rlx.Session("cpu").compile(g.raw)
    out, = rlx.run(compiled, x=np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(out, np.full((2, 3), 2.0, dtype=np.float32))


def test_dsl_node_layer_norm_method():
    with rlx.graph("ln_method") as g:
        x = g.input("x", [2, 4], "f32")
        gm = g.param("g", [4], "f32")
        bt = g.param("b", [4], "f32")
        y = x.layer_norm(gm, bt)
        g.outputs = [y]
    assert y.shape == (2, 4)


def test_dsl_import_from_dsl_module():
    from pyrlx.dsl import Node, graph
    with graph("submod") as g:
        x = g.input("x", [2], "f32")
        y = x * 3.0
        g.outputs = [y]
    assert isinstance(y, Node)


def test_dsl_int_scalar_oob_raises():
    with rlx.graph("oob") as g:
        x = g.input("x", [4], "i8")
        with pytest.raises(ValueError, match="out of range"):
            _ = x + 300


def test_dsl_float_literal_honors_int_dtype():
    with rlx.graph("int_lit") as g:
        x = g.input("x", [4], "i8")
        y = x + 2.0
        g.outputs = [y]
    assert y.dtype == "i8"


def test_dsl_fractional_int_literal_raises():
    with rlx.graph("frac") as g:
        x = g.input("x", [4], "i32")
        with pytest.raises(ValueError, match="integral"):
            _ = x + 2.5


def test_dsl_huge_int_literal_raises():
    with rlx.graph("huge") as g:
        x = g.input("x", [2], "i64")
        with pytest.raises(ValueError, match="2\\*\\*53"):
            _ = x + 10**30
    with rlx.graph("huge_const") as g:
        with pytest.raises(ValueError, match="2\\*\\*53"):
            g.constant(10**30, "i64")
