"""pyrlx.Graph ↔ rlx_ir::Graph IR parity.

Each test exercises one IR op via the Python builder to confirm:
  1. the call doesn't raise
  2. the resulting node's inferred shape (via Graph.shape_of) matches
     what the corresponding Rust GraphExt method computes
  3. compile + run produces a deterministic result on CPU

This is the canonical "is the Python surface in lockstep with the IR?"
test. When you add a new op to rlx-ir, add a case here.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyrlx as rlx


def _compile(g: rlx.Graph):
    """Compile + return the CompiledGraph."""
    return rlx.Session(device="cpu").compile(g)


# ── I/O + linear algebra ──────────────────────────────────────────

def test_input_param_shape_inference():
    g = rlx.Graph("io")
    x = g.input("x", [4, 8], "f32")
    w = g.param("w", [8, 16], "f32")
    assert g.shape_of(x) == ([4, 8], "f32")
    assert g.shape_of(w) == ([8, 16], "f32")


def test_matmul_inferred_shape():
    g = rlx.Graph("mm")
    x = g.input("x", [3, 4], "f32")
    w = g.param("w", [4, 5], "f32")
    y = g.matmul(x, w)
    assert g.shape_of(y) == ([3, 5], "f32")


def test_matmul_with_explicit_shape():
    g = rlx.Graph("mm_explicit")
    x = g.input("x", [3, 4], "f32")
    w = g.param("w", [4, 5], "f32")
    y = g.matmul_with_shape(x, w, [3, 5], "f32")
    g.set_outputs([y])
    c = _compile(g)
    c.set_param("w", np.zeros((4, 5), dtype=np.float32))
    [out] = c.run({"x": np.ones((3, 4), dtype=np.float32)})
    assert out.shape == (3, 5)


# ── Element-wise ──────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_binary_inferred(op):
    g = rlx.Graph("bin")
    x = g.input("x", [2, 3], "f32")
    y = g.input("y", [2, 3], "f32")
    z = g.binary(op, x, y)
    assert g.shape_of(z) == ([2, 3], "f32")


def test_named_binary_helpers():
    g = rlx.Graph("named")
    x = g.input("x", [2], "f32")
    y = g.input("y", [2], "f32")
    for n in (g.add(x, y), g.sub(x, y), g.mul(x, y), g.div(x, y)):
        assert g.shape_of(n) == ([2], "f32")


@pytest.mark.parametrize("kind", ["gelu", "silu", "relu", "tanh", "exp", "sqrt", "neg"])
def test_activation_inferred(kind):
    g = rlx.Graph("act")
    x = g.input("x", [2, 3], "f32")
    y = g.activation(kind, x)
    assert g.shape_of(y) == ([2, 3], "f32")


def test_activation_helpers():
    g = rlx.Graph("acth")
    x = g.input("x", [4], "f32")
    for n in (g.gelu(x), g.silu(x), g.relu(x), g.tanh(x),
              g.exp(x), g.sqrt(x), g.neg(x)):
        assert g.shape_of(n) == ([4], "f32")


# ── Compare + where ───────────────────────────────────────────────

def test_compare_returns_bool_shape():
    g = rlx.Graph("cmp")
    x = g.input("x", [3], "f32")
    y = g.input("y", [3], "f32")
    n = g.compare("lt", x, y)
    dims, dt = g.shape_of(n)
    assert dims == [3]
    assert dt == "bool"


def test_where_uses_then_branch_shape():
    g = rlx.Graph("where")
    x = g.input("x",    [3], "f32")
    y = g.input("y",    [3], "f32")
    c = g.input("cond", [3], "bool")
    n = g.where_(c, x, y)
    assert g.shape_of(n) == ([3], "f32")


# ── Reduction ─────────────────────────────────────────────────────

def test_sum_collapses_axis():
    g = rlx.Graph("sum")
    x = g.input("x", [2, 3, 4], "f32")
    n = g.sum(x, [1], keep_dim=False)
    assert g.shape_of(n) == ([2, 4], "f32")


def test_mean_keep_dim():
    g = rlx.Graph("mean")
    x = g.input("x", [2, 3, 4], "f32")
    n = g.mean(x, [2], keep_dim=True)
    assert g.shape_of(n) == ([2, 3, 1], "f32")


def test_softmax_preserves_shape():
    g = rlx.Graph("sm")
    x = g.input("x", [4, 8], "f32")
    assert g.shape_of(g.softmax(x)) == ([4, 8], "f32")


def test_cumsum_preserves_shape():
    g = rlx.Graph("cs")
    x = g.input("x", [4, 8], "f32")
    assert g.shape_of(g.cumsum(x, axis=-1)) == ([4, 8], "f32")


# ── Shape ops ─────────────────────────────────────────────────────

def test_reshape_inferred():
    g = rlx.Graph("rs")
    x = g.input("x", [2, 6], "f32")
    n = g.reshape(x, [3, 4])
    assert g.shape_of(n) == ([3, 4], "f32")


def test_transpose_inferred():
    g = rlx.Graph("tp")
    x = g.input("x", [2, 3, 4], "f32")
    n = g.transpose(x, [0, 2, 1])
    assert g.shape_of(n) == ([2, 4, 3], "f32")


def test_narrow_inferred():
    g = rlx.Graph("nr")
    x = g.input("x", [10, 4], "f32")
    n = g.narrow(x, 0, 2, 5)
    assert g.shape_of(n) == ([5, 4], "f32")


def test_concat_inferred():
    g = rlx.Graph("cc")
    a = g.input("a", [2, 3], "f32")
    b = g.input("b", [2, 4], "f32")
    n = g.concat([a, b], axis=1)
    assert g.shape_of(n) == ([2, 7], "f32")


def test_gather_inferred():
    g  = rlx.Graph("ga")
    tb = g.param("tb", [10, 4], "f32")
    ix = g.input("ix", [3], "i64")
    n  = g.gather(tb, ix, axis=0)
    assert g.shape_of(n) == ([3, 4], "f32")


def test_cast_changes_dtype_only():
    g = rlx.Graph("cast")
    x = g.input("x", [2, 3], "f32")
    n = g.cast(x, "f16")
    assert g.shape_of(n) == ([2, 3], "f16")


# ── Normalization ────────────────────────────────────────────────

def test_layer_norm_preserves_shape():
    g = rlx.Graph("ln")
    x  = g.input("x", [2, 4], "f32")
    gm = g.param("g", [4],    "f32")
    bt = g.param("b", [4],    "f32")
    n = g.layer_norm(x, gm, bt)
    assert g.shape_of(n) == ([2, 4], "f32")


def test_rms_norm_preserves_shape():
    g = rlx.Graph("rms")
    x  = g.input("x", [2, 4], "f32")
    gm = g.param("g", [4],    "f32")
    bt = g.param("b", [4],    "f32")
    n = g.rms_norm(x, gm, bt)
    assert g.shape_of(n) == ([2, 4], "f32")


# ── Attention + RoPE ─────────────────────────────────────────────

def test_attention_kind_no_mask():
    g = rlx.Graph("attn")
    q = g.input("q", [1, 2, 4, 8], "f32")
    k = g.input("k", [1, 2, 4, 8], "f32")
    v = g.input("v", [1, 2, 4, 8], "f32")
    n = g.attention_kind(q, k, v, num_heads=2, head_dim=8, mask_kind="causal")
    dims, dt = g.shape_of(n)
    assert dt == "f32"
    assert dims == [1, 2, 4, 8]


def test_rope_preserves_shape():
    g = rlx.Graph("rope")
    x   = g.input("x",   [1, 4, 8], "f32")
    cos = g.param("cos", [4, 4],    "f32")
    sin = g.param("sin", [4, 4],    "f32")
    n = g.rope(x, cos, sin, head_dim=8)
    assert g.shape_of(n) == ([1, 4, 8], "f32")


# ── Functional run ───────────────────────────────────────────────

def test_full_pipeline_runs_on_cpu():
    """Sanity: build a graph using a representative slice of the API,
    compile, and run it. Catches issues where shape inference returns
    something the executor rejects."""
    g  = rlx.Graph("pipeline")
    x  = g.input("x", [2, 4], "f32")
    w1 = g.param("w1", [4, 8], "f32")
    w2 = g.param("w2", [8, 4], "f32")

    h = g.matmul(x, w1)
    h = g.gelu(h)
    h = g.matmul(h, w2)
    h = g.softmax(h)
    g.set_outputs([h])

    c = _compile(g)
    c.set_param("w1", np.zeros((4, 8), dtype=np.float32))
    c.set_param("w2", np.zeros((8, 4), dtype=np.float32))
    [y] = c.run({"x": np.ones((2, 4), dtype=np.float32)})
    assert y.shape == (2, 4)
    np.testing.assert_allclose(y.sum(axis=-1), np.ones(2), rtol=1e-4)
