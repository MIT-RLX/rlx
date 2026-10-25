# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The wider ``rlx_ir::Graph`` surface on ``pyrlx.Graph``: shape inference per
op family, closed-form numerics on CPU, the table-driven activation / binary
parsers, RoPE pairing, and DSL forwarding of the new builders."""

import numpy as np
import pytest

import pyrlx as rlx


def dims(g, n):
    return tuple(g.shape_of(n)[0])


def test_shape_manipulation_and_ordering_infer_shapes():
    g = rlx.Graph("ext")
    x = g.input("x", [2, 3], "f32")
    a = g.input("a", [4, 3], "f32")
    sq = g.input("sq", [3, 3], "f64")
    got = [
        dims(g, g.clamp(x, 0.0, 1.0)),
        dims(g, g.pad(x, [(1, 1), (2, 0)], "reflect")),
        dims(g, g.slice(x, 1, 0, 2)),
        dims(g, g.tile(x, [2, 1])),
        dims(g, g.trilu(sq, True)),
        dims(g, g.roll(x, [1], [0])),
        dims(g, g.reverse(x, [1])),
        dims(g, g.cumprod(x, 1)),
        dims(g, g.argmax(x, 1)),
        dims(g, g.sort(x, 1, True)),
        dims(g, g.qr(a, "q")), dims(g, g.qr(a, "r")),
        dims(g, g.svd(a, "u")), dims(g, g.svd(a, "s")), dims(g, g.svd(a, "vt")),
        dims(g, g.det(sq)),
        dims(g, g.cholesky(sq)),
        dims(g, g.eigh(sq)[1]),
        dims(g, g.histogram(x, 8, 0.0, 1.0)),
        dims(g, g.zeros([2, 2])),
        dims(g, g.expand(g.input("e", [1, 3], "f32"), [4, 3])),
    ]
    assert got == [
        (2, 3), (4, 5), (2, 2), (4, 3), (3, 3), (2, 3), (2, 3), (2, 3), (2,), (2, 3),
        (4, 3), (3, 3), (4, 3), (3,), (3, 3), (), (3, 3), (3, 3), (8,), (2, 2), (4, 3),
    ]


def test_layer_ops_infer_shapes():
    g = rlx.Graph("layers")
    img = g.input("img", [1, 2, 4, 4], "f32")
    vol = g.input("vol", [1, 2, 4, 4, 4], "f32")
    sig = g.input("sig", [1, 2, 8], "f32")
    seq = g.input("seq", [2, 5, 4], "f32")
    q = g.input("q", [1, 6, 8], "f32")
    k = g.input("k", [1, 6, 8], "f32")
    v12 = g.input("v12", [1, 6, 12], "f32")
    v8 = g.input("v8", [1, 6, 8], "f32")
    c = g.param("c", [2], "f32")
    m = g.param("m", [4, 4], "f32")
    got = [
        dims(g, g.conv3d(vol, g.param("w3", [3, 2, 3, 3, 3], "f32"), padding=[1])),
        dims(g, g.conv_transpose1d(sig, g.param("w1", [2, 3, 4], "f32"), 4, 2, 1)),
        dims(g, g.im2col(img, [3, 3], padding=[1, 1])),
        dims(g, g.max_pool2d(img, [2, 2])),
        dims(g, g.avg_pool2d(img, [2, 2], stride=[1, 1])),
        dims(g, g.pool2d(img, "max", [2, 2])),
        dims(g, g.interpolate1d(sig, 16, "linear")),
        dims(g, g.interpolate3d(vol, [8, 8, 8])),
        dims(g, g.batch_norm_inference(img, c, c, c, c)),
        dims(g, g.batch_norm_training(img, c, c)),
        dims(g, g.lstm(seq, g.param("wih", [16, 4], "f32"), g.param("whh", [16, 3], "f32"), g.param("b", [16], "f32"), 3)),
        dims(g, g.gru(seq, g.param("gih", [9, 4], "f32"), g.param("ghh", [9, 3], "f32"), g.param("gb1", [9], "f32"),
                      g.param("gb2", [9], "f32"), 3, bidirectional=True)),
        dims(g, g.rnn(seq, g.param("rih", [3, 4], "f32"), g.param("rhh", [3, 3], "f32"), g.param("rb", [3], "f32"), 3)),
        dims(g, g.attention_kind_vdim(q, k, v12, 2, 4, 6)),
        dims(g, g.attention_opts(q, k, v8, 2, 4, scale=0.5, softcap=30.0)),
        dims(g, g.scatter_nd(m, g.input("si", [2, 1], "i64"), g.input("su", [2, 4], "f32"), "add")),
        dims(g, g.gather_nd(m, g.input("gi", [3, 2], "i64"))),
        dims(g, g.triangular_solve(m, g.input("rhs", [4, 2], "f32"))),
        dims(g, g.linear_bias(seq, g.param("lw", [4, 6], "f32"), g.param("lb", [6], "f32"))),
        dims(g, g.linear_bias_act(seq, g.param("law", [4, 6], "f32"), None, "gelu")),
        dims(g, g.scaled_quantize(m)[0]),
        dims(g, g.vector_quantize(m, g.param("cb", [8, 4], "f32"))[0]),
        len(g.residual_vq(m, [g.param("cb1", [8, 4], "f32"), g.param("cb2", [8, 4], "f32")])),
    ]
    assert got == [
        (1, 3, 4, 4, 4), (1, 3, 16), (16, 18), (1, 2, 2, 2), (1, 2, 3, 3), (1, 2, 2, 2),
        (1, 2, 16), (1, 2, 8, 8, 8), (1, 2, 4, 4), (1, 2, 4, 4), (2, 5, 3), (2, 5, 6),
        (2, 5, 3), (1, 6, 12), (1, 6, 8), (4, 4), (3,), (4, 2), (2, 5, 6), (2, 5, 6),
        (4, 4), (4,), 3,
    ]


def test_signal_ops_infer_shapes():
    g = rlx.Graph("signal")
    x = g.input("x", [2, 16], "f32")
    spd = g.input("spd", [3, 3], "f32")
    batch = g.input("batch", [4, 3, 3], "f32")
    got = [
        dims(g, g.fir_conv1d(x, [0.25, 0.5, 0.25], "same")),
        dims(g, g.fir_filtfilt(x, [0.5, 0.5])),
        dims(g, g.biquad(x, [1, 0, 0], [1, 0, 0])),
        dims(g, g.sosfilt(x, [[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]])),
        dims(g, g.iirfilt(x, [1.0], [1.0, -0.5])),
        dims(g, g.resample_poly(x, 2, 1, [0.5, 1.0, 0.5])),
        dims(g, g.hilbert(x)[0]),
        dims(g, g.envelope(x)),
        dims(g, g.bimap(g.param("bw", [2, 3], "f32"), spd)),
        dims(g, g.spd_matrix_fn(batch, "logm")),
        dims(g, g.spd_karcher_mean(batch)),
        dims(g, g.spd_log_map(spd, spd)),
    ]
    assert got == [
        (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 32), (2, 16), (2, 16),
        (2, 2), (4, 3, 3), (3, 3), (3, 3),
    ]


def test_layer_ops_compute_closed_forms_on_cpu():
    g = rlx.Graph("num")
    img = g.input("img", [1, 1, 2, 4], "f32")
    a = g.input("a", [2, 2], "f32")
    b = g.input("b", [2, 1], "f32")
    e = g.input("e", [1, 3], "f32")
    s = g.input("s", [4], "f32")
    xs = g.input("xs", [3, 2], "f32")
    init = g.input("init", [2], "f32")
    body = rlx.Graph("step")
    carry = body.input("carry", [2], "f32")
    xt = body.input("xt", [2], "f32")
    body.set_outputs([body.add(carry, xt)])
    g.set_outputs([
        g.max_pool2d(img, [2, 2]),
        g.triangular_solve(a, b, True, False),
        g.expand(e, [2, 3]),
        g.scan(init, body, 3, xs=[xs]),
        g.biquad(s, [1, 0, 0], [1, 0, 0]),
        g.activation("softplus", g.constant(0.0)),
        g.binary("atan2", g.constant(1.0), g.constant(1.0)),
    ])
    c = rlx.Session("cpu").compile(g)
    outs = rlx.run(
        c,
        img=np.array([1, 5, 2, 0, 3, 4, 7, 8], np.float32),
        a=np.array([2, 0, 1, 4], np.float32),
        b=np.array([4, 10], np.float32),
        e=np.array([1, 2, 3], np.float32),
        s=np.array([1, -2, 3, -4], np.float32),
        xs=np.arange(1, 7, dtype=np.float32),
        init=np.array([0, 100], np.float32),
    )
    want = [
        [5, 8], [2, 2], [1, 2, 3, 1, 2, 3], [9, 112], [1, -2, 3, -4],
        [np.log(2.0)], [np.pi / 4],
    ]
    for got, ref in zip(outs, want):
        np.testing.assert_allclose(np.ravel(got), ref, rtol=1e-5, atol=1e-6)


def test_every_activation_and_binary_op_parses_by_name():
    g = rlx.Graph("names")
    x = g.input("x", [4], "f32")
    acts = ["gelu", "gelu_approx", "silu", "relu", "sigmoid", "tanh", "exp", "log", "sqrt",
            "rsqrt", "neg", "abs", "sin", "cos", "tan", "atan", "recip", "round", "floor",
            "ceil", "sign", "softplus", "elu", "erf", "hard_swish", "hardSigmoid", "mish",
            "softsign", "log-sigmoid"]
    bins = ["add", "sub", "mul", "div", "max", "min", "pow", "mod", "bit_and", "bitOr",
            "bitxor", "shl", "shr", "atan2", "%", "**", "<<"]
    for name in acts:
        g.activation(name, x)
    for name in bins:
        g.binary(name, x, x)
    with pytest.raises(ValueError, match="LogSigmoid"):
        g.activation("nope", x)


@pytest.mark.parametrize("style,want", [("neox", [-3, 2, 1, 4]), ("gptj", [-2, 1, 3, 4])])
def test_rope_style_selects_the_pairing(style, want):
    # A 90° rotation on the first frequency only: NeoX pairs (i, i + d/2),
    # GPT-J pairs (2i, 2i + 1), so the two move different elements.
    g = rlx.Graph("rope")
    x = g.input("x", [1, 1, 4], "f32")
    cos = g.input("cos", [1, 2], "f32")
    sin = g.input("sin", [1, 2], "f32")
    g.set_outputs([g.rope(x, cos, sin, 4, style=style)])
    c = rlx.Session("cpu").compile(g)
    (out,) = rlx.run(
        c,
        x=np.array([1, 2, 3, 4], np.float32),
        cos=np.array([0, 1], np.float32),
        sin=np.array([1, 0], np.float32),
    )
    np.testing.assert_allclose(np.ravel(out), want, atol=1e-6)


def test_quant_scheme_names_resolve():
    g = rlx.Graph("q")
    x = g.input("x", [2, 64], "f32")
    packed = g.param("w", [4096], "u8")
    for scheme in ["Q4_K", "q8_0", "IQ2_XXS"]:
        assert dims(g, g.dequant_matmul_packed(x, packed, scheme, 32)) == (2, 32)
    with pytest.raises(ValueError, match="unknown quant scheme"):
        g.dequant_matmul_packed(x, packed, "Q9_Z", 32)


def test_new_builders_chain_through_the_dsl():
    with rlx.graph("chain") as g:
        x = g.input("x", [1, 2, 4, 4])
        y = g.batch_norm_training(
            g.resize_nearest_2x(g.max_pool2d(x, [2, 2])),
            g.param("g", [2]),
            g.param("b", [2]),
        )
        re, im = g.hilbert(g.input("s", [2, 16]))
        assert y.shape == (1, 2, 4, 4)
        assert re.shape == im.shape == (2, 16)


def test_introspection_lists_params_and_op_kinds():
    g = rlx.Graph("intro")
    x = g.input("x", [2, 3], "f32")
    w = g.param("w", [3, 4], "f32")
    y = g.relu(g.matmul(x, w))
    g.set_outputs([y])
    assert g.param_names() == ["w"]
    assert "MatMul" in g.op_kinds()
    assert g.outputs() == [y]
