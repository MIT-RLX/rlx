# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

#
# splice / parallel / sequential / extract / Partition (CPU): combined and
# split graphs compute what the originals do.

from __future__ import annotations

import numpy as np
import pytest

import pyrlx as rlx

W0 = np.array([2, 2, 2], dtype=np.float32)
W1 = np.array([1, 2, 3], dtype=np.float32)
X = np.array([1, 2, 3], dtype=np.float32)


def layer() -> rlx.Graph:
    """y = x * w over [3]."""
    g = rlx.Graph("layer")
    x = g.input("x", [3], "f32")
    w = g.param("w", [3], "f32")
    g.set_outputs([g.mul(x, w)])
    return g


def run(g: rlx.Graph, params: dict, inputs: dict) -> list[np.ndarray]:
    c = rlx.Session(device="cpu").compile(g.clone())
    for name in g.param_names():
        c.set_param(name, params[name])
    return [np.asarray(o).ravel() for o in c.run(inputs)]


def stack() -> tuple[rlx.Graph, int]:
    g = rlx.Graph("stack")
    [h0] = rlx.splice(g, layer(), prefix="l0.")
    [h1] = rlx.splice(g, layer(), prefix="l1.", bind={"x": h0})
    g.set_outputs([h1])
    return g, h0


PARAMS = {"l0.w": W0, "l1.w": W1}


def test_splice_prefix_and_bind_stack_layers():
    g, _ = stack()
    assert g.param_names() == ["l0.w", "l1.w"]
    [y] = run(g, PARAMS, {"l0.x": X})
    np.testing.assert_allclose(y, X * W0 * W1)


def test_sequential_ties_weights_and_parallel_concatenates():
    s = rlx.sequential(layer(), layer())
    assert s.param_names() == ["w"]
    [y] = run(s, {"w": W1}, {"x": X})
    np.testing.assert_allclose(y, X * W1 * W1)

    p = rlx.parallel(layer(), s, name="both")
    a, b = run(p, {"w": W1}, {"x": X})
    np.testing.assert_allclose(a, X * W1)
    np.testing.assert_allclose(b, X * W1 * W1)


def test_split_parts_run_in_order_and_merge_back():
    g, h0 = stack()
    [whole] = run(g, PARAMS, {"l0.x": X})

    p = rlx.Partition.split_at(g, [h0])
    assert len(p) == p.num_parts() == 2
    assert p.imports(1) == p.exports(0)
    env = {"l0.x": X}
    outs = []
    for i in range(len(p)):
        feed = {n: env[n] for n in p.imports(i)}
        if i == 0:
            feed["l0.x"] = X
        o = run(p.graph(i), PARAMS, feed)
        env.update(zip(p.exports(i), o))
        outs.append(o)
    [(pi, j)] = p.outputs()
    np.testing.assert_allclose(outs[pi][j], whole)
    np.testing.assert_allclose(run(p.merge(), PARAMS, {"l0.x": X})[0], whole)


def test_partition_by_node_list_and_components():
    g, _ = stack()
    n = g.node_count()
    p = rlx.Partition(g, [0 if i < n // 2 else 1 for i in range(n)])
    np.testing.assert_allclose(run(p.merge(), PARAMS, {"l0.x": X})[0], X * W0 * W1)

    two = rlx.Graph("two")
    x = two.input("x", [3], "f32")
    a, b = two.exp(x), two.mul(x, x)
    two.set_outputs([a, b])
    c = rlx.Partition.components(two)
    assert len(c) == 2 and c.imports(1) == [] and c.outputs() == [(0, 0), (1, 0)]
    [sq] = run(rlx.extract(two, [b]), {}, {"x": X})
    np.testing.assert_allclose(sq, X * X)


def test_errors_name_the_problem_and_leave_the_target_unchanged():
    g, h0 = stack()
    before = g.node_count()
    with pytest.raises(ValueError, match='"nope"'):
        rlx.splice(g, layer(), bind={"nope": h0})
    assert g.node_count() == before
    with pytest.raises(ValueError, match="free inputs"):
        rlx.sequential(rlx.parallel(layer(), layer()), layer())
    with pytest.raises(ValueError, match="part 5"):
        rlx.Partition.components(layer()).graph(5)
