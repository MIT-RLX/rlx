# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""``Model``: a graph saved as an ``.rlxp`` model package, opened and run by
name like any provider's model (feature ``models``)."""

import numpy as np
import pytest

import pyrlx as rlx

pytestmark = pytest.mark.skipif(not hasattr(rlx, "Model"), reason="built without `models`")


def affine_package(tmp_path):
    g = rlx.Graph("affine")
    x = g.input("x", [1, 2], "f32")
    w = g.param("w", [2, 2], "f32")
    g.set_outputs([g.matmul(x, w)])
    path = str(tmp_path / "affine.rlxp")
    rlx.save_model(g, path, {"w": np.array([2, 0, 0, 3], np.float32)},
                   {"task": "graph", "family": "affine", "outputs": ["y"]})
    return path


def test_a_saved_graph_runs_as_a_model(tmp_path):
    m = rlx.Model.open(affine_package(tmp_path), device="cpu")
    info = m.info()
    assert (info["provider"], info["family"], info["task"]) == ("rlxp", "affine", "graph")
    assert info["inputs"] == [{"name": "x", "dtype": "f32", "shape": [1, 2]}]
    (y,) = m.run({"x": np.ones((1, 2), np.float32)})
    assert y.shape == (1, 2) and y.tolist() == [[2.0, 3.0]]
    assert m.output_names() == ["y"] and m.output_shapes() == [[1, 2]]
    ((data, dtype),) = m.run_typed({"x": (np.ones(2, np.float32).tobytes(), "f32")})
    assert dtype == "f32" and np.frombuffer(data, np.float32).tolist() == [2.0, 3.0]


def test_providers_and_errors_name_what_exists(tmp_path):
    assert any(p["name"] == "rlxp" for p in rlx.model_providers())
    with pytest.raises(Exception, match="no model provider claims"):
        rlx.Model.open(str(tmp_path / "nothing.bin"))
    m = rlx.Model.open(affine_package(tmp_path))
    with pytest.raises(Exception, match="missing input 'x'"):
        m.run({})
    with pytest.raises(Exception, match="does not generate text"):
        m.generate_text("hi")
