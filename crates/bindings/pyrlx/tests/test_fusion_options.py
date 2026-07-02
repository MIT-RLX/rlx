# RLX - versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Licensed under the GNU General Public License, version 3.

"""FusionOptions and Session.compile_with."""

import numpy as np
import pyrlx as rlx


def _add_mul_graph():
    g = rlx.Graph("fk")
    x = g.input("x", [4], "f32")
    y = g.input("y", [4], "f32")
    g.set_outputs([g.mul(g.add(x, y), y)])
    return g


def test_fusion_options_defaults():
    fo = rlx.FusionOptions()
    assert fo.fk_fusion is True
    assert fo.fuse_region_prologue is True
    assert fo.native_fk_regions is False


def test_compile_with_cpu_matches_default_compile():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    inp = {"x": x, "y": y}

    g1 = _add_mul_graph()
    g2 = _add_mul_graph()
    sess = rlx.Session(device="cpu")
    out_default = sess.compile(g1).run(inp)[0]
    out_opts = sess.compile_with(g2, fusion_options=rlx.FusionOptions()).run(inp)[0]
    np.testing.assert_allclose(out_default, out_opts, rtol=0, atol=1e-5)
