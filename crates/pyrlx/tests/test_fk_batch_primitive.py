# RLX - versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Licensed under the GNU General Public License, version 3.

"""Primitive narrow+relu+concat fuses via MarkBatchSliceRegions on GPU session compile."""

import numpy as np
import pytest

import pyrlx as rlx


@pytest.mark.parametrize("device", ["metal", "cpu"])
def test_primitive_batch_default_session_matches_cpu(device):
    if not rlx.is_available(device):
        pytest.skip(f"{device} not in this build")

    batch_n, c, h, w = 2, 3, 8, 8
    batch = np.linspace(-0.2, 0.2, batch_n * c * h * w, dtype=np.float32).reshape(
        batch_n, c, h, w
    )
    inp = {"batch": batch}

    ref = (
        rlx.Session(device="cpu")
        .compile(rlx.batch_narrow_relu_graph("ref", batch_n, c, h, w))
        .run(inp)[0]
    )

    g = rlx.batch_narrow_relu_graph("prim", batch_n, c, h, w)
    out = rlx.Session(device=device).compile(g).run(inp)[0]

    np.testing.assert_allclose(ref, out, rtol=0, atol=1e-4)
