# RLX - versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Licensed under the GNU General Public License, version 3.

"""FKL batch region with FusionOptions.native_fk()."""

import numpy as np
import pytest

import pyrlx as rlx


def test_fusion_options_native_fk_preset():
    fo = rlx.FusionOptions.native_fk()
    assert fo.native_fk_regions is True
    assert fo.decompose_fusion_regions is False


@pytest.mark.parametrize("device", ["metal", "cpu", "tpu"])
def test_batch_narrow_relu_matches_cpu(device):
    if not rlx.is_available(device):
        pytest.skip(f"{device} not in this build")

    batch_n, c, h, w = 2, 3, 8, 8
    batch = np.linspace(-0.3, 0.3, batch_n * c * h * w, dtype=np.float32).reshape(
        batch_n, c, h, w
    )
    inp = {"batch": batch}

    ref_g = rlx.batch_narrow_relu_graph("ref", batch_n, c, h, w)
    ref = rlx.Session(device="cpu").compile(ref_g).run(inp)[0]

    g = rlx.batch_narrow_relu_graph("fused", batch_n, c, h, w)
    opts = rlx.FusionOptions.native_fk()
    kd = "native" if device != "cpu" else None
    out = rlx.Session(device=device).compile_with(
        g, fusion_options=opts, kernel_dispatch=kd
    ).run(inp)[0]

    np.testing.assert_allclose(ref, out, rtol=0, atol=1e-4)
