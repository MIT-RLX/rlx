# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import pyrlx as rlx


def test_quantize_iq2_xxs_roundtrip():
    # Structured weights — random Gaussian is a poor fit for IQ2_XXS codebooks.
    i = np.arange(256, dtype=np.float32)
    w = np.sin(i * 0.03) * 0.5
    packed = rlx.quantize(w, dtype="IQ2_XXS")
    assert packed.dtype == np.uint8
    assert packed.ndim == 1
    back = rlx.dequant(packed, dtype="IQ2_XXS", num_elements=256)
    assert back.shape == (256,)
    cos = float(np.dot(back, w) / (np.linalg.norm(back) * np.linalg.norm(w) + 1e-8))
    assert cos > 0.80


@pytest.mark.parametrize(
    "dtype",
    ["Q4_1", "IQ4_NL", "TQ2_0", "MXFP4"],
)
def test_quantize_dequant_smoke(dtype: str):
    rng = np.random.default_rng(1)
    n = 256 if dtype != "MXFP4" else 32
    w = rng.standard_normal(n, dtype=np.float32)
    packed = rlx.quantize(w, dtype=dtype)
    back = rlx.dequant(packed, dtype=dtype, num_elements=n)
    assert back.shape == (n,)
    assert np.isfinite(back).all()


def test_quantize_unknown_dtype():
    w = np.zeros(32, dtype=np.float32)
    with pytest.raises(ValueError, match="unknown GGUF dtype"):
        rlx.quantize(w, dtype="NOT_A_SCHEME")
