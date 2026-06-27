# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pyrlx as rlx


def test_write_load_gguf_f32_roundtrip(tmp_path: Path):
    path = tmp_path / "tiny.gguf"
    w = np.arange(6, dtype=np.float32)
    b = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    rlx.write_gguf(
        str(path),
        {
            "w": {"data": w, "shape": [2, 3], "dtype": "F32"},
            "b": {"data": b, "shape": [3], "dtype": "F32"},
        },
        architecture="test",
        metadata={"general.name": "unit"},
    )
    f = rlx.load_gguf(str(path))
    assert f.architecture == "test"
    assert f.metadata()["general.name"] == "unit"
    assert set(f.tensor_names()) == {"w", "b"}
    info = f.tensor_info("w")
    assert info["shape"] == [2, 3]
    assert info["dtype"] == "F32"
    np.testing.assert_allclose(f.dequant_tensor("w"), w)
    np.testing.assert_allclose(f.dequant_tensor("b"), b)


def test_write_load_gguf_quantized_roundtrip(tmp_path: Path):
    path = tmp_path / "q.gguf"
    n = 256
    rng = np.random.default_rng(0)
    w = rng.standard_normal(n, dtype=np.float32) * 0.2
    packed = rlx.quantize(w, dtype="Q8_0")
    rlx.write_gguf(
        str(path),
        {"w": {"data": packed, "shape": [n], "dtype": "Q8_0"}},
        architecture="test",
    )
    f = rlx.load_gguf(str(path))
    back = f.dequant_tensor("w")
    assert back.shape == (n,)
    cos = float(np.dot(back, w) / (np.linalg.norm(back) * np.linalg.norm(w) + 1e-8))
    assert cos > 0.99


def test_write_gguf_quantize_on_write(tmp_path: Path):
    path = tmp_path / "q4.gguf"
    w = np.sin(np.arange(256, dtype=np.float32) * 0.03)
    rlx.write_gguf(
        str(path),
        {"w": {"data": w, "shape": [256], "dtype": "Q4_0"}},
    )
    f = rlx.load_gguf(str(path))
    back = f.dequant_tensor("w")
    assert back.shape == (256,)
    assert np.isfinite(back).all()


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        rlx.load_gguf(str(tmp_path / "nope.gguf"))
