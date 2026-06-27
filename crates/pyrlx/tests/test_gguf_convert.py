# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pyrlx as rlx


def test_convert_safetensors_to_gguf(tmp_path: Path):
    st = pytest.importorskip("safetensors")
    numpy_mod = pytest.importorskip("safetensors.numpy")
    save_file = numpy_mod.save_file

    st_path = tmp_path / "tiny.safetensors"
    out_path = tmp_path / "tiny.gguf"
    w = np.sin(np.arange(512, dtype=np.float32).reshape(2, 256) * 0.01)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    save_file({"blk.weight": w, "blk.bias": b}, str(st_path))

    report = rlx.convert_to_gguf(
        str(st_path),
        str(out_path),
        "Q4_K",
        architecture="test",
        scheme_overrides={"blk.weight": "Q8_0"},
    )
    assert report.tensors == 2
    assert report.compression_ratio() > 1.0
    assert out_path.is_file()

    schemes = dict(report.schemes())
    assert schemes["blk.weight"] == "Q8_0"
    assert schemes["blk.bias"] == "F32"

    gguf = rlx.load_gguf(str(out_path))
    assert gguf.architecture == "test"
    back = gguf.dequant_tensor("blk.weight")
    assert back.shape == (512,)
    cos = float(np.dot(back, w.ravel()) / (np.linalg.norm(back) * np.linalg.norm(w) + 1e-8))
    assert cos > 0.99


def test_convert_unknown_scheme(tmp_path: Path):
    st_path = tmp_path / "empty.safetensors"
    st_path.write_bytes(b"not gguf")
    with pytest.raises((ValueError, FileNotFoundError)):
        rlx.convert_to_gguf(str(st_path), str(tmp_path / "out.gguf"), "NOT_A_SCHEME")
