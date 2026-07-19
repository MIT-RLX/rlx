#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Emit a minimal FLUX/F5-style adaLN ONNX fixture for rlx-onnx-conformance.

Graph (affine-free LayerNorm + broadcast modulation):
  n = LayerNormalization(x, gamma=1, beta=0)
  out = n * (1 + Expand(scale, [B,S,D])) + Expand(shift, [B,S,D])
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

B, S, D = 2, 4, 8
EPS = 1e-5
OPSET = 17


def main() -> None:
    default_out = (
        Path(__file__).resolve().parent.parent / "tests/fixtures/dit_adaln.onnx"
    )
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default_out

    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, S, D])
    scale_info = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [B, 1, D])
    shift_info = helper.make_tensor_value_info("shift", TensorProto.FLOAT, [B, 1, D])
    out_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, [B, S, D])

    gamma = numpy_helper.from_array(np.ones(D, dtype=np.float32), name="gamma")
    beta = numpy_helper.from_array(np.zeros(D, dtype=np.float32), name="beta")
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one")
    expand_shape = numpy_helper.from_array(
        np.array([B, S, D], dtype=np.int64), name="expand_shape"
    )

    nodes = [
        helper.make_node(
            "LayerNormalization",
            inputs=["x", "gamma", "beta"],
            outputs=["n"],
            name="ln",
            axis=-1,
            epsilon=EPS,
        ),
        helper.make_node(
            "Expand",
            inputs=["scale", "expand_shape"],
            outputs=["scale_e"],
            name="expand_scale",
        ),
        helper.make_node(
            "Add",
            inputs=["one", "scale_e"],
            outputs=["one_plus_scale"],
            name="one_plus_scale",
        ),
        helper.make_node(
            "Mul",
            inputs=["n", "one_plus_scale"],
            outputs=["scaled"],
            name="scale_norm",
        ),
        helper.make_node(
            "Expand",
            inputs=["shift", "expand_shape"],
            outputs=["shift_e"],
            name="expand_shift",
        ),
        helper.make_node(
            "Add",
            inputs=["scaled", "shift_e"],
            outputs=["out"],
            name="add_shift",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "dit_adaln",
        [x_info, scale_info, shift_info],
        [out_info],
        initializer=[gamma, beta, one, expand_shape],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="rlx-onnx-conformance",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    print(f"wrote {out} (ir={model.ir_version} opset={OPSET} shape=[{B},{S},{D}])")


if __name__ == "__main__":
    main()
