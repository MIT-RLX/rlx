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

"""Export a tiny MatMul+Relu ONNX (opset 11, IR 8) for rlx-onnx vs ORT benches."""
from __future__ import annotations

import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(512, 512) * 0.02)
        self.b = nn.Parameter(torch.zeros(512))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.w + self.b)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_matmul_relu.onnx"
    m = M().eval()
    x = torch.randn(1, 32, 512)
    torch.onnx.export(
        m,
        x,
        out,
        input_names=["x"],
        output_names=["y"],
        opset_version=11,
        dynamo=False,
    )
    model = onnx.load(out)
    model.ir_version = 8
    g = model.graph
    del g.initializer[:]
    for init in onnx.load(out).graph.initializer:
        arr = onnx.numpy_helper.to_array(init).astype(np.float32)
        t = TensorProto()
        t.name = init.name
        t.dims.extend(arr.shape)
        t.data_type = TensorProto.FLOAT
        t.float_data.extend(arr.flatten().tolist())
        g.initializer.append(t)
    onnx.save(model, out)
    print(f"wrote {out} (ir={model.ir_version} opset={model.opset_import[0].version})")


if __name__ == "__main__":
    main()
