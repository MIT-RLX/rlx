#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Minimal DSL example: build, compile, run on the first available backend.
#
#   cd pyrlx && maturin develop --features cpu,metal
#   python examples/dsl_quickstart.py

from __future__ import annotations

import sys

import numpy as np

import pyrlx as rlx


def main() -> int:
    devices = rlx.available_devices()
    if not devices:
        print("no RLX backends in this wheel — rebuild with maturin --features cpu", file=sys.stderr)
        return 1
    device = devices[0]
    print(f"using device={device!r} (available: {devices})")

    with rlx.graph("dsl_quickstart") as g:
        x = g.input("x", [2, 4], "f32")
        w = g.param("w", [4, 3], "f32")
        b = g.param("b", [3], "f32")
        # operator syntax + scalar literal scale
        y = (x @ w + b).gelu() * 2.0
        g.outputs = [y]
        graph = g.raw

    compiled = rlx.Session(device).compile(graph)
    rlx.set_param(compiled, "w", np.full((4, 3), 0.25, dtype=np.float32))
    rlx.set_param(compiled, "b", np.zeros(3, dtype=np.float32))
    x_in = np.arange(8, dtype=np.float32).reshape(2, 4)
    out, = rlx.run(compiled, x=x_in)

    print("input x:\n", x_in)
    print("output y shape:", out.shape)
    print("output y (first row):", out[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
