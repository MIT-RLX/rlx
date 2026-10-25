# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Picking a backend: what this build has, what suits a graph, and the ways to
choose — name it, ask for the fastest, let a policy decide per graph, or
benchmark and route per run.

    python backends.py
"""

import numpy as np

import pyrlx


def model():
    g = pyrlx.Graph("mlp")
    x = g.input("x", [4, 16])
    g.set_outputs([g.tanh(g.matmul(x, g.param("w", [16, 16])))])
    return g


x = {"x": ((np.arange(64) % 7) / 7).astype(np.float32).reshape(4, 16)}
w = ((np.arange(256) % 5) / 10 - 0.2).astype(np.float32).reshape(16, 16)

available = pyrlx.available_devices()
print("devices:", " ".join(available))
print("metal available:", pyrlx.is_available("metal"))
print("fastest for graph:", pyrlx.fastest_device_for(model()))

reference = None
for device in available:
    c = pyrlx.Session(device).compile(model())
    c.set_param("w", w)
    y = c.run(x)[0]
    reference = y if reference is None else reference
    print(f"session {device}: max|Δ| vs first = {np.abs(y - reference).max():.2e}")

policy = pyrlx.DevicePolicy.all().with_prefer(["metal", "cpu"])
picked = pyrlx.FlexibleSession(policy).compile_resolved(model())
print("flexible session picked:", picked.device)

runner = pyrlx.GraphDevices(model())
runner.set_param("w", w)
print("runner warmed:", " ".join(runner.warm_all()))
for row in runner.benchmark(x, 3):
    print(f"  bench {row.label}: exec_ms={row.median_exec_ns / 1e6:.4f}")
print("runner resolves to:", runner.resolve())
device, _ = runner.run_try(["metal", "cpu"], x)
print("run_try chain:", device)

router = pyrlx.DeviceRouter(model())
router.set_param("w", w)
print("router ran on:", router.run_chain(x)[0])
