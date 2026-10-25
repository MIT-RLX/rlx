# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Inference: build a graph, compile it for a device, set its weights, run it.

    python inference.py [device]

A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
The Swift, JS and C examples of the same name do the same steps.
"""

import sys

import numpy as np

import pyrlx


def classifier():
    g = pyrlx.Graph("classifier")
    x = g.input("x", [1, 4])
    hidden = g.relu(g.matmul(x, g.param("w1", [4, 8])))
    g.set_outputs([g.softmax(g.matmul(hidden, g.param("w2", [8, 3])))])
    return g


def weights(n, seed):
    return ((np.arange(n) * 7919 + seed * 104_729) % 1000 / 1000 - 0.5).astype(np.float32)


device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
session = pyrlx.Session(device)
model = session.compile(classifier())  # compile consumes the graph
print("compiled on:", model.device)
model.set_params({"w1": weights(32, 1).reshape(4, 8), "w2": weights(24, 2).reshape(8, 3)})

probs = model.run({"x": np.array([[0.5, -1, 2, 0.25]], np.float32)})[0]
print("probs:", " ".join(f"{p:.4f}" for p in probs.ravel()))
print(f"probs sum: {probs.sum():.4f}")
print("output shapes:", model.output_shapes(), "dtypes:", model.output_dtypes())
