# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Training: fit y = x·w_true with the Trainer (forward + backward + optimizer
compiled once), checkpoint and resume it, then the same fit by hand with
`grad` and an Optimizer.

    python training.py [device]
"""

import math
import os
import sys
import tempfile

import numpy as np

import pyrlx

W_TRUE = np.array([2, -1, 0.5], np.float32)
BATCH = 8


def regression():
    g = pyrlx.Graph("regression")
    x = g.input("x", [BATCH, 3])
    y = g.input("y", [BATCH, 1])
    w = g.param("w", [3, 1])
    d = g.sub(g.matmul(x, w), y)
    g.set_outputs([g.mean(g.mul(d, d), [0, 1])])
    return g, w


def data(step):
    i = np.arange(BATCH)[:, None]
    k = np.arange(3)[None, :]
    x = (((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1).astype(np.float32)
    return {"x": x, "y": (x @ W_TRUE)[:, None].astype(np.float32)}


device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
g, w = regression()

# ── Trainer: `run` pulls batches from a callback ────────────────────────────
trainer = pyrlx.Trainer(
    g.clone(), wrt=["w"], init={"w": np.zeros(3, np.float32)},
    device=device, optimizer={"kind": "adamw", "lr": 0.1},
)
first = trainer.evaluate(data(0))
summary = trainer.run(
    400,
    data,
    lr=lambda s: 0.1 * 0.5 * (1 + math.cos(math.pi * s / 400)),  # cosine decay
    on_step=lambda s, loss: loss > 1e-6,  # stop once the loss is tiny
)
print(f"trainer loss: {first:.4f} -> {trainer.evaluate(data(0)):.6f} in {trainer.steps()} steps")
print("trainer w:", " ".join(f"{v:.3f}" for v in trainer.params()["w"].ravel()))

# Checkpoint (GGUF: params + optimizer state), restore into a fresh trainer.
path = os.path.join(tempfile.gettempdir(), "rlx-example-train.gguf")
written = trainer.save(path)
restored = pyrlx.Trainer(g.clone(), wrt=["w"], init={"w": np.zeros(3, np.float32)}, device=device)
report = restored.load(path)
print(f"checkpoint: {written} tensors, restored step {report['step']}")
os.remove(path)

# Freezing: a frozen param keeps its value through steps.
restored.freeze(["w"])
before = restored.params()["w"].copy()
restored.step(data(1))
print("frozen:", restored.frozen(), "unchanged:", bool((restored.params()["w"] == before).all()))
restored.unfreeze()

# ── By hand: grad → compile → Optimizer ─────────────────────────────────────
backward = pyrlx.Session(device).compile(pyrlx.grad(g, [w]))  # outputs [loss, dw]
opt = pyrlx.Optimizer("sgd", lr=0.2)
params = np.zeros((3, 1), np.float32)
for step in range(300):
    backward.set_param("w", params)
    loss, dw = backward.run({**data(step), "d_output": np.ones(1, np.float32)})
    params = opt.step({"w": (params, dw.reshape(3, 1))})["w"]
print(f"manual loss: {float(loss.ravel()[0]):.6f} w:", " ".join(f"{v:.3f}" for v in params.ravel()))
