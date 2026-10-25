# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Distributed: a mesh of rlx nodes. Here the three ranks are threads of one
process on loopback (every blocking call releases the GIL); across machines
give each rank the same `peers` (or `discover`) and run one rank per host.

    python distributed.py

1. Collectives. 2. Pipelined inference: rank 0 ships each worker a stage; the
worker compiles it on its own hardware and loads its weights locally. 3.
Data-parallel training: every rank trains its shard, gradients averaged.
"""

import os
import tempfile
import threading

import numpy as np

import pyrlx

WORLD = 3
PER_RANK = 16
tmp = tempfile.mkdtemp(prefix="rlx-dist-")


def blob(name, values):
    path = os.path.join(tmp, name)
    np.asarray(values, np.float32).tofile(path)
    return "file://" + path


def stage():
    g = pyrlx.Graph("stage")
    x = g.input("x", [1, 4])
    g.set_outputs([g.relu(g.matmul(x, g.param("w", [4, 4])))])
    return g


def training_graph():
    g = pyrlx.Graph("lr")
    x = g.input("X", [4, 3])
    y = g.input("y", [4, 1])
    w = g.param("w", [3, 1])
    d = g.sub(g.matmul(x, w), y)
    g.set_outputs([g.mean(g.mul(d, d), [0, 1])])
    return pyrlx.grad(g, [w])


stage_w = {
    1: (np.arange(16) * 5 % 7 / 7 - 0.3).astype(np.float32),
    2: (np.arange(16) * 3 % 5 / 5 - 0.2).astype(np.float32),
}
w_true = np.array([1.5, -2, 0.5], np.float32)
x_all = ((np.arange(PER_RANK * WORLD * 3) * 37 % 23) / 11.5 - 1).astype(np.float32)
y_all = (x_all.reshape(-1, 3) @ w_true).astype(np.float32)


def spec(rank, uris):
    start = rank * PER_RANK
    return {
        "params": [{"name": "w", "uri": uris["w"]}],
        "data": [
            {"input": "X", "uri": uris["X"], "elem": 3, "shard_start": start, "shard_len": PER_RANK},
            {"input": "y", "uri": uris["y"], "elem": 1, "shard_start": start, "shard_len": PER_RANK},
        ],
        "batch": 4, "lr_per_epoch": [0.2] * 40, "device": "cpu",
    }


peers = pyrlx.loopback_peers(WORLD)
results = {}


def worker(rank):
    g = pyrlx.Group.connect(rank, WORLD, {"peers": peers, "device": "cpu"})
    g.all_reduce([rank + 1.0], "sum")
    g.all_gather([rank + 1.0])
    g.broadcast(0, [0.0])  # the root's length; it overwrites
    g.serve_worker({"steps": 2})
    results[rank] = np.asarray(g.serve_trainer()["params"]["w"], np.float32)


threads = [threading.Thread(target=worker, args=(r,)) for r in range(1, WORLD)]
for t in threads:
    t.start()

g = pyrlx.Group.connect(0, WORLD, {"peers": peers, "device": "cpu"})
print(f"mesh: rank {g.rank()} of {g.world()}, leader: {g.is_leader()}")
print("all_reduce sum:", g.all_reduce([1.0], "sum").tolist())
print("all_gather:", g.all_gather([1.0]).tolist())
print("broadcast:", g.broadcast(0, [42.0]).tolist())

caps = g.collect_caps()
print("workers:", ", ".join(f"rank {c['rank']} on {c['platform']}" for c in caps))
s = stage()
for rank in range(1, WORLD):
    g.ship_stage(rank, s, {"device": "cpu", "weights": {"w": blob(f"w{rank}.bin", stage_w[rank])}})
for x in ([1, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]):
    g.send_activation(1, x)
    y = g.recv_activation(WORLD - 1)
    want = np.maximum(np.maximum(np.array(x, np.float32) @ stage_w[1].reshape(4, 4), 0) @ stage_w[2].reshape(4, 4), 0)
    print(f"pipeline max|Δ| vs local = {np.abs(y - want).max():.1e}")

g.collect_caps()
bwd = training_graph()
uris = {"X": blob("X.bin", x_all), "y": blob("y.bin", y_all), "w": blob("w0.bin", [0, 0, 0])}
for rank in range(1, WORLD):
    g.ship_train(rank, bwd, spec(rank, uris))
mine = g.run_train(bwd, spec(0, uris))
w0 = np.asarray(mine["params"]["w"], np.float32)
m = mine["metrics"]
print(f"training loss: {m['first_loss']:.4f} -> {m['last_loss']:.2e}")
print("rank 0 w:", " ".join(f"{v:.3f}" for v in w0), "(true: 1.5 -2 0.5)")
for t in threads:
    t.join()
for rank in range(1, WORLD):
    print(f"rank {rank} w agrees: {bool(np.abs(results[rank] - w0).max() < 1e-5)}")
