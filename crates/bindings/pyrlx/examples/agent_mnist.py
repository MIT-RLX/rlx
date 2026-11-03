# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""MNIST with the agent optimizer: an MLP trained by ``pyrlx.Trainer``, its
optimizer rewritten by a coding agent while it trains.

    python agent_mnist.py --backend keep                 # no agent: must match the plain run
    python agent_mnist.py --backend claude --decisions 4 --max-cost 3

Data: the four MNIST IDX files in ``--data`` (default ``~/.mnist``). The last
5,000 training images are a validation set the agent sees; the test set (the
goal's judge) it never sees. Both the plain run and the agent's run train on
the same batches; the script prints their test accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pyrlx as rlx

BATCH, HIDDEN = 128, 128
GOAL = "a 784-128-10 MLP on MNIST; judged by test-set accuracy after the last update"


def idx(path):
    raw = Path(path).read_bytes()
    dims = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * dims).reshape(shape)


def load(data):
    d = Path(data).expanduser()
    x = idx(d / "train-images-idx3-ubyte").reshape(-1, 784).astype(np.float32) / 255.0
    y = idx(d / "train-labels-idx1-ubyte").astype(np.int64)
    xt = idx(d / "t10k-images-idx3-ubyte").reshape(-1, 784).astype(np.float32) / 255.0
    yt = idx(d / "t10k-labels-idx1-ubyte").astype(np.int64)
    return (x[:-5000], y[:-5000]), (x[-5000:], y[-5000:]), (xt, yt)


def graph():
    g = rlx.Graph("mlp")
    x = g.input("x", [BATCH, 784], "f32")
    t = g.input("t", [BATCH, 10], "f32")
    w1 = g.param("w1", [784, HIDDEN], "f32")
    b1 = g.param("b1", [1, HIDDEN], "f32")
    w2 = g.param("w2", [HIDDEN, 10], "f32")
    b2 = g.param("b2", [1, 10], "f32")
    h = g.relu(g.add(g.matmul(x, w1), b1))
    logits = g.add(g.matmul(h, w2), b2)
    g.set_outputs([g.mean(g.softmax_cross_entropy(logits, t), [0], False)])
    return g


def init(seed=0):
    r = np.random.default_rng(seed)
    he = lambda fan_in, shape: (r.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
    return {"w1": he(784, (784, HIDDEN)), "b1": np.zeros((1, HIDDEN), np.float32),
            "w2": he(HIDDEN, (HIDDEN, 10)), "b2": np.zeros((1, 10), np.float32)}


def evaluate(params, x, y):
    """Loss and accuracy of the MLP on (x, y), in numpy."""
    p = {k: v.reshape(s) for (k, v), s in zip(params.items(), [(784, HIDDEN), (1, HIDDEN), (HIDDEN, 10), (1, 10)])}
    logits = np.maximum(x @ p["w1"] + p["b1"], 0) @ p["w2"] + p["b2"]
    logits -= logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean()), float((logits.argmax(1) == y).mean())


def batches(x, y, steps, seed=1):
    r = np.random.default_rng(seed)
    eye = np.eye(10, dtype=np.float32)
    order = np.array([], np.int64)
    for _ in range(steps):
        if len(order) < BATCH:
            order = np.concatenate([order, r.permutation(len(x))])
        i, order = order[:BATCH], order[BATCH:]
        yield {"x": np.ascontiguousarray(x[i]), "t": eye[y[i]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/.mnist")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--backend", default="keep")
    ap.add_argument("--program", default="rules", choices=["rules", "python"])
    ap.add_argument("--decisions", type=int, default=None)
    ap.add_argument("--max-cost", type=float, default=None)
    ap.add_argument("--validate-every", type=int, default=250)
    ap.add_argument("--run-root", default="rlx_runs")
    a = ap.parse_args()

    (x, y), (xv, yv), (xt, yt) = load(a.data)
    opt = {"kind": "adamw", "lr": 1e-3, "weight_decay": 1e-4}
    names = ["w1", "b1", "w2", "b2"]
    plain = rlx.Trainer(graph(), wrt=names, init=init(), optimizer=opt)
    wrapped = rlx.Trainer(graph(), wrt=names, init=init(), optimizer=opt)
    options = dict(validation=a.validate_every, validation_size=len(xv), every=50,
                   epochs=round(a.steps * BATCH / len(x), 2), train_size=len(x), batch_size=BATCH,
                   same_order_each_epoch=False, run_root=a.run_root)
    if a.decisions is not None:
        options["decisions"] = a.decisions
    if a.max_cost is not None:
        options["max_cost_usd"] = a.max_cost
    agent = rlx.AgentOptimizer(GOAL, a.steps, optimizer=opt, program=a.program,
                               backend=a.backend, **options)
    print(f"run folder: {agent.run_dir()}")
    t0 = time.time()
    same = True
    for step, batch in enumerate(batches(x, y, a.steps), 1):
        lp = plain.step(batch)
        la = agent.train_step(wrapped, batch)
        same = same and lp == la
        if step % a.validate_every == 0:
            vl, va = evaluate(wrapped.params(), xv, yv)
            agent.report_validation(loss=vl, accuracy=va)
            print(f"step {step:5d}  loss plain {lp:.4f} agent {la:.4f}  "
                  f"agent val loss {vl:.4f} acc {va:.4f}  v{agent.version():03d}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    results = agent.close()
    tp = evaluate(plain.params(), xt, yt)
    ta = evaluate(wrapped.params(), xt, yt)
    print(json.dumps({
        "test_accuracy_plain": round(tp[1], 4), "test_accuracy_agent": round(ta[1], 4),
        "test_loss_plain": round(tp[0], 4), "test_loss_agent": round(ta[0], 4),
        "identical_to_plain": same, "decisions": results["decisions"], "applied": results["applied"],
        "reverted": results["reverted"], "cost_usd": round(results["cost_usd"], 4),
        "seconds": round(time.time() - t0, 1), "run_dir": results["run_dir"],
    }, indent=1))


if __name__ == "__main__":
    main()
