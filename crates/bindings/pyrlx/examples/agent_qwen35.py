# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""LoRA on Qwen3.5 0.8B with the agent optimizer: next-token training on
text, judged by the loss on a test set the agent never sees.

    rlx-export Qwen3.5-0.8B-BF16.gguf -o qwen-train.rlxp --variant train --seqs 64
    python agent_qwen35.py qwen-train.rlxp tokenizer.json train/ heldout/ --backend keep --plain
    python agent_qwen35.py ... --backend claude --decisions 6 --max-cost 5

``train`` and ``heldout`` are text files or folders of ``.md`` files. The
held-out text is split in two: the validation half, whose loss the agent sees
every ``--validate-every`` updates, and the test half, measured at the same
updates but never shown to it — the result. ``--plain`` first trains a plain
``Trainer`` on the same windows, measured at the same updates, and checks the
agent's run made the same losses (with ``--backend keep`` they must be equal).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import pyrlx as rlx

GOAL = ("LoRA (rank 8; full-attention q and v, delta-net qkv) on Qwen3.5 0.8B, next-token prediction on "
        "technical documentation; judged by the loss on a held-out test set after the last update")


def read_text(path):
    """A file's text, or every ``.md`` file under a folder, in name order."""
    p = Path(path)
    files = sorted(p.rglob("*.md")) if p.is_dir() else [p]
    return "\n\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)


def windows(tok, path, seq):
    """Non-overlapping ``seq + 1``-token windows of the text: inputs and next-token labels."""
    ids = tok.encode(read_text(path), add_special=False)
    n = len(ids) // (seq + 1)
    w = np.asarray(ids[: n * (seq + 1)], np.float32).reshape(n, seq + 1)
    return [{"input_ids": r[None, :seq].copy(), "labels": r[1:].copy()} for r in w]


def held_out(trainer, data):
    return float(np.mean([trainer.evaluate(b) for b in data]))


def trainer(a, opt):
    # On the device (resident) the gradients never reach the host to be clipped.
    kw = {} if a.resident else {"clip_norm": 1.0}
    if a.schedule:
        # Both runs follow it; the agent's programs scale from its rate.
        kw["schedule"] = {"kind": a.schedule, "total_steps": a.steps, "warmup_steps": a.steps // 20}
    return rlx.Trainer.from_package(a.package, device=a.device, optimizer=opt, resident=a.resident, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("tokenizer")
    ap.add_argument("train")
    ap.add_argument("heldout")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--heldout-windows", type=int, default=32, help="windows in each held-out half")
    ap.add_argument("--device", default="metal")
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--kind", default="adamw")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--schedule", default=None, help="the trainer's own schedule (cosine, linear, …), for both runs")
    ap.add_argument("--backend", default="keep")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--decisions", type=int, default=None)
    ap.add_argument("--max-cost", type=float, default=None)
    ap.add_argument("--validate-every", type=int, default=25)
    ap.add_argument("--run-root", default="rlx_runs")
    a = ap.parse_args()

    tok = rlx.load_tokenizer(a.tokenizer)
    train = windows(tok, a.train, a.seq)
    held = windows(tok, a.heldout, a.seq)
    n = min(a.heldout_windows, len(held) // 2)
    val, test = held[:n], held[n:2 * n]
    batches = [train[i % len(train)] for i in range(a.steps)]
    print(f"{len(train)} training windows, {len(val)} validation, {len(test)} test, seq {a.seq}", flush=True)
    if a.steps > len(train):
        print(f"note: {a.steps} updates repeat the training windows ({a.steps / len(train):.2f} epochs)", flush=True)
    opt = {"kind": a.kind, "lr": a.lr, "weight_decay": 0.0}
    at = [s for s in range(a.validate_every, a.steps + 1, a.validate_every)]
    out = {"updates": [0] + at}

    plain_losses = None
    if a.plain:
        t0 = time.time()
        plain = trainer(a, opt)
        curve = {"val": [held_out(plain, val)], "test": [held_out(plain, test)]}
        plain_losses = []
        for step, b in enumerate(batches, 1):
            plain_losses.append(plain.step(b))
            if step % a.validate_every == 0:
                curve["val"].append(held_out(plain, val))
                curve["test"].append(held_out(plain, test))
        out["plain"] = curve
        out["plain_seconds"] = round(time.time() - t0, 1)
        print(f"plain: test {curve['test'][0]:.4f} -> {curve['test'][-1]:.4f}  "
              f"(lowest {min(curve['test']):.4f})  {out['plain_seconds']}s", flush=True)
        del plain

    t0 = time.time()
    wrapped = trainer(a, opt)
    print(f"{len(wrapped.trainable())} trained params: {wrapped.trainable()[:4]} …", flush=True)
    options = dict(validation=a.validate_every, validation_size=len(val), every=10,
                   epochs=round(a.steps / len(train), 2), train_size=len(train), batch_size=1,
                   same_order_each_epoch=True, run_root=a.run_root)
    if a.decisions is not None:
        options["decisions"] = a.decisions
    if a.max_cost is not None:
        options["max_cost_usd"] = a.max_cost
    agent = rlx.AgentOptimizer(GOAL, a.steps, optimizer=opt, backend=a.backend, **options)
    print(f"run folder: {agent.run_dir()}", flush=True)
    losses = []
    curve = {"val": [held_out(wrapped, val)], "test": [held_out(wrapped, test)]}
    for step, b in enumerate(batches, 1):
        losses.append(agent.train_step(wrapped, b))
        if step % a.validate_every == 0:
            vl, tl = held_out(wrapped, val), held_out(wrapped, test)
            agent.report_validation(loss=vl)  # the test loss is never shown
            curve["val"].append(vl)
            curve["test"].append(tl)
            print(f"step {step:4d}  loss {losses[-1]:.4f}  val {vl:.4f}  test {tl:.4f}  "
                  f"v{agent.version():03d}  {time.time() - t0:.0f}s", flush=True)
    results = agent.close()
    out.update({
        "agent": curve,
        "decisions": results["decisions"], "applied": results["applied"],
        "reverted": results["reverted"], "cost_usd": round(results["cost_usd"], 4),
        "agent_seconds": round(time.time() - t0, 1), "run_dir": results["run_dir"],
    })
    if plain_losses is not None:
        out["identical_to_plain"] = plain_losses == losses
        print("\nupdate   plain val  plain test   agent val  agent test")
        for i, u in enumerate(out["updates"]):
            p, g = out["plain"], curve
            print(f"{u:6d}  {p['val'][i]:10.4f}  {p['test'][i]:10.4f}  {g['val'][i]:10.4f}  {g['test'][i]:10.4f}")
        print(f"\nresult (test loss after the last update): plain {out['plain']['test'][-1]:.4f}, "
              f"agent {curve['test'][-1]:.4f}")
    Path(results["run_dir"], "comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("plain", "agent", "updates")}, indent=1))


if __name__ == "__main__":
    main()
