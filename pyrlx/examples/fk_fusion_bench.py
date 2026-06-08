#!/usr/bin/env python3
# RLX - versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# Licensed under the GNU General Public License, version 3.
"""Minimal FKL fusion timing demo (pyrlx)."""

from __future__ import annotations

import argparse
import time

import numpy as np

import pyrlx as rlx


def _time_run(compiled, inputs: dict, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        compiled.run(inputs)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        compiled.run(inputs)
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return samples[len(samples) // 2] / 1e3  # us


def main() -> None:
    p = argparse.ArgumentParser(description="FKL fusion bench (pyrlx)")
    p.add_argument("--device", default="metal", help="cpu, metal, ...")
    p.add_argument(
        "--batch",
        action="store_true",
        help="also time batch_narrow_relu_graph (default session vs native_fk)",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()
    if not rlx.is_available(args.device):
        raise SystemExit(f"device {args.device!r} not available in this wheel")

    # Resize prologue chain
    g = rlx.Graph("fk_resize")
    x = g.input("x", [1, 3, 56, 56], "f32")
    a = g.input("a", [1, 3, 112, 112], "f32")
    up = g.resize_nearest_2x(x)
    r = g.relu(up)
    g.set_outputs([g.add(r, a)])

    x_np = np.zeros((1, 3, 56, 56), dtype=np.float32)
    a_np = np.zeros((1, 3, 112, 112), dtype=np.float32)
    inp = {"x": x_np, "a": a_np}

    sess = rlx.Session(device=args.device)
    kd = "native" if args.device != "cpu" else None

    g_sep = rlx.Graph("sep")
    x2 = g_sep.input("x", [1, 3, 56, 56], "f32")
    a2 = g_sep.input("a", [1, 3, 112, 112], "f32")
    up2 = g_sep.resize_nearest_2x(x2)
    r2 = g_sep.relu(up2)
    g_sep.set_outputs([g_sep.add(r2, a2)])
    t_sep = _time_run(sess.compile(g_sep), inp, args.warmup, args.runs)

    g_fk = rlx.Graph("fk")
    x3 = g_fk.input("x", [1, 3, 56, 56], "f32")
    a3 = g_fk.input("a", [1, 3, 112, 112], "f32")
    up3 = g_fk.resize_nearest_2x(x3)
    r3 = g_fk.relu(up3)
    g_fk.set_outputs([g_fk.add(r3, a3)])
    t_fk = _time_run(
        sess.compile_with(g_fk, fusion_options=rlx.FusionOptions(), kernel_dispatch=kd),
        inp,
        args.warmup,
        args.runs,
    )

    print(f"device={args.device} resize+relu+add (median us)")
    print(f"  session default compile: {t_sep:.2f}")
    print(f"  compile_with (FKL on):   {t_fk:.2f}")
    if t_fk > 0:
        print(f"  ratio (default/fk):      {t_sep / t_fk:.3f}x")

    if args.batch:
        batch_n, c, h, w = 2, 3, 32, 32
        g_batch = rlx.batch_narrow_relu_graph("batch", batch_n, c, h, w)
        batch_np = np.linspace(-0.2, 0.2, batch_n * c * h * w, dtype=np.float32).reshape(
            batch_n, c, h, w
        )
        binp = {"batch": batch_np}
        t_def = _time_run(sess.compile(g_batch), binp, args.warmup, args.runs)
        t_nat = _time_run(
            sess.compile_with(g_batch, fusion_options=rlx.FusionOptions.native_fk(), kernel_dispatch=kd),
            binp,
            args.warmup,
            args.runs,
        )
        print(f"device={args.device} batch narrow+relu+concat (median us)")
        print(f"  session default (native FKL auto): {t_def:.2f}")
        print(f"  explicit native_fk():          {t_nat:.2f}")
        if t_nat > 0:
            print(f"  ratio (default/native):        {t_def / t_nat:.3f}x")


if __name__ == "__main__":
    main()
