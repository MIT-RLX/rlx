# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
"""Shared helper for the per-model examples.

`export_all_forms(name, model, example)` exports one PyTorch model to RLX in
**every output form** and prints the results so they can be inspected:

  1. bundle          — a runnable RLX file: the serialized HIR graph
                       (`out_<name>/bundle/model.hir.json`) + weights, plus a
                       CPU parity check against PyTorch (cosine + max abs err).
  2. crate / graph   — standalone RLX crate, raw HIR builder (covers every op).
  3. crate / tensor  — standalone RLX crate, PyTorch-like `Tensor` DSL.
  4. crate / flow    — standalone RLX crate, `ModelFlow` blocks.

Everything is left on disk under `out_<name>/` for inspection.
"""

from __future__ import annotations

import collections
import json
import pathlib

import pyrlx

HERE = pathlib.Path(__file__).parent


def _op_kind(node: dict) -> str:
    op = node.get("op")
    return next(iter(op)) if isinstance(op, dict) else str(op)


def _print_parity(summary: dict) -> None:
    parity = (summary.get("rlx_result") or {}).get("parity")
    if parity and parity.get("outputs"):
        o = parity["outputs"][0]
        tag = "PASS ✓" if parity.get("passed") else "FAIL ✗"
        print(f"  1. bundle  parity {tag}: cosine={o['cosine']:.6f}  "
              f"max|err|={o['max_abs_err']:.2e}  ({o['numel']} elems)")
    else:
        print(f"  1. bundle  front-end: {summary.get('num_nodes')} aten nodes "
              f"(rlx_ran={summary.get('rlx_ran')})")


def _print_hir(bundle_dir: pathlib.Path) -> None:
    p = bundle_dir / "model.hir.json"
    if not p.exists():
        return
    nodes = json.loads(p.read_text()).get("nodes", [])
    kinds = collections.Counter(_op_kind(n) for n in nodes)
    top = ", ".join(f"{k}×{v}" for k, v in kinds.most_common(8))
    print(f"             HIR graph: {len(nodes)} nodes  [{top}]")
    print(f"             {p}")


def _find_crate(root: pathlib.Path) -> pathlib.Path | None:
    return next((d for d in sorted(root.glob("rlx-*")) if d.is_dir()), None)


def _print_crate(root: pathlib.Path, style: str, n: int) -> None:
    crate = _find_crate(root)
    if crate is None:
        print(f"  {n}. crate/{style:<6} (no crate dir)")
        return
    src = crate / "src"
    files = sorted(f.name for f in src.glob("*.rs")) if src.exists() else []
    print(f"  {n}. crate/{style:<6} {crate}  src/{{{', '.join(files)}}}")


def export_all_forms(name: str, model, example) -> pathlib.Path:
    """Export `model` to RLX in every form under `out_<name>/` and report."""
    out = HERE / f"out_{name}"
    print(f"\n{'=' * 72}\n {name.upper()}\n{'=' * 72}")

    # 1) HIR-graph bundle (+ graph crate), verified against PyTorch on CPU.
    summary = pyrlx.from_torch(
        model, example, out_dir=str(out), model_name=name,
        emit=("bundle", "crate"), emit_style="graph", verify=True,
    )
    _print_parity(summary)
    _print_hir(out / "bundle")
    _print_crate(out, "graph", 2)

    # 2) alternative crate authoring styles (some ops aren't expressible in the
    #    Tensor DSL — that's reported, not fatal).
    for i, style in enumerate(("tensor", "flow"), start=3):
        d = out / f"style_{style}"
        try:
            pyrlx.from_torch(
                model, example, out_dir=str(d), model_name=name,
                emit=("crate",), emit_style=style, verify=False,
            )
            _print_crate(d, style, i)
        except Exception as e:  # noqa: BLE001 — surface which op is unsupported
            msg = str(e).strip().splitlines()[-1][:80]
            print(f"  {i}. crate/{style:<6} n/a for this model — {msg}")

    print(f"\n  → inspect all forms under {out}/")
    return out
