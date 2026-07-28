# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Pure-logic tests for the ``torch_import`` auto-decompose fallback — the parts
that don't need torch (op-name parsing + the export-breaking bisect). The full
end-to-end path is exercised separately with a real torch install."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "python" / "pyrlx" / "torch_import.py"
    spec = importlib.util.spec_from_file_location("torch_import_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ti = _load()


def test_base_name():
    assert ti._base_name("aten.roll.default") == "aten.roll"
    assert ti._base_name("aten.var.correction") == "aten.var"
    assert ti._base_name("aten.mm") == "aten.mm"
    assert ti._base_name("weird") == "weird"


def test_parse_unsupported_ops():
    err = (
        "Error: aten→rlx registry does not yet support 2 op(s):\n"
        "  aten.roll.default\n  aten.var.correction\n"
        "Extend crates/io/rlx-torch-import/src/lower.rs (add a handler + a SUPPORTED entry)."
    )
    assert ti._parse_unsupported_ops(err) == {"aten.roll.default", "aten.var.correction"}
    # Unrelated failures (or success) yield no ops → the loop stops, no false retries.
    assert ti._parse_unsupported_ops("some other error mentioning aten stuff") == set()
    assert ti._parse_unsupported_ops("") == set()


def test_accept_decompositions_all_clean():
    def export_fn(force):
        return {}  # always exports

    assert ti._accept_decompositions(export_fn, {"a"}, {"b", "c"}) == {"a", "b", "c"}


def test_accept_decompositions_drops_export_breaking_op():
    # `bad`'s decomposition breaks torch.export (e.g. emits prims); it must be
    # dropped while the clean `good` op is still accepted — no crash.
    def export_fn(force):
        if "bad" in force:
            raise RuntimeError("prims::broadcast_in_dim alias annotation")
        return {}

    assert ti._accept_decompositions(export_fn, set(), {"good", "bad"}) == {"good"}


def test_accept_decompositions_none_acceptable():
    def export_fn(force):
        if force:  # any candidate breaks
            raise RuntimeError("nope")
        return {}

    assert ti._accept_decompositions(export_fn, set(), {"x", "y"}) == set()
