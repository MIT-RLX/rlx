# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""`pyrlx` against rlx-bind's runtime API table (`api.idl`), both ways.

Every row this build's features cover resolves to a Python name, and every
public runtime name `pyrlx` exports is a row or a listed host helper. The C
ABI, Swift and JS are held to the same table, so a gap here is a binding that
is no longer one-to-one with the others.
"""

import inspect

import pyrlx
from pyrlx import _pyrlx

REQUIRED, KNOWN = _pyrlx._api_contract()


def _resolve(path):
    obj = pyrlx
    for part in path.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


def test_every_api_row_has_a_python_name():
    missing = [p for p in REQUIRED if _resolve(p) is None]
    assert not missing, (
        "api.idl rows with no Python name (add them to pyrlx, or mark the row "
        f"`py=-` with a why): {missing}"
    )


def test_every_native_export_is_a_row_or_a_host_helper():
    exported = []
    for name in dir(_pyrlx):
        if name.startswith("_"):
            continue
        obj = getattr(_pyrlx, name)
        # The public spelling: `pyrlx` may re-export it under another name.
        public = next((k for k in dir(pyrlx) if getattr(pyrlx, k, None) is obj), name)
        exported.append(public)
        # A host-only class (a typed options builder, a result record) brings
        # its members with it.
        # `Graph`'s builders are the op table's, checked by rlx-bind.
        host_class = not any(k.startswith(public + ".") for k in KNOWN)
        if inspect.isclass(obj) and not host_class and public != "Graph":
            for m in dir(obj):
                if not m.startswith("_"):
                    exported.append(f"{public}.{m}")
    extra = sorted({n for n in exported if n not in KNOWN})
    assert not extra, (
        "pyrlx exports names that are no api.idl row (add a row so every "
        f"binding gets it, or list them under `host py`): {extra}"
    )
