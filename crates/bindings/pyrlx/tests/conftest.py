# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Ensure tests import the maturin package under ``python/``, not the repo
directory namespace at ``pyrlx/`` (which has no ``Graph`` binding)."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_PKG_ROOT = Path(__file__).resolve().parents[1] / "python"
if _PYTHON_PKG_ROOT.is_dir():
    root = str(_PYTHON_PKG_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
