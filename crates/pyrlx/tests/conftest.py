# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
