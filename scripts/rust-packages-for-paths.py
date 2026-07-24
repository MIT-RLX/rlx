#!/usr/bin/env python3
# Map workspace-relative .rs paths → cargo package names (via nearest Cargo.toml).
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r'(?m)^name\s*=\s*"([^"]+)"')


def package_for(rel: str) -> str | None:
    path = (ROOT / rel).resolve()
    if not path.is_file():
        path = Path(rel).resolve()
    if not str(path).startswith(str(ROOT)):
        return None
    # Skip vendor trees.
    parts = path.relative_to(ROOT).parts
    if "vendor" in parts or "target" in parts:
        return None
    cur = path.parent
    while True:
        manifest = cur / "Cargo.toml"
        if manifest.is_file():
            text = manifest.read_text(encoding="utf-8", errors="replace")
            # Workspace roots have [workspace] without [package] sometimes.
            if "[package]" not in text:
                if cur == ROOT:
                    return None
                cur = cur.parent
                if cur == cur.parent:
                    return None
                continue
            m = NAME_RE.search(text)
            if m:
                return m.group(1)
            return None
        if cur == ROOT:
            return None
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent


def main() -> int:
    pkgs: list[str] = []
    seen: set[str] = set()
    for arg in sys.argv[1:]:
        name = package_for(arg)
        if name and name not in seen:
            seen.add(name)
            pkgs.append(name)
    for p in pkgs:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
