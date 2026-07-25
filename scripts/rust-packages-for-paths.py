#!/usr/bin/env python3
# Map workspace-relative .rs paths -> "<cargo package>\t<workspace-rel-dir>".
# The workspace column is '.' for the repo's root workspace, or the directory
# (relative to the repo root) of the *nested* workspace that owns the package
# (e.g. `android` for rlx-jni). The pre-commit hook uses it to keep `cargo -p`
# scoped to the right workspace: a nested-workspace crate cannot be selected
# from the root workspace (`cargo -p rlx-jni` -> "did not match any packages").
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r'(?m)^name\s*=\s*"([^"]+)"')
WORKSPACE_RE = re.compile(r'(?m)^\s*\[workspace\]')


def _read(manifest: Path) -> str:
    return manifest.read_text(encoding="utf-8", errors="replace")


def workspace_dir_for(pkg_dir: Path) -> Path:
    """Nearest ancestor (inclusive) whose Cargo.toml declares [workspace]."""
    cur = pkg_dir
    while True:
        manifest = cur / "Cargo.toml"
        if manifest.is_file() and WORKSPACE_RE.search(_read(manifest)):
            return cur
        if cur == ROOT:
            return ROOT
        parent = cur.parent
        if parent == cur:
            return ROOT
        cur = parent


def package_for(rel: str) -> tuple[str, str] | None:
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
            text = _read(manifest)
            # Workspace roots have [workspace] without [package] sometimes.
            if "[package]" not in text:
                if cur == ROOT:
                    return None
                cur = cur.parent
                if cur == cur.parent:
                    return None
                continue
            m = NAME_RE.search(text)
            if not m:
                return None
            ws = workspace_dir_for(cur)
            return m.group(1), ws.relative_to(ROOT).as_posix()
        if cur == ROOT:
            return None
        parent = cur.parent
        if parent == cur:
            return None
        cur = parent


def main() -> int:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for arg in sys.argv[1:]:
        res = package_for(arg)
        if res and res[0] not in seen:
            seen.add(res[0])
            out.append(res)
    for name, ws in out:
        print(f"{name}\t{ws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
