#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail when a tracked doc file links to a path that does not exist.

Relative links rot silently. Nothing renders an error — GitHub shows a 404
only after a click, and `cargo doc` never sees a README — so a stale link
survives every gate the workspace has. The 2026 regrouping of the crates into
`crates/{core,backends,io,numerics,tooling,bindings}/` broke 59 of them at
once: every `../docs/foo.md` in a crate README was suddenly two levels short,
and no test noticed.

Checks, per tracked `*.md` (plus `llms.txt`, the agent-facing workspace map):

  * **Relative links resolve.** `[text](path)` must name a file or directory
    that exists, relative to the linking file. External schemes (`http`,
    `https`, `mailto`) and bare fragments (`#anchor`) are skipped.
  * **The target is tracked.** A path that exists here but is `.gitignore`d
    resolves locally and 404s for everyone else — the worst kind of broken
    link, because it is invisible on the machine that wrote it.
  * **Same-file anchors resolve.** `[text](#anchor)` must match a heading in
    that same file, using GitHub's slug rules. Cross-file anchors are not
    checked — only that the file exists.

Vendored and third-party trees are skipped: they are upstream copies, and
their links are upstream's problem.

Usage:
  python3 scripts/check-doc-links.py            # exit 1 on a broken link
  python3 scripts/check-doc-links.py --list     # print every link checked
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

SKIP_DIR_PARTS = ("vendor", "third_party", "node_modules")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
# Fenced code blocks hold example output and shell snippets, not real links.
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def slug(heading: str) -> str:
    """GitHub's anchor slug for a heading.

    Lowercase, strip inline-code and emphasis markers, drop everything that is
    not a word character / whitespace / hyphen, then replace **each** remaining
    whitespace character with a dash. That last detail matters: `A & B` loses
    the `&` and keeps *both* surrounding spaces, so the anchor is `a--b`, not
    `a-b`. Underscores survive — they are word characters.
    """
    text = re.sub(r"[`*]", "", heading)
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"\s", "-", text.strip())


_TRACKED: set[pathlib.Path] = set()


def _is_tracked(path: pathlib.Path) -> bool:
    """Whether `path` (or, for a directory, anything under it) is in the index."""
    if not _TRACKED:
        repo = pathlib.Path.cwd()
        _TRACKED.update(
            (repo / f).resolve()
            for f in subprocess.run(
                ["git", "ls-files"], capture_output=True, text=True, check=True
            ).stdout.split()
        )
    if path in _TRACKED:
        return True
    return any(str(t).startswith(str(path) + "/") for t in _TRACKED)


def tracked_markdown() -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md", "llms.txt"], capture_output=True, text=True, check=True
    ).stdout.split()
    return [
        pathlib.Path(f)
        for f in out
        if not any(part in SKIP_DIR_PARTS for part in pathlib.Path(f).parts)
    ]


def main() -> int:
    show_all = "--list" in sys.argv
    broken: list[str] = []
    checked = 0

    for path in tracked_markdown():
        text = path.read_text(errors="replace")
        anchors = {slug(h) for h in HEADING.findall(FENCE.sub("", text))}

        for match in LINK.finditer(FENCE.sub("", text)):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            file_part, _, fragment = target.partition("#")

            if not file_part:
                if fragment and fragment not in anchors:
                    broken.append(f"{path}: anchor #{fragment} has no matching heading")
                continue

            resolved = (path.parent / file_part).resolve()
            if not resolved.exists():
                broken.append(f"{path}: {target} does not exist")
            elif not _is_tracked(resolved):
                # Exists here but not in the repo: a `.gitignore`d or generated
                # path. It resolves on this machine and 404s for everyone else,
                # which is the worst kind of broken link — invisible locally.
                broken.append(f"{path}: {target} exists locally but is not tracked by git")
            elif show_all:
                print(f"  ok  {path} -> {target}")

    if broken:
        print(f"broken Markdown links ({len(broken)}):", file=sys.stderr)
        for line in broken:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nRelative links are resolved from the linking file's directory. "
            "Crate READMEs live three levels below the workspace root "
            "(crates/<group>/<crate>/), so the workspace docs are ../../../docs/.",
            file=sys.stderr,
        )
        return 1

    print(f"doc links OK — {checked} links across {len(tracked_markdown())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
