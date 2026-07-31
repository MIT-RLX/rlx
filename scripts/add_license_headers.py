#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Add or complete RLX dual-license (MIT OR Apache-2.0) file headers in workspace source trees."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = frozenset(
    {"vendor", "target", ".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
)
EXTS = {".rs", ".py", ".sh", ".cpp", ".h", ".c", ".cu", ".cuh", ".wgsl"}
DOC_EXTS = {".md"}
TEXT_DOCS = frozenset({"llms.txt"})
FULL_MARK = "SPDX-License-Identifier: MIT OR Apache-2.0"
COPYRIGHT_MARK = "Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna."
MD_LICENSE_MARKERS = ("## License", "MIT OR Apache-2.0", FULL_MARK)
MD_LICENSE_FOOTER = "\n## License\n\nMIT OR Apache-2.0.\n"
ROOT_DOC_NAMES = frozenset({"AGENTS.md", "CHANGELOG.md"})

LICENSE_BODY = """\
// SPDX-License-Identifier: MIT OR Apache-2.0
"""

SHORT_HEADER_RE = re.compile(
    r"^// RLX [—-] versatile ML compiler \+ runtime\.\n"
    r"// Copyright \(C\) 2026 Eugene Hauptmann, Nataliya Kosmyna\.\n",
    re.MULTILINE,
)

SHORT_HEADER_HASH_RE = re.compile(
    r"^# RLX — versatile ML compiler \+ runtime\.\n"
    r"# Copyright \(C\) 2026 Eugene Hauptmann, Nataliya Kosmyna\.\n",
    re.MULTILINE,
)


def workspace_members() -> list[str]:
    text = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    members: list[str] = []
    in_members = False
    for line in text.splitlines():
        if line.strip() == "members = [":
            in_members = True
            continue
        if in_members:
            if line.strip() == "]":
                break
            m = re.match(r'\s*"([^"]+)"', line)
            if m and not line.strip().startswith("#"):
                members.append(m.group(1))
    return members


def iter_targets() -> list[Path]:
    paths: list[Path] = []
    for member in workspace_members():
        base = ROOT / member
        if not base.exists():
            continue
        if base.is_file():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or any(p in SKIP_PARTS for p in path.parts):
                continue
            if path.suffix in EXTS or path.suffix in DOC_EXTS or path.name == "Justfile":
                paths.append(path)

    for extra in (
        ROOT / "scripts",
        ROOT / "docs",
        ROOT / "rig.sh",
        ROOT / "Justfile",
        ROOT / "llms.txt",
        ROOT / "AGENTS.md",
        ROOT / "CHANGELOG.md",
    ):
        if not extra.exists():
            continue
        if extra.is_file():
            if (
                extra.suffix in EXTS
                or extra.suffix in DOC_EXTS
                or extra.name == "Justfile"
                or extra.name in TEXT_DOCS
                or extra.name in ROOT_DOC_NAMES
            ):
                paths.append(extra)
            continue
        for path in extra.rglob("*"):
            if not path.is_file() or any(p in SKIP_PARTS for p in path.parts):
                continue
            if path.suffix in EXTS or path.suffix in DOC_EXTS:
                paths.append(path)

    return sorted(set(paths))


def comment_prefix(path: Path) -> str:
    if (
        path.suffix in {".py", ".sh"}
        or path.name == "Justfile"
        or path.name in TEXT_DOCS
    ):
        return "#"
    return "//"


def license_body_lines(path: Path) -> str:
    p = comment_prefix(path)
    return "\n".join(
        f"{p}{line[2:]}" if line.startswith("//") else line
        for line in LICENSE_BODY.splitlines()
    )


def full_header(path: Path) -> str:
    p = comment_prefix(path)
    body = license_body_lines(path)
    return (
        f"{p} RLX — versatile ML compiler + runtime.\n"
        f"{p} Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n"
        f"{body}\n"
    )


def expand_short(text: str, path: Path) -> str | None:
    p = comment_prefix(path)
    body = license_body_lines(path)
    if p == "//":
        if SHORT_HEADER_RE.match(text) and FULL_MARK not in text[:2000]:
            return SHORT_HEADER_RE.sub(
                "// RLX — versatile ML compiler + runtime.\n"
                "// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n"
                + body
                + "\n",
                text,
                count=1,
            )
    else:
        if SHORT_HEADER_HASH_RE.match(text) and FULL_MARK not in text[:2000]:
            return SHORT_HEADER_HASH_RE.sub(
                "# RLX — versatile ML compiler + runtime.\n"
                "# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.\n"
                + body
                + "\n",
                text,
                count=1,
            )
    return None


def prepend_header(text: str, path: Path) -> str:
    header = full_header(path)
    if path.suffix == ".py" and text.startswith("#!"):
        first, _, rest = text.partition("\n")
        if rest.startswith("\n"):
            return f"{first}\n{header}{rest.lstrip(chr(10))}"
        return f"{first}\n{header}{rest}"
    if path.suffix == ".sh" and text.startswith("#!"):
        first, _, rest = text.partition("\n")
        return f"{first}\n{header}{rest}"
    return header + text


def has_md_license(text: str) -> bool:
    return any(marker in text for marker in MD_LICENSE_MARKERS)


def process_markdown(path: Path, dry_run: bool) -> str | None:
    text = path.read_text(encoding="utf-8")
    if has_md_license(text):
        return None
    updated = text.rstrip() + MD_LICENSE_FOOTER
    if not dry_run:
        path.write_text(updated, encoding="utf-8", newline="\n")
    return "license-footer"


def process_text_doc(path: Path, dry_run: bool) -> str | None:
    text = path.read_text(encoding="utf-8")
    if FULL_MARK in text[:4000] or COPYRIGHT_MARK in text[:500]:
        return None
    updated = prepend_header(text, path)
    if not dry_run:
        path.write_text(updated, encoding="utf-8", newline="\n")
    return "prepend"


def process(path: Path, dry_run: bool) -> str | None:
    if path.suffix in DOC_EXTS or path.name in ROOT_DOC_NAMES:
        return process_markdown(path, dry_run)
    if path.name in TEXT_DOCS:
        return process_text_doc(path, dry_run)

    text = path.read_text(encoding="utf-8")
    if FULL_MARK in text[:4000]:
        return None

    if COPYRIGHT_MARK in text[:500]:
        updated = expand_short(text, path)
        action = "expand"
    else:
        updated = prepend_header(text, path)
        action = "prepend"

    if updated is None or updated == text:
        return None

    if not dry_run:
        path.write_text(updated, encoding="utf-8", newline="\n")
    return action


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    changed: dict[str, int] = {"expand": 0, "prepend": 0, "license-footer": 0}
    for path in iter_targets():
        rel = path.relative_to(ROOT)
        action = process(path, dry_run)
        if action:
            changed[action] += 1
            print(f"{action}: {rel}")

    mode = "would update" if dry_run else "updated"
    total = sum(changed.values())
    print(
        f"\n{mode} {total} files "
        f"(expand={changed['expand']}, prepend={changed['prepend']}, "
        f"license-footer={changed['license-footer']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
