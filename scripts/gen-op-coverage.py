#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate / verify the op-claim matrix in docs/op-coverage.md.

Reads `OpKind` from `crates/core/rlx-ir/src/op.rs` and each backend's
`SUPPORTED_OPS` (or CoreML companions) from the backend crates, then updates
the **Coverage at a glance** table and the per-op checkmark columns inside
`docs/op-coverage.md`.

Usage:
  python3 scripts/gen-op-coverage.py          # rewrite docs/op-coverage.md
  python3 scripts/gen-op-coverage.py --check  # exit 1 if doc would change
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "op-coverage.md"
OP_RS = ROOT / "crates" / "core" / "rlx-ir" / "src" / "op.rs"

# Column order in the coverage matrix (matches the hand-written tables).
BACKENDS = [
    ("CPU", ROOT / "crates/backends/rlx-cpu/src/supported_ops.rs", "SUPPORTED_OPS"),
    ("MTL", ROOT / "crates/backends/rlx-metal/src/supported_ops.rs", "SUPPORTED_OPS"),
    ("MLX", ROOT / "crates/backends/rlx-mlx/src/supported_ops.rs", "SUPPORTED_OPS"),
    ("WGPU", ROOT / "crates/backends/rlx-wgpu/src/supported_ops.rs", "SUPPORTED_OPS"),
    ("ANE", ROOT / "crates/backends/rlx-coreml/src/supported_ops.rs", "SUPPORTED_OPS"),
    ("CUDA", ROOT / "crates/backends/rlx-cuda/src/supported_ops.rs", "SUPPORTED_OPS"),
    ("ROCm", ROOT / "crates/backends/rlx-rocm/src/supported_ops.rs", "SUPPORTED_OPS"),
    ("TPU", ROOT / "crates/backends/rlx-tpu/src/supported_ops.rs", "SUPPORTED_OPS"),
]

# Extra (not in the 8-column matrix, but claimed in-crate like Vulkan/OneAPI).
EXTRA = [
    ("Vulkan", ROOT / "crates/backends/rlx-vulkan/src/backend.rs", "SUPPORTED_OPS"),
    ("OneAPI", ROOT / "crates/backends/rlx-oneapi/src/backend.rs", "SUPPORTED_OPS"),
]


def parse_opkinds(path: Path) -> list[str]:
    text = path.read_text()
    m = re.search(r"pub enum OpKind\s*\{(.*?)\n\}", text, re.S)
    if not m:
        raise SystemExit(f"OpKind enum not found in {path}")
    body = m.group(1)
    kinds: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        # `Foo,` or `Foo`
        mm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*,?\s*(?://.*)?$", line)
        if mm:
            kinds.append(mm.group(1))
    return kinds


def extract_const_idents(path: Path, const_name: str) -> set[str]:
    text = path.read_text()
    # Find `const NAME` then collect OpKind identifiers until the matching
    # `];` / `};` that closes the slice.
    m = re.search(rf"\bconst {re.escape(const_name)}\b[^=]*=", text)
    if not m:
        raise SystemExit(f"{const_name} not found in {path}")
    rest = text[m.end() :]
    # Prefer bracket form `&=[ ... ]` else brace form `{ ... &[ ... ] }`
    idents: set[str] = set()
    # Strip comments
    rest_nc = re.sub(r"//.*?$", "", rest, flags=re.M)
    # Take until we have seen a top-level `];` after an `[`
    depth = 0
    started = False
    buf = []
    for ch in rest_nc:
        if ch == "[":
            depth += 1
            started = True
            continue
        if ch == "]":
            depth -= 1
            if started and depth == 0:
                break
            continue
        if started and depth > 0:
            buf.append(ch)
    blob = "".join(buf)
    for tok in re.findall(r"(?:OpKind::)?([A-Za-z_][A-Za-z0-9_]*)", blob):
        if tok in {"OpKind", "use", "rlx_ir"}:
            continue
        idents.add(tok)
    return idents


def mark(claimed: bool) -> str:
    return "✅" if claimed else ""


def update_glance(doc: str, counts: dict[str, int], total: int) -> str:
    """Replace the glance table body counts."""
    lines = doc.splitlines(keepends=True)
    out: list[str] = []
    in_glance = False
    for line in lines:
        if line.startswith("### Coverage at a glance"):
            in_glance = True
            out.append(line)
            continue
        if not in_glance:
            out.append(line)
            continue
        # Stay in the glance section through blank lines; leave on the next
        # heading / horizontal rule / non-table prose.
        if line.strip() == "":
            out.append(line)
            continue
        if line.startswith("---") or line.startswith("#") or line.startswith(">"):
            in_glance = False
            out.append(line)
            continue
        if "| Backend | Ops claimed |" in line:
            line = re.sub(r"of \d+", f"of {total}", line)
            out.append(line)
            continue
        if line.startswith("|-------"):
            out.append(line)
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] in counts:
                name = cells[0]
                n = counts[name]
                desc = cells[2] if len(cells) > 2 else ""
                out.append(f"| {name}  | **{n}** | {desc} |\n")
                continue
            # Non-backend table row (shouldn't happen) — keep as-is
            out.append(line)
            continue
        # Prose inside the glance section (e.g. the Total note)
        if "(Total **" in line:
            line = re.sub(
                r"\(Total \*\*\d+\*\* `OpKind`s\.",
                f"(Total **{total}** `OpKind`s.",
                line,
                count=1,
            )
        out.append(line)
    return "".join(out)


def update_matrix_checkmarks(doc: str, claims: dict[str, set[str]]) -> str:
    """Rewrite ✅ columns in category tables; preserve Description."""
    col_names = [b[0] for b in BACKENDS]
    lines = doc.splitlines(keepends=True)
    out = []
    header_cols: list[str] | None = None
    for line in lines:
        if line.startswith("| Op | Description |"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            header_cols = parts
            out.append(line)
            continue
        if header_cols and line.startswith("|----"):
            out.append(line)
            continue
        if header_cols and line.startswith("|") and "`" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 2:
                out.append(line)
                continue
            op_cell = parts[0]
            m = re.match(r"`([A-Za-z_][A-Za-z0-9_]*)`", op_cell)
            if not m:
                out.append(line)
                continue
            op = m.group(1)
            desc = parts[1] if len(parts) > 1 else ""
            cells = [f"`{op}`", desc]
            for i, name in enumerate(col_names):
                claimed = op in claims.get(name, set())
                old = parts[2 + i] if len(parts) > 2 + i else ""
                # Preserve hand annotations (ᵁ/ᴰ/ᴴ) on claimed cells.
                suffix = ""
                for mark_ch in ("ᵁ", "ᴰ", "ᴴ"):
                    if mark_ch in old:
                        suffix += mark_ch
                cells.append((mark(claimed) + suffix) if claimed else "")
            out.append("| " + " | ".join(cells) + " |\n")
            continue
        if header_cols and (not line.startswith("|") or line.strip() == ""):
            header_cols = None
        out.append(line)
    return "".join(out)


def update_source_of_truth(doc: str) -> str:
    """Point the doc at in-crate SUPPORTED_OPS."""
    old = (
        "| Per-backend **legalization contract** | "
        "[`crates/core/rlx-runtime/src/backend.rs`]"
        "(../crates/core/rlx-runtime/src/backend.rs) — the `*_SUPPORTED_OPS` "
        "consts returned by each `Backend::supported_ops()` |"
    )
    new = (
        "| Per-backend **legalization contract** | each backend crate's "
        "`SUPPORTED_OPS` (`crates/backends/rlx-*/src/supported_ops.rs`, "
        "Vulkan/OneAPI in their `backend.rs`) — returned by "
        "`Backend::supported_ops()` |"
    )
    if old in doc:
        doc = doc.replace(old, new)
    else:
        # Fuzzy: replace any line mentioning *_SUPPORTED_OPS in backend.rs
        doc = re.sub(
            r"\| Per-backend \*\*legalization contract\*\* \|.*?\|",
            new,
            doc,
            count=1,
            flags=re.S,
        )
    # Refresh instruction about regenerating
    doc = re.sub(
        r"> \*\*This file is generated/verified from the source consts\.\*\*.*?\n",
        "> **Claim columns are generated/verified from backend `SUPPORTED_OPS`.** "
        "Run `python3 scripts/gen-op-coverage.py` after changing a claim "
        "(or `python3 scripts/gen-op-coverage.py --check` in CI).\n",
        doc,
        count=1,
        flags=re.S,
    )
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if doc would change")
    args = ap.parse_args()

    kinds = parse_opkinds(OP_RS)
    kind_set = set(kinds)
    claims: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for name, path, const in BACKENDS:
        ids = extract_const_idents(path, const)
        unknown = ids - kind_set
        if unknown:
            print(f"warning: {name} claims unknown OpKinds: {sorted(unknown)}", file=sys.stderr)
        claims[name] = ids & kind_set
        counts[name] = len(claims[name])

    for name, path, const in EXTRA:
        ids = extract_const_idents(path, const)
        print(f"# {name}: {len(ids & kind_set)} ops claimed", file=sys.stderr)

    total = len(kinds)
    print(f"# OpKind total: {total}", file=sys.stderr)
    for name, n in counts.items():
        print(f"# {name}: {n}", file=sys.stderr)

    doc = DOC.read_text()
    doc2 = update_source_of_truth(doc)
    doc2 = update_glance(doc2, counts, total)
    doc2 = update_matrix_checkmarks(doc2, claims)

    if doc2 == doc:
        print("docs/op-coverage.md already up to date")
        return 0
    if args.check:
        print("docs/op-coverage.md is out of date; run scripts/gen-op-coverage.py", file=sys.stderr)
        return 1
    DOC.write_text(doc2)
    print(f"updated {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
