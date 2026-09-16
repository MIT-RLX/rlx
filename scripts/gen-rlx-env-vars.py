#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate docs/rlx-env-vars.md from the env registry + leftover tree walk.

Primary source: crates/core/rlx-ir/src/env_registry_data.inc.rs (and Public
metadata). Optionally append an "Unregistered mentions" section for RLX_*
strings still found in the tree but not in the registry (migration leftovers).

Also verifies that every `env::flag` / `env::var` / … string literal in
`crates/**/*.rs` is registered (canonical or alias).

Usage:
  python3 scripts/gen-rlx-env-vars.py          # rewrite docs
  python3 scripts/gen-rlx-env-vars.py --check  # exit 1 if doc or registry drift
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "rlx-env-vars.md"
REGISTRY_INC = ROOT / "crates/core/rlx-ir/src/env_registry_data.inc.rs"
SKIP_DIRS = {".git", "target", "vendor", "node_modules", ".cursor"}
SKIP_PATH_PARTS = ("vendor/mlx", "rlx-mlx-sys/vendor")
# Left boundary matters: without it `KITTEN_RLX_DEBUG_TRIP` matches as
# `RLX_DEBUG_TRIP`, and the registry gains an entry for a variable no code
# ever reads. Four such phantoms existed before this was tightened.
NAME_RE = re.compile(r"(?<![A-Z0-9_])RLX_[A-Z][A-Z0-9_]*")
ENTRY_RE = re.compile(
    r'name:\s*"(RLX_[A-Z0-9_]+)"\s*,\s*group:\s*"([^"]*)"\s*,\s*summary:\s*"([^"]*)"\s*,'
    r"\s*kind:\s*(EnvKind::\w+(?:\s*\{[^}]*\})?|EnvKind::Enum\([^)]*\))\s*,"
    r"\s*stability:\s*(EnvStability::\w+(?:\s*\{[^}]*\})?)\s*,"
    r'\s*aliases:\s*&\[((?:[^\]]*))\]',
    re.S,
)
ENV_CALL_RE = re.compile(
    r'(?:rlx_ir::env::|crate::env::|env::|std::env::|'
    r'rlx_ir::env_registry::|env_registry::)'
    r'(?:flag|flag_or|var|parse_or|var_os)\s*\(\s*"(?<![A-Z0-9_])(RLX_[A-Z][A-Z0-9_]*)"'
)
ENV_LINE_MARKERS = (
    "env::flag",
    "env::flag_or",
    "env::var",
    "env::parse_or",
    "env::var_os",
    "std::env::var",
    "env_registry::flag",
    "env_registry::var",
    "env_registry::flag_or",
    "env_registry::parse_or",
)


def parse_registry() -> tuple[dict[str, dict], set[str]]:
    text = REGISTRY_INC.read_text(encoding="utf-8")
    entries: dict[str, dict] = {}
    aliases: set[str] = set()
    # Simpler line-oriented parse
    blocks = re.split(r"EnvVarEntry\s*\{", text)
    for block in blocks[1:]:
        def field(name: str) -> str | None:
            m = re.search(rf'{name}:\s*"([^"]*)"', block)
            return m.group(1) if m else None

        name = field("name")
        if not name:
            continue
        group = field("group") or "misc"
        summary = field("summary") or ""
        stab_m = re.search(r"stability:\s*EnvStability::(\w+)", block)
        stab = stab_m.group(1) if stab_m else "Bisect"
        replace = None
        if stab == "Deprecated":
            rm = re.search(r'replace_with:\s*"([^"]+)"', block)
            replace = rm.group(1) if rm else None
        am = re.search(r"aliases:\s*&\[(.*?)\]", block, re.S)
        als: list[str] = []
        if am:
            als = re.findall(r'"(RLX_[A-Z0-9_]+)"', am.group(1))
            aliases.update(als)
        kind_m = re.search(r"kind:\s*(EnvKind::\w+)", block)
        kind = kind_m.group(1).replace("EnvKind::", "") if kind_m else "Bool"
        layer_m = re.search(r'layer:\s*EnvLayer::(\w+)(?:\("([^"]*)"\))?', block)
        if layer_m:
            layer = layer_m.group(1)
            if layer_m.group(2):
                layer = f"{layer}({layer_m.group(2)})"
        else:
            layer = "Tooling"
        entries[name] = {
            "group": group,
            "summary": summary,
            "stability": stab,
            "replace_with": replace,
            "aliases": als,
            "kind": kind,
            "layer": layer,
        }
        aliases.add(name)
        aliases.update(als)
    return entries, aliases


def collect_env_reads() -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    crates = ROOT / "crates"
    for dirpath, dirnames, filenames in os.walk(crates):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".rs"):
                continue
            path = Path(dirpath) / fn
            rel = path.relative_to(ROOT).as_posix()
            # registry / env I/O modules may mention names in docs
            if rel.endswith("env_registry_data.inc.rs"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if "RLX_" not in line:
                    continue
                if not any(m in line for m in ENV_LINE_MARKERS):
                    continue
                for m in NAME_RE.finditer(line):
                    n = m.group(0)
                    if n.endswith("_"):
                        continue
                    found[n].add(rel)
    return found


def leftover_mentions(registered: set[str]) -> dict[str, set[str]]:
    """RLX_* strings in the tree that are not registered (docs/benches/etc.)."""
    found: dict[str, set[str]] = defaultdict(set)
    exts = {".rs", ".md", ".py", ".sh", ".toml"}
    doc_rel = DOC.relative_to(ROOT).as_posix()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
        if any(p in rel_dir for p in SKIP_PATH_PARTS):
            continue
        for fn in filenames:
            path = Path(dirpath) / fn
            if path.suffix.lower() not in exts and fn not in ("Justfile", "justfile"):
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel == doc_rel or rel.endswith("env_registry_data.inc.rs"):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in NAME_RE.finditer(text):
                n = m.group(0)
                if len(n) > 80 or n.endswith("_"):
                    continue
                if n not in registered:
                    found[n].add(rel)
    return found


def render_from_registry(entries: dict[str, dict], leftovers: dict[str, set[str]]) -> str:
    lines: list[str] = [
        "# RLX environment variables (`RLX_*`)",
        "",
        "Generated from [`env_registry`](../crates/core/rlx-ir/src/env_registry.rs)",
        "(source of truth). Prefer `CompileOptions` when a setting changes compile",
        "semantics. Curated Public list: `just env-catalog`.",
        "",
        "## Legend",
        "",
        "| Stability | Meaning |",
        "|-----------|---------|",
        "| Public | Stable / documented (`just env-catalog`) |",
        "| Bisect | Escape hatch / parity |",
        "| Internal | Bench / tooling |",
        "| Deprecated | Use replace_with |",
        "",
        f"**Registered names:** {len(entries)}  ",
        f"**Unregistered mentions (migration leftovers):** {len(leftovers)}",
        "",
        "## Groups",
        "",
    ]
    by_group: dict[str, list[str]] = defaultdict(list)
    for name, e in entries.items():
        by_group[e["group"]].append(name)
    for g in sorted(by_group):
        lines.append(f"- [{g}](#{g}) — {len(by_group[g])}")
    lines.append("")

    for g in sorted(by_group):
        lines.append(f"## {g}")
        lines.append("")
        lines.append("| Name | Stability | Kind | Layer | Summary |")
        lines.append("|------|-----------|------|-------|---------|")
        for name in sorted(by_group[g]):
            e = entries[name]
            stab = e["stability"]
            if stab == "Deprecated" and e.get("replace_with"):
                stab = f"Deprecated → `{e['replace_with']}`"
            als = ""
            if e["aliases"]:
                als = " aliases: " + ", ".join(f"`{a}`" for a in e["aliases"])
            lines.append(
                f"| `{name}` | {stab} | {e['kind']} | {e['layer']} | {e['summary']}{als} |"
            )
        lines.append("")

    if leftovers:
        lines.append("## Unregistered mentions")
        lines.append("")
        lines.append(
            "Identifiers still appearing in the tree but not yet in the registry "
            "(docs, benches, or pending migration). Prefer registering or deleting."
        )
        lines.append("")
        lines.append("| Name | Example path |")
        lines.append("|------|--------------|")
        for name in sorted(leftovers):
            primary = sorted(
                leftovers[name],
                key=lambda p: (
                    0 if "/src/" in p else 1,
                    0 if p.endswith(".rs") else 1,
                    len(p),
                    p,
                ),
            )[0]
            lines.append(f"| `{name}` | `{primary}` |")
        lines.append("")

    lines.extend(
        [
            "## Maintenance",
            "",
            "```sh",
            "just gen-rlx-env-vars",
            "# or: python3 scripts/gen-rlx-env-vars.py",
            "```",
            "",
            "Add new names to `env_registry_data.inc.rs`. Unregistered "
            "`env::flag(\"RLX_…\")` call sites fail `just check-rlx-env-vars`.",
            "",
        ]
    )
    return "\n".join(lines)


def check_reads_registered(registered: set[str]) -> list[str]:
    reads = collect_env_reads()
    missing = sorted(n for n in reads if n not in registered)
    return missing


def cargo_registry_markdown() -> str | None:
    """Prefer Rust formatter when the crate builds."""
    try:
        r = subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "rlx-ir",
                "--example",
                "env_registry_dump",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode == 0 and r.stdout.strip().startswith("#"):
            return r.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


# Auto-generated stand-ins that are not descriptions. See the ratchet in main().
PLACEHOLDER_PREFIXES = ("See call sites for", "Read at ")

# At ZERO: every registered variable has a hand-written one-line summary.
# Started at 507. Do not raise it — a new variable needs a real summary, not a
# stand-in, and `Read at <path>` / `See call sites for X` are stand-ins.
MAX_PLACEHOLDER_SUMMARIES = 0


def summary_placeholders(entries: dict[str, dict]) -> list[str]:
    """Names whose summary is an auto-generated stand-in.

    `entries` is keyed by name, so iterating it directly yields strings and
    every lookup silently misses — the first version of this did exactly that
    and reported 0 placeholders out of 507. The assertion below makes a
    vacuous result impossible: if nothing was inspected, that is a bug in this
    function, not a clean registry.
    """
    seen = 0
    out = []
    for name, e in entries.items():
        seen += 1
        if str(e.get("summary", "")).startswith(PLACEHOLDER_PREFIXES):
            out.append(name)
    assert seen == len(entries), "placeholder scan inspected nothing"
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if doc/registry drift")
    ap.add_argument(
        "--allow-unregistered-reads",
        action="store_true",
        help="do not fail on env::flag string literals missing from registry",
    )
    args = ap.parse_args()

    entries, registered = parse_registry()
    if not entries:
        print("failed to parse registry", file=sys.stderr)
        return 1

    missing = check_reads_registered(registered)
    if missing and not args.allow_unregistered_reads:
        print(
            f"unregistered env reads ({len(missing)}); add to env_registry_data.inc.rs:",
            file=sys.stderr,
        )
        for n in missing[:40]:
            print(f"  {n}", file=sys.stderr)
        if len(missing) > 40:
            print(f"  … +{len(missing) - 40} more", file=sys.stderr)
        if args.check:
            return 1

    # Every `RLX_*` read must go through the `rlx_ir::env` shim, not `std::env`.
    #
    # This used to be a soft note, and 208 call sites accumulated under it. The
    # shim is not cosmetic: it layers in-process overrides on top of the real
    # environment, so a variable read via `std::env` cannot be set by a test, by
    # `EnvOverride`, or by any harness that needs to A/B two settings inside one
    # process. Those were exactly the knobs nothing could exercise.
    #
    # The listed exceptions are the two places the shim genuinely cannot reach.
    SHIM_EXEMPT = {
        # Build scripts run before (and independently of) the crate graph, so
        # `rlx-ir` is not linkable there and there is no process to override.
        "crates/backends/rlx-cpu/build.rs",
        # `rlx-gguf` is a deliberately standalone GGUF parser with no rlx-ir
        # dependency; adding one for a trace flag would invert the layering.
        "crates/io/rlx-gguf/src/lib.rs",
        "crates/io/rlx-gguf/tests/iq_tq_real_weights.rs",
    }
    std_env_hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT / "crates"):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".rs"):
                continue
            path = Path(dirpath) / fn
            rel = path.relative_to(ROOT).as_posix()
            if rel.endswith("env.rs") or "env_registry" in rel:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if 'std::env::var("RLX_' in line or "std::env::var_os(\"RLX_" in line:
                    if rel in SHIM_EXEMPT:
                        continue
                    std_env_hits.append(f"{rel}:{i}")
    if std_env_hits:
        print(
            f"{len(std_env_hits)} RLX_* read(s) bypass the rlx_ir::env shim "
            f"(use rlx_ir::env::var / var_os / flag / parse_or):",
            file=sys.stderr,
        )
        for h in std_env_hits[:15]:
            print(f"  {h}", file=sys.stderr)
        if len(std_env_hits) > 15:
            print(f"  … +{len(std_env_hits) - 15} more", file=sys.stderr)

    leftovers = leftover_mentions(registered)
    text = render_from_registry(entries, leftovers)

    if args.check:
        ok = DOC.is_file() and DOC.read_text(encoding="utf-8") == text
        if not ok:
            print(f"{DOC.relative_to(ROOT)} is out of date; run: just gen-rlx-env-vars", file=sys.stderr)
            return 1
        if missing and not args.allow_unregistered_reads:
            return 1
        if std_env_hits:
            return 1

        # Placeholder-summary ratchet.
        #
        # `See call sites for X` / `Read at <path>` are auto-generated stand-ins,
        # not descriptions. Every `Public` entry already has a real summary, so
        # the curated catalogue is clean; the debt is in the `Bisect` /
        # `Internal` inventory. Rather than block on paying all of it down, this
        # freezes the count: a new variable cannot add another placeholder, and
        # the number only moves down. Lower it when you curate a batch.
        placeholders = summary_placeholders(entries)
        if len(placeholders) > MAX_PLACEHOLDER_SUMMARIES:
            print(
                f"{len(placeholders)} placeholder summaries "
                f"(ratchet allows {MAX_PLACEHOLDER_SUMMARIES}); write a real one-line "
                f"summary for:",
                file=sys.stderr,
            )
            for n in placeholders[:15]:
                print(f"  {n}", file=sys.stderr)
            return 1
        if len(placeholders) < MAX_PLACEHOLDER_SUMMARIES:
            print(
                f"placeholder summaries down to {len(placeholders)} "
                f"(ratchet is {MAX_PLACEHOLDER_SUMMARIES}) — lower "
                f"MAX_PLACEHOLDER_SUMMARIES in {Path(__file__).name} to lock it in.",
                file=sys.stderr,
            )
        print(
            f"{DOC.relative_to(ROOT)} is up to date "
            f"({len(entries)} registered, {len(leftovers)} leftover mentions)"
        )
        return 0

    DOC.write_text(text, encoding="utf-8")
    print(
        f"wrote {DOC.relative_to(ROOT)} "
        f"({len(entries)} registered, {len(leftovers)} leftovers, "
        f"{len(missing)} unregistered reads)"
    )
    return 1 if missing and not args.allow_unregistered_reads else 0


if __name__ == "__main__":
    raise SystemExit(main())
