#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Maintain `tests/known_failures/<test>.txt` files (see `rlx_ir::known_failures`).

Collect verdicts, then apply them:

  RLX_KNOWN_FAILURES_REPORT=/tmp/kf.jsonl just test-gpu
  python3 scripts/known-failures.py update /tmp/kf.jsonl --dry-run
  python3 scripts/known-failures.py update /tmp/kf.jsonl --issue owner/repo#N

`update` removes entries whose case now passes (verdict `known-pass`) and adds
an entry per new failure (`fail`, unlisted). New entries need an issue: pass
`--issue`, or the failures are listed and the script exits 1 without adding
them. New entries are limited to the OS they failed on (`device@os`) unless
`--any-os`.

  python3 scripts/known-failures.py list [--issue #N]

prints every entry in the tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "target", "vendor", "node_modules"}

HEADER = """\
# Known failures for tests/{test}.rs — format and env options:
# `rlx_ir::known_failures`. Columns: case  device[@os]  issue  [note].
"""


def issue_eq(listed: str, wanted: str) -> bool:
    return (
        listed == wanted
        or (wanted.startswith("#") and listed.endswith(wanted))
        or (listed.startswith("#") and wanted.endswith(listed))
    )


def entry_key(line: str):
    """(case, device, os) of an entry line, or None for comments/blank."""
    t = line.strip()
    if not t or t.startswith("#"):
        return None
    cols = t.split()
    if len(cols) < 3:
        return None
    dev, _, os_ = cols[1].partition("@")
    return cols[0], dev.lower(), os_ or None


def matches(key, case, device, os_) -> bool:
    k_case, k_dev, k_os = key
    return (
        k_case == case
        and (k_dev == "*" or k_dev == device)
        and (k_os is None or k_os == os_)
    )


def load_report(path: Path):
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            sys.exit(f"{path}:{n}: bad report line: {e}")
    # A case run twice keeps its last verdict.
    last = {}
    for r in rows:
        last[(r["file"], r["case"], r["device"], r["os"])] = r
    return list(last.values())


def cmd_update(args) -> int:
    rows = load_report(Path(args.report))
    by_file = defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)

    unfiled = []
    changed = 0
    for file, frows in sorted(by_file.items()):
        path = Path(file)
        lines = path.read_text().splitlines() if path.exists() else []
        out = list(lines)

        for r in frows:
            if r["verdict"] != "known-pass":
                continue
            before = len(out)
            out = [
                l
                for l in out
                if not (
                    (k := entry_key(l))
                    and matches(k, r["case"], r["device"], r["os"])
                )
            ]
            if len(out) == before:
                print(f"  stale entry not in the tree, remove it at {r['source']}: "
                      f"{r['case']} {r['device']}")
            else:
                print(f"- {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: "
                      f"{r['case']} {r['device']} now passes")

        new = [r for r in frows if r["verdict"] == "fail"]
        for r in new:
            if not args.issue:
                unfiled.append(r)
                continue
            dev = r["device"] if args.any_os else f"{r['device']}@{r['os']}"
            note = r["detail"][:100]
            if not out:
                out = HEADER.format(test=r["test"]).splitlines()
            out.append(f"{r['case']}  {dev}  {args.issue}  {note}".rstrip())
            print(f"+ {path.name}: {r['case']} {dev} {args.issue}")

        if out != lines:
            changed += 1
            if not args.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(out) + "\n")

    if unfiled:
        print(f"\n{len(unfiled)} new failure(s) need an issue (--issue owner/repo#N):")
        for r in unfiled:
            print(f"  {r['test']}::{r['case']} on {r['device']}@{r['os']}: {r['detail'][:100]}")
    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb} {changed} file(s)")
    return 1 if unfiled else 0


def all_lists():
    for p in ROOT.rglob("known_failures/*.txt"):
        if not SKIP_DIRS.intersection(p.parts):
            yield p


def cmd_list(args) -> int:
    for p in sorted(all_lists()):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if entry_key(line) is None:
                continue
            issue = line.split()[2]
            if args.issue and not issue_eq(issue, args.issue):
                continue
            print(f"{p.relative_to(ROOT)}:{n}: {line.strip()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("update", help="apply a verdict report to the lists")
    up.add_argument("report")
    up.add_argument("--issue", help="issue for new entries (#N or owner/repo#N)")
    up.add_argument("--any-os", action="store_true", help="new entries match every OS")
    up.add_argument("--dry-run", action="store_true")
    ls = sub.add_parser("list", help="print every entry")
    ls.add_argument("--issue")
    args = ap.parse_args()
    return {"update": cmd_update, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
