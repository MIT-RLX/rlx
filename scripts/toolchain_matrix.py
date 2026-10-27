#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Run each backend's suite against every external toolchain version we support.

The spec (`scripts/toolchain_matrix.toml`) lists runs: a device, the component
and version under test, the rig (an ssh alias, or `local`), the env that selects
that version (a venv's aiecc, a CUDA toolkit dir, …), the command, and the text
that marks a pass. Each run gets a fresh TMPDIR, so kernels compiled by an
earlier version can never be reused and make a run pass that compiled nothing.

    python3 scripts/toolchain_matrix.py                  # run everything, print a table
    python3 scripts/toolchain_matrix.py --only xdna      # one device (or component)
    python3 scripts/toolchain_matrix.py --update         # also rewrite tested_toolchains.tsv
    python3 scripts/toolchain_matrix.py --upstream       # newer upstream releases than tested?

`--update` replaces the matching rows of
`crates/core/rlx-runtime/src/doctor/tested_toolchains.tsv`, which `cargo rlx
doctor` shows as "tested". `--upstream` exits 1 when a project has a release
newer than anything in that file — the signal to add it to the matrix before a
user reports the break.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "toolchain_matrix.toml"
TSV = ROOT / "crates/core/rlx-runtime/src/doctor/tested_toolchains.tsv"


def version_key(v: str) -> tuple:
    out = []
    for part in v.lstrip("v").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def run_one(run: dict) -> tuple[bool, str]:
    """Run one matrix entry; (passed, last lines of output)."""
    env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in run.get("env", {}).items())
    # A fresh TMPDIR per run (kernel builds are cached under it), removed after.
    script = (
        f"set -o pipefail; T=$(mktemp -d); export TMPDIR=$T; "
        f"[ -f ~/.cargo/env ] && . ~/.cargo/env; "
        f"cd {run.get('workdir', '.')} && env {env} {run['command']}; rc=$?; "
        f'rm -rf "$T"; exit $rc'
    )
    # `~` must expand on the target, so values stay unquoted where they start with it.
    script = script.replace("'~/", "~/'")
    host = run.get("host", "local")
    argv = ["bash", "-lc", script] if host == "local" else ["ssh", host, script]
    try:
        p = subprocess.run(
            argv, capture_output=True, text=True, timeout=run.get("timeout_s", 3600)
        )
    except subprocess.TimeoutExpired:
        return False, "timed out"
    out = (p.stdout or "") + (p.stderr or "")
    passed = p.returncode == 0 and run.get("pass", "") in out
    tail = "\n".join(out.strip().splitlines()[-8:])
    return passed, tail


def load_tsv() -> tuple[list[str], list[list[str]]]:
    header, rows = [], []
    for line in TSV.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            header.append(line)
        else:
            rows.append(line.split("\t"))
    return header, rows


def write_tsv(header: list[str], rows: list[list[str]]) -> None:
    rows.sort(key=lambda r: (r[0], r[1], version_key(r[2]), r[3]))
    TSV.write_text("\n".join(header + ["\t".join(r) for r in rows]) + "\n", encoding="utf-8")


def normalize_tag(tag: str) -> str:
    """`v1.4.4` → `1.4.4`; `202610.2.23.244` (YYYYMM-prefixed) → `2.23.244`;
    `202610.2.23.0_Canonical` → `2.23.0`."""
    t = tag.lstrip("v").split("_")[0]
    parts = t.split(".")
    if len(parts) > 3 and len(parts[0]) == 6 and parts[0].isdigit():
        parts = parts[1:]
    return ".".join(parts)


def upstream(spec: dict) -> int:
    _, rows = load_tsv()
    stale = 0
    for u in spec.get("upstream", []):
        tested = [r[2] for r in rows if r[0] == u["device"] and r[1] == u["component"] and r[4] == "pass"]
        newest_tested = max(tested, key=version_key) if tested else "none"
        try:
            tags = subprocess.run(
                ["gh", "api", f"repos/{u['github']}/releases", "--paginate", "--jq", ".[].tag_name"],
                capture_output=True, text=True, timeout=120, check=True,
            ).stdout.split()
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"{u['component']:<12} cannot query {u['github']}: {e}")
            continue
        # The newest release by version, not GitHub's "latest" marker: XRT marks a
        # date-prefixed 2.23 tag latest while 2.25.0 exists.
        # Plain versions only; a first component >= 1000 is an old year-based
        # scheme (XRT `2019.2.0.3`) that would otherwise sort above every 2.x.
        versions = [
            v for v in (normalize_tag(t) for t in tags)
            if re.fullmatch(r"\d+(\.\d+){1,3}", v) and int(v.split(".")[0]) < 1000
        ]
        tag = max(versions, key=version_key, default="")
        if not tag:
            continue
        newer = not tested or version_key(tag) > version_key(newest_tested)
        stale += newer
        flag = "NEWER — add it to the matrix" if newer else "covered"
        print(f"{u['component']:<12} latest {tag:<12} newest tested {newest_tested:<10} {flag}")
    return 1 if stale else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run entries whose device or component matches")
    ap.add_argument("--version", action="append", help="run only these versions (repeatable)")
    ap.add_argument("--update", action="store_true", help="rewrite tested_toolchains.tsv rows")
    ap.add_argument("--upstream", action="store_true", help="check upstream releases vs tested")
    ap.add_argument("--json", action="store_true", help="machine-readable results")
    args = ap.parse_args()

    spec = tomllib.loads(SPEC.read_text(encoding="utf-8"))
    if args.upstream:
        return upstream(spec)

    runs = [
        r for r in spec.get("run", [])
        if (not args.only or args.only in (r["device"], r["component"]))
        and (not args.version or r["version"] in args.version)
    ]
    results = []
    for r in runs:
        label = f"{r['device']}/{r['component']} {r['version']} @ {r.get('host', 'local')}"
        print(f"… {label}", file=sys.stderr, flush=True)
        ok, tail = run_one(r)
        results.append({**r, "passed": ok, "tail": tail})
        if not ok:
            print(f"   FAILED:\n{tail}", file=sys.stderr)

    if args.json:
        print(json.dumps([{k: v for k, v in r.items() if k != "env"} for r in results], indent=2))
    else:
        print(f"\n{'device':<8} {'component':<10} {'version':<12} {'host':<8} result")
        for r in results:
            print(
                f"{r['device']:<8} {r['component']:<10} {r['version']:<12} "
                f"{r.get('host', 'local'):<8} {'pass' if r['passed'] else 'FAIL'}"
            )

    if args.update and results:
        header, rows = load_tsv()
        today = dt.date.today().isoformat()
        for r in results:
            key = (r["device"], r["component"], r["version"], r["hardware"])
            notes = r.get("notes") or (f"peano {r['peano']}" if r.get("peano") else "")
            # One row per (key, notes): XRT rows differ only by the mlir-aie
            # they ran with; a rerun replaces its own row and nothing else.
            rows = [
                row for row in rows
                if not (tuple(row[:4]) == key and (row[6] if len(row) > 6 else "") in ("", notes))
            ]
            rows.append([*key, "pass" if r["passed"] else "fail", today, notes])
        write_tsv(header, rows)
        print(f"updated {TSV.relative_to(ROOT)}", file=sys.stderr)

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
